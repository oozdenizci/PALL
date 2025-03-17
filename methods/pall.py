import os
from copy import deepcopy
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .base import *
from .er import RehearsalMemory
from models.vit import VisionTransformer
from models.subnet_vit import SubnetVisionTransformer


class PALL(Base):
    def __init__(self, args):
        super(PALL, self).__init__(args)
        self.memory = RehearsalMemory(buffer_size=self.args.mem_budget,
                                      n_tasks=self.args.n_tasks,
                                      cpt=self.args.class_per_task,
                                      dim_x=self.args.dim_input,
                                      device=self.device,
                                      mem_type=self.args.mem_type,
                                      save_logits=True)
        self.alpha = args.alpha
        self.beta = args.beta
        self.k_shot = args.k_shot
        self.per_task_masks = {}
        self.combined_masks = {}
        self.finetuning_hist = {}

    def init_optimizer(self):
        if self.args.optim == "sgd":
            return SGD(self.net.parameters(), lr=self.lr, momentum=self.momentum)
        elif self.args.optim == "adam":
            return Adam(self.net.parameters(), lr=self.lr)
        else:
            raise NotImplementedError

    def der_loss(self, a, b, task_id):
        a_ = a[..., task_id*self.cpt:(task_id+1)*self.cpt]
        b_ = b[..., task_id*self.cpt:(task_id+1)*self.cpt]
        return F.mse_loss(a_, b_)

    def forward(self, x, task, mask=None, mode="train"):
        pred = self.net.forward(x, task, mask, mode)
        if task > 0:
            pred[:, :self.cpt * task].data.fill_(-10e10)
        if task < self.n_tasks - 1:
            pred[:, self.cpt * (task + 1):].data.fill_(-10e10)
        return pred

    def forward_with_features(self, x, task, mask=None, mode="train"):
        pred, features = self.net.forward(x, task, mask, mode, returnt='all')
        if task > 0:
            pred[:, :self.cpt * task].data.fill_(-10e10)
        if task < self.n_tasks - 1:
            pred[:, self.cpt * (task + 1):].data.fill_(-10e10)
        return pred, features

    def evaluate(self, x, task, mask=None, mode="test"):
        if task in self.per_task_masks:
            return self.forward(x, task, mask=self.per_task_masks[task], mode="test")
        else:
            return self.forward(x, task, mask=None, mode="no_mask")

    def extract_logits_and_features(self, data_loader, task_id, norm_features=True):
        features, logits, targets = [], [], []
        with torch.no_grad():
            self.net.eval()
            for x, y in data_loader:
                x, y = x.to(self.device), y.to(self.device)
                pred, feats = self.forward_with_features(x, task_id, mask=self.per_task_masks[task_id], mode="test")
                if norm_features:
                    feats = feats / feats.norm(dim=1).view(-1, 1)
                features.append(feats)
                logits.append(pred)
                targets.append(y)
            features, logits, targets = torch.cat(features), torch.cat(logits), torch.cat(targets)
        self.net.train()
        return features.cpu(), logits.cpu(), targets.cpu()

    def fill_buffer(self, task_id, dataset):
        sel_loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=2)
        features, logits, targets = self.extract_logits_and_features(sel_loader, task_id)
        if self.args.mem_type == "random":
            sel_indices = self.memory.select_indices_by_random(targets)
        else:
            raise NotImplementedError
        x, y = zip(*(sel_loader.dataset[idx] for idx in sel_indices))
        if self.memory.save_logits:
            self.memory.add((x, y, logits[sel_indices].detach()), task_id)
        else:
            self.memory.add((x, y), task_id)

    def learn(self, task_id, dataset):
        assert task_id not in self.per_task_masks, f"[ERROR] {task_id} already present in learned subnet masks"
        assert self.args.weight_decay >= 0.0, f"[ERROR] Invalid weight_decay value: {self.args.weight_decay}"

        loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=2)
        self.opt = self.init_optimizer()

        if isinstance(self.net, SubnetVisionTransformer) or isinstance(self.net, VisionTransformer):
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, self.args.n_epochs)

        for epoch in range(self.args.n_epochs):
            for i, (x, y) in enumerate(loader):
                x, y = x.to(self.device), y.to(self.device)
                loss = self.loss_fn(self.forward(x, task_id, mask=None, mode="train"), y)
                self.opt.zero_grad()
                loss.backward()

                # Set gradients to zero (no backprop) for the active combined subnets
                if self.combined_masks != {}:  # Only do this for tasks 1 and beyond
                    for key in self.combined_masks.keys():
                        key_split = key.split('.')
                        module_attr = key_split[-1]
                        if 'classifier' in key_split or len(key_split) == 2:  # e.g., conv1.weight or classifier.weight
                            module_name = key_split[0]
                            if hasattr(getattr(self.net, module_name), module_attr):
                                if getattr(getattr(self.net, module_name), module_attr) is not None:
                                    getattr(getattr(self.net, module_name), module_attr).grad[self.combined_masks[key] == 1] = 0
                        elif len(key_split) == 4:  # e.g., layer1.0.conv1.weight or encoder_layers.0.mlp_lin1.weight
                            curr_module = getattr(getattr(self.net, key_split[0])[int(key_split[1])], key_split[2])
                            if hasattr(curr_module, module_attr):
                                if getattr(curr_module, module_attr) is not None:
                                    getattr(curr_module, module_attr).grad[self.combined_masks[key] == 1] = 0
                        elif len(key_split) == 5:  # e.g., encoder_layers.0.mha.W_V.weight
                            curr_module = getattr(getattr(getattr(self.net, key_split[0])[int(key_split[1])], key_split[2]), key_split[3])
                            if hasattr(curr_module, module_attr):
                                if getattr(curr_module, module_attr) is not None:
                                    getattr(curr_module, module_attr).grad[self.combined_masks[key] == 1] = 0
                        else:
                            raise NotImplementedError('This should not happen with the currently implemented models!')

                    if self.args.weight_decay != 0.0:
                        for n, p in self.net.named_parameters():    # no weight decay for the scores
                            if n.endswith("weight") or n.endswith("bias"):
                                if self.combined_masks[n] is not None:
                                    p.grad += self.args.weight_decay * p * (1 - self.combined_masks[n])
                else:
                    if self.args.weight_decay != 0.0:
                        for n, p in self.net.named_parameters():    # no weight decay for the scores
                            if n.endswith("weight") or n.endswith("bias"):
                                p.grad += self.args.weight_decay * p

                self.opt.step()

            if self.scheduler is not None:
                self.scheduler.step()

        # Then save the per-task-dependent masks
        self.per_task_masks[task_id] = self.net.get_masks(task_id)

        # Fill up rehearsal memory using the final architecture
        self.fill_buffer(task_id, dataset)

        # Combine task masks to keep track of parameters to-update or not
        if self.combined_masks == {}:
            self.combined_masks = deepcopy(self.per_task_masks[task_id])
        else:
            for key in self.per_task_masks[task_id].keys():
                if self.combined_masks[key] is not None and self.per_task_masks[task_id][key] is not None:
                    self.combined_masks[key] = 1 - ((1 - self.combined_masks[key]) * (1 - self.per_task_masks[task_id][key]))

        # Print sparsity metrics
        connectivity_per_layer = self.print_connectivity(self.combined_masks)
        all_connectivity = self.global_connectivity(self.combined_masks)
        print("Connectivity per Layer: {}".format(connectivity_per_layer))
        print("Global Connectivity: {}".format(all_connectivity))

        # Reinitialize the scores and unused weights
        self.net.reinit_scores()
        self.net.reinit_weights(self.combined_masks)

    def forget(self, task_id):
        assert task_id in self.per_task_masks, f"[ERROR] {task_id} is not learned yet (no per_task_mask found)"

        # (1) Backup the masks of the task to be forgotten from the list
        deleted_masks = deepcopy(self.per_task_masks[task_id])

        # (2) Remove the masks of the task to be forgotten from the list as well as the rehearsal samples from memory
        self.memory.remove(task_id)
        del self.per_task_masks[task_id]
        if isinstance(self.net, SubnetVisionTransformer):
            nn.init.zeros_(self.net.class_tokens[task_id])
            nn.init.normal_(self.net.pos_embeddings[task_id], std=0.02)

        if self.per_task_masks != {}:
            # (3) Create combined masks for until task_id
            prev_tasks = [t for t in self.per_task_masks.keys() if task_id > t]
            combined_prev_task_id = {}
            for prev_task in prev_tasks:
                if combined_prev_task_id == {}:
                    combined_prev_task_id = deepcopy(self.per_task_masks[prev_task])
                else:
                    for key in self.per_task_masks[prev_task].keys():
                        if combined_prev_task_id[key] is not None and self.per_task_masks[prev_task][key] is not None:
                            combined_prev_task_id[key] = 1 - ((1 - combined_prev_task_id[key]) * (1 - self.per_task_masks[prev_task][key]))

            # (4) Create combined masks for after task_id
            after_tasks = [t for t in self.per_task_masks.keys() if task_id < t]
            combined_after_task_id = {}
            for after_task in after_tasks:
                if combined_after_task_id == {}:
                    combined_after_task_id = deepcopy(self.per_task_masks[after_task])
                else:
                    for key in self.per_task_masks[after_task].keys():
                        if combined_after_task_id[key] is not None and self.per_task_masks[after_task][key] is not None:
                            combined_after_task_id[key] = 1 - ((1 - combined_after_task_id[key]) * (1 - self.per_task_masks[after_task][key]))

            # (5) Store indices of what to reset and what to finetune to recover KT
            to_finetune, to_reset, buffer_leak = {}, {}, {}
            if combined_prev_task_id == {}:
                assert combined_after_task_id != {}
                for key in deleted_masks.keys():
                    if deleted_masks[key] is not None and combined_after_task_id[key] is not None:
                        to_reset[key] = torch.logical_not(deleted_masks[key] == 1)
                        to_finetune[key] = torch.logical_and(deleted_masks[key] == 1, combined_after_task_id[key] == 1)
            else:
                if combined_after_task_id == {}:
                    for key in deleted_masks.keys():
                        if deleted_masks[key] is not None and combined_prev_task_id[key] is not None:
                            to_reset[key] = torch.logical_and(deleted_masks[key] == 1, combined_prev_task_id[key] == 0)
                            buffer_leak_ids = [k for k in self.finetuning_hist.keys() if task_id in self.finetuning_hist[k][1]]
                            buffer_leak[key] = torch.zeros_like(deleted_masks[key], dtype=torch.bool)
                            if buffer_leak_ids:
                                for k in buffer_leak_ids:
                                    if self.finetuning_hist[k][0][key] is not None:
                                        buffer_leak[key] = torch.logical_or(buffer_leak[key], torch.logical_and(torch.logical_and(deleted_masks[key] == 1, combined_prev_task_id[key] == 1), self.finetuning_hist[k][0][key] == 1))
                                to_reset[key] = torch.logical_not(torch.logical_or(to_reset[key], buffer_leak[key]))
                                to_finetune[key] = buffer_leak[key]
                            else:
                                to_reset[key] = torch.logical_not(to_reset[key])
                                to_finetune[key] = buffer_leak[key]
                else:
                    for key in deleted_masks.keys():
                        if (deleted_masks[key] is not None and combined_prev_task_id[key] is not None and combined_after_task_id[key] is not None):
                            to_reset[key] = torch.logical_and(deleted_masks[key] == 1, combined_prev_task_id[key] == 0)
                            to_finetune[key] = torch.logical_and(deleted_masks[key] == 1, torch.logical_and(combined_prev_task_id[key] == 0, combined_after_task_id[key] == 1))
                            buffer_leak_ids = [k for k in self.finetuning_hist.keys() if task_id in self.finetuning_hist[k][1]]
                            buffer_leak[key] = torch.zeros_like(deleted_masks[key], dtype=torch.bool)
                            if buffer_leak_ids:
                                for k in buffer_leak_ids:
                                    if self.finetuning_hist[k][0][key] is not None:
                                        buffer_leak[key] = torch.logical_or(buffer_leak[key], torch.logical_and(torch.logical_and(deleted_masks[key] == 1, combined_prev_task_id[key] == 1), self.finetuning_hist[k][0][key] == 1))
                                to_reset[key] = torch.logical_not(torch.logical_or(to_reset[key], buffer_leak[key]))
                                to_finetune[key] = torch.logical_or(to_finetune[key], buffer_leak[key])
                            else:
                                to_reset[key] = torch.logical_not(to_reset[key])
                                to_finetune[key] = to_finetune[key]

            # (6) Reinitialize the weights that were specific to this task
            self.net.reinit_weights(to_reset)

            do_finetune = False
            if to_finetune != {}:
                for key in to_finetune.keys():
                    if torch.any(to_finetune[key]):
                        do_finetune = True

            # (7) Then finetune a subset of these weights which were used in other tasks
            if do_finetune:
                active_tasks = list(self.per_task_masks.keys())
                self.finetuning_hist[task_id] = (to_finetune, active_tasks)

                finetune_opt = self.init_optimizer()
                for i in range(self.k_shot):
                    finetune_opt.zero_grad()
                    loss = 0.0
                    for t_id in active_tasks:
                        if self.alpha > 0.0:
                            x_past, y_past, h_past = self.memory.sample_task(self.args.batch_size // len(active_tasks), t_id)
                            h = self.forward(x_past, t_id, mask=self.per_task_masks[t_id], mode="test")
                            loss += self.alpha * self.der_loss(h, h_past, t_id) / len(active_tasks)
                        if self.beta > 0.0:
                            x_past, y_past, h_past = self.memory.sample_task(self.args.batch_size // len(active_tasks), t_id)
                            h = self.forward(x_past, t_id, mask=self.per_task_masks[t_id], mode="test")
                            loss += self.beta * self.loss_fn(h, y_past) / len(active_tasks)
                    loss.backward()

                    for key in to_finetune.keys():
                        key_split = key.split('.')
                        module_attr = key_split[-1]
                        if 'classifier' in key_split or len(key_split) == 2:  # e.g., conv1.weight or classifier.weight
                            module_name = key_split[0]
                            if hasattr(getattr(self.net, module_name), module_attr):
                                if getattr(getattr(self.net, module_name), module_attr) is not None:
                                    getattr(getattr(self.net, module_name), module_attr).grad[to_finetune[key] == 0] = 0
                        elif len(key_split) == 4:  # e.g., layer1.0.conv1.weight or encoder_layers.0.mlp_lin1.weight
                            curr_module = getattr(getattr(self.net, key_split[0])[int(key_split[1])], key_split[2])
                            if hasattr(curr_module, module_attr):
                                if getattr(curr_module, module_attr) is not None:
                                    getattr(curr_module, module_attr).grad[to_finetune[key] == 0] = 0
                        elif len(key_split) == 5:  # e.g., encoder_layers.0.mha.W_V.weight
                            curr_module = getattr(getattr(getattr(self.net, key_split[0])[int(key_split[1])], key_split[2]), key_split[3])
                            if hasattr(curr_module, module_attr):
                                if getattr(curr_module, module_attr) is not None:
                                    getattr(curr_module, module_attr).grad[to_finetune[key] == 0] = 0
                        else:
                            raise NotImplementedError('This should not happen with the currently implemented models!')

                    if self.args.weight_decay != 0.0:
                        for n, p in self.net.named_parameters():    # make sure no weight decay for the scores
                            if n.endswith("weight") or n.endswith("bias"):
                                if to_finetune[n] is not None:
                                    p.grad += self.args.weight_decay * p * to_finetune[n]

                    finetune_opt.step()

            # (8) Recreate the combined masks
            self.combined_masks = {}
            for task in self.per_task_masks.keys():
                if self.combined_masks == {}:
                    self.combined_masks = deepcopy(self.per_task_masks[task])
                else:
                    for key in self.per_task_masks[task].keys():
                        if self.combined_masks[key] is not None and self.per_task_masks[task][key] is not None:
                            self.combined_masks[key] = 1 - ((1 - self.combined_masks[key]) * (1 - self.per_task_masks[task][key]))

            # Print sparsity metrics
            connectivity_per_layer = self.print_connectivity(self.combined_masks)
            all_connectivity = self.global_connectivity(self.combined_masks)
            print("Connectivity per Layer: {}".format(connectivity_per_layer))
            print("Global Connectivity: {}".format(all_connectivity))

        else:   # If task_id was the only task remaining so far, then reset all.
            self.combined_masks = {}
            self.net.reinit_weights(self.combined_masks)

    def print_connectivity(self, combined_masks, percent=1.0):
        connectivity_dict = {}
        for key in combined_masks.keys():
            mask = combined_masks[key]
            if mask is not None:
                connectivity = torch.sum(mask == 1) / np.prod(mask.shape)
                connectivity_dict[key] = connectivity * percent
        return connectivity_dict

    def global_connectivity(self, combined_masks):
        denum, num = 0, 0
        for key in combined_masks.keys():
            mask = combined_masks[key]
            if mask is not None:
                num += torch.sum(mask == 1).item()
                denum += np.prod(mask.shape)
        return num / denum
