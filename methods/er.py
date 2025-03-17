import random
import torch.utils.data as data
from torch.utils.data import DataLoader
from .base import *
from models.vit import VisionTransformer
from models.subnet_vit import SubnetVisionTransformer


class RehearsalMemory(data.Dataset):
    def __init__(self, buffer_size, n_tasks, cpt, dim_x, device, mem_type="random", save_logits=False):
        super(RehearsalMemory, self).__init__()
        self.buffer = {}
        self.device = device
        self.mem_type = mem_type
        self.save_logits = save_logits
        self.dim_x = dim_x
        self.buffer_size = buffer_size
        self.n_tasks = n_tasks
        self.cpt = cpt
        self.exemplars_per_task = self.buffer_size // self.n_tasks
        self.exemplars_per_class = self.exemplars_per_task // self.cpt

    def __len__(self):
        if not self.buffer:
            return 0
        else:
            n = 0
            for t in self.buffer:
                n += min(self.buffer[t]['num_seen'], self.exemplars_per_task)
            return n

    def select_indices_by_random(self, targets):
        result = []
        for curr_cls in np.unique(targets):
            cls_ind = np.where(targets == curr_cls)[0]
            assert (len(cls_ind) > 0), "No samples to choose from for class {:d}".format(curr_cls)
            assert (self.exemplars_per_class <= len(cls_ind)), "Not enough samples to store"
            result.extend(random.sample(list(cls_ind), self.exemplars_per_class))
        return result

    def add(self, data, t):
        if self.save_logits:
            x, y, h = torch.stack(data[0]).cpu(), torch.as_tensor(data[1]).cpu(), data[2].cpu()
            dim_h = h.shape[-1]
        else:
            x, y = torch.stack(data[0]).cpu(), torch.as_tensor(data[1]).cpu()

        if t not in self.buffer:
            if self.save_logits:
                self.buffer[t] = {
                    "X": torch.zeros([self.exemplars_per_task] + list(self.dim_x)),
                    "Y": torch.zeros(self.exemplars_per_task).long(),
                    "H": torch.zeros([self.exemplars_per_task] + [dim_h]),
                    "num_seen": 0,
                }
            else:
                self.buffer[t] = {
                    "X": torch.zeros([self.exemplars_per_task] + list(self.dim_x)),
                    "Y": torch.zeros(self.exemplars_per_task).long(),
                    "num_seen": 0,
                }

        # Override the buffer for task t with the data
        assert x.shape[0] == self.exemplars_per_task
        self.buffer[t]['num_seen'] += x.shape[0]
        self.buffer[t]['X'] = x
        self.buffer[t]['Y'] = y
        if self.save_logits:
            self.buffer[t]['H'] = h

    def sample_task(self, n, task_id):
        X = []; Y = []; H = []
        assert task_id in self.buffer, f"[ERROR] not found {task_id} in buffer"
        v = self.buffer[task_id]

        indices_by_label = {}
        for idx, label in enumerate(v['Y']):
            if label.item() not in indices_by_label:
                indices_by_label[label.item()] = []
            indices_by_label[label.item()].append(idx)

        if n >= len(indices_by_label):
            sampled_indices = []
            for label, indices in indices_by_label.items():
                num_to_sample = min(n // len(indices_by_label), len(indices))
                sampled_indices.extend(torch.tensor(indices)[torch.randperm(len(indices))[:num_to_sample]])
            sampled_indices = torch.tensor(sampled_indices)

            if n > len(sampled_indices) and v['X'].shape[0] > len(sampled_indices):  # makes it un-stratified
                diff = n - len(sampled_indices)
                candidates = torch.tensor([i for i in torch.randperm(v['X'].shape[0]) if i not in sampled_indices])
                sampled_indices = torch.cat((sampled_indices, candidates[:diff]))
        else:
            sampled_indices = torch.randperm(min(v['num_seen'], v['X'].shape[0]))[:min(min(n, v['num_seen']), v['X'].shape[0])]

        X.append(v['X'][sampled_indices])
        Y.append(v['Y'][sampled_indices])

        if self.save_logits:
            H.append(v['H'][sampled_indices])
            return torch.cat(X, 0).to(self.device), torch.cat(Y, 0).to(self.device), torch.cat(H, 0).to(self.device)
        else:
            return torch.cat(X, 0).to(self.device), torch.cat(Y, 0).to(self.device)

    def remove(self, t):
        X = self.buffer[t]['X']
        Y = self.buffer[t]['Y']
        if self.save_logits:
            H = self.buffer[t]['H']
        del self.buffer[t]
        return X, Y, H if self.save_logits else X, Y


class ER(Base):
    def __init__(self, args):
        super(ER, self).__init__(args)
        self.memory = RehearsalMemory(buffer_size=self.args.mem_budget,
                                      n_tasks=self.args.n_tasks,
                                      cpt=self.args.class_per_task,
                                      dim_x=self.args.dim_input,
                                      device=self.device,
                                      mem_type=self.args.mem_type,
                                      save_logits=False)

    def extract_logits_and_features(self, data_loader, task_id, norm_features=True):
        features, logits, targets = [], [], []
        with torch.no_grad():
            self.net.eval()
            for x, y in data_loader:
                x, y = x.to(self.device), y.to(self.device)
                pred, feats = self.forward_with_features(x, task_id)
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
        loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=2)
        self.opt = self.init_optimizer()

        if isinstance(self.net, SubnetVisionTransformer) or isinstance(self.net, VisionTransformer):
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, self.args.n_epochs)

        self.n_iters = self.args.n_epochs * len(loader)
        for epoch in range(self.args.n_epochs):
            for i, (x, y) in enumerate(loader):
                x, y = x.to(self.device), y.to(self.device)
                loss = self.loss_fn(self.forward(x, task_id), y)

                n_prev_tasks = len(self.prev_tasks)
                for t in self.prev_tasks:
                    x_past, y_past = self.memory.sample_task(self.args.batch_size // n_prev_tasks, t)
                    loss += self.loss_fn(self.forward(x_past, t), y_past) / n_prev_tasks

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

            if self.scheduler is not None:
                self.scheduler.step()

        self.fill_buffer(task_id, dataset)
        self.prev_tasks.append(task_id)

    def forget(self, task_id):
        self.prev_tasks.remove(task_id)
        n_prev_tasks = len(self.prev_tasks)

        self.opt = self.init_optimizer()

        assert self.args.forget_iters is not None

        for i in range(self.args.forget_iters):
            self.opt.zero_grad()
            x_forget, _ = self.memory.sample_task(self.args.batch_size, task_id)
            out = self.forward(x_forget, task_id)

            uniform_target = (torch.ones(out.shape) / self.cpt).to(self.device)
            if task_id > 0:
                uniform_target[:, :self.cpt * task_id].data.fill_(0.0)
            if task_id < self.n_tasks - 1:
                uniform_target[:, self.cpt * (task_id + 1):].data.fill_(0.0)

            loss = self.loss_fn(out, uniform_target)

            if n_prev_tasks > 0:
                for t in self.prev_tasks:
                    x_past, y_past = self.memory.sample_task(self.args.batch_size // n_prev_tasks, t)
                    loss += self.loss_fn(self.forward(x_past, t), y_past) / n_prev_tasks

            loss.backward()
            self.opt.step()

        self.memory.remove(task_id)
