import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .base import *
from .lwf import smooth, modified_kl_div
from models.vit import VisionTransformer
from models.subnet_vit import SubnetVisionTransformer


class LSF(Base):
    def __init__(self, args):
        super(LSF, self).__init__(args)
        self.dim_input = args.dim_input
        self.mnemonic_code = torch.randn(self.n_tasks * self.cpt, *args.dim_input).to(self.device)  # all codes
        self.mnemonic_target = torch.arange(self.n_tasks * self.cpt).to(self.device)
        self.fish = {}
        self.checkpoint = {}
        self.old_net = None
        self.prev_dataset = None

    def init_optimizer_classifier(self):
        if self.args.optim == "sgd":
            return SGD(self.net.classifier.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        elif self.args.optim == "adam":
            return Adam(self.net.classifier.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise NotImplementedError

    def penalty(self):
        if len(self.prev_tasks) == 0:
            return torch.tensor(0.0).to(self.device)
        current_param = self.net.get_params()
        penalty = 0.0
        for t in self.prev_tasks:
            penalty += (self.fish[t] * (current_param - self.checkpoint[t]).pow(2)).sum()
        return penalty

    def learn(self, task_id, dataset):
        self.old_net = copy.deepcopy(self.net)
        self.old_net.eval()
        for p in self.old_net.parameters():
            p.requires_grad = False

        loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=2)
        self.opt = self.init_optimizer()

        if len(self.prev_tasks) > 0:
            self.opt_cls = self.init_optimizer_classifier()
            if isinstance(self.net, SubnetVisionTransformer) or isinstance(self.net, VisionTransformer):
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt_cls, self.args.n_epochs)
            for epoch in range(self.args.n_epochs):
                for i, (x, y) in enumerate(loader):
                    x, y = x.to(self.device), y.to(self.device)
                    loss = self.loss_fn(self.forward(x, task_id), y)
                    self.opt_cls.zero_grad()
                    loss.backward()
                    self.opt_cls.step()
                if self.scheduler is not None:
                    self.scheduler.step()

        lsf_gamma = self.args.lsf_gamma
        ewc_lmbd = self.args.ewc_lmbd

        self.n_iters = self.args.n_epochs * len(loader)

        if isinstance(self.net, SubnetVisionTransformer) or isinstance(self.net, VisionTransformer):
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, self.args.n_epochs)

        for epoch in range(self.args.n_epochs):
            for i, (x, y) in enumerate(loader):
                x, y = x.to(self.device), y.to(self.device)

                target_shape = [x.shape[0]] + [1] * (len(x.shape) - 1)
                lsf_lmbd = torch.rand(*target_shape).to(x.device)
                y_idx = y
                hat_x = lsf_lmbd * x + (1 - lsf_lmbd) * self.mnemonic_code[y_idx]

                x_ = torch.cat([x, hat_x], 0)
                y_ = torch.cat([y, y], 0)
                loss = self.loss_fn(self.forward(x_, task_id), y_)

                n_prev_tasks = len(self.prev_tasks)
                if n_prev_tasks > 0:
                    loss += ewc_lmbd * self.penalty()

                    for t in self.prev_tasks:
                        loss += lsf_gamma * self.loss_fn(
                            self.forward(self.mnemonic_code[t * self.cpt:(t + 1) * self.cpt].view(-1, *self.dim_input), t),
                            self.mnemonic_target[t * self.cpt:(t + 1) * self.cpt]) / n_prev_tasks

                        # lwf
                        outputs = self.forward(x, t)[..., t * self.cpt:(t + 1) * self.cpt]
                        with torch.no_grad():
                            targets = self.old_net.forward(x)[..., t * self.cpt:(t + 1) * self.cpt]
                        loss += self.args.lwf_alpha * modified_kl_div(
                            smooth(F.softmax(targets, dim=-1), 2, 1),
                            smooth(F.softmax(outputs, dim=-1), 2, 1)) / n_prev_tasks

                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

            if self.scheduler is not None:
                self.scheduler.step()

        fish = torch.zeros_like(self.net.get_params())
        for j, (x, y) in enumerate(loader):
            x, y = x.to(self.device), y.to(self.device)
            for ex, lab in zip(x, y):
                self.opt.zero_grad()
                output = self.forward(ex.unsqueeze(0), task_id)
                loss = - F.nll_loss(F.log_softmax(output, dim=1), lab.unsqueeze(0), reduction='none')
                exp_cond_prob = torch.mean(torch.exp(loss.detach().clone()))
                loss = torch.mean(loss)
                loss.backward()
                fish += exp_cond_prob * self.net.get_grads() ** 2

        fish /= (len(loader) * self.args.batch_size)
        self.prev_tasks.append(task_id)
        self.fish[task_id] = fish
        self.checkpoint[task_id] = self.net.get_params().data.clone()

    def forget(self, task_id):
        self.prev_tasks.remove(task_id)
        del self.fish[task_id]
        del self.checkpoint[task_id]
        n_prev_tasks = len(self.prev_tasks)

        self.opt = self.init_optimizer()

        assert self.args.forget_iters is not None

        if self.args.forget_iters:
            for i in range(self.args.forget_iters):
                self.opt.zero_grad()
                out = self.forward(self.mnemonic_code[task_id * self.cpt:(task_id + 1) * self.cpt].view(-1, *self.dim_input), task_id)

                uniform_target = (torch.ones(out.shape) / self.cpt).to(self.device)
                if task_id > 0:
                    uniform_target[:, :self.cpt * task_id].data.fill_(0.0)
                if task_id < self.n_tasks - 1:
                    uniform_target[:, self.cpt * (task_id + 1):].data.fill_(0.0)

                loss = self.loss_fn(out, uniform_target)

                if n_prev_tasks > 0:
                    for t in self.prev_tasks:
                        loss += self.args.lsf_gamma * self.loss_fn(
                            self.forward(self.mnemonic_code[t * self.cpt:(t + 1) * self.cpt].view(-1, *self.dim_input), t),
                            self.mnemonic_target[t * self.cpt:(t + 1) * self.cpt]) / n_prev_tasks

                loss.backward()
                self.opt.step()
