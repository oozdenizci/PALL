import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .base import *
from models.vit import VisionTransformer
from models.subnet_vit import SubnetVisionTransformer


class EWC(Base):
    def __init__(self, args):
        super(EWC, self).__init__(args)
        self.checkpoint = {}
        self.fish = {}

    def penalty(self):
        if len(self.prev_tasks) == 0:
            return torch.tensor(0.0).to(self.device)
        current_param = self.net.get_params()
        penalty = 0.0
        for t in self.prev_tasks:
            penalty += (self.fish[t] * (current_param - self.checkpoint[t]).pow(2)).sum()
        return penalty

    def learn(self, task_id, dataset):
        loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=2)
        self.opt = self.init_optimizer()

        if isinstance(self.net, SubnetVisionTransformer) or isinstance(self.net, VisionTransformer):
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, self.args.n_epochs)

        for epoch in range(self.args.n_epochs):
            for i, (x, y) in enumerate(loader):
                x, y = x.to(self.device), y.to(self.device)
                loss = self.loss_fn(self.forward(x, task_id), y)
                if task_id > 0:
                    loss += self.args.ewc_lmbd * self.penalty()
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
        assert task_id in self.prev_tasks, f"[ERROR] {task_id} not seen before"
        self.prev_tasks.remove(task_id)
        del self.fish[task_id]
        del self.checkpoint[task_id]
