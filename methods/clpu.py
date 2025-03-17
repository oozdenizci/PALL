import copy
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from .base import *
import models
from models.vit import VisionTransformer
from models.subnet_vit import SubnetVisionTransformer


class CLPU(Base):
    def __init__(self, args):
        super(CLPU, self).__init__(args)
        self.side_nets = {}

    def eval_mode(self):
        self.eval()
        for t in self.side_nets:
            self.side_nets[t].eval()

    def train_mode(self):
        self.train()
        for t in self.side_nets:
            self.side_nets[t].train()

    def init_optimizer_per_task(self, task_id):
        if self.args.optim == "sgd":
            return SGD(self.side_nets[task_id].parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        elif self.args.optim == "adam":
            return Adam(self.side_nets[task_id].parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise NotImplementedError

    def forward(self, x, task):
        pred = self.side_nets[task].forward(x) if task in self.side_nets else self.net.forward(x)
        if task > 0:
            pred[:, :self.cpt*task].data.fill_(-10e10)
        if task < self.n_tasks-1:
            pred[:, self.cpt*(task+1):].data.fill_(-10e10)
        return pred

    def forward_with_features(self, x, task):
        pred, features = self.side_nets[task].forward(x, returnt='all') if task in self.side_nets else self.net.forward(x, returnt='all')
        if task > 0:
            pred[:, :self.cpt * task].data.fill_(-10e10)
        if task < self.n_tasks - 1:
            pred[:, self.cpt * (task + 1):].data.fill_(-10e10)
        return pred, features

    def learn(self, task_id, dataset):
        assert task_id not in self.side_nets, f"[ERROR] should not see {task_id} in side nets"

        self.side_nets[task_id] = models.__dict__[self.args.arch.lower()](self.cpt * self.n_tasks).to(self.args.device)

        loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=2)
        opt = self.init_optimizer_per_task(task_id)

        if isinstance(self.net, SubnetVisionTransformer) or isinstance(self.net, VisionTransformer):
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.args.n_epochs)

        for epoch in range(self.args.n_epochs):
            for i, (x, y) in enumerate(loader):
                x, y = x.to(self.device), y.to(self.device)
                h = self.forward(x, task_id)
                loss = self.loss_fn(h, y)
                opt.zero_grad()
                loss.backward()
                opt.step()

            if self.scheduler is not None:
                self.scheduler.step()

    def forget(self, task_id):
        del self.side_nets[task_id]
