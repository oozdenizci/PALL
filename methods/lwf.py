import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .base import *
from models.vit import VisionTransformer
from models.subnet_vit import SubnetVisionTransformer


def smooth(logits, temp, dim):
    log = logits ** (1 / temp)
    return log / torch.sum(log, dim).unsqueeze(1)


def modified_kl_div(old, new):
    return -torch.mean(torch.sum(old * torch.log(new), 1))


class LwF(Base):
    def __init__(self, args):
        super(LwF, self).__init__(args)
        self.old_net = None

    def init_optimizer_classifier(self):
        if self.args.optim == "sgd":
            return SGD(self.net.classifier.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        elif self.args.optim == "adam":
            return Adam(self.net.classifier.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise NotImplementedError

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

        if isinstance(self.net, SubnetVisionTransformer) or isinstance(self.net, VisionTransformer):
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, self.args.n_epochs)

        for epoch in range(self.args.n_epochs):
            for i, (x, y) in enumerate(loader):
                x, y = x.to(self.device), y.to(self.device)
                loss = self.loss_fn(self.forward(x, task_id), y)

                # current loss
                n_prev_tasks = len(self.prev_tasks)
                if n_prev_tasks > 0:
                    for t in self.prev_tasks:
                        outputs = self.forward(x, t)[..., t * self.cpt:(t + 1) * self.cpt]
                        with torch.no_grad():
                            targets = self.old_net.forward(x)[..., t * self.cpt:(t + 1) * self.cpt]
                        loss += self.args.lwf_alpha * modified_kl_div(
                            smooth(F.softmax(targets, dim=-1), self.args.lwf_temp, 1),
                            smooth(F.softmax(outputs, dim=-1), self.args.lwf_temp, 1)) / n_prev_tasks
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

            if self.scheduler is not None:
                self.scheduler.step()

        self.prev_tasks.append(task_id)

    def forget(self, task_id):
        self.prev_tasks.remove(task_id)
