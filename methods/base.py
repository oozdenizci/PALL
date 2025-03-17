import torch
import torch.nn as nn
from torch.optim import SGD, Adam
import numpy as np
import models
import time


class Base(nn.Module):
    def __init__(self, args):
        super(Base, self).__init__()
        self.args = args
        self.net = models.__dict__[args.arch](args.class_per_task * args.n_tasks,
                                              n_tasks=args.n_tasks,
                                              sparsity=args.sparsity,
                                              norm_params=args.norm_params)
        self.device = args.device
        self.n_tasks = args.n_tasks
        self.cpt = args.class_per_task
        self.lr = args.lr
        self.momentum = args.momentum
        self.weight_decay = args.weight_decay
        self.task_status = {}
        self.prev_tasks = []
        self.n_iters = 1
        self.loss_fn = nn.CrossEntropyLoss()
        self.loss_fn_reduction_none = nn.CrossEntropyLoss(reduction='none')
        self.opt = None
        self.scheduler = None

    def init_optimizer(self):
        if self.args.optim == "sgd":
            return SGD(self.net.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        elif self.args.optim == "adam":
            return Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise NotImplementedError

    def forward(self, x, task):
        out = self.net(x)
        if task > 0:
            out[:, :self.cpt * task].data.fill_(-10e10)
        if task < self.n_tasks - 1:
            out[:, self.cpt * (task + 1):].data.fill_(-10e10)
        return out

    def forward_with_features(self, x, task):
        out, features = self.net(x, returnt='all')
        if task > 0:
            out[:, :self.cpt * task].data.fill_(-10e10)
        if task < self.n_tasks - 1:
            out[:, self.cpt * (task + 1):].data.fill_(-10e10)
        return out, features

    def evaluate(self, x, task):
        return self.forward(x, task)  # default to the forward pass

    def eval_mode(self):
        self.eval()

    def train_mode(self):
        self.train()

    def learn(self, task_id, dataset):
        return  # default: do nothing when we want to learn a task

    def forget(self, task_id):
        return  # default: do nothing when we want to forget a task

    def privacy_aware_lifelong_learning(self, task_id, dataset, learn_type):
        t0 = time.time()
        if learn_type == "T":
            if task_id not in self.task_status:  # first time learning the task
                self.task_status[task_id] = learn_type
                self.learn(task_id, dataset)
            else:  # second time consolidate - we do not explore the impact of repetition yet
                raise NotImplementedError
        else:  # learn type is "F" forget
            assert learn_type == "F", f"[ERROR] unknown learning type {learn_type}"
            assert task_id in self.task_status, f"[ERROR] {task_id} was not learned"
            self.task_status[task_id] = "F"
            self.forget(task_id)
        # print('Processed in {} seconds'.format(time.time() - t0))
