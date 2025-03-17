import torch
from torch.utils.data import DataLoader
from .base import Base
from models.vit import VisionTransformer
from models.subnet_vit import SubnetVisionTransformer


class Sequential(Base):
    def __init__(self, args):
        super(Sequential, self).__init__(args)

    def learn(self, task_id, dataset):
        loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=2)
        self.opt = self.init_optimizer()

        if isinstance(self.net, SubnetVisionTransformer) or isinstance(self.net, VisionTransformer):
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, self.args.n_epochs)

        for i in range(self.args.n_epochs):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                self.opt.zero_grad()
                pred = self.forward(x, task_id)
                loss = self.loss_fn(pred, y)
                loss.backward()
                self.opt.step()

            if self.scheduler is not None:
                self.scheduler.step()
