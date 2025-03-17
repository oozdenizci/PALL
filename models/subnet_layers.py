import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from copy import deepcopy


class MaskByScores(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, zeros, ones, sparsity):
        k = 1 + round(float(sparsity) * (scores.numel() - 1))
        k_val = scores.view(-1).kthvalue(k).values.item()
        return torch.where(scores < k_val, zeros.to(scores.device), ones.to(scores.device))

    @staticmethod
    def backward(ctx, g):
        return g, None, None, None


class SubnetClassifier(nn.Linear):
    def __init__(self, in_features, out_features, n_tasks, bias=False):
        super(self.__class__, self).__init__(in_features=in_features, out_features=out_features, bias=bias)

        self.n_tasks = n_tasks
        self.cpt = out_features//n_tasks

        self.wired_masks = {-1: torch.ones_like(self.weight, requires_grad=False)}
        for t in range(self.n_tasks):
            class_wired_mask = torch.zeros_like(self.weight, requires_grad=False)
            class_wired_mask[t * self.cpt:(t + 1) * self.cpt, :] = 1
            self.wired_masks[t] = class_wired_mask.type(torch.uint8)

        self.register_parameter('bias', None)
        self.reset_parameters()

    def forward(self, x, task=-1):
        return F.linear(input=x, weight=self.wired_masks[task].to(self.weight.device) * self.weight, bias=None)


class SubnetLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, sparsity=0.8):
        super(self.__class__, self).__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.sparsity = sparsity

        # Mask Parameters of Weights and Bias
        self.w_m = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_mask = None
        self.zeros_weight, self.ones_weight = torch.zeros(self.w_m.shape), torch.ones(self.w_m.shape)
        if bias:
            self.b_m = nn.Parameter(torch.empty(out_features))
            self.bias_mask = None
            self.zeros_bias, self.ones_bias = torch.zeros(self.b_m.shape), torch.ones(self.b_m.shape)
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        self.init_mask_parameters()

    def forward(self, x, weight_mask=None, bias_mask=None, mode="train"):
        # If training, Get the subnet by sorting the scores
        if mode == "train":
            self.weight_mask = MaskByScores.apply(self.w_m.abs(), self.zeros_weight, self.ones_weight, self.sparsity)
            w_pruned = self.weight_mask * self.weight
            b_pruned = None
            if self.bias is not None:
                self.bias_mask = MaskByScores.apply(self.b_m.abs(), self.zeros_bias, self.ones_bias, self.sparsity)
                b_pruned = self.bias_mask * self.bias

        # If inference, no need to compute the subnetwork
        elif mode == "test":
            w_pruned = weight_mask * self.weight
            b_pruned = bias_mask * self.bias if self.bias is not None else None

        # No masking applied to weights - to evaluate unlearned tasks
        elif mode == "no_mask":
            w_pruned = self.weight
            b_pruned = self.bias if self.bias is not None else None

        else:
            raise Exception("[ERROR] The mode " + str(mode) + " is not supported!")

        return F.linear(input=x, weight=w_pruned, bias=b_pruned)

    def init_mask_parameters(self):
        nn.init.kaiming_uniform_(self.w_m, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w_m)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.b_m, -bound, bound)


class SubnetConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False, sparsity=0.8):
        super(self.__class__, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                             kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.stride = stride
        # self.padding = padding
        self.sparsity = sparsity

        # Mask Parameters of Weight and Bias
        self.w_m = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.weight_mask = None
        self.zeros_weight, self.ones_weight = torch.zeros(self.w_m.shape), torch.ones(self.w_m.shape)
        if bias:
            self.b_m = nn.Parameter(torch.empty(out_channels))
            self.bias_mask = None
            self.zeros_bias, self.ones_bias = torch.zeros(self.b_m.shape), torch.ones(self.b_m.shape)
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        self.init_mask_parameters()

    def forward(self, x, weight_mask=None, bias_mask=None, mode="train", epoch=1):
        # If training, Get the subnet by sorting the scores
        if mode == "train":
            self.weight_mask = MaskByScores.apply(self.w_m.abs(), self.zeros_weight, self.ones_weight, self.sparsity)
            w_pruned = self.weight_mask * self.weight
            b_pruned = None
            if self.bias is not None:
                self.bias_mask = MaskByScores.apply(self.b_m.abs(), self.zeros_bias, self.ones_bias, self.sparsity)
                b_pruned = self.bias_mask * self.bias

        # If inference/test, no need to compute the subnetwork
        elif mode == "test":
            w_pruned = weight_mask * self.weight
            b_pruned = bias_mask * self.bias if self.bias is not None else None

        # No masking applied to weights - to evaluate unlearned tasks
        elif mode == "no_mask":
            w_pruned = self.weight
            b_pruned = self.bias if self.bias is not None else None

        else:
            raise Exception("[ERROR] The mode " + str(mode) + " is not supported!")

        return F.conv2d(input=x, weight=w_pruned, bias=b_pruned, stride=self.stride, padding=self.padding)

    def init_mask_parameters(self):
        nn.init.kaiming_uniform_(self.w_m, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w_m)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.b_m, -bound, bound)
