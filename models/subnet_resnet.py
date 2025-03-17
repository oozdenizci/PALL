import math
import torch.nn.functional as F
from models.base import *
from .subnet_layers import SubnetConv2d, SubnetLinear, SubnetClassifier

__all__ = ['SubnetResNet', 'subnet_resnet18', 'subnet_resnet34', 'subnet_resnet50']


def subnet_conv3x3(in_planes, out_planes, stride=1, sparsity=0.8):
    return SubnetConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False, sparsity=sparsity)


def subnet_conv1x1(in_planes, out_planes, stride=1, sparsity=0.8):
    return SubnetConv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, sparsity=sparsity)


class maskedSequential(nn.Sequential):
    def forward(self, *inputs):
        x = inputs[0]
        mask = inputs[1]
        mode = inputs[2]
        for module in self._modules.values():
            if isinstance(module, SubnetBasicBlock) or isinstance(module, SubnetBottleneck):
                x = module(x, mask, mode)
            else:
                x = module(x)
        return x


class SubnetBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, sparsity=0.8, name=""):
        super(SubnetBasicBlock, self).__init__()
        self.name = name
        self.conv1 = subnet_conv3x3(in_planes, planes, stride, sparsity=sparsity)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=False, affine=False)
        self.conv2 = subnet_conv3x3(planes, planes, sparsity=sparsity)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=False, affine=False)

        self.shortcut = None
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = 1
            self.conv3 = subnet_conv1x1(in_planes, self.expansion * planes, stride=stride, sparsity=sparsity)
            self.bn3 = nn.BatchNorm2d(self.expansion * planes, track_running_stats=False, affine=False)
        self.count = 0

    def forward(self, x, mask, mode='train'):
        name = self.name + ".conv1"
        out = F.relu(self.bn1(self.conv1(x, weight_mask=mask[name + '.weight'], bias_mask=mask[name + '.bias'], mode=mode)))
        name = self.name + ".conv2"
        out = self.bn2(self.conv2(out, weight_mask=mask[name + '.weight'], bias_mask=mask[name + '.bias'], mode=mode))
        if self.shortcut is not None:
            name = self.name + ".conv3"
            out += self.bn3(self.conv3(x, weight_mask=mask[name + '.weight'], bias_mask=mask[name + '.bias'], mode=mode))
        else:
            out += x
        out = F.relu(out)
        return out


class SubnetBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, sparsity=0.8, name=""):
        super(SubnetBottleneck, self).__init__()
        self.name = name
        self.conv1 = subnet_conv1x1(in_planes, planes, sparsity=sparsity)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=False, affine=False)
        self.conv2 = subnet_conv3x3(planes, planes, stride, sparsity=sparsity)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=False, affine=False)
        self.conv3 = subnet_conv1x1(planes, planes * self.expansion, sparsity=sparsity)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, track_running_stats=False, affine=False)

        self.shortcut = None
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = 1
            self.conv4 = subnet_conv1x1(in_planes, self.expansion * planes, stride=stride, sparsity=sparsity)
            self.bn4 = nn.BatchNorm2d(self.expansion * planes, track_running_stats=False, affine=False)
        self.count = 0

    def forward(self, x, mask, mode='train'):
        name = self.name + ".conv1"
        out = F.relu(self.bn1(self.conv1(x, weight_mask=mask[name + '.weight'], bias_mask=mask[name + '.bias'], mode=mode)))
        name = self.name + ".conv2"
        out = F.relu(self.bn2(self.conv2(out, weight_mask=mask[name + '.weight'], bias_mask=mask[name + '.bias'], mode=mode)))
        name = self.name + ".conv3"
        out = self.bn3(self.conv3(out, weight_mask=mask[name + '.weight'], bias_mask=mask[name + '.bias'], mode=mode))
        if self.shortcut is not None:
            name = self.name + ".conv4"
            out += self.bn4(self.conv4(x, weight_mask=mask[name + '.weight'], bias_mask=mask[name + '.bias'], mode=mode))
        else:
            out += x
        out = F.relu(out)
        return out


class SubnetResNet(BaseModel):
    def __init__(self, block, num_blocks, num_classes, nf, n_tasks, sparsity):
        super(SubnetResNet, self).__init__()
        self.in_planes = nf
        self.block = block
        self.num_classes = num_classes
        self.nf = nf
        self.conv1 = subnet_conv3x3(3, nf * 1, 1, sparsity=sparsity)
        self.bn1 = nn.BatchNorm2d(nf * 1, track_running_stats=False, affine=False)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1, sparsity=sparsity, name="layer1")
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2, sparsity=sparsity, name="layer2")
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2, sparsity=sparsity, name="layer3")
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2, sparsity=sparsity, name="layer4")
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = SubnetClassifier(nf * 8 * block.expansion, num_classes, n_tasks, bias=False)

        self.none_masks = {}
        for name, module in self.named_modules():
            if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d):
                self.none_masks[name + '.weight'] = None
                self.none_masks[name + '.bias'] = None

    def _make_layer(self, block, planes, num_blocks, stride, sparsity, name):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        name_count = 0
        for stride in strides:
            new_name = name + "." + str(name_count)
            layers.append(block(self.in_planes, planes, stride, sparsity, new_name))
            self.in_planes = planes * block.expansion
            name_count += 1
        return maskedSequential(*layers)

    def forward(self, x, task=-1, mask=None, mode="train", returnt='out'):
        if mask is None:
            mask = self.none_masks

        out = F.relu(self.bn1(self.conv1(x, weight_mask=mask['conv1.weight'], bias_mask=mask['conv1.bias'], mode=mode)))
        out = self.layer1(out, mask, mode)
        out = self.layer2(out, mask, mode)
        out = self.layer3(out, mask, mode)
        out = self.layer4(out, mask, mode)
        out = self.avgpool(out)
        feature = out.view(out.size(0), -1)

        if returnt == 'features':
            return feature

        out = self.classifier(feature, task)

        if returnt == 'out':
            return out
        elif returnt == 'all':
            return out, feature

        raise NotImplementedError("Unknown return type")

    def get_masks(self, task_id):
        task_mask = {}
        for name, module in self.named_modules():
            if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d):
                task_mask[name + '.weight'] = (module.weight_mask.detach().clone() > 0).type(torch.uint8)
                if getattr(module, 'bias') is not None:
                    task_mask[name + '.bias'] = (module.bias_mask.detach().clone() > 0).type(torch.uint8)
                else:
                    task_mask[name + '.bias'] = None
            elif isinstance(module, SubnetClassifier):
                task_mask[name + '.weight'] = module.wired_masks[task_id].detach().clone().to(module.weight.device)
        return task_mask

    def reinit_scores(self):
        print("reinitializing scores")
        for name, module in self.named_modules():
            if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d):
                module.init_mask_parameters()

    def reinit_weights(self, combined_masks):
        print("reinitializing weights")
        for name, module in self.named_modules():
            if isinstance(module, SubnetLinear) or isinstance(module, SubnetConv2d) or isinstance(module, SubnetClassifier):
                if combined_masks != {}:
                    module.weight.data = torch.where(combined_masks[name + '.weight'], module.weight,
                                                     nn.init.kaiming_uniform_(torch.zeros_like(module.weight), a=math.sqrt(5)))
                    if module.bias is not None:
                        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                        bound = 1 / math.sqrt(fan_in)
                        module.bias.data = torch.where(combined_masks[name + '.bias'], module.bias,
                                                       nn.init.uniform_(torch.zeros_like(module.bias), -bound, bound))
                else:
                    module.weight.data = nn.init.kaiming_uniform_(torch.zeros_like(module.weight), a=math.sqrt(5))
                    if module.bias is not None:
                        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                        bound = 1 / math.sqrt(fan_in)
                        module.bias.data = nn.init.uniform_(torch.zeros_like(module.bias), -bound, bound)


def subnet_resnet18(num_classes, nf=64, norm_params=False, n_tasks=1, sparsity=0.8):
    return SubnetResNet(SubnetBasicBlock, [2, 2, 2, 2], num_classes, nf, n_tasks=n_tasks, sparsity=sparsity)


def subnet_resnet34(num_classes, nf=64, norm_params=False, n_tasks=1, sparsity=0.8):
    return SubnetResNet(SubnetBasicBlock, [3, 4, 6, 3], num_classes, nf, n_tasks=n_tasks, sparsity=sparsity)


def subnet_resnet50(num_classes, nf=64, norm_params=False, n_tasks=1, sparsity=0.8):
    return SubnetResNet(SubnetBottleneck, [3, 4, 6, 3], num_classes, nf, n_tasks=n_tasks, sparsity=sparsity)
