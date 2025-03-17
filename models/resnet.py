import torch.nn.functional as F
from models.base import *

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50']


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm_params=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=norm_params, affine=norm_params)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=norm_params, affine=norm_params)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                conv1x1(in_planes, self.expansion * planes, stride=stride),
                nn.BatchNorm2d(self.expansion * planes, track_running_stats=norm_params, affine=norm_params)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, norm_params=False):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm2d(planes, track_running_stats=norm_params, affine=norm_params)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes, track_running_stats=norm_params, affine=norm_params)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, track_running_stats=norm_params, affine=norm_params)
        self.stride = stride

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                conv1x1(in_planes, self.expansion * planes, stride=stride),
                nn.BatchNorm2d(self.expansion * planes, track_running_stats=norm_params, affine=norm_params)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(BaseModel):
    def __init__(self, block, num_blocks, num_classes, nf, norm_params=False):
        super(ResNet, self).__init__()
        self.in_planes = nf
        self.block = block
        self.num_classes = num_classes
        self.nf = nf
        self.norm_params = norm_params
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1, track_running_stats=self.norm_params, affine=self.norm_params)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self._features = nn.Sequential(self.conv1,
                                       self.bn1,
                                       nn.ReLU(),
                                       self.layer1,
                                       self.layer2,
                                       self.layer3,
                                       self.layer4)
        self.classifier = nn.Linear(nf * 8 * block.expansion, num_classes, bias=False)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.norm_params))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, returnt='out'):
        out = F.relu(self.bn1(self.conv1(x)))  # 64, 32, 32
        out = self.layer1(out)  # -> 64, 32, 32
        out = self.layer2(out)  # -> 128, 16, 16
        out = self.layer3(out)  # -> 256, 8, 8
        out = self.layer4(out)  # -> 512, 4, 4
        out = self.avgpool(out) # -> 512, 1, 1
        feature = out.view(out.size(0), -1)  # 512

        if returnt == 'features':
            return feature

        out = self.classifier(feature)

        if returnt == 'out':
            return out
        elif returnt == 'all':
            return out, feature

        raise NotImplementedError("Unknown return type")


def resnet18(num_classes, nf=64, norm_params=False, n_tasks=1, sparsity=None):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, nf, norm_params)


def resnet34(num_classes, nf=64, norm_params=False, n_tasks=1, sparsity=None):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, nf, norm_params)


def resnet50(num_classes, nf=64, norm_params=False, n_tasks=1, sparsity=None):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, nf, norm_params)
