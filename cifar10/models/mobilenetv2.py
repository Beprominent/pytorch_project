import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# MobileNetV2 Block
# expansion + depthwise + point-wise
class Block(nn.Module):
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride
        expand_plane = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, expand_plane, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_plane)
        self.conv2 = nn.Conv2d(expand_plane, expand_plane, kernel_size=3, stride=self.stride, padding=1, groups=expand_plane,  bias=False)
        self.bn2 = nn.BatchNorm2d(expand_plane)
        self.conv3 = nn.Conv2d(expand_plane, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)
        self.shortcut = nn.Sequential()
        if self.stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out

class MobileNetV2(nn.Module):
    # (expansion, num_output, num_blocks, strides)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # It is different from the paper, but the performance is better than the origin
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]
    def __init__(self):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.laysers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, 10)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.laysers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def _make_layers(self, in_planes):
        layers = []
        for expansion, num_output, num_blocks, strides in self.cfg:
            strides = [strides] + [1] * (num_blocks -1)
            for stride in strides:
                layers.append(Block(in_planes, num_output, expansion, stride))
                in_planes = num_output
        return nn.Sequential(*layers)

def test():
    net = MobileNetV2()
    x = Variable(torch.randn(1, 3, 32, 32))
    y = net(x)
    print(y.size())
