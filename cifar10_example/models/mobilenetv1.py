import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, input_channels, output_channels, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=output_channels,
                               kernel_size=3, stride=stride, padding=1, groups=input_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(input_channels)
        self.conv2 = nn.Conv2d(in_channels=input_channels, out_channels=output_channels,
                               kernel_size=1, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(output_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class MobileNetV1(nn.Module):

    cfg = [64, (128, 2), 128, (256, 2), 256, (512, 2), 512, 512, 512, 512, 512, (1024, 2), 1024]
    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.linear = nn.Linear(1024, 10)

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
