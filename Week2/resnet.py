import torch
import torch.nn as nn


class VanillaBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        self.conv1 = nn.Conv2d(in_channels,out_channels,3,padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.Relu()
        self.conv2 = nn.Conv2d(out_channels,out_channels,3,padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward_0(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,3,padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.Relu()
        self.conv2 = nn.Conv2d(out_channels,out_channels,3,padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels,out_channels,1)
        else:
            self.shortcut + nn.Identity()

    def forward_1(self,x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)

        return out