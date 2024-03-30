import torch.nn as nn

from torchvision.models.resnet import resnet18
from torchvision.models.resnet import resnet34
from torchvision.models.resnet import resnet50

from models.backbones.sucam.layers import UpsamplingAdd
import torch


class Decoder(nn.Module):
    def __init__(self, in_channels, n_classes, version="18"):
        super().__init__()

        if version == "18":
            backbone = resnet18(weights=None, zero_init_residual=True)
        elif version == "34":
            backbone = resnet34(weights=None, zero_init_residual=True)
        elif version == "50":
            backbone = resnet50(weights=None, zero_init_residual=True)

        self.first_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1 = backbone.bn1
        self.relu = backbone.relu

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        shared_out_channels = in_channels
        self.up3_skip = UpsamplingAdd(256, 128, scale_factor=2)
        self.up2_skip = UpsamplingAdd(128, 64, scale_factor=2)
        self.up1_skip = UpsamplingAdd(64, shared_out_channels, scale_factor=2)

        self.segmentation_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, n_classes, kernel_size=1, padding=0),
        )

    def forward(self, x):
        skip_x = {'1': x}
        x = self.first_conv(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        skip_x['2'] = x
        x = self.layer2(x)
        skip_x['3'] = x

        x = self.layer3(x)
        x = self.up3_skip(x, skip_x['3'])
        x = self.up2_skip(x, skip_x['2'])
        x = self.up1_skip(x, skip_x['1'])

        return self.segmentation_head(x)

