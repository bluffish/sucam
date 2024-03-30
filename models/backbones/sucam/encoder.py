import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import BasicBlock
from models.backbones.sucam.layers import *
from mmcv.ops import DeformConv2dPack as DCN

class DepthNet(nn.Module):

    def __init__(self, in_channels, mid_channels, depth_channels):
        super(DepthNet, self).__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.depth_conv = nn.Sequential(
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            ASPP(mid_channels, mid_channels),
            DCN(
                mid_channels, mid_channels,
                3, padding=1, groups=4, im2col_step=128,
            ),
            nn.Conv2d(mid_channels,
                      depth_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0),
        )

    def forward(self, x):
        
        x = self.reduce_conv(x)
        depth = self.depth_conv(x)
        
        return depth


class Encoder(nn.Module):
    def __init__(self, C, D, S, downsample=8):
        super().__init__()
        self.C = C
        self.D = D
        self.S = S

        self.downsample = downsample
        self.version = 'b4'

        self.backbone = EfficientNet.from_pretrained(f'efficientnet-{self.version}')
        self.delete_unused_layers()

        if self.downsample == 16:
            if self.version == 'b0':
                upsampling_in_channels = 320 + 112
            elif self.version == 'b4':
                upsampling_in_channels = 448 + 160
            upsampling_out_channels = 512

        elif self.downsample == 8:
            if self.version == 'b0':
                upsampling_in_channels = 112 + 40
            elif self.version == 'b4':
                upsampling_in_channels = 160 + 56
            upsampling_out_channels = 128

        else:
            raise ValueError(f'Downsample factor {self.downsample} not handled.')

        self.upsampling_layer = UpsamplingConcat(upsampling_in_channels, upsampling_out_channels)
        # self.depth_net = nn.Conv2d(upsampling_out_channels, self.C + self.D + 1, kernel_size=1, padding=0)
        self.depth_net = DepthNet(upsampling_out_channels, upsampling_out_channels, self.C + self.D)

    def delete_unused_layers(self):
        indices_to_delete = []
        for idx in range(len(self.backbone._blocks)):
            if self.downsample == 8:
                if self.version == 'b0' and idx > 10:
                    indices_to_delete.append(idx)
                if self.version == 'b4' and idx > 21:
                    indices_to_delete.append(idx)

        for idx in reversed(indices_to_delete):
            del self.backbone._blocks[idx]

        del self.backbone._conv_head
        del self.backbone._bn1
        del self.backbone._avg_pooling
        del self.backbone._dropout
        del self.backbone._fc

    def get_features(self, x):
        endpoints = dict()

        x = self.backbone._swish(self.backbone._bn0(self.backbone._conv_stem(x)))
        prev_x = x

        for idx, block in enumerate(self.backbone._blocks):
            drop_connect_rate = self.backbone._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.backbone._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = prev_x
            prev_x = x

            if self.downsample == 8:
                if self.version == 'b0' and idx == 10:
                    break
                if self.version == 'b4' and idx == 21:
                    break

        endpoints['reduction_{}'.format(len(endpoints) + 1)] = x

        if self.downsample == 16:
            input_1, input_2 = endpoints['reduction_5'], endpoints['reduction_4']
        elif self.downsample == 8:
            input_1, input_2 = endpoints['reduction_4'], endpoints['reduction_3']

        x = self.upsampling_layer(input_1, input_2)
        return x

    def forward(self, x, gt_depth=None):
        x = self.get_features(x)
        x = self.depth_net(x)

        if gt_depth != None:
            x = gt_depth.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)

            return x, None, gt_depth
        else:
            depth = x[:, :self.D].softmax(dim=1)

            x = depth.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)

            # uniform depth
            # x = torch.ones_like(depth.unsqueeze(1)) * x[:, self.D:(self.D + self.C)].unsqueeze(2)

            return x, None, depth