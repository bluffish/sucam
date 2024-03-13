import torch.nn as nn
from torchvision.models.resnet import resnet18

from models.backbones.fiery.layers import UpsamplingAdd
from models.gpn.density import Density, Evidence
import torch


class Decoder(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()
        backbone = resnet18(weights=None, zero_init_residual=True)
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


class DecoderPostnet(nn.Module):
    def __init__(self, in_channels, n_classes):
        super().__init__()
        backbone = resnet18(weights=None, zero_init_residual=True)
        self.first_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1 = backbone.bn1
        self.relu = backbone.relu

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.n_classes = n_classes

        shared_out_channels = in_channels
        self.up3_skip = UpsamplingAdd(256, 128, scale_factor=2)
        self.up2_skip = UpsamplingAdd(128, 64, scale_factor=2)
        self.up1_skip = UpsamplingAdd(64, shared_out_channels, scale_factor=2)
        self.latent_size = 16

        self.segmentation_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, self.latent_size, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.latent_size),
            nn.ReLU(inplace=True),
        )

        self.flow = Density(dim_latent=self.latent_size, num_mixture_elements=n_classes)
        self.evidence = Evidence(scale='latent-new')

        self.last = nn.Conv2d(n_classes, n_classes, kernel_size=3, padding=1)

        self.p_c = torch.tensor([.015, .2, .05, .735])

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
        x = self.segmentation_head(x)

        x = x.permute(0, 2, 3, 1).to(x.device)
        x = x.reshape(-1, self.latent_size)

        self.p_c = self.p_c.to(x.device)

        log_q_ft_per_class = self.flow(x) + self.p_c.view(1, -1).log()

        beta = self.evidence(
            log_q_ft_per_class, dim=self.latent_size,
            further_scale=2.0).exp()

        beta = beta.reshape(-1, 200, 200, self.n_classes).permute(0, 3, 1, 2).contiguous()
        beta = self.last(beta.log()).exp()

        return beta
