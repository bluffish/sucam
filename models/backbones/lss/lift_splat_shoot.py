from typing import Any

import numpy as np
import torch
from efficientnet_pytorch import EfficientNet
from torch import nn
from torchvision.models.resnet import resnet18
from models.gpn.density import Density, Evidence
from models.backbones.seg.unet import UNet


def gen_dx_bx(x_bound, y_bound, z_bound):
    dx = torch.Tensor([row[2] for row in [x_bound, y_bound, z_bound]])
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [x_bound, y_bound, z_bound]])
    nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [x_bound, y_bound, z_bound]])

    return dx, bx, nx


class QuickCumsum(torch.autograd.Function):
    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        pass

    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = (ranks[1:] != ranks[:-1])

        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        # save kept for backward
        ctx.save_for_backward(kept)

        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)

        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        kept, = ctx.saved_tensors

        back = torch.cumsum(kept, 0)
        back[kept] -= 1

        val = gradx[back]

        return val, None, None


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


class CamEncode(nn.Module):
    def __init__(self, D, C, inC=3):
        super(CamEncode, self).__init__()
        self.D = D
        self.C = C

        self.trunk = EfficientNet.from_pretrained("efficientnet-b0", in_channels=inC)

        self.up1 = Up(320 + 112, 512)
        self.depthnet = nn.Conv2d(512, self.D + self.C, kernel_size=1, padding=0)

    def get_depth_dist(self, x):
        return x.softmax(dim=1)

    def get_depth_feat(self, x):
        x = self.get_eff_depth(x)

        x = self.depthnet(x)

        depth = self.get_depth_dist(x[:, :self.D])
        # new_x = depth.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)
        new_x = depth.unsqueeze(1) * torch.ones_like(x[:, self.D:(self.D + self.C)].unsqueeze(2))

        return x[:, :self.D], new_x

    def get_eff_depth(self, x):
        endpoints = dict()

        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x

        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = prev_x
            prev_x = x

        endpoints['reduction_{}'.format(len(endpoints) + 1)] = x
        x = self.up1(endpoints['reduction_5'], endpoints['reduction_4'])
        return x

    def forward(self, x):
        depth, x = self.get_depth_feat(x)

        return x, depth


class BevEncode(nn.Module):
    def __init__(self, inC, outC):
        super(BevEncode, self).__init__()

        trunk = resnet18(weights=None, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64 + 256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x = self.layer2(x1)
        x = self.layer3(x)

        x = self.up1(x, x1)
        x = self.up2(x)

        return x


class BevEncodePostnet(nn.Module):
    def __init__(self, inC, outC):
        super(BevEncodePostnet, self).__init__()

        self.outC = outC

        trunk = resnet18(zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.latent_size = 16

        self.up1 = Up(64 + 256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(256, self.latent_size, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(self.latent_size),
            nn.ReLU(inplace=True),
        )

        self.flow = Density(dim_latent=self.latent_size, num_mixture_elements=outC)
        self.evidence = Evidence(scale='latent-new')

        self.last = nn.Conv2d(outC, outC, kernel_size=3, padding=1)

        # self.p_c = torch.tensor([.015, .2, .05, .735])
        self.p_c = torch.tensor([.02, .98])

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x = self.layer2(x1)
        x = self.layer3(x)

        x = self.up1(x, x1)
        x = self.up2(x)

        x = x.permute(0, 2, 3, 1).to(x.device)
        x = x.reshape(-1, self.latent_size)

        self.p_c = self.p_c.to(x.device)

        log_q_ft_per_class = self.flow(x) + self.p_c.view(1, -1).log()

        beta = self.evidence(
            log_q_ft_per_class, dim=self.latent_size,
            further_scale=2.0).exp()

        beta = beta.reshape(-1, 200, 200, self.outC).permute(0, 3, 1, 2).contiguous()
        beta = self.last(beta.log()).exp()

        return beta


class LiftSplatShoot(nn.Module):
    def __init__(
            self,
            x_bound=(-50.0, 50.0, 0.5),
            y_bound=(-50.0, 50.0, 0.5),
            z_bound=(-10.0, 10.0, 20.0),
            d_bound=(4.0, 45.0, 1.0),
            final_dim=(224, 480),
            n_classes=4,
            use_seg=False,
            use_dep=False,
            gt=False
    ):
        super(LiftSplatShoot, self).__init__()

        self.inC = 3

        if use_seg:
            self.seg = UNet(n_channels=3, n_classes=n_classes)
            self.inC += 4

        self.use_seg = use_seg
        self.use_dep = use_dep
        self.gt = gt

        self.d_bound = d_bound
        self.final_dim = final_dim
        self.outC = n_classes

        dx, bx, nx = gen_dx_bx(x_bound,
                               y_bound,
                               z_bound,
                               )

        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.downsample = 16
        self.camC = 64
        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape
        self.camencode = CamEncode(self.D, self.camC, inC=self.inC)
        self.bevencode = BevEncode(inC=self.camC, outC=self.outC)
        self.inter = None
    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.final_dim
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        ds = torch.arange(*self.d_bound, dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, intrinsics, extrinsics):
        rotation, translation = extrinsics[..., :3, :3], extrinsics[..., :3, 3]
        B, N, _ = translation.shape

        points = self.frustum.unsqueeze(0).unsqueeze(0).unsqueeze(-1)

        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3], points[:, :, :, :, :, 2:3]), 5)
        combined_transformation = rotation.matmul(torch.inverse(intrinsics))
        points = combined_transformation.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += translation.view(B, N, 1, 1, 1, 3)

        return points

    def get_cam_feats(self, x):
        """Return B x N x D x H/downsample x W/downsample x C
        """
        B, N, C, imH, imW = x.shape

        x = x.view(B * N, C, imH, imW)
        x, depth = self.camencode(x)
        x = x.view(B, N, self.camC, self.D, imH // self.downsample, imW // self.downsample)
        x = x.permute(0, 1, 3, 4, 5, 2)

        return x, depth

    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix,
                                         device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0]) \
               & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1]) \
               & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]

        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B) \
                + geom_feats[:, 1] * (self.nx[2] * B) \
                + geom_feats[:, 2] * B \
                + geom_feats[:, 3]
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]
        # cumsum trick

        x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        # griddify (B x C x Z x X x Y)
        final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x

        # collapse Z

        final = torch.cat(final.unbind(dim=2), 1)

        return final

    def get_voxels(self, images, intrinsics, extrinsics):
        geom = self.get_geometry(intrinsics, extrinsics)
        x, depth = self.get_cam_feats(images)

        x = self.voxel_pooling(geom, x)
        self.inter = x
        return x, depth

    def forward(self, images, intrinsics, extrinsics):
        if self.use_seg:
            seg = self.seg(images.view(-1, 3, 224, 480))
            images = torch.cat((images, seg.view(-1, 6, self.outC, 224, 480)), dim=2)
        else:
            seg = None

        x, depth = self.get_voxels(images, intrinsics, extrinsics)
        x = self.bevencode(x)

        return x, seg, depth

