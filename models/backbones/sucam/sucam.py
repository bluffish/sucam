import numpy as np
import torch
import torch.nn as nn

from models.backbones.sucam.bev_pool.bev_pool import bev_pool
from models.backbones.sucam.decoder import Decoder
from models.backbones.sucam.encoder import Encoder
from tools.geometry import *


class SUCAM(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()

        bev_resolution, bev_start_position, bev_dimension = calculate_birds_eye_view_parameters(
            [-50.0, 50.0, 0.5], [-50.0, 50.0, 0.5], [-10.0, 10.0, 20.0]
        )

        self.d_bound = [2.0, 58.0, .5]

        self.bev_resolution = nn.Parameter(torch.tensor(bev_resolution), requires_grad=False)
        self.bev_start_position = nn.Parameter(torch.tensor(bev_start_position), requires_grad=False)
        self.bev_dimension = nn.Parameter(torch.tensor(bev_dimension), requires_grad=False)

        self.encoder_downsample = 8
        self.encoder_out_channels = 64

        self.frustum = self.create_frustum()
        self.depth_channels, _, _, _ = self.frustum.shape

        self.bev_size = (200, 200)
        self.n_classes = n_classes

        self.encoder = Encoder(C=self.encoder_out_channels, D=self.depth_channels, S=n_classes, downsample=self.encoder_downsample)
        self.decoder = Decoder(in_channels=self.encoder_out_channels, n_classes=n_classes)

    def create_frustum(self):
        h, w = 224, 480
        dh, dw = h // self.encoder_downsample, w // self.encoder_downsample

        depth_grid = torch.arange(*self.d_bound, dtype=torch.float)
        depth_grid = depth_grid.view(-1, 1, 1).expand(-1, dh, dw)
        n_depth_slices = depth_grid.shape[0]

        x_grid = torch.linspace(0, w - 1, dw, dtype=torch.float)
        x_grid = x_grid.view(1, 1, dw).expand(n_depth_slices, dh, dw)
        y_grid = torch.linspace(0, h - 1, dh, dtype=torch.float)
        y_grid = y_grid.view(1, dh, 1).expand(n_depth_slices, dh, dw)

        frustum = torch.stack((x_grid, y_grid, depth_grid), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def forward(self, image, intrinsics, extrinsics, gt_depth=None):
        x, seg, depth = self.calculate_birds_eye_view_features(image, intrinsics, extrinsics, gt_depth=gt_depth)
        bev_output = self.decoder(x)

        return bev_output, seg, depth

    def get_geometry(self, intrinsics, extrinsics):
        rotation, translation = extrinsics[..., :3, :3], extrinsics[..., :3, 3]

        B, N, _ = translation.shape

        points = self.frustum.unsqueeze(0).unsqueeze(0).unsqueeze(-1)

        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3], points[:, :, :, :, :, 2:3]), 5)
        combined_transformation = rotation.matmul(torch.inverse(intrinsics))
        points = combined_transformation.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += translation.view(B, N, 1, 1, 1, 3)

        return points

    def encoder_forward(self, x, gt_depth=None):
        b, n, c, h, w = x.shape

        x = x.view(b * n, c, h, w)
        x, seg, depth = self.encoder(x, gt_depth=gt_depth)
        x = x.view(b, n, *x.shape[1:])
        x = x.permute(0, 1, 3, 4, 5, 2)

        return x, seg, depth

    def projection_to_birds_eye_view(self, x, geom_feats):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bev_start_position - self.bev_resolution / 2.)) / self.bev_resolution).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix,
                                         device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.bev_dimension[0]) \
               & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.bev_dimension[1]) \
               & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.bev_dimension[2])
        x = x[kept]

        geom_feats = geom_feats[kept]

        final = bev_pool(x, geom_feats, B, self.bev_dimension[2], self.bev_dimension[0], self.bev_dimension[1])

        final = torch.cat(final.unbind(dim=2), 1)

        return final

    def calculate_birds_eye_view_features(self, x, intrinsics, extrinsics, gt_depth=None):
        geometry = self.get_geometry(intrinsics, extrinsics)

        x, seg, depth = self.encoder_forward(x, gt_depth=gt_depth)
        x = self.projection_to_birds_eye_view(x, geometry)

        return x, seg, depth