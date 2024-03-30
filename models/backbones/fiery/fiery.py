import numpy as np
import torch
import torch.nn as nn

from models.backbones.fiery.decoder import Decoder
from models.backbones.fiery.encoder import Encoder
from models.backbones.sucam.bev_pool.bev_pool import bev_pool
from tools.geometry import *


class VoxelsSumming(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geometry, ranks):
        x = x.cumsum(0)

        mask = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        mask[:-1] = ranks[1:] != ranks[:-1]

        x, geometry = x[mask], geometry[mask]
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        ctx.save_for_backward(mask)
        ctx.mark_non_differentiable(geometry)

        return x, geometry

    @staticmethod
    def backward(ctx, grad_x, grad_geometry):
        (mask,) = ctx.saved_tensors

        indices = torch.cumsum(mask, 0)
        indices[mask] -= 1

        output_grad = grad_x[indices]

        return output_grad, None, None


class Fiery(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()

        bev_resolution, bev_start_position, bev_dimension = calculate_birds_eye_view_parameters(
            [-50.0, 50.0, 0.5], [-50.0, 50.0, 0.5], [-10.0, 10.0, 20.0]
        )

        self.d_bound = [2.0, 50.0, 1.0]

        self.bev_resolution = nn.Parameter(torch.tensor(bev_resolution), requires_grad=False)
        self.bev_start_position = nn.Parameter(torch.tensor(bev_start_position), requires_grad=False)
        self.bev_dimension = nn.Parameter(torch.tensor(bev_dimension), requires_grad=False)

        self.encoder_downsample = 8
        self.encoder_out_channels = 64

        self.frustum = self.create_frustum()
        self.depth_channels, _, _, _ = self.frustum.shape

        self.bev_size = (200, 200)
        self.n_classes = n_classes

        self.encoder = Encoder(C=self.encoder_out_channels, D=self.depth_channels, downsample=self.encoder_downsample)
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

    def forward(self, image, intrinsics, extrinsics):
        x, depth = self.calculate_birds_eye_view_features(image, intrinsics, extrinsics)
        bev_output = self.decoder(x)

        return bev_output, None, depth

    def get_geometry(self, intrinsics, extrinsics):
        rotation, translation = extrinsics[..., :3, :3], extrinsics[..., :3, 3]

        B, N, _ = translation.shape

        points = self.frustum.unsqueeze(0).unsqueeze(0).unsqueeze(-1)

        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3], points[:, :, :, :, :, 2:3]), 5)
        combined_transformation = rotation.matmul(torch.inverse(intrinsics))
        points = combined_transformation.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += translation.view(B, N, 1, 1, 1, 3)

        return points

    def encoder_forward(self, x):
        b, n, c, h, w = x.shape

        x = x.view(b * n, c, h, w)
        x, depth = self.encoder(x)
        x = x.view(b, n, *x.shape[1:])
        x = x.permute(0, 1, 3, 4, 5, 2)

        return x, depth

    def projection_to_birds_eye_view(self, x, geometry):
        batch, n, d, h, w, c = x.shape
        output = torch.zeros((batch, c, self.bev_size[0], self.bev_size[1]), dtype=torch.float, device=x.device)

        N = n * d * h * w
        for b in range(batch):
            x_b = x[b].reshape(N, c)

            geometry_b = ((geometry[b] - (self.bev_start_position - self.bev_resolution / 2.0)) / self.bev_resolution)
            geometry_b = geometry_b.view(N, 3).long()

            mask = (
                    (geometry_b[:, 0] >= 0)
                    & (geometry_b[:, 0] < self.bev_dimension[0])
                    & (geometry_b[:, 1] >= 0)
                    & (geometry_b[:, 1] < self.bev_dimension[1])
                    & (geometry_b[:, 2] >= 0)
                    & (geometry_b[:, 2] < self.bev_dimension[2])
            )
            x_b = x_b[mask]
            geometry_b = geometry_b[mask]

            ranks = (
                    geometry_b[:, 0] * (self.bev_dimension[1] * self.bev_dimension[2])
                    + geometry_b[:, 1] * (self.bev_dimension[2])
                    + geometry_b[:, 2]
            )
            ranks_indices = ranks.argsort()
            x_b, geometry_b, ranks = x_b[ranks_indices], geometry_b[ranks_indices], ranks[ranks_indices]

            x_b, geometry_b = VoxelsSumming.apply(x_b, geometry_b, ranks)

            bev_feature = torch.zeros((self.bev_dimension[2], self.bev_dimension[0], self.bev_dimension[1], c),
                                      device=x_b.device)
            bev_feature[geometry_b[:, 2], geometry_b[:, 0], geometry_b[:, 1]] = x_b

            bev_feature = bev_feature.permute((0, 3, 1, 2))
            bev_feature = bev_feature.squeeze(0)

            output[b] = bev_feature

        return output

    def projection_to_birds_eye_view_fast(self, x, geom_feats):
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

    def calculate_birds_eye_view_features(self, x, intrinsics, extrinsics):
        geometry = self.get_geometry(intrinsics, extrinsics)

        x, depth = self.encoder_forward(x)
        x = self.projection_to_birds_eye_view(x, geometry)

        return x, depth