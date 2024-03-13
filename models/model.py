import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbones.cvt.cross_view_transformer import CrossViewTransformer
from models.backbones.fiery.fiery import Fiery
from models.backbones.lss.lift_splat_shoot import LiftSplatShoot
from models.backbones.midas.midas_net import MidasNet
from models.backbones.seg.unet import UNet
from tools.loss import ce_loss
import math

backbones = {
    'fiery': Fiery,
    'cvt': CrossViewTransformer,
    'lss': LiftSplatShoot
}

def bin_depths(depth_map, mode, depth_min, depth_max, num_bins, target=False):
    """
    Converts depth map into bin indices
    Args:
        depth_map [torch.Tensor(H, W)]: Depth Map
        mode [string]: Discretiziation mode (See https://arxiv.org/pdf/2005.13423.pdf for more details)
            UD: Uniform discretiziation
            LID: Linear increasing discretiziation
            SID: Spacing increasing discretiziation
        depth_min [float]: Minimum depth value
        depth_max [float]: Maximum depth value
        num_bins [int]: Number of depth bins
        target [bool]: Whether the depth bins indices will be used for a target tensor in loss comparison
    Returns:
        indices [torch.Tensor(H, W)]: Depth bin indices
    """
    if mode == "UD":
        bin_size = (depth_max - depth_min) / num_bins
        indices = ((depth_map - depth_min) / bin_size)
    elif mode == "LID":
        bin_size = 2 * (depth_max - depth_min) / (num_bins * (1 + num_bins))
        indices = -0.5 + 0.5 * torch.sqrt(1 + 8 * (depth_map - depth_min) / bin_size)
    elif mode == "SID":
        indices = num_bins * (torch.log(1 + depth_map) - math.log(1 + depth_min)) / \
            (math.log(1 + depth_max) - math.log(1 + depth_min))
    else:
        raise NotImplementedError

    if target:
        # Remove indicies outside of bounds
        mask = (indices < 0) | (indices > num_bins) | (~torch.isfinite(indices))
        indices[mask] = num_bins

        # Convert to integer
        indices = indices.type(torch.int64)
    return F.one_hot(indices).permute(0, 1, 4, 2, 3)


class Model(nn.Module):
    def __init__(self,
                 devices,
                 backbone='fiery',
                 n_classes=4,
                 opt=None,
                 scaler=None,
                 loss_type='ce',
                 weights=None,
                 use_seg=False,
                 use_dep=False,
                 gt=False
                 ):
        super(Model, self).__init__()

        self.device = devices[0]
        self.devices = devices

        self.weights = weights

        if self.weights is not None:
            self.weights = self.weights.to(self.device)

        self.backbone = None

        self.loss_type = loss_type
        self.n_classes = n_classes
        self.opt = opt
        self.scaler = scaler
        self.gamma = .1
        self.tsne = False

        self.use_seg = use_seg
        self.use_dep = use_dep
        self.gt = gt

        self.create_backbone(backbone)

        print(self.device)

    def create_backbone(self, backbone):
        self.backbone = nn.DataParallel(
            backbones[backbone](n_classes=self.n_classes, use_seg=self.use_seg, use_dep=self.use_dep, gt=self.gt).to(self.device),
            output_device=self.device,
            device_ids=self.devices,
        )

    @staticmethod
    def aleatoric(x):
        pass

    @staticmethod
    def epistemic(x):
        pass

    @staticmethod
    def activate(x):
        pass

    @staticmethod
    def loss(x, gt):
        pass

    def state_dict(self, epoch=-1):
        return {
            'model_state_dict': super().state_dict(),
            'optimizer_state_dict': self.opt.state_dict() if self.opt is not None else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler is not None else None,
            'epoch': epoch
        }

    def load(self, state_dict):
        self.load_state_dict(state_dict['model_state_dict'])

        if self.opt is not None:
            self.opt.load_state_dict(state_dict['optimizer_state_dict'])

        if self.scaler is not None:
            self.scaler.load_state_dict(state_dict['scaler_state_dict'])

    def save(self, path):
        torch.save(self.state_dict(), path)

    def train_step(self, images, segs, depths, intrinsics, extrinsics, labels):
        self.opt.zero_grad(set_to_none=True)

        outs, seg_outs, depth_outs = self(images, intrinsics, extrinsics)
        loss = self.loss(outs, labels.to(self.device))

        if seg_outs is not None:
            loss += self.loss(seg_outs, segs.to(self.device).view(-1, 4, 224, 480))

        if self.use_dep:
            depths = depths.to(self.device)
            b, c = depths.shape[:2]
            # downscaled = depths.reshape(b, c, 14, 16, 30, 16).mean(dim=3).mean(dim=4)
            # downscaled = downscaled.floor().long()
            # downscaled = downscaled.floor().long().clamp(min=0, max=40)
            binned = torch.nn.functional.one_hot(depths, num_classes=41).permute(0, 1, 4, 2, 3)
            # binned = bin_depths(downscaled, "SID", 0, 40, 40, target=True)
            depth_loss = ce_loss(depth_outs, binned.view(b*c, 41, 14, 30).float()).mean()
            loss += depth_loss

        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 5.0)
        self.opt.step()

        return outs, seg_outs, loss

    def forward(self, images, intrinsics, extrinsics):
        return self.backbone(images, intrinsics, extrinsics)
