import torch
import torch.nn as nn
import torch.nn.functional as F

from tools.geometry import *
from torch.cuda.amp.autocast_mode import autocast

from models.backbones.cvt.cross_view_transformer import CrossViewTransformer
from models.backbones.fiery.fiery import Fiery
from models.backbones.sucam.sucam import SUCAM
from models.backbones.lss.lift_splat_shoot import LiftSplatShoot
from tools.loss import DepthLoss
from tools.utils import *
from fvcore.nn import sigmoid_focal_loss

backbones = {
    'fiery': Fiery,
    'sucam': SUCAM,
    'cvt': CrossViewTransformer,
    'lss': LiftSplatShoot
}


class Model(nn.Module):
    def __init__(self,
                 devices,
                 backbone='sucam',
                 n_classes=1,
                 opt=None,
                 loss_type='bce',
                 weights=None,
                 use_seg=False,
                 use_dep=False,
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

        self.use_seg = use_seg
        self.use_dep = use_dep

        self.depth_lambda = .0075
        self.seg_lambda = .05

        if self.use_dep:
            self.dep_loss = DepthLoss(gamma=2, ignore_index=-1, reduction='mean')

        self.create_backbone(backbone)

    def create_backbone(self, backbone):
        self.backbone = nn.DataParallel(
            backbones[backbone](n_classes=self.n_classes).to(self.device),
            output_device=self.device,
            device_ids=self.devices,
        )
        
    def loss(self, x, gt):
        if self.loss_type == 'bce':
            return F.binary_cross_entropy_with_logits(x, gt)
        if self.loss_type == 'focal':
            return sigmoid_focal_loss(x, gt, -1, 2, reduction='mean')
        
    def state_dict(self):
        return {
            'model_state_dict': super().state_dict(),
            'optimizer_state_dict': self.opt.state_dict() if self.opt is not None else None,
        }

    def load(self, state_dict):
        self.load_state_dict(state_dict['model_state_dict'])

        if self.opt is not None:
            self.opt.load_state_dict(state_dict['optimizer_state_dict'])

    def save(self, path):
        torch.save(self.state_dict(), path)

    def train_step(self, images, segs, depths, intrinsics, extrinsics, labels, use_gt_depth=False):
        self.opt.zero_grad(set_to_none=True)
        b, c = images.shape[:2]

        if use_gt_depth:
            depths[depths == -1] = 0
            gt_depth = F.one_hot(depths, num_classes=112).permute(0, 1, 4, 2, 3)
            gt_depth = gt_depth.view(b * c, 112, 28, 60)
        else:
            gt_depth = None

        outs, seg_outs, dep_outs = self(images, intrinsics, extrinsics, gt_depth=gt_depth)
        loss = self.loss(outs, labels.to(self.device))

        with autocast(enabled=False):
            if self.use_dep:
                dep_loss = F.nll_loss(
                    torch.log(dep_outs.clamp(min=1e-8)),
                    depths.to(self.device).view(b * c, 28, 60),
                    reduction='mean',
                    ignore_index=-1
                )

                loss += self.depth_lambda * dep_loss
            else:
                dep_loss = 0

        if self.use_seg:
            seg_loss = F.binary_cross_entropy(seg_outs, segs.to(self.device).view(b * c, 28, 60))
            loss += self.seg_lambda * seg_loss
        else:
            seg_loss = 0

        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 5.0)
        self.opt.step()      

        return outs, seg_outs, dep_outs, loss, seg_loss, dep_loss

    def forward(self, images, intrinsics, extrinsics, gt_depth=None):
        return self.backbone(images, intrinsics, extrinsics, gt_depth=gt_depth)

    @staticmethod
    def activate(x):
        return x.sigmoid()