from models.backbones.fiery.decoder import DecoderPostnet
from models.backbones.lss.lift_splat_shoot import BevEncodePostnet
from models.backbones.cvt.cross_view_transformer import Post

from models.model import *
from tools.loss import *
from tools.uncertainty import *
import torch.nn as nn


class Postnet(Model):
    def __init__(self, *args, **kwargs):
        super(Postnet, self).__init__(*args, **kwargs)

        self.beta_lambda = 0.001
        self.ood_lambda = 0.01
        self.k = 64

        print(f"BETA LAMBDA: {self.beta_lambda}")

    def create_backbone(self, backbone):
        self.backbone = nn.DataParallel(
            backbones[backbone](n_classes=self.n_classes).to(self.device),
            output_device=self.device,
            device_ids=self.devices
        )

        if backbone == 'lss':
            self.backbone.module.bevencode = BevEncodePostnet(inC=self.backbone.module.camC, outC=self.backbone.module.outC).to(self.device)
        if backbone == 'fiery':
            self.backbone.module.decoder = DecoderPostnet(in_channels=self.backbone.module.encoder_out_channels, n_classes=self.backbone.module.n_classes).to(self.device)
        if backbone == 'cvt':
            print(self.backbone.module.n_classes)
            self.backbone.module.to_logits = Post(self.backbone.module.decoder.out_channels, n_classes=self.backbone.module.n_classes).to(self.device)

    @staticmethod
    def aleatoric(alpha, mode='aleatoric'):
        if mode == 'aleatoric':
            soft = Postnet.activate(alpha)
            max_soft, hard = soft.max(dim=1)
            return (1 - max_soft).unsqueeze(1)
        elif mode == 'dissonance':
            return dissonance(alpha)

    @staticmethod
    def epistemic(alpha):
        return vacuity(alpha)

    @staticmethod
    def activate(alpha):
        return alpha / torch.sum(alpha, dim=1, keepdim=True)

    def loss(self, alpha, y, reduction='mean'):
        if self.loss_type == 'ce':
            A = uce_loss(alpha, y, weights=self.weights) + entropy_reg(alpha, beta_reg=self.beta_lambda)
        elif self.loss_type == 'focal':
            A = u_focal_loss(alpha, y, weights=self.weights, n=self.gamma)
        else:
            raise NotImplementedError()

        if reduction == 'mean':
            return A.mean()
        else:
            return A

    def loss_ood(self, alpha, y, ood):
        A = self.loss(alpha, y, reduction='none')
        A *= 1 + (self.epistemic(alpha).detach() * self.k)

        oreg = ood_reg(alpha, ood) * self.ood_lambda

        A = A[~ood.bool()].mean()

        A += oreg

        return A, oreg

    def train_step_ood(self, images, intrinsics, extrinsics, labels, ood):
        self.opt.zero_grad(set_to_none=True)

        if self.scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                outs = self(images, intrinsics, extrinsics)
                loss, oodl = self.loss_ood(outs, labels.to(self.device), ood)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)

            nn.utils.clip_grad_norm_(self.parameters(), 5.0)
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            outs = self(images, intrinsics, extrinsics)
            loss, oodl = self.loss_ood(outs, labels.to(self.device), ood)
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), 5.0)
            self.opt.step()

        preds = self.activate(outs)
        return outs, preds, loss, oodl

    def forward(self, images, intrinsics, extrinsics, limit=None):
        if self.tsne:
            print("Returning intermediate")
            return self.backbone(images, intrinsics, extrinsics)

        evidence = self.backbone(images, intrinsics, extrinsics).relu()

        if limit is not None:
            evidence = evidence.clamp(max=limit)
        alpha = evidence + 1

        return alpha

