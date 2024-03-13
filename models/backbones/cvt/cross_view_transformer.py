import torch.nn as nn

from models.backbones.cvt.decoder import *
from models.backbones.cvt.encoder import *
from models.gpn.density import Density, Evidence


class Shrink(nn.Module):
    def __init__(self):
        super(Shrink, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(7, 5, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(5, 3, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


class Post(nn.Module):
    def __init__(
            self, out_channels,
            dim_last: int = 64,
            n_classes: int = 2,
    ):
        super().__init__()
        self.latent_size = 16
        self.n_classes = n_classes

        self.flow = Density(dim_latent=self.latent_size, num_mixture_elements=n_classes)
        self.evidence = Evidence(scale='latent-new')

        self.to_logits = nn.Sequential(
            nn.Conv2d(out_channels, self.latent_size, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.latent_size))

        self.last = nn.Conv2d(n_classes, n_classes, 1)
        self.p_c = torch.tensor([.02, .98])

    def forward(self, x):
        x = self.to_logits(x)

        x = x.permute(0, 2, 3, 1).to(x.device)
        x = x.reshape(-1, self.latent_size)

        self.p_c = self.p_c.to(x.device)

        log_q_ft_per_class = self.flow(x) + self.p_c.view(1, -1).log()

        beta = self.evidence(
            log_q_ft_per_class, dim=self.latent_size,
            further_scale=2.0).exp()

        beta = beta.reshape(-1, 200, 200, self.n_classes).permute(0, 3, 1, 2).contiguous()
        beta = self.last(beta.log()).exp()
        alpha = beta + 1

        return alpha


class CrossViewTransformer(nn.Module):
    def __init__(
        self,
        dim_last=64,
        n_classes=2,
    ):
        super().__init__()

        self.n_classes = n_classes
        self.encoder = Encoder()
        self.decoder = Decoder(128, [128, 128, 64])

        self.to_logits = nn.Sequential(
            nn.Conv2d(self.decoder.out_channels, dim_last, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim_last),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_last, n_classes, 1))

    def forward(self, images, intrinsics, extrinsics):
        x, atts = self.encoder(images, intrinsics, extrinsics)
        y = self.decoder(x)

        return self.to_logits(y)