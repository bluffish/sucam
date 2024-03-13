from models.model import Model
from tools.uncertainty import *


class Dropout(Model):
    def __init__(self, *args, **kwargs):
        super(Dropout, self).__init__(*args, **kwargs)

    @staticmethod
    def aleatoric(logits):
        return entropy(torch.mean(logits, dim=0), dim=1)

    @staticmethod
    def epistemic(logits):
        pred, _ = logits.max(dim=2)
        var = torch.var(pred, dim=0)
        return (1 - 1 / var).unsqueeze(1)

    @staticmethod
    def activate(logits):
        return torch.mean(torch.softmax(logits, dim=2), dim=0)

    def forward(self, images, intrinsics, extrinsics):
        self.train()
        out = [self.backbone(images, intrinsics, extrinsics) for _ in range(10)]

        return torch.stack(out)

