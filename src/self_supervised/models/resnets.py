import torch
from torch import nn
import torch.nn.functional as F
from pytorchvideo.models.head import create_res_basic_head


class ResNet50(nn.Module):
    def __init__(self, freeze_backbone=False):

        super().__init__()
        self.freeze_backbone = freeze_backbone

        pretrained_model = torch.hub.load(
            "facebookresearch/pytorchvideo:main", model="slow_r50", pretrained=True
        )

        self.backbone = nn.Sequential(*list(pretrained_model.children())[0][:-1])
        self.res_head = create_res_basic_head(in_features=2048, out_features=2048)

        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.res_head(self.backbone(x))
        return x



