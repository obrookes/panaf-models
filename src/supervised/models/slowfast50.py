import torch
from torch import nn
from torch.nn import Conv3d
from pytorchvideo.models.head import create_res_basic_head


class SlowFast50(nn.Module):
    def __init__(self):

        super().__init__()

        slowfast = torch.hub.load(
            "facebookresearch/pytorchvideo:main", model="slowfast_r50", pretrained=True
        )
        self.backbone = nn.Sequential(*list(slowfast.children())[0][:-1])
        self.res_head = create_res_basic_head(
            in_features=2304, out_features=128, pool=None
        )
        self.fc = nn.Linear(in_features=128, out_features=9)

    def forward(self, x):
        output = self.res_head(self.backbone(x))
        return self.fc(output)
