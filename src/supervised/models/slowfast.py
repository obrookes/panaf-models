import torch
from torch import nn
from pytorchvideo.models.head import create_res_basic_head


class SlowFast(nn.Module):
    def __init__(self, model_name="slowfast_r50"):

        """
        Other models:
        'name: slowfast_16x8_r101_50_50'
        'input: slow = torch.rand(1, 3, 64, 244, 244)
                fast = torch.rand(1, 3, 16, 244, 244)

        'name: slowfast_r101
        'input: slow = torch.rand(1, 3, 32, 244, 244)
                fast = torch.rand(1, 3, 8, 244, 244)'
        """

        super().__init__()

        slowfast = torch.hub.load(
            "facebookresearch/pytorchvideo:main", model=model_name, pretrained=True
        )
        self.backbone = nn.Sequential(*list(slowfast.children())[0][:-1])
        self.res_head = create_res_basic_head(
            in_features=2304, out_features=128, pool=None
        )
        self.fc = nn.Linear(in_features=128, out_features=9)

    def forward(self, x):
        output = self.res_head(self.backbone(x))
        return self.fc(output)
