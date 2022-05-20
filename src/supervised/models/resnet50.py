import torch
from torch import nn
from pytorchvideo.models.head import create_res_basic_head


class ResNet50(nn.Module):
    def __init__(self):

        super().__init__()

        # Load pretrained model
        pretrained_model = torch.hub.load(
            "facebookresearch/pytorchvideo:main", model="slow_r50", pretrained=True
        )

        # Strip the head from backbone
        self.backbone = nn.Sequential(*list(pretrained_model.children())[0][:-1])

        # Attach a new head with specified class number (hard coded for now...)
        self.res_head = create_res_basic_head(in_features=2048, out_features=500)

        self.fc = nn.Linear(in_features=500, out_features=9)

    def forward(self, x):
        output = self.res_head(self.backbone(x))
        return self.fc(output)
