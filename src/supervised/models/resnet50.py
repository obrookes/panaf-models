import torch
from torch import nn
from torch.nn import Conv3d
from pytorchvideo.models.head import create_res_basic_head


class ResNet50(nn.Module):
    def __init__(self):

        super().__init__()

        pretrained_model = torch.hub.load(
            "facebookresearch/pytorchvideo:main", model="slow_r50", pretrained=True
        )

        self.backbone = nn.Sequential(*list(pretrained_model.children())[0][:-1])
        self.res_head = create_res_basic_head(in_features=2048, out_features=500)
        self.fc = nn.Linear(in_features=500, out_features=9)

    def forward(self, x):
        output = self.res_head(self.backbone(x))
        return self.fc(output)


class TemporalResNet50(nn.Module):
    def __init__(self):

        super().__init__()

        # Load pretrained model
        pretrained_model = torch.hub.load(
            "facebookresearch/pytorchvideo:main", model="slow_r50", pretrained=True
        )

        pretrained_model.blocks[0].conv = Conv3d(
            2,
            64,
            kernel_size=(1, 7, 7),
            stride=(1, 2, 2),
            padding=(0, 3, 3),
            bias=False,
        )

        self.backbone = nn.Sequential(*list(pretrained_model.children())[0][:-1])
        self.res_head = create_res_basic_head(in_features=2048, out_features=128)
        self.fc = nn.Linear(in_features=128, out_features=9)

    def forward(self, x):
        output = self.res_head(self.backbone(x))
        return self.fc(output)


class SoftmaxEmbedderResNet50(nn.Module):
    def __init__(self):

        super().__init__()

        # Load pretrained model
        pretrained_model = torch.hub.load(
            "facebookresearch/pytorchvideo:main", model="slow_r50", pretrained=True
        )

        self.backbone = nn.Sequential(*list(pretrained_model.children())[0][:-1])
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.res_head = create_res_basic_head(in_features=2048, out_features=128)
        self.fc = nn.Linear(in_features=128, out_features=9)

    def forward(self, x):
        embedding = self.res_head(self.backbone(x))
        pred = self.fc(embedding)
        return embedding, pred
