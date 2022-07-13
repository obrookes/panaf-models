import torch
from torch import nn
from torch.nn import Conv3d
from pytorchvideo.models.head import create_res_basic_head


class ResNet50Embedder(nn.Module):
    def __init__(self, freeze_backbone=False):

        super().__init__()
        self.freeze_backbone = freeze_backbone

        pretrained_model = torch.hub.load(
            "facebookresearch/pytorchvideo:main", model="slow_r50", pretrained=True
        )

        self.backbone = nn.Sequential(*list(pretrained_model.children())[0][:-1])
        self.head = create_res_basic_head(in_features=2048, out_features=1024)

        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        output = self.head(self.backbone(x))
        return output


class TemporalResNet50Embedder(nn.Module):
    def __init__(self, freeze_backbone=False):

        super().__init__()
        self.freeze_backbone = freeze_backbone

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
        self.head = create_res_basic_head(in_features=2048, out_features=1024)

        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        output = self.head(self.backbone(x))
        return output


class ResNet50(nn.Module):
    def __init__(self, freeze_backbone=False, out_features=9):

        super().__init__()
        self.freeze_backbone = freeze_backbone

        pretrained_model = torch.hub.load(
            "facebookresearch/pytorchvideo:main", model="slow_r50", pretrained=True
        )

        self.backbone = nn.Sequential(*list(pretrained_model.children())[0][:-1])

        self.res_head = create_res_basic_head(in_features=2048, out_features=500)
        self.fc = nn.Linear(in_features=500, out_features=out_features)

        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        output = self.res_head(self.backbone(x))
        return self.fc(output)


class MinorityResNet50(nn.Module):
    def __init__(self, freeze_backbone=False):

        super().__init__()
        self.freeze_backbone = freeze_backbone

        pretrained_model = torch.hub.load(
            "facebookresearch/pytorchvideo:main", model="slow_r50", pretrained=True
        )

        self.backbone = nn.Sequential(*list(pretrained_model.children())[0][:-1])

        self.res_head = create_res_basic_head(in_features=2048, out_features=500)
        self.fc = nn.Linear(in_features=500, out_features=6)

        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        output = self.res_head(self.backbone(x))
        return self.fc(output)


class TemporalResNet50(nn.Module):
    def __init__(self, freeze_backbone=False, out_features=9):

        super().__init__()
        self.freeze_backbone = freeze_backbone

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
        self.fc = nn.Linear(in_features=128, out_features=out_features)

        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        output = self.res_head(self.backbone(x))
        return self.fc(output)


class TemporalSoftmaxEmbedderResNet50(nn.Module):
    def __init__(self, freeze_backbone=False, out_features=9):

        super().__init__()
        self.freeze_backbone = freeze_backbone

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
        self.fc = nn.Linear(in_features=128, out_features=out_features)

        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        embedding = self.res_head(self.backbone(x))
        pred = self.fc(embedding)
        return embedding, pred


class SoftmaxEmbedderResNet50(nn.Module):
    def __init__(self, freeze_backbone=False, out_features=9):

        super().__init__()
        self.freeze_backbone = freeze_backbone

        # Load pretrained model
        pretrained_model = torch.hub.load(
            "facebookresearch/pytorchvideo:main", model="slow_r50", pretrained=True
        )

        self.backbone = nn.Sequential(*list(pretrained_model.children())[0][:-1])
        self.res_head = create_res_basic_head(in_features=2048, out_features=128)
        self.fc = nn.Linear(in_features=128, out_features=out_features)

        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        embedding = self.res_head(self.backbone(x))
        pred = self.fc(embedding)
        return embedding, pred
