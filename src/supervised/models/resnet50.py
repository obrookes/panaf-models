import torch
from torch import nn
from torch.nn import Conv3d
from pytorchvideo.models.head import create_res_basic_head


class ThreeStreamNetwork(nn.Module):
    def __init__(self, device, freeze_backbone=False):
        super().__init__()

        self.device = device

        self.rgb_stream = ResNet50Embedder(freeze_backbone=freeze_backbone)
        self.flow_stream = TemporalResNet50Embedder(freeze_backbone=freeze_backbone)
        self.pose_stream = ResNet50Embedder(freeze_backbone=freeze_backbone)

        self.fc1 = nn.Linear(in_features=3072, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=512)
        self.fc3 = nn.Linear(in_features=512, out_features=9)

    def forward(self, x):
        rgb_emb = self.rgb_stream(x["spatial_sample"].permute(0, 2, 1, 3, 4))
        flow_emb = self.flow_stream(x["flow_sample"].permute(0, 2, 1, 3, 4))
        pose_emb = self.pose_stream(x["dense_sample"].permute(0, 2, 1, 3, 4))

        emb = torch.cat((rgb_emb, flow_emb, pose_emb), dim=1)
        pred = self.fc3(self.fc2(self.fc1(emb)))
        return pred


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
    def __init__(self, freeze_backbone=False):

        super().__init__()
        self.freeze_backbone = freeze_backbone

        pretrained_model = torch.hub.load(
            "facebookresearch/pytorchvideo:main", model="slow_r50", pretrained=True
        )

        self.backbone = nn.Sequential(*list(pretrained_model.children())[0][:-1])

        self.res_head = create_res_basic_head(in_features=2048, out_features=500)
        self.fc = nn.Linear(in_features=500, out_features=9)

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
        self.res_head = create_res_basic_head(in_features=2048, out_features=128)
        self.fc = nn.Linear(in_features=128, out_features=9)

        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        output = self.res_head(self.backbone(x))
        return self.fc(output)


class TemporalSoftmaxEmbedderResNet50(nn.Module):
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
        self.res_head = create_res_basic_head(in_features=2048, out_features=128)
        self.fc = nn.Linear(in_features=128, out_features=9)

        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        embedding = self.res_head(self.backbone(x))
        pred = self.fc(embedding)
        return embedding, pred


class SoftmaxEmbedderResNet50(nn.Module):
    def __init__(self, freeze_backbone=False):

        super().__init__()
        self.freeze_backbone = freeze_backbone

        # Load pretrained model
        pretrained_model = torch.hub.load(
            "facebookresearch/pytorchvideo:main", model="slow_r50", pretrained=True
        )

        self.backbone = nn.Sequential(*list(pretrained_model.children())[0][:-1])
        self.res_head = create_res_basic_head(in_features=2048, out_features=128)
        self.fc = nn.Linear(in_features=128, out_features=9)

        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        embedding = self.res_head(self.backbone(x))
        pred = self.fc(embedding)
        return embedding, pred
