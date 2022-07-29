import torch
from torch import nn
from .resnet50 import (
    ResNet50,
    TemporalResNet50,
    ResNet50Embedder,
    TemporalResNet50Embedder,
    SoftmaxEmbedderResNet50,
    TemporalSoftmaxEmbedderResNet50,
)

# ====> Models for supervised training <====
# ==========================================


class ResNet50S(nn.Module):
    """
    Single stream network which output logits or softmax
    for rgb stream.
    """

    def __init__(self, freeze_backbone=False, out_features=9):
        super().__init__()
        self.rgb_stream = ResNet50(
            freeze_backbone=freeze_backbone, out_features=out_features
        )

    def forward(self, x):
        rgb_score = self.rgb_stream(x["spatial_sample"].permute(0, 2, 1, 3, 4))
        return rgb_score


class RGBFlowNetworkSF(nn.Module):
    """
    Dual stream network with score fusion (SF):
    An average of the softmax output for rgb and
    optical flow stream is computed.
    """

    def __init__(self, freeze_backbone=False, out_features=9):
        super().__init__()

        self.rgb_stream = ResNet50(
            freeze_backbone=freeze_backbone, out_features=out_features
        )
        self.flow_stream = TemporalResNet50(
            freeze_backbone=freeze_backbone, out_features=out_features
        )

    def forward(self, x):
        rgb_score = self.rgb_stream(x["spatial_sample"].permute(0, 2, 1, 3, 4))
        flow_score = self.flow_stream(x["flow_sample"].permute(0, 2, 1, 3, 4))
        pred = (rgb_score + flow_score) / 2
        return pred


class RGBDenseNetworkSF(nn.Module):
    """
    Dual stream network with score fusion (SF):
    An average of the softmax output for rgb and
    dense pose stream is computed.
    """

    def __init__(self, freeze_backbone=False, out_features=9):
        super().__init__()

        self.rgb_stream = ResNet50(
            freeze_backbone=freeze_backbone, out_features=out_features
        )
        self.pose_stream = ResNet50(
            freeze_backbone=freeze_backbone, out_features=out_features
        )

    def forward(self, x):
        rgb_score = self.rgb_stream(x["spatial_sample"].permute(0, 2, 1, 3, 4))
        pose_score = self.pose_stream(x["flow_sample"].permute(0, 2, 1, 3, 4))
        pred = (rgb_score + pose_score) / 2
        return pred


class ThreeStreamNetworkSF(nn.Module):
    """
    Three stream network with score fusion (SF):
    An average of the softmax output for each stream
    is computed.
    """

    def __init__(self, freeze_backbone=False, out_features=9):
        super().__init__()

        self.rgb_stream = ResNet50(
            freeze_backbone=freeze_backbone, out_features=out_features
        )
        self.flow_stream = TemporalResNet50(
            freeze_backbone=freeze_backbone, out_features=out_features
        )
        self.pose_stream = ResNet50(
            freeze_backbone=freeze_backbone, out_features=out_features
        )

    def forward(self, x):
        rgb_score = self.rgb_stream(x["spatial_sample"].permute(0, 2, 1, 3, 4))
        flow_score = self.flow_stream(x["flow_sample"].permute(0, 2, 1, 3, 4))
        pose_score = self.pose_stream(x["dense_sample"].permute(0, 2, 1, 3, 4))
        pred = (rgb_score + flow_score + pose_score) / 3
        return pred


class ThreeStreamNetworkLF(nn.Module):
    """
    Three stream network with late fusion (LF):
    1024-dimensional embeddings from each stream
    are concatenated before being passed through fc layers.
    """

    def __init__(self, freeze_backbone=False):
        super().__init__()

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


# ======> Models for triplet loss <===========
# ============================================


class SpatialStreamNetworkEmbedderSoftmax(nn.Module):
    """
    Single, spatial stream network that outputs a 128-dimensional
    vector and logits.
    """

    def __init__(self, freeze_backbone=False, out_features=9):
        super().__init__()

        self.rgb_stream = SoftmaxEmbedderResNet50(
            freeze_backbone=freeze_backbone, out_features=out_features
        )

    def forward(self, x):
        emb, pred = self.rgb_stream(x["spatial_sample"].permute(0, 2, 1, 3, 4))
        return emb, pred


class DualStreamNetworkEmbedderSoftmax(nn.Module):
    """
    Dual stream network where each stream outputs
    a 128-dimensional vector and logits. In the forward
    method both vector and logits are averaged over all
    streams.
    """

    def __init__(self, freeze_backbone=False, out_features=9):
        super().__init__()

        self.rgb_stream = SoftmaxEmbedderResNet50(
            freeze_backbone=freeze_backbone, out_features=out_features
        )
        self.flow_stream = TemporalSoftmaxEmbedderResNet50(
            freeze_backbone=freeze_backbone, out_features=out_features
        )

    def forward(self, x):
        rgb_emb, rgb_pred = self.rgb_stream(x["spatial_sample"].permute(0, 2, 1, 3, 4))
        flow_emb, flow_pred = self.flow_stream(x["flow_sample"].permute(0, 2, 1, 3, 4))

        emb = (rgb_emb + flow_emb) / 2
        pred = (rgb_pred + flow_pred) / 2
        return emb, pred


class ThreeStreamNetworkEmbedderSoftmax(nn.Module):
    """
    Three stream network where each stream outputs
    a 128-dimensional vector and logits. In the forward
    method both vector and logits are averaged over all
    streams.
    """

    def __init__(self, freeze_backbone=False, out_features=9):
        super().__init__()

        self.rgb_stream = SoftmaxEmbedderResNet50(
            freeze_backbone=freeze_backbone, out_features=out_features
        )
        self.flow_stream = TemporalSoftmaxEmbedderResNet50(
            freeze_backbone=freeze_backbone, out_features=out_features
        )
        self.pose_stream = SoftmaxEmbedderResNet50(
            freeze_backbone=freeze_backbone, out_features=out_features
        )

    def forward(self, x):
        rgb_emb, rgb_pred = self.rgb_stream(x["spatial_sample"].permute(0, 2, 1, 3, 4))
        flow_emb, flow_pred = self.flow_stream(x["flow_sample"].permute(0, 2, 1, 3, 4))
        pose_emb, pose_pred = self.pose_stream(x["dense_sample"].permute(0, 2, 1, 3, 4))

        emb = (rgb_emb + flow_emb + pose_emb) / 3
        pred = (rgb_pred + flow_pred + pose_pred) / 3
        return emb, pred
