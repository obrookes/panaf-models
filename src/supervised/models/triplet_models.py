import torch
from torch import nn
from torch.nn import Conv3d
import torch.nn.functional as F
from pytorchvideo.models.head import create_res_basic_head

# ======> Models for triplet loss <===========
# ============================================


class DualStreamNetworkEmbeddingSoftmax(nn.Module):
    """
    Dual stream network where each stream outputs
    a 128-dimensional vector and logits. In the forward
    method both vector and logits are averaged over all
    streams.
    """

    def __init__(
        self,
        freeze_backbone=False,
        embedding_size=128,
        num_classes=9,
        type_of_fusion="convolutional",
    ):
        super().__init__()

        # ==> Spatial network <==
        pretrained_spatial_model = torch.hub.load(
            "facebookresearch/pytorchvideo:main", model="slow_r50", pretrained=True
        )
        self.spatial_backbone = nn.Sequential(
            *list(pretrained_spatial_model.children())[0][:-1]
        )

        if freeze_backbone:
            for param in self.spatial_backbone.parameters():
                param.requires_grad = False

        self.spatial_head = create_res_basic_head(in_features=2048, out_features=1024)
        self.spatial_embedder = nn.Sequential(
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=embedding_size),
        )

        # ==> Temporal network <==
        pretrained_temporal_model = torch.hub.load(
            "facebookresearch/pytorchvideo:main", model="slow_r50", pretrained=True
        )

        pretrained_temporal_model.blocks[0].conv = Conv3d(
            2,
            64,
            kernel_size=(1, 7, 7),
            stride=(1, 2, 2),
            padding=(0, 3, 3),
            bias=False,
        )

        self.temporal_backbone = nn.Sequential(
            *list(pretrained_temporal_model.children())[0][:-1]
        )

        if freeze_backbone:
            for param in self.temporal_backbone.parameters():
                param.requires_grad = False

        self.temporal_head = create_res_basic_head(in_features=2048, out_features=1024)
        self.temporal_embedder = nn.Sequential(
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=embedding_size),
        )

        # ==> Fusion components <==
        self.conv1 = nn.Sequential(
            nn.Conv3d(4096, 2048, 1, stride=1, padding=1, dilation=1, bias=True),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )
        self.fusion_head = create_res_basic_head(
            in_features=2048, out_features=1024, pool_kernel_size=(1, 5, 5)
        )

        # ==> Embedder <==

        self.embedder = nn.Sequential(
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=embedding_size),
        )

        self.predictor = nn.Sequential(
            nn.Linear(in_features=embedding_size, out_features=64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=64, out_features=num_classes),
        )

        # ==> Fuser <==
        if type_of_fusion == "convolutional":
            self.fusion_forward = self._convolutional_fusion
        elif type_of_fusion == "average":
            self.fusion_forward = self._average_fusion
        elif type_of_fusion == "element_wise":
            self.fusion_forward = self._elemwise_mul

    def _convolutional_fusion(self, x):
        spatial_data = x["spatial_sample"].permute(0, 2, 1, 3, 4)
        temporal_data = x["flow_sample"].permute(0, 2, 1, 3, 4)

        spatial_fmaps = self.spatial_backbone(spatial_data)
        temporal_fmaps = self.temporal_backbone(temporal_data)

        y = torch.cat((spatial_fmaps, temporal_fmaps), dim=1)
        for i in range(spatial_fmaps.size(1)):
            y[:, (2 * i), :, :] = spatial_fmaps[:, i, :, :]
            y[:, (2 * i + 1), :, :] = temporal_fmaps[:, i, :, :]

        fused_output = self.fusion_head(self.conv1(y))
        return self.embedder(fused_output)

    def _average_fusion(self, x):
        spatial_data = x["spatial_sample"].permute(0, 2, 1, 3, 4)
        temporal_data = x["flow_sample"].permute(0, 2, 1, 3, 4)

        spatial_fmaps = self.spatial_backbone(spatial_data)
        spatial_embedding = self.spatial_embedder(self.spatial_head(spatial_fmaps))

        temporal_fmaps = self.temporal_backbone(temporal_data)
        temporal_embedding = self.temporal_embedder(self.temporal_head(temporal_fmaps))

        embedding = (spatial_embedding + temporal_embedding) / 2
        return embedding

    def _elemwise_mul(self, x):
        spatial_data = x["spatial_sample"].permute(0, 2, 1, 3, 4)
        temporal_data = x["flow_sample"].permute(0, 2, 1, 3, 4)

        spatial_fmaps = self.spatial_backbone(spatial_data)
        spatial_embedding = self.spatial_embedder(self.spatial_head(spatial_fmaps))

        temporal_fmaps = self.temporal_backbone(temporal_data)
        temporal_embedding = self.temporal_embedder(self.temporal_head(temporal_fmaps))

        embedding = torch.mul(spatial_embedding, temporal_embedding)
        return F.normalize(embedding, p=2, dim=1)

    def forward(self, x):
        embedding = self.fusion_forward(x)
        scores = self.predictor(embedding)
        return embedding, scores


class TripleStreamNetworkEmbeddingSoftmax(nn.Module):
    """
    Dual stream network where each stream outputs
    a 128-dimensional vector and logits. In the forward
    method both vector and logits are averaged over all
    streams.
    """

    def __init__(
        self,
        freeze_backbone=False,
        embedding_size=128,
        num_classes=9,
        type_of_fusion="convolutional",
    ):
        super().__init__()

        # ==> Spatial network <==
        pretrained_spatial_model = torch.hub.load(
            "facebookresearch/pytorchvideo:main", model="slow_r50", pretrained=True
        )
        self.spatial_backbone = nn.Sequential(
            *list(pretrained_spatial_model.children())[0][:-1]
        )

        if freeze_backbone:
            for param in self.spatial_backbone.parameters():
                param.requires_grad = False

        self.spatial_head = create_res_basic_head(in_features=2048, out_features=1024)
        self.spatial_embedder = nn.Sequential(
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=embedding_size),
        )

        # ==> Temporal network <==
        pretrained_temporal_model = torch.hub.load(
            "facebookresearch/pytorchvideo:main", model="slow_r50", pretrained=True
        )

        pretrained_temporal_model.blocks[0].conv = Conv3d(
            2,
            64,
            kernel_size=(1, 7, 7),
            stride=(1, 2, 2),
            padding=(0, 3, 3),
            bias=False,
        )

        self.temporal_backbone = nn.Sequential(
            *list(pretrained_temporal_model.children())[0][:-1]
        )

        if freeze_backbone:
            for param in self.temporal_backbone.parameters():
                param.requires_grad = False

        self.temporal_head = create_res_basic_head(in_features=2048, out_features=1024)
        self.temporal_embedder = nn.Sequential(
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=embedding_size),
        )

        # ==> Dense pose network <==
        pretrained_pose_model = torch.hub.load(
            "facebookresearch/pytorchvideo:main", model="slow_r50", pretrained=True
        )
        self.pose_backbone = nn.Sequential(
            *list(pretrained_pose_model.children())[0][:-1]
        )

        if freeze_backbone:
            for param in self.pose_backbone.parameters():
                param.requires_grad = False

        self.pose_head = create_res_basic_head(in_features=2048, out_features=1024)
        self.pose_embedder = nn.Sequential(
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=embedding_size),
        )

        # ==> Fusion components <==
        self.conv1 = nn.Sequential(
            nn.Conv3d(6144, 2048, 1, stride=1, padding=1, dilation=1, bias=True),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )
        self.fusion_head = create_res_basic_head(
            in_features=2048, out_features=1024, pool_kernel_size=(1, 5, 5)
        )

        # ==> Embedder <==

        self.embedder = nn.Sequential(
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=512, out_features=embedding_size),
        )

        self.predictor = nn.Sequential(
            nn.Linear(in_features=embedding_size, out_features=64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=64, out_features=num_classes),
        )

        # ==> Fuser <==
        if type_of_fusion == "convolutional":
            self.fusion_forward = self._convolutional_fusion
        elif type_of_fusion == "average":
            self.fusion_forward = self._average_fusion
        elif type_of_fusion == "element_wise":
            self.fusion_forward = self._elemwise_mul

    def _shared_step(self, x):
        spatial_data = x["spatial_sample"].permute(0, 2, 1, 3, 4)
        temporal_data = x["flow_sample"].permute(0, 2, 1, 3, 4)
        pose_data = x["dense_sample"].permute(0, 2, 1, 3, 4)
        return spatial_data, temporal_data, pose_data

    def _convolutional_fusion(self, x):
        spatial_data, temporal_data, pose_data = self._shared_step(x)

        spatial_fmaps = self.spatial_backbone(spatial_data)
        temporal_fmaps = self.temporal_backbone(temporal_data)
        pose_fmaps = self.pose_backbone(pose_data)

        y = torch.cat((spatial_fmaps, temporal_fmaps, pose_fmaps), dim=1)
        # Unsure what this is doing...
        for i in range(spatial_fmaps.size(1)):
            y[:, (2 * i), :, :] = spatial_fmaps[:, i, :, :]
            y[:, (2 * i + 1), :, :] = temporal_fmaps[:, i, :, :]
            y[:, (2 * i + 2), :, :] = pose_fmaps[:, i, :, :]

        fused_output = self.fusion_head(self.conv1(y))
        return self.embedder(fused_output)

    def _average_fusion(self, x):
        spatial_data, temporal_data, pose_data = self._shared_step(x)

        spatial_fmaps = self.spatial_backbone(spatial_data)
        spatial_embedding = self.spatial_embedder(self.spatial_head(spatial_fmaps))

        temporal_fmaps = self.temporal_backbone(temporal_data)
        temporal_embedding = self.temporal_embedder(self.temporal_head(temporal_fmaps))

        pose_fmaps = self.pose_backbone(pose_data)
        pose_embedding = self.pose_embedder(self.pose_head(pose_fmaps))

        embedding = (spatial_embedding + temporal_embedding + pose_embedding) / 3
        return embedding

    def _elemwise_mul(self, x):
        spatial_data, temporal_data, pose_data = self._shared_step(x)

        spatial_fmaps = self.spatial_backbone(spatial_data)
        spatial_embedding = self.spatial_embedder(self.spatial_head(spatial_fmaps))

        temporal_fmaps = self.temporal_backbone(temporal_data)
        temporal_embedding = self.temporal_embedder(self.temporal_head(temporal_fmaps))

        pose_fmaps = self.pose_backbone(pose_data)
        pose_embedding = self.pose_embedder(self.pose_head(pose_fmaps))

        embedding = torch.mul(spatial_embedding, temporal_embedding)
        embedding = torch.mul(embedding, pose_embedding)
        return F.normalize(embedding, p=2, dim=1)

    def forward(self, x):
        embedding = self.fusion_forward(x)
        scores = self.predictor(embedding)
        return embedding, scores
