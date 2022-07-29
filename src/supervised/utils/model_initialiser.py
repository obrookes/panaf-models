from src.supervised.models import (
    ResNet50S,  # single rgb-stream
    RGBDenseNetworkSF,  # dual rgb and pose
    RGBFlowNetworkSF,  # dual rgb + flow
    ThreeStreamNetworkSF,  # triple-stream
    # => triplet models <=
    SpatialStreamNetworkEmbedderSoftmax,
    DualStreamNetworkEmbeddingSoftmax,
    TripleStreamNetworkEmbeddingSoftmax,
    SlowFastEmbedder,
    MViT,
)


def initialise_model(name, freeze_backbone, out_features=9):
    if name == "r":
        model = ResNet50S(freeze_backbone=freeze_backbone, out_features=out_features)
    elif name == "rd":
        model = RGBDenseNetworkSF(
            freeze_backbone=freeze_backbone, out_features=out_features
        )
    elif name == "rf":
        model = RGBFlowNetworkSF(
            freeze_backbone=freeze_backbone, out_features=out_features
        )
    elif name == "rdf":
        model = ThreeStreamNetworkSF(
            freeze_backbone=freeze_backbone, out_features=out_features
        )
    elif name == "mvit_base_16x4":
        model = MViT(model_name=name, out_features=out_features)
    elif name == "mvit_base_32x3":
        model = MViT(model_name=name, out_features=out_features)
    else:
        raise NameError(f"The model initialisation: {name} does not exist!")
    return model


def initialise_triplet_model(
    name, freeze_backbone, embedding_size=128, num_classes=9, type_of_fusion=None
):
    if name == "r":
        model = SpatialStreamNetworkEmbedderSoftmax(
            freeze_backbone=freeze_backbone, out_features=num_classes
        )
    elif name == "rf":
        model = DualStreamNetworkEmbeddingSoftmax(
            freeze_backbone=freeze_backbone,
            embedding_size=embedding_size,
            num_classes=num_classes,
            type_of_fusion=type_of_fusion,
        )
    elif name == "rdf":
        model = TripleStreamNetworkEmbeddingSoftmax(
            freeze_backbone=freeze_backbone,
            embedding_size=embedding_size,
            num_classes=num_classes,
            type_of_fusion=type_of_fusion,
        )
    elif name == "slowfast_r50":
        model = SlowFastEmbedder(model_name=name)
    elif name == "slowfast_r101":
        model = SlowFastEmbedder(model_name=name)
    else:
        raise NameError(f"The model initialisation: {name} does not exist!")
    return model
