from src.supervised.models import (
    ResNet50S,  # single rgb-stream
    RGBDenseNetworkSF,  # dual rgb and pose
    RGBFlowNetworkSF,  # dual rgb + flow
    ThreeStreamNetworkSF,  # triple-stream
    # => triplet models <=
    SpatialStreamNetworkEmbedderSoftmax,
    DualStreamNetworkEmbedderSoftmax,
    ThreeStreamNetworkEmbedderSoftmax,
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


def initialise_triplet_model(name, freeze_backbone, out_features=9):
    if name == "r":
        model = SpatialStreamNetworkEmbedderSoftmax(
            freeze_backbone=freeze_backbone, out_features=out_features
        )
    elif name == "rf":
        model = DualStreamNetworkEmbedderSoftmax(
            freeze_backbone=freeze_backbone, out_features=out_features
        )
    elif name == "rdf":
        model = ThreeStreamNetworkEmbedderSoftmax(
            freeze_backbone=freeze_backbone, out_features=out_features
        )
    elif name == "slowfast_r50":
        model = SlowFastEmbedder(model_name=name)
    elif name == "slowfast_r101":
        model = SlowFastEmbedder(model_name=name)
    else:
        raise NameError(f"The model initialisation: {name} does not exist!")
    return model
