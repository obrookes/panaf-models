from src.supervised.models import (
    ResNet50S,  # single rgb-stream
    RGBDenseNetworkSF,  # dual rgb and pose
    RGBFlowNetworkSF,  # dual rgb + flow
    ThreeStreamNetworkSF,  # triple-stream
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
    else:
        raise NameError(f"The model initialisation: {name} does not exist!")
    return model
