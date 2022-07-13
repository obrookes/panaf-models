from src.supervised.models import (
    ResNet50S, # single rgb-stream
    RGBDenseNetworkSF, # dual rgb and pose
    RGBFlowNetworkSF, # dual rgb + flow
    ThreeStreamNetworkSF # triple-stream
)

def initialise_model(name, freeze_backbone):
    if name == 'r':
        model = ResNet50S(freeze_backbone=freeze_backbone)
    elif name == 'rd':
        model = RGBDenseNetworkSF(freeze_backbone=freeze_backbone)
    elif name == 'rf':
        model = RGBFlowNetworkSF(freeze_backbone=freeze_backbone)
    elif name == 'rdf':
        model = ThreeStreamNetworkSF(freeze_backbone=freeze_backbone)
    else:
        raise NameError(f"The model initialisation: {name} does not exist!")
    return model

