import torch
from torch import nn
from torch.nn import Conv3d
from pytorchvideo.models.vision_transformers import create_multiscale_vision_transformers, create_vit_basic_head


class MViT(nn.Module):
    def __init__(self, model_name="mvit_base_16x4"):
        
        """
        Other models: 
        'name: mvit_base_32x3'
        'input: torch.rand(1, 3, 32, 224, 224)
        """

        super().__init__()

        self.model = torch.hub.load(
            "facebookresearch/pytorchvideo:main", model=model_name, pretrained=True
        )
        self.model.head = create_vit_basic_head(in_features=768, out_features=128)
        self.fc = nn.Linear(in_features=128, out_features=9)

    def forward(self, x):
        output = self.model(x)
        return self.fc(output)
