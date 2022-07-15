import torch
from torch import nn
from typing import Optional
from pytorchvideo.layers.utils import set_attributes
from pytorchvideo.models.simclr import SimCLR


class SimCLR(SimCLR):
    def __init__(
        self,
        mlp: nn.Module,
        backbone: Optional[nn.Module] = None,
        temperature: float = 0.07,
    ):
        super().__init__(mlp, backbone, temperature)
        torch._C._log_api_usage_once("PYTORCHVIDEO.model.SimCLR.__init__")
        set_attributes(self, locals())

    def get_representations(self, x: torch.Tensor):
        return self.backbone(x)
