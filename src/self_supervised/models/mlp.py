from torch import nn
import torch.nn.functional as F


class MLP(nn.Module):

    # ENCODE
    # encode -> representations
    # (b, 2048) -> (b, 128)

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(2048, 2048, bias=True),
            nn.BatchNorm1d(2048),
            nn.ReLU(),
            nn.Linear(2048, 128, bias=False),
        )

    def forward(self, x):
        x = self.model(x)
        return F.normalize(x, dim=1)
