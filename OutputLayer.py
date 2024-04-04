import torch
import torch.nn as nn


class OutputLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, z: torch.Tensor):
        return nn.functional.sigmoid(self.layer(z.abs()))