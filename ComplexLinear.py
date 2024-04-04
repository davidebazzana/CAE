import torch
import torch.nn as nn
from utils import get_init_bound, apply_layer


class ComplexLinear(nn.Module):
    def __init__(self, in_features, out_features) -> None:
        super().__init__()
        self.layer = nn.Linear(in_features=in_features, out_features=out_features, bias=False)
        self.normalization = nn.BatchNorm1d(num_features=out_features)
        
        fan_in = in_features
        
        self.magnitude_bias = nn.Parameter(torch.empty((1, out_features)))
        nn.init.uniform_(self.magnitude_bias, -get_init_bound(fan_in), get_init_bound(fan_in))

        self.phase_bias = nn.Parameter(torch.empty((1, out_features)))
        nn.init.constant_(self.phase_bias, val=0)

    def forward(self, x: torch.Tensor):
        return apply_layer(z=x, module=self, normalization=self.normalization)