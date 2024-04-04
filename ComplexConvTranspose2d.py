import torch
import torch.nn as nn
from utils import get_init_bound, apply_layer


class ComplexConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, h_in, dilation=1):
        super().__init__()
        self.layer = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            bias=False
        )
        self.normalization = nn.BatchNorm2d(num_features=out_channels)
        
        self.kernel_size = torch.nn.modules.utils._pair(kernel_size)
        fan_in = in_channels * self.kernel_size[0] * self.kernel_size[1]
        
        self.magnitude_bias = nn.Parameter(torch.empty((1, out_channels, 1, 1)))
        nn.init.uniform_(self.magnitude_bias, -get_init_bound(fan_in), get_init_bound(fan_in))

        self.phase_bias = nn.Parameter(torch.empty((1, out_channels, 1, 1)))
        nn.init.constant_(self.phase_bias, val=0)

    def forward(self, x: torch.Tensor):
        return apply_layer(z=x, module=self, normalization=self.normalization)