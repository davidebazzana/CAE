import torch
import torch.nn as nn
from ComplexConv2d import ComplexConv2d
from ComplexConvTranspose2d import ComplexConvTranspose2d
from ComplexLinear import ComplexLinear
from OutputLayer import OutputLayer


class CAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # 32x32 -> 16x16
        self.conv_1 = ComplexConv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1, h_in=32)
        # 16x16 -> 16x16
        self.conv_2 = ComplexConv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, h_in=16)
        # 16x16 -> 8x8
        self.conv_3 = ComplexConv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, h_in=16)
        # 8x8 -> 8x8
        self.conv_4 = ComplexConv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, h_in=8)
        # 8x8 -> 4x4
        self.conv_5 = ComplexConv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, h_in=8)
        self.encoder = nn.Sequential(
            self.conv_1,
            self.conv_2,
            self.conv_3,
            self.conv_4,
            self.conv_5
        )
        self.enc_linear = ComplexLinear(in_features=64*4*4, out_features=512)
        self.dec_linear = ComplexLinear(in_features=512, out_features=64*4*4)
        # 4x4 -> 8x8
        self.conv_t_1 = ComplexConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1, h_in=4)
        # 8x8 -> 8x8
        self.conv_d_1 = ComplexConv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, h_in=8)
        # 8x8 -> 16x16
        self.conv_t_2 = ComplexConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1, h_in=8)
        # 16x16 -> 16x16
        self.conv_d_2 = ComplexConv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1, h_in=16)
        # 16x16 -> 32x32
        self.conv_t_3 = ComplexConvTranspose2d(in_channels=16, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1, h_in=16)
        self.decoder = nn.Sequential(
            self.conv_t_1,
            self.conv_d_1,
            self.conv_t_2,
            self.conv_d_2,
            self.conv_t_3
        )
        self.output_layer = OutputLayer()
        nn.init.constant_(self.output_layer.layer.weight, 1)
        nn.init.constant_(self.output_layer.layer.bias, 0)

    def preprocess(x_real: torch.Tensor):
        x_imaginary = torch.zeros_like(x_real)
        return torch.complex(x_real*torch.cos(x_imaginary), x_real*torch.sin(x_imaginary))

    def forward(self, x: torch.Tensor):
        z = CAE.preprocess(x)
        latent = self.encoder(z)
        latent = torch.reshape(latent, (-1, 64*4*4))
        latent = self.enc_linear(latent)
        latent = self.dec_linear(latent)
        latent = torch.reshape(latent, (-1, 64, 4, 4))
        output = self.decoder(latent)
        if self.training:
            return self.output_layer(output)
        else:
            return self.output_layer(output), output
