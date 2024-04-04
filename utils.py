import torch
import torch.nn as nn
import math

def get_init_bound(fan_in: int):
    return 1 / math.sqrt(fan_in)

def stable_angle(x: torch.tensor, eps=1e-8):
    imag = x.imag
    y = x.clone()
    y.imag[(imag < eps) & (imag > -1.0 * eps)] = eps
    return torch.angle(y)

def apply_layer(z: torch.Tensor, module: nn.Module, normalization):
    psi = torch.complex(module.layer(z.real), module.layer(z.imag))
    synchrony_term = psi.abs() + module.magnitude_bias
    output_phase = stable_angle(psi) + module.phase_bias
    classic_term = module.layer(z.abs()) + module.magnitude_bias
    intermediate_magnitude = 0.5*synchrony_term + 0.5*classic_term
    output_magnitude = nn.functional.relu(normalization(intermediate_magnitude))
    output = torch.complex(output_magnitude*torch.cos(output_phase), output_magnitude*torch.sin(output_phase))
    return output