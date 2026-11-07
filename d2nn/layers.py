import torch
import torch.nn as nn
import numpy as np

from d2nn import config
from . import optics

def _get_quarter_circle_mask(region_type: str, size: int, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    Y, X = torch.meshgrid(
        torch.arange(size, device=device),
        torch.arange(size, device=device),
        indexing='ij'
    )

    if region_type == "NW":
        cx, cy = 0, 0
    elif region_type == "NE":
        cx, cy = size - 1, 0
    elif region_type == "SW":
        cx, cy = 0, size - 1
    elif region_type == "SE":
        cx, cy = size - 1, size - 1
    else:
        cx, cy = 0, 0

    dist_sq = (X - cx) ** 2 + (Y - cy) ** 2

    mask = (dist_sq <= size ** 2).float()
    return mask

class PixelDiffractiveLayer(nn.Module):

    def __init__(self, output_res: int = config.SUB_SIZE,
                 smooth_init: bool = False,
                 init_modes: int = 31):
        super().__init__()
        self.output_res = output_res

        if smooth_init:
            assert init_modes % 2 == 1 and init_modes <= output_res, (
                "init_modes must be odd and <= output_res"
            )

            coeffs_r = torch.randn(init_modes, init_modes) * config.PIXEL_INIT_STD
            coeffs_i = torch.randn(init_modes, init_modes) * config.PIXEL_INIT_STD
            coeffs = torch.complex(coeffs_r, coeffs_i)

            padded = torch.zeros(output_res, output_res, dtype=torch.complex64)
            c = output_res // 2
            h = init_modes // 2
            padded[c - h: c + h + 1, c - h: c + h + 1] = coeffs
            spatial = torch.fft.ifft2(torch.fft.ifftshift(padded))
            init = spatial.real

            cur_std = init.std()
            if cur_std > 1e-8:
                init = init * (config.PIXEL_INIT_STD / cur_std)
            self.phase = nn.Parameter(init)
        else:
            self.phase = nn.Parameter(
                torch.randn(output_res, output_res) * config.PIXEL_INIT_STD
            )

    def get_phase_mask(self) -> torch.Tensor:
        return self.phase

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        phase = self.phase.to(x.device)
        return x * torch.exp(1j * phase)
