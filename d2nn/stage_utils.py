import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from d2nn import config
from d2nn import data
from d2nn import optics
from d2nn import losses
from d2nn.layers import PixelDiffractiveLayer
from d2nn.layers import _get_quarter_circle_mask
from d2nn.training import _set_seed
from d2nn.training import _sample_batch
from d2nn.training import _place_image
from d2nn.training import _make_differential_gallery
from d2nn.readout import _get_balanced_indices
from d2nn.readout import _make_differential_masks

def pad_input_to_canvas(img: torch.Tensor, canvas_size: int = config.SUB_SIZE,
                          upsample: int = 1,
                          centroid: tuple = None,
                          noise_std: float = 0.0) -> torch.Tensor:
    if upsample != 1:
        img = F.interpolate(
            img.unsqueeze(0).unsqueeze(0),
            scale_factor=upsample, mode="bilinear", align_corners=False
        ).squeeze(0).squeeze(0)
    canvas = torch.zeros((canvas_size, canvas_size), dtype=img.dtype, device=img.device)
    h, w = img.shape[-2:]
    if centroid is None:
        cx, cy = canvas_size // 2, canvas_size // 2
    else:
        cx, cy = centroid
    x1, y1 = cx - w // 2, cy - h // 2
    x2, y2 = x1 + w, y1 + h
    sx1 = max(0, -x1);       sy1 = max(0, -y1)
    sx2 = min(w, canvas_size - x1); sy2 = min(h, canvas_size - y1)
    dx1 = max(0, x1);        dy1 = max(0, y1)
    dx2 = min(canvas_size, x2); dy2 = min(canvas_size, y2)
    ww = min(sx2 - sx1, dx2 - dx1); hh = min(sy2 - sy1, dy2 - dy1)
    if ww > 0 and hh > 0:
        canvas[dy1:dy1 + hh, dx1:dx1 + ww] = img[sy1:sy1 + hh, sx1:sx1 + ww]
    if noise_std > 0:
        zero_mask = (canvas == 0)
        noise = torch.randn_like(canvas) * noise_std
        canvas = torch.where(zero_mask, noise, canvas)
    return canvas

class SubNetNL(nn.Module):
    def __init__(self, n_layers: int, region_type: str = "NW",
                 smooth_init: bool = False, init_modes: int = 31):
        super().__init__()
        self.n_layers = n_layers
        self.region_type = region_type
        self.propagator = optics.FreeSpacePropagation(
            n=config.SUB_SIZE, dx=config.DX,
            distance=config.DISTANCE, wavelength=config.WAVELENGTH,
        )
        self.layers = nn.ModuleList([
            PixelDiffractiveLayer(config.SUB_SIZE, smooth_init=smooth_init,
                                   init_modes=init_modes)
            for _ in range(n_layers)
        ])
        self.register_buffer(
            "mask", _get_quarter_circle_mask(region_type, config.SUB_SIZE)
        )

    def forward(self, x):
        for layer in self.layers:
            phase = layer.get_phase_mask().to(x.device)
            phase = phase * self.mask
            x = x * torch.exp(1j * phase)
            x = self.propagator(x)
        return x

    def get_spatial_phase(self, layer_idx):
        return self.layers[layer_idx].get_phase_mask() * self.mask
