import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from d2nn import config
from d2nn import data
from d2nn import optics
from d2nn.layers import PixelDiffractiveLayer
from d2nn.layers import _get_quarter_circle_mask
from d2nn import losses
from d2nn.stage_utils import SubNetNL
from d2nn.stage_utils import pad_input_to_canvas
from d2nn.stage1_centroid import compute_quarter_circle_centroids
from d2nn.readout import _make_differential_masks
from d2nn.readout import _get_balanced_indices
from d2nn.training import _set_seed
from d2nn.training import _sample_batch
from d2nn.training import _make_differential_gallery

CENTS, _ = compute_quarter_circle_centroids()

N_LAYERS = 3

class FullNet3L(nn.Module):
    def __init__(self, smooth_init=False, init_modes=31):
        super().__init__()
        self.n_layers = N_LAYERS
        self.propagator = optics.FreeSpacePropagation(
            n=config.N, dx=config.DX,
            distance=config.DISTANCE, wavelength=config.WAVELENGTH,
        )
        self.layers = nn.ModuleList([
            PixelDiffractiveLayer(config.N, smooth_init=smooth_init,
                                   init_modes=init_modes)
            for _ in range(N_LAYERS)
        ])
        global_mask = torch.zeros((config.N, config.N), dtype=torch.float32)
        n_off = config.SUB_SIZE - config.OVERLAP
        for r, ox, oy in [("NW", 0, 0), ("NE", n_off, 0),
                          ("SW", 0, n_off), ("SE", n_off, n_off)]:
            m_sub = _get_quarter_circle_mask(r, config.SUB_SIZE)
            global_mask[oy:oy + config.SUB_SIZE, ox:ox + config.SUB_SIZE] = torch.max(
                global_mask[oy:oy + config.SUB_SIZE, ox:ox + config.SUB_SIZE],
                m_sub)
        self.register_buffer("global_mask", global_mask)

        self.register_buffer("region_mask", torch.zeros((4, config.N, config.N),
                                                         dtype=torch.float32))
        for ri, (r, ox, oy) in enumerate([("NW", 0, 0), ("NE", n_off, 0),
                                            ("SW", 0, n_off), ("SE", n_off, n_off)]):
            m = _get_quarter_circle_mask(r, config.SUB_SIZE)
            self.region_mask[ri, oy:oy + config.SUB_SIZE, ox:ox + config.SUB_SIZE] = m

    def forward(self, x):
        for layer in self.layers:
            phase = layer.get_phase_mask().to(x.device)
            phase = phase * self.global_mask
            x = x * torch.exp(1j * phase)
            x = self.propagator(x)
        return x

def merge_3layer_subs(subs):
    first = next(iter(subs.values()))
    device = next(first.parameters()).device
    full = FullNet3L().to(device)
    n_off = config.SUB_SIZE - config.OVERLAP
    for li in range(N_LAYERS):
        canvas = torch.zeros((config.N, config.N), dtype=torch.float32, device=device)
        count  = torch.zeros((config.N, config.N), dtype=torch.float32, device=device)
        for region, ox, oy in [("NW", 0, 0), ("NE", n_off, 0),
                                 ("SW", 0, n_off), ("SE", n_off, n_off)]:
            phase_sub = subs[region].get_spatial_phase(li).to(device)
            mask_sub  = _get_quarter_circle_mask(region, config.SUB_SIZE, device)
            canvas[oy:oy + config.SUB_SIZE, ox:ox + config.SUB_SIZE] += phase_sub
            count [oy:oy + config.SUB_SIZE, ox:ox + config.SUB_SIZE] += mask_sub
        phase_avg = canvas / torch.clamp(count, min=1.0)
        with torch.no_grad():
            full.layers[li].phase.copy_(phase_avg)
    return full
