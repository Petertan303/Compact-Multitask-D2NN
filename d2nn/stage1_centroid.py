import os
import json
import time
import numpy as np
import torch
import torch.optim as optim

from d2nn import config
from d2nn import data
from d2nn import optics
from d2nn import losses
from d2nn.training import _set_seed
from d2nn.training import _sample_batch
from d2nn.training import _place_image
from d2nn.training import _make_differential_gallery
from d2nn.readout import _get_balanced_indices
from d2nn.readout import _make_differential_masks

def compute_quarter_circle_centroids(size: int = config.SUB_SIZE):

    R = float(size)
    offset = 4.0 * R / (3.0 * np.pi)
    last = size - 1
    cents = {
        "NW": (offset,        offset       ),
        "NE": (last - offset, offset       ),
        "SW": (offset,        last - offset),
        "SE": (last - offset, last - offset),
    }

    return {r: (int(round(c[0])), int(round(c[1]))) for r, c in cents.items()}, offset
