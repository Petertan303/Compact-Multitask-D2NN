import argparse
import json
import os
import random
import time
import numpy as np
import torch
import torch.optim as optim

from d2nn import config
from d2nn import data
from . import losses
from . import optics

def _set_seed(seed: int = 0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _sample_batch(dataset, batch_size: int, device):
    indices = random.sample(range(len(dataset)), min(batch_size, len(dataset)))
    imgs, labels = [], []
    for idx in indices:
        img, label = dataset[idx]
        if hasattr(img, "squeeze") and img.dim() == 3:
            img = img.squeeze(0)
        imgs.append(img)
        labels.append(int(label) % 10)
    return (torch.stack(imgs).to(device),
            torch.tensor(labels, device=device, dtype=torch.long),
            [int(dataset[i][1]) for i in indices])

def _place_image(img_2d: torch.Tensor, target_xy_pixel: tuple,
                 canvas_size: int, scale: float = 1,
                 placement: str = "centre") -> torch.Tensor:
    if placement == "random_sub":
        import torch.nn.functional as F
        h0 = w0 = img_2d.shape[-1]
        h1, w1 = int(h0 * scale), int(w0 * scale)
        lo, hi = (h1 // 2) + 1, canvas_size - (h1 // 2) - 1
        if hi <= lo:
            cx = cy = canvas_size // 2
        else:
            cx = int(torch.randint(lo, hi + 1, (1,)).item())
            cy = int(torch.randint(lo, hi + 1, (1,)).item())
        target_xy_pixel = (cx, cy)
    return optics.center_image_in_canvas(img_2d, target_xy_pixel, canvas_size, scale=scale)

def _make_differential_gallery(cx, cy, sub_size, dx, limit_pos, limit_neg,
                               sigma, n_classes, device):
    gal_pos = torch.zeros(n_classes, sub_size, sub_size, device=device)
    gal_neg = torch.zeros(n_classes, sub_size, sub_size, device=device)

    Y, X = torch.meshgrid(torch.arange(sub_size, device=device),
                          torch.arange(sub_size, device=device), indexing="ij")
    X = (X - cx) * dx
    Y = (Y - cy) * dx

    for c in range(n_classes):

        angle_pos = c * (2 * np.pi / n_classes)
        x_pos = limit_pos * np.cos(angle_pos)
        y_pos = limit_pos * np.sin(angle_pos)
        D2_pos = (X - x_pos) ** 2 + (Y - y_pos) ** 2
        gal_pos[c] = torch.exp(-D2_pos / (2 * sigma ** 2))

        angle_neg = angle_pos + np.pi + (np.pi / n_classes)
        x_neg = limit_neg * np.cos(angle_neg)
        y_neg = limit_neg * np.sin(angle_neg)
        D2_neg = (X - x_neg) ** 2 + (Y - y_neg) ** 2
        gal_neg[c] = torch.exp(-D2_neg / (2 * sigma ** 2))

    return gal_pos, gal_neg
