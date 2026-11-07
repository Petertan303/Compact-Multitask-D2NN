import argparse
import os
import random
import numpy as np
import torch
from torch.utils.data import Subset

from d2nn import config
from d2nn import data
from . import optics
from . import training as trainmod

def _get_balanced_indices(dataset, n_per_class: int, n_classes: int = 10, seed: int = 0):
    rng = random.Random(seed)
    by_label = {i: [] for i in range(n_classes)}
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    for idx in indices:
        if all(len(v) >= n_per_class for v in by_label.values()): break
        _, label = dataset[idx]
        lab = int(label) % n_classes
        if len(by_label[lab]) < n_per_class: by_label[lab].append(idx)
    out = []
    for c in range(n_classes): out.extend(by_label[c][:n_per_class])
    return out

def _make_differential_masks(cx, cy, sub_size, dx, limit_pos, limit_neg, sigma, n_classes, device):
    gal_pos = torch.zeros(n_classes, sub_size, sub_size, device=device)
    gal_neg = torch.zeros(n_classes, sub_size, sub_size, device=device)
    Y, X = torch.meshgrid(torch.arange(sub_size, device=device), torch.arange(sub_size, device=device), indexing='ij')
    X = (X - cx) * dx
    Y = (Y - cy) * dx
    for c in range(n_classes):
        angle_pos = c * (2 * np.pi / n_classes)
        x_pos = limit_pos * np.cos(angle_pos)
        y_pos = limit_pos * np.sin(angle_pos)
        gal_pos[c] = torch.exp(-((X - x_pos)**2 + (Y - y_pos)**2) / (2 * sigma**2))

        angle_neg = angle_pos + np.pi + (np.pi / n_classes)
        x_neg = limit_neg * np.cos(angle_neg)
        y_neg = limit_neg * np.sin(angle_neg)
        gal_neg[c] = torch.exp(-((X - x_neg)**2 + (Y - y_neg)**2) / (2 * sigma**2))

    mask_pos = (gal_pos > 0.5 * gal_pos.max()).float()
    mask_neg = (gal_neg > 0.5 * gal_neg.max()).float()
    return mask_pos, mask_neg
