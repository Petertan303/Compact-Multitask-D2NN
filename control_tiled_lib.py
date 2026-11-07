import os

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))

from d2nn import config
from d2nn import optics
from d2nn import losses
from d2nn.layers import PixelDiffractiveLayer
from d2nn.stage_utils import pad_input_to_canvas
from d2nn.training import _make_differential_gallery
from d2nn.readout import _make_differential_masks
from d2nn.readout import _get_balanced_indices

DEV = config.DEVICE
REG = config.REGIONS
TILE = 150
OFFS = {"NW": (0, 0), "NE": (TILE, 0), "SW": (0, TILE), "SE": (TILE, TILE)}
CENT = TILE // 2
RI = {r: i for i, r in enumerate(REG)}

NOISE = 0.01
CE_SCALE = 100.0
ALPHA_CE = 1e-2
ALPHA_ANTI = 40.0
BATCH = 256
UPSAMPLE = 4
LR = 1e-1
N_LAYERS = 3

def ring_params():
    lp = config.FULL_SIZE * config.SPREAD_SCALE * config.LIMIT_SCALE
    ln = lp * 1.4
    sg = config.FULL_SIZE * config.SIGMA_SCALE
    return lp, ln, sg

class SquareSubNet(nn.Module):

    def __init__(self, n_layers=N_LAYERS, size=TILE):
        super().__init__()
        self.n_layers = n_layers
        self.size = size
        self.propagator = optics.FreeSpacePropagation(
            n=size, dx=config.DX, distance=config.DISTANCE,
            wavelength=config.WAVELENGTH)
        self.layers = nn.ModuleList([
            PixelDiffractiveLayer(size, smooth_init=False, init_modes=31)
            for _ in range(n_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = x * torch.exp(1j * layer.get_phase_mask().to(x.device))
            x = self.propagator(x)
        return x

    def get_spatial_phase(self, layer_idx):
        return self.layers[layer_idx].get_phase_mask()

class TiledFullNet(nn.Module):

    def __init__(self, n_layers=N_LAYERS):
        super().__init__()
        self.n_layers = n_layers
        self.propagator = optics.FreeSpacePropagation(
            n=config.N, dx=config.DX, distance=config.DISTANCE,
            wavelength=config.WAVELENGTH)
        self.layers = nn.ModuleList([
            PixelDiffractiveLayer(config.N, smooth_init=False, init_modes=31)
            for _ in range(n_layers)])
        self.register_buffer("global_mask",
                             torch.ones((config.N, config.N),
                                        dtype=torch.float32))
        rm = torch.zeros((4, config.N, config.N), dtype=torch.float32)
        for r, (ox, oy) in OFFS.items():
            rm[RI[r], oy:oy + TILE, ox:ox + TILE] = 1.0
        self.register_buffer("region_mask", rm)

    def forward(self, x):
        for layer in self.layers:
            phase = layer.get_phase_mask().to(x.device) * self.global_mask
            x = x * torch.exp(1j * phase)
            x = self.propagator(x)
        return x

def tile_subs(subs, n_layers=N_LAYERS):
    device = next(next(iter(subs.values())).parameters()).device
    full = TiledFullNet(n_layers).to(device)
    for li in range(n_layers):
        canvas = torch.zeros((config.N, config.N), dtype=torch.float32,
                             device=device)
        for region, (ox, oy) in OFFS.items():
            canvas[oy:oy + TILE, ox:ox + TILE] = \
                subs[region].get_spatial_phase(li).to(device)
        with torch.no_grad():
            full.layers[li].phase.copy_(canvas)
    return full

def build_masks(tile=TILE):
    lp, ln, sg = ring_params()
    ring, pos300, neg300 = {}, {}, {}
    for r in REG:
        mp, mn = _make_differential_masks(CENT, CENT, tile, config.DX,
                                          lp, ln, sg, config.N_CLASSES, DEV)
        mpb, mnb = (mp > 0.5).float(), (mn > 0.5).float()
        mp300 = torch.zeros(config.N_CLASSES, config.N, config.N, device=DEV)
        mn300 = torch.zeros(config.N_CLASSES, config.N, config.N, device=DEV)
        ox, oy = OFFS[r]
        mp300[:, oy:oy + tile, ox:ox + tile] = mpb
        mn300[:, oy:oy + tile, ox:ox + tile] = mnb
        pos300[r], neg300[r] = mp300, mn300
        ring[r] = ((mp300.sum(0) + mn300.sum(0)) > 0).float()
    return ring, pos300, neg300

def build_sub_masks(tile=TILE):
    lp, ln, sg = ring_params()
    gp, gn = _make_differential_gallery(CENT, CENT, tile, config.DX, lp, ln,
                                        sg, config.N_CLASSES, DEV)
    mp, mn = (gp > 0.5).float(), (gn > 0.5).float()
    ring = ((mp.sum(0) + mn.sum(0)) > 0).float()
    return ring, mp, mn

def sub_canvas(imgs, tile=TILE):
    if imgs.dim() == 4:
        imgs = imgs.reshape(imgs.shape[0], imgs.shape[-2], imgs.shape[-1])
    imgs = losses.normalize_input_to_canvas_energy(
        imgs, target_l2_sq=config.CANVAS_L2_SQ)
    out = torch.stack([pad_input_to_canvas(imgs[i], tile, upsample=UPSAMPLE,
                                           centroid=(CENT, CENT),
                                           noise_std=NOISE)
                       for i in range(imgs.shape[0])])
    return out.to(torch.complex64)

def full_input(imgs, slot_region, anchor_region=None):
    ar = anchor_region or slot_region
    ox, oy = OFFS[ar]
    if imgs.dim() == 2:
        imgs = imgs.unsqueeze(0)
    canvases = []
    for i in range(imgs.shape[0]):
        it = imgs[i]
        if it.dim() == 3:
            it = it.squeeze(0)
        sc = pad_input_to_canvas(it, TILE, upsample=UPSAMPLE,
                                 centroid=(CENT, CENT), noise_std=NOISE)
        if sc.max() > 0:
            sc = sc / sc.max()
        big = torch.zeros(config.N, config.N, device=DEV)
        big[oy:oy + TILE, ox:ox + TILE] = sc
        canvases.append(big)
    return torch.stack(canvases).to(torch.complex64)

def empty_input(anchor_region):
    ox, oy = OFFS[anchor_region]
    sc = torch.randn(TILE, TILE) * NOISE
    big = torch.zeros(config.N, config.N, device=DEV)
    big[oy:oy + TILE, ox:ox + TILE] = sc
    return big.unsqueeze(0).to(torch.complex64)

def differential_scores(I, pos, neg):
    ep = torch.stack([(I * pos[c]).flatten(1).sum(1)
                      for c in range(config.N_CLASSES)], dim=1)
    en = torch.stack([(I * neg[c]).flatten(1).sum(1)
                      for c in range(config.N_CLASSES)], dim=1)
    return ep, en

def eval_acc(net, ds_test, pos300, neg300, n_per_class=100, batch=64):
    net.eval()
    accs = {}
    with torch.no_grad():
        for r in REG:
            idx = _get_balanced_indices(ds_test[r], n_per_class=n_per_class)
            c = tot = 0
            for s in range(0, len(idx), batch):
                ids = idx[s:s + batch]
                ims = torch.stack([ds_test[r][i][0] for i in ids])
                labs = [int(ds_test[r][i][1]) % 10 for i in ids]
                x = full_input(ims, r)
                I = torch.abs(net(x)) ** 2
                ep, en = differential_scores(I, pos300[r], neg300[r])
                pred = (ep - en).argmax(1).tolist()
                c += sum(int(p == l) for p, l in zip(pred, labs))
                tot += len(ids)
            accs[r] = c / max(tot, 1)
    net.train()
    return accs

def energy_partition(net, ds_test, ring, n_per_class=10, batch=32):
    net.eval()
    m2, m3 = {}, {}
    with torch.no_grad():
        for r in REG:
            idx = _get_balanced_indices(ds_test[r], n_per_class=n_per_class)
            acc = torch.zeros(4, device=DEV)
            for s in range(0, len(idx), batch):
                ims = torch.stack([ds_test[r][i][0] for i in idx[s:s + batch]])
                I = torch.abs(net(full_input(ims, r))) ** 2
                tot = I.flatten(1).sum(1)[:, None] + 1e-12
                own_ring = (I * ring[r]).flatten(1).sum(1)[:, None] / tot
                other_ring = sum((I * ring[r2]).flatten(1).sum(1)[:, None] / tot
                                 for r2 in REG if r2 != r)
                own_off = (I * net.region_mask[RI[r]] * (1 - ring[r])
                           ).flatten(1).sum(1)[:, None] / tot
                rest = 1 - own_ring - other_ring - own_off
                acc += torch.cat([own_ring, own_off, other_ring, rest],
                                 dim=1).sum(0)
            m2[r] = (acc / len(idx)).tolist()
        for a in REG:
            for b in REG:
                if a == b:
                    continue
                idx = _get_balanced_indices(ds_test[a], n_per_class=n_per_class)
                acc = torch.zeros(3, device=DEV)
                for s in range(0, len(idx), batch):
                    ims = torch.stack([ds_test[a][i][0]
                                       for i in idx[s:s + batch]])
                    I = torch.abs(net(full_input(ims, a,
                                                 anchor_region=b))) ** 2
                    tot = I.flatten(1).sum(1)[:, None] + 1e-12
                    b_ring = (I * ring[b]).flatten(1).sum(1)[:, None] / tot
                    a_ring = (I * ring[a]).flatten(1).sum(1)[:, None] / tot
                    oth = sum((I * ring[r2]).flatten(1).sum(1)[:, None] / tot
                              for r2 in REG if r2 not in (a, b))
                    acc += torch.cat([b_ring, a_ring, oth], dim=1).sum(0)
                m3[a + "->" + b] = (acc / len(idx)).tolist()
    net.train()
    return m2, m3
