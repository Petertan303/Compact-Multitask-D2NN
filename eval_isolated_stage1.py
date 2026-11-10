"""Accuracy of the four isolated stage-1 networks, with three seeds.

This is the "independent networks" row of the comparison: each task is
recognised by its own quarter-disk network, nothing is shared and there is no
second stage.  The readout, the spot layout, the placement at the quarter-disk
centroid and the class-balanced sampling from the official test split are the
same as in the merged device, so the two rows of the table are directly
comparable.

Seed 0 is the checkpoint the merged device was initialised from, written by
train_stage1.py as `checkpoints/sub_<R>_3layer_bg1e3.pth`; seeds 1 and 2 are
`checkpoints/sub_<R>_3layer_bg1e3_s{1,2}.pth`.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))

from d2nn import config, data
from d2nn.stage_utils import SubNetNL, pad_input_to_canvas
from d2nn.training import _make_differential_gallery
from d2nn.readout import _get_balanced_indices
from d2nn.layers import _get_quarter_circle_mask

DEV = config.DEVICE
REG = list(config.REGIONS)
NAME = {"NW": "MNIST", "NE": "Fashion", "SW": "EMNIST", "SE": "Shapes"}
NC = config.N_CLASSES
CENTS = {"NW": (90, 90), "NE": (121, 90), "SW": (90, 121), "SE": (121, 121)}


def sub_masks(r):
    lp = config.FULL_SIZE * config.SPREAD_SCALE * config.LIMIT_SCALE
    ln = lp * 1.4
    sg = config.FULL_SIZE * config.SIGMA_SCALE
    cx, cy = CENTS[r]
    gp, gn = _make_differential_gallery(cx, cy, config.SUB_SIZE, config.DX,
                                        lp, ln, sg, NC, DEV)
    return (gp > 0.5).float(), (gn > 0.5).float()


@torch.no_grad()
def one_region(net, ds_test, r, pos, neg, n_per_class):
    idx = _get_balanced_indices(ds_test[r], n_per_class=n_per_class, seed=0)
    cx, cy = CENTS[r]
    ok, n = 0, 0
    for i in range(0, len(idx), 64):
        sel = idx[i:i + 64]
        canv = []
        for j in sel:
            im = ds_test[r][int(j)][0]
            if im.dim() == 3:
                im = im.squeeze(0)
            sc = pad_input_to_canvas(im, config.SUB_SIZE, upsample=4,
                                     centroid=(cx, cy), noise_std=0.01)
            if sc.max() > 0:
                sc = sc / sc.max()
            canv.append(sc)
        cv = torch.stack(canv).to(DEV).to(torch.complex64)
        I = torch.abs(net(cv)) ** 2
        ep = (I.unsqueeze(1) * pos).flatten(start_dim=2).sum(dim=2)
        en = (I.unsqueeze(1) * neg).flatten(start_dim=2).sum(dim=2)
        lab = torch.tensor([int(ds_test[r][int(j)][1]) % NC for j in sel],
                           device=DEV)
        ok += int((torch.argmax(ep - en, dim=1) == lab).sum())
        n += len(sel)
    return ok / n, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=100)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--sub-tag", default="bg1e3")
    a = ap.parse_args()

    ds_test = data.load_datasets(train=False)
    paths = {}
    for r in REG:
        paths[(r, 0)] = os.path.join(HERE, "checkpoints",
                                     "sub_%s_%dlayer_%s.pth"
                                     % (r, a.layers, a.sub_tag))
        for s in (1, 2):
            paths[(r, s)] = os.path.join(HERE, "checkpoints", "sub_%s_%dlayer_%s_s%d.pth"
                                         % (r, a.layers, a.sub_tag, s))
    out = {}
    for s in (0, 1, 2):
        accs = {}
        for r in REG:
            net = SubNetNL(n_layers=a.layers, region_type=r).to(DEV)
            net.load_state_dict(torch.load(paths[(r, s)], map_location=DEV,
                                           weights_only=False))
            net.eval()
            pos, neg = sub_masks(r)
            acc, n = one_region(net, ds_test, r, pos, neg, a.n_per_class)
            accs[r] = acc
            print("seed %d  %-7s acc %6.2f%%  (n=%d)"
                  % (s, NAME[r], 100 * acc, n), flush=True)
        out["seed%d" % s] = accs
    per_region = {r: [out["seed%d" % s][r] for s in (0, 1, 2)] for r in REG}
    means = [float(np.mean([out[k][r] for r in REG])) for k in out]
    print("\nper-region mean over seeds:")
    for r in REG:
        print("  %-7s %6.2f%%  (seeds %s)"
              % (NAME[r], 100 * np.mean(per_region[r]),
                 ["%.2f" % (100 * x) for x in per_region[r]]))
    print("  overall mean %.2f%%, std over seeds %.2f"
          % (100 * np.mean(means), 100 * np.std(means)))
    out["overall_mean"] = means
    json.dump(out, open(os.path.join(HERE, "results",
                                     "isolated_stage1.json"), "w"), indent=1)


if __name__ == "__main__":
    main()

