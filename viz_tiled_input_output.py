"""Input and detector-plane views for the 150x150 tiled control arms.

Same layout as viz_input_output.py: four rows, one per channel, two
columns (input intensity and detector-plane intensity on a log window).  The
misplaced view puts a foreign dataset in the same window, using the cyclic
assignment NW receives NE data, NE receives SW, SW receives SE, SE receives NW,
which is the assignment the anti-crosstalk term is trained on.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))

import eval_tiled_control as etc
import control_tiled_lib as t15
from d2nn import config, data
from d2nn.readout import _get_balanced_indices

DEV = config.DEVICE
REG = list(config.REGIONS)
NAME = {"NW": "MNIST", "NE": "Fashion", "SW": "EMNIST", "SE": "Shapes"}
WRONG_FOR_CORNER = {"NW": "NE", "NE": "SW", "SW": "SE", "SE": "NW"}
CLS = 5


@torch.no_grad()
def panel(net, cv, host, pos, neg, ring):
    I = torch.abs(net(cv)) ** 2
    P = I.flatten(1).sum() + 1e-12
    ep = float((I[0] * pos[host].sum(0)).sum()) / P
    en = float((I[0] * neg[host].sum(0)).sum()) / P
    frac = float((I[0] * ring[host]).sum()) / P
    return I[0].cpu().numpy(), ep, en, frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--view", choices=("correct", "wrong", "both"),
                    default="both")
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--decades", type=float, default=3.0)
    a = ap.parse_args()

    net = t15.TiledFullNet().to(DEV)
    net.load_state_dict(torch.load(a.ckpt, map_location=DEV,
                                   weights_only=False))
    net.eval()
    ring, pos, neg = t15.build_masks()
    ds = data.load_datasets(train=False)

    views = ([0] if a.view == "correct" else
             [1] if a.view == "wrong" else [0, 1])
    n_rows = len(views) * len(REG)
    fig, axes = plt.subplots(n_rows, 2, figsize=(6.6, 2.05 * n_rows),
                             squeeze=False)
    row = 0
    for v in views:
        for host in REG:
            src = host if v == 0 else WRONG_FOR_CORNER[host]
            j = _get_balanced_indices(ds[src], n_per_class=1, seed=0)[CLS]
            img = ds[src][int(j)][0]
            if img.dim() == 3:
                img = img.squeeze(0)
            cv = t15.full_input(img, src, host)
            I_out, ep, en, frac = panel(net, cv, host, pos, neg, ring)
            I_in = torch.abs(cv[0]) ** 2
            for c, I in enumerate((I_in.cpu().numpy(), I_out)):
                ax = axes[row][c]
                if c == 0:
                    ax.imshow(I, cmap="gray", origin="upper")
                else:
                    ax.imshow(np.log10(I + 1e-14), cmap="inferno",
                              origin="upper",
                              vmin=np.log10(I.max()) - a.decades,
                              vmax=np.log10(I.max()))
                ax.set_xticks([])
                ax.set_yticks([])
                if c == 0:
                    ax.set_ylabel("%s -> %s" % (NAME[src], host), fontsize=7.5)
                if row == 0:
                    ax.set_title("input" if c == 0 else
                                 "detector plane (3 decades)", fontsize=8)
            print("%-4s %-16s own spots %.2f%%  positive %.2f%% negative "
                  "%.2f%%" % (host, "correct" if v == 0 else "misplaced",
                              100 * frac, 100 * ep, 100 * en), flush=True)
            row += 1
    fig.suptitle("tiled control  %s  (%s)" % (a.tag,
                                             os.path.basename(a.ckpt)),
                 fontsize=9)
    fig.tight_layout()
    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "io_%s_%s.png" % (a.tag, a.view))
    fig.savefig(p, dpi=220)
    print("saved ->", p)


if __name__ == "__main__":
    main()

