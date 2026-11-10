"""Input and detector-plane views only (no intermediate layers).

Eight rows: for each of the four channels, the sample placed in its own
corner and a foreign sample placed in that same corner.  Two columns: the
input intensity and the output intensity on a 3-decade log window.  Each row
prints the energy fractions and the detection readout s_hat.
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

import train_stage2 as t2
import train_stage2_shared as fd
from d2nn import config, data, losses
from d2nn.readout import _get_balanced_indices

DEV = config.DEVICE
REGIONS = list(config.REGIONS)
WRONG_FOR_CORNER = {"NW": "NE", "NE": "SW", "SW": "SE", "SE": "NW"}
NAME = {"NW": "MNIST", "NE": "Fashion", "SW": "EMNIST", "SE": "Shapes"}
CLS = 5


@torch.no_grad()
def forward_io(full, canvas):
    x = canvas.to(torch.complex64)
    for layer in full.layers:
        phase = layer.get_phase_mask().to(x.device) * full.global_mask
        x = x * torch.exp(1j * phase)
        x = full.propagator(x)
    return torch.abs(canvas[0]) ** 2, torch.abs(x[0]) ** 2


def readout(det, r, pos, neg):
    tot = float(det.sum()) + 1e-12
    ep = float((det * pos[r].sum(0)).sum())
    en = float((det * neg[r].sum(0)).sum())
    per_c = []
    for c in range(config.N_CLASSES):
        per_c.append((float((det * pos[r][c]).sum())
                      - float((det * neg[r][c]).sum())) / tot)
    return max(per_c), ep / tot, en / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", default="io")
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--decades", type=float, default=3.0)
    ap.add_argument("--no-overlays", action="store_true")
    ap.add_argument("--view", choices=("correct", "wrong", "both"),
                    default="both",
                    help="which placement to draw; 'both' stacks the two "
                         "views of every channel in one figure")
    a = ap.parse_args()

    full = t2.FullNetNL(a.layers).to(DEV)
    full.load_state_dict(torch.load(a.ckpt, map_location=DEV,
                                    weights_only=False))
    full.eval()
    print("phase std per layer: %s rad"
          % [round(float(l.phase.std().detach()), 2) for l in full.layers])

    pos, neg = t2.build_masks300()
    ring4 = torch.zeros(config.N, config.N, device=DEV)
    core4 = torch.zeros(config.N, config.N, device=DEV)
    big_rings = fd.build_big_rings()[0]
    for r in REGIONS:
        ring4 = torch.maximum(ring4, big_rings[r])
        core4 = torch.maximum(core4, losses.spot_ring(pos[r], neg[r]))
    ds_test = data.load_datasets(train=False)

    views = ([0] if a.view == "correct" else
             [1] if a.view == "wrong" else [0, 1])
    n_rows = len(views) * len(REGIONS)
    fig, axes = plt.subplots(n_rows, 2, figsize=(6.6, 2.05 * n_rows),
                             squeeze=False)
    row = 0
    for v in views:
        for r in REGIONS:
            if v == 0:
                label, ds = "own %s" % NAME[r], ds_test[r]
            else:
                label = "wrong %s" % NAME[WRONG_FOR_CORNER[r]]
                ds = ds_test[WRONG_FOR_CORNER[r]]
            j = _get_balanced_indices(ds, n_per_class=1, seed=0)[CLS]
            canv = t2.test_canvas(r, fd._img(ds, j))
            I_in, I_out = forward_io(full, canv)
            tot = float(I_out.sum()) + 1e-12
            f_big = float((I_out * ring4).sum()) / tot
            f_core = float((I_out * core4).sum()) / tot
            own_core = float((I_out * losses.spot_ring(pos[r], neg[r])).sum()) / tot
            s_max, f_pos, f_neg = readout(I_out, r, pos, neg)
            p = I_out.flatten() / tot
            n_eff = float(1.0 / (p * p).sum())
            f_top = float(torch.topk(p, 900).values.sum())
            for c, I in enumerate((I_in, I_out)):
                ax = axes[row][c]
                In = I.cpu().numpy()
                vmax = float(In.max())
                if c == 0:
                    ax.imshow(In, cmap="gray", origin="upper", vmin=0.0,
                              vmax=max(vmax, 1e-9))
                else:
                    ax.imshow(np.log10(In + 1e-14), cmap="inferno",
                              origin="upper",
                              vmin=np.log10(vmax) - a.decades,
                              vmax=np.log10(vmax))
                    if not a.no_overlays:
                        ax.contour(core4.cpu().numpy(), levels=[0.5],
                                   colors="cyan", linewidths=0.6)
                        ox, oy = t2.slot(r)
                        ax.add_patch(plt.Rectangle((ox - .5, oy - .5),
                                                   config.SUB_SIZE,
                                                   config.SUB_SIZE, fill=False,
                                                   edgecolor="white", lw=0.8))
                ax.set_xticks([])
                ax.set_yticks([])
                if c == 0:
                    ax.set_ylabel("%s -> %s" % (label, r), fontsize=7.5)
                if row == 0:
                    ax.set_title("input" if c == 0
                                 else "detector plane (3 decades)",
                                 fontsize=8)
            print("%-4s %-16s big-ring %6.2f%%  core4 %6.2f%%  own-core "
                  "%6.2f%%  pos/neg own %.3f%%/%.3f%%  s_hat_max %+.5f  "
                  "N_eff %.0f px  brightest-1%% %.1f%%"
                  % (r, label, 100 * f_big, 100 * f_core, 100 * own_core,
                     100 * f_pos, 100 * f_neg, s_max, n_eff, 100 * f_top),
                  flush=True)
            row += 1
    fig.suptitle("%s  (%s)" % (a.tag, os.path.basename(a.ckpt)), fontsize=9)
    fig.tight_layout()
    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "io_%s_%s.png" % (a.tag, a.view))
    fig.savefig(p, dpi=220)
    print("saved ->", p)


if __name__ == "__main__":
    main()
