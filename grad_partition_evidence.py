"""Roughness partition evidence for R1.2.

R1.2 asks whether the smooth-shared / rough-exclusive partition reported in
Fig. 3(b) is something training created, or merely the sqrt(2) smoothing that
averaging two uncorrelated phase profiles produces in the overlap band.  This
script measures the spatial-gradient magnitude |grad phi| of the phase maps
under four conditions and reports the same statistic for the null hypothesis
itself:

  subNets      the four stage-1 quarter-disk sub-networks, taken separately
  mergedInit   the phase-averaged merge of those four, epoch 0 of stage 2
  trained      the fine-tuned shared device (true mainline, 1000 epochs)
  randomInit   an untrained random phase screen (the device the stage-2 recipe
               would start from without phase averaging), and, once available,
               the same stage-2 recipe started from that screen

For every layer it reports the median, mean and 90th percentile of |grad phi|
over the exclusive part of a quarter disk (covered once) and over the shared
part (covered twice), and the ratio shared/exclusive.  It also measures the
averaging factor directly: inside each pairwise overlap band, the median
|grad| of the averaged pair divided by the mean of the medians of the two
original sub-nets.  For two uncorrelated profiles that factor is 1/sqrt(2) =
0.7071, which is the number R1.2 quotes.
"""
import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))

import train_stage2 as t2
from d2nn import config
from d2nn.stage_utils import SubNetNL
from d2nn.layers import _get_quarter_circle_mask

DEV = config.DEVICE
REG = list(config.REGIONS)
N = config.N
PAIRS = [("NW", "NE"), ("NW", "SW"), ("NE", "SE"), ("SW", "SE")]
OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)
N_LAYERS = 3
SUB_TAG = "bg1e3"


def grad_mag(phi):
    """Central-difference gradient magnitude of a phase map, edges excluded."""
    gx = (phi[:, 2:] - phi[:, :-2]) * 0.5
    gy = (phi[2:, :] - phi[:-2, :]) * 0.5
    g = torch.zeros_like(phi)
    g[1:-1, 1:-1] = torch.sqrt(gx[1:-1, :] ** 2 + gy[:, 1:-1] ** 2)
    return g


def stats(v):
    v = v.detach().cpu().numpy().ravel()
    v = v[np.isfinite(v)]
    return {"n": int(v.size), "median": float(np.median(v)),
            "mean": float(v.mean()), "p90": float(np.percentile(v, 90)),
            "p99": float(np.percentile(v, 99))}


def load_subs():
    subs = {}
    for r in REG:
        p = os.path.join(HERE, "checkpoints",
                         "sub_%s_%dlayer_%s.pth" % (r, N_LAYERS, SUB_TAG))
        s = SubNetNL(n_layers=N_LAYERS, region_type=r).to(DEV)
        s.load_state_dict(torch.load(p, map_location=DEV, weights_only=False))
        subs[r] = s
    return subs


def sub_phase_canvas(sub, r, layer):
    """Embed one sub-net's phase into the 300x300 frame, masked to its disk."""
    canvas = torch.zeros(N, N, device=DEV)
    ox, oy = t2.OFFSETS[r]
    m = _get_quarter_circle_mask(r, config.SUB_SIZE, DEV)
    canvas[oy:oy + config.SUB_SIZE, ox:ox + config.SUB_SIZE] = \
        sub.get_spatial_phase(layer).to(DEV) * m
    mask = torch.zeros(N, N, device=DEV)
    mask[oy:oy + config.SUB_SIZE, ox:ox + config.SUB_SIZE] = m
    return canvas, mask


def device_layers(kind, ckpt=None):
    """Return the list of three phase maps for one device state."""
    if kind in ("mergedInit", "trained"):
        full = t2.build_init("merged", N_LAYERS, SUB_TAG).to(DEV)
        if kind == "trained":
            full.load_state_dict(torch.load(ckpt, map_location=DEV,
                                            weights_only=False))
        return [l.phase.detach() for l in full.layers]
    if kind == "randomInit":
        if ckpt:
            full = t2.build_init("merged", N_LAYERS, SUB_TAG).to(DEV)
            full.load_state_dict(torch.load(ckpt, map_location=DEV,
                                            weights_only=False))
            return [l.phase.detach() for l in full.layers]
        return [torch.randn(N, N, device=DEV) * float(a) for a in
                [config.PIXEL_INIT_STD] * N_LAYERS]
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trained", required=True,
                    help="checkpoint of the fine-tuned shared device")
    ap.add_argument("--rand-trained", default=None,
                    help="optional checkpoint of the same recipe started from "
                         "a random phase screen")
    ap.add_argument("--tag", default="full1k")
    a = ap.parse_args()

    t2.SUB_DIR = os.path.join(HERE, "checkpoints")
    excl, ovlp, shared_all = t2.build_grad_masks()
    excl = {r: excl[r].to(DEV) for r in REG}
    shared_all = shared_all.to(DEV)
    excl_all = torch.zeros(N, N, device=DEV)
    for r in REG:
        excl_all = torch.maximum(excl_all, excl[r])

    devices = [("mergedInit", None), ("trained", a.trained),
               ("randomInit", None)]
    if a.rand_trained:
        devices.append(("randomInit_trained", a.rand_trained))

    out = {"tag": a.tag, "regions": {
        "exclusive_px": float(excl_all.sum()),
        "shared_px": float(shared_all.sum())}}

    # ---- stage-1 sub-nets and the averaging factor (the sqrt(2) null) ----
    subs = load_subs()
    avg_factor = {}
    for pair in PAIRS:
        a1, a2 = pair
        per_layer = []
        for li in range(N_LAYERS):
            ca, ma = sub_phase_canvas(subs[a1], a1, li)
            cb, mb = sub_phase_canvas(subs[a2], a2, li)
            band = ((ma > 0) & (mb > 0))
            g_avg = grad_mag(0.5 * (ca + cb))[band]
            g_a, g_b = grad_mag(ca)[band], grad_mag(cb)[band]
            m_avg = float(g_avg.median())
            ref = 0.5 * (float(g_a.median()) + float(g_b.median()))
            per_layer.append({"band_px": int(band.sum()),
                              "median_avg": m_avg,
                              "median_a": float(g_a.median()),
                              "median_b": float(g_b.median()),
                              "ratio_vs_subnets": m_avg / ref})
        avg_factor["%s+%s" % pair] = per_layer
        print("averaging factor %s+%s per layer: %s (null 0.7071)"
              % (a1, a2, [round(x["ratio_vs_subnets"], 3)
                          for x in per_layer]), flush=True)
    out["averaging_factor"] = avg_factor
    out["subnets"] = {r: [stats(grad_mag(subs[r].get_spatial_phase(li).to(DEV)))
                          for li in range(N_LAYERS)] for r in REG}

    # ---- |grad phi| split by exclusive and shared regions, per device ----
    fig, axes = plt.subplots(len(devices), N_LAYERS,
                             figsize=(3.1 * N_LAYERS, 2.3 * len(devices)),
                             squeeze=False)
    for di, (kind, ck) in enumerate(devices):
        phases = device_layers(kind.split("_")[0], ck)
        rows = []
        for li, phi in enumerate(phases):
            g = grad_mag(phi)
            e = stats(g[excl_all > 0])
            s = stats(g[shared_all > 0])
            rows.append({"layer": li + 1, "exclusive": e, "shared": s,
                         "ratio_median": s["median"] / e["median"],
                         "ratio_mean": s["mean"] / e["mean"]})
            ax = axes[di][li]
            bins = np.logspace(np.log10(max(1e-4, float(
                torch.quantile(g[g > 0], 0.01)))),
                np.log10(float(g.max())), 60)
            ax.hist(g[excl_all > 0].cpu().numpy(), bins=bins,
                    histtype="step", label="exclusive", color="tab:blue")
            ax.hist(g[shared_all > 0].cpu().numpy(), bins=bins,
                    histtype="step", label="shared", color="tab:orange")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title("%s  layer %d  shared/exclusive %.2f"
                         % (kind, li + 1, rows[-1]["ratio_median"]),
                         fontsize=8)
            ax.tick_params(labelsize=6)
            if li == 0:
                ax.legend(fontsize=6)
            print("%-20s layer %d  exclusive median %.4f  shared median %.4f "
                  "ratio %.3f" % (kind, li + 1, e["median"], s["median"],
                                  rows[-1]["ratio_median"]), flush=True)
        out.setdefault("devices", {})[kind] = rows
    fig.suptitle("|grad phi| in exclusive and shared regions  (%s)" % a.tag,
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "grad_hist_%s.png" % a.tag), dpi=200)

    # ---- phase maps ----
    fig, axes = plt.subplots(len(devices), N_LAYERS,
                             figsize=(3.1 * N_LAYERS, 2.3 * len(devices)),
                             squeeze=False)
    for di, (kind, ck) in enumerate(devices):
        for li, phi in enumerate(device_layers(kind.split("_")[0], ck)):
            ax = axes[di][li]
            ax.imshow(phi.cpu().numpy(), cmap="twilight", vmin=-np.pi,
                      vmax=np.pi)
            ax.contour(shared_all.cpu().numpy(), levels=[0.5], colors="k",
                       linewidths=0.8)
            ax.set_xticks([])
            ax.set_yticks([])
            if di == 0:
                ax.set_title("layer %d" % (li + 1), fontsize=8)
            if li == 0:
                ax.set_ylabel(kind, fontsize=8)
    fig.suptitle("phase maps, outline = shared region  (%s)" % a.tag,
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "phase_maps_%s.png" % a.tag), dpi=200)

    json.dump(out, open(os.path.join(OUT, "grad_evidence_%s.json" % a.tag),
                        "w"), indent=1)
    print("saved ->", os.path.join(OUT, "grad_evidence_%s.json" % a.tag))


if __name__ == "__main__":
    main()
