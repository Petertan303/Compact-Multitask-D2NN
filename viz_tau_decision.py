"""Per-class detection decision and the paper-style three-panel figure.

Test set: 100 samples per class in every channel.  For each channel we collect
the differential readout s_c = (E+_c - E-_c)/P_out for the ten classes under
three illumination conditions: the own dataset in its own corner (target), a
foreign dataset in that corner (off-target), and a noise-only window (empty).

Three threshold calibrations are compared:
  empty   tau = 99th percentile of the empty-window s_max       (FAR_empty=1%)
  youden  tau that maximises (TPR - FAR) against off-target inputs
  fixed   tau = 0.2, the value quoted in the submitted manuscript
A channel is drawn red when its largest s_c reaches tau.
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
import train_stage2_shared as fd
from d2nn import config, data
from d2nn.readout import _get_balanced_indices

DEV = config.DEVICE
REGIONS = list(config.REGIONS)
NAME = {"NW": "MNIST", "NE": "Fashion", "SW": "EMNIST", "SE": "Shapes"}
NC = config.N_CLASSES
N_PER_CLASS = 100
BATCH = 128


@torch.no_grad()
def shat_all(full, canvases, host, pos, neg):
    vals = np.empty((canvases.shape[0], NC))
    for i in range(0, canvases.shape[0], BATCH):
        out = full(canvases[i:i + BATCH].to(torch.complex64))
        I = torch.abs(out) ** 2
        P = I.flatten(1).sum(1) + 1e-8
        ep = (I.unsqueeze(1) * pos[host]).flatten(start_dim=2).sum(dim=2)
        en = (I.unsqueeze(1) * neg[host]).flatten(start_dim=2).sum(dim=2)
        vals[i:i + BATCH] = ((ep - en) / P.unsqueeze(1)).cpu().numpy()
    return vals


@torch.no_grad()
def shat_one(full, canvas, host, pos, neg):
    return shat_all(full, canvas.unsqueeze(0), host, pos, neg)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", default="tau")
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--decades", type=float, default=3.0)
    a = ap.parse_args()

    full = t2.FullNetNL(a.layers).to(DEV)
    full.load_state_dict(torch.load(a.ckpt, map_location=DEV,
                                    weights_only=False))
    full.eval()
    pos, neg = t2.build_masks300()
    ds = data.load_datasets(train=False)

    data_by_region = {}
    for host in REGIONS:
        own_idx = _get_balanced_indices(ds[host], n_per_class=N_PER_CLASS,
                                        seed=0)
        own_lab = np.array([int(ds[host][j][1]) % NC for j in own_idx])
        canv = torch.cat([t2.test_canvas(host, fd._img(ds[host], j))
                          for j in own_idx])
        s_own = shat_all(full, canv, host, pos, neg)

        s_off, off_src = [], []
        off_pairs = []
        for src in REGIONS:
            if src == host:
                continue
            sidx = _get_balanced_indices(ds[src], n_per_class=N_PER_CLASS,
                                         seed=0)
            cc = torch.cat([t2.test_canvas(host, fd._img(ds[src], j))
                            for j in sidx])
            s_off.append(shat_all(full, cc, host, pos, neg))
            off_pairs += [(src, int(j)) for j in sidx]
        s_off = np.concatenate(s_off, axis=0)

        blank = torch.stack([fd.blank_canvas(host) for _ in range(500)])
        s_emp = shat_all(full, blank, host, pos, neg)

        data_by_region[host] = {
            "s_own": s_own, "lab": own_lab, "s_off": s_off, "s_emp": s_emp,
            "own_idx": own_idx, "off_pairs": off_pairs,
            "m_own": s_own.max(1), "m_off": s_off.max(1),
            "m_emp": s_emp.max(1),
            "hit_own": s_own.argmax(1) == own_lab,
        }
        print("%s  own s_max mean %.4f p05 %.4f | off mean %.4f p99 %.4f | "
              "empty mean %.4f p99 %.4f"
              % (NAME[host], data_by_region[host]["m_own"].mean(),
                 np.percentile(data_by_region[host]["m_own"], 5),
                 data_by_region[host]["m_off"].mean(),
                 np.percentile(data_by_region[host]["m_off"], 99),
                 data_by_region[host]["m_emp"].mean(),
                 np.percentile(data_by_region[host]["m_emp"], 99)), flush=True)

    schemes = {}
    cand = np.unique(np.concatenate([d["m_own"] for d in
                                     data_by_region.values()]
                                    + [d["m_off"] for d in
                                       data_by_region.values()]))
    for host in REGIONS:
        d = data_by_region[host]
        best, best_j = None, -2.0
        grid = np.quantile(np.concatenate([d["m_own"], d["m_off"]]),
                           np.linspace(0.0, 1.0, 401))
        for th in grid:
            tpr = float((d["m_own"] >= th).mean())
            far = float((d["m_off"] >= th).mean())
            if tpr - far > best_j:
                best_j, best = tpr - far, float(th)
        schemes.setdefault("empty", {})[host] = float(
            np.percentile(d["m_emp"], 99))
        schemes.setdefault("youden", {})[host] = best
        schemes.setdefault("fixed", {})[host] = 0.2

    summary = {}
    for scheme, taus in schemes.items():
        rows = {}
        for host in REGIONS:
            d = data_by_region[host]
            th = taus[host]
            det = d["m_own"] >= th
            rows[host] = {
                "tau": th,
                "tpr": float(det.mean()),
                "tpr_and_right_class": float((det & d["hit_own"]).mean()),
                "far_offtarget": float((d["m_off"] >= th).mean()),
                "far_empty": float((d["m_emp"] >= th).mean()),
            }
        summary[scheme] = rows
        print("\n%s: %s" % (scheme, {NAME[h]: "tau=%.4f TPR=%.2f "
                                     "TPR@right=%.2f FAR_off=%.3f FAR_empty=%.3f"
                                     % (rows[h]["tau"], rows[h]["tpr"],
                                        rows[h]["tpr_and_right_class"],
                                        rows[h]["far_offtarget"],
                                        rows[h]["far_empty"])
                                     for h in REGIONS}))
    json.dump(summary, open(os.path.join(HERE, "results",
                                         "tau_%s.json" % a.tag), "w"), indent=1)

    for scheme in ("empty", "youden", "fixed"):
        make_figure(full, ds, data_by_region, schemes[scheme], pos, neg,
                    a.tag, scheme, a.decades)


@torch.no_grad()
def make_figure(full, ds, db, taus, pos, neg, tag, scheme, decades):
    fig, axes = plt.subplots(len(REGIONS), 6,
                             figsize=(11.4, 2.35 * len(REGIONS)),
                             squeeze=False)
    for ri, host in enumerate(REGIONS):
        d = db[host]
        th = taus[host]
        k_own = int(np.argmin(np.abs(d["m_own"] - np.median(d["m_own"]))))
        k_off = int(np.argmin(np.abs(d["m_off"] - np.median(d["m_off"]))))
        own_canvas = t2.test_canvas(host, fd._img(ds[host], d["own_idx"][k_own]))
        osrc, oj = d["off_pairs"][k_off]
        off_canvas = t2.test_canvas(host, fd._img(ds[osrc], oj))
        blocks = [(own_canvas, d["s_own"][k_own], d["lab"][k_own],
                   "target region"),
                  (off_canvas, d["s_off"][k_off], -1, "off-target region")]
        for bi, (canv, sv, lab, blk) in enumerate(blocks):
            I_in = torch.abs(canv[0]) ** 2
            out = full(canv.to(torch.complex64))
            I_out = torch.abs(out[0]) ** 2
            ax = axes[ri][bi * 3]
            ax.imshow(I_in.cpu().numpy(), cmap="gray", origin="upper")
            ax.set_xticks([])
            ax.set_yticks([])
            if ri == 0:
                ax.set_title("input", fontsize=8)
            if bi == 1 and ri == 0:
                ax.set_title("input  (off-target region)", fontsize=8)
            ax = axes[ri][bi * 3 + 1]
            In = I_out.cpu().numpy()
            ax.imshow(np.log10(In + 1e-14), cmap="inferno", origin="upper",
                      vmin=np.log10(In.max()) - decades, vmax=In.max())
            ax.set_xticks([])
            ax.set_yticks([])
            if ri == 0:
                ax.set_title("output", fontsize=8)
            ax = axes[ri][bi * 3 + 2]
            det = float(sv.max()) >= th
            colors = ["#c0392b" if (det and int(sv.argmax()) == c) else "#2e6da4"
                      for c in range(NC)]
            edges = ["#f1c40f" if (lab >= 0 and c == lab) else "none"
                     for c in range(NC)]
            lw = [1.2 if (lab >= 0 and c == lab) else 0.0 for c in range(NC)]
            ax.bar(range(NC), sv, color=colors, edgecolor=edges, linewidth=lw)
            ax.axhline(th, color="0.35", lw=0.9)
            ax.axhline(0, color="k", lw=0.5)
            ax.set_xticks(range(NC))
            ax.set_xticklabels([str(c) for c in range(NC)], fontsize=6)
            ax.tick_params(labelsize=6)
            top = max(abs(float(sv.min())), abs(float(sv.max())), th) * 1.25
            ax.set_ylim(-top, top)
            if ri == 0:
                ax.set_title("class index    tau=%.4f  %s"
                             % (th, "detected" if det else "nothing"),
                             fontsize=8)
            if lab >= 0:
                ax.text(0.02, 0.96, "true class %d" % lab,
                        transform=ax.transAxes, fontsize=6, va="top",
                        color="#f1c40f")
            if bi * 3 == 0:
                ax.set_ylabel("")
            axes[ri][0].set_ylabel("%s in %s" % (NAME[host], host), fontsize=8)
    fig.suptitle("%s  threshold scheme: %s" % (tag, scheme), fontsize=10)
    fig.tight_layout()
    out_dir = os.path.join(HERE, "results")
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "tau_%s_%s.png" % (tag, scheme))
    fig.savefig(p, dpi=200)
    print("saved ->", p)


if __name__ == "__main__":
    main()
