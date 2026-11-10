"""Accuracy and false-alarm-rate evaluation for one or several devices.

Sampling.  Everything reported here comes from the official test split,
`d2nn.data.load_datasets(train=False)`, the same split every training run used
and the one no training loop ever touches: MNIST test (10,000 images),
Fashion-MNIST test (10,000), EMNIST letters test split restricted to the ten
letter classes used here, and the test folder of the synthetic Shapes set.
From each dataset `--n-per-class` images are drawn balanced over the ten
classes with a fixed seed, so every cell of every table states how many
samples stand behind it.

Illumination, one non-empty window per canvas:
  own      the dataset that owns the window is placed in that window
  off-tgt  a foreign dataset is placed in that window
  empty    noise-only window, nothing placed

Readout.  On the small spots, S_c = E+_c - E-_c and s_c = S_c / P_out, with
P_out the total output power of the same sample.  A channel reports an object
when max_c s_c reaches its threshold.  Thresholds: the 99th percentile of the
empty-window distribution, the value maximising TPR minus FAR against
off-target inputs, and any fixed value passed with --tau.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))

import train_stage2 as t2
import train_stage2_shared as fd
from d2nn import config, data, losses
from d2nn.readout import _get_balanced_indices

DEV = config.DEVICE
REGIONS = list(config.REGIONS)
NAME = {"NW": "MNIST", "NE": "Fashion", "SW": "EMNIST", "SE": "Shapes"}
NC = config.N_CLASSES
CHUNK = 256
N_EMPTY = 500


@torch.no_grad()
def readout_batched(full, canvases, host, pos, neg, spots):
    """Return the ten normalised readouts, their maximum, and the fraction of
    output power inside this channel's own twenty spots."""
    shat, s_max, core = [], [], []
    for i in range(0, len(canvases), CHUNK):
        cv = torch.cat(canvases[i:i + CHUNK])
        out = full(cv.to(torch.complex64))
        I = torch.abs(out) ** 2
        P = I.flatten(1).sum(1) + 1e-12
        ep = (I.unsqueeze(1) * pos[host]).flatten(start_dim=2).sum(dim=2)
        en = (I.unsqueeze(1) * neg[host]).flatten(start_dim=2).sum(dim=2)
        s = (ep - en) / P.unsqueeze(1)
        shat.append(s.cpu().numpy())
        s_max.append(s.max(dim=1).values.cpu().numpy())
        core.append(((I * spots).flatten(1).sum(1) / P).cpu().numpy())
    return (np.concatenate(shat), np.concatenate(s_max), np.concatenate(core))


def sample_readout(full, host, ds_test, src, n_per_class, pos, neg, spots):
    """Draw a class-balanced test subset of `src`, place it in `host`'s window,
    and return the ten readouts, their maximum, the own-spot power fraction and
    the sample indices."""
    idx = _get_balanced_indices(ds_test[src], n_per_class=n_per_class, seed=0)
    cv = [t2.test_canvas(host, fd._img(ds_test[src], j)) for j in idx]
    shat, s_max, core = readout_batched(full, cv, host, pos, neg, spots)
    return shat, s_max, core, idx


def evaluate(name, full, ds_test, pos, neg, spots, n_per_class, taus_fixed):
    per = {}
    for host in REGIONS:
        s_own, m_own, c_own, idx = sample_readout(
            full, host, ds_test, host, n_per_class, pos, neg, spots[host])
        lab = np.array([int(ds_test[host][j][1]) % NC for j in idx])
        acc = float((s_own.argmax(1) == lab).mean())

        off = {}
        for src in REGIONS:
            if src == host:
                continue
            _, m_q, c_q, _ = sample_readout(
                full, host, ds_test, src, n_per_class, pos, neg, spots[host])
            off[NAME[src]] = (m_q, c_q)
        m_off = np.concatenate([off[k][0] for k in off])
        c_off = float(np.mean([off[k][1].mean() for k in off]))

        blanks = [fd.blank_canvas(host).unsqueeze(0) for _ in range(N_EMPTY)]
        _, m_emp, c_emp = readout_batched(full, blanks, host, pos, neg,
                                          spots[host])

        tau_empty = float(np.percentile(m_emp, 99))
        grid = np.quantile(np.concatenate([m_own, m_off]),
                           np.linspace(0.0, 1.0, 401))
        best, tau_yd = -2.0, float(grid[0])
        for x in grid:
            j = float((m_own >= x).mean()) - float((m_off >= x).mean())
            if j > best:
                best, tau_yd = j, float(x)
        det = m_own >= tau_yd
        row = {
            "n_own": int(m_own.size), "n_off": int(m_off.size),
            "n_empty": int(m_emp.size),
            "acc": acc,
            "s_own_mean": float(m_own.mean()),
            "s_own_p05": float(np.percentile(m_own, 5)),
            "s_off_mean": float(m_off.mean()),
            "s_off_p99": float(np.percentile(m_off, 99)),
            "s_empty_mean": float(m_emp.mean()),
            "s_empty_p99": tau_empty,
            "core_own": float(c_own.mean()), "core_off": c_off,
            "core_empty": float(c_emp.mean()),
            "tau_empty": tau_empty,
            "far_off_at_tau_empty": float((m_off >= tau_empty).mean()),
            "far_empty_at_tau_empty": float((m_emp >= tau_empty).mean()),
            "tau_youden": tau_yd,
            "tpr_at_tau_youden": float(det.mean()),
            "tpr_right_class_at_tau_youden": (
                float((s_own[det].argmax(1) == lab[det]).mean())
                if det.any() else float("nan")),
            "far_off_at_tau_youden": float((m_off >= tau_yd).mean()),
            "far_empty_at_tau_youden": float((m_emp >= tau_yd).mean()),
            "by_source": {
                k: {"s_mean": float(off[k][0].mean()),
                    "s_p99": float(np.percentile(off[k][0], 99)),
                    "far_at_tau_empty": float((off[k][0] >= tau_empty).mean()),
                    "far_at_tau_youden": float((off[k][0] >= tau_yd).mean()),
                    "far_at_fixed": [float((off[k][0] >= x).mean())
                                     for x in taus_fixed]}
                for k in off},
            "tpr_at_fixed": [float((m_own >= x).mean()) for x in taus_fixed],
            "far_off_at_fixed": [float((m_off >= x).mean())
                                 for x in taus_fixed],
            "sweep": sweep(m_own, m_off, m_emp),
        }
        per[host] = row
    return {"device": name, "n_per_class": n_per_class,
            "taus_fixed": taus_fixed,
            "mean_acc": float(np.mean([per[r]["acc"] for r in REGIONS])),
            "channels": per}


def sweep(m_own, m_off, m_emp):
    grid = np.linspace(0.0, float(np.percentile(m_own, 99.9)), 201)
    return {"tau": grid.tolist(),
            "tpr": [float((m_own >= x).mean()) for x in grid],
            "far_off": [float((m_off >= x).mean()) for x in grid],
            "far_empty": [float((m_emp >= x).mean()) for x in grid]}


def print_report(o):
    print("\n==== %s ====" % o["device"])
    print("accuracy, balanced %d per class from the official test split"
          % o["n_per_class"])
    for r in REGIONS:
        d = o["channels"][r]
        print("  %-7s acc=%6.2f%%  (n=%d)  s_own %.4f p05 %.4f | "
              "s_off %.4f p99 %.4f | s_empty %.4f p99 %.4f | own spots "
              "%.2f%% off %.2f%% empty %.2f%%"
              % (NAME[r], 100 * d["acc"], d["n_own"], d["s_own_mean"],
                 d["s_own_p05"], d["s_off_mean"], d["s_off_p99"],
                 d["s_empty_mean"], d["s_empty_p99"], 100 * d["core_own"],
                 100 * d["core_off"], 100 * d["core_empty"]))
    print("  mean accuracy %.2f%%" % (100 * o["mean_acc"]))
    print("false alarm, thresholds per channel")
    for r in REGIONS:
        d = o["channels"][r]
        print("  %-7s tau_empty=%.4f: FAR_off=%.3f FAR_empty=%.3f | "
              "tau_youden=%.4f: TPR=%.3f class-correct=%.3f FAR_off=%.3f "
              "FAR_empty=%.3f"
              % (NAME[r], d["tau_empty"], d["far_off_at_tau_empty"],
                 d["far_empty_at_tau_empty"], d["tau_youden"],
                 d["tpr_at_tau_youden"], d["tpr_right_class_at_tau_youden"],
                 d["far_off_at_tau_youden"], d["far_empty_at_tau_youden"]))
        for k, v in d["by_source"].items():
            print("        off-target from %-7s s_mean %.4f s_p99 %.4f "
                  "FAR@tau_empty %.3f FAR@tau_youden %.3f"
                  % (k, v["s_mean"], v["s_p99"], v["far_at_tau_empty"],
                     v["far_at_tau_youden"]))
    fixed = o["taus_fixed"]
    print("fixed thresholds %s" % fixed)
    for r in REGIONS:
        d = o["channels"][r]
        print("  %-7s TPR %s  FAR_off %s"
              % (NAME[r], ["%.3f" % x for x in d["tpr_at_fixed"]],
                 ["%.3f" % x for x in d["far_off_at_fixed"]]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True,
                    help="name=path pairs; give several seeds of one recipe "
                         "to get a spread")
    ap.add_argument("--n-per-class", type=int, default=100)
    ap.add_argument("--tau", type=float, nargs="+", default=[0.2])
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--sub-tag", default="bg1e3")
    ap.add_argument("--out", default="accuracy_far")
    a = ap.parse_args()

    t2.SUB_DIR = os.path.join(HERE, "checkpoints")
    pos, neg = t2.build_masks300()
    spots = {r: losses.spot_ring(pos[r], neg[r]) for r in REGIONS}
    ds_test = data.load_datasets(train=False)
    print("official test split sizes: %s"
          % {NAME[r]: len(ds_test[r]) for r in REGIONS}, flush=True)
    all_out = []
    for spec in a.ckpt:
        nm, path = spec.split("=", 1)
        full = t2.build_init("merged", a.layers, a.sub_tag).to(DEV)
        full.load_state_dict(torch.load(path, map_location=DEV,
                                        weights_only=False))
        full.eval()
        o = evaluate(nm, full, ds_test, pos, neg, spots, a.n_per_class, a.tau)
        print_report(o)
        all_out.append(o)
    if len(all_out) > 1:
        accs = [o["mean_acc"] for o in all_out]
        print("\nacross %d devices: mean accuracy %.2f%% +- %.2f"
              % (len(accs), 100 * float(np.mean(accs)),
                 100 * float(np.std(accs))))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    p = os.path.join(HERE, "results", "%s.json" % a.out)
    json.dump(all_out, open(p, "w"), indent=1)
    print("saved ->", p)

if __name__ == "__main__":
    main()
