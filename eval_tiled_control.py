"""Equal-budget non-overlapping control for R1.1 and R2.1.

Four disjoint 150x150 squares inside the same 300x300 canvas, same three
layers, same 8 mm spacing, same datasets, same split and same differential
readout as the shared-aperture device.  The only difference is geometry: no
pixel is used by two tasks, so nothing is shared.

Arms that can be evaluated:
  ce_paste  the four squares trained on the classification term only, pasted
            side by side, never fine-tuned jointly
  bg1e3     the same pasting, then fine-tuned with the stage-2 recipe that
            carries the background term but no anti-crosstalk term
  lxt       the same pasting, fine-tuned with the stage-2 recipe that includes
            the anti-crosstalk term

The reported quantities are the same ones eval_accuracy_far.py
reports for the shared device: per-channel accuracy, the distribution of the
largest normalised readout under own, off-target and empty illumination, false
alarm rates at three threshold calibrations, and the share of output power
falling on the host channel's own detector spots.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))

from d2nn import config, data
from d2nn.readout import _get_balanced_indices
import control_tiled_lib as t15

DEV = config.DEVICE
REG = list(config.REGIONS)
NAME = {"NW": "MNIST", "NE": "Fashion", "SW": "EMNIST", "SE": "Shapes"}
NC = config.N_CLASSES
BATCH = 128
N_EMPTY = 500


@torch.no_grad()
def distributions(net, cv, host, pos, neg, ring):
    s_max, core, pred = [], [], []
    for i in range(0, cv.shape[0], BATCH):
        I = torch.abs(net(cv[i:i + BATCH])) ** 2
        P = I.flatten(1).sum(1) + 1e-12
        ep = (I.unsqueeze(1) * pos[host]).flatten(start_dim=2).sum(dim=2)
        en = (I.unsqueeze(1) * neg[host]).flatten(start_dim=2).sum(dim=2)
        s = (ep - en) / P.unsqueeze(1)
        s_max.append(s.max(dim=1).values.cpu().numpy())
        core.append(((I * ring[host]).flatten(1).sum(1) / P).cpu().numpy())
        pred.append(s.argmax(1).cpu().numpy())
    return np.concatenate(s_max), np.concatenate(core), np.concatenate(pred)


def imgs_of(ds, idx):
    out = []
    for j in idx:
        im = ds[int(j)][0]
        if im.dim() == 3:
            im = im.squeeze(0)
        out.append(im)
    return torch.stack(out)


def evaluate(arm, net, ds_test, pos, neg, ring, n_per_class, taus_fixed):
    res = {}
    for host in REG:
        idx = _get_balanced_indices(ds_test[host], n_per_class=n_per_class,
                                    seed=0)
        lab = np.array([int(ds_test[host][j][1]) % NC for j in idx])
        m_own, c_own, p_own = distributions(
            net, t15.full_input(imgs_of(ds_test[host], idx), host), host,
            pos, neg, ring)
        acc = float((p_own == lab).mean())

        m_off, c_off, by_src = [], [], {}
        for src in REG:
            if src == host:
                continue
            sidx = _get_balanced_indices(ds_test[src], n_per_class=n_per_class,
                                         seed=0)
            m_q, c_q, _ = distributions(
                net, t15.full_input(imgs_of(ds_test[src], sidx), src, host),
                host, pos, neg, ring)
            by_src[NAME[src]] = {"s_mean": float(m_q.mean()),
                                 "s_p99": float(np.percentile(m_q, 99))}
            m_off.append(m_q)
            c_off.append(c_q)
        m_off = np.concatenate(m_off)
        c_off = float(np.mean(c_off))

        emp = torch.cat([t15.empty_input(host) for _ in range(N_EMPTY)])
        m_emp, c_emp, _ = distributions(net, emp, host, pos, neg, ring)

        tau_empty = float(np.percentile(m_emp, 99))
        grid = np.quantile(np.concatenate([m_own, m_off]),
                           np.linspace(0, 1, 401))
        best, tau_yd = -2.0, float(grid[0])
        for x in grid:
            jj = float((m_own >= x).mean()) - float((m_off >= x).mean())
            if jj > best:
                best, tau_yd = jj, float(x)
        det = m_own >= tau_yd
        res[host] = {
            "n_own": int(m_own.size), "n_off": int(m_off.size),
            "n_empty": int(m_emp.size), "acc": acc,
            "s_own_mean": float(m_own.mean()),
            "s_own_p05": float(np.percentile(m_own, 5)),
            "s_off_mean": float(m_off.mean()),
            "s_off_p99": float(np.percentile(m_off, 99)),
            "s_empty_mean": float(m_emp.mean()),
            "s_empty_p99": tau_empty,
            "core_own": float(c_own.mean()), "core_off": c_off,
            "core_empty": float(c_emp.mean()),
            "confinement_db": float(10.0 * np.log10((c_own.mean() + 1e-12)
                                                    / (c_off + 1e-12))),
            "readout_db": float(10.0 * np.log10((m_own.mean() + 1e-12)
                                                / (m_off.mean() + 1e-12))),
            "tau_youden": tau_yd,
            "tpr_at_tau_youden": float(det.mean()),
            "far_off_at_tau_youden": float((m_off >= tau_yd).mean()),
            "far_off_at_tau_empty": float((m_off >= tau_empty).mean()),
            "far_empty_at_tau_empty": float((m_emp >= tau_empty).mean()),
            "tpr_at_fixed": [float((m_own >= x).mean()) for x in taus_fixed],
            "far_off_at_fixed": [float((m_off >= x).mean())
                                 for x in taus_fixed],
            "by_source": by_src,
        }
    return {"arm": arm, "n_per_class": n_per_class, "taus_fixed": taus_fixed,
            "mean_acc": float(np.mean([res[r]["acc"] for r in REG])),
            "mean_confinement_db": float(np.mean([res[r]["confinement_db"]
                                                  for r in REG])),
            "channels": res}


def print_report(o):
    print("\n==== %s ====" % o["arm"])
    for r in REG:
        d = o["channels"][r]
        print("  %-7s acc=%6.2f%%  s_own %.4f p05 %.4f | s_off %.4f p99 %.4f "
              "| s_empty %.4f | spots own/off/empty %.2f/%.2f/%.2f%% | "
              "conf %+0.2f dB | readout %+0.2f dB"
              % (NAME[r], 100 * d["acc"], d["s_own_mean"], d["s_own_p05"],
                 d["s_off_mean"], d["s_off_p99"], d["s_empty_mean"],
                 100 * d["core_own"], 100 * d["core_off"],
                 100 * d["core_empty"], d["confinement_db"], d["readout_db"]))
        print("        tau_empty=%.4f FAR_off=%.3f FAR_empty=%.3f | "
              "tau_youden=%.4f TPR=%.3f FAR_off=%.3f | fixed %s TPR %s "
              "FAR_off %s"
              % (d["s_empty_p99"], d["far_off_at_tau_empty"],
                 d["far_empty_at_tau_empty"], d["tau_youden"],
                 d["tpr_at_tau_youden"], d["far_off_at_tau_youden"],
                 o["taus_fixed"], ["%.3f" % x for x in d["tpr_at_fixed"]],
                 ["%.3f" % x for x in d["far_off_at_fixed"]]))
    print("  mean accuracy %.2f%%, mean confinement %.2f dB"
          % (100 * o["mean_acc"], o["mean_confinement_db"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", nargs="+", required=True, help="name=path pairs")
    ap.add_argument("--n-per-class", type=int, default=100)
    ap.add_argument("--tau", type=float, nargs="+", default=[0.2, 0.1, 0.05])
    ap.add_argument("--out", default="tiled_control")
    a = ap.parse_args()

    ring, pos, neg = t15.build_masks()
    ds_test = data.load_datasets(train=False)
    out_all = []
    for spec in a.arm:
        nm, path = spec.split("=", 1)
        net = t15.TiledFullNet().to(DEV)
        net.load_state_dict(torch.load(path, map_location=DEV,
                                       weights_only=False))
        net.eval()
        o = evaluate(nm, net, ds_test, pos, neg, ring, a.n_per_class, a.tau)
        print_report(o)
        out_all.append(o)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    p = os.path.join(HERE, "results", "%s.json" % a.out)
    json.dump(out_all, open(p, "w"), indent=1)
    print("saved ->", p)


if __name__ == "__main__":
    main()

