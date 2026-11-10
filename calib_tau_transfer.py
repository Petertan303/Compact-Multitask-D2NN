"""Can the decision threshold tau be calibrated on the training split?

Readout maxima are collected for three illuminations (own, off-target, empty)
from two pools and three input conventions:

  test    official test split, test convention (the patch is divided by its own
          peak intensity)
  trainA  official training split, test convention.  The only difference from
          `test` is that the device has seen these images during training, so
          this pair measures the memorisation part of the transfer.
  trainB  official training split, training convention (input energy normalised
          to SUB_SIZE^2 before the 4x upsampling), i.e. exactly the pipeline the
          weights were fitted under.

Empty windows (noise only) are identical for the three conventions.  tau_empty
is the 99th percentile of the empty-window readout maximum, tau_youden
maximises TPR minus FAR against off-target inputs.  Every threshold is then
applied to the test pool, because that is what would be reported.
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
import eval_accuracy_far as eaf
from d2nn import config, data, losses
from d2nn.readout import _get_balanced_indices

DEV = config.DEVICE
REGIONS = list(config.REGIONS)
NAME = {"NW": "MNIST", "NE": "Fashion", "SW": "EMNIST", "SE": "Shapes"}
NC = config.N_CLASSES


def youden(m_own, m_off):
    grid = np.quantile(np.concatenate([m_own, m_off]), np.linspace(0, 1, 401))
    best, th = -2.0, float(grid[0])
    for x in grid:
        j = float((m_own >= x).mean()) - float((m_off >= x).mean())
        if j > best:
            best, th = j, float(x)
    return th


@torch.no_grad()
def train_conv_readout(full, host, ds_tr, src, n_per_class, pos, neg, spots):
    idx = _get_balanced_indices(ds_tr[src], n_per_class=n_per_class, seed=0)
    rows = []
    for i in range(0, len(idx), 128):
        sel = idx[i:i + 128]
        imgs = []
        for j in sel:
            im = fd._img(ds_tr[src], j).unsqueeze(0)
            im = losses.normalize_input_to_canvas_energy(
                im, target_l2_sq=config.CANVAS_L2_SQ)[0]
            imgs.append(im)
        imgs = torch.stack(imgs).to(DEV)
        rows.append(t2.train_canvas_batch(host, imgs))
    _, m, _ = eaf.readout_batched(full, [torch.cat(rows)], host, pos, neg,
                                  spots)
    return m, len(idx)


def pool_readouts(full, host, pool, n_per_class, pos, neg, spots, empty,
                  convention):
    own, off = [], []
    if convention == "test":
        for src in REGIONS:
            _, m, _, _ = eaf.sample_readout(full, host, pool, src,
                                            n_per_class, pos, neg, spots[host])
            (off if src != host else own).append(m)
    else:
        for src in REGIONS:
            m, _ = train_conv_readout(full, host, pool, src, n_per_class, pos,
                                      neg, spots[host])
            (off if src != host else own).append(m)
    return np.concatenate(own), np.concatenate(off)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--n-per-class", type=int, default=50)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--sub-tag", default="bg1e3")
    ap.add_argument("--out", default="tau_transfer")
    a = ap.parse_args()

    t2.SUB_DIR = os.path.join(HERE, "checkpoints")
    pos, neg = t2.build_masks300()
    spots = {r: losses.spot_ring(pos[r], neg[r]) for r in REGIONS}
    ds_test = data.load_datasets(train=False)
    ds_train = data.load_datasets(train=True)
    print("pool sizes: test %s, train %s"
          % ({NAME[r]: len(ds_test[r]) for r in REGIONS},
             {NAME[r]: len(ds_train[r]) for r in REGIONS}))

    for spec in a.ckpt:
        nm, path = spec.split("=", 1)
        full = t2.build_init("merged", a.layers, a.sub_tag).to(DEV)
        full.load_state_dict(torch.load(path, map_location=DEV,
                                        weights_only=False))
        full.eval()
        print("\n==== %s ====" % nm)
        res = {}
        for host in REGIONS:
            blanks = [fd.blank_canvas(host).unsqueeze(0)
                      for _ in range(eaf.N_EMPTY)]
            _, m_emp, _ = eaf.readout_batched(full, blanks, host, pos, neg,
                                              spots[host])
            te_own, te_off = pool_readouts(full, host, ds_test, a.n_per_class,
                                           pos, neg, spots, m_emp, "test")
            tA_own, tA_off = pool_readouts(full, host, ds_train, a.n_per_class,
                                           pos, neg, spots, m_emp, "test")
            tB_own, tB_off = pool_readouts(full, host, ds_train, a.n_per_class,
                                           pos, neg, spots, m_emp, "train")
            calib = {
                "test": (youden(te_own, te_off), np.percentile(m_emp, 99)),
                "trainA": (youden(tA_own, tA_off), np.percentile(m_emp, 99)),
                "trainB": (youden(tB_own, tB_off), np.percentile(m_emp, 99)),
            }
            print("  %-7s readout maxima: test own %.4f off %.4f | "
                  "trainA own %.4f off %.4f | trainB own %.4f off %.4f"
                  % (NAME[host], te_own.mean(), te_off.mean(), tA_own.mean(),
                     tA_off.mean(), tB_own.mean(), tB_off.mean()))
            row = {}
            for tag, (tau_yd, tau_emp) in calib.items():
                row[tag] = {
                    "tau_youden": float(tau_yd),
                    "tpr_on_test": float((te_own >= tau_yd).mean()),
                    "far_off_on_test": float((te_off >= tau_yd).mean()),
                    "tau_empty": float(tau_emp),
                    "far_off_at_tau_empty": float((te_off >= tau_emp).mean()),
                }
                print("      %-7s tau_youden=%.4f -> TPR_test %.3f "
                      "FAR_off_test %.3f   (tau_empty=%.4f FAR_off %.3f)"
                      % (tag, tau_yd, row[tag]["tpr_on_test"],
                         row[tag]["far_off_on_test"], tau_emp,
                         row[tag]["far_off_at_tau_empty"]))
            res[host] = row
        p = os.path.join(HERE, "results", "%s_%s.json" % (a.out, nm))
        json.dump(res, open(p, "w"), indent=1)
        print("  saved ->", p)


if __name__ == "__main__":
    main()
