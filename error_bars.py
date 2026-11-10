"""Confidence intervals for the rates reported by eval_accuracy_far.py.

Two different sources of spread are kept apart.
  Sampling spread: the number of test images behind a rate is finite, so the
  95% Wilson interval of a binomial proportion is given for every accuracy,
  detection rate and false-alarm rate.
  Training spread: when several seeds of the same recipe are present, the mean
  and the standard deviation across seeds are given as well.
"""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

REGIONS = ["NW", "NE", "SW", "SE"]
NAME = {"NW": "MNIST", "NE": "Fashion", "SW": "EMNIST", "SE": "Shapes"}


def wilson(p, n, z=1.96):
    if n <= 0:
        return float("nan"), float("nan")
    den = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, ctr - half), min(1.0, ctr + half)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", nargs="+", required=True)
    a = ap.parse_args()

    entries = []
    for path in a.json:
        d = json.load(open(path))
        entries += d if isinstance(d, list) else [d]

    for ent in entries:
        label = ent.get("device") or ent.get("arm") or os.path.basename(
            a.json[0])
        print("\n==== %s ====" % label)
        accs, means = [], []
        for r in REGIONS:
            d = ent["channels"][r]
            lo, hi = wilson(d["acc"], d["n_own"])
            tlo, thi = wilson(d["tpr_at_tau_youden"], d["n_own"])
            flo, fhi = wilson(d["far_off_at_tau_youden"], d["n_off"])
            elo, ehi = wilson(d["far_empty_at_tau_youden"], d["n_empty"])
            print("  %-7s acc %5.2f%% [%5.2f, %5.2f]  n=%d"
                  % (NAME[r], 100 * d["acc"], 100 * lo, 100 * hi, d["n_own"]))
            print("          门限 %.4f  检出 %5.1f%% [%5.1f, %5.1f]  "
                  "放错虚警 %5.2f%% [%5.2f, %5.2f]  空窗虚警 %5.2f%% "
                  "[%5.2f, %5.2f]"
                  % (d["tau_youden"], 100 * d["tpr_at_tau_youden"],
                     100 * tlo, 100 * thi, 100 * d["far_off_at_tau_youden"],
                     100 * flo, 100 * fhi, 100 * d["far_empty_at_tau_youden"],
                     100 * elo, 100 * ehi))
            accs.append(d["acc"])
        print("  mean accuracy over channels %.2f%%" % (100 * np.mean(accs)))

    by_dev = {}
    for ent in entries:
        nm = re_seed(ent.get("device") or ent.get("arm") or "")
        by_dev.setdefault(nm, []).append(ent["mean_acc"])
    print("\n跨种子的波动（同一配方）")
    for nm, v in by_dev.items():
        if len(v) > 1:
            print("  %-22s 平均 %.2f%%, 标准差 %.2f, 样本数 %d"
                  % (nm, 100 * np.mean(v), 100 * np.std(v, ddof=1), len(v)))


def re_seed(name):
    base = name.split("_s")[0]
    return base if base else name


if __name__ == "__main__":
    main()
