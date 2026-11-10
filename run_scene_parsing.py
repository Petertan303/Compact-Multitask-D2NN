"""Sliding-window scene parsing accuracy, with confidence intervals.

Sweeps the fixed device across a scene in steps of --step pixels (150 px is
half a window, 300 px a whole window) and reports three statistics, the same
three the manuscript quotes:

  home     reads of a quadrant that contains an object of that quadrant's own
           family, at the position where that object is centred
  offset   reads of a quadrant that contains only objects of other families;
           a correct read here is a false claim, so the good value is zero
  merged   each object counted once, taken from the window in which its own
           family's quadrant gave the strongest differential signal

Empty quadrants are reported with their signal distribution so a presence
threshold can be placed without guessing.  Wilson 95% intervals are given
because the number of reads is in the hundreds to thousands.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import scene_scan as sc
import train_stage2 as t2
from d2nn import config

REGIONS = sc.REGIONS
DEV = config.DEVICE


def wilson(p, n, z=1.96):
    if n <= 0:
        return float("nan"), float("nan")
    den = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * max(0.0, ctr - hw), 100 * min(1.0, ctr + hw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True,
                    help="name=path pairs of shared-aperture devices")
    ap.add_argument("--grid", type=int, default=8,
                    help="scene cells per axis")
    ap.add_argument("--step", type=int, default=150)
    ap.add_argument("--n-full", type=int, default=None)
    ap.add_argument("--n-partial", type=int, default=None)
    ap.add_argument("--images-per-class", type=int, default=10)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--sub-tag", default="bg1e3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="scene_parsing")
    a = ap.parse_args()

    n_full = a.n_full if a.n_full is not None else a.grid * a.grid - 6
    n_partial = (a.n_partial if a.n_partial is not None
                 else max(0, a.grid * a.grid - n_full - 2))

    pos, neg = t2.build_masks300()
    sc.prepare_masks(pos, neg)
    pool = sc.build_image_pool(a.seed, a.images_per_class)
    world, cells = sc.build_scene(a.seed, a.grid, n_full, n_partial, pool)
    boxes = sc.object_boxes(cells)
    pad = sc.prepare(world, a.step)
    positions = sc.window_positions(world.shape[0], a.step)
    print("scene: grid %d, cells %d (full %d, partial %d, empty %d), "
          "objects %d, world %d px, step %d, windows %d"
          % (a.grid, a.grid * a.grid, n_full, n_partial,
             a.grid * a.grid - n_full - n_partial, len(boxes),
             world.shape[0], a.step, len(positions)), flush=True)

    results = {}
    for spec in a.ckpt:
        nm, path = spec.split("=", 1)
        net = t2.FullNetNL(a.layers).to(DEV)
        net.load_state_dict(torch.load(path, map_location=DEV,
                                       weights_only=False))
        net.eval()
        n_home = n_home_ok = n_off = n_off_ok = n_empty = 0
        empty_sig = []
        obj_reads = {}
        for wy, wx in positions:
            readout, info = sc.scan_window(net, pad, wy, wx, boxes, pos, neg)
            for region in REGIONS:
                ro = readout[region]
                full = info[region]["full"]
                if not full and not info[region]["part"]:
                    n_empty += 1
                    empty_sig.append(ro["smax"])
                    continue
                if not full:
                    continue
                ok = any(b["region"] == region and ro["pred"] == b["label"]
                         for b in full)
                if any(b["region"] == region for b in full):
                    n_home += 1
                    n_home_ok += int(ok)
                else:
                    n_off += 1
                    n_off_ok += int(ok)
                for b in full:
                    key = (b["center"][0], b["center"][1], b["region"],
                           b["label"])
                    obj_reads.setdefault(key, []).append(
                        {"smax": ro["smax"],
                         "ok": b["region"] == region
                         and ro["pred"] == b["label"]})
        n_obj = n_obj_ok = 0
        n_obj_any = 0
        per_family = {r: [0, 0] for r in REGIONS}
        per_family_any = {r: [0, 0] for r in REGIONS}
        for key, reads in obj_reads.items():
            n_obj += 1
            best = max(reads, key=lambda x: x["smax"])
            n_obj_ok += int(best["ok"])
            per_family[key[2]][0] += int(best["ok"])
            per_family[key[2]][1] += 1
            any_ok = any(x["ok"] for x in reads)
            n_obj_any += int(any_ok)
            per_family_any[key[2]][0] += int(any_ok)
            per_family_any[key[2]][1] += 1
        rows = {}
        for label, (ok, n) in (("home", (n_home_ok, n_home)),
                               ("offset", (n_off_ok, n_off)),
                               ("merged_best_signal", (n_obj_ok, n_obj)),
                               ("merged_any_window", (n_obj_any, n_obj))):
            lo, hi = wilson(ok / max(n, 1), n)
            rows[label] = {"ok": ok, "n": n, "acc": ok / max(n, 1),
                           "ci95": [lo, hi]}
        rows["empty_quadrants"] = {"n": n_empty,
                                   "smax_median": float(np.median(empty_sig))
                                   if empty_sig else 0.0,
                                   "smax_max": float(np.max(empty_sig))
                                   if empty_sig else 0.0}
        rows["merged_per_family"] = {"%s" % r: per_family[r] for r in REGIONS}
        rows["merged_per_family_any"] = {"%s" % r: per_family_any[r]
                                         for r in REGIONS}
        results[nm] = rows
        print("\n==== %s ====" % nm)
        for label, text in (("home", "object centred in its own quadrant"),
                             ("offset", "foreign object claimed by the host"),
                             ("merged_any_window",
                              "object read right in some window"),
                             ("merged_best_signal",
                              "strongest-signal window was the right one")):
            d = rows[label]
            print("  %-36s %5d/%5d = %5.2f%%   95%% CI [%.2f, %.2f]"
                  % (text, d["ok"], d["n"], 100 * d["acc"],
                     d["ci95"][0], d["ci95"][1]))
        print("  empty quadrants %d, signal median %.4f, max %.4f"
              % (rows["empty_quadrants"]["n"],
                 rows["empty_quadrants"]["smax_median"],
                 rows["empty_quadrants"]["smax_max"]))
        print("  merged per family: " + "  ".join(
            "%s %d/%d" % (r, v[0], v[1])
            for r, v in rows["merged_per_family"].items()))
        print("  merged per family, any window: " + "  ".join(
            "%s %d/%d" % (r, v[0], v[1])
            for r, v in rows["merged_per_family_any"].items()))

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    p = os.path.join(HERE, "results", "%s.json" % a.out)
    json.dump(results, open(p, "w"), indent=1)
    print("\nsaved ->", p)


if __name__ == "__main__":
    main()
