import os
import json
import argparse

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))

from d2nn import config
from d2nn import data
from d2nn.stage_utils import pad_input_to_canvas
from d2nn.readout import _get_balanced_indices
import eval_accuracy as ev2

DEV = config.DEVICE
REG = list(config.REGIONS)
NC = config.N_CLASSES
NOISE = 0.01
TAUS = np.round(np.arange(0.0, 0.601, 0.02), 3)
RES = os.path.join(HERE, "results")

def input_canvas(img, slot, geom):
    cx, cy = geom["CENTS"][slot]
    ox, oy = geom["OFF"][slot]
    sub = geom["SUB"]
    if img.dim() == 3:
        img = img.squeeze(0)
    sc = pad_input_to_canvas(img, sub, upsample=4,
                             centroid=(cx, cy), noise_std=NOISE)
    if sc.max() > 0:
        sc = sc / sc.max()
    big = torch.zeros(config.N, config.N, device=DEV)
    big[oy:oy + sub, ox:ox + sub] = sc
    return big

def empty_canvas(slot, geom):
    cx, cy = geom["CENTS"][slot]
    ox, oy = geom["OFF"][slot]
    sub = geom["SUB"]
    sc = torch.zeros(sub, sub)
    sc = pad_input_to_canvas(sc, sub, upsample=1,
                             centroid=(cx, cy), noise_std=NOISE)
    big = torch.zeros(config.N, config.N, device=DEV)
    big[oy:oy + sub, ox:ox + sub] = sc
    return big

def per_slot_scores(I, p_out, slot, ring300, pos300, neg300):
    ep = torch.stack([(I * pos300[slot][c]).flatten(1).sum(1)
                      for c in range(NC)], dim=1)
    en = torch.stack([(I * neg300[slot][c]).flatten(1).sum(1)
                      for c in range(NC)], dim=1)
    raw = ep - en
    shat = raw / (p_out[:, None] + 1e-12)
    pq = (I * ring300[slot]).flatten(1).sum(1)
    shat_q = raw / (pq[:, None] + 1e-12)
    return raw, shat, shat_q

def stats_of(v):
    v = np.asarray(v, dtype=np.float64)
    return {"n": int(v.size), "mean": float(v.mean()),
            "median": float(np.median(v)), "p90": float(np.percentile(v, 90)),
            "p99": float(np.percentile(v, 99)), "max": float(v.max())}

STATS = ("max_shat", "margin", "ringfrac")

def evaluate(tag, path, ds_test, ring300, pos300, neg300, n_per_class, batch,
             geom):
    full = ev2.load_full(path)
    if full is None:
        print(f"[{tag}] MISSING {path}")
        return None, None
    out = {"ckpt": path, "n_per_class": n_per_class,
           "matrix_max_shat": {}, "matrix_far": {}, "matrix_tpr_detail": {},
           "matrix_max_shat_quadnorm": {}, "matrix_acc_cond_detected": {},
           "matrix_margin": {}, "matrix_ringfrac": {},
           "far_margin": {}, "far_ringfrac": {},
           "background": {}, "sweep": {}}
    dists = {}
    idx_all = {r: _get_balanced_indices(ds_test[r], n_per_class=n_per_class)
               for r in REG}
    with torch.no_grad():
        for A in REG:
            for B in REG:
                mx, mxq, mg, rf, correct, tot = [], [], [], [], 0, 0
                det = det_ok = 0
                for s in range(0, len(idx_all[A]), batch):
                    ids = idx_all[A][s:s + batch]
                    canv = torch.stack([input_canvas(ds_test[A][i][0], B, geom)
                                        for i in ids])
                    I = torch.abs(full(canv.to(torch.complex64))) ** 2
                    p_out = I.flatten(1).sum(1)
                    raw, shat, shat_q = per_slot_scores(I, p_out, B, ring300,
                                                        pos300, neg300)
                    mx += shat.max(1).values.tolist()
                    mxq += shat_q.max(1).values.tolist()
                    top2 = shat.topk(2, dim=1).values
                    mg += (top2[:, 0] - top2[:, 1]).tolist()
                    rf += ((I * ring300[B]).flatten(1).sum(1)
                          / (p_out + 1e-12)).tolist()
                    pred = shat.argmax(1).tolist()
                    labs = [int(ds_test[A][i][1]) % NC for i in ids]
                    correct += sum(int(p == l) for p, l in zip(pred, labs))
                    hit = (shat.max(1).values >= 0.2)
                    det += int(hit.sum())
                    det_ok += sum(int(hit[j]) and int(pred[j] == labs[j])
                                  for j in range(len(ids)))
                    tot += len(ids)
                mx = np.array(mx)
                key = A + "->" + B
                arrs = {"max_shat": mx, "margin": np.array(mg),
                        "ringfrac": np.array(rf), "max_shat_quadnorm":
                        np.array(mxq)}
                for st, arr in arrs.items():
                    dists[f"{tag}::{st}::{key}"] = arr
                    if st in STATS:
                        out["matrix_" + st][key] = stats_of(arr)
                        out_key = ("matrix_far" if st == "max_shat"
                                   else "far_" + st)
                        out[out_key][key] = {
                            float(t): float((arr >= t).mean()) for t in TAUS}
                out["matrix_tpr_detail"][key] = {
                    "acc_unconditional": correct / max(tot, 1),
                    "TPR_at_0.2": float((mx >= 0.2).mean()),
                    "frac_shat_ge_0.2": float((mx >= 0.2).mean()),
                }
                out["matrix_acc_cond_detected"][key] = (det_ok / det
                                                        if det else None)
                print(f"  {tag} {key:8s} max_shat mean={mx.mean():.4f} "
                      f"p99={np.percentile(mx, 99):.4f} max={mx.max():.4f} "
                      f"FAR@0.2={float((mx >= 0.2).mean())*100:.2f}%",
                      flush=True)

        for B in REG:
            canv = torch.stack([empty_canvas(B, geom) for _ in range(64)])
            I = torch.abs(full(canv.to(torch.complex64))) ** 2
            p_out = I.flatten(1).sum(1)
            raw, shat, shat_q = per_slot_scores(I, p_out, B, ring300, pos300,
                                                neg300)
            out["background"][B] = stats_of(shat.max(1).values.tolist())
            print(f"  {tag} empty->{B}  max_shat mean="
                  f"{out['background'][B]['mean']:.4f} "
                  f"max={out['background'][B]['max']:.4f}", flush=True)

    tpr = {float(t): float(np.mean([out["matrix_far"][r + "->" + r][t]
                                    for r in REG])) for t in TAUS}
    far = {float(t): float(np.mean([out["matrix_far"][a + "->" + b][t]
                                    for a in REG for b in REG if a != b]))
           for t in TAUS}
    out["sweep"] = {"tau": [float(t) for t in TAUS],
                    "mean_TPR": [tpr[float(t)] for t in TAUS],
                    "mean_FAR": [far[float(t)] for t in TAUS]}
    out["at_tau_0.2"] = {
        "mean_TPR": tpr[0.2], "mean_FAR": far[0.2],
        "max_off_target_shat": float(max(out["matrix_max_shat"][a + "->" + b]
                                         ["max"]
                                         for a in REG for b in REG if a != b)),
        "p99_off_target_shat": float(np.mean(
            [out["matrix_max_shat"][a + "->" + b]["p99"]
             for a in REG for b in REG if a != b])),
        "tau_for_FAR_1pct": next((float(t) for t in TAUS
                                  if far[float(t)] <= 0.01), None),
        "tau_for_FAR_0pct": next((float(t) for t in TAUS
                                  if far[float(t)] == 0.0), None),
    }
    return out, dists

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-class", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--ckpt", required=True, help="device checkpoint")
    ap.add_argument("--geom", choices=["petal", "tile150"], default="petal")
    ap.add_argument("--out-suffix", default="",
                    help="append to the output file name so several devices "
                         "of the same geometry can coexist")
    ap.add_argument("--regions", nargs="+", default=None,
                    help="tasks present in the device, e.g. --regions NW NE SW "
                         "for the three-task run; the slot matrix shrinks to "
                         "3x3 and misplaced probes come only from those tasks")
    args = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    if args.regions:
        REG[:] = [r for r in config.REGIONS if r in args.regions]
        print("regions under test:", ",".join(REG))
    torch.manual_seed(0)
    if args.geom == "petal":
        ring300, pos300, neg300 = ev2.build_masks()
        geom = {"CENTS": ev2.CENTS, "OFF": ev2.OFF, "SUB": config.SUB_SIZE}
    else:
        import control_tiled_lib as t150
        ring300, pos300, neg300 = t150.build_masks()
        geom = {"CENTS": {r: (t150.CENT, t150.CENT) for r in REG},
                "OFF": t150.OFFS, "SUB": t150.TILE}
    ds_test = data.load_datasets(train=False)
    nets = [(os.path.splitext(os.path.basename(args.ckpt))[0], args.ckpt)]
    result = {"protocol": {
        "normalisation": "shat_c = (E+ - E-)/P_out, P_out = total intensity "
                         "over the full 300x300 detector plane (unitary => "
                         "equals incident power)",
        "n_per_class": args.n_per_class,
        "test_sets": "same balanced 100/class index sets as the accuracy table",
        "input": "eval_accuracy.make_input convention (peak-norm, noise pad)"},
        "nets": {}}
    all_dists = {}
    for tag, path in nets:
        r, dd = evaluate(tag, path, ds_test, ring300, pos300, neg300,
                         args.n_per_class, args.batch, geom)
        if r:
            result["nets"][tag] = r
            all_dists.update(dd)
    out_path = os.path.join(RES,
                            f"false_alarm_{args.geom}{args.out_suffix}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=1)
    np.savez_compressed(out_path.replace(".json", "_dists.npz"), **all_dists)
    print("saved ->", out_path.replace(".json", "_dists.npz"))
    print("\nsaved ->", out_path)
