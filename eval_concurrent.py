import argparse
import itertools
import json
import os
import sys

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))

from d2nn import config
from d2nn import data
from d2nn.stage_utils import pad_input_to_canvas
from d2nn.stage2_net import FullNet3L
from d2nn.stage2_net import CENTS
from d2nn.readout import _make_differential_masks

import control_tiled_lib as t15

REG = list(config.REGIONS)
N_OFF = config.SUB_SIZE - config.OVERLAP
NOISE = 0.01
QS = {"NW": (slice(0, config.SUB_SIZE), slice(0, config.SUB_SIZE)),
      "NE": (slice(0, config.SUB_SIZE), slice(N_OFF, config.N)),
      "SW": (slice(N_OFF, config.N), slice(0, config.SUB_SIZE)),
      "SE": (slice(N_OFF, config.N), slice(N_OFF, config.N))}

def petal_masks():
    lp = config.FULL_SIZE * config.SPREAD_SCALE * config.LIMIT_SCALE
    ln = lp * 1.4
    sg = config.FULL_SIZE * config.SIGMA_SCALE
    pos, neg = {}, {}
    for r in REG:
        mp, mn = _make_differential_masks(CENTS[r][0], CENTS[r][1],
                                          config.SUB_SIZE, config.DX, lp, ln,
                                          sg, config.N_CLASSES, config.DEVICE)
        pos[r] = (mp > 0.5).float().reshape(config.N_CLASSES, -1)
        neg[r] = (mn > 0.5).float().reshape(config.N_CLASSES, -1)
    return pos, neg

def geom_ctx(geom):
    if geom == "petal":
        pos, neg = petal_masks()
        win = QS
        cents = CENTS
        side = config.SUB_SIZE

        crop = QS
    else:
        _, pos, neg = t15.build_masks()
        pos = {r: pos[r].reshape(config.N_CLASSES, -1) for r in REG}
        neg = {r: neg[r].reshape(config.N_CLASSES, -1) for r in REG}
        win = {r: (slice(t15.OFFS[r][1], t15.OFFS[r][1] + t15.TILE),
                   slice(t15.OFFS[r][0], t15.OFFS[r][0] + t15.TILE))
               for r in REG}
        cents = {r: (t15.CENT, t15.CENT) for r in REG}
        side = t15.TILE

        crop = {r: (slice(0, config.N), slice(0, config.N)) for r in REG}
    return pos, neg, win, cents, side, crop

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geom", default="petal", choices=("petal", "tile150"))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tag", default="device")
    ap.add_argument("--n-per-class", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--regions", nargs="+", default=None,
                    help="subset of tasks present in the device, e.g. "
                         "--regions NW NE SW for the three-task run; the "
                         "combination list shrinks to 2**len-1 entries")
    a = ap.parse_args()
    if a.regions:
        REG[:] = [r for r in config.REGIONS if r in a.regions]
        print("regions under test:", ",".join(REG))

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    pos, neg, win, cents, side, crop = geom_ctx(a.geom)

    if a.geom == "petal":
        net = FullNet3L().to(config.DEVICE)
        st = torch.load(a.ckpt, map_location=config.DEVICE, weights_only=False)
        for k in set(st.keys()) - set(net.state_dict().keys()):
            del st[k]
        net.load_state_dict(st)
    else:
        net = t15.TiledFullNet().to(config.DEVICE)
        net.load_state_dict(torch.load(a.ckpt, map_location=config.DEVICE,
                                       weights_only=False))
    net.eval()

    ds = data.load_datasets(train=False)
    pools = {}
    for r in REG:
        byc = {c: [] for c in range(config.N_CLASSES)}
        for i in range(len(ds[r])):
            _, lab = ds[r][i]
            byc[int(lab) % config.N_CLASSES].append(i)
        pools[r] = {c: rng.choice(v, size=min(len(v), 10 * a.n_per_class),
                                  replace=False) for c, v in byc.items()}

    def label_plan(r, n):
        reps = int(np.ceil(n / config.N_CLASSES))
        labs = []
        imgs = []
        for rep in range(reps):
            cl = list(range(config.N_CLASSES))
            rng.shuffle(cl)
            labs += cl
            imgs += [ds[r][pools[r][c][rep]][0] for c in cl]
        return labs[:n], torch.stack(imgs[:n])

    n = config.N_CLASSES * a.n_per_class
    combos = [c for k in range(1, len(REG) + 1)
              for c in itertools.combinations(REG, k)]
    out = {"tag": a.tag, "geom": a.geom, "ckpt": a.ckpt, "n_per_region": n,
           "per_combo": [], "by_k": {}, "per_region_overall": {}}
    tot = {r: [0, 0] for r in REG}
    for combo in combos:
        plan = {r: label_plan(r, n) for r in combo}
        ok = {r: 0 for r in combo}
        for s in range(0, n, a.batch):
            e = min(s + a.batch, n)
            canvas = torch.zeros(e - s, config.N, config.N,
                                 device=config.DEVICE)
            for r in combo:
                labs, imgs = plan[r]
                patches = []
                for it in imgs[s:e]:
                    if it.dim() == 3:
                        it = it.squeeze(0)
                    sc = pad_input_to_canvas(it.to(config.DEVICE), side,
                                             upsample=config.INPUT_SCALE,
                                             centroid=cents[r],
                                             noise_std=NOISE)
                    if sc.max() > 0:
                        sc = sc / sc.max()
                    patches.append(sc)
                p = torch.stack(patches)
                canvas[:, win[r][0], win[r][1]] += p
            with torch.no_grad():
                o = net(canvas.to(torch.complex64))
                I3 = (o.abs() ** 2).float()
            for r in combo:
                labs = plan[r][0][s:e]
                Ic = I3[:, crop[r][0], crop[r][1]].reshape(o.shape[0], -1)
                d = (Ic @ pos[r].T - Ic @ neg[r].T).cpu().numpy()
                pred = d.argmax(1)
                y = np.array(labs)
                ok[r] += int((pred == y).sum())
                tot[r][0] += int((pred == y).sum())
                tot[r][1] += len(y)
        row = {"combo": "+".join(combo),
               "acc": {r: round(ok[r] / n, 4) for r in combo},
               "mean": round(float(np.mean([ok[r] / n for r in combo])), 4)}
        out["per_combo"].append(row)
        out["by_k"].setdefault(str(len(combo)), {"ok": 0, "n": 0})
        out["by_k"][str(len(combo))]["ok"] += sum(ok.values())
        out["by_k"][str(len(combo))]["n"] += len(combo) * n
        print(f"  {row['combo']:<12} " +
              "  ".join(f"{r}={row['acc'][r]*100:5.1f}" for r in combo) +
              f"   mean={row['mean']*100:5.2f}")
    for k, v in out["by_k"].items():
        v["acc"] = v["ok"] / v["n"]
        print(f"k={k}: {v['acc']*100:.2f}%  (n={v['n']})")
    for r in REG:
        out["per_region_overall"][r] = tot[r][0] / tot[r][1]
        print(f"overall {r}: {out['per_region_overall'][r]*100:.2f}% "
              f"(n={tot[r][1]})")
    path = os.path.join(HERE, "results", f"concurrent_{a.tag}.json")
    json.dump(out, open(path, "w"), indent=1)
    print("saved ->", path)

if __name__ == "__main__":
    main()
