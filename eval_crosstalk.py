import argparse
import json
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import eval_concurrent as ecb
from d2nn import config
from d2nn import data
from d2nn.readout import _get_balanced_indices

DEV = config.DEVICE
NC = config.N_CLASSES
NOISE = 0.01

def embed(pos, neg, region):
    side = config.SUB_SIZE
    wy, wx = ecb.QS[region]
    out = []
    for m in (pos, neg):
        big = torch.zeros(NC, config.N, config.N, device=DEV)
        big[:, wy, wx] = m.reshape(NC, side, side)
        out.append(big)
    return out[0], out[1]

def input_canvas(img, region):
    if img.dim() == 3:
        img = img.squeeze(0)
    sc = ecb.pad_input_to_canvas(img.to(DEV), config.SUB_SIZE,
                                 upsample=config.INPUT_SCALE,
                                 centroid=ecb.CENTS[region], noise_std=NOISE)
    if float(sc.max()) > 0:
        sc = sc / sc.max()
    big = torch.zeros(config.N, config.N, device=DEV)
    wy, wx = ecb.QS[region]
    big[wy, wx] = sc
    return big.to(torch.complex64)

def blank_canvas(region):
    z = torch.zeros(config.INPUT_SIZE, config.INPUT_SIZE, device=DEV)
    sc = ecb.pad_input_to_canvas(z, config.SUB_SIZE, upsample=1,
                                 centroid=ecb.CENTS[region], noise_std=NOISE)
    big = torch.zeros(config.N, config.N, device=DEV)
    wy, wx = ecb.QS[region]
    big[wy, wx] = sc.to(DEV)
    return big.to(torch.complex64)

@torch.no_grad()
def measure(net, stack, ring, pos300, neg300, own, batch):
    own_f, oth_f, d_max, pred = [], [], [], []
    for s in range(0, len(stack), batch):
        I = (net(stack[s:s + batch]).abs() ** 2).float()
        If = I.reshape(I.shape[0], -1)
        P = If.sum(1) + 1e-12
        own_f.append(((If * ring[own].reshape(1, -1)).sum(1) / P).tolist())
        oth = torch.zeros_like(P)
        for q in ring:
            if q != own:
                oth = oth + (If * ring[q].reshape(1, -1)).sum(1)
        oth_f.append((oth / P).tolist())
        d = If @ pos300[own].reshape(NC, -1).T - If @ neg300[own].reshape(NC, -1).T
        shat = d / P[:, None]
        d_max.append(shat.max(1).values.tolist())
        pred.append(shat.argmax(1).tolist())
    return (np.array([x for a in own_f for x in a]),
            np.array([x for a in oth_f for x in a]),
            np.array([x for a in d_max for x in a]),
            np.array([x for a in pred for x in a]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--regions", nargs="+", default=None,
                    help="tasks present in the device; default all four")
    ap.add_argument("--n-per-class", type=int, default=100)
    ap.add_argument("--n-foreign", type=int, default=100)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--tag", default="device")
    a = ap.parse_args()
    regs = ([r for r in config.REGIONS if r in a.regions] if a.regions
            else list(config.REGIONS))
    net = ecb.FullNet3L().to(DEV)
    st = torch.load(a.ckpt, map_location=DEV, weights_only=False)
    for k in set(st.keys()) - set(net.state_dict().keys()):
        del st[k]
    net.load_state_dict(st)
    net.eval()
    pos, neg = ecb.petal_masks()
    pos300, neg300, ring = {}, {}, {}
    for r in regs:
        p3, n3 = embed(pos[r], neg[r], r)
        pos300[r], neg300[r] = p3, n3
        ring[r] = ((p3.sum(0) + n3.sum(0)) > 0).float()
    ds = data.load_datasets(train=False)
    out = {"tag": a.tag, "ckpt": a.ckpt, "regions": regs, "per_region": {}}
    for r in regs:
        idx = _get_balanced_indices(ds[r], n_per_class=a.n_per_class)
        stack = torch.stack([(ds[r][i][0].squeeze(0) if ds[r][i][0].dim() == 3
                              else ds[r][i][0]) for i in idx]).to(DEV)
        labs = np.array([int(ds[r][i][1]) % NC for i in idx])
        canv = torch.stack([input_canvas(stack[i], r)
                            for i in range(len(stack))])
        own, oth, dmax, pred = measure(net, canv, ring, pos300, neg300, r,
                                       a.batch)
        rec = {"acc": float((pred == labs).mean()),
               "own_ring": float(own.mean()), "other_rings": float(oth.mean()),
               "conf_dB": 10 * math.log10(max(own.mean(), 1e-12)
                                          / max(oth.mean(), 1e-12)),
               "D_own_mean": float(dmax.mean()),
               "D_own_p50": float(np.median(dmax)),
               "D_own_p99": float(np.percentile(dmax, 99)),
               "tpr_at_0p2": float((dmax >= 0.2).mean())}
        fmax, fmean, fall = [], [], []
        for q in regs:
            if q == r:
                continue
            fidx = _get_balanced_indices(ds[q], n_per_class=a.n_foreign)[:1000]
            fst = torch.stack([(ds[q][i][0].squeeze(0)
                                if ds[q][i][0].dim() == 3 else ds[q][i][0])
                               for i in fidx]).to(DEV)
            fcv = torch.stack([input_canvas(fst[i], r)
                               for i in range(len(fst))])
            _, _, fd, _ = measure(net, fcv, ring, pos300, neg300, r, a.batch)
            fmax.append(fd.max())
            fmean.append(fd.mean())
            fall.append(fd)
        rec["D_foreign_mean"] = float(np.mean(fmean))
        rec["D_foreign_max"] = float(np.max(fmax))
        blanks = torch.stack([blank_canvas(r) for _ in range(200)])
        _, _, bmax, _ = measure(net, blanks, ring, pos300, neg300, r, a.batch)
        fal = np.concatenate(fall)
        rec["far_foreign_at_0p2"] = float((fal >= 0.2).mean())
        rec["D_blank_max"] = float(bmax.max())
        thr = rec["D_blank_max"]
        rec["tpr_at_blank_thr"] = float((dmax >= thr).mean())
        rec["far_foreign_at_blank_thr"] = float((fal >= thr).mean())
        out["per_region"][r] = rec
        print("%s acc=%5.1f%% own=%5.2f%% other=%5.2f%% conf=%+5.1fdB "
              "D_own_p50=%.4f D_for_max=%.4f D_blank_max=%.4f"
              % (r, 100 * rec["acc"], 100 * rec["own_ring"],
                 100 * rec["other_rings"], rec["conf_dB"], rec["D_own_p50"],
                 rec["D_foreign_max"], rec["D_blank_max"]), flush=True)
    path = os.path.join(HERE, "results", f"ring_energy_{a.tag}.json")
    json.dump(out, open(path, "w"), indent=1)
    print("saved ->", path)

if __name__ == "__main__":
    main()
