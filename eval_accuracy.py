import os
import json

import torch

from d2nn import config
from d2nn import data
from d2nn.stage2_net import FullNet3L
from d2nn.stage2_net import CENTS
from d2nn.stage_utils import pad_input_to_canvas
from d2nn.readout import _make_differential_masks
from d2nn.readout import _get_balanced_indices

DEV = config.DEVICE
NOISE = 0.01
N_OFF = config.SUB_SIZE - config.OVERLAP
OFF = {"NW": (0, 0), "NE": (N_OFF, 0), "SW": (0, N_OFF), "SE": (N_OFF, N_OFF)}
REG = config.REGIONS
RI = {r: i for i, r in enumerate(REG)}
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

def load_full(path):
    if not os.path.exists(path):
        return None
    full = FullNet3L().to(DEV)
    full.load_state_dict(torch.load(path, map_location=DEV, weights_only=False))
    full.eval()
    return full

def build_masks():
    lp = config.FULL_SIZE * config.SPREAD_SCALE * config.LIMIT_SCALE
    ln = lp * 1.4
    sg = config.FULL_SIZE * config.SIGMA_SCALE
    ring, pos300, neg300 = {}, {}, {}
    for r in REG:
        cx, cy = CENTS[r]
        mp, mn = _make_differential_masks(cx, cy, config.SUB_SIZE, config.DX,
                                          lp, ln, sg, config.N_CLASSES, DEV)
        mpb, mnb = (mp > 0.5).float(), (mn > 0.5).float()
        mp300 = torch.zeros(config.N_CLASSES, config.N, config.N, device=DEV)
        mn300 = torch.zeros(config.N_CLASSES, config.N, config.N, device=DEV)
        ox, oy = OFF[r]
        mp300[:, oy:oy + config.SUB_SIZE, ox:ox + config.SUB_SIZE] = mpb
        mn300[:, oy:oy + config.SUB_SIZE, ox:ox + config.SUB_SIZE] = mnb
        pos300[r], neg300[r] = mp300, mn300
        ring[r] = ((mp300.sum(0) + mn300.sum(0)) > 0).float()
    return ring, pos300, neg300

def make_input(region, img, anchor_region=None):
    ar = anchor_region or region
    cx, cy = CENTS[ar]
    sc = pad_input_to_canvas(img, config.SUB_SIZE, upsample=4,
                             centroid=(cx, cy), noise_std=NOISE)
    if sc.max() > 0:
        sc = sc / sc.max()
    big = torch.zeros(config.N, config.N, device=DEV)
    ox, oy = OFF[ar]
    big[oy:oy + config.SUB_SIZE, ox:ox + config.SUB_SIZE] = sc
    return big.unsqueeze(0).to(torch.complex64)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True,
                    help="merged-device checkpoints to evaluate")
    ap.add_argument("--n-per-class", type=int, default=100)
    ap.add_argument("--n-energy", type=int, default=10)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    nets = [(os.path.splitext(os.path.basename(p))[0], p) for p in a.ckpt]
    ring, pos300, neg300 = build_masks()
    ds_test = data.load_datasets(train=False)
    idx_acc = {r: _get_balanced_indices(ds_test[r], n_per_class=a.n_per_class)
               for r in REG}
    idx_en = {r: _get_balanced_indices(ds_test[r], n_per_class=a.n_energy)
              for r in REG}

    results = {}
    for tag, path in nets:
        full = load_full(path)
        if full is None:
            print(f"[{tag}] MISSING {path} -- skipped")
            continue
        rm = full.region_mask
        print(f"\n=== {tag} ===")

        acc_diff, acc_pos = {}, {}
        with torch.no_grad():
            for r in REG:
                cd = cp = tot = 0
                for i in idx_acc[r]:
                    img = ds_test[r][i][0]
                    if img.dim() == 3:
                        img = img.squeeze(0)
                    I = torch.abs(full(make_input(r, img))) ** 2
                    I = I[0]
                    ep = [(I * pos300[r][c]).sum().item() for c in range(10)]
                    en = [(I * neg300[r][c]).sum().item() for c in range(10)]
                    lab = int(ds_test[r][i][1]) % 10
                    if max(range(10), key=lambda c: ep[c] - en[c]) == lab:
                        cd += 1
                    if max(range(10), key=lambda c: ep[c]) == lab:
                        cp += 1
                    tot += 1
                acc_diff[r], acc_pos[r] = cd / tot, cp / tot
        print("M1 acc (differential): " +
              "  ".join("%s=%.3f" % (r, acc_diff[r]) for r in REG))
        print("M4 acc (pos-only)    : " +
              "  ".join("%s=%.3f" % (r, acc_pos[r]) for r in REG))

        m2 = {}
        with torch.no_grad():
            for r in REG:
                acc = torch.zeros(4, device=DEV)
                for i in idx_en[r]:
                    img = ds_test[r][i][0]
                    if img.dim() == 3:
                        img = img.squeeze(0)
                    I = torch.abs(full(make_input(r, img))) ** 2
                    I = I[0]
                    tot = I.sum() + 1e-12
                    own_ring = (I * ring[r]).sum() / tot
                    other_ring = sum((I * ring[r2]).sum() / tot
                                     for r2 in REG if r2 != r)
                    own_q_off = (I * rm[RI[r]] * (1 - ring[r])).sum() / tot
                    rest = 1 - own_ring - other_ring - own_q_off
                    acc += torch.stack([own_ring, own_q_off, other_ring, rest])
                m2[r] = (acc / len(idx_en[r])).tolist()
        print("M2 correct-position energy "
              "[ownRing, ownQtrOffRing, otherRings, rest]:")
        for r in REG:
            print("   %-3s %s" % (r,
                  "  ".join("%.1f%%" % (100 * v) for v in m2[r])))

        m3 = {}
        with torch.no_grad():
            for A in REG:
                for B in REG:
                    if A == B:
                        continue
                    acc = torch.zeros(3, device=DEV)
                    for i in idx_en[A]:
                        img = ds_test[A][i][0]
                        if img.dim() == 3:
                            img = img.squeeze(0)
                        I = torch.abs(full(make_input(A, img,
                                                      anchor_region=B))) ** 2
                        I = I[0]
                        tot = I.sum() + 1e-12
                        b_ring = (I * ring[B]).sum() / tot
                        a_ring = (I * ring[A]).sum() / tot
                        oth = sum((I * ring[r2]).sum() / tot
                                  for r2 in REG if r2 not in (A, B))
                        acc += torch.stack([b_ring, a_ring, oth])
                    m3[A + "->" + B] = (acc / len(idx_en[A])).tolist()
        print("M3 wrong-position energy "
              "[B_ring(false act.), A_ring(loyal), other rings]:")
        for k in sorted(m3):
            print("   %-8s %s" % (k,
                  "  ".join("%.2f%%" % (100 * v) for v in m3[k])))

        results[tag] = {"M1_acc_differential": acc_diff,
                        "M4_acc_pos_only": acc_pos,
                        "M2_energy_correct": m2,
                        "M3_energy_wrongpos": m3}

    out = a.out or os.path.join(RES, "accuracy_eval.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print("\nsaved -> " + out)

if __name__ == "__main__":
    main()
