import os
import json
import argparse

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
CKPT_DIR = os.path.join(HERE, "checkpoints")

from control_tiled_lib import SquareSubNet
from control_tiled_lib import TiledFullNet
from control_tiled_lib import tile_subs
from control_tiled_lib import build_masks
from control_tiled_lib import energy_partition
from control_tiled_lib import eval_acc
from control_tiled_lib import REG
from control_tiled_lib import DEV
from control_tiled_lib import TILE
from d2nn import data

def load_subs(layers, tag, seed):
    subs = {}
    for r in REG:
        s = SquareSubNet(layers).to(DEV)
        s.load_state_dict(torch.load(
            os.path.join(CKPT_DIR, f"sub150_{r}_{layers}layer_{tag}_"
                                   f"s{seed}.pth"),
            map_location=DEV, weights_only=False))
        subs[r] = s
    return subs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--tag", default="bg1e3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-acc", type=int, default=100)
    ap.add_argument("--n-energy", type=int, default=20)
    args = ap.parse_args()
    torch.manual_seed(0)
    ring, pos, neg = build_masks()
    ds_test = data.load_datasets(train=False)
    out = {"protocol": {"acc": "differential argmax, 100/class, tile geometry",
                        "energy": "fractions of total output power, "
                                  "%d/class" % args.n_energy,
                        "geometry": "4 x 150x150 squares, rings at quadrant "
                                    "centres (identical radii to the "
                                    "proposed device)"},
           "device": {}}
    tag2 = f"tile150_{args.layers}l_{args.tag}_s{args.seed}"
    states = [("pasted_init", None)]
    for ep in (1000, 500):
        p = os.path.join(CKPT_DIR,
                         f"full_tile150_1000ep_{args.layers}l_tiled_"
                         f"{args.tag}_s{args.seed}_ep{ep:04d}.pth")
        if os.path.exists(p):
            states.append((f"finetuned_ep{ep}", p))
    for name, path in states:
        if path is None:
            net = tile_subs(load_subs(args.layers, args.tag, args.seed),
                            args.layers).to(DEV)

            init_path = os.path.join(
                CKPT_DIR, f"full_tile150_{args.layers}l_tiled_{args.tag}_"
                          f"s{args.seed}_pasted_only.pth")
            torch.save(net.state_dict(), init_path)
            print("  pasted-only ckpt ->", init_path)
        else:
            net = TiledFullNet(args.layers).to(DEV)
            net.load_state_dict(torch.load(path, map_location=DEV,
                                           weights_only=False))
        net.eval()
        accs = eval_acc(net, ds_test, pos, neg, n_per_class=args.n_acc)
        m2, m3 = energy_partition(net, ds_test, ring,
                                  n_per_class=args.n_energy)
        mean = sum(accs.values()) / len(accs)
        print(f"\n=== {name} === M1 mean={100*mean:.1f}%  " +
              "  ".join(f"{r}={100*accs[r]:.1f}" for r in REG))
        print("M2 [ownRing, ownSqOffRing, otherRings, rest]:")
        for r in REG:
            print("   %-3s %s" % (r, "  ".join("%.2f%%" % (100 * v)
                                               for v in m2[r])))
        print("M3 [B_ring(false act.), A_ring(loyal), other]:")
        for k in sorted(m3):
            print("   %-8s %s" % (k, "  ".join("%.2f%%" % (100 * v)
                                               for v in m3[k])))
        out["device"][name] = {"M1_acc": accs, "M1_mean": mean,
                               "M2_energy_correct": m2,
                               "M3_energy_wrongpos": m3,
                               "ckpt": path}

    with open(os.path.join(RES,
                           f"tile150_crosstalk_{args.tag}_s{args.seed}.json"),
              "w") as f:
        json.dump(out, f, indent=1)
    print("\nsaved ->", os.path.join(
        RES, f"tile150_crosstalk_{args.tag}_s{args.seed}.json"))

if __name__ == "__main__":
    main()
