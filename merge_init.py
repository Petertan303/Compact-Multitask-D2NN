import os
import json

import torch

from d2nn import config
from d2nn import data
from d2nn.stage_utils import SubNetNL
from d2nn.stage2_net import FullNet3L
from d2nn.stage2_net import merge_3layer_subs
from d2nn.stage2_net import CENTS
from d2nn.readout import _get_balanced_indices

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
CKPT_DIR = os.path.join(HERE, "checkpoints")

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2")
    args = ap.parse_args()
    subs = {}
    for r in config.REGIONS:
        ck = os.path.join(CKPT_DIR, f"sub_{r}_3layer_{args.tag}.pth")
        if not os.path.exists(ck):
            raise FileNotFoundError(ck)
        s = SubNetNL(n_layers=3, region_type=r).to(config.DEVICE)
        s.load_state_dict(torch.load(ck, map_location=config.DEVICE))
        subs[r] = s

    full = merge_3layer_subs(subs).to(config.DEVICE)
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(RES, exist_ok=True)
    ckpt_path = os.path.join(CKPT_DIR, f"full_p2_{args.tag}_init.pth")
    torch.save(full.state_dict(), ckpt_path)
    print(f"saved -> {ckpt_path}")

    import train_stage2 as base
    pos, neg = base.build_masks300()
    ds_test = data.load_datasets(train=False)
    accs = base.test_full(full, ds_test, pos, neg)
    avg = sum(accs.values()) / len(accs)
    out = os.path.join(RES, f"merge_init_{args.tag}.json")
    with open(out, "w") as f:
        json.dump({"ckpt": ckpt_path, "accs": accs, "avg": avg}, f, indent=2)
    print("merged init accs: " +
          "  ".join(f"{r}={accs[r]*100:.2f}%" for r in config.REGIONS) +
          f"  AVG={avg*100:.2f}%")
    print(f"-> {out}")

if __name__ == "__main__":
    main()
