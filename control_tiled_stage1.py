import os
import json
import time
import argparse

import torch
import torch.optim as optim

from control_tiled_lib import SquareSubNet
from control_tiled_lib import build_sub_masks
from control_tiled_lib import sub_canvas
from control_tiled_lib import REG
from control_tiled_lib import CE_SCALE
from control_tiled_lib import ALPHA_CE
from control_tiled_lib import ALPHA_ANTI
from control_tiled_lib import LR
from control_tiled_lib import BATCH
from control_tiled_lib import DEV
from d2nn import config
from d2nn import data
from d2nn import losses
from d2nn.training import _set_seed
from d2nn.training import _sample_batch
from d2nn.readout import _get_balanced_indices

HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(HERE, "checkpoints")
RES = os.path.join(HERE, "results")
EPOCHS = 300
NOISE = 0.01

def train_region(region, ckpt, n_layers, alpha_bg, seed, epochs=EPOCHS,
                 alpha_anti=ALPHA_ANTI):
    _set_seed(seed)
    torch.manual_seed(seed)
    ds = data.load_datasets(train=True)[region]
    ring, mask_pos, mask_neg = build_sub_masks()
    sub = SquareSubNet(n_layers=n_layers).to(DEV)
    opt = optim.Adam(sub.parameters(), lr=LR)
    sched = optim.lr_scheduler.LinearLR(opt, start_factor=1.0,
                                        end_factor=1e-6 / LR,
                                        total_iters=epochs)
    t0 = time.time()
    for epoch in range(epochs):
        opt.zero_grad()
        imgs, labels, _ = _sample_batch(ds, BATCH, DEV)
        canvas = sub_canvas(imgs)
        out = sub(canvas.to(DEV))
        L = losses.target_loss(out_field=out, labels=labels,
                                mask_pos=mask_pos, mask_neg=mask_neg,
                                ce_scale=CE_SCALE, alpha_ce=ALPHA_CE,
                                alpha_anti_neg=alpha_anti, alpha_bg=alpha_bg)
        if torch.is_complex(L):
            L = L.real
        if torch.isnan(L) or torch.isinf(L):
            sched.step()
            continue
        L.backward()
        gn = torch.nn.utils.clip_grad_norm_(sub.parameters(), 1.0)
        if torch.isfinite(gn):
            opt.step()
        sched.step()
        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"  [{region}] ep {epoch:3d}: L={float(L):.3f} "
                  f"gn={float(gn):.2f} lr={opt.param_groups[0]['lr']:.2e} "
                  f"[{time.time() - t0:.0f}s]", flush=True)
    torch.save(sub.state_dict(), ckpt)

def test_region(region, ckpt, n_layers, n_per_class=100):
    ring, mask_pos, mask_neg = build_sub_masks()
    ds_test = data.load_datasets(train=False)[region]
    sub = SquareSubNet(n_layers=n_layers).to(DEV)
    sub.load_state_dict(torch.load(ckpt, map_location=DEV, weights_only=False))
    sub.eval()
    idx = _get_balanced_indices(ds_test, n_per_class=n_per_class)
    correct = total = 0
    with torch.no_grad():
        for i in idx:
            img, lab = ds_test[i]
            if img.dim() == 3:
                img = img.squeeze(0)
            x = sub_canvas(img.unsqueeze(0))
            if x.real.max() > 0:
                x = x / x.real.max()
            I = torch.abs(sub(x.to(DEV))) ** 2
            I = I[0]
            ep = [(I * mask_pos[c]).sum().item() for c in range(10)]
            en = [(I * mask_neg[c]).sum().item() for c in range(10)]
            pred = max(range(10), key=lambda c: ep[c] - en[c])
            if pred == int(lab) % 10:
                correct += 1
            total += 1
    sub.train()
    return correct / max(total, 1)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--alpha-bg", type=float, default=1e3)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--tag", default="bg1e3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ce-only", action="store_true",
                    help="accuracy term only: L_anti and L_bg switched off")
    ap.add_argument("--no-anti", action="store_true",
                    help="keep L_bg but switch L_anti off (isolate the negative"
                         " spot penalty)")
    args = ap.parse_args()
    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(RES, exist_ok=True)
    accs = {}
    for region in REG:
        ck = os.path.join(CKPT_DIR, f"sub150_{region}_{args.layers}layer_"
                                    f"{args.tag}_s{args.seed}.pth")
        bg = 0.0 if args.ce_only else args.alpha_bg
        anti = 0.0 if (args.ce_only or args.no_anti) else ALPHA_ANTI
        print(f"\n=== 150sq {region} dataset={config.REGION_TASKS[region]} "
              f"layers={args.layers} alpha_bg={bg:g} alpha_anti={anti:g} "
              f"seed={args.seed} ===", flush=True)
        t0 = time.time()
        train_region(region, ck, args.layers, bg, args.seed, args.epochs, anti)
        acc = test_region(region, ck, args.layers)
        accs[region] = {"acc": acc, "dataset": config.REGION_TASKS[region],
                        "ckpt": ck, "elapsed_s": round(time.time() - t0, 1)}
        print(f"  {region} acc (own test set, 100/class) = {acc * 100:.2f}%",
              flush=True)
    mean = sum(v["acc"] for v in accs.values()) / len(accs)
    print(f"\n150sq independent mean = {mean * 100:.2f}%")
    out = os.path.join(RES, f"phase1_150_{args.tag}_s{args.seed}_accs.json")
    with open(out, "w") as f:
        json.dump({"tag": args.tag, "seed": args.seed,
                   "layers": args.layers,
                   "alpha_bg": 0.0 if args.ce_only else args.alpha_bg,
                   "alpha_anti": 0.0 if args.ce_only else ALPHA_ANTI,
                   "ce_only": bool(args.ce_only),
                   "epochs": args.epochs, "tile": 150,
                   "accs": {r: accs[r]["acc"] for r in accs},
                   "mean": mean, "detail": accs}, f, indent=2)
    print("saved ->", out)
