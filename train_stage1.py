import os
import json
import time

import torch
import torch.optim as optim

from d2nn import config
from d2nn import data
from d2nn import losses
from d2nn.stage_utils import SubNetNL
from d2nn.stage_utils import pad_input_to_canvas
from d2nn.stage1_centroid import compute_quarter_circle_centroids
from d2nn.training import _set_seed
from d2nn.training import _sample_batch
from d2nn.training import _make_differential_gallery
from d2nn.readout import _get_balanced_indices
from d2nn.readout import _make_differential_masks

EPOCHS = 300
LR = 1e-1
BATCH = 256
NOISE = 0.01
CE_SCALE = 100.0
ALPHA_CE = 1e-2
ALPHA_ANTI = 40.0
ALPHA_BG = 1e4
UPSAMPLE = 4

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
CKPT_DIR = os.path.join(HERE, "checkpoints")
CENTS, _ = compute_quarter_circle_centroids()

def build_target(region):
    cx, cy = CENTS[region]
    lp = config.FULL_SIZE * config.SPREAD_SCALE * config.LIMIT_SCALE
    ln = lp * 1.4
    sg = config.FULL_SIZE * config.SIGMA_SCALE
    gp, gn = _make_differential_gallery(cx, cy, config.SUB_SIZE, config.DX,
                                        lp, ln, sg, config.N_CLASSES,
                                        config.DEVICE)
    return (gp > 0.5).float(), (gn > 0.5).float()

def train_region(region, ckpt, n_layers, alpha_bg=1e4, epochs=EPOCHS, seed=0):
    _set_seed(seed)
    ds = data.load_datasets(train=True)
    own_ds = ds[region]
    mask_pos, mask_neg = build_target(region)
    cx, cy = CENTS[region]

    sub = SubNetNL(n_layers=n_layers, region_type=region).to(config.DEVICE)
    opt = optim.Adam(sub.parameters(), lr=LR)
    sched = optim.lr_scheduler.LinearLR(opt, start_factor=1.0,
                                        end_factor=1e-6 / LR,
                                        total_iters=epochs)
    t0 = time.time()
    for epoch in range(epochs):
        opt.zero_grad()
        imgs, labels, _ = _sample_batch(own_ds, BATCH, config.DEVICE)
        imgs = losses.normalize_input_to_canvas_energy(
            imgs, target_l2_sq=config.CANVAS_L2_SQ)
        canvas = torch.stack([
            pad_input_to_canvas(imgs[i], config.SUB_SIZE, upsample=UPSAMPLE,
                                centroid=(cx, cy), noise_std=NOISE)
            for i in range(imgs.shape[0])
        ])
        out = sub(canvas.to(torch.complex64))
        L = losses.target_loss(out_field=out, labels=labels,
                                mask_pos=mask_pos, mask_neg=mask_neg,
                                ce_scale=CE_SCALE, alpha_ce=ALPHA_CE,
                                alpha_anti_neg=ALPHA_ANTI, alpha_bg=alpha_bg)
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
            print(f"  [{region}] ep {epoch:3d}: L={float(L.item()):.3f} "
                  f"gn={float(gn):.2f} lr={opt.param_groups[0]['lr']:.2e} "
                  f"[{time.time() - t0:.0f}s]", flush=True)
    torch.save(sub.state_dict(), ckpt)

def test_region(region, ckpt, n_layers, n_per_class=100):
    ds_test = data.load_datasets(train=False)
    cx, cy = CENTS[region]
    lp = config.FULL_SIZE * config.SPREAD_SCALE * config.LIMIT_SCALE
    ln = lp * 1.4
    sg = config.FULL_SIZE * config.SIGMA_SCALE
    mp, mn = _make_differential_masks(cx, cy, config.SUB_SIZE, config.DX,
                                      lp, ln, sg, config.N_CLASSES,
                                      config.DEVICE)
    sub = SubNetNL(n_layers=n_layers, region_type=region).to(config.DEVICE)
    sub.load_state_dict(torch.load(ckpt, map_location=config.DEVICE))
    sub.eval()
    idx = _get_balanced_indices(ds_test[region], n_per_class=n_per_class)
    correct = total = 0
    with torch.no_grad():
        for i in idx:
            img, lab = ds_test[region][i]
            if hasattr(img, "squeeze") and img.dim() == 3:
                img = img.squeeze(0)
            sc = pad_input_to_canvas(img, config.SUB_SIZE, upsample=UPSAMPLE,
                                     centroid=(cx, cy), noise_std=NOISE)
            if sc.max() > 0:
                sc = sc / sc.max()
            x = sc.unsqueeze(0).to(config.DEVICE).to(torch.complex64)
            out = sub(x)
            I = torch.abs(out) ** 2
            ep = [(I[0] * mp[c_]).sum().item() for c_ in range(config.N_CLASSES)]
            en = [(I[0] * mn[c_]).sum().item() for c_ in range(config.N_CLASSES)]
            pred = max(range(config.N_CLASSES), key=lambda c_: ep[c_] - en[c_])
            if pred == int(lab) % 10:
                correct += 1
            total += 1
    sub.train()
    return correct / max(total, 1)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--alpha-bg", type=float, default=1e4)
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--regions", nargs="+", default=config.REGIONS)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    n_layers = args.layers
    os.makedirs(RES, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    accs = {}
    for region in args.regions:
        ckpt = os.path.join(CKPT_DIR,
                            f"sub_{region}_{n_layers}layer_{args.tag}.pth")
        print(f"\n=== {region}  dataset={config.REGION_TASKS[region]}  "
              f"layers={n_layers}  alpha_bg={args.alpha_bg:g}  "
              f"centroid={CENTS[region]} ===", flush=True)
        t0 = time.time()
        train_region(region, ckpt, n_layers, args.alpha_bg, args.epochs,
                     args.seed)
        acc = test_region(region, ckpt, n_layers)
        accs[region] = {"acc": acc,
                        "dataset": config.REGION_TASKS[region],
                        "layers": n_layers,
                        "ckpt": ckpt,
                        "elapsed_s": round(time.time() - t0, 1)}
        print(f"  {region} acc (own test set, 100/class) = {acc * 100:.2f}%",
              flush=True)
    out_json = os.path.join(RES, f"phase1_{args.tag}_{n_layers}l_s{args.seed}.json")
    with open(out_json, "w") as f:
        json.dump(accs, f, indent=2)
    print(f"\nDone. -> {out_json}")
