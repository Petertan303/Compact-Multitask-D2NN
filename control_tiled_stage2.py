import os
import json
import time
import argparse

import torch
import torch.optim as optim

from control_tiled_lib import SquareSubNet
from control_tiled_lib import TiledFullNet
from control_tiled_lib import tile_subs
from control_tiled_lib import build_masks
from control_tiled_lib import full_input
from control_tiled_lib import differential_scores
from control_tiled_lib import REG
from control_tiled_lib import RI
from control_tiled_lib import DEV
from control_tiled_lib import TILE
from control_tiled_lib import OFFS
from control_tiled_lib import CE_SCALE
from control_tiled_lib import ALPHA_CE
from control_tiled_lib import ALPHA_ANTI
from control_tiled_lib import BATCH
from control_tiled_lib import LR
from control_tiled_lib import N_LAYERS
from d2nn import config
from d2nn import data
from d2nn import losses
from d2nn import training as ctrain
from d2nn.readout import _get_balanced_indices

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
CKPT_DIR = os.path.join(HERE, "checkpoints")

def load_subs(layers, tag, seed):
    subs = {}
    for r in REG:
        p = os.path.join(CKPT_DIR, f"sub150_{r}_{layers}layer_{tag}_s{seed}.pth")
        s = SquareSubNet(layers).to(DEV)
        s.load_state_dict(torch.load(p, map_location=DEV, weights_only=False))
        subs[r] = s
    return subs

def train_canvas_batch(region, imgs):
    from control_tiled_lib import sub_canvas
    imgs_c = sub_canvas(imgs)
    ox, oy = OFFS[region]
    big = torch.zeros(imgs_c.shape[0], config.N, config.N, device=DEV)
    big[:, oy:oy + TILE, ox:ox + TILE] = imgs_c.real
    return big.to(torch.complex64)

def test_full(full, ds_test, pos, neg, n_per_class=100):
    full.eval()
    accs = {}
    with torch.no_grad():
        for r in REG:
            idx = _get_balanced_indices(ds_test[r], n_per_class=n_per_class)
            c = tot = 0
            for s in range(0, len(idx), 64):
                ids = idx[s:s + 64]
                ims = torch.stack([ds_test[r][i][0] for i in ids])
                labs = [int(ds_test[r][i][1]) % 10 for i in ids]
                I = torch.abs(full(full_input(ims, r))) ** 2
                ep, en = differential_scores(I, pos[r], neg[r])
                pred = (ep - en).argmax(1).tolist()
                c += sum(int(p == l) for p, l in zip(pred, labs))
                tot += len(ids)
            accs[r] = c / max(tot, 1)
    full.train()
    return accs

def forward_loss_region(full, region, ds, pos, neg, alpha_bg,
                        alpha_anti=ALPHA_ANTI):
    imgs, labels, _ = ctrain._sample_batch(ds[region], BATCH, DEV)
    canvases = train_canvas_batch(region, imgs)
    out = full(canvases)
    return losses.target_loss(out_field=out, labels=labels,
                               mask_pos=pos[region], mask_neg=neg[region],
                               ce_scale=CE_SCALE, alpha_ce=ALPHA_CE,
                               alpha_anti_neg=alpha_anti, alpha_bg=alpha_bg)

def rebalance_weights(weights, accs, dyn_penalty):
    easy = max(accs, key=lambda r: accs[r])
    old_w = weights[easy]
    new_easy = old_w * (1.0 - dyn_penalty)
    freed = old_w - new_easy
    w = dict(weights)
    w[easy] = new_easy
    for r in w:
        if r != easy:
            w[r] += freed / (len(w) - 1)
    tot = sum(w.values())
    return {r: v / tot for r, v in w.items()}, easy

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", choices=["tiled", "random"], default="tiled")
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--layers", type=int, default=N_LAYERS)
    ap.add_argument("--alpha-bg", type=float, default=1e3)
    ap.add_argument("--tag", default="bg1e3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--curri-interval", type=int, default=100)
    ap.add_argument("--dyn-penalty", type=float, default=0.30)
    ap.add_argument("--no-curriculum", action="store_true")
    ap.add_argument("--no-anti", action="store_true",
                    help="keep L_bg but switch L_anti off (isolate the negative"
                         " spot penalty)")
    args = ap.parse_args()
    use_curri = not args.no_curriculum
    anti = 0.0 if args.no_anti else ALPHA_ANTI
    os.makedirs(RES, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    tag = f"tile150_{args.epochs}ep_{args.layers}l_{args.init}_{args.tag}" \
          f"_s{args.seed}" + ("" if use_curri else "_nocurri")
    curve_path = os.path.join(RES, f"curve_{tag}.json")

    ctrain._set_seed(args.seed)
    torch.manual_seed(args.seed)
    if args.init == "tiled":
        full = tile_subs(load_subs(args.layers, args.tag, args.seed),
                         args.layers).to(DEV)
    else:
        full = TiledFullNet(args.layers).to(DEV)
    print(f"[{tag}] params={sum(p.numel() for p in full.parameters())} "
          f"trainable px/layer={int(full.region_mask.sum()/args.layers)} "
          f"alpha_bg={args.alpha_bg:g} alpha_anti={anti:g} "
          f"curriculum={use_curri}", flush=True)

    ring, pos, neg = build_masks()
    ds_train = data.load_datasets(train=True)
    ds_test = data.load_datasets(train=False)

    points = []
    last_accs = None

    def dump():
        with open(curve_path, "w") as f:
            json.dump({"tag": tag, "init": args.init, "epochs": args.epochs,
                       "eval_every": args.eval_every, "alpha_bg": args.alpha_bg,
                   "alpha_anti": anti,
                       "seed": args.seed, "curriculum": use_curri,
                       "points": points}, f, indent=1)

    def record(ep, weights=None):
        nonlocal last_accs
        t0 = time.time()
        accs = test_full(full, ds_test, pos, neg)
        last_accs = accs
        avg = sum(accs.values()) / len(accs)
        points.append({"epoch": ep, "accs": accs, "avg": avg,
                       "region_weights": weights})
        dump()
        print(f"  [eval ep {ep}] " +
              "  ".join(f"{r}={accs[r]*100:.2f}%" for r in REG) +
              f"  AVG={avg*100:.2f}%" +
              (f"  W=" + " ".join(f"{r}:{weights[r]:.3f}" for r in REG)
               if weights else "") + f"  [{time.time()-t0:.0f}s]", flush=True)

    weights = {r: 0.25 for r in REG}
    w_tensor = torch.tensor([weights[r] for r in REG])
    record(0, weights=weights)
    opt = optim.Adam(full.parameters(), lr=LR)
    sched = optim.lr_scheduler.LinearLR(opt, start_factor=1.0,
                                        end_factor=1e-6 / LR,
                                        total_iters=300 * len(REG))
    t0 = time.time()
    for ep in range(args.epochs):
        if use_curri:
            pick = torch.multinomial(w_tensor, 4, replacement=True)
            epoch_regions = [REG[i] for i in pick.tolist()]
        else:
            epoch_regions = list(REG)
        ep_loss = {}
        for r in epoch_regions:
            ri = RI[r]
            L = forward_loss_region(full, r, ds_train, pos, neg, args.alpha_bg,
                                    anti)
            if torch.is_complex(L):
                L = L.real
            if torch.isnan(L) or torch.isinf(L):
                continue
            ep_loss[r] = float(L.item())
            L.backward()
            for layer in full.layers:
                p = layer.phase
                if p.grad is not None:
                    p.grad.copy_(p.grad * full.region_mask[ri])
            gn = torch.nn.utils.clip_grad_norm_(full.parameters(), 1.0)
            if torch.isfinite(gn):
                opt.step()
            sched.step()
            opt.zero_grad()
        if (ep + 1) % 100 == 0 and (ep + 1) < args.epochs:
            torch.save(full.state_dict(),
                       os.path.join(CKPT_DIR, f"full_{tag}_ep{ep+1:04d}.pth"))
        if (ep + 1) % 50 == 0:
            finite = [v for v in ep_loss.values() if v == v]
            print(f"  ep {ep+1:4d}/{args.epochs}: avgL="
                  f"{sum(finite)/max(1,len(finite)):.0f} "
                  f"lr={opt.param_groups[0]['lr']:.2e} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        if (ep + 1) % args.eval_every == 0 and (ep + 1) < args.epochs:
            record(ep + 1, weights=weights)
            if (use_curri and last_accs is not None
                    and (ep + 1) % args.curri_interval == 0):
                weights, easy = rebalance_weights(weights, last_accs,
                                                  args.dyn_penalty)
                w_tensor = torch.tensor([weights[r] for r in REG])
                points[-1]["region_weights_after"] = weights
                dump()
                print(f"  [curriculum @ ep {ep+1}] easiest={easy} -> " +
                      " ".join(f"{r}:{weights[r]:.3f}" for r in REG), flush=True)
    final = os.path.join(CKPT_DIR, f"full_{tag}_ep{args.epochs:04d}.pth")
    torch.save(full.state_dict(), final)
    record(args.epochs, weights=weights)
    with open(os.path.join(RES, f"phase2_{tag}_summary.json"), "w") as f:
        json.dump({"tag": tag, "final_ckpt": final,
                   "final_accs": last_accs,
                   "final_avg": sum(last_accs.values()) / len(last_accs),
                   "final_weights": weights}, f, indent=1)
    print("saved ->", final)

if __name__ == "__main__":
    main()
