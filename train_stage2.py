import os
import json
import time
import argparse

import torch
import torch.nn as nn
import torch.optim as optim

from d2nn import config
from d2nn import data
from d2nn import losses
from d2nn.stage_utils import SubNetNL
from d2nn.stage_utils import pad_input_to_canvas
from d2nn.stage1_centroid import compute_quarter_circle_centroids
from d2nn.layers import PixelDiffractiveLayer
from d2nn.layers import _get_quarter_circle_mask
from d2nn import optics
from d2nn.training import _set_seed
from d2nn.training import _sample_batch
from d2nn.training import _make_differential_gallery
from d2nn.readout import _get_balanced_indices

NOISE = 0.01
CE_SCALE = 100.0
ALPHA_CE = 1e-2
ALPHA_ANTI = 40.0
ALPHA_BG = 1e4
BATCH = 256
UPSAMPLE = 4
LR = 1e-1
N_OFF = config.SUB_SIZE - config.OVERLAP
CENTS, _ = compute_quarter_circle_centroids()
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "results")
SUB_DIR = os.path.join(HERE, "checkpoints")
CKPT_DIR = os.path.join(HERE, "checkpoints")

REGIONS = config.REGIONS
OFFSETS = {"NW": (0, 0), "NE": (N_OFF, 0), "SW": (0, N_OFF), "SE": (N_OFF, N_OFF)}

class FullNetNL(nn.Module):

    def __init__(self, n_layers, smooth_init=False, init_modes=31):
        super().__init__()
        self.n_layers = n_layers
        self.propagator = optics.FreeSpacePropagation(
            n=config.N, dx=config.DX,
            distance=config.DISTANCE, wavelength=config.WAVELENGTH,
        )
        self.layers = nn.ModuleList([
            PixelDiffractiveLayer(config.N, smooth_init=smooth_init,
                                  init_modes=init_modes)
            for _ in range(n_layers)
        ])
        global_mask = torch.zeros((config.N, config.N), dtype=torch.float32)
        for r, (ox, oy) in OFFSETS.items():
            m_sub = _get_quarter_circle_mask(r, config.SUB_SIZE)
            global_mask[oy:oy + config.SUB_SIZE, ox:ox + config.SUB_SIZE] = torch.max(
                global_mask[oy:oy + config.SUB_SIZE, ox:ox + config.SUB_SIZE],
                m_sub)
        self.register_buffer("global_mask", global_mask)
        self.register_buffer("region_mask", torch.zeros((4, config.N, config.N),
                                                         dtype=torch.float32))
        for ri, (r, (ox, oy)) in enumerate(OFFSETS.items()):
            m = _get_quarter_circle_mask(r, config.SUB_SIZE)
            self.region_mask[ri, oy:oy + config.SUB_SIZE, ox:ox + config.SUB_SIZE] = m

    def forward(self, x):
        for layer in self.layers:
            phase = layer.get_phase_mask().to(x.device)
            phase = phase * self.global_mask
            x = x * torch.exp(1j * phase)
            x = self.propagator(x)
        return x

def merge_subs(subs, n_layers):
    first = next(iter(subs.values()))
    device = next(first.parameters()).device
    full = FullNetNL(n_layers).to(device)
    for li in range(n_layers):
        canvas = torch.zeros((config.N, config.N), dtype=torch.float32,
                             device=device)
        count = torch.zeros((config.N, config.N), dtype=torch.float32,
                            device=device)
        for region, (ox, oy) in OFFSETS.items():
            phase_sub = subs[region].get_spatial_phase(li).to(device)
            mask_sub = _get_quarter_circle_mask(region, config.SUB_SIZE, device)
            canvas[oy:oy + config.SUB_SIZE, ox:ox + config.SUB_SIZE] += phase_sub
            count[oy:oy + config.SUB_SIZE, ox:ox + config.SUB_SIZE] += mask_sub
        with torch.no_grad():
            full.layers[li].phase.copy_(canvas / torch.clamp(count, min=1.0))
    return full

def slot(region):
    return OFFSETS[region]

def build_masks300():
    lp = config.FULL_SIZE * config.SPREAD_SCALE * config.LIMIT_SCALE
    ln = lp * 1.4
    sg = config.FULL_SIZE * config.SIGMA_SCALE
    pos, neg = {}, {}
    for r in REGIONS:
        cx, cy = CENTS[r]
        gp, gn = _make_differential_gallery(cx, cy, config.SUB_SIZE, config.DX,
                                            lp, ln, sg, config.N_CLASSES,
                                            config.DEVICE)
        mp = torch.zeros(config.N_CLASSES, config.N, config.N,
                         device=config.DEVICE)
        mn = torch.zeros(config.N_CLASSES, config.N, config.N,
                         device=config.DEVICE)
        ox, oy = slot(r)
        mp[:, oy:oy + config.SUB_SIZE, ox:ox + config.SUB_SIZE] = (gp > 0.5).float()
        mn[:, oy:oy + config.SUB_SIZE, ox:ox + config.SUB_SIZE] = (gn > 0.5).float()
        pos[r], neg[r] = mp, mn
    return pos, neg

def build_grad_masks():
    rm = torch.stack([
        _get_quarter_circle_mask(r, config.SUB_SIZE)
        for r in REGIONS
    ])
    cov = torch.zeros(4, config.N, config.N)
    for ri, (r, (ox, oy)) in enumerate(OFFSETS.items()):
        cov[ri, oy:oy + config.SUB_SIZE, ox:ox + config.SUB_SIZE] += rm[ri]
    total_cov = cov.sum(0)
    excl, ovlp = {}, {}
    for ri, (r, (ox, oy)) in enumerate(OFFSETS.items()):
        own = cov[ri]
        excl[r] = ((own > 0) & (total_cov == 1)).float()
        ovlp[r] = ((own > 0) & (total_cov == 2)).float()
    ovlp_union = (total_cov == 2).float()
    excl = {r: m.to(config.DEVICE) for r, m in excl.items()}
    ovlp = {r: m.to(config.DEVICE) for r, m in ovlp.items()}
    return excl, ovlp, ovlp_union.to(config.DEVICE)

def train_canvas_batch(region, imgs):
    cx, cy = CENTS[region]
    b = imgs.shape[0]
    big = torch.zeros(b, config.N, config.N, device=config.DEVICE)
    ox, oy = slot(region)
    for i in range(b):
        sc = pad_input_to_canvas(imgs[i], config.SUB_SIZE, upsample=UPSAMPLE,
                                 centroid=(cx, cy), noise_std=NOISE)
        big[i, oy:oy + config.SUB_SIZE, ox:ox + config.SUB_SIZE] = sc
    return big

def test_canvas(region, img):
    cx, cy = CENTS[region]
    sc = pad_input_to_canvas(img, config.SUB_SIZE, upsample=UPSAMPLE,
                             centroid=(cx, cy), noise_std=NOISE)
    if sc.max() > 0:
        sc = sc / sc.max()
    ox, oy = slot(region)
    big = torch.zeros(config.N, config.N, device=config.DEVICE)
    big[oy:oy + config.SUB_SIZE, ox:ox + config.SUB_SIZE] = sc.to(config.DEVICE)
    return big.unsqueeze(0).to(torch.complex64)

def test_full(full, ds_test, pos, neg, n_per_class=100):
    full.eval()
    accs = {}
    with torch.no_grad():
        for r in REGIONS:
            ox, oy = slot(r)
            mp = pos[r][:, oy:oy + config.SUB_SIZE, ox:ox + config.SUB_SIZE]
            mn = neg[r][:, oy:oy + config.SUB_SIZE, ox:ox + config.SUB_SIZE]
            idx = _get_balanced_indices(ds_test[r], n_per_class=n_per_class)
            c = t = 0
            for i in idx:
                img, lab = ds_test[r][i]
                if hasattr(img, "squeeze") and img.dim() == 3:
                    img = img.squeeze(0)
                x = test_canvas(r, img)
                out = full(x)
                I = torch.abs(out) ** 2
                Ic = I[0, oy:oy + config.SUB_SIZE, ox:ox + config.SUB_SIZE]
                ep = [(Ic * mp[c_]).sum().item() for c_ in range(config.N_CLASSES)]
                en = [(Ic * mn[c_]).sum().item() for c_ in range(config.N_CLASSES)]
                pred = max(range(config.N_CLASSES),
                           key=lambda c_: ep[c_] - en[c_])
                if pred == int(lab) % 10:
                    c += 1
                t += 1
            accs[r] = c / max(t, 1)
    full.train()
    return accs

def forward_loss_region(full, region, ds, pos, neg, batch_size,
                        alpha_bg=ALPHA_BG):
    imgs, labels, _ = _sample_batch(ds[region], batch_size, config.DEVICE)
    imgs = losses.normalize_input_to_canvas_energy(
        imgs, target_l2_sq=config.CANVAS_L2_SQ)
    canvases = train_canvas_batch(region, imgs).to(torch.complex64)
    out = full(canvases)
    return losses.target_loss(out_field=out, labels=labels,
                               mask_pos=pos[region], mask_neg=neg[region],
                               ce_scale=CE_SCALE, alpha_ce=ALPHA_CE,
                               alpha_anti_neg=ALPHA_ANTI, alpha_bg=alpha_bg)

def build_init(init_kind, n_layers, tag="v2"):
    if init_kind == "merged":
        subs = {}
        for r in REGIONS:
            ckpt = os.path.join(SUB_DIR, f"sub_{r}_{n_layers}layer_{tag}.pth")
            if not os.path.exists(ckpt):
                raise FileNotFoundError(
                    f"phase-1 {tag} ckpt missing: {ckpt} -- run train_stage1.py first")
            s = SubNetNL(n_layers=n_layers, region_type=r).to(config.DEVICE)
            s.load_state_dict(torch.load(ckpt, map_location=config.DEVICE))
            subs[r] = s
        return merge_subs(subs, n_layers).to(config.DEVICE)
    if init_kind == "random":
        return FullNetNL(n_layers).to(config.DEVICE)
    raise ValueError(init_kind)

def _py(o):
    if isinstance(o, dict):
        return {k: _py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_py(v) for v in o]
    if torch.is_tensor(o):
        return o.item() if o.dim() == 0 else o.detach().cpu().tolist()
    return o

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
    ap.add_argument("--init", choices=["merged", "random"], required=True)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--schedule", choices=["stretched", "anchored"],
                    default="anchored")
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--alpha-bg", type=float, default=1e4)
    ap.add_argument("--tag", default="v2")
    ap.add_argument("--sub-tag", default=None,
                   help="phase-1 sub-net tag for merged init (default: --tag)")
    ap.add_argument("--curri-interval", type=int, default=100)
    ap.add_argument("--dyn-penalty", type=float, default=0.30)
    ap.add_argument("--no-curriculum", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sub-dir", default=None,
                    help="directory holding the phase-1 sub-net checkpoints")
    ap.add_argument("--out-dir", default=None,
                    help="directory for curve/ckpt output (default: "
                         "results)")
    args = ap.parse_args()
    sub_tag = args.sub_tag if args.sub_tag else args.tag
    global SUB_DIR, OUT_DIR
    if args.sub_dir:
        SUB_DIR = os.path.abspath(args.sub_dir)
    if args.out_dir:
        OUT_DIR = os.path.abspath(args.out_dir)

    use_curri = not args.no_curriculum
    tag = f"p2_{args.epochs}ep_{args.layers}l_{args.init}_{args.schedule}"
    if args.tag != "v2":
        tag = f"{tag}_{args.tag}"
    if args.seed != 0:
        tag = f"{tag}_s{args.seed}"
    os.makedirs(OUT_DIR, exist_ok=True)
    curve_path = os.path.join(OUT_DIR, f"curve_{tag}.json")

    _set_seed(args.seed)
    torch.manual_seed(args.seed)
    full = build_init(args.init, args.layers, sub_tag).to(config.DEVICE)
    n_params = sum(p.numel() for p in full.parameters())
    excl, ovlp, ovlp_union = build_grad_masks()
    n_excl = {r: int(excl[r].sum()) for r in REGIONS}
    n_ovlp = {r: int(ovlp[r].sum()) for r in REGIONS}
    print(f"init={args.init}  schedule={args.schedule}  "
          f"alpha_bg={args.alpha_bg:g}  params={n_params}  "
          f"curriculum={use_curri} (interval={args.curri_interval}, "
          f"penalty={args.dyn_penalty:g})", flush=True)
    print(f"grad-masks: exclusive px per region = {n_excl}", flush=True)
    print(f"grad-masks: overlap px per region   = {n_ovlp}, "
          f"union={int(ovlp_union.sum())}", flush=True)

    pos, neg = build_masks300()
    ds_train = data.load_datasets(train=True)
    ds_test = data.load_datasets(train=False)

    points = []
    last_accs = None

    def record(epoch, grad_stats=None, weights=None):
        nonlocal last_accs
        t_eval = time.time()
        accs = test_full(full, ds_test, pos, neg)
        last_accs = accs
        avg = sum(accs.values()) / len(accs)
        pt = {"epoch": epoch, "accs": accs, "avg": avg}
        if grad_stats is not None:
            pt["grad"] = grad_stats
        if weights is not None:
            pt["region_weights"] = weights
        points.append(_py(pt))
        with open(curve_path, "w") as f:
            json.dump({"tag": tag, "init": args.init, "epochs": args.epochs,
                       "eval_every": args.eval_every,
                       "curriculum": use_curri,
                       "curri_interval": args.curri_interval,
                       "dyn_penalty": args.dyn_penalty,
                       "points": points},
                      f, indent=2)
        print(f"  [eval ep {epoch}] " +
              "  ".join(f"{r}={accs[r]*100:.2f}%" for r in REGIONS) +
              f"  AVG={avg*100:.2f}%" +
              (f"  W=" + " ".join(f"{r}:{weights[r]:.3f}" for r in REGIONS)
               if weights else "") +
              f"  [{time.time() - t_eval:.0f}s]", flush=True)

    record(0)

    weights = {r: 1.0 / len(REGIONS) for r in REGIONS}
    w_tensor = torch.tensor([weights[r] for r in REGIONS])

    opt = optim.Adam(full.parameters(), lr=LR)
    sched_total = ((300 if args.schedule == "anchored"
                    else args.epochs) * len(REGIONS))
    sched = optim.lr_scheduler.LinearLR(opt, start_factor=1.0,
                                        end_factor=1e-6 / LR,
                                        total_iters=sched_total)
    region_idx = {r: i for i, r in enumerate(REGIONS)}

    t0 = time.time()
    for ep in range(args.epochs):

        if use_curri:
            pick = torch.multinomial(w_tensor, 4, replacement=True)
            epoch_regions = [REGIONS[i] for i in pick.tolist()]
        else:
            epoch_regions = list(REGIONS)

        gs = {li: {"excl": 0.0, "ovlp": 0.0, "n": 0}
              for li in range(full.n_layers)}
        ov_sum_g = [torch.zeros_like(ovlp_union) for _ in range(full.n_layers)]
        ov_sum_abs = [torch.zeros_like(ovlp_union) for _ in range(full.n_layers)]
        n_steps = {r: 0 for r in REGIONS}

        ep_loss = {}
        for r in epoch_regions:
            ri = region_idx[r]
            L = forward_loss_region(full, r, ds_train, pos, neg, BATCH,
                                    args.alpha_bg)
            if torch.is_complex(L):
                L = L.real
            if torch.isnan(L) or torch.isinf(L):
                ep_loss[r] = float("nan")
                continue
            ep_loss[r] = float(L.item())
            L.backward()

            with torch.no_grad():
                for li, layer in enumerate(full.layers):
                    p = layer.phase
                    if p.grad is None:
                        continue
                    g = p.grad
                    ga = g.abs()
                    e_den = excl[r].sum()
                    o_den = ovlp[r].sum()
                    if e_den > 0:
                        gs[li]["excl"] += (ga * excl[r]).sum().item() / e_den
                    if o_den > 0:
                        gs[li]["ovlp"] += (ga * ovlp[r]).sum().item() / o_den
                    gs[li]["n"] += 1
                    ov_sum_g[li] += g * ovlp[r]
                    ov_sum_abs[li] += ga * ovlp[r]
            n_steps[r] += 1

            for layer in full.layers:
                p = layer.phase
                if p.grad is None:
                    continue
                g = p.grad * full.region_mask[ri]
                p.grad.copy_(g)
            gn = torch.nn.utils.clip_grad_norm_(full.parameters(), 1.0)
            if torch.isfinite(gn):
                opt.step()
            sched.step()
            opt.zero_grad()

        finite = [v for v in ep_loss.values() if v == v]
        if (ep + 1) % 10 == 0 or ep == args.epochs - 1:
            avg_l = sum(finite) / max(1, len(finite))
            steps_line = " ".join(f"{r}:{n_steps[r]}" for r in REGIONS)
            print(f"  ep {ep+1:4d}/{args.epochs}: avg L={avg_l:.0f} "
                  f"lr={opt.param_groups[0]['lr']:.2e} steps[{steps_line}] "
                  f"[{time.time() - t0:.0f}s]", flush=True)
        if (ep + 1) % 100 == 0 and (ep + 1) < args.epochs:
            ck = os.path.join(CKPT_DIR, f"full_{tag}_ep{ep+1:04d}.pth")
            torch.save(full.state_dict(), ck)

        grad_stats = {}
        for li in range(full.n_layers):
            d = gs[li]
            n = max(d["n"], 1)
            cancel = ((ov_sum_g[li].abs() / (ov_sum_abs[li] + 1e-12))
                      * (ov_sum_abs[li] > 0)).sum().item()
            cancel = cancel / max(int((ov_sum_abs[li] > 0).sum()), 1)
            grad_stats[str(li)] = {
                "excl_mean": d["excl"] / n,
                "ovlp_mean": d["ovlp"] / n,
                "ovlp_excl_ratio": (d["ovlp"] / n) / max(d["excl"] / n, 1e-12),
                "cancel": cancel,
            }
        if (ep + 1) % args.eval_every == 0 and (ep + 1) < args.epochs:

            rebalanced = None
            record(ep + 1, grad_stats=grad_stats, weights=weights)
            if (use_curri and last_accs is not None
                    and (ep + 1) % args.curri_interval == 0):
                weights, easy = rebalance_weights(
                    weights, last_accs, args.dyn_penalty)
                w_tensor = torch.tensor([weights[r] for r in REGIONS])
                rebalanced = easy
                print(f"  [curriculum @ ep {ep + 1}] easiest={rebalanced} "
                      f"-> weights=" +
                      " ".join(f"{r}:{weights[r]:.3f}" for r in REGIONS),
                      flush=True)

                points[-1]["region_weights_after"] = weights
                with open(curve_path, "w") as f:
                    json.dump({"tag": tag, "init": args.init,
                               "epochs": args.epochs,
                               "eval_every": args.eval_every,
                               "curriculum": use_curri,
                               "curri_interval": args.curri_interval,
                               "dyn_penalty": args.dyn_penalty,
                               "points": points},
                              f, indent=2)

    os.makedirs(CKPT_DIR, exist_ok=True)
    final_ckpt = os.path.join(CKPT_DIR, f"full_{tag}_ep{args.epochs:04d}.pth")
    torch.save(full.state_dict(), final_ckpt)
    record(args.epochs, grad_stats=grad_stats, weights=weights)

    last = points[-1]
    summary = {"tag": tag, "init": args.init, "epochs": args.epochs,
               "final_ckpt": final_ckpt,
               "final_accs": last["accs"], "final_avg": last["avg"],
               "final_weights": weights}
    with open(os.path.join(OUT_DIR, f"phase2_{tag}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nDone. final avg = {last['avg']*100:.2f}%  "
          f"final weights = {weights}", flush=True)

if __name__ == "__main__":
    main()
