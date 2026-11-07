import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.optim as optim

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))

import train_stage2 as base
from d2nn import config
from d2nn import data
from d2nn import losses
from d2nn.training import _sample_batch
from d2nn.training import _set_seed
from d2nn.stage_utils import SubNetNL
from d2nn.stage_utils import pad_input_to_canvas
from d2nn.layers import _get_quarter_circle_mask

from d2nn.readout import _get_balanced_indices

REG = list(config.REGIONS)
NC = config.N_CLASSES
DEV = config.DEVICE
NOISE = 0.01
LR, BATCH = base.LR, base.BATCH
RES = os.path.join(HERE, "results")
CKPT_DIR = os.path.join(HERE, "checkpoints")

def load_subs(regions, n_layers, tag):
    subs = {}
    for r in regions:
        ckpt = os.path.join(base.SUB_DIR, f"sub_{r}_{n_layers}layer_{tag}.pth")
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"phase-1 {tag} ckpt missing: {ckpt}")
        s = SubNetNL(n_layers=n_layers, region_type=r).to(DEV)
        s.load_state_dict(torch.load(ckpt, map_location=DEV))
        subs[r] = s
    return subs

def merge_subset(subs, regions, n_layers):
    full = base.FullNetNL(n_layers).to(DEV)
    for li in range(n_layers):
        canvas = torch.zeros((config.N, config.N), dtype=torch.float32,
                             device=DEV)
        count = torch.zeros_like(canvas)
        for r in regions:
            ox, oy = base.OFFSETS[r]
            canvas[oy:oy + config.SUB_SIZE,
                   ox:ox + config.SUB_SIZE] += subs[r].get_spatial_phase(
                       li).to(DEV)
            count[oy:oy + config.SUB_SIZE, ox:ox + config.SUB_SIZE] += \
                _get_quarter_circle_mask(r, config.SUB_SIZE, DEV)
        with torch.no_grad():
            full.layers[li].phase.copy_(canvas / torch.clamp(count, min=1.0))
    return full

def blank_window(region):
    cx, cy = base.CENTS[region]
    ox, oy = base.OFFSETS[region]
    sc = pad_input_to_canvas(torch.zeros(config.INPUT_SIZE, config.INPUT_SIZE),
                             config.SUB_SIZE, upsample=1, centroid=(cx, cy),
                             noise_std=NOISE)
    big = torch.zeros(config.N, config.N, device=DEV)
    big[oy:oy + config.SUB_SIZE, ox:ox + config.SUB_SIZE] = sc.to(DEV)
    return big

def scores_and_power(out, pos, neg):
    I = (out.abs() ** 2).float()
    If = I.reshape(I.shape[0], -1)
    P = If.sum(1) + 1e-12
    pf = pos.reshape(NC, -1)
    nf = neg.reshape(NC, -1)
    return If @ pf.T - If @ nf.T, P

def gate_loss(s, P, labels, tth, marg, own):
    lo, hi = (tth - marg) * P, (tth + marg) * P
    pen_all = torch.relu(s - lo[:, None])
    if own is None:
        return pen_all.mean()
    keep = pen_all.clone()
    pos_idx = torch.nonzero(labels >= 0).squeeze(1)
    if pos_idx.numel():

        keep[pos_idx, labels[pos_idx]] = 0.0
        need = torch.relu(hi[pos_idx] - s[pos_idx, labels[pos_idx]])
        return keep.mean() + need.mean()
    return keep.mean()

def margin_loss(out, pos, neg, labels, mg, mg_neg):
    s, P = scores_and_power(out, pos, neg)
    sn = s / P[:, None]
    top2 = sn.topk(2, dim=1).values
    m = top2[:, 0] - top2[:, 1]
    if labels is None:
        return torch.relu(m - mg_neg).mean()
    return torch.relu(mg - m).mean() + torch.relu(m - mg_neg).mean()

def own_images(ds, region, b):
    imgs, labels, _ = _sample_batch(ds[region], b, DEV)
    imgs = losses.normalize_input_to_canvas_energy(
        imgs, target_l2_sq=config.CANVAS_L2_SQ)
    return imgs, labels

def subset_for(r, a):
    if not a.concurrent:
        return [r]
    k = int(np.random.choice([1, 2, 3, 4], p=a.k_probs))
    k = max(1, min(k, len(REG)))
    others = [q for q in REG if q != r]
    np.random.shuffle(others)
    return [r] + others[:k - 1]

def lit_batch(region, subset, ds, b, dev):
    big = torch.zeros(b, config.N, config.N, device=dev)
    labels = None
    all_labels = {}
    for q in subset:
        imgs, lab = own_images(ds, q, b)
        big = big + base.train_canvas_batch(q, imgs)
        all_labels[q] = lab
        if q == region:
            labels = lab
    return big, labels, all_labels

def negative_batch(region, subset, ds, b, dev, blank_share):
    k_for = int(round(b * (1.0 - blank_share)))
    k_bla = b - k_for
    big = torch.zeros(b, config.N, config.N, device=dev)
    others = [q for q in subset if q != region]
    for q in others:
        imgs, _ = own_images(ds, q, b)
        big = big + base.train_canvas_batch(q, imgs)
    rows = []
    if k_for > 0:
        src = [q for q in REG if q != region]
        pick = src[int(np.random.randint(len(src)))]
        imgs, _ = own_images(ds, pick, k_for)
        rows.append(base.train_canvas_batch(region, imgs))
    if k_bla > 0:
        rows.append(torch.stack([blank_window(region)] * k_bla).to(dev))
    for r_ in rows:
        big[:r_.shape[0]] = big[:r_.shape[0]] + r_
    return big, k_for, k_bla

def step_one_region(full, region, ds, pos, neg, opt, a, stats):
    subset = subset_for(region, a)
    cv, labels, all_labels = lit_batch(region, subset, ds, a.batch_own, DEV)
    out = full(cv.to(torch.complex64))
    targets = subset if a.multi_target else [region]
    total = torch.zeros((), device=DEV)
    for q in targets:
        total = total + losses.target_loss(
            out_field=out, labels=all_labels[q],
            mask_pos=pos[q], mask_neg=neg[q],
            ce_scale=base.CE_SCALE, alpha_ce=base.ALPHA_CE,
            alpha_anti_neg=base.ALPHA_ANTI, alpha_bg=a.alpha_bg)
    labels = all_labels[region]
    stats["ce"] = float(total.detach())
    if a.gate_weight > 0:
        ncv, k_for, k_bla = negative_batch(region, subset, ds, a.batch_neg,
                                          DEV, a.blank_share)
        nout = full(ncv.to(torch.complex64))
        if a.gate_mode == "margin":
            g_own = float(margin_loss(out, pos[region], neg[region], labels,
                                      a.mg, a.mg_neg).detach())
            g_neg = float(margin_loss(nout, pos[region], neg[region], None,
                                      a.mg, a.mg_neg).detach())
        else:
            s, P = scores_and_power(out, pos[region], neg[region])
            g_own = float(gate_loss(s, P, labels, a.tth, a.marg,
                                    own=True).detach())
            sn, Pn = scores_and_power(nout, pos[region], neg[region])
            labs = torch.full((sn.shape[0],), -1, dtype=torch.long)
            g_neg = float(gate_loss(sn, Pn, labs, a.tth, a.marg,
                                    own=None).detach())
        total = total + a.gate_weight * (g_own + g_neg)
        stats["gate_own"], stats["gate_neg"] = g_own, g_neg
        stats["n_for"], stats["n_bla"] = k_for, k_bla
    if torch.is_complex(total):
        total = total.real
    if not torch.isfinite(total):
        return float("nan")
    opt.zero_grad()
    total.backward()

    keep = full.region_mask[[REG.index(q) for q in targets]].amax(0)
    for layer in full.layers:
        if layer.phase.grad is not None:
            layer.phase.grad.copy_(layer.phase.grad * keep)
    gn = torch.nn.utils.clip_grad_norm_(full.parameters(), 1.0)
    if torch.isfinite(gn):
        opt.step()
    stats["gnorm"] = float(gn)
    return float(total.detach())

@torch.no_grad()
def evaluate(full, ds_test, pos, neg, a, n_per_class=100):
    out = {"acc": {}, "tpr_own": {}, "far_foreign": {}, "far_blank": {},
           "score_own": {}, "score_foreign": {}, "score_blank": {},
           "margin_own_p50": {}, "margin_foreign_p99": {},
           "tpr_mg_own": {}, "far_mg_foreign": {}}
    for r in REG:
        idx = _get_balanced_indices(ds_test[r], n_per_class=n_per_class)
        ok = 0
        seen = 0
        svals = []
        mvals = []
        for i in idx:
            img, lab = ds_test[r][i]
            lab = int(lab) % NC
            if img.dim() == 3:
                img = img.squeeze(0)
            cv = base.test_canvas(r, img).to(torch.complex64)
            s, P = scores_and_power(full(cv), pos[r], neg[r])
            sn = (s / P)[0]
            svals.append(float(sn.max()))
            top2 = sn.sort(descending=True).values
            mvals.append(float(top2[0] - top2[1]))
            ok += int(int(sn.argmax()) == lab)
            seen += int(float(sn.max()) >= a.tth)
        out["acc"][r] = ok / len(idx)
        out["tpr_own"][r] = seen / len(idx)
        out["score_own"][r] = float(np.mean(svals))
        out["margin_own_p50"][r] = float(np.median(mvals))
        out["tpr_mg_own"][r] = float(np.mean(np.array(mvals) >= a.mg))

        ftrig, fn, fmarg = 0, 0, []
        for q in REG:
            if q == r:
                continue
            sub = _get_balanced_indices(ds_test[q], n_per_class=20)[:60]
            for i in sub:
                img, _ = ds_test[q][i]
                if img.dim() == 3:
                    img = img.squeeze(0)
                cv = base.test_canvas(r, img).to(torch.complex64)
                s, P = scores_and_power(full(cv), pos[r], neg[r])
                sn2 = (s / P)[0]
                t2 = sn2.sort(descending=True).values
                fmarg.append(float(t2[0] - t2[1]))
                fn += 1
                ftrig += int(float((s / P)[0].max()) >= a.tth)
        out["far_foreign"][r] = ftrig / max(fn, 1)
        if fmarg:
            out["margin_foreign_p99"][r] = float(np.percentile(fmarg, 99))
            out["far_mg_foreign"][r] = float(np.mean(np.array(fmarg) >= a.mg))
        btrig = 0
        for _ in range(200):
            cv = blank_window(r).unsqueeze(0).to(torch.complex64)
            s, P = scores_and_power(full(cv), pos[r], neg[r])
            btrig += int(float((s / P)[0].max()) >= a.tth)
        out["far_blank"][r] = btrig / 200
    out["mean_acc"] = float(np.mean(list(out["acc"].values())))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="gate",
                    choices=("baseline", "gate", "concurrent", "gate_concurrent"))
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha-bg", type=float, default=1e3)
    ap.add_argument("--tth", type=float, default=0.2)
    ap.add_argument("--marg", type=float, default=0.05)
    ap.add_argument("--gate-mode", choices=("score", "margin"), default="score")
    ap.add_argument("--mg", type=float, default=0.02,
                    help="margin hinge threshold (margin mode only)")
    ap.add_argument("--mg-neg", type=float, default=0.005,
                    help="largest margin a foreign object or blank aperture may "
                         "show (margin mode only)")
    ap.add_argument("--gate-weight", type=float, default=1.0,
                    help="the hinge is written in power units, so a weight of "
                         "1 makes it directly comparable with the "
                         "classification term (typically 1e3 at init)")
    ap.add_argument("--batch-own", type=int, default=BATCH)
    ap.add_argument("--batch-neg", type=int, default=64)
    ap.add_argument("--blank-share", type=float, default=0.5,
                    help="share of the negative probes that are blank windows")
    ap.add_argument("--k-probs", type=float, nargs=4, default=[0.25, 0.3, 0.25, 0.2],
                    help="probability of illuminating 1, 2, 3, 4 regions")
    ap.add_argument("--lr-anchor", type=int, default=300,
                    help="the linear LR decay reaches its floor after this many "
                         "epochs; set it equal to --epochs for a full-length "
                         "schedule (the stage-2 default is 300)")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--sub-tag", default="bg1e3",
                    help="phase-1 tag used for the merged initialisation")
    ap.add_argument("--out-dir", default=RES)
    ap.add_argument("--ckpt-dir", default=CKPT_DIR)
    ap.add_argument("--name-suffix", default="",
                    help="appended to every output name so several settings of "
                         "the same arm can coexist")
    ap.add_argument("--multi-target", action="store_true",
                    help="with --concurrent: supervise every lit region in the "
                         "same forward pass, instead of one target plus "
                         "distractors")
    ap.add_argument("--regions", nargs="+", default=None,
                    help="subset of the four quarter-circle tasks to train, e.g."
                         " --regions NW NE SW for a three-task device; the merged"
                         " initialisation then leaves the omitted aperture"
                         " transparent")
    ap.add_argument("--curriculum", action="store_true",
                    help="stage-2 curriculum: every --curri-interval epochs "
                         "penalise the easiest region's sampling weight, and "
                         "draw the epoch's four regions by multinomial")
    ap.add_argument("--curri-interval", type=int, default=100)
    ap.add_argument("--dyn-penalty", type=float, default=0.30)
    a = ap.parse_args()

    use_gate = a.arm in ("gate", "gate_concurrent")
    a.gate_weight = a.gate_weight if use_gate else 0.0
    a.concurrent = a.arm in ("concurrent", "gate_concurrent")
    if a.multi_target and not a.concurrent:
        print("--multi-target needs --concurrent; turning concurrency on")
        a.concurrent = True
    a.neg_frac = 1.0 if use_gate else 0.0
    a.blank_frac = a.neg_frac

    a.k_probs = np.array(a.k_probs, dtype=float)
    a.k_probs = a.k_probs / a.k_probs.sum()
    os.makedirs(a.out_dir, exist_ok=True)
    os.makedirs(a.ckpt_dir, exist_ok=True)
    name = a.arm + a.name_suffix
    if a.gate_mode == "margin":
        name += "_margin"
    if a.multi_target:
        name += "_multi"
    if a.curriculum:
        name += "_curri"
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    _set_seed(a.seed)

    if a.regions:
        bad = [r for r in a.regions if r not in config.REGIONS]
        if bad:
            raise SystemExit(f"unknown regions: {bad}")
        REG[:] = [r for r in config.REGIONS if r in a.regions]
        full = merge_subset(load_subs(REG, a.layers, a.sub_tag), REG,
                            a.layers).to(DEV)
        print("three-task device: regions=%s, merged from %d sub-networks"
              % (",".join(REG), len(REG)), flush=True)
    else:
        full = base.build_init("merged", a.layers, a.sub_tag).to(DEV)
    pos, neg = base.build_masks300()
    ds_train = data.load_datasets(train=True)
    ds_test = data.load_datasets(train=False)
    opt = optim.Adam(full.parameters(), lr=LR)
    sched = optim.lr_scheduler.LinearLR(opt, start_factor=1.0,
                                        end_factor=1e-6 / LR,
                                        total_iters=a.lr_anchor * len(REG))
    stats = {}
    curve = []
    curri_w = {r: 1.0 / len(REG) for r in REG}
    w_tensor = torch.tensor([curri_w[r] for r in REG], dtype=torch.float32)
    print("arm=%s epochs=%d tth=%g marg=%g gate_weight=%g concurrent=%s "
          "lr_anchor=%d params=%d" % (a.arm, a.epochs, a.tth, a.marg,
                                      a.gate_weight, a.concurrent, a.lr_anchor,
                                      sum(p.numel() for p in full.parameters())),
          flush=True)
    t0 = time.time()
    for ep in range(a.epochs):
        per_region = []
        if a.curriculum:
            order = [REG[i] for i in torch.multinomial(
                w_tensor, len(REG), replacement=True).tolist()]
        else:
            order = list(REG)
        for r in order:
            per_region.append(step_one_region(full, r, ds_train, pos, neg, opt, a,
                                          stats))
            sched.step()
            opt.zero_grad()
        L = float(np.mean([x for x in per_region if x == x] or [float("nan")]))
        if (ep + 1) % a.eval_every == 0 or ep == a.epochs - 1:
            full.eval()
            ev = evaluate(full, ds_test, pos, neg, a, n_per_class=20)
            full.train()
            rec = {"epoch": ep + 1, "loss": L, "parts": dict(stats),
                   "acc": ev["acc"], "mean_acc": ev["mean_acc"],
                   "tpr_own": ev["tpr_own"], "far_foreign": ev["far_foreign"],
                   "far_blank": ev["far_blank"], "score_own": ev["score_own"]}
            curve.append(rec)
            if (a.curriculum and (ep + 1) % a.curri_interval == 0
                    and ep + 1 < a.epochs):
                curri_w, easy = base.rebalance_weights(curri_w, ev["acc"],
                                                       a.dyn_penalty)
                w_tensor = torch.tensor([curri_w[r] for r in REG],
                                        dtype=torch.float32)
                rec["curriculum"] = dict(curri_w)
                print("    curriculum: easiest %s -> weights %s"
                      % (easy, {k: round(v, 3) for k, v in curri_w.items()}),
                      flush=True)
            print("  ep %4d L=%.0f ce=%.0f gate_own=%.4f gate_neg=%.4f "
                  "acc=%.1f%% tpr=%.2f far_for=%.2f far_blank=%.3f "
                  "score_own=%s | margin p50=%s p99for=%s TPR@mg=%.2f FAR@mg=%.3f "
                  "lr=%.1e [%.0fs]"
                  % (ep + 1, L if L == L else -1,
                     stats.get("ce", float("nan")),
                     stats.get("gate_own", float("nan")),
                     stats.get("gate_neg", float("nan")),
                     100 * ev["mean_acc"],
                     float(np.mean(list(ev["tpr_own"].values()))),
                     float(np.mean(list(ev["far_foreign"].values()))),
                     float(np.mean(list(ev["far_blank"].values()))),
                     " ".join("%.4f" % ev["score_own"][r] for r in REG),
                     " ".join("%.4f" % ev["margin_own_p50"][r] for r in REG),
                     " ".join("%.4f" % (ev["margin_foreign_p99"].get(r, float("nan")))
                              for r in REG),
                     float(np.mean(list(ev["tpr_mg_own"].values()))),
                     float(np.mean([ev["far_mg_foreign"].get(r, float("nan"))
                                    for r in REG])),
                     opt.param_groups[0]["lr"],
                     time.time() - t0), flush=True)
            torch.save(full.state_dict(),
                       os.path.join(a.ckpt_dir, "ckpt_%s_s%d_ep%04d.pth"
                                    % (name, a.seed, ep + 1)))
    json.dump({"arm": a.arm, "seed": a.seed, "tth": a.tth, "marg": a.marg,
               "name": name, "lr_anchor": a.lr_anchor,
               "gate_weight": a.gate_weight, "concurrent": a.concurrent,
               "epochs": a.epochs, "alpha_bg": a.alpha_bg, "curve": curve},
              open(os.path.join(a.out_dir, "train_%s_s%d.json"
                               % (name, a.seed)), "w"), indent=1)
    print("done in %.0f s" % (time.time() - t0), flush=True)

if __name__ == "__main__":
    main()
