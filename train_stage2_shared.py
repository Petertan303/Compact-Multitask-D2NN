"""Finished device: classification and anti-crosstalk trained together.

Per epoch: four passes, one per focus region.  Each pass draws one batch in
which every corner holds one sample, the focus corner holding the dataset that
belongs to it and the other three corners holding foreign datasets (cyclic
assignment).  The loss is

    L = L_target(focus) + W_INTER * <E_ring4 / E_tot>

    L_target = alpha_ce * CE(logits, y)
               + alpha_anti_neg * <E^-(y) / E_tot>
               + alpha_bg     * <E_bg / E_tot>

L_target is measured on the correctly placed samples only.  The defaults are
the weights the reported device was trained with: alpha_ce 300, gamma 300,
alpha_anti_neg 40, alpha_bg 1e3 and W_INTER 1e3.  ring4 is the
four-corner union of the 20-spot rings built with the enlarged thresholds
(pos 0.05, neg 0.01 of the gaussian peak, radius ~11-14 px), and it is
penalised on the misplaced samples only.  Evaluation keeps the un-enlarged
spot masks (>0.5) so the reported numbers stay comparable with every other
device.

Why these numbers: every pass ends with clip_grad_norm_(parameters, 1.0), so
what matters is the share of the unit step each term takes, not the loss
values.  Measured shares are 52% classification and 38% confinement at
alpha_ce 30, 79% and 17% at 100, and 92% and 7% at 300.  Letting the two
conflicting gradients split the step evenly stalls both; giving classification
the larger share improves accuracy and confinement at the same time.  With
these defaults the device reaches 91.1% mean accuracy on the test split
(95.5, 81.4, 87.6 and 100.0 per task, three seeds, standard deviation 0.06)
and 91.6% with all four sub-apertures illuminated at once.

The classification logits are the normalised readout s_c = S_c / E_tot, where
S_c = E^+_c - E^-_c is the differential power on the small spots; --ce-mode
raw restores the mainline convention of using S_c itself.  With the input
energy normalised to SUB_SIZE^2 the raw logits reach thousands and the
softmax saturates, which is what made the classification term die after the
first few hundred epochs of every earlier run.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

HERE = os.path.dirname(os.path.abspath(__file__))

import train_stage2 as t2
from d2nn import config
from d2nn import data
from d2nn import losses
from d2nn.readout import _get_balanced_indices
from d2nn.training import _make_differential_gallery
from d2nn.training import _sample_batch

DEV = config.DEVICE
REGIONS = list(config.REGIONS)
WRONG_FOR_CORNER = {"NW": "NE", "NE": "SW", "SW": "SE", "SE": "NW"}
N_LAYERS = 3
SUB_TAG = "bg1e3"

W_INTER = 1000.0
ALPHA_CE = 300.0
ALPHA_ANTI_NEG = 40.0
ALPHA_BG = 1e3
CE_SCALE_NORM = 3e2
CE_SCALE_RAW = 1e2
BATCH_OWN = 64
BATCH_WRONG = 32
LR = 0.1
POS_TH = 0.05
NEG_TH = 0.01


def build_big_rings():
    lp = config.FULL_SIZE * config.SPREAD_SCALE * config.LIMIT_SCALE
    ln = lp * 1.4
    sg = config.FULL_SIZE * config.SIGMA_SCALE
    rings = {}
    for r in REGIONS:
        cx, cy = t2.CENTS[r]
        gp, gn = _make_differential_gallery(cx, cy, config.SUB_SIZE, config.DX,
                                            lp, ln, sg, config.N_CLASSES, DEV)
        mp = (gp > POS_TH).float()
        mn = (gn > NEG_TH).float()
        one = ((mp.sum(dim=0) + mn.sum(dim=0)) > 1e-2).float()
        ox, oy = t2.slot(r)
        big = torch.zeros(config.N, config.N, device=DEV)
        big[oy:oy + config.SUB_SIZE, ox:ox + config.SUB_SIZE] = one
        rings[r] = big
    ring4 = torch.zeros(config.N, config.N, device=DEV)
    for r in REGIONS:
        ring4 = torch.maximum(ring4, rings[r])
    return rings, ring4


def target_parts(out_field, labels, mask_pos, mask_neg, ce_mode, ce_scale,
                 alpha_ce, alpha_anti_neg, alpha_bg):
    """L_target and its three terms, returned separately for the log."""
    I = torch.abs(out_field) ** 2
    tot = I.flatten(1).sum(dim=1) + 1e-8
    E_pos = torch.stack([(I * mask_pos[k]).flatten(1).sum(dim=1)
                         for k in range(mask_pos.shape[0])], dim=1)
    E_neg = torch.stack([(I * mask_neg[k]).flatten(1).sum(dim=1)
                         for k in range(mask_neg.shape[0])], dim=1)
    S = E_pos - E_neg
    logits = S / tot.unsqueeze(1) if ce_mode == "normalized" else S
    L_ce = F.cross_entropy(ce_scale * logits, labels)
    L_neg = ((I * mask_neg[labels]).flatten(1).sum(dim=1) / tot).mean()
    bg_mask = torch.clamp(1.0 - mask_pos[labels] - mask_neg[labels], min=0.0)
    L_bg = ((I * bg_mask).flatten(1).sum(dim=1) / tot).mean()
    loss = alpha_ce * L_ce + alpha_anti_neg * L_neg + alpha_bg * L_bg
    return loss, float(L_ce.detach()), float(L_neg.detach()), float(L_bg.detach())


def _img(ds, j):
    im = ds[j][0]
    if hasattr(im, "squeeze") and im.dim() == 3:
        im = im.squeeze(0)
    return im


def mixed_canvas(ds_train, focus):
    rows, labels = [], None
    imgs, lab, _ = _sample_batch(ds_train[focus], BATCH_OWN, DEV)
    imgs = losses.normalize_input_to_canvas_energy(imgs,
                                                   target_l2_sq=config.CANVAS_L2_SQ)
    rows.append(t2.train_canvas_batch(focus, imgs))
    labels = lab
    for corner in REGIONS:
        if corner == focus:
            continue
        ds_name = WRONG_FOR_CORNER[corner]
        oimg, _, _ = _sample_batch(ds_train[ds_name], BATCH_WRONG, DEV)
        oimg = losses.normalize_input_to_canvas_energy(
            oimg, target_l2_sq=config.CANVAS_L2_SQ)
        rows.append(t2.train_canvas_batch(corner, oimg))
    return torch.cat(rows, dim=0), labels


def blank_canvas(region):
    """Noise-only window: the "nothing was placed here" probe."""
    from d2nn.stage_utils import pad_input_to_canvas
    cx, cy = t2.CENTS[region]
    ox, oy = t2.slot(region)
    sc = pad_input_to_canvas(torch.zeros(config.INPUT_SIZE, config.INPUT_SIZE),
                             config.SUB_SIZE, upsample=1, centroid=(cx, cy),
                             noise_std=0.01)
    big = torch.zeros(config.N, config.N, device=DEV)
    big[oy:oy + config.SUB_SIZE, ox:ox + config.SUB_SIZE] = sc.to(DEV)
    return big


@torch.no_grad()
def evaluate(full, ds_test, pos, neg, ring4, core4, n_per_class=20):
    full.eval()
    out = {"correct": {}, "wrong": {}}
    for r in REGIONS:
        ds = ds_test[r]
        idx = _get_balanced_indices(ds, n_per_class=n_per_class, seed=0)
        frac_ring4, frac_core, correct, n = 0.0, 0.0, 0, 0
        shat_ok = []
        for i in range(0, len(idx), 64):
            sel = idx[i:i + 64]
            canv = torch.cat([t2.test_canvas(r, _img(ds, j)) for j in sel],
                             dim=0)
            I = torch.abs(full(canv)) ** 2
            tot = I.flatten(1).sum(dim=1) + 1e-8
            frac_ring4 += float(((I * ring4).flatten(1).sum(1) / tot).sum())
            frac_core += float(((I * core4).flatten(1).sum(1) / tot).sum())
            lab = torch.tensor([int(ds[j][1]) % config.N_CLASSES for j in sel],
                               device=DEV)
            ep = (I.unsqueeze(1) * pos[r]).flatten(start_dim=2).sum(dim=2)
            en = (I.unsqueeze(1) * neg[r]).flatten(start_dim=2).sum(dim=2)
            correct += int((torch.argmax(ep - en, dim=1) == lab).sum())
            shat_ok += list(((ep - en) / tot.unsqueeze(1)).max(dim=1).values
                            .detach().cpu().numpy())
            n += len(sel)
        out["correct"][r] = {"acc": correct / n,
                             "ring4": frac_ring4 / n, "core4": frac_core / n,
                             "shat_mean": float(np.mean(shat_ok)),
                             "shat_p05": float(np.percentile(shat_ok, 5))}

        wname = WRONG_FOR_CORNER[r]
        widx = _get_balanced_indices(ds_test[wname], n_per_class=n_per_class,
                                     seed=0)
        frac_ring4, frac_core, shat_max, shat_neg_max = 0.0, 0.0, 0.0, 0.0
        shat_bad = []
        n = 0
        for i in range(0, len(widx), 64):
            sel = widx[i:i + 64]
            canv = torch.cat([t2.test_canvas(r, _img(ds_test[wname], j))
                              for j in sel], dim=0)
            I = torch.abs(full(canv)) ** 2
            tot = I.flatten(1).sum(dim=1) + 1e-8
            frac_ring4 += float(((I * ring4).flatten(1).sum(1) / tot).sum())
            frac_core += float(((I * core4).flatten(1).sum(1) / tot).sum())
            ep = (I.unsqueeze(1) * pos[r]).flatten(start_dim=2).sum(dim=2)
            en = (I.unsqueeze(1) * neg[r]).flatten(start_dim=2).sum(dim=2)
            shat_max = max(shat_max, float(((ep - en) / tot.unsqueeze(1)).max()))
            shat_neg_max = max(shat_neg_max, float((-en / tot.unsqueeze(1)).max()))
            shat_bad += list(((ep - en) / tot.unsqueeze(1)).max(dim=1).values
                             .detach().cpu().numpy())
            n += len(sel)
        out["wrong"][r] = {"ring4": frac_ring4 / n, "core4": frac_core / n,
                           "shat_max": shat_max,
                           "shat_mean": float(np.mean(shat_bad)),
                           "shat_p99": float(np.percentile(shat_bad, 99))}
    full.train()
    out["mean_acc"] = float(np.mean([out["correct"][r]["acc"]
                                     for r in REGIONS]))
    out["mean_ring4_wrong"] = float(np.mean([out["wrong"][r]["ring4"]
                                             for r in REGIONS]))
    out["contrast_db"] = float(np.mean([
        10.0 * np.log10((out["correct"][r]["shat_p05"] + 1e-12)
                        / (out["wrong"][r]["shat_p99"] + 1e-12))
        for r in REGIONS]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=None,
                    help="checkpoint cadence; defaults to --eval-every")
    ap.add_argument("--w-inter", type=float, default=W_INTER)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--no-freeze", action="store_true",
                    help="train every pixel instead of the focus quarter disk")
    ap.add_argument("--inter-rows", choices=("all", "wrong"), default="wrong",
                    help="'all' follows v6: the interference term is measured "
                         "on every row, correct placements included. 'wrong' "
                         "restricts it to the misplaced rows, leaving the "
                         "classification rows to L_target alone")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="shared_stage2")
    ap.add_argument("--init", choices=("merged", "random"), default="merged")
    ap.add_argument("--alpha-ce", type=float, default=ALPHA_CE)
    ap.add_argument("--alpha-anti-neg", type=float, default=ALPHA_ANTI_NEG)
    ap.add_argument("--alpha-bg", type=float, default=ALPHA_BG)
    ap.add_argument("--ce-mode", choices=("normalized", "raw"),
                    default="normalized",
                    help="normalized: logits are the scale-free readout "
                         "s_c = S_c / E_tot; raw: logits are the differential "
                         "power S_c itself, as the mainline does")
    ap.add_argument("--ce-scale", type=float, default=None,
                    help="soft-max temperature gamma; default 100 for both "
                         "modes")
    ap.add_argument("--no-aperture-mask", action="store_true",
                    help="open the whole 300x300 aperture instead of the four "
                         "quarter disks")
    ap.add_argument("--init-std", type=float, default=None,
                    help="standard deviation of the random phase screen at "
                         "initialisation (mainline default 0.2 rad; a fully "
                         "developled diffuser corresponds to ~1.8 rad)")
    a = ap.parse_args()
    if a.ce_scale is None:
        a.ce_scale = (CE_SCALE_NORM if a.ce_mode == "normalized"
                      else CE_SCALE_RAW)
    if a.save_every is None:
        a.save_every = a.eval_every

    t2.SUB_DIR = os.path.join(HERE, "checkpoints")
    out_dir = os.path.join(HERE, "results")
    ckpt_dir = os.path.join(HERE, "checkpoints")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    if a.init_std is not None:
        config.PIXEL_INIT_STD = float(a.init_std)
    full = t2.build_init(a.init, N_LAYERS, SUB_TAG).to(DEV)
    print("initial phase std per layer: %s rad"
          % [round(float(l.phase.std()), 2) for l in full.layers], flush=True)
    if a.no_aperture_mask:
        full.global_mask = torch.ones_like(full.global_mask)
    pos, neg = t2.build_masks300()
    rings, ring4 = build_big_rings()
    core4 = torch.zeros(config.N, config.N, device=DEV)
    for r in REGIONS:
        core4 = torch.maximum(core4, losses.spot_ring(pos[r], neg[r]))
    print("big ring4 coverage %.1f%% of canvas;  core4 coverage %.1f%%"
          % (100 * float(ring4.sum()) / (config.N * config.N),
             100 * float(core4.sum()) / (config.N * config.N)), flush=True)
    print("per-corner big ring coverage: %s"
          % {r: round(100 * float(rings[r].sum()) / 90000, 1) for r in REGIONS},
          flush=True)

    ds_train = data.load_datasets(train=True)
    ds_test = data.load_datasets(train=False)
    opt = optim.Adam(full.parameters(), lr=a.lr)
    sched = optim.lr_scheduler.LinearLR(opt, start_factor=1.0,
                                        end_factor=1e-6 / a.lr,
                                        total_iters=a.epochs * len(REGIONS))
    curve = []
    print("epochs=%d w_inter=%g alpha_ce=%g alpha_anti_neg=%g alpha_bg=%g "
          "ce_mode=%s ce_scale=%g batch_own=%d batch_wrong=%d inter_rows=%s "
          "freeze=%s lr=%.2g"
          % (a.epochs, a.w_inter, a.alpha_ce, a.alpha_anti_neg, a.alpha_bg,
             a.ce_mode, a.ce_scale, BATCH_OWN, BATCH_WRONG, a.inter_rows,
             not a.no_freeze, a.lr), flush=True)
    t0 = time.time()
    for ep in range(a.epochs):
        tgt_sum, int_sum = 0.0, 0.0
        ce_sum, neg_sum, bg_sum = 0.0, 0.0, 0.0
        for focus in REGIONS:
            cv, labels = mixed_canvas(ds_train, focus)
            opt.zero_grad()
            out = full(cv.to(torch.complex64))
            I = torch.abs(out) ** 2
            if a.inter_rows == "wrong":
                I = I[BATCH_OWN:]
            tot = I.flatten(1).sum(dim=1) + 1e-8
            inter = ((I * ring4).flatten(1).sum(dim=1) / tot).mean()
            no_target = (a.alpha_ce == 0.0 and a.alpha_anti_neg == 0.0
                         and a.alpha_bg == 0.0)
            if no_target:
                L_target = torch.zeros((), device=DEV)
                ce_p = neg_p = bg_p = 0.0
            else:
                L_target, ce_p, neg_p, bg_p = target_parts(
                    out_field=out[:BATCH_OWN], labels=labels,
                    mask_pos=pos[focus], mask_neg=neg[focus],
                    ce_mode=a.ce_mode, ce_scale=a.ce_scale,
                    alpha_ce=a.alpha_ce, alpha_anti_neg=a.alpha_anti_neg,
                    alpha_bg=a.alpha_bg)
            L = L_target + a.w_inter * inter
            L.backward()
            if not a.no_freeze:
                keep = full.region_mask[REGIONS.index(focus)]
                for layer in full.layers:
                    if layer.phase.grad is not None:
                        layer.phase.grad.copy_(layer.phase.grad * keep)
            gn = torch.nn.utils.clip_grad_norm_(full.parameters(), 1.0)
            if torch.isfinite(gn):
                opt.step()
            sched.step()
            tgt_sum += float(L_target.detach())
            int_sum += float(inter.detach())
            ce_sum, neg_sum, bg_sum = (ce_sum + ce_p, neg_sum + neg_p,
                                       bg_sum + bg_p)
        if (ep + 1) % a.eval_every == 0 or ep + 1 == a.epochs:
            ev = evaluate(full, ds_test, pos, neg, ring4, core4)
            rec = {"epoch": ep + 1, "L_target": tgt_sum / 4,
                   "L_inter": int_sum / 4, "L_ce": ce_sum / 4,
                   "L_neg": neg_sum / 4, "L_bg": bg_sum / 4, "eval": ev}
            curve.append(rec)
            print("  ep %4d tgt=%.0f (ce=%.3f neg=%.4f bg=%.4f) inter=%.4f acc=%.1f%% "
                  "ring4_wrong=%.2f%% core4_wrong=%.2f%% "
                  "ring4_own=%s shat_for_max=%s contrast=%.1f dB [%.0fs]"
                  % (ep + 1, tgt_sum / 4, ce_sum / 4, neg_sum / 4, bg_sum / 4,
                     int_sum / 4,
                     100 * ev["mean_acc"],
                     100 * ev["mean_ring4_wrong"],
                     100 * float(np.mean([ev["wrong"][r]["core4"]
                                          for r in REGIONS])),
                     {r: round(100 * ev["correct"][r]["ring4"], 1)
                      for r in REGIONS},
                     {r: round(ev["wrong"][r]["shat_max"], 5)
                      for r in REGIONS}, ev["contrast_db"],
                     time.time() - t0), flush=True)
            if (ep + 1) % a.save_every == 0 or ep + 1 == a.epochs:
                torch.save(full.state_dict(),
                           os.path.join(ckpt_dir, "%s_s%d_ep%04d.pth"
                                        % (a.name, a.seed, ep + 1)))
    json.dump({"name": a.name, "seed": a.seed, "epochs": a.epochs,
               "inter_rows": a.inter_rows,
               "w_inter": a.w_inter, "alpha_ce": a.alpha_ce,
               "alpha_anti_neg": a.alpha_anti_neg, "alpha_bg": a.alpha_bg,
               "ce_mode": a.ce_mode, "ce_scale": a.ce_scale,
               "init": a.init,
               "aperture": "full" if a.no_aperture_mask else "quarter-disks",
               "batch_own": BATCH_OWN, "batch_wrong": BATCH_WRONG,
               "freeze": not a.no_freeze, "pos_th": POS_TH, "neg_th": NEG_TH,
               "ring4_px": float(ring4.sum()), "curve": curve},
              open(os.path.join(out_dir, "train_%s_s%d.json"
                               % (a.name, a.seed)), "w"), indent=1)
    print("done in %.0f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
