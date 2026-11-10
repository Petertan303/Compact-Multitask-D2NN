"""Stage 2 of the equal-budget control, with the anti-crosstalk term.

The four 150x150 square sub-apertures trained on the classification term only
are pasted side by side into the 300x300 canvas and then fine-tuned with the
same stage-2 recipe as the shared-aperture device:
  four passes per epoch, every row lights all four windows at once (the focus
  window holds its own data, the other three hold foreign data in a cyclic
  assignment), 64 own rows and 3x32 foreign rows;
  L = alpha_ce CE(gamma s_hat) + alpha_neg E_neg + alpha_bg E_bg
      + W_inter E_big-ring, the last term measured on the foreign rows only;
alpha_ce 300 by default with the normalised readout s_hat = S / P_out and
gamma 300,
  alpha_neg 40, alpha_bg 1e3, W_inter 1e3, Adam lr 0.1 with a linear decay to
  1e-6 over epochs x 4 steps, gradient clip 1.0, and a gradient freeze that
  lets only the focus square update.
The only difference from the shared device is geometry: four disjoint squares,
no shared pixel.  Comparing this arm with the shared device separates what the
anti-crosstalk term buys from what the sharing buys.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.optim as optim

HERE = os.path.dirname(os.path.abspath(__file__))

import eval_tiled_control as etc
import control_tiled_lib as t15
from d2nn import config, data, losses
from d2nn.training import _make_differential_gallery, _sample_batch

DEV = config.DEVICE
REG = list(config.REGIONS)
WRONG_FOR_CORNER = {"NW": "NE", "NE": "SW", "SW": "SE", "SE": "NW"}
CKPT_DIR = os.path.join(HERE, "checkpoints")
BATCH_OWN, BATCH_WRONG = 64, 32
W_INTER, ALPHA_CE, ALPHA_NEG, ALPHA_BG, CE_SCALE = 1e3, 300.0, 40.0, 1e3, 3e2
POS_TH, NEG_TH = 0.05, 0.01
LR = 0.1


def big_rings():
    lp, ln, sg = t15.ring_params()
    rings, out = {}, torch.zeros(config.N, config.N, device=DEV)
    for r in REG:
        gp, gn = _make_differential_gallery(t15.CENT, t15.CENT, t15.TILE,
                                            config.DX, lp, ln, sg,
                                            config.N_CLASSES, DEV)
        one = (((gp > POS_TH).float().sum(0) + (gn > NEG_TH).float().sum(0))
               > 1e-2).float()
        big = torch.zeros(config.N, config.N, device=DEV)
        ox, oy = t15.OFFS[r]
        big[oy:oy + t15.TILE, ox:ox + t15.TILE] = one
        rings[r] = big
        out = torch.maximum(out, big)
    return rings, out


def freeze_masks():
    m = {}
    for r in REG:
        mm = torch.zeros(config.N, config.N, device=DEV)
        ox, oy = t15.OFFS[r]
        mm[oy:oy + t15.TILE, ox:ox + t15.TILE] = 1.0
        m[r] = mm
    return m


def place(window, imgs):
    """Energy-normalised, upsampled, noise-padded patch inside one square."""
    imgs = losses.normalize_input_to_canvas_energy(
        imgs, target_l2_sq=config.CANVAS_L2_SQ)
    ox, oy = t15.OFFS[window]
    big = torch.zeros(imgs.shape[0], config.N, config.N, device=DEV)
    for i in range(imgs.shape[0]):
        sc = t15.pad_input_to_canvas(imgs[i], t15.TILE, upsample=t15.UPSAMPLE,
                                     centroid=(t15.CENT, t15.CENT),
                                     noise_std=t15.NOISE)
        big[i, oy:oy + t15.TILE, ox:ox + t15.TILE] = sc
    return big


def mixed_canvas(ds_train, focus):
    rows, labels = [], None
    imgs, lab, _ = _sample_batch(ds_train[focus], BATCH_OWN, DEV)
    rows.append(place(focus, imgs))
    labels = lab
    for corner in REG:
        if corner == focus:
            continue
        oimg, _, _ = _sample_batch(ds_train[WRONG_FOR_CORNER[corner]],
                                   BATCH_WRONG, DEV)
        rows.append(place(corner, oimg))
    return torch.cat(rows, dim=0), labels


def target_parts(out_field, labels, pos, neg, host):
    I = torch.abs(out_field) ** 2
    tot = I.flatten(1).sum(dim=1) + 1e-8
    E_pos = torch.stack([(I * pos[host][k]).flatten(1).sum(dim=1)
                         for k in range(config.N_CLASSES)], dim=1)
    E_neg = torch.stack([(I * neg[host][k]).flatten(1).sum(dim=1)
                         for k in range(config.N_CLASSES)], dim=1)
    S = E_pos - E_neg
    L_ce = torch.nn.functional.cross_entropy(
        CE_SCALE * (S / tot.unsqueeze(1)), labels)
    L_neg = ((I * neg[host][labels]).flatten(1).sum(dim=1) / tot).mean()
    bg = torch.clamp(1.0 - pos[host][labels] - neg[host][labels], min=0.0)
    L_bg = ((I * bg).flatten(1).sum(dim=1) / tot).mean()
    return (ALPHA_CE * L_ce + ALPHA_NEG * L_neg + ALPHA_BG * L_bg,
            float(L_ce.detach()), float(L_neg.detach()), float(L_bg.detach()))


def main():
    global ALPHA_CE
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=100)
    ap.add_argument("--w-inter", type=float, default=W_INTER)
    ap.add_argument("--alpha-ce", type=float, default=ALPHA_CE,
                    help="classification weight; the shared-aperture device "
                         "uses the same default.  The control quoted in the "
                         "response was first run with --alpha-ce 30")
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--sub-tag", default="ce",
                    help="stage-1 tag of the 150x150 square sub-networks")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="tile150_lxt")
    a = ap.parse_args()
    ALPHA_CE = a.alpha_ce

    os.makedirs(os.path.join(HERE, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    subs = {}
    for r in REG:
        s = t15.SquareSubNet(3).to(DEV)
        s.load_state_dict(torch.load(
            os.path.join(CKPT_DIR, "sub150_%s_3layer_%s_s%d.pth"
                         % (r, a.sub_tag, a.seed)), map_location=DEV,
            weights_only=False))
        subs[r] = s
    net = t15.tile_subs(subs, 3).to(DEV)

    ring_small, pos, neg = t15.build_masks()
    _, ring4 = big_rings()
    keep = freeze_masks()
    print("tiled control: big-ring coverage %.1f%% of canvas, per-corner %s"
          % (100 * float(ring4.sum()) / (config.N * config.N),
             {r: round(100 * float(t15.build_masks()[0][r].sum())
                       / (config.N * config.N), 1) for r in REG}), flush=True)

    ds_train = data.load_datasets(train=True)
    ds_test = data.load_datasets(train=False)
    opt = optim.Adam(net.parameters(), lr=a.lr)
    sched = optim.lr_scheduler.LinearLR(opt, start_factor=1.0,
                                        end_factor=1e-6 / a.lr,
                                        total_iters=a.epochs * len(REG))
    curve = []
    print("epochs=%d w_inter=%g alpha_ce=%g alpha_neg=%g alpha_bg=%g lr=%.2g"
          % (a.epochs, a.w_inter, a.alpha_ce, ALPHA_NEG, ALPHA_BG, a.lr),
          flush=True)
    t0 = time.time()
    for ep in range(a.epochs):
        tgt_sum, int_sum, ce_sum = 0.0, 0.0, 0.0
        for focus in REG:
            cv, labels = mixed_canvas(ds_train, focus)
            opt.zero_grad()
            out = net(cv.to(torch.complex64))
            I_all = torch.abs(out) ** 2
            tot = I_all.flatten(1).sum(dim=1) + 1e-8
            mis = I_all[BATCH_OWN:]
            inter = ((mis * ring4).flatten(1).sum(dim=1)
                     / (mis.flatten(1).sum(dim=1) + 1e-8)).mean()
            L_target, ce_p, _, _ = target_parts(out[:BATCH_OWN], labels, pos,
                                                neg, focus)
            (L_target + a.w_inter * inter).backward()
            for layer in net.layers:
                if layer.phase.grad is not None:
                    layer.phase.grad.copy_(layer.phase.grad * keep[focus])
            gn = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            if torch.isfinite(gn):
                opt.step()
            sched.step()
            tgt_sum += float(L_target.detach())
            int_sum += float(inter.detach())
            ce_sum += ce_p
        if (ep + 1) % a.eval_every == 0 or ep + 1 == a.epochs:
            ev = etc.evaluate("%s_ep%d" % (a.name, ep + 1), net, ds_test, pos,
                              neg, ring_small, 20, [0.2])
            curve.append({"epoch": ep + 1, "L_target": tgt_sum / 4,
                          "L_inter": int_sum / 4, "L_ce": ce_sum / 4,
                          "eval": ev})
            print("  ep %4d tgt=%.0f ce=%.3f inter=%.4f acc=%.1f%% "
                  "conf=%.2f dB tau_youden TPR/FAR_off %.2f/%.3f [%.0fs]"
                  % (ep + 1, tgt_sum / 4, ce_sum / 4, int_sum / 4,
                     100 * ev["mean_acc"], ev["mean_confinement_db"],
                     float(np.mean([ev["channels"][r]["tpr_at_tau_youden"]
                                    for r in REG])),
                     float(np.mean([ev["channels"][r]["far_off_at_tau_youden"]
                                    for r in REG])), time.time() - t0),
                  flush=True)
            if (ep + 1) % a.save_every == 0 or ep + 1 == a.epochs:
                torch.save(net.state_dict(), os.path.join(
                    HERE, "checkpoints", "%s_s%d_ep%04d.pth"
                    % (a.name, a.seed, ep + 1)))
    json.dump({"name": a.name, "seed": a.seed, "epochs": a.epochs,
               "w_inter": a.w_inter, "alpha_ce": ALPHA_CE,
               "alpha_neg": ALPHA_NEG, "alpha_bg": ALPHA_BG,
               "ce_scale": CE_SCALE, "curve": curve},
              open(os.path.join(HERE, "results", "train_%s_s%d.json"
                                % (a.name, a.seed)), "w"), indent=1)
    print("done in %.0f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
