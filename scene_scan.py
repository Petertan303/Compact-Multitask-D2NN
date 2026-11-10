"""Scene construction and sliding-window scan for the deployment demo.

A world scene of GRID x GRID cells of 300 x 300 px is built from test images
of the four families: every non-empty cell carries one object per family it
draws, placed at the placement centre of that family's sub-aperture, so the
scene is exactly what a fixed device would see while the stage moves under it.

The device itself never moves.  A window is a 300 x 300 crop of the scene; its
four quadrant patches (212 x 212 at the sub-aperture offsets) are cropped,
filled with noise where empty, peak-normalised where an object is present and
superposed on the canvas.  All four sub-apertures are therefore illuminated at
the same time in every window, which is the condition the confinement term is
trained for.

Object centres are computed with the same convention the scene composer uses
(row = cell + sub-aperture offset in y + centroid y, column likewise), which
is what decides whether an object lies fully inside a quadrant patch or is
split across its border.
"""
import math
import random

import numpy as np
import torch

from d2nn import config, data
from d2nn.stage_utils import pad_input_to_canvas

REGIONS = list(config.REGIONS)
NC = config.N_CLASSES
SUB = config.SUB_SIZE
N_OFF = SUB - config.OVERLAP
OFF = {"NW": (0, 0), "NE": (N_OFF, 0), "SW": (0, N_OFF), "SE": (N_OFF, N_OFF)}
CENTS = {"NW": (90, 90), "NE": (121, 90), "SW": (90, 121), "SE": (121, 121)}
QS = {"NW": (slice(0, SUB), slice(0, SUB)),
      "NE": (slice(0, SUB), slice(N_OFF, config.N)),
      "SW": (slice(N_OFF, config.N), slice(0, SUB)),
      "SE": (slice(N_OFF, config.N), slice(N_OFF, config.N))}
CELL = config.N
NOISE_STD = 0.01
HALF = 28 * config.INPUT_SCALE // 2          # object half footprint, px
DEV = config.DEVICE


def build_image_pool(seed, images_per_class):
    """Distinct test images per class per family."""
    rng = random.Random(seed + 777)
    ds_test = data.load_datasets(train=False)
    pool = {}
    for region in REGIONS:
        by_label = {c: [] for c in range(NC)}
        for idx in range(len(ds_test[region])):
            _, lab = ds_test[region][idx]
            by_label[int(lab) % NC].append(idx)
        pool[region] = {}
        for c in range(NC):
            picks = rng.sample(by_label[c],
                               min(images_per_class, len(by_label[c])))
            pool[region][c] = [ds_test[region][i][0] for i in picks]
    del ds_test
    return pool


def build_scene(seed, grid, n_full, n_partial, pool):
    rng = random.Random(seed)
    cell_types = (["full"] * n_full + ["partial"] * n_partial +
                  ["empty"] * (grid * grid - n_full - n_partial))
    rng.shuffle(cell_types)
    cells = []
    for i in range(grid * grid):
        r, c = divmod(i, grid)
        ctype = cell_types[i]
        pick = {}
        if ctype != "empty":
            n_obj = 4 if ctype == "full" else 2
            for region in rng.sample(REGIONS, n_obj):
                lab = rng.randrange(NC)
                pick[region] = (lab, rng.randrange(len(pool[region][lab])))
        cells.append({"row": r, "col": c, "type": ctype, "pick": pick})
    world = torch.zeros(grid * CELL, grid * CELL, dtype=torch.float32)
    for cell in cells:
        r0, c0 = cell["row"] * CELL, cell["col"] * CELL
        for region, (lab, k) in cell["pick"].items():
            img = pool[region][lab][k]
            if img.dim() == 3:
                img = img.squeeze(0)
            patch = pad_input_to_canvas(img, SUB, upsample=config.INPUT_SCALE,
                                        centroid=CENTS[region], noise_std=0.0)
            ox, oy = OFF[region]
            world[r0 + oy:r0 + oy + SUB, c0 + ox:c0 + ox + SUB] += patch
    return world, cells


def object_boxes(cells):
    boxes = []
    for cell in cells:
        for region, (lab, _k) in cell["pick"].items():
            ox, oy = OFF[region]
            cx, cy = CENTS[region]
            boxes.append({"center": (cell["row"] * CELL + oy + cy,
                                     cell["col"] * CELL + ox + cx),
                          "region": region, "label": lab})
    return boxes


def classify_patch(box, boxes):
    r0, c0, r1, c1 = box
    full, part = [], []
    for b in boxes:
        cr, cc = b["center"]
        if (cr - HALF >= r0 and cc - HALF >= c0 and
                cr + HALF <= r1 and cc + HALF <= c1):
            full.append(b)
        elif (cr - HALF < r1 and cc - HALF < c1 and
              cr + HALF > r0 and cc + HALF > c0):
            part.append(b)
    return full, part


def window_positions(world_side, step):
    """Serpentine order of window corners, positions are multiples of step."""
    n = world_side // step
    order = []
    for r in range(n):
        cols = range(n) if r % 2 == 0 else range(n - 1, -1, -1)
        for c in cols:
            order.append((r * step, c * step))
    return order


@torch.no_grad()
def scan_window(net, world_pad, wy, wx, boxes, pos, neg):
    """One window: build the canvas, propagate, read every quadrant."""
    canvas = torch.zeros(config.N, config.N, device=DEV)
    info = {}
    for region in REGIONS:
        ox, oy = OFF[region]
        pr0, pc0 = wy + oy, wx + ox
        patch = world_pad[pr0:pr0 + SUB, pc0:pc0 + SUB].clone()
        patch = torch.where(patch == 0, torch.randn_like(patch) * NOISE_STD,
                            patch)
        pmax = patch.max()
        if pmax > 0.05:
            patch = patch / pmax
        rows, cols = QS[region]
        canvas[rows, cols] += patch.to(DEV)
        full, part = classify_patch((pr0, pc0, pr0 + SUB, pc0 + SUB), boxes)
        info[region] = {"full": full, "part": part}
    out = net(canvas.unsqueeze(0).to(torch.complex64))
    I = torch.abs(out[0]) ** 2
    If = I.reshape(-1)
    P = float(If.sum()) + 1e-8
    readout = {}
    for region in REGIONS:
        pm, nm = _MASKS[region]
        diff = ((If @ pm) - (If @ nm)).cpu().numpy() / P
        order = np.argsort(diff)[::-1]
        readout[region] = {"pred": int(order[0]),
                           "smax": float(diff[order[0]]),
                           "margin": float(diff[order[0]] - diff[order[1]])}
    return readout, info


_MASKS = None


def prepare_masks(pos, neg):
    """Flatten the detector masks once so each window costs one matmul per
    quadrant instead of twenty reductions with a device sync each."""
    global _MASKS
    _MASKS = {r: (pos[r].flatten(1).T.contiguous(),
                  neg[r].flatten(1).T.contiguous()) for r in REGIONS}
    return _MASKS


def prepare(world, step):
    pad = torch.zeros(world.shape[0] + step, world.shape[1] + step,
                      dtype=torch.float32)
    pad[:world.shape[0], :world.shape[1]] = world
    return pad
