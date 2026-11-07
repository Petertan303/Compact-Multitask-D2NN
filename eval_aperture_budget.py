import os
import sys

import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from d2nn import config
from d2nn.layers import _get_quarter_circle_mask

N = config.N
off = N - config.SUB_SIZE
SLOTS = {"NW": (0, 0), "NE": (off, 0), "SW": (0, off), "SE": (off, off)}
cov = torch.zeros(N, N)
for r, (ox, oy) in SLOTS.items():
    m = _get_quarter_circle_mask(r, config.SUB_SIZE, "cpu")
    cov[oy:oy + config.SUB_SIZE, ox:ox + config.SUB_SIZE] += m
u4 = float((cov > 0).sum())
per = {r: float(m.sum()) for r, m in
       ((r, _get_quarter_circle_mask(r, config.SUB_SIZE, "cpu"))
        for r in SLOTS)}
print("one quarter disk area      : %.0f px" % per["NW"])
print("union of four disks        : %.0f px/layer  (%.0f parameters)"
      % (u4, 3 * u4))
for drop in ("SE",):
    keep = [r for r in SLOTS if r != drop]
    c = torch.zeros(N, N)
    for r in keep:
        ox, oy = SLOTS[r]
        c[oy:oy + config.SUB_SIZE, ox:ox + config.SUB_SIZE] += _get_quarter_circle_mask(
            r, config.SUB_SIZE, "cpu")
    u3 = float((c > 0).sum())
    active = sum(float(_get_quarter_circle_mask(r, config.SUB_SIZE, "cpu").sum())
                 for r in keep)
    print("three disks, sum of areas  : %.0f px/layer" % active)
    print("union without %s           : %.0f px/layer  (%.0f parameters), "
          "difference %.0f" % (drop, u3, 3 * u3, u4 - u3))
print("four isolated 212x212 squares: %.0f px/layer" % (4 * config.SUB_SIZE ** 2))
print("four isolated quarter disks  : %.0f px/layer (sum of areas)"
      % (4 * per["NW"]))
print("four 150x150 squares         : %.0f px/layer" % (4 * 150 ** 2))
