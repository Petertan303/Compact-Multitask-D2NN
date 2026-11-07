import os
import sys
import collections

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))

from d2nn import config
from d2nn.data import load_datasets
from d2nn.data import SyntheticShapes

def counts(ds):
    c = collections.Counter()
    if hasattr(ds, "indices") and hasattr(ds.dataset, "targets"):

        all_t = ds.dataset.targets
        t = [int(all_t[i]) for i in ds.indices]
    else:
        t = ds.targets
        t = t.tolist() if hasattr(t, "tolist") else list(t)
    for x in t:
        c[int(x)] += 1
    return c

train = load_datasets(train=True)
test = load_datasets(train=False)
print("DATA_PATH:", os.path.abspath(config.DATA_PATH))
print("regions: NW=MNIST  NE=Fashion-MNIST  SW=EMNIST-letters(A-J)  "
      "SE=SyntheticShapes")
for reg in ("NW", "NE", "SW", "SE"):
    ctr, cte = counts(train[reg]), counts(test[reg])
    print(f"\n{reg}: train n={sum(ctr.values())}  test n={sum(cte.values())}  "
          f"classes={len(ctr)}")
    print(f"   per-class train: min={min(ctr.values())} max={max(ctr.values())}"
          f"   per-class test: min={min(cte.values())} max={max(cte.values())}")

print("\nShapes class names:", SyntheticShapes.CLASS_NAMES)
print("raw image side:", config.INPUT_SIZE_RAW, "upsample x", config.INPUT_SCALE,
      "-> placed patch", config.INPUT_SIZE, "px on the",
      f"{config.N}x{config.N} aperture; classes per region: 10")
print("eval protocol: differential readout, balanced 100 images/class "
      "(1000 per region) from the official test split")
