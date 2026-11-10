"""Per-class census of the four task datasets, for R1.5 (describe the Shapes set).

Reports, for every task and split actually used by `d2nn.data.load_datasets`:
the number of files, the image side in pixels, and how many distinct images
exist.  The distinct-image count matters: the synthetic Shapes set is drawn by
`d2nn/shapes.py` with a fixed centre, a fixed scale and a fixed line width, so
within a class every sample is the same picture.  That is the reason the Shapes
channel converges to 100% while the other three do not, and it must be stated
in the manuscript.
"""
import glob
import hashlib
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "public_release"))
from d2nn import config, data

NAME = {"NW": "MNIST", "NE": "Fashion", "SW": "EMNIST letters (10 classes)",
        "SE": "synthetic Shapes"}
SHAPES_DIR = os.path.join(config.DATA_PATH, "synthetic_shapes_%s", "%d")


def main():
    for train in (True, False):
        split = "train" if train else "test"
        ds = data.load_datasets(train=train)
        print("== %s split ==" % split)
        for r in ("NW", "NE", "SW"):
            d = ds[r]
            sizes = {tuple(np.asarray(d[i][0]).shape) for i in
                     range(0, len(d), max(1, len(d) // 200))}
            uniq = len({hash(d[i][0].numpy().tobytes())
                        for i in range(0, len(d),
                                       max(1, len(d) // 2000))})
            print("  %-28s n=%-6d shapes %s  distinct in sample %d"
                  % (NAME[r], len(d), sizes, uniq))
        for lab in range(10):
            fs = sorted(glob.glob(SHAPES_DIR % (split, lab) + "/*.png"))
            hashes = {hashlib.md5(np.asarray(Image.open(f),
                                             dtype=np.uint8)).hexdigest()
                      for f in fs[:300]}
            shape = Image.open(fs[0]).size if fs else None
            print("  Shapes class %-2d files %-6d size %-8s distinct in "
                  "first 300: %d" % (lab, len(fs), shape, len(hashes)))


if __name__ == "__main__":
    main()
