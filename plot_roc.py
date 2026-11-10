"""Threshold sweeps as ROC-style curves, one panel per channel.

Reads the json written by eval_accuracy_far.py, which stores for every channel
the grid of thresholds together with the detection rate under correct
placement and the false-alarm rate under off-target and empty illumination.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REGIONS = ["NW", "NE", "SW", "SE"]
NAME = {"NW": "MNIST", "NE": "Fashion", "SW": "EMNIST", "SE": "Shapes"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", nargs="+", required=True)
    ap.add_argument("--tag", default="roc")
    a = ap.parse_args()

    fig, axes = plt.subplots(1, len(REGIONS),
                             figsize=(3.0 * len(REGIONS), 2.8))
    for path in a.json:
        entries = json.load(open(path))
        if isinstance(entries, dict):
            entries = [entries]
        for ent in entries:
            for ax, r in zip(axes, REGIONS):
                d = ent["channels"][r]
                far = 100 * np.array(d["sweep"]["far_off"])
                tpr = 100 * np.array(d["sweep"]["tpr"])
                ax.plot(far, tpr, lw=1.4,
                        label="%s off-target" % ent.get("device", ""))
                ax.axvline(1.0, color="0.6", lw=0.7, ls=":")
                x = 100 * d["far_off_at_tau_youden"]
                y = 100 * d["tpr_at_tau_youden"]
                ax.plot([x], [y], "o", ms=5)
                ax.annotate("%.3f" % d["tau_youden"], (x, y), fontsize=6,
                            xytext=(3, 3), textcoords="offset points")
                ax.set_xlabel("false alarm on off-target input [%]",
                              fontsize=7)
                ax.set_ylabel("detection rate [%]", fontsize=7)
                ax.set_title("%s  dotted line: 1%% empty-window FAR" % NAME[r],
                             fontsize=7.5)
                ax.tick_params(labelsize=7)
                ax.legend(fontsize=6)
    fig.tight_layout()
    p = os.path.join(HERE, "results", "%s.png" % a.tag)
    fig.savefig(p, dpi=200)
    print("saved ->", p)


if __name__ == "__main__":
    main()
