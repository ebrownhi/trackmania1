"""Shared helpers for the real-simulator (L2F -> gym-pybullet-drones) experiments."""

import json
import os
import sys
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/claude-501/mpl")
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402,F401

OUT = os.path.join(ROOT, "results", "real_sim")
os.makedirs(OUT, exist_ok=True)


def save_json(name: str, obj) -> str:
    path = os.path.join(OUT, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=float)
    return path


def mean_std(x):
    x = np.asarray(x, dtype=np.float64)
    return {"mean": float(x.mean()), "std": float(x.std()), "n": int(x.size), "values": x.tolist()}


def fmt(ms) -> str:
    return f"{ms['mean']:.3f} ± {ms['std']:.3f}"
