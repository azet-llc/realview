"""One-step validation on held-out drives: generator error vs. 'copy previous chunk'.

    python scripts/val.py --ckpt runs/base/last.pt --data data/prep
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from realview.meanflow import generate, make_source  # noqa: E402
from realview.run import default_device, load_model  # noqa: E402

F, STRIDE = 8, 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data/prep")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--device", default=default_device())
    a = ap.parse_args()
    s = torch.load(a.ckpt, map_location="cpu")
    model, mean, std, sigma = load_model(a.ckpt, a.device)
    paths = [p for p in sorted(glob.glob(os.path.join(a.data, "*.npz"))) if any(h in p for h in s["holdout"])]
    rng = np.random.default_rng(0)
    xs, ps, fs = [], [], []
    for i in range(a.n):
        z = np.load(paths[i % len(paths)])
        st = 2 * rng.integers(F // 2, (len(z["lat"]) - F) // 2)
        xs.append(z["lat"][st:st + F]); ps.append(z["lat"][st - F:st]); fs.append(z["feat"][st:st + F:STRIDE])
    t = lambda v: torch.from_numpy(np.stack(v)).float().to(a.device)
    x, prev, f = (t(xs) - mean) / std, (t(ps) - mean) / std, t(fs)
    torch.manual_seed(0)
    out = {}
    for name, p, ff in [("model", prev, f), ("model_cold", torch.zeros_like(prev), f),
                        ("model_zero_feats", prev, torch.zeros_like(f)),
                        ("model_shuffled_feats", prev, f.roll(1, 0))]:
        g = torch.cat([generate(model, make_source(p[i:i + 16], sigma), ff[i:i + 16]) for i in range(0, a.n, 16)])
        out[name] = round(float((g - x).pow(2).mean()), 4)
    out["copy_prev"] = round(float((prev - x).pow(2).mean()), 4)
    out["predict_zero"] = round(float(x.pow(2).mean()), 4)
    print({"step": s["step"], **out})


if __name__ == "__main__":
    main()
