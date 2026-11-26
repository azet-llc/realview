"""Train the generator. Resumes from --out/last.pt if it exists.

    python scripts/train.py --data data/prep --out runs/base --steps 30000
"""
import argparse
import copy
import glob
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from realview.run import default_device  # noqa: E402
from realview.meanflow import generate, loss_fn, make_source  # noqa: E402
from realview.model import RealView  # noqa: E402

F = 8          # output frames per chunk
STRIDE = 2     # one encoder frame per 2 output frames (18 Hz -> 36 fps in the post)


def load(data, holdout):
    clips = []
    for p in sorted(glob.glob(os.path.join(data, "*.npz"))):
        if any(h in os.path.basename(p) for h in holdout):
            continue
        z = np.load(p)
        clips.append((torch.from_numpy(z["lat"]).float(), torch.from_numpy(z["feat"])))
    return clips


def batch(clips, B, mean, std, dev):
    xs, ps, pps, fs, pfs = [], [], [], [], []
    for _ in range(B):
        lat, feat = clips[np.random.randint(len(clips))]
        s = 2 * np.random.randint(F, (len(lat) - F) // 2)
        xs.append(lat[s:s + F])
        ps.append(lat[s - F:s])
        pps.append(lat[s - 2 * F:s - F])
        fs.append(feat[s:s + F:STRIDE].float())
        pfs.append(feat[s - F:s:STRIDE].float())
    n = lambda v: (torch.stack(v).to(dev) - mean) / std
    return n(xs), n(ps), n(pps), torch.stack(fs).to(dev), torch.stack(pfs).to(dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/prep")
    ap.add_argument("--out", default="runs/base")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--sigma", type=float, default=0.7, help="noise added to the previous generation")
    ap.add_argument("--holdout", default="", help="comma-separated clip-name substrings kept out of training")
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--device", default=default_device())
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = a.device
    holdout = [h for h in a.holdout.split(",") if h]
    clips = load(a.data, holdout)
    allz = torch.cat([c[0][::10] for c in clips])
    mean = allz.mean((0, 2, 3), keepdim=True)[0].to(dev)  # (C,1,1)
    std = allz.std((0, 2, 3), keepdim=True)[0].to(dev)
    print(f"clips={len(clips)} frames={sum(len(c[0]) for c in clips)}", flush=True)

    model = RealView(d=a.d, depth=a.depth, heads=a.d // 64).to(dev)
    ema = copy.deepcopy(model).requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0, betas=(0.9, 0.99))
    step = 0
    ck = os.path.join(a.out, "last.pt")
    if os.path.exists(ck):
        s = torch.load(ck, map_location=dev)
        model.load_state_dict(s["model"]); ema.load_state_dict(s["ema"]); opt.load_state_dict(s["opt"])
        step = s["step"]
        print("resumed at", step, flush=True)
    print(f"params={sum(p.numel() for p in model.parameters()) / 1e6:.1f}M", flush=True)

    def save(path):
        torch.save({"model": model.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(),
                    "step": step, "cfg": model.cfg, "mean": mean.cpu(), "std": std.cpu(),
                    "sigma": a.sigma, "holdout": holdout}, path + ".tmp")
        os.replace(path + ".tmp", path)

    t0, hist = time.time(), []
    while step < a.steps:
        x, prev, pprev, feats, pfeats = batch(clips, a.bs, mean, std, dev)
        if step > 2000 and np.random.rand() < 0.25:          # learn to recover from its own output
            prev = generate(ema, make_source(pprev, a.sigma), pfeats)
        cold = torch.rand(a.bs, device=dev) < 0.3           # start from nothing often, so features must carry the scene
        prev = torch.where(cold.view(-1, 1, 1, 1, 1), torch.zeros_like(prev), prev)
        src = make_source(prev, a.sigma)
        for g in opt.param_groups:
            g["lr"] = a.lr * min(1.0, (step + 1) / 1000)
        loss, mse = loss_fn(model, x, src, feats)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - 0.999)
        step += 1
        hist.append(mse.item())
        if step % 100 == 0:
            el = time.time() - t0
            print(json.dumps({"step": step, "mse": round(float(np.mean(hist)), 4), "it_s": round(100 / el, 2)}), flush=True)
            hist, t0 = [], time.time()
        if step % 1000 == 0 or step == a.steps:
            save(ck)
        if step % 10000 == 0:
            save(os.path.join(a.out, f"step{step}.pt"))


if __name__ == "__main__":
    main()
