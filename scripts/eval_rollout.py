"""Full autoregressive rollouts on held-out clips, with ablations and baselines.

For each clip: real features, shuffled features (another held-out clip), zero features.
Metrics:
  psnr        whole-clip PSNR vs. the real frames (low frequencies dominate it, so a still frame scores well)
  track@0.4s  for every generated 8-frame chunk, is the nearest real chunk (16x12 thumbnails) of the same
              clip within one chunk (0.4 s) of the true time? A still frame or a wrong drive scores near
              chance, which is printed alongside.

    python scripts/eval_rollout.py --ckpt weights/realview.pt --clips VIDEO [VIDEO ...]
"""
import argparse
import json
import os
import sys

import torch
import torch.nn.functional as Fn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from realview.encoders import Latents, VisionEncoder  # noqa: E402
from realview.run import F, batched, load_model, psnr, read_video, rollout  # noqa: E402


def thumbs(v):
    return Fn.interpolate(v, (12, 16), mode="area").reshape(-1, F, 3 * 12 * 16).mean(1)


def track(gen, real):
    g, r = thumbs(gen), thumbs(real)
    d = torch.cdist(g, r)
    hit = (d.argmin(1) - torch.arange(len(g))).abs() <= 1
    return float(hit.float().mean()), min(1.0, 3 / len(g))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    dev = a.device
    model, mean, std, sigma = load_model(a.ckpt, dev)
    enc, vae = VisionEncoder().to(dev), Latents(a.weights).to(dev)
    vids = [read_video(c, a.seconds) for c in a.clips]
    n = min(len(v) for v in vids)
    vids = [v[:n] for v in vids]
    feats = [batched(enc, v, dev) for v in vids]
    rows = []
    for i, real in enumerate(vids):
        row = {"clip": os.path.relpath(a.clips[i], os.path.dirname(os.path.dirname(os.path.dirname(a.clips[i]))))}
        for name, f in [("real", feats[i]), ("shuffled", feats[(i + 1) % len(vids)]), ("zero", torch.zeros_like(feats[i]))]:
            lat, _ = rollout(model, f, mean, std, sigma, dev)
            gen = batched(vae.decode, lat, dev)
            row[f"psnr_{name}"] = round(psnr(gen, real), 2)
            row[f"track_{name}"], chance = track(gen, real)
        mframe = real.mean(0, keepdim=True).expand_as(real)
        row["psnr_mean_frame"] = round(psnr(mframe, real), 2)
        row["track_chance"] = round(chance, 3)
        rows.append(row)
        print(json.dumps(row), flush=True)
    keys = [k for k in rows[0] if k != "clip"]
    print(json.dumps({"MEAN": {k: round(sum(r[k] for r in rows) / len(rows), 3) for k in keys}, "clips": len(rows), "seconds": n / 20}))


if __name__ == "__main__":
    main()
