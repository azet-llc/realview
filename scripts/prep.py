"""Turn driving videos into training arrays: per-frame VAE latents (the targets)
and per-frame vision-encoder tokens (the only thing the generator is shown).

    python scripts/prep.py --videos DIR --out data/prep --weights weights
"""
import argparse
import glob
import os
import sys

import av
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from realview.run import default_device  # noqa: E402
from realview.encoders import H, W, Latents, VisionEncoder  # noqa: E402


def frames(path):
    with av.open(path) as c:
        for f in c.decode(video=0):
            yield f.to_ndarray(format="rgb24", width=W, height=H)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", default="data/prep")
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--device", default=default_device())
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    enc = VisionEncoder().to(a.device)
    vae = Latents(a.weights).to(a.device)
    paths = sorted(glob.glob(os.path.join(a.videos, "**", "*.hevc"), recursive=True)
                   + glob.glob(os.path.join(a.videos, "**", "*.mp4"), recursive=True))
    for i, p in enumerate(paths):
        name = os.path.relpath(p, a.videos).replace("/", "__").replace("|", "_")
        dst = os.path.join(a.out, name + ".npz")
        if os.path.exists(dst):
            continue
        lats, feats = [], []
        buf = []
        for fr in list(frames(p)) + [None]:
            if fr is not None:
                buf.append(fr)
            if len(buf) == 64 or (fr is None and buf):
                x = torch.from_numpy(np.stack(buf)).to(a.device).permute(0, 3, 1, 2).float() / 255
                lats.append(vae.encode(x).half().cpu())
                feats.append(enc(x).half().cpu())
                buf = []
        np.savez(dst, lat=torch.cat(lats).numpy(), feat=torch.cat(feats).numpy())
        print(f"[{i + 1}/{len(paths)}] {name} frames={sum(len(l) for l in lats)}", flush=True)


if __name__ == "__main__":
    main()
