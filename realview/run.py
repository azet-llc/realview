"""Run RealView on a driving video and write  real | generated  side by side.

    python -m realview.run --video drive.mp4 --out out.mp4
    python -m realview.run --video drive.mp4 --feats shuffle:other.mp4   # ablation
    python -m realview.run --video drive.mp4 --feats zero                 # ablation

The generator only ever sees vision-encoder tokens and its own previous chunk.
"""
import argparse
import json
import os
import time

import av
import numpy as np
import torch

from .encoders import H, W, Latents, VisionEncoder
from .meanflow import generate, make_source
from .model import RealView

FPS = 20  # comma2k19 camera rate; the post's 36 fps / 18 Hz keep the same 2:1 ratio here
F, STRIDE = 8, 2
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def read_video(path, seconds=None):
    """Center-crop to 4:3, resize to 256x192, resample to 20 fps."""
    out, next_t = [], 0.0
    with av.open(path) as c:
        s = c.streams.video[0]
        # raw comma .hevc has no timestamps and a bogus 25 fps tag; it is 20 fps
        rate = FPS if path.endswith(".hevc") else float(s.average_rate or FPS)
        for i, f in enumerate(c.decode(video=0)):
            t = float(f.time) if f.time is not None and not path.endswith(".hevc") else i / rate
            if t + 1e-6 < next_t:
                continue
            next_t += 1.0 / FPS
            img = f.to_ndarray(format="rgb24")
            h, w = img.shape[:2]
            if w * 3 > h * 4:
                cw = h * 4 // 3; img = img[:, (w - cw) // 2:(w - cw) // 2 + cw]
            else:
                ch = w * 3 // 4; img = img[(h - ch) // 2:(h - ch) // 2 + ch]
            img = av.VideoFrame.from_ndarray(np.ascontiguousarray(img), format="rgb24").to_ndarray(
                format="rgb24", width=W, height=H)
            out.append(img)
            if seconds and len(out) >= seconds * FPS:
                break
    n = len(out) // F * F
    return torch.from_numpy(np.stack(out[:n])).permute(0, 3, 1, 2).float() / 255


def batched(fn, x, dev, bs=64):
    return torch.cat([fn(x[i:i + bs].to(dev)).float().cpu() for i in range(0, len(x), bs)])


def psnr(a, b):
    return float(10 * torch.log10(1 / ((a - b) ** 2).mean().clamp_min(1e-10)))


def load_model(ckpt, dev):
    s = torch.load(ckpt, map_location="cpu")
    m = RealView(**s["cfg"])
    m.load_state_dict(s["model"])  # released files hold one state dict; training checkpoints hold model + ema
    return m.to(dev).eval(), s["mean"].to(dev), s["std"].to(dev), s.get("sigma", 0.7)


@torch.no_grad()
def rollout(model, feats, mean, std, sigma, dev, steps=1, seed=0):
    torch.manual_seed(seed)
    n = feats.shape[0] // F
    prev = torch.zeros(1, F, model.C, model.H, model.W, device=dev)
    outs, t0 = [], time.time()
    for k in range(n):
        f = feats[k * F:(k + 1) * F:STRIDE][None].to(dev)
        prev = generate(model, make_source(prev, sigma), f, steps=steps)
        outs.append(prev[0] * std + mean)
    if dev == "mps":
        torch.mps.synchronize()
    elif dev == "cuda":
        torch.cuda.synchronize()
    return torch.cat(outs), (time.time() - t0) / n


def write_mp4(path, frames, fps=FPS):
    with av.open(path, "w") as c:
        s = c.add_stream("libx264", rate=fps)
        s.width, s.height, s.pix_fmt = frames.shape[2], frames.shape[1], "yuv420p"
        s.options = {"crf": "20"}
        for fr in frames:
            for p in s.encode(av.VideoFrame.from_ndarray(fr, format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "weights", "realview.pt"))
    ap.add_argument("--weights", default=os.path.join(ROOT, "weights"))
    ap.add_argument("--out", default="realview_out.mp4")
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--steps", type=int, default=1)
    ap.add_argument("--sigma", type=float, default=None, help="noise on the previous chunk (default: training value)")
    ap.add_argument("--feats", default="real", help="real | zero | shuffle:<other video>")
    ap.add_argument("--device", default=default_device())
    a = ap.parse_args()
    dev = a.device

    model, mean, std, sigma = load_model(a.ckpt, dev)
    sigma = a.sigma if a.sigma is not None else sigma
    enc, vae = VisionEncoder().to(dev), Latents(a.weights).to(dev)
    real = read_video(a.video, a.seconds)
    feats = batched(enc, real, dev)
    if a.feats == "zero":
        feats = torch.zeros_like(feats)
    elif a.feats.startswith("shuffle:"):
        other = batched(enc, read_video(a.feats.split(":", 1)[1], a.seconds), dev)
        feats = other[:len(feats)]
        real = real[:len(feats)]
    lat, sec_per_chunk = rollout(model, feats, mean, std, sigma, dev, a.steps)
    gen = batched(vae.decode, lat, dev)
    recon = batched(lambda x: vae.decode(vae.encode(x)), real, dev)   # VAE ceiling

    per_chunk = [psnr(gen[i:i + F], real[i:i + F]) for i in range(0, len(gen), F)]
    freeze = gen[:F].mean(0, keepdim=True).expand_as(real)              # "just copy" baseline
    stats = {
        "video": os.path.basename(a.video), "feats": a.feats, "seconds": len(real) / FPS,
        "psnr_generated": round(psnr(gen, real), 2),
        "psnr_vae_ceiling": round(psnr(recon, real), 2),
        "psnr_frozen_first_chunk": round(psnr(freeze, real), 2),
        "psnr_first_10s": round(float(np.mean(per_chunk[:len(per_chunk) // 6 or 1])), 2),
        "psnr_last_10s": round(float(np.mean(per_chunk[-(len(per_chunk) // 6 or 1):])), 2),
        "ms_per_chunk": round(sec_per_chunk * 1000, 1), "denoise_steps": a.steps, "sigma": sigma, "device": dev,
    }
    side = torch.cat([real, gen], 3).permute(0, 2, 3, 1).mul(255).round().byte().numpy()
    write_mp4(a.out, side)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
