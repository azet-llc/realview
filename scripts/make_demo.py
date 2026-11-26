"""Three-panel demo: camera | RealView | RealView fed another drive's features.

    python scripts/make_demo.py --ckpt weights/realview.pt --video A --other B --out assets/realview
Writes <out>.mp4 and <out>.gif (the GIF is the first --gif-seconds).
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from realview.run import default_device  # noqa: E402
from realview.encoders import Latents, VisionEncoder  # noqa: E402
from realview.run import FPS, batched, load_model, read_video, rollout, write_mp4  # noqa: E402


def label(frames, texts):
    h = 22
    out = np.zeros((frames.shape[0], frames.shape[1] + h, frames.shape[2], 3), np.uint8)
    out[:, h:] = frames
    bar = Image.new("RGB", (frames.shape[2], h), (12, 12, 14))
    d = ImageDraw.Draw(bar)
    font = ImageFont.load_default()
    for name in ("DejaVuSans.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"):
        try:
            font = ImageFont.truetype(name, 13)
            break
        except OSError:
            pass
    w = frames.shape[2] // len(texts)
    for i, t in enumerate(texts):
        d.text((i * w + 8, 4), t, fill=(235, 235, 240), font=font)
    out[:, :h] = np.asarray(bar)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--video", required=True)
    ap.add_argument("--other", required=True)
    ap.add_argument("--out", default="assets/realview")
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--gif-seconds", type=float, default=8)
    ap.add_argument("--gif-skip", type=float, default=4, help="skip the cold start in the GIF")
    ap.add_argument("--device", default=default_device())
    a = ap.parse_args()
    dev = a.device
    model, mean, std, sigma = load_model(a.ckpt, dev)
    enc, vae = VisionEncoder().to(dev), Latents(a.weights).to(dev)
    real, other = read_video(a.video, a.seconds), read_video(a.other, a.seconds)
    n = min(len(real), len(other))
    real, other = real[:n], other[:n]
    panels = [real]
    for f in (batched(enc, real, dev), batched(enc, other, dev)):
        lat, _ = rollout(model, f, mean, std, sigma, dev)
        panels.append(batched(vae.decode, lat, dev))
    frames = torch.cat(panels, 3).permute(0, 2, 3, 1).mul(255).round().byte().numpy()
    frames = label(frames, ["Camera (never shown to the generator)", "RealView", "RealView fed another drive"])
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    write_mp4(a.out + ".mp4", frames)
    s0 = int(a.gif_skip * FPS)
    clip = frames[s0:s0 + int(a.gif_seconds * FPS)]
    tmp = a.out + ".tmp.mp4"
    write_mp4(tmp, clip)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", tmp, "-vf",
                    "fps=12,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer:bayer_scale=4",
                    a.out + ".gif"], check=True)
    os.replace(tmp, a.out + ".clip.mp4")
    print(a.out + ".mp4", a.out + ".gif")


if __name__ == "__main__":
    main()
