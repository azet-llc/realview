"""Small two-panel GIFs (camera | RealView) for the README gallery.

    python scripts/make_gallery.py --ckpt weights/realview.pt --out assets/gallery \
        --clip day=VIDEO --clip night=VIDEO --clip blind=VIDEO:zero
`name=VIDEO` uses the clip's own features; `:zero` blinds the generator (all features set to zero).
"""
import argparse
import os
import subprocess
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from realview.encoders import Latents, VisionEncoder  # noqa: E402
from realview.run import FPS, batched, default_device, load_model, read_video, rollout, write_mp4  # noqa: E402
from make_demo import label  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--out", default="assets/gallery")
    ap.add_argument("--clip", action="append", required=True)
    ap.add_argument("--seconds", type=float, default=16)
    ap.add_argument("--gif-skip", type=float, default=4)
    ap.add_argument("--gif-seconds", type=float, default=8)
    ap.add_argument("--device", default=default_device())
    a = ap.parse_args()
    dev = a.device
    model, mean, std, sigma = load_model(a.ckpt, dev)
    enc, vae = VisionEncoder().to(dev), Latents(a.weights).to(dev)
    os.makedirs(a.out, exist_ok=True)
    for spec in a.clip:
        name, path = spec.split("=", 1)
        zero = path.endswith(":zero")
        path = path.removesuffix(":zero")
        real = read_video(path, a.seconds)
        f = batched(enc, real, dev)
        if zero:
            f = torch.zeros_like(f)
        lat, _ = rollout(model, f, mean, std, sigma, dev)
        gen = batched(vae.decode, lat, dev)[: len(real)]
        frames = torch.cat([real[: len(gen)], gen], 3).permute(0, 2, 3, 1).mul(255).round().byte().numpy()
        frames = label(frames, ["Camera", "RealView, features zeroed" if zero else "RealView"])
        s0 = int(a.gif_skip * FPS)
        tmp = os.path.join(a.out, name + ".tmp.mp4")
        write_mp4(tmp, frames[s0:s0 + int(a.gif_seconds * FPS)])
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", tmp, "-vf",
                        "fps=12,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer:bayer_scale=4",
                        os.path.join(a.out, name + ".gif")], check=True)
        os.remove(tmp)
        print(name, os.path.getsize(os.path.join(a.out, name + ".gif")), flush=True)


if __name__ == "__main__":
    main()
