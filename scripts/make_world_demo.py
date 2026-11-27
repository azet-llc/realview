"""World-model GIFs for the README.

  <out>/world.gif         camera | dream with the recorded actions | dream nudged left | dream nudged right
  <out>/gallery/closed_loop.gif   starting 0.8 m off the path: the cloned policy and the RL policy, each in its own dream (2x speed)

    python scripts/make_world_demo.py --video weights/sample_day.mp4 --log weights/sample_day.npz --out assets
The bar under each dream is the kinematic offset from the recorded path; the ticks are the 1.2 m lane limit.
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from realview.dream import dream, load_inputs, nudge, paint  # noqa: E402
from realview.encoders import Latents, VisionEncoder  # noqa: E402
from realview.run import default_device, load_model, write_mp4  # noqa: E402
from realview.world import D_MAX, Head, load_world  # noqa: E402
from make_demo import label  # noqa: E402

SPAN = 2.4  # metres shown on each side of the bar


def with_bar(frames, d):
    """frames (N,3,H,W) in [0,1] -> uint8 (N,H+10,W,3) with an offset bar. d is per 10 Hz tick, + = left."""
    f = frames.permute(0, 2, 3, 1).mul(255).round().byte().numpy()
    n, h, w, _ = f.shape
    out = np.full((n, h + 10, w, 3), 24, np.uint8)
    out[:, :h] = f
    x = lambda m: int(np.clip(w / 2 - m / SPAN * w / 2, 2, w - 3))
    for i in range(n):
        m = d[min(i // 2, len(d) - 1)] if d is not None else None
        out[i, h + 4:h + 6] = 70
        if m is None:
            continue
        for lim in (-D_MAX, D_MAX):
            out[i, h + 1:h + 9, x(lim)] = 150
        out[i, h + 1:h + 9, x(m) - 2:x(m) + 3] = (240, 70, 70) if abs(m) > D_MAX else (240, 240, 245)
    return out


def gif(frames, path, width=None):
    tmp = path + ".tmp.mp4"
    write_mp4(tmp, frames)
    scale = f"scale={width}:-2:flags=lanczos," if width else ""
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", tmp, "-vf",
                    f"fps=10,{scale}split[a][b];[a]palettegen=max_colors=64[p];[b][p]paletteuse=dither=bayer:bayer_scale=4",
                    path], check=True)
    os.remove(tmp)
    print(path, os.path.getsize(path), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--world", default="weights/realview_world.pt")
    ap.add_argument("--ckpt", default="weights/realview.pt")
    ap.add_argument("--weights", default="weights")
    ap.add_argument("--policies", default="weights")
    ap.add_argument("--out", default="assets")
    ap.add_argument("--start", type=float, default=10)
    ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--loop-seconds", type=float, default=20)
    ap.add_argument("--steer", type=float, default=0.02)
    ap.add_argument("--offset", type=float, default=0.8, help="where the closed-loop drives start, metres left of the path")
    ap.add_argument("--device", default=default_device())
    a = ap.parse_args()
    dev = a.device
    world, gen = load_world(a.world, dev), load_model(a.ckpt, dev)
    enc, vae = VisionEncoder().to(dev), Latents(a.weights).to(dev)

    real, warm, speed, route = load_inputs(a.video, a.log, a.start, a.seconds, enc, dev)
    panels, names = [with_bar(real, None)], ["Camera: what really happened"]
    for name, rate in (("Dream: recorded actions", 0.0), ("Dream: steered left", a.steer), ("Dream: steered right", -a.steer)):
        toks, d, _ = dream(world, warm, speed, route, dev, extra=nudge(len(speed), rate) if rate else None)
        panels.append(with_bar(paint(gen, vae, warm, toks, dev), d))
        names.append(f"{name} ({d[-1]:+.1f} m)" if rate else name)
    n = min(len(p) for p in panels)
    gif(label(np.concatenate([p[:n] for p in panels], 2), names), os.path.join(a.out, "world.gif"), width=860)

    real, warm, speed, route = load_inputs(a.video, a.log, a.start, a.loop_seconds, enc, dev, offset=a.offset)
    panels, names = [], []
    for name, f in (("Cloned from human logs", "policy_bc.pt"), ("After RL in the world model", "policy_rl.pt")):
        pol = Head(frames=2, out=1).to(dev)
        pol.load_state_dict(torch.load(os.path.join(a.policies, f), map_location=dev))
        toks, d, _ = dream(world, warm, speed, route, dev, policy=pol, offset=a.offset)
        end = int(np.argmax(np.abs(d) > D_MAX)) if (np.abs(d) > D_MAX).any() else None
        panels.append(with_bar(paint(gen, vae, warm, toks, dev), d))
        names.append(name + (f": out of lane at {end / 10:.1f} s" if end is not None else f": ends {abs(d[-1]):.1f} m off"))
    n = min(len(p) for p in panels)
    gif(label(np.concatenate([p[:n:2] for p in panels], 2), names), os.path.join(a.out, "gallery", "closed_loop.gif"))  # 2x speed


if __name__ == "__main__":
    main()
