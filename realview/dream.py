"""Drive inside the world model and watch it.

    python -m realview.dream --video drive.mp4 --log drive.npz --out dream.mp4               # replay the recorded actions
    python -m realview.dream --video drive.mp4 --log drive.npz --steer 0.02                  # nudge it to the left
    python -m realview.dream --video drive.mp4 --log drive.npz --offset 0.8 --policy weights/policy_rl.pt
                                              # start 0.8 m left of the path and let a policy steer back

The camera is read for 2.4 s only. After that every frame is dreamed: a kinematic model turns the
steering into a pose, the world model predicts the next 0.4 s of encoder tokens at that pose, and the
v0.1 generator paints them.
`--log` is an .npz with per-frame `speed` (m/s) and `yaw` (rad/s, + = left) at 20 fps. Without it the
car holds --speed on a straight road.
"""
import argparse
import json
import os

import numpy as np
import torch

from .encoders import Latents, VisionEncoder
from .run import F, FPS, ROOT, batched, default_device, load_model, read_video, rollout, write_mp4
from .world import D_MAX, YAW_S, Head, T, kin, load_world, step, to_grid, to_tokens, warp

WARM = 24  # real encoder ticks shown before the dream starts (2.4 s): context for both models


def nudge(n_ticks, rate, hold=12):
    """Steer one way for 1.2 s, then back for 1.2 s: the car ends up beside the recorded path, parallel to it."""
    e = torch.zeros(n_ticks)
    e[:hold], e[hold:2 * hold] = rate, -rate
    return e


@torch.no_grad()
def dream(world, feats, speed, route, dev, extra=None, policy=None, offset=0.0):
    """feats: (WARM, 48, 384) real tokens at 10 Hz. speed, route: (n,) per tick after the warm-up.
    Returns dreamed tokens (n, 48, 384) and the kinematic offset / heading error per tick."""
    model, mean, std = world
    z = ((to_grid(feats[-T:].float().to(dev)) - mean) / std)[None]
    d, psi = torch.full((1,), float(offset), device=dev), torch.zeros(1, device=dev)
    out, ds, ps = [], [], []
    for h in range(len(speed) // T):
        sl = slice(h * T, (h + 1) * T)
        sp, rt = speed[sl][None].to(dev), route[sl][None].to(dev)
        if policy is not None:
            ego = rt + policy(z[:, -2:], sp[:, 0])[:, 0] / YAW_S
        else:
            ego = rt + (extra[sl][None].to(dev) if extra is not None else 0)
        dd, pp = kin(d, psi, sp, ego, rt)
        z = step(model, z, sp, rt, dd, pp)
        d, psi = dd[:, -1], pp[:, -1]
        out.append(to_tokens(z[0] * std + mean).cpu())
        ds += dd[0].tolist()
        ps += pp[0].tolist()
    return torch.cat(out), np.array(ds), np.array(ps)


@torch.no_grad()
def paint(gen, vae, warm_feats, dreamed, dev):
    """Tokens at 10 Hz -> frames at 20 fps through the v0.1 generator. The warm-up frames are dropped."""
    model, mean, std, sigma = gen
    f = torch.cat([warm_feats.float(), dreamed]).repeat_interleave(2, 0)
    lat, _ = rollout(model, f, mean, std, sigma, dev)
    return batched(vae.decode, lat, dev)[2 * len(warm_feats):]


def load_inputs(video, log, start, seconds, enc, dev, speed=30.0, offset=0.0):
    """Returns real frames from the dream start on, warm-up tokens, and per-tick speed / route yaw.
    offset: metres left of the recorded path to start from (the warm-up frames are warped to that view)."""
    n = int(seconds * 10) // T * T
    f0 = int(start * FPS)
    assert f0 >= 2 * WARM, f"--start must be at least {2 * WARM / FPS} s: the models need that much real video first"
    real = read_video(video, start + seconds + 1)[f0 - 2 * WARM:f0 + 2 * n]
    warm = real[:2 * WARM:2]
    warm = batched(enc, warp(warm, torch.full((len(warm),), float(offset)), torch.zeros(len(warm))), dev)
    if log:
        z = np.load(log)
        tick = lambda k: torch.from_numpy(z[k][f0 - 2:f0 - 2 + 2 * n].reshape(-1, 2).mean(1)).float()
        sp, route = tick("speed"), tick("yaw")
    else:
        sp, route = torch.full((n,), speed), torch.zeros(n)
    return real[2 * WARM:], warm, sp, route


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--log", default=None)
    ap.add_argument("--world", default=os.path.join(ROOT, "weights", "realview_world.pt"))
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "weights", "realview.pt"))
    ap.add_argument("--weights", default=os.path.join(ROOT, "weights"))
    ap.add_argument("--out", default="dream.mp4")
    ap.add_argument("--start", type=float, default=10, help="second of the video where the dream takes over")
    ap.add_argument("--seconds", type=float, default=12)
    ap.add_argument("--steer", type=float, default=0.0, help="extra yaw rate in rad/s for 1.2 s, then back (+ = left)")
    ap.add_argument("--offset", type=float, default=0.0, help="start this many metres left of the recorded path")
    ap.add_argument("--policy", default=None, help="steering policy from scripts/closed_loop.py")
    ap.add_argument("--device", default=default_device())
    a = ap.parse_args()
    dev = a.device
    world, gen = load_world(a.world, dev), load_model(a.ckpt, dev)
    enc, vae = VisionEncoder().to(dev), Latents(a.weights).to(dev)
    real, warm, speed, route = load_inputs(a.video, a.log, a.start, a.seconds, enc, dev, offset=a.offset)
    policy = None
    if a.policy:
        policy = Head(frames=2, out=1).to(dev)
        policy.load_state_dict(torch.load(a.policy, map_location=dev))
    toks, d, psi = dream(world, warm, speed, route, dev, extra=nudge(len(speed), a.steer) if a.steer else None,
                         policy=policy, offset=a.offset)
    frames = paint(gen, vae, warm, toks, dev)
    side = torch.cat([real[:len(frames)], frames[:len(real)]], 3).permute(0, 2, 3, 1).mul(255).round().byte().numpy()
    write_mp4(a.out, side)
    print(json.dumps({"out": a.out, "seconds": len(frames) / FPS, "final_offset_m": round(float(d[-1]), 2),
                      "max_abs_offset_m": round(float(np.abs(d).max()), 2), "left_lane": bool(np.abs(d).max() > D_MAX)}))


if __name__ == "__main__":
    main()
