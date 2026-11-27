"""World-model mode: the same transformer, run in encoder-feature space as a latent predictor.

    state   = the last 0.4 s of vision-encoder tokens (4 ticks x 48 tokens)
    control = where the car will be over the next 0.4 s: speed, the yaw rate of the recorded drive
              (the road to follow), and its pose relative to that recorded path (offset d, heading error psi)
    output  = the next 0.4 s of tokens; the frozen v0.1 generator paints them

The pose comes from a 2-state kinematic model driven by the steering action:

    psi' = yaw_ego - yaw_route        (heading error, + = pointing left)
    d'   = speed * sin(psi)           (lateral offset in metres, + = left of the recorded path)

Off-route views do not exist in the logs, so prep_world.py synthesises them with a ground-plane warp.
"""
import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from .model import RealView

T = 4            # encoder ticks per chunk (10 Hz)
DT = 0.1         # seconds per tick
YAW_S, SPEED_S, PSI_S = 20.0, 1 / 30.0, 50.0   # bring yaw rate (rad/s), speed (m/s), heading error (rad) to ~unit range
FOCAL = 200.1    # comma EON focal length (910 px at 1164 wide) at 256 wide
ROW_M = 0.8206   # px of ground shift per metre of offset per row below the horizon (camera height 1.22 m)
HOOD = 0.76     # the hood starts at this fraction of the image height
D_MAX = 1.2      # metres off the recorded path that counts as leaving the lane


def warp(img, d, psi):
    """View from a camera moved d metres left and yawed psi rad left. img (N,3,H,W); d, psi (N,).
    Yaw slides the image; offset shears the ground plane only (rows below the horizon).
    The car's own hood (bottom rows) stays put."""
    N, _, H, W = img.shape
    v = torch.arange(H, device=img.device).float()
    shift = (FOCAL * psi[:, None] + d[:, None] * ROW_M * (v - H / 2).clamp_min(0)[None]) * (W / 256)
    shift = shift * (v < HOOD * H)[None]
    u = torch.arange(W, device=img.device).float()[None, None] - shift[:, :, None]
    grid = torch.stack([(u + 0.5) / W * 2 - 1, ((v + 0.5) / H * 2 - 1)[None, :, None].expand(N, H, W)], -1)
    return Fn.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=False)


def restore(d, psi, speed):
    """Yaw-rate correction of a damped steer-back controller (also the reference recovery command)."""
    return -0.6 / speed.clip(5.0) * d - 0.8 * psi   # numpy or torch


def sloppy_driver(speed, rng, n_std=0.012, tau=2.0, dt=0.05, on_path=False):
    """A driver who wanders off the recorded path and steers back. Returns d, psi and the extra yaw rate."""
    n = len(speed)
    d, psi, extra = np.zeros(n), np.zeros(n), np.zeros(n)
    x = np.zeros(2) if on_path else np.array([rng.uniform(-0.6, 0.6), rng.uniform(-0.01, 0.01)])
    noise = 0.0
    for i in range(n):
        d[i], psi[i] = x
        noise += dt * (-noise / tau) + n_std * np.sqrt(2 * dt / tau) * rng.standard_normal()
        extra[i] = restore(x[0], x[1], speed[i]) + noise
        x = x + dt * np.array([speed[i] * np.sin(x[1]), extra[i]])
    return d, psi, extra


def to_grid(feat):
    """(..., 48, 384) tokens -> (..., 384, 6, 8)"""
    return feat.transpose(-1, -2).reshape(*feat.shape[:-2], feat.shape[-1], 6, 8)


def to_tokens(z):
    """(..., 384, 6, 8) -> (..., 48, 384)"""
    return z.flatten(-2).transpose(-1, -2)


def control_vec(speed, route, d, psi):
    """(B,T) each -> (B, 4T) conditioning vector"""
    return torch.cat([speed * SPEED_S, route * YAW_S, d, psi * PSI_S], -1)


def load_world(ckpt, dev):
    s = torch.load(ckpt, map_location="cpu")
    m = RealView(**s["cfg"])
    m.load_state_dict(s["model"])
    return m.to(dev).eval(), s["mean"].to(dev), s["std"].to(dev)


def predict(model, z_prev, speed, route, d, psi):
    """One 0.4 s tick of the simulator. z_prev: normalised (B,T,384,6,8); the rest (B,T).
    The network outputs the change from the previous chunk (the mean-flow one-step map with the noise off)."""
    B = z_prev.shape[0]
    zero, one = torch.zeros(B, device=z_prev.device), torch.ones(B, device=z_prev.device)
    return z_prev - model(z_prev, zero, one, to_tokens(z_prev), act=control_vec(speed, route, d, psi))


step = torch.no_grad()(predict)


def kin(d, psi, speed, yaw_ego, yaw_route):
    """Advance the kinematic model through one chunk from the pose (B,) at its start.
    speed and yaw rates are (B,T). Returns the pose after every tick, (B,T) each."""
    ds, ps = [], []
    for i in range(speed.shape[1]):
        d = d + DT * speed[:, i] * torch.sin(psi)
        psi = psi + DT * (yaw_ego[:, i] - yaw_route[:, i])
        ds.append(d); ps.append(psi)
    return torch.stack(ds, 1), torch.stack(ps, 1)


def load_clips(data, holdout, keep):
    """Arrays from prep.py + prep_world.py at 10 Hz. keep=True: only held-out drives; False: only training drives."""
    out = []
    for p in sorted(glob.glob(os.path.join(data, "*.npz"))):
        w = os.path.join(data.rstrip("/") + "_world", os.path.basename(p).replace(".npz", ".world.npz"))
        if any(h in os.path.basename(p) for h in holdout) != keep or not os.path.exists(w):
            continue
        a, b = np.load(p), np.load(w)
        n = min(len(a["feat"]), len(b["featw"])) // 2 * 2
        c = {"name": os.path.basename(p).split("__video")[0],
             "feat": torch.from_numpy(a["feat"][:n:2]), "featw": torch.from_numpy(b["featw"][:n:2])}
        for k in ("speed", "yaw", "extra"):                       # rates: average the two frames of a tick
            c[k] = torch.from_numpy(b[k][:n].reshape(-1, 2).mean(1)).float()
        for k in ("d", "psi"):                                    # states: value at the tick
            c[k] = torch.from_numpy(b[k][:n:2]).float()
        out.append(c)
    return out


class Head(nn.Module):
    """Small reader of encoder frames. The lane probe (d, psi) and the driving policy share it."""

    def __init__(self, frames=2, out=1):
        super().__init__()
        self.tok = nn.Linear(384, 32)
        self.net = nn.Sequential(nn.Linear(frames * 48 * 32 + 1, 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(),
                                 nn.Linear(256, out))

    def forward(self, z, speed):
        """z: normalised (B,frames,384,6,8); speed (B,) m/s"""
        x = self.tok(to_tokens(z)).flatten(1)
        return self.net(torch.cat([x, speed[:, None] * SPEED_S], 1))
