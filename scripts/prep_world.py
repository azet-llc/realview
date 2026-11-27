"""Add what the world model needs to the arrays from prep.py:

  speed, yaw   ego speed (m/s) and yaw rate (rad/s, + = left) per frame, from the comma2k19 logs
  d, psi       a simulated off-route trajectory (metres left of the recorded path, heading error)
  extra        the extra yaw rate that produces it
  featw        encoder tokens of every frame warped to that off-route view

    python scripts/prep_world.py --videos videos --logs logs --out data/prep_world
    (logs = the three parquet files under data/ in huggingface.co/datasets/commaai/comma2k19)
"""
import argparse
import glob
import os
import sys

import numpy as np
import pyarrow.parquet as pq
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from realview.encoders import VisionEncoder  # noqa: E402
from realview.run import default_device  # noqa: E402
from realview.world import sloppy_driver, warp  # noqa: E402
from prep import frames  # noqa: E402


def actions(log, n):
    ft = np.array(log["global_pose__frame_times"])
    p, v = np.array(log["global_pose__frame_positions"]), np.array(log["global_pose__frame_velocities"])
    x, y, z = p[0]
    lon, lat = np.arctan2(y, x), np.arctan2(z, np.hypot(x, y))
    east = np.array([-np.sin(lon), np.cos(lon), 0])
    north = np.array([-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)])
    speed = np.linalg.norm(v, axis=1)
    yaw = np.convolve(np.gradient(np.unwrap(np.arctan2(v @ north, v @ east)), ft), np.ones(9) / 9, "same")
    # heading from GPS velocity is noise when the car crawls: fall back to steering-angle kinematics there
    steer = np.interp(ft, log["processed_log__CAN__steering_angle__t"], log["processed_log__CAN__steering_angle__value"])
    slow = 0.5 * speed * np.tan(np.deg2rad(steer) / 15.4) / 2.7   # 0.5 = fitted against the GPS yaw rate at speed
    yaw = np.where(speed > 5, yaw, slow)
    pad = lambda a: np.concatenate([a, np.repeat(a[-1:], max(0, n - len(a)))])[:n]
    return pad(speed).astype(np.float32), pad(yaw).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True)
    ap.add_argument("--logs", required=True)
    ap.add_argument("--out", default="data/prep_world")
    ap.add_argument("--shard", default="0/1", help="k/n: handle every n-th clip starting at k (run n copies in parallel)")
    ap.add_argument("--device", default=default_device())
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    enc = VisionEncoder().to(a.device)
    logs = {}
    for f in sorted(glob.glob(os.path.join(a.logs, "*.parquet"))):
        t = pq.read_table(f, columns=["segment_id", "log"])
        logs.update(zip(t.column(0).to_pylist(), t.column(1).to_pylist()))
    paths = sorted(glob.glob(os.path.join(a.videos, "**", "*.hevc"), recursive=True))
    k, n = map(int, a.shard.split("/"))
    for i, p in list(enumerate(paths))[k::n]:
        rel = os.path.relpath(p, a.videos)
        dst = os.path.join(a.out, rel.replace("/", "__").replace("|", "_") + ".world.npz")
        if os.path.exists(dst):
            continue
        fr = np.stack(list(frames(p)))
        speed, yaw = actions(logs[os.path.dirname(rel)], len(fr))
        d, psi, extra = sloppy_driver(speed, np.random.default_rng(i))
        out = []
        for s in range(0, len(fr), 64):
            x = torch.from_numpy(fr[s:s + 64]).to(a.device).permute(0, 3, 1, 2).float() / 255
            dd, pp = (torch.from_numpy(v[s:s + 64]).float().to(a.device) for v in (d, psi))
            out.append(enc(warp(x, dd, pp)).half().cpu())
        np.savez(dst, featw=torch.cat(out).numpy(), speed=speed, yaw=yaw,
                 d=d.astype(np.float32), psi=psi.astype(np.float32), extra=extra.astype(np.float32))
        print(f"[{i + 1}/{len(paths)}] {rel} d_std={d.std():.2f} d_max={np.abs(d).max():.2f} psi_std={psi.std():.4f}", flush=True)


if __name__ == "__main__":
    main()
