"""Does the world model put the car where it is told to? Three checks on the held-out drives.

  1. lane probe   a small reader of encoder tokens -> (d, psi), trained on real warped frames of the training
                  drives and scored on real warped frames of the held-out drives. It is the measuring stick below.
  2. open loop    dream forward from a real off-route start and compare with the real (warped) future tokens
                  at 0.4 .. 3.2 s: with the true pose, with the pose withheld, with the route zeroed,
                  with another drive's speed and route.
  3. steering     start on the recorded path, steer along a wandering course the logs never contained, and compare
                  where the kinematic model says the car is with where the probe sees it in the dream.

    python scripts/world_eval.py --ckpt weights/realview_world.pt --data data/prep \
      --holdout "2018-08-17--12-07-08,2018-08-10--22-42-26"
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from realview.run import default_device  # noqa: E402
from realview.world import PSI_S, Head, T, kin, load_clips, load_world, sloppy_driver, step, to_grid  # noqa: E402


def corr(a, b):
    return float(np.corrcoef(a.flatten().cpu().numpy(), b.flatten().cpu().numpy())[0, 1])


def probe_set(clips, mean, std, dev):
    z = torch.cat([c[k] for c in clips for k in ("featw", "feat")]).float()
    y = torch.cat([torch.stack([c["d"], c["psi"] * PSI_S], 1) * w for c in clips for w in (1, 0)])
    sp = torch.cat([c["speed"] for c in clips for _ in (0, 1)])
    return z, y.to(dev), sp.to(dev), lambda i: (to_grid(z[i].to(dev))[:, None] - mean) / std


def train_probe(train, mean, std, dev, steps=3000):
    torch.manual_seed(0)
    probe = Head(frames=1, out=2).to(dev)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-2)
    z, y, sp, get = probe_set(train, mean, std, dev)
    for _ in range(steps):
        i = torch.randint(len(z), (256,))
        loss = (probe(get(i), sp[i]) - y[i]).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return probe.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="data/prep")
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--probe", default="runs/world/probe.pt")
    ap.add_argument("--device", default=default_device())
    a = ap.parse_args()
    dev = a.device
    holdout = a.holdout.split(",")
    model, mean, std = load_world(a.ckpt, dev)
    test = load_clips(a.data, holdout, keep=True)
    norm = lambda f: (to_grid(f.float().to(dev)) - mean) / std
    out = {"held_out_clips": len(test)}

    # 1. lane probe
    if os.path.exists(a.probe):
        probe = Head(frames=1, out=2).to(dev)
        probe.load_state_dict(torch.load(a.probe, map_location=dev))
        probe.eval()
    else:
        probe = train_probe(load_clips(a.data, holdout, keep=False), mean, std, dev)
        os.makedirs(os.path.dirname(a.probe), exist_ok=True)
        torch.save(probe.state_dict(), a.probe)
    with torch.no_grad():
        z, y, sp, get = probe_set(test, mean, std, dev)
        idx = torch.cat([torch.arange(len(c["d"])) + 2 * sum(len(t["d"]) for t in test[:j]) for j, c in enumerate(test)])
        pr = torch.cat([probe(get(idx[i:i + 512]), sp[idx[i:i + 512]]) for i in range(0, len(idx), 512)])
        out["probe_real_held_out"] = {"r_offset": round(corr(pr[:, 0], y[idx, 0]), 3),
                                      "r_heading": round(corr(pr[:, 1], y[idx, 1]), 3),
                                      "offset_mae_m": round(float((pr[:, 0] - y[idx, 0]).abs().mean()), 3)}

    # 2. open loop against the real off-route future
    torch.manual_seed(0)
    H = 8
    starts = [(j, s) for j, c in enumerate(test) for s in range(2 * T, len(c["feat"]) - H * T - 1, 20)]
    drive = [c["name"].split("__")[0] for c in test]
    other = [next(k for k in range(len(test)) if drive[k] != drive[j]) for j in range(len(test))]
    errs = {k: torch.zeros(H) for k in ("true_pose", "pose_withheld", "route_zeroed", "other_drive_speed_route", "copy_last_chunk")}
    with torch.no_grad():
        for b in range(0, len(starts), 128):
            st = starts[b:b + 128]
            z0 = torch.stack([norm(test[j]["featw"][s - T:s]) for j, s in st])
            real = torch.stack([norm(test[j]["featw"][s:s + H * T]) for j, s in st])
            get = lambda key, lag, pick=int: torch.stack([test[pick(j)][key][s - lag:s - lag + H * T] for j, s in st]).to(dev)
            pose = (get("d", 0), get("psi", 0))
            sets = {"true_pose": (get("speed", 1), get("yaw", 1), *pose),
                    "pose_withheld": (get("speed", 1), get("yaw", 1), 0 * pose[0], 0 * pose[1]),
                    "route_zeroed": (get("speed", 1), 0 * get("yaw", 1), *pose),
                    "other_drive_speed_route": (get("speed", 1, lambda j: other[j]), get("yaw", 1, lambda j: other[j]), *pose)}
            for name, ctl in sets.items():
                z = z0
                for h in range(H):
                    sl = slice(h * T, (h + 1) * T)
                    z = step(model, z, *(c[:, sl] for c in ctl))
                    errs[name][h] += (z - real[:, sl]).pow(2).mean((1, 2, 3, 4)).sum().cpu()
            for h in range(H):
                errs["copy_last_chunk"][h] += (z0 - real[:, h * T:(h + 1) * T]).pow(2).mean((1, 2, 3, 4)).sum().cpu()
    out["open_loop_token_mse"] = {"starts": len(starts), "horizons_s": [0.4, 0.8, 1.6, 3.2],
                                  **{k: [round(float(v[h] / len(starts)), 3) for h in (0, 1, 3, 7)] for k, v in errs.items()}}

    # 3. steering it somewhere the logs never went
    H = 20
    starts = [(j, s) for j, c in enumerate(test) for s in range(2 * T, len(c["feat"]) - H * T - 1, 40)]
    rng = np.random.default_rng(1)
    with torch.no_grad():
        z = torch.stack([norm(test[j]["feat"][s - T:s]) for j, s in starts])
        speed = torch.stack([test[j]["speed"][s - 1:s - 1 + H * T] for j, s in starts])
        route = torch.stack([test[j]["yaw"][s - 1:s - 1 + H * T] for j, s in starts]).to(dev)
        extra = torch.from_numpy(np.stack([sloppy_driver(sp.numpy(), rng, dt=0.1, on_path=True)[2] for sp in speed])).float().to(dev)
        speed = speed.to(dev)
        d = psi = torch.zeros(len(starts), device=dev)
        kd, kp, pd, pp = [], [], [], []
        for h in range(H):
            sl = slice(h * T, (h + 1) * T)
            ds, ps = kin(d, psi, speed[:, sl], route[:, sl] + extra[:, sl], route[:, sl])
            z = step(model, z, speed[:, sl], route[:, sl], ds, ps)
            d, psi = ds[:, -1], ps[:, -1]
            p = probe(z[:, -1:], speed[:, sl][:, -1])
            kd.append(d); kp.append(psi * PSI_S); pd.append(p[:, 0]); pp.append(p[:, 1])
    kd, kp, pd, pp = (torch.stack(v) for v in (kd, kp, pd, pp))
    hs = {"0.4-2s": slice(0, 5), "2-4s": slice(5, 10), "4-8s": slice(10, 20)}
    out["steered_dream"] = {"episodes": len(starts), "kinematic_offset_std_m": round(float(kd.std()), 3),
                            "r_offset": {k: round(corr(pd[s], kd[s]), 3) for k, s in hs.items()},
                            "r_heading": {k: round(corr(pp[s], kp[s]), 3) for k, s in hs.items()},
                            "offset_mae_m": {k: round(float((pd[s] - kd[s]).abs().mean()), 3) for k, s in hs.items()}}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
