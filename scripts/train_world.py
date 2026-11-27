"""Train the world model: (last 0.4 s of encoder tokens, control) -> next 0.4 s of tokens.
Plain regression, unrolled over K chunks so the model learns to continue from its own output.
Resumes from --out/last.pt if it exists.

    python scripts/train_world.py --data data/prep --out runs/world --steps 6000 \
      --holdout "2018-08-17--12-07-08,2018-08-10--22-42-26"
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from realview.model import RealView  # noqa: E402
from realview.run import default_device  # noqa: E402
from realview.world import T, load_clips, predict, to_grid  # noqa: E402

K = 4  # chunks unrolled per sample (1.6 s)


def sample(clips, B, mean, std, dev):
    """Chunk k covers ticks [s, s+T): its pose is that of those ticks, its speed and route yaw rate are
    those of ticks [s-1, s+T-1), the interval that leads into them.
    Each clip exists twice: as recorded (pose 0) and warped along an off-route trajectory. Half the time
    the starting chunk comes from the other version, so the pose jumps and copying the previous chunk
    is wrong: the model has to read the pose it is given."""
    x, p, ctl = [], [], []
    for _ in range(B):
        c = clips[np.random.randint(len(clips))]
        off = np.random.rand() < 0.5                 # version of the targets
        poff = off ^ (np.random.rand() < 0.5)        # version of the starting chunk
        s = np.random.randint(T + 1, len(c["feat"]) - K * T)
        x.append(c["featw" if off else "feat"][s:s + K * T])
        p.append(c["featw" if poff else "feat"][s - T:s])
        ctl.append(torch.stack([c["speed"][s - 1:s - 1 + K * T], c["yaw"][s - 1:s - 1 + K * T],
                                c["d"][s:s + K * T] * off, c["psi"][s:s + K * T] * off]))
    n = lambda v: (to_grid(torch.stack(v).float().to(dev)) - mean) / std
    return n(x), n(p), torch.stack(ctl).to(dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/prep")
    ap.add_argument("--out", default="runs/world")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--holdout", default="")
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--device", default=default_device())
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    dev = a.device
    holdout = [h for h in a.holdout.split(",") if h]
    clips = load_clips(a.data, holdout, keep=False)
    allf = torch.cat([c[k][::10].float() for c in clips for k in ("feat", "featw")])
    mean = allf.mean((0, 1)).view(-1, 1, 1).to(dev)   # per encoder channel
    std = allf.std((0, 1)).view(-1, 1, 1).to(dev)
    print(f"clips={len(clips)} ticks={sum(len(c['feat']) for c in clips)}", flush=True)

    model = RealView(frames=T, lat_c=384, lat_h=6, lat_w=8, patch=1, enc_frames=T, d=a.d, depth=a.depth,
                     heads=a.d // 64, act_dim=4 * T).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0, betas=(0.9, 0.99))
    step = 0
    ck = os.path.join(a.out, "last.pt")
    if os.path.exists(ck):
        s = torch.load(ck, map_location=dev)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"])
        step = s["step"]
        print("resumed at", step, flush=True)
    print(f"params={sum(p.numel() for p in model.parameters()) / 1e6:.1f}M", flush=True)

    t0, hist = time.time(), []
    while step < a.steps:
        x, z, ctl = sample(clips, a.bs, mean, std, dev)
        for g in opt.param_groups:
            g["lr"] = a.lr * min(1.0, (step + 1) / 500) * (0.1 if step > 0.8 * a.steps else 1.0)
        errs = []
        for k in range(K):
            sl = slice(k * T, (k + 1) * T)
            z = predict(model, z, *ctl[:, :, sl].unbind(1))
            errs.append((z - x[:, sl]).pow(2).mean())
        loss = torch.stack(errs).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        hist.append([e.item() for e in errs])
        if step % 100 == 0:
            m = np.mean(hist, 0)
            print(json.dumps({"step": step, "mse_0.4s": round(float(m[0]), 4), f"mse_{0.4 * K:.1f}s": round(float(m[-1]), 4),
                              "it_s": round(100 / (time.time() - t0), 2)}), flush=True)
            hist, t0 = [], time.time()
        if step % 500 == 0 or step == a.steps:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step, "cfg": model.cfg,
                        "mean": mean.cpu(), "std": std.cpu(), "holdout": holdout}, ck + ".tmp")
            os.replace(ck + ".tmp", ck)


if __name__ == "__main__":
    main()
