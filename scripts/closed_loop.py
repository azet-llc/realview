"""Closed loop: a steering policy drives inside the world model and has to get back into its lane.

Every 0.4 s the policy is handed the route's yaw rate (the way a driving stack is handed a planned path)
and reads the last two dreamed encoder frames. It answers with a correction to that yaw rate. A 2-state
kinematic model turns the steering into a pose and the world model dreams the next 0.4 s at that pose.
Episodes start from real off-route frames, 0.3 to 1.0 m beside the recorded path. More than 1.2 m off
counts as leaving the lane.

  bc    behaviour cloning on the logged drives. The human never leaves the path, so the correction it learns is zero
  aug   no world model: cloning on the warped frames, labelled with a hand-written steer-back rule
  ppo   reinforcement learning inside the world model, starting from bc (training drives only, no rule, no labels)
  eval  all three on starts from the held-out drives, plus a check on real warped pixels

    python scripts/closed_loop.py bc   --world weights/realview_world.pt --holdout A,B
    python scripts/closed_loop.py aug  --world weights/realview_world.pt --holdout A,B
    python scripts/closed_loop.py ppo  --world weights/realview_world.pt --holdout A,B
    python scripts/closed_loop.py eval --world weights/realview_world.pt --holdout A,B
"""
import argparse
import copy
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from realview.run import default_device  # noqa: E402
from realview.world import D_MAX, YAW_S, Head, T, kin, load_clips, load_world, restore, step, to_grid  # noqa: E402

START = (0.3, 1.0)  # metres off the path at the start of an episode


class Env:
    """A batch of dream drives. Observation = (last two dreamed frames, speed). Action = yaw-rate correction."""

    def __init__(self, clips, world, dev, horizon):
        self.clips, self.dev, self.h = clips, dev, horizon
        self.model, self.mean, self.std = world

    def reset(self, starts):
        c, dev = self.clips, self.dev
        self.z = (to_grid(torch.stack([c[j]["featw"][s - T:s] for j, s in starts]).float().to(dev)) - self.mean) / self.std
        n = self.h * T
        self.speed = torch.stack([c[j]["speed"][s - 1:s - 1 + n] for j, s in starts]).to(dev)
        self.route = torch.stack([c[j]["yaw"][s - 1:s - 1 + n] for j, s in starts]).to(dev)
        self.d, self.psi = (torch.stack([c[j][k][s - 1] for j, s in starts]).to(dev) for k in ("d", "psi"))
        self.t = 0
        self.alive = torch.ones(len(starts), dtype=torch.bool, device=dev)
        return self.obs()

    def obs(self):
        return self.z[:, -2:], self.speed[:, min(self.t, self.h - 1) * T]

    def step(self, fix):
        """fix: (B,) rad/s added to the route's yaw rate for 0.4 s. Returns obs, reward, alive-before-this-step."""
        sl = slice(self.t * T, (self.t + 1) * T)
        ds, ps = kin(self.d, self.psi, self.speed[:, sl], self.route[:, sl] + fix[:, None], self.route[:, sl])
        self.z = step(self.model, self.z, self.speed[:, sl], self.route[:, sl], ds, ps)
        self.d, self.psi = ds[:, -1], ps[:, -1]
        was = self.alive
        off = self.d.abs() > D_MAX
        reward = torch.where(off, torch.full_like(self.d, -10.0), 1 - (self.d / D_MAX) ** 2)
        self.alive = was & ~off
        self.t += 1
        return self.obs(), reward, was


def starts_for(clips, horizon, every):
    return [(j, s) for j, c in enumerate(clips) for s in range(2 * T, len(c["feat"]) - horizon * T - 1, every)
            if START[0] <= abs(float(c["d"][s - 1])) <= START[1]]


def train_bc(clips, mean, std, dev, steps=2000, aug=False):
    torch.manual_seed(0); np.random.seed(0)
    pol = Head(frames=2, out=1).to(dev)
    opt = torch.optim.AdamW(pol.parameters(), lr=1e-3, weight_decay=1e-2)
    key = "featw" if aug else "feat"
    for i in range(steps):
        b = [(j, np.random.randint(2, len(clips[j]["feat"]))) for j in np.random.randint(len(clips), size=256)]
        z = (to_grid(torch.stack([clips[j][key][s - 2:s] for j, s in b]).float().to(dev)) - mean) / std
        sp = torch.stack([clips[j]["speed"][s - 1] for j, s in b])
        # the human drives the route, so the correction to clone is zero. aug: the label a human never gave
        y = restore(*(torch.stack([clips[j][k][s - 1] for j, s in b]) for k in ("d", "psi")), sp) if aug else 0 * sp
        loss = (pol(z, sp.to(dev))[:, 0] - y.to(dev) * YAW_S).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (i + 1) % 500 == 0:
            print(json.dumps({"step": i + 1, "loss": round(loss.item(), 5)}), flush=True)
    return pol


def train_ppo(pol, clips, world, dev, iters, envs=256, horizon=25, gamma=0.97, lam=0.95, clip=0.2):
    torch.manual_seed(0); np.random.seed(0)
    pol = copy.deepcopy(pol)
    val = Head(frames=2, out=1).to(dev)
    log_std = torch.full((1,), np.log(0.15), device=dev, requires_grad=True)   # 0.15 / YAW_S = 0.0075 rad/s
    opt = torch.optim.Adam([{"params": pol.parameters(), "lr": 3e-4}, {"params": val.parameters(), "lr": 1e-3},
                            {"params": [log_std], "lr": 1e-3}])
    env, st = Env(clips, world, dev, horizon), starts_for(clips, horizon, 1)
    for it in range(iters):
        opt.param_groups[0]["lr"] = 3e-4 * (1 - it / iters)   # the policy settles as the step shrinks
        with torch.no_grad():
            (z, sp) = env.reset([st[k] for k in np.random.randint(len(st), size=envs)])
            Z, S, A, LP, R, V, M = [], [], [], [], [], [], []
            for _ in range(horizon):
                mu = pol(z, sp)[:, 0]
                act = mu + log_std.exp() * torch.randn_like(mu)
                lp = -0.5 * ((act - mu) / log_std.exp()) ** 2 - log_std
                Z.append(z); S.append(sp); A.append(act); LP.append(lp); V.append(val(z, sp)[:, 0])
                (z, sp), r, was = env.step(act / YAW_S)
                R.append(r); M.append(was.float())
            adv, g, nxt = [], torch.zeros(envs, device=dev), val(z, sp)[:, 0]
            for t in reversed(range(horizon)):
                live_next = M[t + 1] if t + 1 < horizon else env.alive.float()
                delta = R[t] + gamma * nxt * live_next - V[t]
                g = delta + gamma * lam * live_next * g
                adv.insert(0, g); nxt = V[t]
            Z, S, A, LP, V, M, adv = (torch.cat(v) for v in (Z, S, A, LP, V, M, adv))
            keep = M.bool()   # steps after the car left the lane are not experience
            Z, S, A, LP, ret, adv = Z[keep], S[keep], A[keep], LP[keep], (adv + V)[keep], adv[keep]
            adv = (adv - adv.mean()) / (adv.std() + 1e-6)
        for _ in range(4):
            for idx in torch.randperm(len(Z), device=dev).split(1600):
                mu = pol(Z[idx], S[idx])[:, 0]
                lp = -0.5 * ((A[idx] - mu) / log_std.exp()) ** 2 - log_std
                ratio = (lp - LP[idx]).exp()
                loss = -torch.min(ratio * adv[idx], ratio.clamp(1 - clip, 1 + clip) * adv[idx]).mean() \
                    + 0.5 * (val(Z[idx], S[idx])[:, 0] - ret[idx]).pow(2).mean()
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(list(pol.parameters()) + list(val.parameters()), 1.0)
                opt.step()
        if (it + 1) % 5 == 0:
            print(json.dumps({"ppo_iter": it + 1, "left_lane": round(1 - env.alive.float().mean().item(), 3),
                              "mean_reward": round(torch.stack(R).mean().item(), 3),
                              "end_abs_offset_m": round(env.d.abs()[env.alive].mean().item(), 3),
                              "std_rad_s": round(log_std.exp().item() / YAW_S, 4)}), flush=True)
    return pol


@torch.no_grad()
def drive(fix_fn, clips, world, dev, horizon=50, every=10, seed=0):
    """20 s episodes from real off-route starts. fix_fn(z, speed, env) -> yaw-rate correction (rad/s)."""
    torch.manual_seed(seed)
    env, st = Env(clips, world, dev, horizon), starts_for(clips, horizon, every)
    obs = env.reset(st)
    d0 = env.d.abs().mean().item()
    for _ in range(horizon):
        obs, _, _ = env.step(fix_fn(*obs, env))
    ok = env.alive.float()
    return {"episodes": len(st), "start_abs_offset_m": round(d0, 2),
            "left_lane_pct": round(100 * (1 - ok.mean().item()), 1),
            "back_within_0.3m_pct": round(100 * (ok * (env.d.abs() < 0.3)).mean().item(), 1),
            "end_abs_offset_m": round((env.d.abs() * ok).sum().item() / max(ok.sum().item(), 1), 2)}


@torch.no_grad()
def real_pixels(pol, clips, mean, std, dev):
    """On real warped held-out frames (no world model): does the correction point back toward the path?"""
    st = [(j, s) for j, c in enumerate(clips) for s in range(2, len(c["feat"]), 2)]
    z = (to_grid(torch.stack([clips[j]["featw"][s - 2:s] for j, s in st]).float().to(dev)) - mean) / std
    sp = torch.stack([clips[j]["speed"][s - 1] for j, s in st])
    fix = pol(z, sp.to(dev))[:, 0].cpu() / YAW_S
    d, psi = (torch.stack([clips[j][k][s - 1] for j, s in st]) for k in ("d", "psi"))
    r = lambda a, b: round(float(np.corrcoef(a.numpy(), b.numpy())[0, 1]), 3)
    return {"frames": len(st), "r_with_steer_back_rule": r(fix, restore(d, psi, sp)), "r_with_minus_offset": r(fix, -d)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["bc", "aug", "ppo", "eval"])
    ap.add_argument("--world", required=True)
    ap.add_argument("--data", default="data/prep")
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--out", default="runs/world")
    ap.add_argument("--iters", type=int, default=150)
    ap.add_argument("--device", default=default_device())
    a = ap.parse_args()
    dev, holdout = a.device, a.holdout.split(",")
    world = load_world(a.world, dev)
    _, mean, std = world

    def load(name):
        p = Head(frames=2, out=1).to(dev)
        p.load_state_dict(torch.load(os.path.join(a.out, name), map_location=dev))
        return p.eval()

    if a.mode in ("bc", "aug"):
        pol = train_bc(load_clips(a.data, holdout, keep=False), mean, std, dev, aug=a.mode == "aug")
        torch.save(pol.state_dict(), os.path.join(a.out, f"policy_{a.mode}.pt"))
    elif a.mode == "ppo":
        pol = train_ppo(load("policy_bc.pt"), load_clips(a.data, holdout, keep=False), world, dev, a.iters)
        torch.save(pol.state_dict(), os.path.join(a.out, "policy_rl.pt"))
    else:
        test = load_clips(a.data, holdout, keep=True)
        out = {}
        for name in ("policy_bc.pt", "policy_aug.pt", "policy_rl.pt"):
            pol = load(name)
            out[name] = {"dream_closed_loop": drive(lambda z, sp, env: pol(z, sp)[:, 0] / YAW_S, test, world, dev),
                         "real_warped_pixels": real_pixels(pol, test, mean, std, dev)}
        # references that read the true pose instead of the picture
        out["no_correction"] = drive(lambda z, sp, env: 0 * sp, test, world, dev)
        out["steer_back_rule_on_true_pose"] = drive(lambda z, sp, env: restore(env.d, env.psi, sp), test, world, dev)
        print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
