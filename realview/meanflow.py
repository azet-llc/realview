"""Mean-flow objective (Geng et al., 2025, arXiv:2505.13447) on a bridge whose
source is the previous generation plus noise, not pure noise:

    src = prev + sigma * eps,   z_t = (1 - t) * x + t * src,   v = src - x
    u_target = v - (t - r) * d/dt u(z_t, r, t)      (total derivative via JVP)

One step at inference: x = src - u(src, r=0, t=1).
"""
import torch
from torch.func import jvp


def sample_rt(B, device, p_neq=0.25):
    a = torch.sigmoid(torch.randn(B, device=device) - 0.4)
    b = torch.sigmoid(torch.randn(B, device=device) - 0.4)
    t, r = torch.maximum(a, b), torch.minimum(a, b)
    eq = torch.rand(B, device=device) > p_neq
    r = torch.where(eq, t, r)
    return r, t


def make_source(prev, sigma):
    return prev + sigma * torch.randn_like(prev)


def loss_fn(model, x, src, feats, gamma=0.5, c=1e-3):
    B = x.shape[0]
    r, t = sample_rt(B, x.device)
    tb = t.view(B, 1, 1, 1, 1)
    z = (1 - tb) * x + tb * src
    v = src - x
    fn = lambda z_, r_, t_: model(z_, r_, t_, feats)
    u, dudt = jvp(fn, (z, r, t), (v, torch.zeros_like(r), torch.ones_like(t)))
    target = (v - (t - r).view(B, 1, 1, 1, 1) * dudt).detach()
    err = (u - target).pow(2).flatten(1).mean(1)
    w = 1.0 / (err.detach() + c).pow(gamma)  # adaptive weighting from the paper
    return (w * err).mean(), err.mean()


@torch.no_grad()
def generate(model, src, feats, steps=1):
    B = src.shape[0]
    z = src
    ts = torch.linspace(1, 0, steps + 1, device=src.device)
    for i in range(steps):
        t = ts[i].expand(B)
        r = ts[i + 1].expand(B)
        z = z - (t - r).view(B, 1, 1, 1, 1) * model(z, r, t, feats)
    return z
