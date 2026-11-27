"""Conditional mean-flow transformer.

Generates an 8-frame latent chunk from (a) the previous generation with added
noise and (b) frozen vision-encoder features. It never sees camera pixels.
Attention is written out by hand because PyTorch's fused SDPA has no
forward-mode AD on MPS, and the mean-flow loss needs a JVP through the net.
"""
import math
import torch
import torch.nn as nn


def t_embed(t, dim):
    half = dim // 2
    f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t[:, None].float() * f[None]  # no x1000: d/dt of the embedding feeds the mean-flow target
    return torch.cat([a.sin(), a.cos()], -1)


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.n1 = nn.LayerNorm(d, elementwise_affine=False)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d, elementwise_affine=False)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(approximate="tanh"), nn.Linear(4 * d, d))
        self.ada = nn.Linear(d, 6 * d)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def forward(self, x, c):
        s1, b1, g1, s2, b2, g2 = self.ada(c)[:, None].chunk(6, -1)
        B, N, D = x.shape
        h = self.n1(x) * (1 + s1) + b1
        q, k, v = self.qkv(h).view(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = ((q @ k.transpose(-1, -2)) * (D // self.h) ** -0.5).softmax(-1) @ v
        x = x + g1 * self.proj(a.transpose(1, 2).reshape(B, N, D))
        h = self.n2(x) * (1 + s2) + b2
        return x + g2 * self.mlp(h)


class RealView(nn.Module):
    """z: (B, F, C, H, W) latents; feats: (B, E, T, Df) encoder tokens."""

    def __init__(self, frames=8, lat_c=4, lat_h=24, lat_w=32, patch=4,
                 enc_frames=4, enc_tokens=48, enc_dim=384, d=384, depth=12, heads=6, act_dim=0):
        super().__init__()
        self.cfg = dict(frames=frames, lat_c=lat_c, lat_h=lat_h, lat_w=lat_w, patch=patch,
                        enc_frames=enc_frames, enc_tokens=enc_tokens, enc_dim=enc_dim,
                        d=d, depth=depth, heads=heads, act_dim=act_dim)
        self.F, self.C, self.H, self.W, self.p, self.d = frames, lat_c, lat_h, lat_w, patch, d
        self.n_sp = (lat_h // patch) * (lat_w // patch)
        self.inp = nn.Linear(lat_c * patch * patch, d)
        self.pos = nn.Parameter(torch.randn(1, self.n_sp, d) * 0.02)
        self.fpos = nn.Parameter(torch.randn(1, frames, 1, d) * 0.02)
        self.cond = nn.Sequential(nn.LayerNorm(enc_dim), nn.Linear(enc_dim, d))
        self.cpos = nn.Parameter(torch.randn(1, enc_frames, enc_tokens, d) * 0.02)
        # encoder tokens sit on the same 8x6 grid as the latent patches: add them in place
        self.align = nn.Linear(d, d)
        self.temb = nn.Sequential(nn.Linear(2 * 256, d), nn.SiLU(), nn.Linear(d, d))
        # world-model mode: a per-chunk control vector (speed, route yaw rate, pose) goes in twice:
        # into the time embedding, and each frame's slice added onto that frame's tokens
        self.act = nn.Sequential(nn.Linear(act_dim, d), nn.SiLU(), nn.Linear(d, d)) if act_dim else None
        self.act_tok = nn.Linear(act_dim // frames, d) if act_dim else None
        self.blocks = nn.ModuleList([Block(d, heads) for _ in range(depth)])
        self.out_n = nn.LayerNorm(d, elementwise_affine=False)
        self.out_ada = nn.Linear(d, 2 * d)
        self.out = nn.Linear(d, lat_c * patch * patch)
        for m in (self.out_ada, self.out):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, z, r, t, feats, act=None):
        B, F, C, H, W = z.shape
        p = self.p
        x = z.reshape(B, F, C, H // p, p, W // p, p).permute(0, 1, 3, 5, 2, 4, 6)
        x = self.inp(x.reshape(B, F, self.n_sp, C * p * p)) + self.pos[:, None] + self.fpos
        cf = self.cond(feats)
        x = x + self.align(cf).repeat_interleave(F // cf.shape[1], 1)
        if self.act_tok is not None:  # act is (B, channels * F), channel-major
            x = x + self.act_tok(act.view(B, -1, F).transpose(1, 2))[:, :, None]
        x = x.reshape(B, F * self.n_sp, self.d)
        c = (cf + self.cpos).reshape(B, -1, self.d)
        n_c = c.shape[1]
        x = torch.cat([c, x], 1)
        e = self.temb(torch.cat([t_embed(t, 256), t_embed(t - r, 256)], -1))
        if self.act is not None:
            e = e + self.act(act)
        for blk in self.blocks:
            x = blk(x, e)
        s, b = self.out_ada(e)[:, None].chunk(2, -1)
        x = self.out(self.out_n(x[:, n_c:]) * (1 + s) + b)
        x = x.reshape(B, F, H // p, W // p, C, p, p).permute(0, 1, 4, 2, 5, 3, 6)
        return x.reshape(B, F, C, H, W)
