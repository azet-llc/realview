"""Frozen pieces: a public vision encoder (stand-in for Tesla's Universal Vision
Encoder) and a tiny VAE that maps 256x192 frames to 4x24x32 latents."""
import os
import sys
import torch
import torch.nn.functional as Fn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "third_party"))
from taesd import TAESD  # noqa: E402

W, H = 256, 192            # output frame size (4:3, matches the comma camera)
ENC_W, ENC_H = 224, 168     # 16x12 DINOv2 patches -> pooled to 8x6 = 48 tokens
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class VisionEncoder(torch.nn.Module):
    def __init__(self, name="facebook/dinov2-small"):
        super().__init__()
        from transformers import AutoModel
        self.m = AutoModel.from_pretrained(name).eval()
        for p in self.m.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, img):  # img: (N,3,h,w) in [0,1]
        x = Fn.interpolate(img, (ENC_H, ENC_W), mode="bilinear", antialias=True, align_corners=False)
        x = (x - MEAN.to(x)) / STD.to(x)
        tok = self.m(pixel_values=x).last_hidden_state[:, 1:]  # drop CLS
        N, _, D = tok.shape
        tok = tok.transpose(1, 2).reshape(N, D, ENC_H // 14, ENC_W // 14)
        return Fn.avg_pool2d(tok, 2).flatten(2).transpose(1, 2)  # (N,48,384)


class Latents(torch.nn.Module):
    def __init__(self, wdir):
        super().__init__()
        self.t = TAESD(os.path.join(wdir, "taesd_encoder.pth"), os.path.join(wdir, "taesd_decoder.pth")).eval()
        for p in self.t.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def encode(self, img):
        return self.t.encoder(img)

    @torch.no_grad()
    def decode(self, lat):
        return self.t.decoder(lat).clamp(0, 1)
