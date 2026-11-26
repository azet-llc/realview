"""Strip a training checkpoint down to the released file (fp16 weights, no optimizer).

    python scripts/export.py runs/base/last.pt weights/realview.pt --which ema
"""
import sys

import torch

src, dst = sys.argv[1], sys.argv[2]
which = sys.argv[sys.argv.index("--which") + 1] if "--which" in sys.argv else "ema"
s = torch.load(src, map_location="cpu")
torch.save({"model": {k: v.half() for k, v in s[which].items()}, "cfg": s["cfg"], "mean": s["mean"],
            "std": s["std"], "sigma": s["sigma"], "step": s["step"], "weights": which, "holdout": s["holdout"]}, dst)
print(f"{dst}: step {s['step']} ({which})")
