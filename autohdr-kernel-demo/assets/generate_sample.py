"""
generate_sample.py — creates assets/sample.jpg

Generates a synthetic warm-toned interior-like gradient image.
Run once:  python autohdr-kernel-demo/assets/generate_sample.py
"""

from pathlib import Path

import numpy as np
from PIL import Image

H, W = 768, 1152

rng = np.random.default_rng(42)

y = np.linspace(0.2, 1.0, H)[:, None]   # top-dark, bottom-bright
x = np.linspace(0.4, 1.2, W)[None, :]   # left-dim, right-warm

# Warm-toned gradient (r heavy, g mid, b low)
r = np.clip(x * 0.85 + rng.normal(0, 0.015, (H, W)), 0.0, 1.0)
g = np.clip(y * 0.65 + rng.normal(0, 0.010, (H, W)), 0.0, 1.0)
b = np.clip(y * 0.35 + 0.05 + rng.normal(0, 0.010, (H, W)), 0.0, 1.0)

img_array = (np.stack([r, g, b], axis=2) * 255).astype(np.uint8)
img = Image.fromarray(img_array, mode="RGB")

out = Path(__file__).parent / "sample.jpg"
img.save(out, quality=92)
print(f"Saved synthetic sample image → {out}  ({W}×{H} px)")
