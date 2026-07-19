"""State variants: same composition, different look (e.g. an "enraged" state
with glowing eyes and an aura), produced by img2img over the picked final.

Why img2img and not txt2img with edited tags: at the same seed, changing even
one tag in txt2img shifts the whole composition, and layer masks cut on the
base image would no longer line up. img2img at moderate denoise keeps the
pixels that matter (silhouette, face, costume topology) while the added tags
land the new elements. Denoise ~0.55 is the empirical sweet spot: below ~0.45
the new elements barely appear, above ~0.65 identity starts to drift. A small
cfg bump (7 -> 8) helps the added tags win without touching composition.
"""
from __future__ import annotations

import os

from PIL import Image

from .config import Spec
from .sdapi import Client


def make(spec: Spec, client: Client, variant_name: str, src_path: str,
         seed: int) -> str:
    """Run the img2img state variant described by spec["variants"][name]
    over src_path (normally the hires final). Reusing the final's seed keeps
    the noise pattern consistent with the base image.

    Variant config fields: ``denoise`` (default 0.55), ``cfg`` (optional
    cfg_scale override), ``add`` (positive tags), ``neg_add`` (optional
    negative tags).

    Saves work_dir/{name}_{variant}_seed{seed}.png and returns the path.
    """
    v = spec["variants"][variant_name]
    init = Image.open(src_path).convert("RGB")
    kw: dict = {}
    if "cfg" in v:
        kw["cfg_scale"] = v["cfg"]
    im = client.img2img(
        init,
        spec.positive(v.get("add", "")),
        spec.negative(v.get("neg_add", "")),
        seed,
        v.get("denoise", 0.55),
        **kw,
    )
    p = os.path.join(spec.work_dir, f"{spec.name}_{variant_name}_seed{seed}.png")
    im.save(p)
    print(f"variant '{variant_name}' saved:", p, im.size)
    return p
