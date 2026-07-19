"""Local retouching: re-roll one region of an otherwise-final image
(a stray extra hand, a broken accessory) without touching the rest.

Practical notes from production use:

* Extra hands/limbs are the classic escapee - candidate review checklists
  should explicitly count hands, they slip past a composition-level glance.
* Make the mask generously larger than the visible defect: a tight box leaves
  fingertips or edge halos just outside it, forcing a second pass.
* On ornate costumes SD inpaint loves to "decorate" the patch (invented gold
  embroidery, marble-like latent texture) instead of continuing plain fabric.
  Steer with prompt_add ("plain fabric, simple cloth folds") and keep denoise
  moderate. If two or three rolls in the same spot all decorate, stop rolling:
  a deterministic clone/heal fill (copy a nearby clean patch, seamless-blend
  it in) beats the sampler there - do that outside this module.
"""
from __future__ import annotations

import os

from PIL import Image, ImageDraw

from .config import Spec
from .sdapi import Client

Box = tuple[int, int, int, int]


def inpaint_region(spec: Spec, client: Client, src_path: str,
                   box_or_mask: Box | str | Image.Image, prompt_add: str = "",
                   denoise: float = 0.55, seed: int = 0) -> str:
    """Inpaint one region of src_path and save the result to work_dir.

    ``box_or_mask``: an (x0, y0, x1, y1) pixel rectangle, a path to a
    grayscale mask image, or a PIL "L" mask (white = editable). The base
    positive prompt plus ``prompt_add`` describes what should fill the region.

    Saves work_dir/{stem}_retouch_seed{seed}.png (never overwrites the
    source - keep the original until the patch is verified) and returns the
    path.
    """
    src = Image.open(src_path).convert("RGB")
    if isinstance(box_or_mask, Image.Image):
        mask = box_or_mask.convert("L")
    elif isinstance(box_or_mask, str):
        mask = Image.open(box_or_mask).convert("L")
    else:
        mask = Image.new("L", src.size, 0)
        ImageDraw.Draw(mask).rectangle(box_or_mask, fill=255)

    im = client.img2img(src, spec.positive(prompt_add), spec.negative(),
                        seed, denoise, mask=mask)
    stem = os.path.splitext(os.path.basename(src_path))[0]
    p = os.path.join(spec.work_dir, f"{stem}_retouch_seed{seed}.png")
    im.save(p)
    print("retouch saved:", p)
    return p
