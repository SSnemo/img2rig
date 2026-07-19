"""Pose keyframes: full-frame replacement frames (e.g. attack windup / swing)
derived from the picked final by img2img / inpaint.

Why full-frame replacement at all: a cutout puppet rig can only bend what was
painted - rotating an arm layer past the painted content exposes holes, so a
genuinely different pose (arms overhead, a downward swing) needs new pixels,
not new transforms. These frames are swapped in whole at runtime.

Two candidate routes are rolled side by side, because each fails differently:

* Route A - whole-image img2img (moderate denoise, ~0.60-0.70). Global
  lighting stays coherent, but the lower body can drift off the rig canvas.
* Route B - upper-body masked inpaint (higher denoise, ~0.70-0.80, the mask
  gives permission so denoise can go hotter). Everything outside the mask is
  pixel-identical to the base, so the frame is naturally aligned with masks
  and pivots cut on the base image.

The mask (route B) encodes two hard-learned rules:

* It extends well above the head: raised arms need empty canvas to be painted
  into. A mask that hugs the figure leaves the new limbs nowhere to go.
* The face (plus any headwear) is punched out as an ellipse - an identity
  lock. High-denoise inpaint will otherwise redraw the face into a stranger.

Prompting note: tag-trained checkpoints respond far more strongly to
canonical booru-style tags ("arms up") than to natural-language pose prose;
if a pose refuses to land, switch the frame's ``add`` to the canonical tag
and weight it up (:1.4-1.6) rather than describing harder.
"""
from __future__ import annotations

import os
from typing import Any

from PIL import Image, ImageDraw

from .config import Spec
from .sdapi import Client, contact_sheet

DENOISE_DEFAULT = {"img2img": [0.60, 0.70], "inpaint": [0.70, 0.80]}


def build_mask(size: tuple[int, int], mask_cfg: dict[str, Any]) -> Image.Image:
    """Build the upper-body inpaint mask from fractional coordinates.

    ``band`` [top, bottom] is a full-width editable band as fractions of H
    (top should sit near 0 to leave headroom for raised arms). ``face_hole``
    is an ellipse (center/radius as fractions of W/H) punched out of the band
    to lock identity.
    """
    w, h = size
    mask = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(mask)
    top, bottom = mask_cfg["band"]
    d.rectangle([0, int(top * h), w, int(bottom * h)], fill=255)
    hole = mask_cfg.get("face_hole")
    if hole:
        cx, cy = hole["center"]
        rx, ry = hole["radius"]
        d.ellipse([int((cx - rx) * w), int((cy - ry) * h),
                   int((cx + rx) * w), int((cy + ry) * h)], fill=0)
    return mask


def _frame_prompts(spec: Spec, frame_cfg: dict[str, Any]) -> tuple[str, str]:
    """Keyframe prompts are built around the ``identity`` line instead of the
    spec's normal subject + pose_constraints: the standing-pose constraints
    would fight the new pose, so only quality/identity/style carry over and
    the frame's ``add`` states the pose."""
    p = spec["prompt"]
    kf = spec["keyframes"]
    parts = [p["quality"], kf["identity"], p.get("style", ""), frame_cfg["add"]]
    pos = ", ".join(s for s in parts if s)
    neg = spec.negative(frame_cfg.get("neg_add", ""))
    return pos, neg


def candidates(spec: Spec, client: Client, base_path: str) -> list[str]:
    """Roll keyframe candidates for every frame in spec["keyframes"]["frames"]
    via both routes, over each route's denoise ladder x a few seeds, into
    work_dir/keyframes/. One contact sheet per frame for eyeballing.

    Returns the list of candidate image paths.
    """
    kf = spec["keyframes"]
    base = Image.open(base_path).convert("RGB")
    out = os.path.join(spec.work_dir, "keyframes")
    os.makedirs(out, exist_ok=True)

    mask = build_mask(base.size, kf["mask"])
    mask.save(os.path.join(out, "mask_upper.png"))

    denoise = kf.get("denoise", DENOISE_DEFAULT)
    seed0 = kf.get("seed0", spec["gen"]["seed0"])
    n_seeds = kf.get("seeds", 2)
    tile = tuple(spec["gen"].get("contact_tile", [300, 439]))

    all_paths: list[str] = []
    for frame, frame_cfg in kf["frames"].items():
        pos, neg = _frame_prompts(spec, frame_cfg)
        paths: list[str] = []
        for route, use_mask in (("A", False), ("B", True)):
            key = "inpaint" if use_mask else "img2img"
            for d in denoise.get(key, DENOISE_DEFAULT[key]):
                for s in range(n_seeds):
                    seed = seed0 + s
                    im = client.img2img(
                        base, pos, neg, seed, d,
                        mask=mask if use_mask else None,
                        cfg_scale=kf.get("cfg", 8),
                    )
                    name = f"{frame}_{route}_seed{seed}_d{int(d * 100)}.png"
                    p = os.path.join(out, name)
                    im.save(p)
                    paths.append(p)
                    print("done", name)
        contact_sheet(paths, tile, cols=4,
                      out_path=os.path.join(out, f"contact_{frame}.png"))
        print("contact sheet:", frame)
        all_paths.extend(paths)
    return all_paths
