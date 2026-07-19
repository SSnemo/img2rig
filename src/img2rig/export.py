"""Final packaging: per-layer PNG pack + rig manifest, and full-frame
pose-keyframe mattes.

Everything is scaled from source resolution into the runtime canvas
declared in spec["rig"]["canvas"], and the manifest (rig.txt) records where
each cropped layer sits and how it is parented.
"""
from __future__ import annotations

import os

import cv2
import numpy as np
from PIL import Image

from .config import Spec
from .sdapi import Client


# ---- edge treatment ----

def feather(arr: np.ndarray, px: int = 2) -> np.ndarray:
    """Soften the alpha edge of a full-frame RGBA layer.

    Premultiplied runtimes turn hard mask edges into dark fringes; a ~2 px
    feather (blur the alpha, never brighten it) hides them.
    """
    a = arr[..., 3].astype(np.float32)
    a = cv2.GaussianBlur(a, (px * 2 + 1, px * 2 + 1), 0)
    out = arr.copy()
    out[..., 3] = np.clip(np.minimum(arr[..., 3].astype(np.float32), a),
                          0, 255).astype(np.uint8)
    return out


def bleed(arr: np.ndarray, iters: int = 6) -> np.ndarray:
    """Color bleed: push the RGB of the nearest opaque pixels outward into
    transparent/semi-transparent area.

    GPU linear sampling mixes the RGB stored under zero alpha into the
    visible edge; source images leave background color there, which shows
    up as a bright fringe on every scaled sprite. Growing the edge color a
    few pixels outward makes the sampled neighborhood self-colored.
    """
    rgb = arr[..., :3].copy()
    known = arr[..., 3] > 200
    k = np.ones((3, 3), np.uint8)
    for _ in range(iters):
        grow = cv2.dilate(known.astype(np.uint8), k).astype(bool) & ~known
        if not grow.any():
            break
        blur = cv2.blur(rgb * known[..., None].astype(np.uint8), (3, 3)).astype(np.float32)
        cnt = cv2.blur(known.astype(np.float32), (3, 3))
        fill = (blur / np.maximum(cnt[..., None], 1e-4)).astype(np.uint8)
        rgb[grow] = fill[grow]
        known |= grow
    out = arr.copy()
    out[..., :3] = rgb
    return out


# ---- layer pack + manifest ----

def layer_pack(spec: Spec, layers: dict[str, np.ndarray]) -> str:
    """Export the final layer pack and write rig.txt to out_dir.

    `layers` maps layer name -> full-frame RGBA at source resolution (the
    dict cleanup.assemble returns). Only names present in
    spec["rig"]["layers"] are exported; each is feathered, color-bled,
    bbox-cropped and rescaled by the uniform factor canvas_height /
    source_height (portrait packs fit by height; width is not cropped).

    Manifest format (one item per line):
        canvas <w> <h>
        node <name> <parent> <z> <pivot_x> <pivot_y>
        layer <name> <file> <x> <y> <w> <h> <pivot_x> <pivot_y> <parent> <z> [spring <k> <c>]
    Pivots in the spec are source-resolution coordinates and are rescaled
    here; layers without a pivot get their bbox center. A composited
    pack_preview.png is written to work_dir as the final eyeball check.

    Returns the rig.txt path.
    """
    rig = spec["rig"]
    cw, ch = rig["canvas"]
    src_h = next(iter(layers.values())).shape[0]
    sc = ch / src_h

    lines = [
        "# layered rig manifest v1",
        "# node <name> <parent> <z> <pivot_x> <pivot_y>",
        "# layer <name> <file> <x> <y> <w> <h> <pivot_x> <pivot_y> <parent> <z> [spring <k> <c>]",
        f"canvas {cw} {ch}",
    ]
    for name, node in rig.get("nodes", {}).items():
        px, py = node["pivot"]
        lines.append(f"node {name} {node['parent']} {node.get('z', -1)} "
                     f"{px * sc:.1f} {py * sc:.1f}")

    placed: list[tuple[str, int, int]] = []
    for name, lcfg in sorted(rig["layers"].items(),
                             key=lambda kv: kv[1].get("z", 0)):
        arr = layers.get(name)
        if arr is None:
            print(f"{name}: no layer image, skipped")
            continue
        arr = bleed(feather(arr))
        ys, xs = np.where(arr[..., 3] > 0)
        if len(ys) == 0:
            print(f"{name}: empty, skipped")
            continue
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        cut = arr[y0:y1, x0:x1]
        w = max(1, int((x1 - x0) * sc + 0.5))
        h = max(1, int((y1 - y0) * sc + 0.5))
        fn = f"{spec.name}_{name}.png"
        Image.fromarray(cut).resize((w, h), Image.LANCZOS).save(
            os.path.join(spec.out_dir, fn))
        pv = lcfg.get("pivot") or ((x0 + x1) / 2, (y0 + y1) / 2)
        ln = (f"layer {name} {fn} {x0 * sc:.1f} {y0 * sc:.1f} {w} {h} "
              f"{pv[0] * sc:.1f} {pv[1] * sc:.1f} {lcfg['parent']} "
              f"{lcfg.get('z', 0)}")
        if lcfg.get("spring"):
            sk, sd = lcfg["spring"]
            ln += f" spring {sk} {sd}"
        lines.append(ln)
        placed.append((fn, int(x0 * sc), int(y0 * sc)))
        print(f"{name}: {w}x{h} @({x0 * sc:.0f},{y0 * sc:.0f})")

    rig_path = os.path.join(spec.out_dir, "rig.txt")
    with open(rig_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # composited preview at canvas size (placed list is already z-ascending)
    prev = Image.new("RGBA", (cw, ch), (40, 40, 56, 255))
    for fn, x, y in placed:
        prev.alpha_composite(Image.open(os.path.join(spec.out_dir, fn))
                             .convert("RGBA"), (max(0, x), max(0, y)))
    prev.convert("RGB").save(os.path.join(spec.work_dir, "pack_preview.png"))
    print("rig.txt + layers ->", spec.out_dir)
    return rig_path


# ---- pose keyframe matting ----

def keyframe(spec: Spec, client: Client, frame_name: str,
             candidate_path: str, base_path: str) -> str:
    """Matte a full-frame pose keyframe off its background and export it as
    out_dir/kf_<frame_name>.png at the rig canvas size.

    Pose keyframes are whole-image replacements (same character, new pose),
    so the matte is a single figure cut. SAM positive prompts are a body
    grid (valid because the keyframe shares its composition with the base
    image) PLUS auto points at the centroids of large candidate-vs-base
    diff blobs - the diff is exactly where the pose moved (e.g. newly
    raised arms), which a static grid would miss entirely.

    Optional spec overrides under spec["keyframes"]: seg_pos / seg_neg
    (fractional [x, y] point lists); the default grid centers on the face
    x from keyframes.mask.face_hole when present. Candidate-mask choice
    targets the character area window midpoint from spec["split"] when a
    "character" part exists there.

    Writes two previews to work_dir: kf_<frame>_matte.png (mask tinted on
    the candidate, prompt points drawn: positive green, auto yellow,
    negative red) and kf_<frame>_preview.png (result on a dark backdrop).
    Returns the exported PNG path.
    """
    im = Image.open(candidate_path).convert("RGB")
    W, H = im.size
    total = W * H
    base = Image.open(base_path).convert("RGB").resize((W, H))
    kf = spec.get("keyframes") or {}

    cx = kf.get("mask", {}).get("face_hole", {}).get("center", [0.5, 0.0])[0]
    pos_frac = kf.get("seg_pos") or [
        [cx, 0.14], [cx, 0.30], [cx, 0.42], [cx, 0.55],
        [cx - 0.05, 0.72], [cx + 0.05, 0.72], [cx, 0.88],
    ]
    neg_frac = kf.get("seg_neg") or [
        [0.04, 0.05], [0.96, 0.05], [0.04, 0.95], [0.96, 0.95],
        [0.05, 0.50], [0.95, 0.50],
    ]
    pos = [[x * W, y * H] for x, y in pos_frac]
    neg = [[x * W, y * H] for x, y in neg_frac]

    # auto correction points: big diff blobs in the upper frame = new limbs
    diff = np.abs(np.asarray(im, np.int16) - np.asarray(base, np.int16)).sum(2) > 90
    diff = cv2.morphologyEx(diff.astype(np.uint8), cv2.MORPH_OPEN,
                            np.ones((9, 9), np.uint8))
    n, _, stats, cent = cv2.connectedComponentsWithStats(diff)
    blobs = [(stats[i, cv2.CC_STAT_AREA], [float(cent[i][0]), float(cent[i][1])])
             for i in range(1, n)
             if stats[i, cv2.CC_STAT_AREA] > 0.0015 * total
             and cent[i][1] < 0.55 * H]
    extra = [p for _, p in sorted(blobs, key=lambda b: -b[0])[:5]]
    print("auto diff points:", [[int(a), int(b)] for a, b in extra])

    cands = client.sam_points(im, pos + extra, neg)
    if not cands:
        raise RuntimeError("SAM returned no masks for the keyframe")
    # pick the mask closest to the expected figure coverage
    target = 0.30
    for part in (spec.get("split") or {}).get("parts", []):
        if part.get("name") == "character":
            target = (part["min"] + part["max"]) / 2
            break
    best = min(cands, key=lambda m: abs(m.mean() - target))
    print(f"mask frac: {best.mean():.3f} (target {target:.2f})")

    # CC filter: keep the figure-sized blobs, close small gaps, then fill
    # only genuinely small holes (real gaps like an arm/torso window stay)
    mask = best.astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    keep = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] > 0.025 * total:
            keep[lab == i] = 1
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    inv = (1 - keep).astype(np.uint8)
    n2, lab2, stats2, _ = cv2.connectedComponentsWithStats(inv)
    for i in range(1, n2):
        if stats2[i, cv2.CC_STAT_AREA] < 0.003 * total:
            keep[lab2 == i] = 1

    # soft edge, then color bleed: a keyframe is one flat image, so TELEA
    # from the silhouette outward is enough (the layer pack uses the
    # iterative bleed instead)
    alpha = cv2.GaussianBlur(keep * 255, (3, 3), 0)
    rgb = np.asarray(im, np.uint8).copy()
    rgb = cv2.inpaint(rgb, (alpha < 16).astype(np.uint8), 4, cv2.INPAINT_TELEA)

    cw, ch = spec["rig"]["canvas"]
    out_im = Image.fromarray(np.dstack([rgb, alpha]).astype(np.uint8), "RGBA") \
        .resize((cw, ch), Image.LANCZOS)
    dst = os.path.join(spec.out_dir, f"kf_{frame_name}.png")
    out_im.save(dst)

    # previews: matte overlay with prompt points, and result on dark backdrop
    ov = np.asarray(im, np.uint8).copy()
    sel = keep.astype(bool)
    ov[sel] = (ov[sel] * 0.55 + np.array([60, 220, 120]) * 0.45).astype(np.uint8)
    # colors in RGB order (the array comes from PIL, not cv2.imread)
    for pts, color in ((pos, (40, 220, 40)), (extra, (255, 220, 0)),
                       (neg, (255, 50, 50))):
        for x, y in pts:
            cv2.circle(ov, (int(x), int(y)), max(4, W // 200), color, -1)
    Image.fromarray(ov).resize((W // 2, H // 2)).save(
        os.path.join(spec.work_dir, f"kf_{frame_name}_matte.png"))
    backdrop = Image.new("RGBA", (cw, ch), (40, 40, 56, 255))
    Image.alpha_composite(backdrop, out_im).convert("RGB").save(
        os.path.join(spec.work_dir, f"kf_{frame_name}_preview.png"))
    print("saved", dst)
    return dst
