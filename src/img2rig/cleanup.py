"""Refinement stage: point-prompt re-masking, derived layers, occlusion
fill, and mutually-exclusive layer assembly.

Every function here drops probe/overlay images into work_dir. Those images
are the pipeline's core interaction design: an agent looks at them, edits
the point tables and geometry parameters in the character spec, and re-runs
the step until the masks are right. Nothing in this module hardcodes a
character - the part names, point tables and priority order all come from
spec["cleanup"], and each optional section (hair_split / glow / inpaint)
simply skips its step when absent.
"""
from __future__ import annotations

import os

import cv2
import numpy as np
from PIL import Image

from .config import Spec
from .sdapi import Client
from .split import PALETTE, cut_rgba, load_mask, masks_dir, save_mask, tint


# ---- small morphology helpers ----

def _kernel(size: int) -> np.ndarray:
    """Elliptical structuring element of the given diameter in pixels
    (spec fields like roi_dilate / eye_pad_dilate are diameters)."""
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def _closed(m: np.ndarray, r: int) -> np.ndarray:
    """Morphological close with radius r (spec close_r is a radius)."""
    if r <= 0:
        return m
    return cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE,
                            _kernel(2 * r + 1)).astype(bool)


def _dilated(m: np.ndarray, size: int) -> np.ndarray:
    return cv2.dilate(m.astype(np.uint8), _kernel(size)).astype(bool)


def _cc_filter(m: np.ndarray, min_area: int) -> np.ndarray:
    """Keep only connected components of at least min_area pixels."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    out = np.zeros_like(m, dtype=bool)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out |= labels == i
    return out


def _pick_small(cands: list[np.ndarray], total: int, amax: float) -> int:
    """Pick the largest candidate whose area stays under amax (fraction of
    the prompt image); fall back to the smallest when all overflow. Small
    facial parts drift toward "the whole lower face" - the cap catches it."""
    order = sorted(range(len(cands)), key=lambda i: int(cands[i].sum()))
    pick = None
    for i in order:
        if cands[i].sum() / total <= amax:
            pick = i
    return order[0] if pick is None else pick


# ---- pass 1a: small facial parts ----

def face_parts(spec: Spec, client: Client, hires_path: str) -> dict[str, np.ndarray]:
    """SAM point-prompt masks for small facial parts on a face crop.

    Running SAM on the full frame makes a tiny part flood into the whole
    lower face - cropping and upscaling first is the standard treatment for
    small parts. Parts flagged `band: true` go through a second, upscaled
    crop (spec["cleanup"]["eyes_band"]): at face-crop scale an eye is only
    ~20 px tall and a few pixels of point drift flips the mask entirely.

    Point tables live in spec["cleanup"]["face_parts"], in face-crop
    coordinates (band parts are converted internally). Partial masks are
    acceptable for parts largely hidden by hair - the covered portion was
    never visible anyway.

    Saves face_probe.png / eyes_probe.png (the images to read point
    coordinates from), face_parts_overlay.png (all picks color-coded) and
    per-part masks at full hires resolution. Returns {name: bool mask}.
    """
    cfg = spec["cleanup"]
    x0, y0, x1, y1 = cfg["face_crop"]
    hires = Image.open(hires_path).convert("RGB")
    crop = hires.crop((x0, y0, x1, y1))
    crop.save(os.path.join(spec.work_dir, "face_probe.png"))

    band_cfg = cfg.get("eyes_band")
    band_im = None
    bx0 = by0 = bx1 = by1 = scale = 0
    if band_cfg:
        bx0, by0, bx1, by1 = band_cfg["box"]
        scale = band_cfg.get("scale", 3)
        band_im = crop.crop((bx0, by0, bx1, by1)).resize(
            ((bx1 - bx0) * scale, (by1 - by0) * scale), Image.LANCZOS)
        band_im.save(os.path.join(spec.work_dir, "eyes_probe.png"))

    results: dict[str, np.ndarray] = {}  # face-crop space
    for name, part in cfg["face_parts"].items():
        use_band = bool(part.get("band")) and band_im is not None
        if use_band:
            pos = [[(p[0] - bx0) * scale, (p[1] - by0) * scale] for p in part["pos"]]
            neg = [[(p[0] - bx0) * scale, (p[1] - by0) * scale] for p in part["neg"]]
            img = band_im
        else:
            pos, neg, img = part["pos"], part["neg"], crop
        cands = client.sam_points(img, pos, neg)
        if not cands:
            print(f"{name}: no masks")
            continue
        total = img.width * img.height
        pick = _pick_small(cands, total, part["amax"])
        m = cands[pick]
        if use_band:
            # band coords -> face-crop coords: shrink back, then place
            small = Image.fromarray(m.astype(np.uint8) * 255).resize(
                (bx1 - bx0, by1 - by0), Image.BILINEAR)
            m = np.zeros((crop.height, crop.width), bool)
            m[by0:by1, bx0:bx1] = np.array(small) > 127
        results[name] = m
        print(f"{name}: candidates "
              f"{[f'{c.sum() / total:.3%}' for c in cands]}, picked #{pick}")

    ov = np.array(crop)
    for i, (name, m) in enumerate(results.items()):
        ov = tint(ov, m, PALETTE[i % len(PALETTE)], 0.65)
    Image.fromarray(ov).save(os.path.join(spec.work_dir, "face_parts_overlay.png"))

    out: dict[str, np.ndarray] = {}
    for name, m in results.items():
        full = np.zeros((hires.height, hires.width), bool)
        full[y0:y1, x0:x1] = m
        save_mask(spec, name, full)
        out[name] = full
    return out


# ---- pass 1b: large body parts ----

def body_parts(spec: Spec, client: Client, hires_path: str) -> dict[str, np.ndarray]:
    """SAM point-prompt masks for large parts on the full hires frame,
    replacing the draft-stage DINO boxes.

    Table: spec["cleanup"]["body_parts"], name -> {pos, neg, amin, amax}
    in hires coordinates. Area window (fraction of frame) guards against
    drift and flood: inside the window the largest candidate wins; when all
    candidates miss, the one closest to the window midpoint is taken so the
    agent still gets something to look at.

    Practical prompt-table lessons: densify positive points along thin parts
    (arms) or the layer comes back sparse; densify negative points wherever
    the background shares a color family with the part; add positive points
    on semi-transparent fabric or SAM treats it as background and the layer
    ships with holes.

    Saves per-part masks plus body_overlay.png (all parts color-coded) and
    character_overlay.png (background dimmed by the character matte).
    Returns {name: bool mask}.
    """
    hires = Image.open(hires_path).convert("RGB")
    rgb = np.array(hires)
    total = hires.width * hires.height
    results: dict[str, np.ndarray] = {}
    for name, part in spec["cleanup"]["body_parts"].items():
        cands = client.sam_points(hires, part["pos"], part["neg"])
        if not cands:
            print(f"{name}: no masks")
            continue
        inwin = [i for i in range(len(cands))
                 if part["amin"] <= cands[i].sum() / total <= part["amax"]]
        if inwin:
            pick = max(inwin, key=lambda i: int(cands[i].sum()))
        else:
            mid = (part["amin"] + part["amax"]) / 2
            pick = min(range(len(cands)),
                       key=lambda i: abs(cands[i].sum() / total - mid))
        results[name] = cands[pick]
        save_mask(spec, name, cands[pick])
        print(f"{name}: candidates "
              f"{[f'{c.sum() / total:.2%}' for c in cands]}, picked #{pick}")

    ov = rgb.copy()
    i = 0
    for name, m in results.items():
        if name == "character":
            continue
        ov = tint(ov, m, PALETTE[i % len(PALETTE)])
        i += 1
    Image.fromarray(ov).resize((hires.width // 2, hires.height // 2)).save(
        os.path.join(spec.work_dir, "body_overlay.png"))
    if "character" in results:
        ov2 = rgb.copy()
        bg = ~results["character"]
        ov2[bg] = (ov2[bg] * 0.25).astype(np.uint8)
        Image.fromarray(ov2).resize((hires.width // 2, hires.height // 2)).save(
            os.path.join(spec.work_dir, "character_overlay.png"))
    return results


# ---- pass 2a: geometric front/back hair split ----

def hair_split(spec: Spec, hires_path: str) -> dict[str, np.ndarray]:
    """Split the hair mask into front and back layers - geometrically.

    SAM cannot split a continuous, same-colored hair mass: point prompts
    anywhere inside it return the whole thing. The side strands are a
    geometric fact instead: hair below the eye line AND outside the cheek
    columns is the back layer. A connected-component pass (min_blob) throws
    fringe-bottom crumbs back into the front layer.

    Config: spec["cleanup"]["hair_split"] = {eyeline_y, cheek_x: [L, R],
    min_blob, source?} in hires coordinates. Returns {} (step skipped) when
    the section is absent. Saves mask_hair_front / mask_hair_back and
    hair_split_overlay.png (front green, back red).
    """
    cfg = spec["cleanup"].get("hair_split")
    if not cfg:
        return {}
    hires = Image.open(hires_path).convert("RGB")
    W, H = hires.size
    hair = load_mask(spec, cfg.get("source", "hair"), (W, H))
    if hair is None:
        raise FileNotFoundError(
            "hair mask not found - run body_parts (or draft) first")
    cl, cr = cfg["cheek_x"]
    yy, xx = np.mgrid[0:H, 0:W]
    back = hair & (yy > cfg["eyeline_y"]) & ((xx < cl) | (xx > cr))
    back = _cc_filter(back, cfg.get("min_blob", 800))
    front = hair & ~back
    save_mask(spec, "hair_front", front)
    save_mask(spec, "hair_back", back)
    print(f"hair split (geometric): front={int(front.sum())} "
          f"back={int(back.sum())} px")

    ov = np.array(hires)
    ov = tint(ov, front, (80, 220, 80), 0.55)
    ov = tint(ov, back, (255, 90, 90), 0.55)
    Image.fromarray(ov).resize((W // 2, H // 2)).save(
        os.path.join(spec.work_dir, "hair_split_overlay.png"))
    return {"hair_front": front, "hair_back": back}


# ---- pass 2b: derived emissive (glow) layer ----

def glow_layer_name(spec: Spec) -> str:
    """Rig layer name the derived glow mask feeds.

    Priority: an explicit `layer` key in the glow section; otherwise the one
    rig layer that no other stage produces (not in the assemble order and
    not a fallback layer); otherwise the literal name "glow".
    """
    g = (spec.get("cleanup") or {}).get("glow") or {}
    if "layer" in g:
        return g["layer"]
    rig = spec.get("rig") or {}
    produced = set((spec.get("cleanup") or {}).get("assemble", {}).get("order", []))
    produced |= {"head_base", "body_remainder", "character"}
    extra = [n for n in (rig.get("layers") or {}) if n not in produced]
    return extra[0] if len(extra) == 1 else "glow"


def derive_glow(spec: Spec, base4k_path: str, variant4k_path: str) -> np.ndarray | None:
    """Derive an emissive layer from the base-vs-variant pixel diff.

    The variant is an img2img re-render, so a whole-frame |diff| threshold
    mostly finds img2img noise. The usable signal is a one-way channel gain
    (variant brighter in the glow tint channel, and that channel dominant)
    inside a dilated ROI around the masks listed in
    spec["cleanup"]["glow"]["roi"]. A tighter ROI additionally accepts any
    large full-color difference - glow recolors can raise the other channels
    too and defeat the single-channel test (e.g. red-violet eye glow).

    Config keys: variant, roi, roi_dilate, red_gain, any_diff, min_blob,
    and optionally tight_dilate / channel / layer. Dilate values are kernel
    diameters in pixels. Returns the boolean glow mask at variant resolution
    (also saved as mask "glow"), or None when the section is absent.
    Saves glow_overlay.png on the variant image for eyeballing.
    """
    cfg = spec["cleanup"].get("glow")
    if not cfg:
        return None
    base = np.asarray(Image.open(base4k_path).convert("RGB"), dtype=np.int16)
    var = np.asarray(Image.open(variant4k_path).convert("RGB"), dtype=np.int16)
    if base.shape != var.shape:
        raise ValueError("base and variant images must share a resolution")
    H, W = var.shape[:2]

    roi = np.zeros((H, W), bool)
    for name in cfg["roi"]:
        m = load_mask(spec, name, (W, H))
        if m is not None:
            roi |= m
    wide = _dilated(roi, cfg.get("roi_dilate", 61))
    tight = _dilated(roi, cfg.get("tight_dilate", 15))

    ch = cfg.get("channel", 0)  # emissive tint channel, default red
    other = (ch + 2) % 3       # the channel a pure tint should NOT raise
    gain = ((var[..., ch] - base[..., ch] > cfg.get("red_gain", 50))
            & (var[..., ch] > var[..., other] + 30))
    anydiff = np.abs(var - base).sum(axis=2) > cfg.get("any_diff", 100)

    glow = (gain & wide) | (tight & anydiff)
    glow = cv2.morphologyEx(glow.astype(np.uint8), cv2.MORPH_CLOSE,
                            _kernel(9)).astype(bool)
    keep = _cc_filter(glow, cfg.get("min_blob", 300))
    save_mask(spec, "glow", keep)
    print(f"glow (channel gain in roi): {int(keep.sum())} px")

    ov = tint(np.asarray(var, dtype=np.uint8).copy(), keep, (60, 255, 120))
    Image.fromarray(ov).resize((W // 4, H // 4)).save(
        os.path.join(spec.work_dir, "glow_overlay.png"))
    return keep


# ---- pass 3: occlusion fill ----

def _occlusion_zones(spec: Spec, size: tuple[int, int]
                     ) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """(hair-rim band clipped to the face zone, dilated eye pad, face rect).

    Hole-placement lessons baked in: holes cut on the hair body make the
    fill paint floating color patches (the layer underneath there is hair,
    not skin); holes leaking into the background make the fill hallucinate.
    The correct hole is exactly the skin the head base will reveal in
    motion: a thin ring just inside the hair silhouette, clipped to the
    face area, plus a pad around the eyes (a blink squashes the eye layer
    and exposes the head base - without the pad that region is a
    mutual-exclusion void that renders as black).
    """
    cfg = spec["cleanup"]["inpaint"]
    W, H = size
    hair = np.zeros((H, W), bool)
    for n in ("hair_front", "hair_back"):
        m = load_mask(spec, n, (W, H))
        if m is not None:
            hair |= m
    rim_px = cfg.get("rim_px", 12)
    rim = hair & ~cv2.erode(hair.astype(np.uint8),
                            _kernel(2 * rim_px + 1)).astype(bool)

    rect = cfg.get("face_rect")  # optional override, target coords
    if rect is None:
        # face_crop is in hires coords - rescale via a stored hires-res mask
        sc = 1.0
        for n in ("character", "hair_front", "hair"):
            p = os.path.join(masks_dir(spec), f"mask_{n}.png")
            if os.path.exists(p):
                sc = W / Image.open(p).size[0]
                break
        rect = [int(round(c * sc)) for c in spec["cleanup"]["face_crop"]]
    yy, xx = np.mgrid[0:H, 0:W]
    zone = (xx >= rect[0]) & (yy >= rect[1]) & (xx < rect[2]) & (yy < rect[3])

    eyepad = np.zeros((H, W), bool)
    for n in cfg.get("pad_masks", ["eye_L", "eye_R"]):
        m = load_mask(spec, n, (W, H))
        if m is not None:
            eyepad |= m
    eyepad = _dilated(eyepad, cfg.get("eye_pad_dilate", 13))
    return rim & zone, eyepad, rect


def fill_occlusion(spec: Spec, base4k_path: str) -> str | None:
    """Fill the occlusion holes so the head base survives motion reveals.

    TELEA alone pulls color from the adjacent hair, so the hole is first
    flat-filled with the median skin tone sampled around
    spec["cleanup"]["inpaint"]["skin_sample"] (a mid-tone cheek patch),
    then a small-radius TELEA pass blends the edges. Pixels outside the
    hole are never touched. For a thin motion-reveal ring this is enough;
    a diffusion inpaint pass (Client.img2img with the hole mask) can be
    layered on top for larger holes.

    Writes work_dir/head_filled.png (full frame), mask_head_hole.png, and
    inpaint_preview.png (original | filled over the face zone). Returns the
    filled image path, or None (step skipped) when the section is absent.
    """
    cfg = spec["cleanup"].get("inpaint")
    if not cfg:
        return None
    arr = np.array(Image.open(base4k_path).convert("RGB"))
    H, W = arr.shape[:2]
    rim, eyepad, rect = _occlusion_zones(spec, (W, H))
    hole = rim | eyepad
    print(f"occlusion hole (hair rim + eye pad): {int(hole.sum())} px")

    sx, sy = cfg["skin_sample"]
    patch = arr[max(0, sy - 20):sy + 20, max(0, sx - 20):sx + 20]
    skin = np.median(patch.reshape(-1, 3), axis=0).astype(np.uint8)
    print(f"sampled skin tone: {skin.tolist()}")

    filled = arr.copy()
    filled[hole] = skin
    filled = cv2.inpaint(
        filled,
        cv2.dilate(hole.astype(np.uint8), np.ones((3, 3), np.uint8)) * 255,
        4, cv2.INPAINT_TELEA)
    filled[hole] = (filled[hole] * 0.4 + skin * 0.6).astype(np.uint8)

    out_path = os.path.join(spec.work_dir, "head_filled.png")
    Image.fromarray(filled).save(out_path)
    save_mask(spec, "head_hole", hole)

    x0, y0 = max(0, rect[0]), max(0, rect[1])
    x1, y1 = min(W, rect[2]), min(H, rect[3])
    if x1 > x0 and y1 > y0:
        duo = np.concatenate([arr[y0:y1, x0:x1], filled[y0:y1, x0:x1]], axis=1)
        Image.fromarray(duo).resize((duo.shape[1] // 2, duo.shape[0] // 2)).save(
            os.path.join(spec.work_dir, "inpaint_preview.png"))
    return out_path


# ---- pass 4: mutually-exclusive assembly ----

def assemble(spec: Spec, base4k_path: str, variant4k_path: str | None = None,
             filled_path: str | None = None) -> dict[str, np.ndarray]:
    """Assemble final layers with mutual exclusion - every character pixel
    lands in exactly one layer, in the priority order of
    spec["cleanup"]["assemble"]["order"] (first grab wins).

    The character domain is deliberately NOT the raw character matte: SAM's
    character mask tends to exclude headwear and props as non-person, and
    can carry holes of its own, so intersecting parts with it would erase
    them. The domain is the hole-closed union of the matte with every part
    mask, reduced to its largest connected component; background rejection
    is each part's own negative points' job. Derived glow layers skip the
    domain entirely for the same reason.

    Part masks get a small morphological close (close_r) only - a large
    radius bulges masks into their neighbors and the priority table then
    steals their pixels.

    Fallbacks so no pixel is lost:
    - head_base: unassigned domain pixels above head_zone_y, plus (when the
      inpaint section exists) the hair-rim ring and eye pad, with pixels
      taken from the occlusion-filled frame. The pad intentionally overlaps
      the hair/eye layers - it is the backing that shows through in motion.
    - body_remainder: everything else, keeping the largest connected
      component plus any blob over remainder_min_blob px. Filtering harder
      than that once dropped detached costume fragments (hem and drape
      pieces) and left see-through holes in the composited character.

    Returns {layer name: full-frame RGBA}. Saves final per-layer masks as
    mask_final_<name>.png (plus mask_final_character.png for QA) and a
    color-coded assemble_overlay.png.
    """
    cfg = spec["cleanup"]["assemble"]
    big = np.array(Image.open(base4k_path).convert("RGBA"))
    H, W = big.shape[:2]
    char = load_mask(spec, "character", (W, H))
    if char is None:
        raise FileNotFoundError(
            "character mask not found - run body_parts (or draft) first")

    order = list(cfg["order"])
    close_r = cfg.get("close_r", 2)
    raw: dict[str, np.ndarray] = {}
    for n in order:
        m = load_mask(spec, n, (W, H))
        if m is None:
            print(f"{n}: mask missing, skipped")
            continue
        raw[n] = _closed(m, close_r)

    dom = char.copy()
    for m in raw.values():
        dom |= m
    dom8 = cv2.morphologyEx(dom.astype(np.uint8), cv2.MORPH_CLOSE, _kernel(15))
    nn, ll, ss, _ = cv2.connectedComponentsWithStats(dom8, 8)
    dom = (ll == 1 + int(np.argmax(ss[1:, cv2.CC_STAT_AREA]))) if nn > 1 \
        else dom8.astype(bool)
    for n in raw:
        raw[n] &= dom

    taken = np.zeros((H, W), bool)
    excl: dict[str, np.ndarray] = {}
    for n in order:
        if n not in raw:
            continue
        excl[n] = raw[n] & ~taken
        taken |= excl[n]

    yy = np.mgrid[0:H, 0:W][0]
    head = dom & (yy < cfg["head_zone_y"]) & ~taken
    if spec["cleanup"].get("inpaint"):
        rim, eyepad, _ = _occlusion_zones(spec, (W, H))
        head |= rim | eyepad
    excl["head_base"] = head
    taken |= head

    rem = dom & ~taken
    nr, lr, sr, _ = cv2.connectedComponentsWithStats(rem.astype(np.uint8), 8)
    rem2 = np.zeros_like(rem)
    if nr > 1:
        big_i = 1 + int(np.argmax(sr[1:, cv2.CC_STAT_AREA]))
        min_blob = cfg.get("remainder_min_blob", 3000)
        for i in range(1, nr):
            if i == big_i or sr[i, cv2.CC_STAT_AREA] > min_blob:
                rem2 |= lr == i
    excl["body_remainder"] = rem2

    # Overlap expansion - the anti-tearing pass. Mutual exclusion gives every
    # pixel exactly one owner, so any relative layer motion opens a gap that
    # shows the background. Fix: each layer additionally clones source pixels
    # a few px into territory owned by HIGHER-z layers. At rest the clone is
    # hidden under the upper layer (composite is pixel-identical); in motion
    # the gap reveals painted costume instead of a hole. Lower extends under
    # upper only, so top silhouettes stay crisp. Small facial parts are
    # skipped - their backing is head_base's job (eye pad / hair rim).
    overlap = cfg.get("overlap_px", 14)
    if overlap > 0:
        rig_layers = spec.get("rig", {}).get("layers", {})
        z_of = {n: rig_layers[n].get("z", 0) for n in excl if n in rig_layers}
        facial = {"eye_L", "eye_R", "brow_L", "brow_R", "mouth"}
        k = _kernel(2 * overlap + 1)
        ext: dict[str, np.ndarray] = {}
        for n, z in z_of.items():
            if n in facial or not excl[n].any():
                continue
            higher = np.zeros((H, W), bool)
            for m, mz in z_of.items():
                if mz > z:
                    higher |= excl[m]
            grow = cv2.dilate(excl[n].astype(np.uint8), k).astype(bool)
            ext[n] = excl[n] | (grow & higher & dom)
        for n, m in ext.items():
            excl[n] = m

    for n, m in excl.items():
        save_mask(spec, f"final_{n}", m)
    save_mask(spec, "final_character", dom)
    print("layer px:", {n: int(m.sum()) for n, m in excl.items() if m.any()})

    filled = np.array(Image.open(filled_path).convert("RGBA")) \
        if filled_path else big
    layers: dict[str, np.ndarray] = {}
    for n, m in excl.items():
        if not m.any():
            continue
        layers[n] = cut_rgba(filled if n == "head_base" else big, m)

    glow = load_mask(spec, "glow", (W, H))
    if glow is not None and variant4k_path:
        var = np.array(Image.open(variant4k_path).convert("RGBA"))
        # No domain clip on derived layers: the glow can extend past the
        # character matte (halos, floating accessories).
        layers[glow_layer_name(spec)] = cut_rgba(var, glow)

    ov = np.array(Image.open(base4k_path).convert("RGB"))
    for i, (n, m) in enumerate(excl.items()):
        if m.any():
            ov = tint(ov, m, PALETTE[i % len(PALETTE)], 0.5)
    Image.fromarray(ov).resize((W // 4, H // 4)).save(
        os.path.join(spec.work_dir, "assemble_overlay.png"))
    return layers


# ---- acceptance checks ----

def qa_sheets(spec: Spec, layers: dict[str, np.ndarray],
              base4k_path: str) -> dict[str, str]:
    """The acceptance triple - three images that together answer "is this
    layer stack shippable?":

    1. flatten_diff.png - composite every layer in rig z order (glow hidden,
       its pixels come from the variant frame) and diff against the source
       inside the character region. Red = pixels the stack lost or moved;
       the head patch and feathered edges are expected noise.
    2. layers_contact_sheet.png - every layer on white, labeled, one glance.
    3. wiggle_test.png - each spring-bound layer shifted a few px and
       re-composited; seams and holes that motion will expose show up here
       before they show up on a device.

    Returns {check name: image path} in work_dir.
    """
    rig_layers = spec["rig"]["layers"]
    glow_name = glow_layer_name(spec)
    zorder = sorted((n for n in layers if n in rig_layers),
                    key=lambda n: rig_layers[n].get("z", 0))
    zorder += [n for n in layers if n not in rig_layers]
    base = np.array(Image.open(base4k_path).convert("RGBA"))
    H, W = base.shape[:2]

    def composite(shift: tuple[str, tuple[int, int]] | None = None,
                  exclude: tuple[str, ...] = ()) -> np.ndarray:
        canvas = np.zeros((H, W, 4), np.float32)
        for n in zorder:
            if n in exclude:
                continue
            a = layers[n].astype(np.float32)
            if shift and n == shift[0]:
                a = np.roll(a, shift[1], axis=(0, 1))
            al = a[..., 3:4] / 255.0
            canvas[..., :3] = a[..., :3] * al + canvas[..., :3] * (1 - al)
            canvas[..., 3:4] = np.maximum(canvas[..., 3:4], a[..., 3:4])
        return canvas.astype(np.uint8)

    out: dict[str, str] = {}

    # 1. flatten check (source frames have no alpha - compare the character
    #    region only, everything outside it is background by construction)
    flat = composite(exclude=(glow_name,))
    charm = load_mask(spec, "final_character", (W, H))
    if charm is None:
        charm = flat[..., 3] > 0
    diff = np.abs(flat[..., :3].astype(int) - base[..., :3].astype(int)).sum(2)
    bad = (diff > 40) & charm
    print(f"flatten diff >40: {int(bad.sum())} px "
          f"({bad.sum() / max(1, charm.sum()):.2%} of character) "
          f"(head patch / feathered edges expected)")
    db = np.zeros((H, W, 3), np.uint8)
    db[charm] = (40, 40, 40)
    db[bad] = (255, 60, 60)
    p = os.path.join(spec.work_dir, "flatten_diff.png")
    Image.fromarray(db).resize((W // 5, H // 5)).save(p)
    out["flatten_diff"] = p

    # 2. contact sheet: layers on white, labeled
    cells = []
    for n in zorder:
        a = layers[n].astype(np.float32)
        al = a[..., 3:4] / 255.0
        cell = (a[..., :3] * al + 255 * (1 - al)).astype(np.uint8)
        cell = np.array(Image.fromarray(cell).resize((W // 8, H // 8)))
        cv2.putText(cell, n, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (200, 30, 30), 2)
        cells.append(cell)
    cols = 4
    rows = [np.concatenate(cells[i:i + cols], axis=1)
            for i in range(0, len(cells), cols)]
    wmax = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, wmax - r.shape[1]), (0, 0)),
                   constant_values=255) for r in rows]
    p = os.path.join(spec.work_dir, "layers_contact_sheet.png")
    Image.fromarray(np.concatenate(rows, axis=0)).save(p)
    out["contact_sheet"] = p

    # 3. wiggle test: shift the layers that will actually move at runtime
    movers = [n for n in zorder if rig_layers.get(n, {}).get("spring")]
    if not movers:
        movers = zorder[-3:]
    panels = []
    for n in movers:
        c = composite(shift=(n, (12, 8)))[..., :3]
        panel = np.array(Image.fromarray(c).resize((W // 6, H // 6)))
        cv2.putText(panel, n, (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 80, 80), 2)
        panels.append(panel)
    p = os.path.join(spec.work_dir, "wiggle_test.png")
    Image.fromarray(np.concatenate(panels, axis=1)).save(p)
    out["wiggle"] = p
    return out
