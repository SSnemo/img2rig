"""First-pass draft layering: text-grounded boxes -> SAM masks -> PSD draft.

Stage [2] of the pipeline. A GroundingDINO text prompt proposes boxes, SAM
turns them into masks, and an area window keeps only the plausible pick.
The result is a rough but complete layer stack that an agent inspects (via
the candidate strips and overlay written to work_dir) and then refines with
the point-prompt tools in cleanup.py.
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image

from .config import Spec
from .sdapi import Client

#: Distinct overlay colors, cycled by part index in every preview image.
PALETTE: list[tuple[int, int, int]] = [
    (255, 80, 80), (255, 160, 40), (60, 220, 60), (60, 160, 255),
    (255, 60, 255), (40, 230, 230), (255, 255, 90), (160, 90, 255),
    (255, 110, 200), (90, 130, 255), (200, 255, 120), (255, 200, 40),
]


# ---- shared mask store (work_dir/masks/mask_<name>.png) ----

def masks_dir(spec: Spec) -> str:
    """Directory holding every intermediate mask for this character."""
    d = os.path.join(spec.work_dir, "masks")
    os.makedirs(d, exist_ok=True)
    return d


def save_mask(spec: Spec, name: str, mask: np.ndarray) -> str:
    """Persist a boolean mask as mask_<name>.png; returns the path."""
    path = os.path.join(masks_dir(spec), f"mask_{name}.png")
    Image.fromarray(mask.astype(np.uint8) * 255).save(path)
    return path


def load_mask(spec: Spec, name: str,
              size: tuple[int, int] | None = None) -> np.ndarray | None:
    """Load mask_<name>.png as a boolean array, or None if absent.

    size=(W, H) resizes to the target resolution (masks are stored at
    whatever resolution the stage that made them worked in).
    """
    path = os.path.join(masks_dir(spec), f"mask_{name}.png")
    if not os.path.exists(path):
        return None
    im = Image.open(path).convert("L")
    if size is not None and im.size != tuple(size):
        im = im.resize(tuple(size), Image.BILINEAR)
    return np.array(im) > 127


def tint(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int],
         strength: float = 0.6) -> np.ndarray:
    """Return a copy of an RGB array with `mask` blended toward `color`."""
    out = rgb.copy()
    c = np.array(color, dtype=np.float32)
    out[mask] = (out[mask] * (1.0 - strength) + c * strength).astype(np.uint8)
    return out


def cut_rgba(image: Image.Image | np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Full-frame RGBA copy of `image` with alpha zeroed outside `mask`."""
    arr = np.array(image.convert("RGBA")) if isinstance(image, Image.Image) else image
    out = arr.copy()
    out[..., 3] = np.where(mask, arr[..., 3], 0)
    return out


def split_lr(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split one mask into (left, right) halves at its bbox center column.

    Draft-stage convenience for paired parts that DINO detects as a single
    blob (eyes -> eye_L/eye_R, arms -> arm_L/arm_R).
    """
    xs = np.where(mask.any(axis=0))[0]
    if len(xs) == 0:
        return mask.copy(), np.zeros_like(mask)
    cx = int((xs.min() + xs.max()) // 2)
    left = mask.copy()
    left[:, cx:] = False
    right = mask.copy()
    right[:, :cx] = False
    return left, right


# ---- draft layering ----

def _candidate_strip(rgb: np.ndarray, cands: list[np.ndarray], out_path: str) -> None:
    """Horizontal strip of every SAM candidate tinted red - the image the
    agent looks at when a part's prompt or area window needs iterating."""
    h, w = rgb.shape[:2]
    tiles = [np.array(Image.fromarray(tint(rgb, c, (255, 40, 40)))
                      .resize((max(1, w // 4), max(1, h // 4)))) for c in cands]
    Image.fromarray(np.concatenate(tiles, axis=1)).save(out_path)


def draft(spec: Spec, client: Client, hires_path: str) -> dict[str, np.ndarray]:
    """Run the DINO text-prompt table (spec["split"]["parts"]) over the hires
    frame and produce one draft mask per part.

    Each part entry: {name, prompts, thr, min, max, pick}. Prompts are tried
    in order until one yields a candidate inside the area window
    [min, max] (fraction of frame). The area window is the defense against
    DINO's habit of boxing the entire figure for clothing words and
    small-part words - without it "sleeve" happily returns the whole body.
    pick chooses among in-window candidates: small | mid | large.

    Masks are saved to work_dir/masks/mask_<name>.png; per-part candidate
    strips, a combined draft_overlay.png and split_log.txt land next to them
    as the agent's observation surface. Returns {name: bool mask} at hires
    resolution.
    """
    im = Image.open(hires_path).convert("RGB")
    rgb = np.array(im)
    total = im.width * im.height
    mdir = masks_dir(spec)
    masks: dict[str, np.ndarray] = {}
    log: list[str] = []

    for part in spec["split"]["parts"]:
        name = part["name"]
        pref = part.get("pick", "mid")
        found = False
        for prompt in part["prompts"]:
            try:
                cands = client.sam_text(im, prompt, part["thr"])
            except Exception as exc:  # network / extension errors: log and move on
                log.append(f"{name}/{prompt}: API error {exc}")
                continue
            if not cands:
                log.append(f"{name}/{prompt}: no masks")
                continue
            _candidate_strip(rgb, cands, os.path.join(mdir, f"{name}_candidates.png"))
            areas = [int(c.sum()) for c in cands]
            ok = [i for i in np.argsort(areas)
                  if part["min"] <= areas[i] / total <= part["max"]]
            if not ok:
                log.append(f"{name}/{prompt}: out of area window, "
                           f"areas={[f'{a / total:.3%}' for a in areas]}")
                continue
            pick = ok[0] if pref == "small" else (
                ok[-1] if pref == "large" else ok[len(ok) // 2])
            masks[name] = cands[pick]
            save_mask(spec, name, cands[pick])
            log.append(f"{name}/{prompt}: picked {areas[pick] / total:.3%} of frame")
            found = True
            break
        if not found:
            log.append(f"{name}: FAILED all prompts -> left for cleanup stage")

    ov = rgb.copy()
    for i, m in enumerate(masks.values()):
        ov = tint(ov, m, PALETTE[i % len(PALETTE)])
    Image.fromarray(ov).resize((im.width // 2, im.height // 2)).save(
        os.path.join(mdir, "draft_overlay.png"))
    with open(os.path.join(spec.work_dir, "split_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(log) + "\n")
    print("\n".join(log))
    return masks


# ---- PSD assembly ----

def to_psd(spec: Spec, layers: dict[str, np.ndarray], out_path: str) -> str:
    """Assemble full-frame RGBA layers into a PSD for manual touch-up.

    `layers` maps layer name -> full-frame RGBA array (see cut_rgba). Layer
    order follows spec["rig"]["layers"] z where the name is known (front-most
    first in the file); unknown names keep dict order after those.

    pytoshop is an optional dependency: `pip install img2rig[psd]`.
    """
    try:
        from pytoshop import enums
        from pytoshop.user import nested_layers as nl
    except ImportError as exc:
        raise ImportError(
            "PSD export requires the optional dependency pytoshop; "
            "install it with `pip install img2rig[psd]` (or `pip install "
            "pytoshop`). All other pipeline stages work without it."
        ) from exc

    rig_layers = (spec.get("rig") or {}).get("layers", {})
    known = sorted((n for n in layers if n in rig_layers),
                   key=lambda n: -rig_layers[n].get("z", 0))
    order = known + [n for n in layers if n not in rig_layers]

    items = []
    W = H = 0
    for name in order:
        arr = layers[name]
        H, W = arr.shape[:2]
        ys, xs = np.where(arr[..., 3] > 0)
        if len(ys) == 0:
            continue
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        c = arr[y0:y1, x0:x1]
        items.append(nl.Image(
            name=name, top=int(y0), left=int(x0),
            channels={0: c[..., 0], 1: c[..., 1], 2: c[..., 2], -1: c[..., 3]}))
    if not items:
        raise ValueError("no non-empty layers to write")

    # Two pytoshop landmines, learned the hard way:
    # - Compression.raw on purpose: the RLE path crashes when pytoshop's
    #   cython extensions are not compiled (the common pure-python install).
    # - The docs describe size as (height, width), but the implementation
    #   unpacks `width, height = size` - so pass (W, H).
    psd = nl.nested_layers_to_psd(items, color_mode=enums.ColorMode.rgb,
                                  size=(W, H), compression=enums.Compression.raw)
    with open(out_path, "wb") as f:
        psd.write(f)
    return out_path
