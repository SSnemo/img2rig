"""Base illustration generation: candidate batches and hires finalization.

Workflow: roll a wide batch of txt2img candidates at the checkpoint's native
resolution, eyeball the contact sheet (agent or human), then re-run the picked
seed with hires-fix to get the working-resolution final.

Hard-won notes baked into this module:

* SD1.5-family checkpoints degrade fast above their native training size -
  a standing figure asked for directly above ~832x1216 tends to grow a second
  head or extra torso. So candidates are generated at ``sd.base_size`` and all
  enlargement happens through hires-fix (finalize) or a GAN upscaler
  (upscale_4k), never by asking the sampler for a bigger canvas.
* Expect a low hit rate on candidates (pose constraints, hand count, symmetry
  all fail independently) - roll wide and pick, don't re-roll one seed.
* finalize reuses the exact prompt and seed of the picked candidate: hires-fix
  refines the same latent trajectory, so the composition is preserved.
  Changing even one tag at the same seed shifts the composition - which is why
  state variants and pose keyframes are img2img jobs (see variant.py /
  keyframes.py), never a txt2img re-roll with edited tags.
"""
from __future__ import annotations

import os

import requests

from .config import Spec
from .sdapi import TIMEOUT, Client, _b64_file, _im_of, contact_sheet


def candidates(spec: Spec, client: Client) -> list[str]:
    """Generate spec["gen"]["candidates"] txt2img candidates (seed0, seed0+1,
    ...) into work_dir/candidates/ and build a contact sheet next to them.

    Returns the list of candidate image paths.
    """
    g = spec["gen"]
    w, h = spec.sd["base_size"]
    out = os.path.join(spec.work_dir, "candidates")
    os.makedirs(out, exist_ok=True)
    paths: list[str] = []
    for i in range(g["candidates"]):
        seed = g["seed0"] + i
        im = client.txt2img(spec.positive(), spec.negative(), w, h, seed)
        p = os.path.join(out, f"{spec.name}_cand_{i:02d}_seed{seed}.png")
        im.save(p)
        paths.append(p)
        print(f"candidate {i:02d} seed={seed}")
    tile = tuple(g.get("contact_tile", [300, 439]))
    sheet_path = os.path.join(spec.work_dir, "contact_sheet.png")
    contact_sheet(paths, tile, cols=4, out_path=sheet_path)
    print("contact sheet:", sheet_path)
    return paths


def finalize(spec: Spec, client: Client, seed: int) -> str:
    """Re-run the picked candidate seed with hires-fix enabled.

    Same prompt + same seed + hires-fix reproduces the chosen composition at
    ``base_size * hires.scale``. Saved as
    work_dir/{name}_final_hires_seed{seed}.png - the canonical working image
    every later stage (variants, keyframes, split) starts from.
    """
    w, h = spec.sd["base_size"]
    im = client.txt2img(spec.positive(), spec.negative(), w, h, seed,
                        hires=spec.sd.get("hires", {}))
    p = os.path.join(spec.work_dir, f"{spec.name}_final_hires_seed{seed}.png")
    im.save(p)
    print("hires final saved:", p, im.size)
    return p


def upscale_4k(spec: Spec, client: Client, src_path: str, scale: float = 2.0,
               upscaler: str | None = None) -> str:
    """Optional GAN upscale of a finished image via the extras endpoint
    (no diffusion, purely deterministic - safe to run last, after retouching).

    Typically takes the hires final to ~4K for layer cutting: masks cut at
    higher resolution have cleaner edges. Defaults to the hires upscaler from
    the spec unless a dedicated one is given.
    """
    if upscaler is None:
        upscaler = spec.sd.get("hires", {}).get("upscaler", "R-ESRGAN 4x+ Anime6B")
    r = requests.post(f"{client.api}/sdapi/v1/extra-single-image", json={
        "image": _b64_file(src_path),
        "upscaling_resize": scale,
        "upscaler_1": upscaler,
    }, timeout=TIMEOUT)
    r.raise_for_status()
    im = _im_of(r.json()["image"])
    stem = os.path.splitext(os.path.basename(src_path))[0]
    p = os.path.join(spec.work_dir, f"{stem}_x{scale:g}.png")
    im.save(p)
    print("upscaled:", p, im.size)
    return p
