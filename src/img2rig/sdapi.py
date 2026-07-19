"""Thin client for an AUTOMATIC1111-compatible Stable Diffusion WebUI API,
plus the sd-webui-segment-anything extension (GroundingDINO + SAM).

Everything speaks base64 PNG over HTTP; nothing here depends on which
checkpoint is loaded. The pipeline never distributes model weights - point
the WebUI at your own.
"""
from __future__ import annotations

import base64
import io

import numpy as np
import requests
from PIL import Image

TIMEOUT = 1800


def _b64_of(im: Image.Image) -> str:
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def _b64_file(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _im_of(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def _mask_of(b64: str) -> np.ndarray:
    return np.array(_im_of(b64).convert("L")) > 127


class Client:
    def __init__(self, api: str, sd_cfg: dict | None = None):
        self.api = api.rstrip("/")
        self.cfg = sd_cfg or {}

    # ---- diffusion ----

    def _common(self, payload: dict) -> dict:
        payload.setdefault("sampler_name", self.cfg.get("sampler", "DPM++ 2M Karras"))
        payload.setdefault("steps", self.cfg.get("steps", 30))
        payload.setdefault("cfg_scale", self.cfg.get("cfg", 7))
        if self.cfg.get("checkpoint"):
            payload.setdefault("override_settings", {})[
                "sd_model_checkpoint"] = self.cfg["checkpoint"]
        return payload

    def txt2img(self, prompt: str, negative: str, w: int, h: int, seed: int,
                hires: dict | None = None, **kw) -> Image.Image:
        payload = self._common({
            "prompt": prompt, "negative_prompt": negative,
            "width": w, "height": h, "seed": seed, **kw,
        })
        if hires:
            payload.update({
                "enable_hr": True,
                "hr_scale": hires.get("scale", 2.0),
                "hr_upscaler": hires.get("upscaler", "R-ESRGAN 4x+ Anime6B"),
                "denoising_strength": hires.get("denoise", 0.4),
                "hr_second_pass_steps": hires.get("steps", 20),
            })
        r = requests.post(f"{self.api}/sdapi/v1/txt2img", json=payload, timeout=TIMEOUT)
        r.raise_for_status()
        return _im_of(r.json()["images"][0])

    def img2img(self, init: Image.Image, prompt: str, negative: str, seed: int,
                denoise: float, mask: Image.Image | None = None, **kw) -> Image.Image:
        payload = self._common({
            "init_images": [_b64_of(init)], "prompt": prompt,
            "negative_prompt": negative, "seed": seed,
            "denoising_strength": denoise,
            "width": init.width, "height": init.height, **kw,
        })
        if mask is not None:
            payload.update({
                "mask": _b64_of(mask), "mask_blur": kw.get("mask_blur", 16),
                "inpainting_fill": 1, "inpaint_full_res": False,
                "inpainting_mask_invert": 0,
            })
        r = requests.post(f"{self.api}/sdapi/v1/img2img", json=payload, timeout=TIMEOUT)
        r.raise_for_status()
        return _im_of(r.json()["images"][0])

    # ---- segmentation (sd-webui-segment-anything) ----

    def sam_points(self, image: Image.Image, pos: list, neg: list) -> list[np.ndarray]:
        """SAM point-prompt mode: the workhorse of the agent's visual loop.
        Returns candidate boolean masks (typically 3)."""
        r = requests.post(f"{self.api}/sam/sam-predict", json={
            "sam_model_name": self.cfg.get("sam_model", "sam_vit_h_4b8939.pth"),
            "input_image": _b64_of(image),
            "sam_positive_points": pos,
            "sam_negative_points": neg,
        }, timeout=600)
        r.raise_for_status()
        return [_mask_of(m) for m in r.json().get("masks", [])]

    def sam_text(self, image: Image.Image, prompt: str, thr: float) -> list[np.ndarray]:
        """GroundingDINO text prompt -> SAM masks (first-pass draft layering)."""
        r = requests.post(f"{self.api}/sam/sam-predict", json={
            "sam_model_name": self.cfg.get("sam_model", "sam_vit_h_4b8939.pth"),
            "input_image": _b64_of(image),
            "dino_enabled": True,
            "dino_model_name": self.cfg.get("dino_model", "GroundingDINO_SwinT_OGC (694MB)"),
            "dino_text_prompt": prompt,
            "dino_box_threshold": thr,
        }, timeout=600)
        r.raise_for_status()
        return [_mask_of(m) for m in r.json().get("masks", [])]


# ---- shared helpers ----

def contact_sheet(paths: list[str], tile: tuple[int, int], cols: int = 4,
                  out_path: str | None = None) -> Image.Image:
    """Grid thumbnail sheet - the artifact the agent (or human) eyeballs to
    pick candidates."""
    tw, th = tile
    rows = max(1, (len(paths) + cols - 1) // cols)
    sheet = Image.new("RGB", (tw * cols, th * rows), (20, 20, 24))
    for i, p in enumerate(paths):
        sheet.paste(Image.open(p).resize((tw, th)), ((i % cols) * tw, (i // cols) * th))
    if out_path:
        sheet.save(out_path)
    return sheet
