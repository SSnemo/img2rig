"""Character spec loading. One YAML file drives the whole pipeline."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass
class Spec:
    """Parsed character spec. Raw dict access via spec[...] for pipeline
    sections; typed helpers for the fields everything touches."""

    raw: dict[str, Any]
    path: str

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    @property
    def name(self) -> str:
        return self.raw["name"]

    @property
    def work_dir(self) -> str:
        d = self._resolve(self.raw.get("work_dir", f"work/{self.name}"))
        os.makedirs(d, exist_ok=True)
        return d

    @property
    def out_dir(self) -> str:
        d = self._resolve(self.raw.get("out_dir", f"out/{self.name}"))
        os.makedirs(d, exist_ok=True)
        return d

    @property
    def api(self) -> str:
        return self.raw["sd"]["api"]

    @property
    def sd(self) -> dict[str, Any]:
        return self.raw["sd"]

    def positive(self, extra: str = "") -> str:
        p = self.raw["prompt"]
        parts = [p["quality"], p["subject"], p.get("style", ""), p["pose_constraints"]]
        if extra:
            parts.insert(3, extra)
        return ", ".join(s for s in parts if s)

    def negative(self, extra: str = "") -> str:
        n = self.raw["prompt"]["negative"]
        return f"{n}, {extra}" if extra else n

    def _resolve(self, p: str) -> str:
        if os.path.isabs(p):
            return p
        return os.path.normpath(os.path.join(os.path.dirname(self.path), p))


def load(path: str) -> Spec:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    for key in ("name", "sd", "prompt"):
        if key not in raw:
            raise ValueError(f"character spec missing required section: {key}")
    return Spec(raw=raw, path=os.path.abspath(path))
