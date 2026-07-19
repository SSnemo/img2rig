"""img2rig: turn a single Stable Diffusion character illustration into a
layered, riggable cutout through an agent-driven pipeline.

Stages: generate (candidates -> hires final) -> variants / pose keyframes ->
split (DINO+SAM draft layering) -> cleanup (refinement) -> export (rig pack).
One YAML spec (see examples/character.yaml) drives everything.
"""
from __future__ import annotations

__version__ = "0.1.0"
