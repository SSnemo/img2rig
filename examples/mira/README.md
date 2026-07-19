# Mira — the demo character

A complete worked example, produced end-to-end by the pipeline with an agent
operating the visual loop (8 review rounds, no human mask edits):

- `mira.yaml` — her full spec: prompts, SD parameters, the SAM point tables as
  they ended up after iteration, and the rig tree. This is the file to copy
  when starting your own character.
- `rig/` — the exported layer pack (13 layers + `rig.txt`). Renders with
  `img2rig preview --spec examples/mira/mira.yaml`, the C++ reference player,
  or ~50 lines of any engine's sprite code.

Provenance: base illustration generated locally with an SD1.5-family
checkpoint (seed recorded in the spec, reproducible), layered and exported by
this repository's pipeline. To the extent permissible, the image assets in
this directory are dedicated to the public domain under
[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/). Attribution
appreciated, not required.

Notes from her production run (also folded into the agent playbook):

- Her brows sit fully under the bangs — the layers were dropped, and the rig
  degrades gracefully (name-driven behaviors just skip missing layers).
- Her long dress hides the legs — `skirt` (with a spring) replaces the usual
  `legs` layer.
- The dark navy pauldron inlays read as background to SAM until positive
  points were placed directly on them: watch for same-value-as-background
  costume elements.
- The bang-occluded right eye needed a third crop level (6x) plus a geometric
  socket clamp; the left eye segmented in one shot at 3x.
- A cloned "neck pad" was added to the z1 underlay so head sway reveals
  painted collar instead of background — the cheap version of occlusion
  backfill, good enough below ~0.1 rad of head travel.
- Layer tearing under big motion (stagger-scale leans) was eliminated by
  the assemble stage's `overlap_px: 18` expansion; her rest-pose composite
  is pixel-identical with or without it (flatten diff 0.65% both ways).
- `rig/kf_windup.png` / `rig/kf_swing.png` are her full-frame strike
  keyframes (booru-tag weighted img2img over the base, matted by SAM +
  identity-diff union + dark-costume keying — the near-black cape survives
  none of the three alone). The C++ runtime swaps them in during
  `strikeWindup`/`strikeRelease`; the Python preview doesn't use them.
