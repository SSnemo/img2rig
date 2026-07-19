# The pipeline

Six stages take a text prompt to a breathing character. Stages 1–2 are pure
generation; 3–5 turn one picked image into layers; 6 is the runtime. Every
stage reads the same `character.yaml` spec and drops its intermediates in
`work_dir` — including the probe/overlay/contact-sheet images that the
operator (human or agent, see [agent-playbook.md](agent-playbook.md)) reviews
between runs.

```
[1] generate    txt2img candidate batch -> contact sheet -> pick -> hires finalize
[2] variants    img2img state variants (glow/rage/damage) + pose keyframes
[3] split       GroundingDINO text prompts -> SAM masks -> draft layers / PSD
[4] cleanup     SAM point-prompt refinement, geometric splits, occlusion fill,
                derived emissive layer, mutual-exclusion assembly, QA sheets
[5] export      feather + color bleed + bbox crop -> layer PNGs + rig.txt
[6] runtime     header-only C++ player: springs, breathing, blinks, strikes
```

## Requirements

- An [AUTOMATIC1111-compatible](https://github.com/AUTOMATIC1111/stable-diffusion-webui)
  Stable Diffusion WebUI running with `--api`
- The [sd-webui-segment-anything](https://github.com/continue-revolution/sd-webui-segment-anything)
  extension with a SAM checkpoint and GroundingDINO (set
  `sam_use_local_groundingdino: true` to avoid a runtime GitHub fetch; the
  DINO model name must include its size suffix, e.g.
  `"GroundingDINO_SwinT_OGC (694MB)"`)
- A checkpoint of your choice. **img2rig ships no model weights** and is
  model-agnostic; tag-trained anime checkpoints respond best to the
  booru-style prompt templates in the example spec.

## Stage notes — the load-bearing details

Each of these was learned the hard way; the pipeline encodes them, and the
spec exposes the knobs.

**[1] Generation.**
- Candidate batches at the model's native size (832x1216 for SD1.5-family;
  larger grows extra heads). Enlarge only via hires-fix and GAN upscalers.
- Expect a *low* hit rate against the layering constraints (~1 in 12 is
  normal). Roll wide, pick strictly — see the acceptance checklist in the
  playbook.
- The `pose_constraints` prompt block is not style, it is engineering: arms
  away from the body (touching arms can't be separated), plain dark
  background (clean matting), full body facing viewer (rig symmetry).
- Don't fight the model for held weapons/props — obedience is poor and a
  separately composited prop layer rigs better anyway.

**[2] Variants & keyframes.**
- Same-seed txt2img with edited tags **drifts the whole composition** —
  variants must be img2img on the picked image. Denoise ~0.55 is the sweet
  spot: identity and framing hold, the new elements (glowing eyes, aura)
  land. Below ~0.45 the change is too weak; above ~0.7 identity erodes.
- Pose keyframes push much hotter denoise (0.6–0.9), so they protect
  identity structurally instead: an upper-body mask with the face (and any
  signature headwear) punched *out* of it, plus headroom above the head so
  raised arms have somewhere to be painted.
- Pose words work best as weighted booru tags (`(arms up:1.4)`), not prose.

**[3] Split (draft).**
- DINO text prompts localize parts, SAM masks them. **Every part needs an
  area window** (fraction of frame): clothing and small-part words routinely
  box the entire figure, and the window is the only defense.
- SAM inference runs on the hires image; masks upscale bilinearly to the 4K
  layer source. Quality holds, VRAM doesn't blow up.

**[4] Cleanup.** The refinement stage; the playbook covers the operator
loop. Key algorithms:
- Small facial parts segment from a **two-level crop pyramid** (face crop,
  then a 3x-scaled eye band) — at full frame a single SAM point bleeds into
  half the face.
- Front/back hair is a **geometric** split, not a semantic one: SAM returns
  the whole mass wherever you click inside same-colored hair. Below the eye
  line + outside the cheeks = the back strands.
- SAM's whole-figure mask tends to exclude headwear/horns as "not person" —
  never intersect derived accessory masks with it; use seeded connected
  components instead.
- Emissive layer = base-vs-variant pixel diff, but **only** a single-channel
  gain inside a dilated ROI around the glowing parts. A global |diff| is
  img2img noise everywhere.
- Occlusion fill (the skin that shows when hair swings): the hole is the
  narrow rim that will actually be exposed, the base fill is a flat
  sampled-skin tone (naive TELEA drags hair color in), and a feathered
  low-denoise inpaint melts the edges.
- Assembly is mutual-exclusion by priority (mask-confidence order, first
  grab wins), with a small morphological close (a big radius swallows
  neighboring parts) and connected-component filtering on the remainder.
- **Overlap expansion** (`assemble.overlap_px`) is the anti-tearing pass:
  after exclusion, each lower-z layer clones source pixels a few px into
  its higher-z neighbors' territory. The clone is invisible at rest (the
  upper layer covers it) and is exactly what motion reveals instead of a
  background-colored gap. The flatten-diff QA proves the rest pose is
  pixel-identical either way.

**[5] Export.**
- 2 px alpha feather (premultiplied pipelines turn dirty edges into dark
  rims), then **color bleed**: transparent pixels inherit RGB from the
  nearest opaque pixels, or linear sampling mixes the background color into
  every layer edge (reads as a white halo in-engine).
- Layers crop to bbox, rescale to the rig canvas, and `rig.txt` records
  where they sit — see [rig-format.md](rig-format.md).

**[6] Runtime.**
- Two headers, no dependencies, draw callback of your choice. Springs +
  breath + blink carry idle; procedural impulses carry hits/staggers/deaths;
  full-frame keyframes carry the big attack reads. Motions never lock game
  timing — logic is the source of truth, the rig only performs.
