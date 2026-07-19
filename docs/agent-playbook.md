# The agent playbook

img2rig is designed to be **operated by an LLM agent with vision** (Claude
Code or similar). This is not a gimmick; it is the reason the pipeline has no
"manual cleanup" stage. The insight:

> The human labor in AI-assisted layer separation consists of exactly two
> jobs — the *clicker* (placing interactive segmentation prompts) and the
> *QA eye* (judging results). Both are visual-judgment loops, and both can be
> run by an agent that can look at an image: look → choose points → call the
> API → look at the overlay → adjust the point table → rerun.

Every pipeline function therefore writes its observations to disk as images
(probes, colored overlays, contact sheets, diff maps). Those images are the
agent's UI. The point tables and thresholds live in `character.yaml`; the
loop is: run a stage → read its images → edit the spec → rerun. Nothing is
hidden in interactive state, so every iteration is reproducible.

A full character was produced this way end-to-end — six review rounds across
five cleanup passes, zero human intervention. What follows is the distilled
operating manual, in pipeline order.

## 1. Candidate selection (the QA eye)

Review the contact sheet against hard constraints — these are *layering*
requirements, not taste:

- [ ] full body, facing viewer (≤ ~15° turn)
- [ ] arms clear of the torso (touching = unseparable)
- [ ] plain dark background, no frames/borders/text (models love adding card
      frames — keep them in the negative prompt)
- [ ] hair strand ends distinguishable; facial features unoccluded
- [ ] no crossed arms, no props crossing the body

Expect to reject most of a batch (1 hit in 12 is a normal rate). Rejection
reasons cluster; feed them back into the negative prompt for the next roll
(`hands clasped`, `card frame, ornate border`, …).

## 2. Point-prompt segmentation (the clicker)

- **Start from the probe image.** Never guess coordinates; read them off the
  probe/overlay you just generated. After every SAM run, look at the overlay
  before touching the point table.
- **Positive points trace the part's spine; negative points fence the
  neighbors.** For a figure mask: positives down the body midline, negatives
  in each background quadrant. For an arm: a chain of points along
  shoulder→elbow→wrist→hand; sparse points give sparse masks.
- **Area windows are non-negotiable.** Every part gets a plausible
  min/max fraction of frame; out-of-window masks are auto-rejected. This is
  the defense against SAM/DINO handing you the whole figure for "glove".
- **Small parts need the crop pyramid.** In a full frame, one point on an
  eye floods into the face. Crop the face, and for eyes/brows crop again at
  3x. A 2 px point error at eye scale is a miss — expect one or two
  reposition rounds; that's what the loop is for.
- **When semantics fail, go geometric.** Inside a mass of same-colored hair,
  SAM returns the whole mass no matter where you click. Front/back hair,
  left/right of a symmetric skirt, etc. are geometry problems: split by
  lines/regions read off the probe, then clean with connected components.
- **Background bleed is semantic, not morphological.** When a background
  ornament touches the figure with similar colors, erosion/bridge-cutting
  fails (they're broadly connected). One negative point placed on the
  ornament excludes it cleanly.
- **Don't trust the figure mask for accessories.** SAM's "person" concept
  excludes horns, crowns, oversized props. Intersecting accessory masks with
  the figure mask deletes them; anchor accessories with seeded connected
  components instead.

## 3. Occlusion inpainting

Three essentials, learned from painting-over-the-wrong-thing failures:

1. **The hole is the reveal band, not the occluder.** Punch out only the
   narrow strip the lower layer will actually expose in motion (hair rim ≈
   12 px, the dilated eye socket for blinks) — carving the whole occluder
   asks SD to repaint hair it can see.
2. **Base fill = sampled flat tone.** Median-sample the actual skin (avoid
   highlight zones), flat-fill the hole, then a small-radius TELEA only to
   blend edges. Raw TELEA drags adjacent hair color across the hole.
3. **Melt the patch.** `mask_blur ≈ 16`, denoise ≈ 0.55 over the hole makes
   the patch invisible; without it you get a "band-aid" seam.

For pose keyframes, invert the trick: hot denoise (0.6–0.9) for a real pose
change, identity protected by *cutting the face out of the mask* and leaving
headroom for the new limbs.

## 4. Acceptance — the three artifacts

Run all three after assembly; each catches a different failure class:

1. **Flatten diff** — recomposite all layers, diff against the source over
   the figure region. A few percent (patches + feathered edges) is expected;
   streaks or blocks mean a layer went missing or double-assigned.
2. **Layer contact sheet** — every layer on white, labeled and thumbnailed.
   Catches bleed-through (background chunks riding in a mask) and starved
   layers at a glance.
3. **Wiggle test** — shift each articulated layer ±12 px and recomposite.
   This is the cheap proxy for rig motion: any hole or tear it exposes, the
   spring physics will expose worse.

Also do one **in-engine check**: load the exported rig and look for pale
edge halos (color bleed missed a layer) and content showing *through* the
figure (a starved mask left a hole — semi-transparent fabrics are the usual
culprit; add positive points there and rerun).

## 5. Small interventions between stages

The pipeline's functions are library calls and the masks directory is the
contract: any stage can be followed by a few lines of numpy that read a mask,
apply a geometric correction, and write it back. The demo character needed
four such moves — a y-floor clamp on the hair masks (a collar shadow was
riding with the head), a distant-crumb connected-component filter on the arm
masks, a third crop level for one bang-occluded eye, and a cloned "neck pad"
appended to the z1 underlay so head sway reveals painted collar instead of
background. Don't force these through the spec; script them, and record what
you did in the character's notes.

## 6. General discipline

- One variable per iteration: change the point table *or* a threshold, not
  both, or you can't attribute the result.
- Log every round's parameters in the spec file (comments are fine) — the
  next character reuses the structure with new coordinates, and the diffs
  are the documentation.
- Keep every rejected candidate and mask on disk (`work_dir` is cheap).
  Yesterday's reject explains today's artifact.
- The pipeline is per-character deterministic given the spec (fixed seeds
  everywhere), so "rerun the stage" is always safe.
