---
name: img2rig-pipeline
description: Operate the img2rig pipeline - turn a single SD-generated illustration into a layered, breathing character rig via an agent visual loop. Use when the user wants to generate a rigged 2D character, run any img2rig stage (gen/variant/keyframes/split/cleanup/export), or iterate SAM point tables against probe images.
---

# Operating img2rig

You are the operator of a generate → segment → assemble → export pipeline.
Your two jobs are the **clicker** (author SAM point tables in the character
spec) and the **QA eye** (judge every stage's output images). Full doctrine:
`docs/agent-playbook.md`; stage reference: `docs/pipeline.md`.

## Ground rules

1. **Look before you write.** Every stage drops probes/overlays/contact
   sheets in `work_dir`. Read the image, then edit `character.yaml`, then
   rerun the stage. Never guess coordinates; read them off the probe.
2. One variable per iteration. Point table *or* threshold, not both.
3. All state lives in the spec + `work_dir`. Rerunning any stage is safe.
4. Never ship weights, and never write game/project-specific content into
   this repo's code — everything character-specific belongs in the spec.

## Workflow

```
img2rig gen        --spec char.yaml      # candidates + contact sheet
# -> review contact_sheet.png against the hard checklist (playbook §1)
img2rig finalize   --spec char.yaml --seed <picked>
img2rig variant    --spec char.yaml --name rage --seed <picked>
img2rig keyframes  --spec char.yaml      # windup/swing candidate ladders
# -> review per-frame contact sheets; pick per frame
img2rig split      --spec char.yaml      # DINO/SAM draft masks + overlays
# -> review overlays; then iterate cleanup point tables:
img2rig cleanup    --spec char.yaml      # refine masks, fill, derive, assemble
# -> loop: overlay/probe images -> edit spec points -> rerun (typically 4-6 rounds)
# -> gate on the three QA artifacts: flatten_diff / contact sheet / wiggle test
img2rig export     --spec char.yaml      # layer pack + rig.txt (+ kf_*.png)
```

## Judging quick-reference

- Candidate hit rate ~1/12 is normal; reject hard, recycle reject reasons
  into the negative prompt.
- Small parts (eyes/brows/mouth): only ever segment inside the crop pyramid.
- Same-colored regions (hair front/back): geometric split, not more points.
- Accessories (horns/crowns): never intersect with the figure mask.
- Inpaint holes: the reveal band only; flat skin fill; melt with mask_blur.
- After export, load the rig in-engine (or the reference player) and check
  for edge halos and see-through holes before calling it done.
