# The `rig.txt` format

A rig is a directory containing one `rig.txt` manifest plus one PNG per layer
(and optionally full-frame pose keyframes). The format is a deliberately tiny
line-based text file — trivially parseable in any language, diff-friendly, and
hand-editable while tuning.

```
# comment
canvas <w> <h>
node <name> <parent> <z> <pivot_x> <pivot_y>
layer <name> <file> <x> <y> <w> <h> <pivot_x> <pivot_y> <parent> <z> [spring <k> <c>]
```

- **canvas** — the rig's logical coordinate space. All positions below are in
  canvas pixels. Renderers scale uniformly (`display_height / canvas_h`).
- **node** — a virtual pivot (no image). `parent` is another node's name or
  `root`. Parents must be declared before children.
- **layer** — one image. `x y w h` is the layer's bounding box on the canvas;
  `pivot` is its rotation anchor; `z` is draw order (ascending = back to
  front). The optional `spring k c` suffix attaches an angular spring
  (stiffness / damping) that lags parent motion — hair, capes, skirts.

## Name-driven conventions

The runtime recognizes a small vocabulary of names instead of an animation
file format. Rigs that omit any of these simply lose that behavior — nothing
breaks:

| Name | Behavior |
|---|---|
| `body_pivot` (node) | breath bob, hit recoil (`kickX`), lean, sink apply here |
| `head_pivot` (node) | nod/shake (`headAngle`) + slight breathing nod |
| `arm_L`, `arm_R` | extra rotation around their own pivot (shoulder joint) |
| `eye_L`, `eye_R` | blink: vertical squash anchored at the bottom edge |
| `sigil` | emissive layer; alpha driven by rage/charge (default hidden) |

Pivot placement rules of thumb: head = neck root, cape = shoulders, hair =
hair root, arm = shoulder. The export tool defaults any unspecified pivot to
the layer's bbox center.

## Pose keyframes (optional)

Two files in the rig directory, sharing the rig canvas and framing:

- `kf_windup.png` — held during an attack windup (drawn with tremble jitter)
- `kf_swing.png` — flashed for ~0.2 s at release, then layered drawing resumes

A rotate-the-cutouts rig can never show a pose the source illustration
doesn't contain; whole-frame swaps cover the big reads (arms overhead, a full
swing) while springs and layers carry everything in between. Missing files
degrade gracefully to layered posing.

## Coordinate example

```
canvas 1232 1800
node body_pivot root -1 635.0 1110.2
node head_pivot body_pivot -1 677.2 384.9
layer hair_front hair_front.png 546.2 6.7 360 363 677.2 170.2 head_pivot 14 spring 60.0 8.0
```

`hair_front` is a 360x363 image whose top-left sits at (546.2, 6.7) on the
canvas, rotates around (677.2, 170.2) — the hair root — follows `head_pivot`,
draws at z=14, and swings on a light, snappy spring.
