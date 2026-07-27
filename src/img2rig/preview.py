"""Animated preview of an exported rig, no engine required.

A faithful Python port of the runtime's solver (rig_math.h/rig_player.h):
breath, blink, head sway, spring physics, hit/stagger/enrage impulses. Renders
frames with PIL and writes an animated GIF (or a WebP with alpha) - the
quickest way to eyeball an export and the tool that makes README demos.

This duplicates the C++ math on purpose: the runtime stays dependency-free,
and the preview stays pure Python. If you change one, change both (the C++
unit tests pin the solver's behavior).
"""
from __future__ import annotations

import math
import os
import random
from dataclasses import dataclass, field

from PIL import Image

from .config import Spec


# ---- rig.txt parsing (mirrors img2rig::parse) ----

@dataclass
class Node:
    name: str
    parent: str
    z: int
    px: float = 0.0
    py: float = 0.0
    file: str = ""
    x: float = 0.0
    y: float = 0.0
    w: float = 0.0
    h: float = 0.0
    is_layer: bool = False
    spring: tuple[float, float] | None = None
    parent_idx: int = -1
    spring_a: float = 0.0
    spring_v: float = 0.0
    prev_parent: float = 0.0  # parent angle last frame (for velocity drive)


@dataclass
class Rig:
    canvas: tuple[float, float]
    nodes: list[Node] = field(default_factory=list)


def parse(path: str) -> Rig:
    rig = Rig(canvas=(0, 0))
    with open(path, encoding="utf-8") as f:
        for line in f:
            t = line.split()
            if not t or t[0].startswith("#"):
                continue
            if t[0] == "canvas":
                rig.canvas = (float(t[1]), float(t[2]))
            elif t[0] == "node":
                rig.nodes.append(Node(name=t[1], parent=t[2], z=int(t[3]),
                                      px=float(t[4]), py=float(t[5])))
            elif t[0] == "layer":
                n = Node(name=t[1], parent=t[9], z=int(t[10]), file=t[2],
                         x=float(t[3]), y=float(t[4]), w=float(t[5]),
                         h=float(t[6]), px=float(t[7]), py=float(t[8]),
                         is_layer=True)
                if len(t) > 11 and t[11] == "spring":
                    n.spring = (float(t[12]), float(t[13]))
                rig.nodes.append(n)
    names = {n.name: i for i, n in enumerate(rig.nodes)}
    for n in rig.nodes:
        n.parent_idx = names.get(n.parent, -1)
    return rig


# ---- solver (mirrors img2rig::solve + the RigPlayer parameter machine) ----

@dataclass
class Params:
    breath: float = 0.0
    head_angle: float = 0.0
    eye_open: float = 1.0
    lean: float = 0.0
    kick_x: float = 0.0
    sink: float = 0.0
    rage: float = 0.0
    fade: float = 1.0
    arm_l: float = 0.0
    arm_r: float = 0.0
    # ankle-anchored whole-rig tilt in radians (hit reactions): every layer
    # rotates rigidly around a ground anchor at the canvas bottom - feet
    # stationary, head max, zero relative layer displacement by construction
    bend: float = 0.0


def _rot(px: float, py: float, cx: float, cy: float, a: float) -> tuple[float, float]:
    s, c = math.sin(a), math.cos(a)
    ox, oy = cx - px, cy - py
    return px + ox * c - oy * s, py + ox * s + oy * c


def solve(rig: Rig, p: Params, dt: float) -> list[tuple[float, float, float, float]]:
    """Returns per-node (cx, cy, rot, alpha); advances springs when dt > 0."""
    acc: list[tuple[float, float, float]] = []  # rot, dx, dy per node
    out: list[tuple[float, float, float, float]] = []
    for n in rig.nodes:
        rot, dx, dy = acc[n.parent_idx] if n.parent_idx >= 0 else (0.0, 0.0, 0.0)
        local = 0.0
        if not n.is_layer:
            if n.name == "body_pivot":
                # translations only - lean joined bend as a ground-anchored
                # whole-rig tilt (a mid-body rotation pivot moves layers
                # below it the opposite way: a second rotation center)
                dx += p.kick_x
                dy += p.sink - p.breath * 6.0
            elif n.name == "head_pivot":
                local = p.head_angle + p.breath * 0.012
        elif n.spring:
            if dt > 0:
                # Inertial lag driven by the parent chain's angular VELOCITY.
                # Velocity, not angle (angle drive biases cloth against held
                # rotations), and bend deliberately EXCLUDED: a spring lag is
                # an opposite rotation around its own pivot - a second
                # rotation center that reads as cloth counter-rotating during
                # hits. Hit tilts stay single-center rigid.
                k, c = n.spring
                ang_vel = max(-6.0, min(6.0, (rot - n.prev_parent) / dt))
                n.prev_parent = rot
                n.spring_v += (-k * n.spring_a - c * n.spring_v
                               - ang_vel * k * 0.7) * dt
                n.spring_a += n.spring_v * dt
            local = n.spring_a
        acc.append((rot + local, dx, dy))
        if not n.is_layer:
            out.append((0, 0, 0, 0))
            continue
        cx, cy = n.x + n.w * 0.5, n.y + n.h * 0.5
        arm = p.arm_l if n.name == "arm_L" else p.arm_r if n.name == "arm_R" else 0.0
        if arm:
            cx, cy = _rot(n.px, n.py, cx, cy, arm)
        chain = []
        pi = n.parent_idx
        while pi >= 0 and len(chain) < 4:
            chain.append(pi)
            pi = rig.nodes[pi].parent_idx
        chain_rot = 0.0
        for ci in reversed(chain):
            pn = rig.nodes[ci]
            pr = ((p.head_angle + p.breath * 0.012)
                  if pn.name == "head_pivot" else 0.0)
            if pr:
                cx, cy = _rot(pn.px, pn.py, cx, cy, pr)
            chain_rot += pr
        wx, wy = cx + dx, cy + dy
        rot_out = chain_rot + local + arm
        tilt = p.bend + p.lean  # impulse tilt + held pose tilt
        if tilt and rig.canvas[1] > 0:
            # THE single rotation center: all whole-body rotation happens
            # rigidly around the ground anchor; articulation joints (head,
            # arms) are the only sanctioned exceptions
            ax, gy = _bend_anchor(rig), rig.canvas[1]
            wx, wy = _rot(ax, gy, wx, wy, tilt)
            rot_out += tilt
        out.append((wx, wy, rot_out, p.fade))
    return out


def _bend_anchor(rig: Rig) -> float:
    for n in rig.nodes:
        if not n.is_layer and n.name == "body_pivot":
            return n.px
    return rig.canvas[0] * 0.5


class PlayerSim:
    """Scripted stand-in for RigPlayer: idle behaviors plus trigger impulses."""

    def __init__(self, rig: Rig, seed: int = 7):
        self.rig = rig
        self.rng = random.Random(seed)
        self.t = 0.0
        self.blink_in = 1.2
        self.eye_open = 1.0
        self.eye_shut = 0.0
        self.lean = self.lean_target = 0.0
        self.lean_return = 0.0
        self.kick = self.kick_v = 0.0
        self.head_imp = self.head_imp_v = 0.0
        self.sink = self.sink_target = 0.0
        self.rage = self.rage_target = 0.0
        # hit-reaction machine (hitstop / bend / squash / shake / flash)
        self.hitstop = 0.0
        self.bend = self.bend_v = 0.0
        self.squash, self.squash_v = 1.0, 0.0
        self.shake_t, self.shake_amp = 9e9, 0.0
        self.flash = 0.0
        self.sink_pulse = 0.0

    def excite(self, v: float) -> None:
        for n in self.rig.nodes:
            if n.spring:
                n.spring_v += v * (1 if self.rng.random() > 0.5 else -1)

    def trigger(self, motion: str) -> None:
        # Hit reactions: hitstop + squash + graded bend + lockstep shake +
        # flash. The old lean/kick pendulum hinged the body around one pivot
        # (visible upper/lower separation) and swayed like a metronome.
        if motion == "hit":
            self.hitstop = 0.07
            self.bend_v += 0.6  # radians/s of ankle tilt
            self.squash_v -= 1.6
            self.shake_t, self.shake_amp = 0.0, 5.0
            self.flash = 0.10
            self.eye_shut = 0.16
            # no head snap, no spring kicks: a hit moves the rig around
            # exactly one center
        elif motion == "stagger":
            self.hitstop = 0.10
            self.bend_v += 0.85  # radians/s of ankle tilt
            self.squash_v -= 2.6
            self.shake_t, self.shake_amp = 0.0, 9.0
            self.flash = 0.14
            self.eye_shut = 0.3
            self.sink_pulse = 14.0  # knee-buckle, self-decaying
            # no head snap, no spring kicks: single-center only
        elif motion == "enrage":
            self.rage_target = 1.0
            self.shake_t, self.shake_amp = 0.0, 6.5  # uniform tremor
        elif motion == "calm":
            self.rage_target = 0.0

    def step(self, dt: float) -> Params:
        if self.hitstop > 0:  # impact freeze: time stops for a beat
            self.hitstop -= dt
            self.flash = max(0.0, self.flash - dt * 0.6)
            return self._params()
        self.t += dt
        # critically damped: one overshoot max, never a pendulum. These
        # springs are stiff (c*dt must stay < 2), so integrate in substeps -
        # a 15 fps preview would otherwise diverge.
        n = max(1, int(dt * 120) + 1)
        h = dt / n
        for _ in range(n):
            self.bend_v += (-140.0 * self.bend - 23.6 * self.bend_v) * h
            self.bend += self.bend_v * h
            sq = self.squash - 1.0
            self.squash_v += (-260.0 * sq - 32.0 * self.squash_v) * h
            self.squash += self.squash_v * h
        self.squash = min(1.3, max(0.7, self.squash))
        self.shake_t += dt
        self.sink_pulse *= max(0.0, 1.0 - dt * 3.0)
        self.flash = max(0.0, self.flash - dt)
        self.blink_in -= dt
        if self.eye_shut > 0:
            self.eye_shut -= dt
            self.eye_open = max(0.08, self.eye_open - dt * 14)
        elif self.blink_in < 0:
            self.eye_open -= dt * 12
            if self.eye_open <= 0.08:
                self.blink_in = 2.2 + 2.6 * self.rng.random()
                self.eye_open = 0.08
        else:
            self.eye_open = min(1.0, self.eye_open + dt * 8)
        if self.lean_return > 0:
            self.lean_return -= dt
            if self.lean_return <= 0:
                self.lean_target = 0.0
        self.lean += (self.lean_target - self.lean) * min(1.0, dt * 6.0)
        # near-critically damped: one dart out, one settle, never a metronome
        self.kick_v += (-90.0 * self.kick - 19.0 * self.kick_v) * dt
        self.kick += self.kick_v * dt
        self.head_imp_v += (-70.0 * self.head_imp - 7.5 * self.head_imp_v) * dt
        self.head_imp += self.head_imp_v * dt
        self.sink += (self.sink_target - self.sink) * min(1.0, dt * 4.0)
        self.rage += (self.rage_target - self.rage) * min(1.0, dt * 3.0)
        return self._params()

    def _params(self) -> Params:
        return Params(
            breath=0.5 + 0.5 * math.sin(self.t * 6.2832 / 3.8),
            head_angle=0.02 * math.sin(self.t * 6.2832 / 7.3) + self.head_imp * 0.05,
            eye_open=self.eye_open,
            lean=self.lean,
            kick_x=self.kick,
            sink=self.sink + self.sink_pulse,
            rage=self.rage,
            bend=self.bend,
        )

    # whole-rig impact effects, applied uniformly at draw time
    def shake_offset(self) -> float:
        return self.shake_amp * math.exp(-self.shake_t * 7.0) * (
            0.7 * math.sin(self.shake_t * 6.2832 * 28)
            + 0.3 * math.sin(self.shake_t * 6.2832 * 9))

    def squash_y(self) -> float:
        return self.squash

    def squash_x(self) -> float:
        return 1.0 + (1.0 - self.squash) * 0.7

    def flash_amount(self) -> float:
        return min(1.0, self.flash * 7.0)


# scripted timelines: (time, motion) events over a total duration
SCRIPTS: dict[str, tuple[float, list[tuple[float, str]]]] = {
    "idle": (8.0, []),
    "showcase": (10.0, [(3.0, "hit"), (5.0, "stagger"), (7.0, "enrage")]),
}


def render_gif(rig_path: str, out_path: str, script: str = "idle",
               height: int = 540, fps: int = 20,
               background: tuple[int, int, int] = (24, 24, 34)) -> str:
    """Render an animated preview of a rig directory to `out_path` (GIF, or
    lossless WebP when the filename ends in .webp)."""
    rig = parse(rig_path)
    rig_dir = os.path.dirname(rig_path)
    sc = height / rig.canvas[1]
    width = int(rig.canvas[0] * sc)

    layers: dict[int, Image.Image] = {}
    order = sorted((i for i, n in enumerate(rig.nodes) if n.is_layer),
                   key=lambda i: rig.nodes[i].z)
    for i in order:
        n = rig.nodes[i]
        im = Image.open(os.path.join(rig_dir, n.file)).convert("RGBA")
        layers[i] = im.resize((max(1, int(n.w * sc)), max(1, int(n.h * sc))),
                              Image.LANCZOS)

    duration, events = SCRIPTS[script]
    sim = PlayerSim(rig)
    frames: list[Image.Image] = []
    dt = 1.0 / fps
    fired = [False] * len(events)
    for f in range(int(duration * fps)):
        t = f * dt
        for ei, (et, motion) in enumerate(events):
            if not fired[ei] and t >= et:
                sim.trigger(motion)
                fired[ei] = True
        p = sim.step(dt)
        xf = solve(rig, p, dt)
        frame = Image.new("RGBA", (width, height), (*background, 255))
        chr_canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        for i in order:
            n = rig.nodes[i]
            cx, cy, rot, alpha = xf[i]
            im = layers[i]
            if n.name == "sigil":
                if p.rage < 0.01:
                    continue
                alpha *= p.rage
            w, h = im.size
            eye_drop = 0.0
            if n.name in ("eye_L", "eye_R"):
                open_ = max(0.08, p.eye_open)
                eye_drop = h * (1.0 - open_) * 0.5
                im = im.resize((w, max(1, int(h * open_))))
                h = im.size[1]
            if abs(rot) > 1e-4:
                im = im.rotate(-math.degrees(rot), resample=Image.BICUBIC,
                               expand=True)
                w, h = im.size
            if alpha < 0.999:
                im = im.copy()
                im.putalpha(im.getchannel("A").point(lambda a: int(a * alpha)))
            chr_canvas.alpha_composite(
                im, (int(cx * sc - w / 2), int(cy * sc - h / 2 + eye_drop)))
        # whole-rig impact effects: uniform transforms can't tear layers apart
        sq = sim.squash_y()
        if abs(sq - 1.0) > 1e-3:  # squash about the feet
            nw, nh = int(width * sim.squash_x()), int(height * sq)
            chr_canvas = chr_canvas.resize((nw, nh), Image.BILINEAR)
            pad = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            pad.alpha_composite(chr_canvas, ((width - nw) // 2, height - nh))
            chr_canvas = pad
        fl = sim.flash_amount()
        if fl > 0.01:  # impact flash: blend toward white, alpha preserved
            import numpy as _np
            arr = _np.array(chr_canvas)
            rgb = arr[..., :3].astype(_np.float32)
            arr[..., :3] = (rgb + (255 - rgb) * (0.32 * fl)).astype(_np.uint8)
            chr_canvas = Image.fromarray(arr)
        # paste (not alpha_composite): the shake offset can be negative
        frame.paste(chr_canvas, (int(sim.shake_offset() * sc), 0), chr_canvas)
        frames.append(frame.convert("P", palette=Image.ADAPTIVE)
                      if out_path.endswith(".gif") else frame)

    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0,
                   **({"lossless": True} if out_path.endswith(".webp") else {}))
    print(f"{script}: {len(frames)} frames -> {out_path}")
    return out_path


def run(spec: Spec, script: str = "idle", out_name: str | None = None,
        **kw) -> str:
    rig_path = os.path.join(spec.out_dir, "rig.txt")
    out = os.path.join(spec.work_dir, out_name or f"preview_{script}.gif")
    return render_gif(rig_path, out, script=script, **kw)
