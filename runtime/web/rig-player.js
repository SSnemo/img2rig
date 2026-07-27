/**
 * img2rig web reference player - renders a rig.txt layer pack on a 2D canvas.
 *
 * A faithful JS port of the runtime solver (runtime/cpp/img2rig/rig_math.h)
 * and the procedural motion machine: breath, blink, head sway, spring
 * physics, hit/stagger/enrage/collapse/die impulses. No dependencies, no
 * build step - plain ES module.
 *
 * If you change the solver here, change rig_math.h and preview.py too; the
 * C++ unit tests pin the behavior.
 *
 * Usage:
 *   import { RigPlayer } from "./rig-player.js";
 *   const rig = await RigPlayer.load("path/to/rig.txt");
 *   function frame(t) { rig.update(dt); rig.draw(ctx, x, y, height); }
 *   rig.trigger("stagger");
 */

// ---- rig.txt parsing (mirrors img2rig::parse) ----

function parseRig(text) {
  const nodes = [];
  let canvas = [0, 0];
  for (const line of text.split(/\r?\n/)) {
    const t = line.trim().split(/\s+/);
    if (!t[0] || t[0].startsWith("#")) continue;
    if (t[0] === "canvas") {
      canvas = [parseFloat(t[1]), parseFloat(t[2])];
    } else if (t[0] === "node") {
      nodes.push({ name: t[1], parent: t[2], z: parseInt(t[3]),
                   px: parseFloat(t[4]), py: parseFloat(t[5]),
                   isLayer: false, spring: null });
    } else if (t[0] === "layer") {
      const n = { name: t[1], file: t[2],
                  x: parseFloat(t[3]), y: parseFloat(t[4]),
                  w: parseFloat(t[5]), h: parseFloat(t[6]),
                  px: parseFloat(t[7]), py: parseFloat(t[8]),
                  parent: t[9], z: parseInt(t[10]),
                  isLayer: true, spring: null,
                  springA: 0, springV: 0, prevParent: 0 };
      if (t[11] === "spring") n.spring = [parseFloat(t[12]), parseFloat(t[13])];
      nodes.push(n);
    }
  }
  const idx = new Map(nodes.map((n, i) => [n.name, i]));
  for (const n of nodes) n.parentIdx = idx.has(n.parent) ? idx.get(n.parent) : -1;
  return { canvas, nodes };
}

function rotAround(px, py, cx, cy, a) {
  const s = Math.sin(a), c = Math.cos(a);
  const ox = cx - px, oy = cy - py;
  return [px + ox * c - oy * s, py + ox * s + oy * c];
}

// ---- solver (mirrors img2rig::solve) ----

function solve(rig, p, dt) {
  const acc = [], out = [];
  for (const n of rig.nodes) {
    let [rot, dx, dy] = n.parentIdx >= 0 ? acc[n.parentIdx] : [0, 0, 0];
    let local = 0;
    if (!n.isLayer) {
      if (n.name === "body_pivot") {
        local = p.lean; dx += p.kickX; dy += p.sink - p.breath * 6.0;
      } else if (n.name === "head_pivot") {
        local = p.headAngle + p.breath * 0.012;
      }
    } else if (n.spring) {
      if (dt > 0) {
        // Inertial lag driven by the parent's angular VELOCITY (bend
        // included): zero at steady state, so cloth lags while the parent
        // moves and then always realigns exactly. An angle-proportional
        // drive biases the equilibrium against any held tilt - cloth
        // visibly rotating the wrong way.
        const [k, c] = n.spring;
        const rotP = rot + (p.bend || 0);
        const angVel = Math.max(-6, Math.min(6, (rotP - n.prevParent) / dt));
        n.prevParent = rotP;
        n.springV += (-k * n.springA - c * n.springV - angVel * k * 0.7) * dt;
        n.springA += n.springV * dt;
      }
      local = n.springA;
    }
    acc.push([rot + local, dx, dy]);
    if (!n.isLayer) { out.push([0, 0, 0, 0]); continue; }

    let cx = n.x + n.w * 0.5, cy = n.y + n.h * 0.5;
    const arm = n.name === "arm_L" ? p.armL : n.name === "arm_R" ? p.armR : 0;
    if (arm) [cx, cy] = rotAround(n.px, n.py, cx, cy, arm);
    const chain = [];
    for (let pi = n.parentIdx; pi >= 0 && chain.length < 4;
         pi = rig.nodes[pi].parentIdx) chain.push(pi);
    let chainRot = 0;
    for (let ci = chain.length - 1; ci >= 0; ci--) {
      const pn = rig.nodes[chain[ci]];
      const pr = pn.name === "body_pivot" ? p.lean
               : pn.name === "head_pivot" ? p.headAngle + p.breath * 0.012 : 0;
      if (pr) [cx, cy] = rotAround(pn.px, pn.py, cx, cy, pr);
      chainRot += pr;
    }
    let wx = cx + dx, wy = cy + dy, rotOut = chainRot + local + arm;
    if (p.bend && rig.canvas[1] > 0) {
      // rigid tilt about the ground anchor: identical transform for every
      // layer, so nothing can separate
      const bp = rig.nodes.find(m => !m.isLayer && m.name === "body_pivot");
      const ax = bp ? bp.px : rig.canvas[0] * 0.5;
      [wx, wy] = rotAround(ax, rig.canvas[1], wx, wy, p.bend);
      rotOut += p.bend;
    }
    out.push([wx, wy, rotOut, p.fade]);
  }
  return out;
}

// ---- player ----

export class RigPlayer {
  /** Load rig.txt and all layer images. baseUrl defaults to rig.txt's dir. */
  static async load(rigUrl) {
    const text = await (await fetch(rigUrl)).text();
    const rig = parseRig(text);
    const base = rigUrl.slice(0, rigUrl.lastIndexOf("/") + 1);
    const player = new RigPlayer(rig);
    await Promise.all(rig.nodes.filter(n => n.isLayer).map(n =>
      new Promise((res, rej) => {
        const im = new Image();
        im.onload = () => { player.images.set(n.name, im); res(); };
        im.onerror = () => rej(new Error("failed to load " + n.file));
        im.src = base + n.file;
      })));
    return player;
  }

  constructor(rig) {
    this.rig = rig;
    this.images = new Map();
    this.order = rig.nodes.map((n, i) => i).filter(i => rig.nodes[i].isLayer)
      .sort((a, b) => rig.nodes[a].z - rig.nodes[b].z);
    this.reset();
  }

  reset() {
    this.t = 0; this.blinkIn = 1.2; this.eyeOpen = 1; this.eyeShut = 0;
    this.lean = 0; this.leanTarget = 0; this.leanReturn = 0;
    this.kick = 0; this.kickV = 0; this.headImp = 0; this.headImpV = 0;
    this.sink = 0; this.sinkTarget = 0; this.rage = 0; this.rageTarget = 0;
    this.fade = 1; this.fadeTarget = 1;
    this.armL = 0; this.armLT = 0; this.armR = 0; this.armRT = 0;
    this.armRate = 5;
    // hit-reaction machine (hitstop / bend / squash / shake / flash)
    this.hitstop = 0; this.bend = 0; this.bendV = 0;
    this.squash = 1; this.squashV = 0;
    this.shakeT = 9e9; this.shakeAmp = 0;
    this.flash = 0; this.headLag = 0; this.sinkPulse = 0;
    for (const n of this.rig.nodes) if (n.spring) { n.springA = 0; n.springV = 0; }
    this.params = null;
    this.xf = null;
  }

  excite(v) {
    for (const n of this.rig.nodes)
      if (n.spring) n.springV += v * (Math.random() > 0.5 ? 1 : -1);
  }

  /** hit | lunge | stagger | collapse | recover | enrage | calm | heal | windup | die */
  trigger(motion) {
    switch (motion) {
      // hit reactions: hitstop + squash + graded bend + lockstep shake +
      // flash (the old lean/kick pendulum hinged and swayed)
      case "hit": // bendV in radians/s of ankle tilt; cloth follows the bend
        this.hitstop = 0.07; this.bendV += 0.6; this.squashV -= 1.6;
        this.shakeT = 0; this.shakeAmp = 5; this.flash = 0.10;
        this.eyeShut = 0.16; this.headLag = 0.05; break;
      case "lunge":
        this.kickV -= 260; this.leanTo(-0.06, 0.35); break;
      case "stagger":
        this.hitstop = 0.10; this.bendV += 0.85; this.squashV -= 2.6;
        this.shakeT = 0; this.shakeAmp = 9; this.flash = 0.14;
        this.eyeShut = 0.3; this.headLag = 0.06; this.sinkPulse = 14; break;
      case "collapse":
        this.sinkTarget = 26; this.leanTo(0.13, 0); break;
      case "recover":
        this.sinkTarget = 0; this.leanTo(0, 0); this.fadeTarget = 1;
        this.eyeShut = 0; break;
      case "enrage":
        this.rageTarget = 1; this.headImpV += 6; this.excite(4.0); break;
      case "calm": this.rageTarget = 0; break;
      case "heal": this.headImpV -= 2.5; break;
      case "windup": this.leanTo(-0.045, 0.4); break;
      case "die":
        this.fadeTarget = 0.22; this.sinkTarget = 44; this.leanTo(0.38, 0);
        this.eyeShut = 9e9; break;
    }
  }

  leanTo(target, ret) { this.leanTarget = target; this.leanReturn = ret; }

  update(dt) {
    if (dt <= 0) return;
    dt = Math.min(dt, 0.05); // tab-switch guard: clamp runaway frame gaps
    if (this.hitstop > 0) { // impact freeze: time stops for a beat
      this.hitstop -= dt;
      this.flash = Math.max(0, this.flash - dt * 0.6);
      return;
    }
    this.t += dt;
    if (this.headLag > 0) { // delayed head snap (overlap/follow-through)
      this.headLag -= dt;
      if (this.headLag <= 0) this.headImpV += 5.5;
    }
    // critically damped: one overshoot max, never a pendulum. Stiff springs
    // (c*dt must stay < 2): substep so low frame rates don't diverge.
    const nSub = Math.max(1, Math.floor(dt * 120) + 1), hSub = dt / nSub;
    for (let s = 0; s < nSub; s++) {
      this.bendV += (-140 * this.bend - 23.6 * this.bendV) * hSub;
      this.bend += this.bendV * hSub;
      const sq = this.squash - 1;
      this.squashV += (-260 * sq - 32 * this.squashV) * hSub;
      this.squash += this.squashV * hSub;
    }
    this.squash = Math.min(1.3, Math.max(0.7, this.squash));
    this.shakeT += dt;
    this.sinkPulse *= Math.max(0, 1 - dt * 3);
    this.flash = Math.max(0, this.flash - dt);
    this.blinkIn -= dt;
    if (this.eyeShut > 0) {
      this.eyeShut -= dt;
      this.eyeOpen = Math.max(0.08, this.eyeOpen - dt * 14);
    } else if (this.blinkIn < 0) {
      this.eyeOpen -= dt * 12;
      if (this.eyeOpen <= 0.08) {
        this.blinkIn = 2.2 + 2.6 * Math.random();
        this.eyeOpen = 0.08;
      }
    } else {
      this.eyeOpen = Math.min(1, this.eyeOpen + dt * 8);
    }
    if (this.leanReturn > 0) {
      this.leanReturn -= dt;
      if (this.leanReturn <= 0) this.leanTarget = 0;
    }
    this.lean += (this.leanTarget - this.lean) * Math.min(1, dt * 6);
    this.kickV += (-90 * this.kick - 9 * this.kickV) * dt;
    this.kick += this.kickV * dt;
    this.headImpV += (-70 * this.headImp - 7.5 * this.headImpV) * dt;
    this.headImp += this.headImpV * dt;
    this.sink += (this.sinkTarget - this.sink) * Math.min(1, dt * 4);
    this.rage += (this.rageTarget - this.rage) * Math.min(1, dt * 3);
    this.fade += (this.fadeTarget - this.fade) * Math.min(1, dt * 2.5);
    this.armL += (this.armLT - this.armL) * Math.min(1, dt * this.armRate);
    this.armR += (this.armRT - this.armR) * Math.min(1, dt * this.armRate);

    this.params = {
      breath: 0.5 + 0.5 * Math.sin(this.t * 6.2832 / 3.8),
      headAngle: 0.02 * Math.sin(this.t * 6.2832 / 7.3) + this.headImp * 0.05,
      eyeOpen: this.eyeOpen, lean: this.lean, kickX: this.kick,
      sink: this.sink + this.sinkPulse, rage: this.rage, fade: this.fade,
      armL: this.armL, armR: this.armR, bend: this.bend,
    };
    this.xf = solve(this.rig, this.params, dt);
  }

  // whole-rig impact effects, applied uniformly at draw time
  shakeOffset() {
    return this.shakeAmp * Math.exp(-this.shakeT * 7) *
      (0.7 * Math.sin(this.shakeT * 6.2832 * 28) +
       0.3 * Math.sin(this.shakeT * 6.2832 * 9));
  }
  flashAmount() { return Math.min(1, this.flash * 7); }

  /** Draw at (x, y) = character center, scaled to `height` display pixels. */
  draw(ctx, x, y, height) {
    if (!this.xf) return;
    const [cw, ch] = this.rig.canvas;
    const sc = height / ch;
    const p = this.params;
    // whole-rig squash (about the feet) + shake: one uniform ctx transform,
    // so no layer can displace relative to another
    ctx.save();
    const sqY = this.squash, sqX = 1 + (1 - this.squash) * 0.7;
    if (Math.abs(sqY - 1) > 1e-3) {
      const bottom = y + height / 2;
      ctx.translate(x, bottom);
      ctx.scale(sqX, sqY);
      ctx.translate(-x, -bottom);
    }
    ctx.translate(this.shakeOffset() * sc, 0);
    const flash = this.flashAmount();
    for (const i of this.order) {
      const n = this.rig.nodes[i];
      const [wx, wy, rot, alphaBase] = this.xf[i];
      const img = this.images.get(n.name);
      if (!img) continue;
      let alpha = alphaBase;
      if (n.name === "sigil") {
        alpha *= p.rage;
        if (alpha < 0.01) continue;
      }
      let w = n.w * sc, h = n.h * sc, eyeDrop = 0;
      if (n.name === "eye_L" || n.name === "eye_R") {
        const open = Math.max(0.08, p.eyeOpen);
        eyeDrop = h * (1 - open) * 0.5; // blink anchors the bottom edge
        h *= open;
      }
      ctx.save();
      ctx.globalAlpha = alpha;
      const bright = p.fade * (1 + 0.9 * flash); // impact flash: brightness lift
      if (Math.abs(bright - 1) > 1e-3) ctx.filter = `brightness(${bright})`;
      ctx.translate(x + (wx - cw * 0.5) * sc, y + (wy - ch * 0.5) * sc + eyeDrop);
      if (rot) ctx.rotate(rot);
      ctx.drawImage(img, -w / 2, -h / 2, w, h);
      ctx.restore();
    }
    ctx.restore();
  }
}
