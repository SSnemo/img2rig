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
                  isLayer: true, spring: null, springA: 0, springV: 0 };
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
        const [k, c] = n.spring;
        n.springV += (-k * n.springA - c * n.springV - rot * k * 0.6) * dt;
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
    out.push([cx + dx, cy + dy, chainRot + local + arm, p.fade]);
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
      case "hit":
        this.kickV += 340; this.headImpV += 5; this.eyeShut = 0.18;
        this.excite(2.2); break;
      case "lunge":
        this.kickV -= 260; this.leanTo(-0.06, 0.35); break;
      case "stagger":
        this.leanTo(0.24, 1.0); this.kickV += 260; this.headImpV += 6;
        this.eyeShut = 0.3; this.excite(3.5); break;
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
    this.t += dt;
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
      sink: this.sink, rage: this.rage, fade: this.fade,
      armL: this.armL, armR: this.armR,
    };
    this.xf = solve(this.rig, this.params, dt);
  }

  /** Draw at (x, y) = character center, scaled to `height` display pixels. */
  draw(ctx, x, y, height) {
    if (!this.xf) return;
    const [cw, ch] = this.rig.canvas;
    const sc = height / ch;
    const p = this.params;
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
      if (p.fade < 0.999) ctx.filter = `brightness(${p.fade})`;
      ctx.translate(x + (wx - cw * 0.5) * sc, y + (wy - ch * 0.5) * sc + eyeDrop);
      if (rot) ctx.rotate(rot);
      ctx.drawImage(img, -w / 2, -h / 2, w, h);
      ctx.restore();
    }
  }
}
