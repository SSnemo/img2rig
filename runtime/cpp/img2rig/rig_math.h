#pragma once
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

// img2rig runtime - pure math core: rig.txt parsing, procedural parameters,
// spring physics and world-transform solving. Header-only, no rendering or
// I/O dependencies; unit-testable by direct include. Rendering lives in
// rig_player.h behind a draw callback.
//
// Model: each layer is a rotated quad (rigid transform), no mesh deformation.
// "Aliveness" comes from timing and physical lag (springs on hair/cape/cloth),
// not from vertex warping. Two levels of virtual pivot nodes are recognized by
// name: "body_pivot" (breath bob, hit kick, lean, sink) and "head_pivot"
// (nod/shake, breath nod). Layers named "arm_L"/"arm_R" get an extra rotation
// around their own pivot (shoulder joint) driven by Params::armL/armR.
namespace img2rig {

struct Node {
    std::string name, file, parent;
    float x = 0, y = 0, w = 0, h = 0; // layer bbox in rig-canvas coordinates (0 for virtual nodes)
    float px = 0, py = 0;             // rotation pivot in rig-canvas coordinates
    int parentIdx = -1;
    int z = 0;
    bool isLayer = false;
    bool hasSpring = false;
    float k = 0, c = 0;             // spring stiffness / damping
    float springA = 0, springV = 0; // runtime spring angle / angular velocity
};

struct Doc {
    float canvasW = 0, canvasH = 0;
    std::vector<Node> nodes; // node lines precede layer lines (parents before children)
    bool valid() const { return canvasW > 0 && !nodes.empty(); }
};

// rig.txt grammar (see docs/rig-format.md):
//   canvas <w> <h>
//   node <name> <parent> <z> <pivot_x> <pivot_y>
//   layer <name> <file> <x> <y> <w> <h> <pivot_x> <pivot_y> <parent> <z> [spring <k> <c>]
//   # comment
inline Doc parse(const std::string& text) {
    Doc d;
    std::istringstream in(text);
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ls(line);
        std::string kind;
        ls >> kind;
        if (kind == "canvas") {
            ls >> d.canvasW >> d.canvasH;
        } else if (kind == "node") {
            Node n;
            ls >> n.name >> n.parent >> n.z >> n.px >> n.py;
            d.nodes.push_back(n);
        } else if (kind == "layer") {
            Node n;
            n.isLayer = true;
            ls >> n.name >> n.file >> n.x >> n.y >> n.w >> n.h >> n.px >> n.py >>
                n.parent >> n.z;
            std::string s;
            if (ls >> s && s == "spring") {
                n.hasSpring = true;
                ls >> n.k >> n.c;
            }
            d.nodes.push_back(n);
        }
    }
    // resolve parent indices (root = -1); parents are guaranteed to precede children
    for (auto& n : d.nodes) {
        n.parentIdx = -1;
        if (n.parent == "root") continue;
        for (int i = 0; i < (int)d.nodes.size(); i++)
            if (d.nodes[i].name == n.parent) { n.parentIdx = i; break; }
    }
    return d;
}

// Procedural parameter set - the motion state machine's output, fed to solve()
// every frame. There are no animation files; all motion is curves over these.
struct Params {
    float breath = 0;    // 0..1 breathing phase
    float headAngle = 0; // radians: idle sway + hit shake
    float eyeOpen = 1;   // 1=open 0=shut (blink / hit reaction)
    float lean = 0;      // radians: body tilt (stagger / collapse)
    float kickX = 0;     // hit recoil, rig-canvas px
    float sink = 0;      // vertical sink (collapse / death), rig-canvas px
    float rage = 0;      // 0..1 alpha of the emissive "sigil" layer
    float fade = 1;      // overall brightness (death fade-out)
    // Height-graded horizontal bend (the hit-reaction primitive). Unlike
    // `lean` - a rigid rotation around body_pivot that reads as a hinge -
    // bend shears each layer by its height above the ground (feet planted,
    // head max), so the body flexes like a reed with no separation line.
    float bendX = 0;
    // Shoulder-joint angles (radians). Name-driven: only layers called
    // arm_L / arm_R respond; rigs without arm layers degrade gracefully to
    // torso-only expression.
    float armL = 0;
    float armR = 0;
};

// Spring step: semi-implicit Euler. `drive` couples the parent's angular
// motion in, so a head shake makes the hair lag and whip back.
inline void springStep(float& a, float& v, float k, float c, float drive, float dt) {
    v += (-k * a - c * v + drive) * dt;
    a += v * dt;
}

struct WorldXf {
    float cx = 0, cy = 0; // layer center in rig-canvas coordinates
    float rot = 0;
    float alpha = 1;
};

// Solve world transforms for every node. dt>0 advances springs; dt=0 is a
// pure solve (for test assertions).
// Virtual-node rules (name-driven, identical across all characters):
//   body_pivot: rot=lean, offset=(kickX, sink - breath*bob)
//   head_pivot: rot+=headAngle (+ a small breathing nod)
// Layer nodes inherit the accumulated parent chain; spring layers add their
// own springA. Callers apply eyeOpen (scaleY squash) and per-layer alpha
// (e.g. the emissive layer uses Params::rage) at draw time.
inline void solve(Doc& d, const Params& p, float dt, std::vector<WorldXf>& out) {
    struct Acc { float rot, dx, dy, prevRot; };
    std::vector<Acc> acc(d.nodes.size());
    out.resize(d.nodes.size());
    for (size_t i = 0; i < d.nodes.size(); i++) {
        Node& n = d.nodes[i];
        float rot = 0, dx = 0, dy = 0;
        if (n.parentIdx >= 0) {
            rot = acc[n.parentIdx].rot;
            dx = acc[n.parentIdx].dx;
            dy = acc[n.parentIdx].dy;
        }
        float localRot = 0;
        if (!n.isLayer) {
            if (n.name == "body_pivot") {
                localRot = p.lean;
                dx += p.kickX;
                dy += p.sink - p.breath * 6.0f;
            } else if (n.name == "head_pivot") {
                localRot = p.headAngle + p.breath * 0.012f;
            }
        } else if (n.hasSpring) {
            if (dt > 0) {
                // Parent angular-velocity approximation: drive against the
                // accumulated parent angle so the spring lags then follows.
                springStep(n.springA, n.springV, n.k, n.c, -rot * n.k * 0.6f, dt);
            }
            localRot = n.springA;
        }
        float total = rot + localRot;
        acc[i] = {total, dx, dy, total};

        if (!n.isLayer) continue;
        float cx = n.x + n.w * 0.5f, cy = n.y + n.h * 0.5f;
        // Arm layers rotate around their own pivot (shoulder) first, then
        // follow the parent chain (root-to-leaf order).
        float armRot = (n.name == "arm_L")   ? p.armL
                       : (n.name == "arm_R") ? p.armR
                                             : 0.0f;
        if (armRot != 0.0f) {
            float s = std::sin(armRot), co = std::cos(armRot);
            float ox = cx - n.px, oy = cy - n.py;
            cx = n.px + ox * co - oy * s;
            cy = n.py + ox * s + oy * co;
        }
        // Rotate the layer center around each ancestor pivot, root first.
        // A two-level virtual chain (body/head) is all the model needs.
        int pi = n.parentIdx;
        float rcx = cx, rcy = cy;
        float chainRot = 0;
        int chain[4], cn = 0;
        while (pi >= 0 && cn < 4) {
            chain[cn++] = pi;
            pi = d.nodes[pi].parentIdx;
        }
        for (int ci = cn - 1; ci >= 0; ci--) {
            Node& pn = d.nodes[chain[ci]];
            float pr = (pn.name == "body_pivot")   ? p.lean
                       : (pn.name == "head_pivot") ? (p.headAngle + p.breath * 0.012f)
                                                   : 0.0f;
            if (pr != 0.0f) {
                float s = std::sin(pr), co = std::cos(pr);
                float ox = rcx - pn.px, oy = rcy - pn.py;
                rcx = pn.px + ox * co - oy * s;
                rcy = pn.py + ox * s + oy * co;
            }
            chainRot += pr;
        }
        out[i].cx = rcx + dx;
        out[i].cy = rcy + dy;
        out[i].rot = chainRot + localRot + armRot;
        out[i].alpha = p.fade;
        if (p.bendX != 0.0f && d.canvasH > 0) {
            // profile from the layer's *resting* center: stable under sink/kick
            float rcy0 = n.y + n.h * 0.5f;
            float f = rcy0 < d.canvasH ? (d.canvasH - rcy0) / d.canvasH : 0.0f;
            f = std::pow(f, 1.7f);
            out[i].cx += p.bendX * 90.0f * f;
            out[i].rot += p.bendX * 0.10f * f;
        }
    }
}

} // namespace img2rig
