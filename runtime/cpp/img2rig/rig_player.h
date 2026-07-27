#pragma once
// img2rig runtime - reference player: loading, a procedural motion state
// machine, and drawing through a user-supplied callback. Renderer-agnostic:
// you provide one function that draws a rotated, tinted textured quad and
// returns false if the image is missing. Layers draw back-to-front in the
// main pass; each layer is one quad.
//
// Time: feed update() your *world* dt. If your game freezes time for a
// cinematic, the character freezes with it - springs and blinking included -
// which is the behavior you want.
#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include "rig_math.h"

namespace img2rig {

struct Vec2 { float x = 0, y = 0; };
struct Color { float r = 1, g = 1, b = 1, a = 1; };

// Draw `file` (path relative to the rig directory, as listed in rig.txt)
// centered at `center`, size `size` in screen units, rotated `rot` radians,
// modulated by `tint`. Return false if the image could not be drawn (missing
// asset) - the player uses this for graceful keyframe fallback.
using DrawLayerFn = std::function<bool(const std::string& file, Vec2 center,
                                       Vec2 size, Color tint, float rot)>;

// Game-event vocabulary. Map your own events onto these; every motion is a
// short procedural impulse or a held target, never a canned clip, so motions
// never lock gameplay timing.
enum class Motion {
    Hit,      // took a hit: head shake + recoil + eye shut
    Lunge,    // landed a hit on the opponent: short forward pulse
    Stagger,  // guard broken: big lean + recoil + springs excited
    Collapse, // incapacitated (held state until Recover)
    Recover,  // release Collapse
    Enrage,   // power-up: emissive layer on + full-body shudder
    Heal,     // small upward perk
    Windup,   // generic attack tell: slight rear-back
    Die,      // fall + sink + fade
};

// Attack silhouettes for the strike pose machine. Pick per attack flavor:
// Sweep = lateral swing, Smash = overhead slam (the height drop reads),
// Burst = gathered release (the emissive glow reads).
enum class StrikeStyle { Sweep, Smash, Burst };

class RigPlayer {
public:
    // rigTxt = path to rig.txt; layer files resolve relative to its directory
    // through your draw callback.
    bool load(const std::string& rigTxt) {
        std::ifstream f(rigTxt);
        if (!f) return false;
        std::stringstream ss;
        ss << f.rdbuf();
        Doc d = parse(ss.str());
        if (!d.valid()) return false;
        doc_ = std::move(d);
        auto slash = rigTxt.find_last_of("/\\");
        relDir_ = slash == std::string::npos ? "" : rigTxt.substr(0, slash + 1);
        order_.clear();
        for (int i = 0; i < (int)doc_.nodes.size(); i++)
            if (doc_.nodes[i].isLayer) order_.push_back(i);
        std::sort(order_.begin(), order_.end(),
                  [&](int a, int b) { return doc_.nodes[a].z < doc_.nodes[b].z; });
        resetState();
        return true;
    }
    void unload() { doc_ = {}; }
    bool loaded() const { return doc_.valid(); }

    void trigger(Motion m) {
        using M = Motion;
        switch (m) {
            // Hit reactions use hitstop + squash + height-graded bend +
            // whole-rig decaying shake + flash instead of the old rigid
            // lean/kick pendulum: a single-pivot rotation reads as a hinge
            // (upper and lower body visibly separate) and an underdamped
            // kick sways like a metronome. Bend flexes the silhouette
            // continuously, the shake moves every layer in lockstep (no
            // relative displacement), and the freeze sells the weight.
            case M::Hit:
                hitstop_ = 0.07f;
                bendV_ += 0.6f; // radians/s of ankle tilt
                squashV_ -= 1.6f;
                shakeT_ = 0;
                shakeAmp_ = 5.0f;
                flash_ = 0.10f;
                eyeShut_ = 0.16f;
                // no head snap and no spring kicks: a hit moves the rig
                // around exactly one center (hair whipped around its own
                // root otherwise)
                break;
            case M::Lunge:
                kickV_ -= 260;
                leanTo(-0.06f, 0.35f);
                break;
            case M::Stagger:
                hitstop_ = 0.10f;
                bendV_ += 0.85f; // radians/s of ankle tilt
                squashV_ -= 2.6f;
                shakeT_ = 0;
                shakeAmp_ = 9.0f;
                flash_ = 0.14f;
                eyeShut_ = 0.3f;
                sinkPulse_ = 14.0f; // brief knee-buckle, decays on its own
                // no head snap, no spring kicks: single-center only
                break;
            case M::Collapse:
                sinkTarget_ = 26;
                leanTo(0.13f, 0);
                break;
            case M::Recover:
                sinkTarget_ = 0;
                leanTo(0, 0);
                break;
            case M::Enrage:
                rageTarget_ = 1;
                shakeT_ = 0;      // power-surge tremor as a uniform shake -
                shakeAmp_ = 6.5f; // random spring kicks read as separation
                break;
            case M::Heal: headImpV_ -= 2.5f; break;
            case M::Windup:
                leanTo(-0.045f, 0.4f);
                break;
            case M::Die:
                fadeTarget_ = 0.22f;
                sinkTarget_ = 44;
                leanTo(0.38f, 0);
                eyeShut_ = 9e9f;
                break;
        }
    }

    // ---- strike pose machine: windup hold (with tremble) -> release impulse
    // -> settle. Trigger the release slightly *before* your impact resolves so
    // the swing frame lands exactly on the hit.
    //
    // Arm angles are clamped small (~0.16 rad) on purpose: layers cut from a
    // single illustration have nothing painted underneath the arms, so large
    // rotations expose holes and tear the costume. Pose readability comes from
    // lean/sink/kick/springs/glow instead - or from full-frame keyframes (below).
    void strikeWindup(StrikeStyle s, bool vertical, bool fury) {
        pose_ = Pose::Windup;
        poseStyle_ = s;
        poseFury_ = fury;
        poseT_ = trembleT_ = 0;
        armRate_ = 6.5f;
        float amp = fury ? 1.25f : 1.0f;
        switch (s) {
            case StrikeStyle::Sweep: // side lean, arm pulled back (higher if vertical)
                armRT_ = (vertical ? 0.16f : 0.12f) * amp;
                armLT_ = -0.04f * amp;
                leanTo(0.07f * amp, 0);
                sinkTarget_ = -6;
                break;
            case StrikeStyle::Smash: // arms up + body rising rear-back (the rise reads)
                armLT_ = -0.12f * amp;
                armRT_ = 0.12f * amp;
                leanTo(0.08f * amp, 0);
                sinkTarget_ = -13;
                break;
            case StrikeStyle::Burst: // rear-back gather, glow ramps (the glow reads)
                armLT_ = -0.06f * amp;
                armRT_ = 0.06f * amp;
                leanTo(0.13f * amp, 0);
                sinkTarget_ = -4;
                chargeT_ = fury ? 1.0f : 0.85f;
                chargeRate_ = 1.8f;
                break;
        }
        if (fury) chargeT_ = std::max(chargeT_, 0.55f);
    }
    void strikeRelease() {
        if (pose_ == Pose::None) { // no windup happened; still give the impulse
            poseStyle_ = StrikeStyle::Sweep;
            poseFury_ = false;
        }
        pose_ = Pose::Release;
        poseT_ = 0;
        armRate_ = 17.0f;
        float amp = poseFury_ ? 1.2f : 1.0f;
        switch (poseStyle_) {
            case StrikeStyle::Sweep: // whip through: forward dart + shake
                armRT_ = -0.14f * amp;
                armLT_ = 0.04f;
                leanTo(-0.13f * amp, 0.4f);
                kickV_ -= 380 * amp;
                shakeT_ = 0;
                shakeAmp_ = 5.0f * amp;
                break;
            case StrikeStyle::Smash: // slam: the downward spike reads
                armLT_ = 0.08f * amp;
                armRT_ = -0.08f * amp;
                leanTo(-0.10f * amp, 0.4f);
                slam_ = 26.0f * amp;
                shakeT_ = 0;
                shakeAmp_ = 6.0f * amp;
                break;
            case StrikeStyle::Burst: // forward pulse + glow flash
                armLT_ = 0.03f;
                armRT_ = -0.03f;
                leanTo(-0.10f * amp, 0.4f);
                kickV_ -= 320 * amp;
                charge_ = std::min(1.0f, charge_ + 0.5f);
                shakeT_ = 0;
                shakeAmp_ = 6.0f * amp;
                break;
        }
        sinkTarget_ = 0;
        chargeT_ = 0;
        chargeRate_ = 5.0f;
    }
    void strikeCancel() { // interrupted or finished: settle back
        if (pose_ == Pose::None) return;
        pose_ = Pose::None;
        armLT_ = armRT_ = 0;
        armRate_ = 5.0f;
        chargeT_ = 0;
        chargeRate_ = 4.0f;
        sinkTarget_ = 0;
    }
    bool strikePosed() const { return pose_ != Pose::None; }

    void update(float dt) {
        if (!loaded() || dt <= 0) return;
        if (hitstop_ > 0) { // impact freeze: time itself stops for a beat
            hitstop_ -= dt;
            flash_ = std::max(0.0f, flash_ - dt * 0.6f);
            return;
        }
        t_ += dt;
        // critically damped: one overshoot max, never a pendulum. Stiff
        // springs (c*dt must stay < 2): integrate in substeps so low frame
        // rates don't diverge.
        {
            int n = std::max(1, (int)(dt * 120.0f) + 1);
            float h = dt / n;
            for (int s = 0; s < n; s++) {
                bendV_ += (-140.0f * bend_ - 23.6f * bendV_) * h;
                bend_ += bendV_ * h;
                float sq = squash_ - 1.0f;
                squashV_ += (-260.0f * sq - 32.0f * squashV_) * h;
                squash_ += squashV_ * h;
            }
            squash_ = std::min(1.3f, std::max(0.7f, squash_));
        }
        shakeT_ += dt;
        sinkPulse_ *= std::max(0.0f, 1.0f - dt * 3.0f);
        flash_ = std::max(0.0f, flash_ - dt);
        if (pose_ == Pose::Windup) {
            poseT_ += dt;
            trembleT_ -= dt; // sustained tremble (denser and stronger when furious)
            if (trembleT_ <= 0) {
                trembleT_ = poseFury_ ? 0.26f : 0.55f;
                shakeT_ = 0; // tremble = uniform shake pulses, never spring kicks
                shakeAmp_ = poseFury_ ? 2.4f : 1.2f;
            }
        } else if (pose_ == Pose::Release) {
            poseT_ += dt;
            if (poseT_ > 0.45f) strikeCancel();
        }
        armL_ += (armLT_ - armL_) * std::min(1.0f, dt * armRate_);
        armR_ += (armRT_ - armR_) * std::min(1.0f, dt * armRate_);
        charge_ += (chargeT_ - charge_) * std::min(1.0f, dt * chargeRate_);
        slam_ = std::max(0.0f, slam_ - dt * 130.0f); // slam spike decays fast
        // Blinking: random interval; hits and death force the eyes shut.
        blinkIn_ -= dt;
        if (eyeShut_ > 0) {
            eyeShut_ -= dt;
            eyeOpen_ = std::max(0.08f, eyeOpen_ - dt * 14);
        } else if (blinkIn_ < 0) {
            eyeOpen_ -= dt * 12;
            if (eyeOpen_ <= 0.08f) {
                blinkIn_ = 2.2f + 2.6f * frand();
                eyeOpen_ = 0.08f;
            }
        } else {
            eyeOpen_ = std::min(1.0f, eyeOpen_ + dt * 8);
        }
        // Lean eases toward its target, with an optional timed return.
        if (leanReturn_ > 0) {
            leanReturn_ -= dt;
            if (leanReturn_ <= 0) leanTarget_ = 0;
        }
        lean_ += (leanTarget_ - lean_) * std::min(1.0f, dt * 6.0f);
        // near-critically damped (c ~ 2*sqrt(k)): one dart out, one settle
        // back, never a metronome
        springStep(kick_, kickV_, 90.0f, 19.0f, 0, dt);
        springStep(headImp_, headImpV_, 70.0f, 7.5f, 0, dt); // head-shake impulse
        sink_ += (sinkTarget_ - sink_) * std::min(1.0f, dt * 4.0f);
        rage_ += (rageTarget_ - rage_) * std::min(1.0f, dt * 3.0f);
        fade_ += (fadeTarget_ - fade_) * std::min(1.0f, dt * 2.5f);

        Params p;
        p.breath = 0.5f + 0.5f * std::sin(t_ * 6.2832f / 3.8f);
        p.headAngle = 0.02f * std::sin(t_ * 6.2832f / 7.3f) + headImp_ * 0.05f;
        p.eyeOpen = eyeOpen_;
        p.lean = lean_;
        p.kickX = kick_;
        p.sink = sink_ + slam_ + sinkPulse_;
        p.rage = rage_;
        p.fade = fade_;
        p.armL = armL_;
        p.armR = armR_;
        p.bendX = bend_;
        params_ = p;
        solve(doc_, p, dt, xf_);
    }

    // Whole-rig impact effects, applied uniformly at draw time (uniform =
    // zero relative layer displacement, so nothing can tear):
    // shake: decaying dual-frequency offset in canvas px.
    float shakeOffset() const {
        return shakeAmp_ * std::exp(-shakeT_ * 7.0f) *
               (0.7f * std::sin(shakeT_ * 6.2832f * 28.0f) +
                0.3f * std::sin(shakeT_ * 6.2832f * 9.0f));
    }
    // squash: vertical scale about the character's feet (pair with a slight
    // horizontal widen). 1 = none.
    float squashY() const { return squash_; }
    float squashX() const { return 1.0f + (1.0f - squash_) * 0.7f; }
    // flash: 0..1 brighten-toward-white amount for the impact frames.
    float flashAmount() const { return std::min(1.0f, flash_ * 7.0f); }

    // ---- full-frame pose keyframes ----
    // A rotate-the-cutouts rig is capped by what the source illustration
    // contains: it cannot show a pose the painting never had (e.g. arms
    // raised overhead when the original arms hang at the sides). Big pose
    // reads use whole-frame swaps instead: windup shows kf_windup.png (held,
    // trembling), release flashes kf_swing.png for an instant, then layered
    // drawing resumes for the settle (springs still ringing). The hard cut is
    // masked by your impact flash/bloom - standard fighting-game practice.
    // Keyframes share the rig canvas and framing, so they align pixel-perfect.
    // If the files are missing, the callback returns false and the player
    // silently falls back to layered posing - deploy per character at leisure.
    bool drawStrikeKeyframe(const DrawLayerFn& drawFn, Vec2 center, float height) {
        if (pose_ == Pose::None) return false;
        if (pose_ == Pose::Release && poseT_ > kSwingShowSec)
            return false; // the swing frame only flashes; settle uses layers
        const char* f = pose_ == Pose::Windup ? "kf_windup.png" : "kf_swing.png";
        float sc = height / doc_.canvasH;
        Vec2 size{doc_.canvasW * sc, doc_.canvasH * sc};
        Vec2 at = center;
        float breath = 0.5f + 0.5f * std::sin(t_ * 6.2832f / 3.8f);
        at.x += kick_ * sc;
        at.y += (sink_ + slam_ - breath * 6.0f) * sc;
        float rot = lean_;
        if (pose_ == Pose::Windup) {
            float j = poseFury_ ? 2.4f : 1.1f;
            at.x += (frand() - 0.5f) * j;
            at.y += (frand() - 0.5f) * j;
            rot += (frand() - 0.5f) * 0.005f * (poseFury_ ? 1.6f : 1.0f);
        }
        float b = fade_;
        return drawFn(relDir_ + f, at, size, {b, b, b, 1}, rot);
    }
    // Keep slightly longer than the lead with which you call strikeRelease()
    // before impact, so a hitstop lands while the swing frame is showing.
    static constexpr float kSwingShowSec = 0.22f;

    // center = character anchor on screen, height = display height (uniform scale)
    void draw(const DrawLayerFn& drawFn, Vec2 center, float height) {
        if (!loaded() || xf_.empty()) return;
        if (drawStrikeKeyframe(drawFn, center, height)) return;
        float sc = height / doc_.canvasH;
        float shakeX = shakeOffset() * sc;
        float sqY = squashY(), sqX = squashX();
        float bottomY = center.y + height * 0.5f;
        float lift = 1.0f + 0.9f * flashAmount(); // impact flash: brightness lift
        for (int i : order_) {
            auto& n = doc_.nodes[i];
            auto& w = xf_[i];
            float al = w.alpha;
            Vec2 size{n.w * sc, n.h * sc};
            if (n.name == "sigil") {
                // Emissive layer: charge glow and rage stack (Burst windup
                // ramps it; fury keeps it pulsing).
                float glow = std::min(
                    1.0f, rage_ + charge_ * (0.75f + 0.25f * std::sin(t_ * 22.0f)));
                al *= glow;
                if (al < 0.01f) continue;
            }
            float eyeDrop = 0.0f; // blink anchors the bottom edge: the upper lid
            if (n.name == "eye_L" || n.name == "eye_R") { // sweeps down (center-scaling
                float open = std::max(0.08f, eyeOpen_);   //  reads as closing upward)
                eyeDrop = n.h * sc * (1.0f - open) * 0.5f;
                size.y *= open;
            }
            Vec2 at{center.x + (w.cx - doc_.canvasW * 0.5f) * sc,
                    center.y + (w.cy - doc_.canvasH * 0.5f) * sc + eyeDrop};
            if (sqY != 1.0f) { // squash about the feet, uniform across layers
                at.y = bottomY - (bottomY - at.y) * sqY;
                at.x = center.x + (at.x - center.x) * sqX;
                size.x *= sqX;
                size.y *= sqY;
            }
            at.x += shakeX;
            float b = fade_ * lift;
            drawFn(relDir_ + n.file, at, size, {b, b, b, al}, w.rot);
        }
    }

    // Direct parameter access for debug consoles / tooling.
    void setParam(const std::string& key, float v) {
        if (key == "lean") leanTarget_ = v;
        else if (key == "sink") sinkTarget_ = v;
        else if (key == "rage") rageTarget_ = v;
        else if (key == "fade") fadeTarget_ = v;
        else if (key == "kick") kickV_ += v;
        else if (key == "blink") eyeShut_ = v; // force eyes shut for v seconds
        else if (key == "arml") armLT_ = v;
        else if (key == "armr") armRT_ = v;
        else if (key == "charge") chargeT_ = v;
    }

    const Doc& doc() const { return doc_; }
    const Params& params() const { return params_; }

private:
    void resetState() {
        t_ = 0; blinkIn_ = 2.5f; eyeOpen_ = 1; eyeShut_ = 0;
        lean_ = leanTarget_ = 0; leanReturn_ = 0;
        kick_ = kickV_ = 0; headImp_ = headImpV_ = 0;
        sink_ = sinkTarget_ = 0; rage_ = rageTarget_ = 0; fade_ = fadeTarget_ = 1;
        pose_ = Pose::None; poseT_ = trembleT_ = 0;
        armL_ = armLT_ = armR_ = armRT_ = 0; armRate_ = 5.0f;
        charge_ = chargeT_ = 0; chargeRate_ = 4.0f; slam_ = 0;
        hitstop_ = 0; bend_ = bendV_ = 0; squash_ = 1; squashV_ = 0;
        shakeT_ = 9e9f; shakeAmp_ = 0; flash_ = 0; sinkPulse_ = 0;
    }
    void leanTo(float target, float returnAfter) {
        leanTarget_ = target;
        leanReturn_ = returnAfter;
    }
    void exciteSprings(float v) {
        for (auto& n : doc_.nodes)
            if (n.hasSpring) n.springV += v * (frand() > 0.5f ? 1.0f : -1.0f);
    }
    float frand() { // deterministic LCG: no <random>, reproducible replays
        rnd_ = rnd_ * 1664525u + 1013904223u;
        return (float)((rnd_ >> 8) & 0xFFFF) / 65535.0f;
    }

    Doc doc_;
    Params params_;
    std::vector<WorldXf> xf_;
    std::vector<int> order_;
    std::string relDir_;
    float t_ = 0, blinkIn_ = 2.5f, eyeOpen_ = 1, eyeShut_ = 0;
    float lean_ = 0, leanTarget_ = 0, leanReturn_ = 0;
    float kick_ = 0, kickV_ = 0, headImp_ = 0, headImpV_ = 0;
    float sink_ = 0, sinkTarget_ = 0, rage_ = 0, rageTarget_ = 0;
    float fade_ = 1, fadeTarget_ = 1;
    enum class Pose { None, Windup, Release };
    Pose pose_ = Pose::None;
    StrikeStyle poseStyle_ = StrikeStyle::Sweep;
    bool poseFury_ = false;
    float poseT_ = 0, trembleT_ = 0;
    float armL_ = 0, armLT_ = 0, armR_ = 0, armRT_ = 0, armRate_ = 5.0f;
    float charge_ = 0, chargeT_ = 0, chargeRate_ = 4.0f;
    float slam_ = 0; // slam spike stacked on sink, decays fast
    // hit-reaction machine (hitstop / bend / squash / shake / flash)
    float hitstop_ = 0;
    float bend_ = 0, bendV_ = 0;
    float squash_ = 1, squashV_ = 0;
    float shakeT_ = 9e9f, shakeAmp_ = 0;
    float flash_ = 0, sinkPulse_ = 0;
    unsigned rnd_ = 12345;
};

} // namespace img2rig
