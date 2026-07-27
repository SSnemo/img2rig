// Unit tests for the img2rig runtime. Plain asserts, no framework.
// Build: see runtime/cpp/CMakeLists.txt (or any C++17 compiler:
//   c++ -std=c++17 -I.. tests/rig_test.cpp -o rig_test && ./rig_test)
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>

#include "../img2rig/rig_math.h"
#include "../img2rig/rig_player.h"

using namespace img2rig;

static int checks = 0;
#define CHECK(cond)                                              \
    do {                                                         \
        if (!(cond)) {                                           \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            std::exit(1);                                        \
        }                                                        \
        checks++;                                                \
    } while (0)

static const char* kRig =
    "# comment line\n"
    "canvas 1000 1500\n"
    "node body_pivot root -1 500 900\n"
    "node head_pivot body_pivot -1 520 300\n"
    "layer cape cape.png 100 400 300 800 500 400 body_pivot 0 spring 30 5\n"
    "layer body body.png 350 400 300 900 500 900 body_pivot 1\n"
    "layer arm_L arm_l.png 250 500 150 400 320 520 body_pivot 2\n"
    "layer head head.png 400 100 240 300 520 300 head_pivot 3\n"
    "layer eye_L eye_l.png 460 200 40 25 480 212 head_pivot 4\n";

int main() {
    // ---- parsing ----
    Doc d = parse(kRig);
    CHECK(d.valid());
    CHECK(d.canvasW == 1000 && d.canvasH == 1500);
    CHECK(d.nodes.size() == 7);
    CHECK(!d.nodes[0].isLayer && d.nodes[0].name == "body_pivot");
    CHECK(d.nodes[0].parentIdx == -1);          // root
    CHECK(d.nodes[1].parentIdx == 0);           // head_pivot -> body_pivot
    CHECK(d.nodes[2].isLayer && d.nodes[2].hasSpring);
    CHECK(d.nodes[2].k == 30 && d.nodes[2].c == 5);
    CHECK(!d.nodes[3].hasSpring);
    CHECK(d.nodes[5].parentIdx == 1);           // head -> head_pivot

    // ---- spring: converges to rest with no drive ----
    float a = 1.0f, v = 0.0f;
    for (int i = 0; i < 600; i++) springStep(a, v, 40.0f, 6.0f, 0.0f, 1.0f / 60.0f);
    CHECK(std::fabs(a) < 0.01f && std::fabs(v) < 0.05f);

    // ---- solve: neutral params keep layer centers at their bbox centers ----
    std::vector<WorldXf> xf;
    Params p;
    solve(d, p, 0, xf);
    CHECK(std::fabs(xf[3].cx - 500.0f) < 1e-3);   // body center x
    CHECK(std::fabs(xf[3].cy - 850.0f) < 1e-3);   // body center y
    CHECK(std::fabs(xf[3].rot) < 1e-6);

    // breath lifts the body-parented layers (offset applied at body_pivot)
    p.breath = 1.0f;
    solve(d, p, 0, xf);
    CHECK(std::fabs(xf[3].cy - (850.0f - 6.0f)) < 1e-3);
    p.breath = 0.0f;

    // lean rotates layers around body_pivot; head chain accumulates headAngle
    p.lean = 0.3f;
    solve(d, p, 0, xf);
    CHECK(std::fabs(xf[3].rot - 0.3f) < 1e-4);
    {   // hand-rotate the body center around the pivot and compare
        float ox = 500.0f - 500.0f, oy = 850.0f - 900.0f;
        float s = std::sin(0.3f), c = std::cos(0.3f);
        float ex = 500.0f + ox * c - oy * s, ey = 900.0f + ox * s + oy * c;
        CHECK(std::fabs(xf[3].cx - ex) < 1e-3 && std::fabs(xf[3].cy - ey) < 1e-3);
    }
    p.headAngle = 0.2f;
    solve(d, p, 0, xf);
    CHECK(std::fabs(xf[5].rot - 0.5f) < 1e-4);    // head: lean + headAngle
    CHECK(std::fabs(xf[3].rot - 0.3f) < 1e-4);    // body unaffected by head
    p = Params{};

    // arm layers rotate around their own pivot on top of the chain
    p.armL = 0.4f;
    solve(d, p, 0, xf);
    CHECK(std::fabs(xf[4].rot - 0.4f) < 1e-4);
    CHECK(std::fabs(xf[3].rot) < 1e-6);
    p = Params{};

    // fade propagates to every layer's alpha (virtual nodes carry no draw state)
    p.fade = 0.5f;
    solve(d, p, 0, xf);
    for (size_t i = 0; i < d.nodes.size(); i++)
        if (d.nodes[i].isLayer) CHECK(std::fabs(xf[i].alpha - 0.5f) < 1e-6);
    p = Params{};

    // springs are velocity-driven: a sudden parent rotation makes the cape
    // lag transiently, but a HELD rotation must end with the cape exactly
    // realigned (an angle-proportional drive would leave it permanently
    // rotated against the body - the "cape separates" bug)
    p.lean = 0.3f;
    for (int i = 0; i < 30; i++) solve(d, p, 1.0f / 60.0f, xf);
    CHECK(std::fabs(xf[2].rot - 0.3f) > 1e-3);    // cape lags the step
    CHECK(std::fabs(xf[3].rot - 0.3f) < 1e-4);    // body follows exactly
    for (int i = 0; i < 600; i++) solve(d, p, 1.0f / 60.0f, xf);
    CHECK(std::fabs(xf[2].rot - 0.3f) < 0.01f);   // ...then realigns fully

    // ---- player: load from disk, update, draw through the callback ----
    const char* tmp = "rig_test_tmp.txt";
    {
        std::ofstream f(tmp);
        f << kRig;
    }
    RigPlayer player;
    CHECK(player.load(tmp));
    CHECK(player.loaded());
    player.update(1.0f / 60.0f);
    int drawn = 0;
    int zPrev = -999;
    player.draw([&](const std::string& file, Vec2, Vec2 size, Color tint, float) {
        drawn++;
        CHECK(!file.empty() && size.x > 0 && size.y > 0 && tint.a > 0);
        return true;
    }, {400, 300}, 600.0f);
    CHECK(drawn == 5);
    (void)zPrev;

    // bendX is a rigid ankle tilt: every layer gets the SAME rotation (zero
    // relative displacement), and higher layers displace more only because
    // they sit farther from the ground anchor
    p = Params{};
    p.bendX = 0.1f;
    solve(d, p, 0, xf);
    {
        float dxHead = std::fabs(xf[5].cx - (400.0f + 240.0f * 0.5f));
        float dxBody = std::fabs(xf[3].cx - 500.0f);
        CHECK(dxHead > dxBody && dxBody > 0.0f);
        CHECK(std::fabs(xf[5].rot - xf[3].rot) < 1e-6); // uniform, no gradient
        // exactness: the body-layer center rotated by hand about the anchor
        float s = std::sin(0.1f), c = std::cos(0.1f);
        float ox = 500.0f - 500.0f, oy = 850.0f - 1500.0f; // anchor (bp.px, canvasH)
        CHECK(std::fabs(xf[3].cx - (500.0f + ox * c - oy * s)) < 1e-3);
        CHECK(std::fabs(xf[3].cy - (1500.0f + ox * s + oy * c)) < 1e-3);
    }
    p = Params{};

    // hitstop freezes time: params don't advance while the timer runs
    player.trigger(Motion::Hit); // hitstop 0.07s
    float breathBefore = player.params().breath;
    player.update(0.03f);
    CHECK(player.params().breath == breathBefore); // frozen
    for (int i = 0; i < 30; i++) player.update(1.0f / 60.0f);
    CHECK(player.params().breath != breathBefore); // resumed
    CHECK(std::fabs(player.params().bendX) > 1e-3); // bend impulse landed
    for (int i = 0; i < 120; i++) player.update(1.0f / 60.0f);
    CHECK(std::fabs(player.params().bendX) < 3e-4); // settled, no pendulum

    // motion triggers change state without exploding
    player.trigger(Motion::Stagger);
    for (int i = 0; i < 120; i++) player.update(1.0f / 60.0f);
    player.strikeWindup(StrikeStyle::Smash, false, true);
    for (int i = 0; i < 30; i++) player.update(1.0f / 60.0f);
    CHECK(player.strikePosed());
    // keyframe assets missing: draw falls back to layered rendering
    drawn = 0;
    player.draw([&](const std::string& file, Vec2, Vec2, Color, float) {
        if (file.find("kf_") != std::string::npos) return false;
        drawn++;
        return true;
    }, {400, 300}, 600.0f);
    CHECK(drawn == 5);
    player.strikeRelease();
    for (int i = 0; i < 60; i++) player.update(1.0f / 60.0f);
    CHECK(!player.strikePosed());                  // auto-settled
    player.trigger(Motion::Die);
    for (int i = 0; i < 300; i++) player.update(1.0f / 60.0f);
    CHECK(player.params().fade < 0.5f);

    std::remove(tmp);
    std::printf("OK: %d checks passed\n", checks);
    return 0;
}
