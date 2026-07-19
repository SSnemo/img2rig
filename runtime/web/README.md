# Web reference player

A dependency-free ES module that renders a `rig.txt` layer pack on a 2D
canvas, with the same solver and motion machine as the C++ runtime.

```bash
# from the repository root
python -m http.server 8000
# open http://localhost:8000/runtime/web/  (loads the bundled demo character)
# or point at any rig: .../runtime/web/?rig=/path/to/rig.txt
```

Embedding:

```js
import { RigPlayer } from "./rig-player.js";
const rig = await RigPlayer.load("assets/mychar/rig.txt");
// each animation frame:
rig.update(dt);                     // seconds; freeze time to freeze her
rig.draw(ctx, x, y, displayHeight); // (x, y) = character center
rig.trigger("stagger");             // hit | stagger | enrage | die | ...
```

The solver is duplicated by design across `rig_math.h` (C++, unit-tested),
`preview.py` (offline GIF renderer) and `rig-player.js` (this file) so each
stays dependency-free. If you change one, change all three.
