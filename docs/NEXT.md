# Next implementation roadmap (post–Betterment B5)

This file answers two questions:

1. **How far did we drift from the original mission?**
2. **What should we implement next?**

Product north star remains [VISION.md](./VISION.md). Phase exit tests remain
in [ROADMAP.md](./ROADMAP.md). CPU-studio polish tracks B0–B5 are **done** —
see [BETTERMENT.md](./BETTERMENT.md). Do not open a B6 track unless a new
CPU honesty hole appears.

---

## Original mission (from the initial prompts + VISION)

Build a **studio + compiler + recipe library** over existing engines.

| Pillar | Original ask |
|---|---|
| Product shape | Canva on MuJoCo / Isaac Lab — not Photoshop, not a new simulator |
| Wedge | Unitree-class **humanoid first** (G1), not SO-ARM / LeLab turf |
| Success metric | **Eval video** + pass/fail English, not TensorBoard |
| Loop | recipe → data → train → eval video → promote or iterate |
| Compiler | Job spec → adapter → engine; UI is a projection of the spec |
| Engines | Playground / mjlab first; Isaac Lab when needed; LeRobot for imitation |
| Refuse to own | New physics, new policy family, new dataset format, new orchestrator, fleet ops |
| Beginner bar | Pick recipe → Train → watch video without installing CUDA / Isaac Sim |
| Honest failure | Block + next step when GPU / engine missing — never fake a walk |

Phased build from the same prompt: **contract → studio shell → demos →
second engine / cloud → real robot**. Each phase leaves recipes and videos,
not just code.

---

## Mission scorecard (2026-09-28)

### Still on mission (do not reopen)

| Mission item | Status |
|---|---|
| Studio + compiler, not a simulator | **held** — adapters wrap gymnasium / mujoco / Playground / mjlab / Isaac / LeRobot |
| Job spec SoT; recipe catalog + `facts` as UI contracts | **held** |
| Four rooms (Evaluate inside Runs) | **held** |
| Eval-video-first CPU loop | **held** — Cartpole, G1 stand, pick-and-place + gold + CI |
| Fail closed on GPU / deploy / dishonest recipes | **held** — walk blocks; no `g1-reach`; no stand-as-walk; `sim_only` |
| Humanoid named wedge (G1 in catalog + walk recipe) | **held** as compile/launch path |
| Non-goals (physics / policy / format / orchestrator / fleet) | **held** |

### Intentional substitutions (honest, still mission-shaped)

These look like “not what VISION named,” but they follow the trap section:
do not ship theater.

| Vision named | What ships | Why it is not a betrayal |
|---|---|---|
| ACT / diffusion on demos | Linear-BC on CPU; ACT **launch** when LeRobot+GPU | CPU laptop cannot pretend ACT trained |
| G1 walk train | Launch path + `ht proof walk`; **no live clip in CI** | No GPU in this environment; fake gold forbidden |
| Gamepad / XR teleop | Canvas, WASD, stick → same table-frame takes | Same Data contract; Unitree XR later |
| Balance / locomotion RL on laptop | G1 **stand** hold + wave | Stand is not walk; copy says so |
| Hosted GPU for beginners | OSMO harvest **harness**; Docker CPU Phase 1 | Live pool proof still required |
| Real robot Try-on | Fail-closed Deploy gate only | Gate before torque is correct |

### Where we **have** deferred (real gaps vs the original mission)

| Gap | Severity | Why it matters |
|---|---|---|
| **No live walking `eval.mp4` from a real GPU** (Phase 3c) | **critical** | The wedge is humanoid locomotion. Without this, G1 walk is a compiler demo, not the product. |
| **No live OSMO / hosted-GPU harvest** (Phase 3d) | **high** | Beginners were promised “compute is someone else’s problem.” Laptop still cannot finish walk. |
| **No live ACT train** (Phase 3e) | **high** | Imitation path is still linear-BC for anyone without LeRobot+GPU. |
| **No honest G1 loco-manipulation recipe** (Isaac PickPlace pin) | **med** | Vision’s “canned loco-manipulation” is still stand / mustard / blocked walk. |
| **No Unitree driver / hardware eval** (Phase 4 live) | **expected later** | Correctly gated; must follow live walk. |
| **Thin recipe library** (4 recipes, 1 GPU) | **med** | Vision asked for 5–10 video-backed tasks. Library *is* the product. |
| **Power-user escape hatch still thin** | **low** | Spec JSON under Advanced exists; export runnable script / pin engine versions is light. |
| **NL / rich scene authoring** | **low / later** | Vision listed sentence + 3D rearrange as optional inputs — not the wedge exit. |

### Process deviation (recent work)

After Phase 4b landed, the last merged tracks were **Betterment B0–B5 + UI
theme/intuition** — polish of the CPU studio that already worked.

That was useful (honesty gates, one-train, Deploy/H1 honesty, Next cue,
teal theme). It was **not** progress on the humanoid wedge’s exit test.

**Verdict:** Architecture and honesty stayed loyal to the mission. The
*implementation center of gravity* drifted from “prove G1 walk on video”
to “make the laptop studio clearer.” B5 closes that polish pass. Next
work must return to the phase board.

```
Mission alignment:        HIGH on contracts / honesty / non-goals
Mission completion:       PARTIAL — CPU Canva loop yes; humanoid video no
Recent deviation:         Betterment/UI over Phase 3c live proof
Corrective action:        Resume ROADMAP Phase 3c → 3d → 3e → PickPlace → 4
```

---

## Ordered next steps (implementation)

Do these in order. Do **not** invent engines, policies, formats, or
orchestrators. Do **not** check in a fake walk clip.

### N1 — Phase 3c: live GPU walk proof  ← **next**

**Who:** operator with an NVIDIA GPU box (not this CPU CI agent alone).

**Exit test.**

```bash
pip install playground   # or mjlab / Isaac Lab
ht proof walk
# optional: ht proof walk --prefer mjlab
# optional: HT_ISAAC_CLI=… ht proof walk --prefer isaaclab
```

Must produce a **walking** `eval.mp4`, stamp `facts.engine`, studio Plays
it with a backend badge. Harness already exists; live clip does not.

| Work | Notes |
|---|---|
| Run `ht proof walk` on a real GPU | Capture run id, engine, duration |
| Fix any harvest / facts bugs found live | Only if proof fails honestly |
| Document “verified on &lt;GPU&gt; / &lt;engine&gt;” in ROADMAP | Date + hardware note |
| Still **no** gold walk clip in git | Keep |

**Blocker if agent lacks GPU:** land any harness fixes discovered; leave
“live clip” unchecked until an operator (or GPU-backed agent) runs proof.

### N2 — Phase 3d: live OSMO (or equivalent) harvest

**Exit test.** From a machine **without** CUDA, with OSMO logged in:
`ht train` on `g1-walk` submits → polls → rsyncs `eval.mp4`, studio shows
`facts.launch=osmo` / `facts.device=remote`.

| Work | Notes |
|---|---|
| Live OSMO pool proof once | Credentials + pool required |
| Harden poll / timeout copy from live failures | Fail closed stays |
| HF Jobs path | Same harvest contract later — only after OSMO or instead if OSMO blocked |

This is how the beginner bar (“no CUDA install”) reconnects to the wedge.

### N3 — Phase 3e: live ACT on pick-and-place demos

**Exit test.** GPU + `lerobot[training]` trains ACT on the same local demos;
`facts.policy=act`; CPU linear-BC unchanged. No finger grasping.

| Work | Notes |
|---|---|
| Live ACT train on GPU | Same demos as linear-BC |
| Studio badge / Runs facts show `policy=act` | Already projected when stamped |
| Keep mustard-in-bowl success criterion | Do not widen to grasping |

### N4 — Honest G1 manipulation recipe (only with a real upstream id)

**Exit test.** Pin a **confirmed** Isaac Lab / mjlab G1 PickPlace (or
equivalent) task string under an honest recipe name. No invented env ids.

| Work | Notes |
|---|---|
| Confirm upstream task id on a GPU box | Write it down in recipe + ROADMAP |
| New recipe + adapter pin + studio catalog card | Fail closed without engine |
| Gold only if `availability: cpu` | GPU recipes: no fake success clip |

If no upstream id is confirmable, **skip** — keep `pick-and-place` as the
arm demo. Deleting `g1-reach` already satisfied honesty.

### N5 — Phase 4: live hardware (only after N1)

**Exit test.** Passed sim walk → reduced-speed Unitree driver; NaN /
pose-limit kills; hardware eval clears `sim_only`.

| Work | Notes |
|---|---|
| Unitree reduced-speed driver | Not stubbed success |
| Hardware eval profile that Deploy accepts | `HT_HARDWARE_PROFILE` |
| Studio Deploy stays fail-closed until then | Gate already correct |

### N6 — Grow the recipe library (after N1 at least)

Vision’s “5–10 video-backed tasks.” Prefer **reproducible pins** over UI.

Candidates (only with real engines / demos):

- Second locomotion (rough terrain / tracking) if upstream env exists
- One more imitation recipe on the same teleop stack
- H1 recipe **only after** G1 walk has a live clip

### N7 — Optional studio polish (only if it serves the loop)

Betterment B0–B5 are done. Further UI work is **optional** and must not
block N1–N5. Acceptable later:

- Spec inspector / export script for power users
- Stronger English failure modes from `facts` (“fell at 1.2s”)
- Accessibility / motion prefs

Do **not** start a B6 theme rewrite or fifth room.

---

## What not to do next

- Fake walk / ACT / OSMO success on CPU CI
- Re-open `g1-reach` or invent `G1Reach-v0`
- Hardware torque before a live walk video
- Finger grasping before live ACT
- New physics / policy / dataset format / cluster product
- Competing with LeLab on SO-ARM101
- Another long polish-only train while 3c is still open

---

## How to use this file

1. Default agent / PR focus: **N1**, then N2–N5 in order.
2. If the environment has no GPU / OSMO / robot, ship harness fixes and
   docs honesty — do not claim the live exit test.
3. Update [ROADMAP.md](./ROADMAP.md) status tables when an exit test is
   met; update this file’s scorecard date when the verdict changes.
4. Keep [BETTERMENT.md](./BETTERMENT.md) as historical polish — reopen
   only for new CPU honesty holes.
