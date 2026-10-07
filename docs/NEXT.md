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

## Conclusion (2026-09-30) — **LOCKED**

**Can you conclude without a GPU?** Yes — on architecture, honesty, and
CPU-safe paper wraps. **No** — on the humanoid wedge exit test (live walking
`eval.mp4`). That remains the only gate that matters for claiming Phase 3c.

| Question | Answer |
|---|---|
| Did we stay a studio/compiler over engines? | **Yes** |
| Did we fake walk / ACT / OSMO / GR00T on CPU? | **No** |
| Is the laptop Canva loop done? | **Yes** (Cartpole, stand, mustard + gold + CI) |
| Is G1 walk a product video yet? | **No** — needs `ht proof walk` on NVIDIA GPU |
| Are SOTA papers “implemented”? | **Pinned / harnessed**, not live-trained: Playground/mjlab walk, BeyondMimic→`g1-track`+LAFAN1 Hub, ACT check, Isaac Mimic pins, GR00T-via-LeRobot check, HF Jobs check |
| What closes the story? | One GPU operator run of N1b, then N2b OSMO (or HF) harvest |

```
CPU-safe work:            DONE (harness, pins, Hub, intuition, honesty) — LOCKED
Live wedge (N1b→N5):      BLOCKED here — no NVIDIA GPU / OSMO / robot
Mission alignment:        HIGH
Mission completion:       PARTIAL — stop polishing; run N1b on a GPU box
Do not reopen:            B6 theme, more Hub pins, or fifth room until N1b lands
```

### Operator handoff — close N1b (pick one)

**A — Local NVIDIA GPU box**

```bash
cd Humanoid-Training   # existing clone is fine
git pull
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[playground]'
pip install 'jax[cuda12]==0.9.2'   # required — bare playground pulls jax 0.11 + CPU jaxlib
python -m humanoid_training.cli proof walk --check   # want ok:true, jax backend gpu
python -m humanoid_training.cli proof walk
```

Paste the run id (or `proof_3c.json`) into an issue/PR so ROADMAP Phase 3c
can flip to done.

**B — Remote GPU host over SSH**

Same commands as A on any NVIDIA box you can reach. Copy `proof_3c.json`
back when finished.

**C — No local GPU (Phase 3d instead)**

Add OSMO credentials, or `HF_TOKEN` + Hugging Face CLI, then pursue **N2**
live harvest — not a fake local Phase 3c.

Until A, B, or C finishes: do not mark Phase 3c done; do not check in a
stand clip as walk gold.

---

## Mission scorecard (2026-09-30; was 2026-09-28)

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
| **No honest G1 loco-manipulation recipe** (Isaac PickPlace pin) | **partial** | `g1-pickplace` pinned; live Robomimic train still needs Isaac + dataset |
| **No Unitree driver / hardware eval** (Phase 4 live) | **expected later** | Correctly gated; must follow live walk. |
| **Thin recipe library** (few video-backed) | **improving** | CPU trio + walk/rough/track + PickPlace×2 — still thin vs 5–10 **video-backed** |
| **Power-user escape hatch still thin** | **low** | Export + English + Hub pins done; engine version pins still light. |
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

### N1 — Phase 3c: live GPU walk proof

**N1a — operator harness (done on CPU).** Host preflight, durable report,
studio projection — so a GPU operator has a clear path and evidence file.

| Work | Status |
|---|---|
| `assess_walk_proof_host` + `ht proof walk --check` | done |
| `GET /api/proof/walk` | done |
| Write `proof_3c.json` beside a proof run | done |
| Studio G1 walk Train page shows Phase 3c readiness | done |
| Still no fake walk success on CPU CI | held |

**N1b — live clip (next; needs NVIDIA GPU).** ← **current focus**

```bash
pip install playground   # or mjlab / Isaac Lab
ht proof walk --check    # should report ok:true on the GPU box
ht proof walk
```

Must produce a **walking** `eval.mp4`, stamp `facts.engine`, write
`proof_3c.json`, studio Plays it with a backend badge.

| Work | Notes |
|---|---|
| Run `ht proof walk` on a real GPU | Capture run id, engine, duration |
| Fix any harvest / facts bugs found live | Only if proof fails honestly |
| Document “verified on &lt;GPU&gt; / &lt;engine&gt;” in ROADMAP | Date + hardware note |
| Still **no** gold walk clip in git | Keep |

This CPU environment cannot close N1b. Do not claim Phase 3c done.

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

### N4 — Honest G1 manipulation recipe (done pin; live train still needs Isaac)

**Exit test (pin met).** Recipe `g1-pickplace` pins
`Isaac-PickPlace-Locomanipulation-G1-Abs-v0` with `workflow: robomimic`
(not rsl_rl). Compiles Robomimic train script; launch fail-closes without
`HT_ISAAC_DATASET`. CPU mustard stays `pick-and-place`.

| Work | Status |
|---|---|
| Confirm upstream task id from Isaac docs | done |
| New recipe + adapter robomimic workflow + catalog card | done |
| No gold clip | done |
| Live Robomimic train on GPU + Mimic hdf5 | **not done** (needs Isaac + dataset) |

Fixed-base alternative: done pin (`g1-pickplace-fixed`).

### N5 — Phase 4: live hardware (only after N1b)

**Exit test.** Passed sim walk → reduced-speed Unitree driver; NaN /
pose-limit kills; hardware eval clears `sim_only`.

| Work | Notes |
|---|---|
| Unitree reduced-speed driver | Not stubbed success |
| Hardware eval profile that Deploy accepts | `HT_HARDWARE_PROFILE` |
| Studio Deploy stays fail-closed until then | Gate already correct |

### N6 — Grow the recipe library (after N1b at least)

Vision’s “5–10 video-backed tasks.” Prefer **reproducible pins** over UI.

Candidates (only with real engines / demos):

- Second locomotion (rough terrain / tracking) if upstream env exists
- One more imitation recipe on the same teleop stack
- H1 recipe **only after** G1 walk has a live clip

### N7 — Studio polish that serves the loop (partial)

| Work | Status |
|---|---|
| `ht export <spec\|run_id>` power-user escape hatch | done |
| English summary from `facts` (Python + studio) | done |
| Accessibility / motion prefs | done (`prefers-reduced-motion`) |
| **N7d** API projects `run.english`; studio prefers it | done |
| **N7e** export reports `runnable_here` + `next_step` | done |
| **N7f** process intuition (OSMO group, Data-first imitate, blocked→ready, ladder) | done |
| **N7g** GR00T/Arena check-only (`ht proof groot`) | done |
| **N7h** Hub pins + LAFAN1 motion resolve + Data room | done |

Do **not** start a B6 theme rewrite or fifth room.

---

## Fallacy fixes (2026-09-29)

| Fallacy | Fix |
|---|---|
| PickPlace `launch_here` without `HT_ISAAC_DATASET` | Require resolvable Mimic hdf5 |
| Pill "works on this GPU" when path is OSMO | Catalog `launch_path` + OSMO promise copy |
| Rough showed Phase 3c flat-proof UI | `studio.proof` catalog field (`walk` only on flat) |
| `ht proof act` looked like a live train | Check-only; command points at `ht train` mustard |
| Robomimic no-mp4 blamed on headless | Studio branches on `facts.video=missing` |

| Bug | Fix |
|---|---|
| OSMO-only made `g1-pickplace` `launch_here` | Robomimic recipes require `isaac_local_ready` only |
| Phase 3c assess `ok:true` via OSMO without GPU | Walk engines for proof use local Isaac; `ok` needs GPU |
| Robomimic exit 0 + no mp4 → status `blocked` | Pass with `facts.video=missing` + honest note |
| Pill said "works on this GPU" on OSMO-only | Pill projects GPU vs OSMO from health |

Still open (needs hardware): N1b live walk, N2b OSMO live, N3b ACT live, N4b Robomimic live.

## Further roadmap (operator order)

| Priority | Item | Needs | Status |
|---|---|---|---|
| **1** | **N1b** live `ht proof walk` on a GPU box | NVIDIA GPU + Playground/mjlab/Isaac | **next** |
| 2 | **N2b** live OSMO harvest from a laptop | OSMO credentials + pool | harness ready (`ht proof osmo`) |
| 3 | **N2c** live HF Jobs harvest (same contract) | HF_TOKEN + `hf` CLI | harness ready (`ht proof hf-jobs`); live harvest TODO |
| 4 | **N3b** live ACT on pick-and-place demos | GPU + `lerobot[training]` | harness ready (`ht proof act`) |
| 5 | N4b live Robomimic train for `g1-pickplace` | Isaac Lab + `HT_ISAAC_DATASET` | pin done (+ fixed-base sibling) |
| 6 | N5 Unitree hardware driver (after N1b) | Robot + live walk | gate only |
| 7 | N6 more video-backed recipes | After walk proof | rough + track + fixed PickPlace + LAFAN1 Hub pin |
| 8 | N7 polish / GR00T / Hub pins | Optional | N7b–N7h done on CPU |

### Shipped this track (CPU-safe)

| Track | What |
|---|---|
| **N2a** | `ht proof osmo` + `GET /api/proof/osmo` host preflight |
| **N3a** | `ht proof act` + `GET /api/proof/act` host preflight |
| **N6a** | `g1-walk-rough` → `Mjlab-Velocity-Rough-Unitree-G1` / `Isaac-Velocity-Rough-G1-v0` |
| **N7b** | `prefers-reduced-motion` studio CSS |
| **N7d** | `GET /api/runs` + `/api/runs/{id}` (+ SSE) attach `english` from facts; studio `runEnglish()` |
| **N7e** | `ht export` returns `runnable_here` / `next_step`; writes `export_meta.json` + honest README |
| **N7f** | Tasks: OSMO≠local ready; imitate opens Data; blocked primary→ready task; post-pass ladder; unsaved→block Save |
| **N2c** | `ht proof hf-jobs` + `GET /api/proof/hf-jobs` (CLI+token; live harvest not wired) |
| **N6b** | Recipes `g1-track` (mjlab Tracking + `HT_MJLAB_MOTION`), `g1-pickplace-fixed` (Isaac fixed-base Mimic) |
| **N6c** | Hub `hf:user/dataset` → inspect/cache (`huggingface_hub`); Data room accepts Hub ids |
| **N7g** | `ht proof groot` + `GET /api/proof/groot` (check-only; fine-tune stays NVIDIA course) |
| **N6d / N7h** | Hub pins (`ht datasets pins`, `GET /api/datasets/pins`): LAFAN1 G1 CSV/NPZ, LeRobot PushT example, GR00T-via-LeRobot path; `ht datasets motion` / `POST /api/datasets/motion` caches motion without claiming ACT |
| **N1b lock** | Operator handoff A/B/C in NEXT/ROADMAP/README; `assess_walk_proof_host.handoff`; studio G1 walk Train shows the same paths when blocked |

`GET /api/proof` bundles walk + osmo + act + hf_jobs + groot readiness.

---

## What not to do next

- Fake walk / ACT / OSMO success on CPU CI
- Re-open `g1-reach` or invent `G1Reach-v0`
- Hardware torque before a live walk video
- Finger grasping before live ACT
- New physics / policy / dataset format / cluster product
- Competing with LeLab on SO-ARM101
- Another long polish-only train while N1b is still open

---

## How to use this file

1. Default PR focus: **N1b**, then N2–N5 in order.
2. If the environment has no GPU / OSMO / robot, ship harness fixes and
   docs honesty — do not claim the live exit test.
3. Update [ROADMAP.md](./ROADMAP.md) status tables when an exit test is
   met; update this file’s scorecard date when the verdict changes.
4. Keep [BETTERMENT.md](./BETTERMENT.md) as historical polish — reopen
   only for new CPU honesty holes.
5. After each shipped track, refresh the **Further roadmap** table above.
