# Betterment roadmap

Improve what already ships. Do **not** use this file to invent engines,
policies, recipes, orchestrators, or live GPU / hardware proofs.

The phase roadmap ([ROADMAP.md](./ROADMAP.md)) still owns walk proof,
OSMO live harvest, ACT-on-GPU, and Unitree torque. This file owns making
the **CPU studio loop** clearer, honester, and harder to misuse.

```
B0  stop silent wrong trains          ← done
B1  one train at a time + health strip ← done
B2  copy / docs match the four rooms  ← done
B3  Deploy + Robots honesty           ← done
B4  status, Data, first-run friction  ← done
B5  interaction tests + small cleanup ← next
```

Each track has an **exit test**. If the test fails, the track is not
done — even if the UI looks nicer.

---

## What we are not bettering here

Leave these to [ROADMAP.md](./ROADMAP.md) (or later):

- Live G1 walk / ACT / OSMO cluster proof
- New Isaac PickPlace env ids
- Hardware driver / clearing `sim_only`
- A fifth studio room
- New physics, policy families, dataset formats, or cluster orchestrators

---

## What is already solid

Do not re-litigate these; build on them.

| Area | Solid today |
|---|---|
| Compiler contract | Spec → recipe → adapter → `manifest.json` + `facts` + optional `eval.mp4` |
| Catalog | Studio projects recipe `studio.*` fields; no ready/later id hardcoding |
| CPU loop | Cartpole, G1 stand, pick-and-place + gold clips + CI frame compare |
| Fail-closed GPU | Walk blocks on CPU; no stand-as-walk; no `g1-reach` |
| Imitation | Empty keep refused; failure-only keep fails mustard; `policy=act\|linear-bc` |
| Runs | Compare same-recipe; Deploy preflight; `sim_only` until hardware |

---

## B0 — Stop silent wrong trains (done)

**Problem.** A beginner can record canvas takes, skip **Save**, and click
**Train**. The “no demos saved” hint vanishes once takes exist, and Train
falls back to scripted demos. That is an honesty hole.

**Exit test.**

1. Open Pick and place → record at least one canvas take → do **not** Save.
2. Click Train → studio **blocks or confirms** with an explicit choice:
   save demos first, or train on scripted demos on purpose.
3. After Save (or explicit “use scripted”), Train proceeds as today.
4. A test covers the pending-trajectory path (behavior, not only a
   string-presence assert in `app.js`).

| Deliverable | Status |
|---|---|
| Keep / restore a visible “unsaved takes” warning while `pendingTrajectories.length > 0` | done |
| Train refuses silent scripted fallback when unsaved takes exist | done |
| Behavioral test for unsaved → Train | done (`studio/gates.js` + `tests/test_studio_gates.py`) |

Primary paths: `studio/gates.js`, `studio/app.js` (`boundDemoHint`, `trainCurrent`,
`pendingTrajectories`).

---

## B1 — One train at a time + readiness strip (done)

**Problem.** Studio trains are fire-and-forget daemon threads. Double-click
Train (or train from two tabs) can overlap MuJoCo/GLFW and scramble logs.
The header only shows `local · v…` even though `/api/health` already knows
engine probes.

**Exit test.**

1. Start Train on Cartpole; a second Train while the first is
   `queued`/`running` is refused or clearly queued as “one at a time.”
2. Header (or a one-line readiness strip) projects GPU / Playground /
   mjlab / Isaac / OSMO / render hints from `/api/health` — enough for a
   beginner to see why G1 walk is “later.”

| Deliverable | Status |
|---|---|
| Server: at most one active in-process train (or explicit queue) | done (409 Conflict) |
| Studio: disable Train + show “already training …” across rooms | done (`trainBusy` + 409 message) |
| Project health engines (and render/video possible when known) in the chrome | done (`formatHealthStrip`) |

Primary paths: `src/humanoid_training/server.py`, `studio/app.js`,
`studio/index.html` / `styles.css`.

---

## B2 — Copy and docs match the four rooms (done)

**Problem.** README still says “Skip G1 reach.” Vision still describes five
rooms / Evaluate as its own room. Recipe and robot blurbs drift from the
phase board. Beginners read two products.

**Exit test.**

1. README “What to click” never mentions `g1-reach`.
2. Vision’s studio map matches four rooms (Evaluate lives in Runs).
3. Pick-and-place / H1 / Architecture OSMO lines do not claim unfinished
   phases as if they were the next click.
4. Cartpole `recipe.yaml` gold notes are not duplicated nonsense.

| Deliverable | Status |
|---|---|
| README: drop reach; keep Cartpole → stand → mustard → walk-blocked | done |
| VISION: four-room map; Cartpole-first Phase 0 aligns with ROADMAP | done |
| Recipe / robot catalog copy hygiene (H1 “catalog only”, pick-and-place Phase wording) | done |
| ARCHITECTURE: OSMO harvest = harness done, live proof not done | done |
| Deduplicate cartpole `gold.notes` in `recipe.yaml` | done |
| Reliable `./run-studio.sh` (reuse install / repair broken venv) | done |

Primary paths: `README.md`, `docs/VISION.md`, `docs/ARCHITECTURE.md`,
`robots/catalog.yaml`, `recipes/*/recipe.yaml`, `run-studio.sh`.

---

## B3 — Deploy and Robots honesty (done)

**Problem.** Every finished run gets a clickable **Deploy to robot**, so
Cartpole and stand feel hardware-ready until the gate text appears. The
Robots room hardcodes “Unitree G1” while the catalog lists CartPole and
H1; choosing H1 still shows `start_here` smoke tasks and feels broken.
The scene badge hardcodes `G1`.

**Exit test.**

1. After preflight `ok: false`, Deploy is disabled (or demoted) and the
   “why sim-only” list remains the explanation.
2. Robots lede and scene mark project the selected catalog robot.
3. H1 is marked catalog-only / Use disabled until an H1 recipe exists —
   no fake H1 train path.

| Deliverable | Status |
|---|---|
| Deploy button state follows preflight (not “always clickable”) | done |
| Robots room copy from catalog, not hardcoded G1 prose | done |
| Scene `robot-mark` from selected robot | done |
| H1: honest empty / catalog-only state | done |

Primary paths: `studio/app.js` (`deployButtonState`, `loadDeployPreflight`,
`renderRobots`), `robots/catalog.yaml`.

---

## B4 — Status color, Data in the loop, first-run friction (done)

**Problem.** Blocked (“can't train here”) and failed (“finished — did not
pass”) share orange. The Task → Train → Video steps ignore Data, so demos
feel optional. First G1 train may download ~30MB Menagerie with no preview;
headless success without `eval.mp4` still surprises people.

**Exit test.**

1. Blocked and failed use distinct status colors / wording in Runs.
2. Imitation recipes show Data in the step strip (or an explicit “demos”
   step) before Train.
3. G1 stand / first Menagerie fetch: Train page warns about the download;
   offline failure names Menagerie, not a raw stack.
4. Empty-video state mentions `HT_NO_RENDER` / display briefly when no
   `eval.mp4` but metrics passed.

| Deliverable | Status |
|---|---|
| Distinct blocked vs failed styling | done (`--blocked` vs `--bad` / `--warn`) |
| Data visible in steps for imitate recipes | done (`stepsHTML` + Data button) |
| Menagerie first-run copy + clearer offline error | done (`train_hint` + `assets.py`) |
| Honest empty-video copy when render skipped | done (`HT_NO_RENDER` copy) |

Primary paths: `studio/app.js`, `studio/styles.css`,
`src/humanoid_training/assets.py` (error text), recipe `train_hint`s.

---

## B5 — Interaction tests and small cleanup

**Problem.** Studio contracts are mostly “string exists in `app.js`.” That
misses the unsaved-demo bug class. Dead CSS and mustard-named helpers make
the code look unfinished.

**Exit test.**

1. At least one Playwright (or equivalent) flow: Save demos → Train, or
   unsaved → blocked Train; plus compare same-recipe gate; plus deploy
   preflight HTML.
2. Dead `.start-here` CSS gone; `mustardObject` renamed to the generic
   target helper (behavior unchanged).
3. Optional: short CONTRIBUTING / troubleshooting pointer from README
   (Menagerie, display, Docker missing, walk exit 12).

| Deliverable | Status |
|---|---|
| Interaction tests for Save/Train, compare, deploy preflight | **not done** |
| Rename mustard helpers; remove dead CSS | partial (`recordTargetObject` + dead CSS gone; alias kept) |
| CONTRIBUTING or README troubleshooting section | **not done** |

Primary paths: `tests/`, `studio/app.js`, `studio/styles.css`, `README.md`.

---

## How to work a betterment track

1. Branch off the latest merged betterment / phase tip:
   `cursor/betterment-bN-<slug>-197a`.
2. Change the **smallest** surface that fixes the exit test (usually
   studio + one contract test). Prefer projecting server facts over
   hardcoding recipe ids in the browser.
3. Do not add a studio control that cannot be expressed in the job spec
   or catalog.
4. Keep fail-closed: never substitute stand clips, never invent env ids,
   never claim ACT/walk/hardware success on CPU.
5. Update this file’s status table when a track’s exit test is met.
6. Full `pytest` green before claiming done.

Suggested merge order: **B0 → B1 → B2 → B3 → B4 → B5**. Parallelize only
when tracks do not touch the same Train path (e.g. B2 docs can ride
beside B0).

---

## Severity cheat sheet (from the audit)

| Priority | Theme | Track |
|---|---|---|
| high | Unsaved demos → silent scripted train | B0 |
| high | Overlapping studio trains | B1 |
| high | Studio tests miss real Train flows | B5 |
| med | Stale reach / five-room / phase copy | B2 |
| med | Deploy always clickable; H1 empty; scene `G1` | B3 |
| med | Blocked vs failed color; Data orphaned; Menagerie | B4 |
| low | Dead CSS, mustard names, license, a11y pass | B5 / later |
