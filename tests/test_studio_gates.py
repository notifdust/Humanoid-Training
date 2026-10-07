from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GATES = ROOT / "studio" / "gates.js"


def _run_gates(expr: str) -> dict | str | bool | list:
    script = f"""
const fs = require('fs');
const vm = require('vm');
const code = fs.readFileSync({json.dumps(str(GATES))}, 'utf8');
const ctx = {{ window: {{}}, globalThis: {{}} }};
ctx.globalThis = ctx;
vm.createContext(ctx);
vm.runInContext(code, ctx);
const HTGates = ctx.window.HTGates || ctx.globalThis.HTGates;
const result = {expr};
process.stdout.write(JSON.stringify(result));
"""
    proc = subprocess.run(
        ["node", "-e", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    return json.loads(proc.stdout)


def test_gates_js_exists() -> None:
    assert GATES.is_file()


def test_unsaved_demo_gate_proceeds_when_empty() -> None:
    out = _run_gates("HTGates.decideUnsavedDemoTrain(0)")
    assert out["action"] == "proceed"


def test_unsaved_demo_gate_blocks_when_pending() -> None:
    out = _run_gates("HTGates.decideUnsavedDemoTrain(2)")
    assert out["action"] == "block_save_first"
    assert "unsaved" in out["message"].lower()
    assert "Save" in out["message"] or "save" in out["message"]
    assert "scripted" in out["scriptedLabel"].lower()


def test_unsaved_takes_hint_visible_copy() -> None:
    assert _run_gates("HTGates.unsavedTakesHint(0)") == ""
    hint = _run_gates("HTGates.unsavedTakesHint(1)")
    assert "unsaved" in hint.lower()
    assert "Save" in hint


def test_catalog_groups_osmo_out_of_local_ready() -> None:
    recipes = [
        {"id": "cart", "launch_here": True, "launch_path": "local", "start_here": True, "title": "Cart"},
        {"id": "walk", "launch_here": True, "launch_path": "osmo", "title": "Walk"},
        {"id": "pick", "launch_here": False, "launch_path": "blocked", "title": "Pick"},
    ]
    grouped = _run_gates(f"HTGates.groupCatalogRecipes({json.dumps(recipes)})")
    assert [r["id"] for r in grouped["ready"]] == ["cart"]
    assert [r["id"] for r in grouped["harvest"]] == ["walk"]
    assert [r["id"] for r in grouped["later"]] == ["pick"]
    assert _run_gates(
        f"HTGates.isLocalLaunch({json.dumps(recipes[1])})"
    ) is False
    assert _run_gates(
        f"HTGates.isHarvestLaunch({json.dumps(recipes[1])})"
    ) is True


def test_next_ladder_skips_finished_start_here() -> None:
    recipes = [
        {"id": "a", "start_here": True, "launch_here": True, "launch_path": "local", "title": "A"},
        {"id": "b", "start_here": True, "launch_here": True, "launch_path": "local", "title": "B"},
        {"id": "c", "start_here": True, "launch_here": True, "launch_path": "local", "title": "C"},
    ]
    runs = [{"recipe": "a", "status": "passed"}]
    nxt = _run_gates(
        f"HTGates.nextLadderRecipe({json.dumps(recipes)}, {json.dumps(runs)}, 'a')"
    )
    assert nxt["id"] == "b"


def test_blocked_recipe_primary_opens_ready() -> None:
    blocked = {"id": "walk", "launch_here": False, "launch_path": "blocked", "title": "Walk"}
    ready = {"id": "cart", "launch_here": True, "launch_path": "local", "title": "Cartpole"}
    out = _run_gates(
        f"HTGates.decideRecipePrimary({json.dumps(blocked)}, {json.dumps(ready)})"
    )
    assert out["action"] == "open_ready"
    assert out["recipeId"] == "cart"
    assert "Cartpole" in out["label"]


def test_run_primary_blocked_and_ladder() -> None:
    ready = {"id": "cart", "title": "Cartpole", "launch_here": True, "launch_path": "local"}
    blocked = _run_gates(
        f"HTGates.decideRunPrimary({{status:'blocked'}}, {json.dumps(ready)}, null)"
    )
    assert blocked["action"] == "open_ready"
    assert blocked["recipeId"] == "cart"

    ladder = {"id": "stand", "title": "G1 stand"}
    passed = _run_gates(
        f"HTGates.decideRunPrimary({{status:'passed', recipe:'cart'}}, {json.dumps(ready)}, {json.dumps(ladder)})"
    )
    assert passed["action"] == "open_ready"
    assert passed["recipeId"] == "stand"
    assert "stand" in passed["label"].lower()


def test_data_step_done_requires_demos_or_visit() -> None:
    imitate = {"imitate": True}
    assert _run_gates(f"HTGates.dataStepDone({json.dumps(imitate)}, [], false)") is False
    assert _run_gates(f"HTGates.dataStepDone({json.dumps(imitate)}, ['x'], false)") is True
    assert _run_gates(f"HTGates.dataStepDone({json.dumps(imitate)}, [], true)") is True
    assert _run_gates("HTGates.dataStepDone({imitate:false}, [], false)") is True


def test_compare_gate_requires_two_same_recipe() -> None:
    runs = [
        {"run_id": "a", "recipe": "cartpole-balance"},
        {"run_id": "b", "recipe": "cartpole-balance"},
        {"run_id": "c", "recipe": "g1-stand"},
    ]
    empty = _run_gates(
        f"HTGates.decideCompareSelection([], {json.dumps(runs)})"
    )
    assert empty["ok"] is False
    assert "two" in empty["reason"].lower()

    mismatch = _run_gates(
        f"HTGates.decideCompareSelection(['a','c'], {json.dumps(runs)})"
    )
    assert mismatch["ok"] is False
    assert "same task" in mismatch["reason"].lower()

    ok = _run_gates(
        f"HTGates.decideCompareSelection(['a','b'], {json.dumps(runs)})"
    )
    assert ok["ok"] is True
    assert ok["recipe"] == "cartpole-balance"
    assert len(ok["runs"]) == 2


def test_deploy_gate_follows_preflight() -> None:
    blocked = _run_gates(
        "HTGates.decideDeployButton({status:'blocked'}, null)"
    )
    assert blocked["disabled"] is True
    assert "never trained" in blocked["title"].lower()

    running = _run_gates(
        "HTGates.decideDeployButton({status:'running'}, null)"
    )
    assert running["disabled"] is True

    checking = _run_gates(
        "HTGates.decideDeployButton({status:'passed'}, null)"
    )
    assert checking["disabled"] is True
    assert "Checking hardware gate" in checking["title"]

    refused = _run_gates(
        "HTGates.decideDeployButton({status:'passed'}, {ok:false, reasons:['sim-only']})"
    )
    assert refused["disabled"] is True
    assert "sim-only" in refused["title"]

    cleared = _run_gates(
        "HTGates.decideDeployButton({status:'passed'}, {ok:true})"
    )
    assert cleared["disabled"] is False
    assert "unwired" in cleared["title"].lower() or "fail-closed" in cleared["title"].lower()


def test_studio_wires_gates_and_train_busy() -> None:
    js = (ROOT / "studio" / "app.js").read_text(encoding="utf-8")
    html = (ROOT / "studio" / "index.html").read_text(encoding="utf-8")
    assert 'src="/gates.js"' in html
    assert "HTGates.decideUnsavedDemoTrain" in js
    assert "HTGates.decideCompareSelection" in js
    assert "HTGates.decideDeployButton" in js
    assert "HTGates.groupCatalogRecipes" in js
    assert "HTGates.decideRecipePrimary" in js
    assert "HTGates.decideRunPrimary" in js
    assert "HTGates.nextLadderRecipe" in js
    assert "block_save_first" in js
    assert "Can harvest remotely" in js
    assert "Demos →" in js
    assert "id=\"unsaved-demo-hint\"" in js
    assert "trainBusy" in js
    assert "function formatHealthStrip" in js
    assert "walk:later" in js
    assert "function mustardObject" not in js
    assert "function recordTargetObject" in js
    assert "Why this stays sim-only" in js
    assert "Optional filter" in html
