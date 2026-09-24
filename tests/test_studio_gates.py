from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GATES = ROOT / "studio" / "gates.js"


def _run_gates(expr: str) -> dict:
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


def test_unsaved_demo_gate_confirms_when_pending() -> None:
    out = _run_gates("HTGates.decideUnsavedDemoTrain(2)")
    assert out["action"] == "confirm_discard_or_cancel"
    assert "unsaved" in out["message"].lower()
    assert "Save" in out["message"] or "save" in out["message"]
    assert "scripted" in out["confirmLabel"].lower()
    assert "Save" in out["cancelLabel"] or "save" in out["cancelLabel"]


def test_unsaved_takes_hint_visible_copy() -> None:
    assert _run_gates("HTGates.unsavedTakesHint(0)") == ""
    hint = _run_gates("HTGates.unsavedTakesHint(1)")
    assert "unsaved" in hint.lower()
    assert "Save" in hint


def test_studio_wires_gates_and_train_busy(monkeypatch: pytest.MonkeyPatch) -> None:
    js = (ROOT / "studio" / "app.js").read_text(encoding="utf-8")
    html = (ROOT / "studio" / "index.html").read_text(encoding="utf-8")
    assert 'src="/gates.js"' in html
    assert "HTGates.decideUnsavedDemoTrain" in js
    assert "id=\"unsaved-demo-hint\"" in js
    assert "trainBusy" in js
    assert "function formatHealthStrip" in js
    assert "walk:later" in js
