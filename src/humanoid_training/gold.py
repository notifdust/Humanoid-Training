"""Phase 2.5: recipe gold eval clips and a coarse resemblance check.

Gold lives on disk next to the recipe (`gold/eval.mp4`). The studio projects
`has_gold` from that file; this module must not hardcode recipe ids.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from humanoid_training.errors import AdapterUnavailable, RecipeError
from humanoid_training.recipes import Recipe, default_user_spec, list_recipes, load_recipe
from humanoid_training.runner import run_job

GOLD_CLIP = "eval.mp4"
GOLD_NOTES = "notes.md"
GOLD_FILES = (GOLD_CLIP, GOLD_NOTES)
_COMPARE_SIZE = 96
_MAX_MEAN_ABS = 28.0
_DURATION_LO = 0.45
_DURATION_HI = 2.2


def cpu_recipes() -> list[Recipe]:
    """Recipes that must ship a gold eval clip (availability: cpu)."""
    return [r for r in list_recipes() if r.as_public_dict()["availability"] == "cpu"]


def gpu_recipes() -> list[Recipe]:
    return [r for r in list_recipes() if r.as_public_dict()["availability"] != "cpu"]


def gold_dir(recipe: Recipe | str, root: Path | None = None) -> Path:
    rec = load_recipe(recipe) if isinstance(recipe, str) else recipe
    if root is not None:
        return Path(root) / rec.id / "gold"
    return rec.path / "gold"


def gold_clip_path(recipe: Recipe | str, root: Path | None = None) -> Path:
    return gold_dir(recipe, root=root) / GOLD_CLIP


def ffmpeg_bin() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise AdapterUnavailable("ffmpeg is not on PATH. Install ffmpeg to record or compare gold clips.")
    return path


def ffprobe_bin() -> str:
    path = shutil.which("ffprobe")
    if not path:
        raise AdapterUnavailable("ffprobe is not on PATH. Install ffmpeg (it includes ffprobe).")
    return path


def probe_video(path: Path) -> dict[str, Any]:
    """Machine-readable video facts. Fail closed if the file is not a video."""
    dest = Path(path)
    if not dest.is_file():
        raise FileNotFoundError(f"missing video {dest}")
    raw = subprocess.run(
        [
            ffprobe_bin(),
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(dest),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if raw.returncode != 0:
        err = (raw.stderr or raw.stdout or "ffprobe failed").strip()
        raise AdapterUnavailable(f"not a video ({dest.name}): {err[:300]}")
    data = json.loads(raw.stdout or "{}")
    streams = [s for s in (data.get("streams") or []) if s.get("codec_type") == "video"]
    if not streams:
        raise AdapterUnavailable(f"{dest} has no video stream")
    video = streams[0]
    duration = _as_float(video.get("duration")) or _as_float((data.get("format") or {}).get("duration"))
    fps = _fps(video.get("avg_frame_rate") or video.get("r_frame_rate"))
    nframes = _as_int(video.get("nb_frames"))
    if duration is None and nframes and fps:
        duration = nframes / fps
    width = _as_int(video.get("width"))
    height = _as_int(video.get("height"))
    if not width or not height:
        raise AdapterUnavailable(f"{dest} video has no width/height")
    if duration is None or duration <= 0:
        raise AdapterUnavailable(f"{dest} video has no duration")
    return {
        "path": str(dest),
        "width": width,
        "height": height,
        "duration": float(duration),
        "codec": video.get("codec_name"),
        "nb_frames": nframes,
        "fps": fps,
        "size": dest.stat().st_size,
    }


def extract_rgb_frame(path: Path, when: float, *, width: int = _COMPARE_SIZE, height: int = _COMPARE_SIZE) -> np.ndarray:
    """One decoded frame, scaled. `when` is seconds from start."""
    t = max(0.0, float(when))
    proc = subprocess.run(
        [
            ffmpeg_bin(),
            "-v",
            "error",
            "-ss",
            f"{t:.3f}",
            "-i",
            str(path),
            "-frames:v",
            "1",
            "-vf",
            f"scale={int(width)}:{int(height)}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        check=False,
        capture_output=True,
    )
    if proc.returncode != 0 or not proc.stdout:
        err = (proc.stderr or b"").decode("utf-8", "replace").strip()
        raise AdapterUnavailable(f"could not decode a frame from {path.name}: {err[:300]}")
    arr = np.frombuffer(proc.stdout, dtype=np.uint8)
    expected = int(width) * int(height) * 3
    if arr.size != expected:
        raise AdapterUnavailable(
            f"decoded {arr.size} bytes from {path.name}, expected {expected}"
        )
    return arr.reshape(int(height), int(width), 3)


@dataclass(frozen=True)
class CompareResult:
    ok: bool
    reason: str
    mean_abs: float | None = None
    duration_a: float | None = None
    duration_b: float | None = None
    size_a: tuple[int, int] | None = None
    size_b: tuple[int, int] | None = None


def compare_eval_videos(produced: Path, gold: Path) -> CompareResult:
    """Coarse check: same-ish duration and downscaled frames. Not a learned judge."""
    try:
        a = probe_video(produced)
        b = probe_video(gold)
    except (FileNotFoundError, AdapterUnavailable, json.JSONDecodeError) as err:
        return CompareResult(False, str(err))
    if a["width"] != b["width"] or a["height"] != b["height"]:
        return CompareResult(
            False,
            f"frame size {a['width']}x{a['height']} vs gold {b['width']}x{b['height']}",
            duration_a=a["duration"],
            duration_b=b["duration"],
            size_a=(a["width"], a["height"]),
            size_b=(b["width"], b["height"]),
        )
    ratio = a["duration"] / b["duration"] if b["duration"] else 0.0
    if ratio < _DURATION_LO or ratio > _DURATION_HI:
        return CompareResult(
            False,
            f"duration {a['duration']:.2f}s vs gold {b['duration']:.2f}s (ratio {ratio:.2f})",
            duration_a=a["duration"],
            duration_b=b["duration"],
            size_a=(a["width"], a["height"]),
            size_b=(b["width"], b["height"]),
        )
    diffs: list[float] = []
    for frac in (0.15, 0.5, 0.85):
        ta = min(max(a["duration"] * frac, 0.0), max(a["duration"] - 0.05, 0.0))
        tb = min(max(b["duration"] * frac, 0.0), max(b["duration"] - 0.05, 0.0))
        fa = extract_rgb_frame(produced, ta)
        fb = extract_rgb_frame(gold, tb)
        diffs.append(float(np.mean(np.abs(fa.astype(np.float32) - fb.astype(np.float32)))))
    mean_abs = float(np.mean(diffs)) if diffs else 255.0
    if mean_abs > _MAX_MEAN_ABS:
        return CompareResult(
            False,
            f"frames differ too much (mean abs {mean_abs:.1f} > {_MAX_MEAN_ABS})",
            mean_abs=mean_abs,
            duration_a=a["duration"],
            duration_b=b["duration"],
            size_a=(a["width"], a["height"]),
            size_b=(b["width"], b["height"]),
        )
    return CompareResult(
        True,
        f"ok mean_abs={mean_abs:.1f} duration={a['duration']:.2f}s",
        mean_abs=mean_abs,
        duration_a=a["duration"],
        duration_b=b["duration"],
        size_a=(a["width"], a["height"]),
        size_b=(b["width"], b["height"]),
    )


def record_gold(
    recipe: Recipe | str,
    *,
    dest_root: Path | None = None,
    runs_dir: Path | None = None,
    log=None,
) -> dict[str, Any]:
    """Train the beginner spec and copy eval.mp4 into the recipe gold folder."""
    if os.environ.get("HT_NO_RENDER") == "1":
        raise AdapterUnavailable(
            "HT_NO_RENDER=1 is set. Unset it (and give MuJoCo a display or xvfb) to record gold clips."
        )
    rec = load_recipe(recipe) if isinstance(recipe, str) else recipe
    public = rec.as_public_dict()
    if public["availability"] != "cpu":
        raise RecipeError(
            f"{rec.id} is availability={public['availability']}. Gold eval.mp4 is only for CPU recipes."
        )
    spec = default_user_spec(rec.id)
    tmp_created = runs_dir is None
    tmp_parent = Path(runs_dir) if runs_dir is not None else Path(tempfile.mkdtemp(prefix="ht-gold-"))
    tmp_parent.mkdir(parents=True, exist_ok=True)
    emit = log or (lambda _m: None)
    emit(f"recording gold for {rec.id}")
    try:
        manifest = run_job(spec, runs_dir=tmp_parent, log=log)
        status = manifest.get("status")
        if status not in {"passed", "completed"}:
            raise AdapterUnavailable(
                f"gold record for {rec.id} did not finish (status={status}): {manifest.get('error')}"
            )
        run_dir = Path(manifest["run_dir"])
        video = run_dir / GOLD_CLIP
        if not video.is_file():
            raise AdapterUnavailable(
                f"gold record for {rec.id} wrote no eval.mp4. Need ffmpeg and a display (or xvfb)."
            )
        dest = gold_dir(rec, root=dest_root)
        dest.mkdir(parents=True, exist_ok=True)
        clip = dest / GOLD_CLIP
        shutil.copy2(video, clip)
        notes = _notes_text(rec, manifest)
        (dest / GOLD_NOTES).write_text(notes, encoding="utf-8")
        probe = probe_video(clip)
        emit(f"wrote {clip} ({probe['width']}x{probe['height']} {probe['duration']:.2f}s)")
        return {
            "ok": True,
            "recipe": rec.id,
            "path": str(clip),
            "notes": str(dest / GOLD_NOTES),
            "probe": probe,
            "facts": dict(manifest.get("facts") or {}),
            "status": status,
            "passed": bool((manifest.get("metrics") or {}).get("passed")),
            "run_id": manifest.get("run_id"),
        }
    finally:
        if tmp_created:
            shutil.rmtree(tmp_parent, ignore_errors=True)


def record_all_gold(*, dest_root: Path | None = None, log=None) -> list[dict[str, Any]]:
    rows = []
    for rec in cpu_recipes():
        rows.append(record_gold(rec, dest_root=dest_root, log=log))
    return rows


def _notes_text(recipe: Recipe, manifest: dict[str, Any]) -> str:
    public = recipe.as_public_dict()
    facts = dict(manifest.get("facts") or {})
    kind = facts.get("kind") or public.get("method") or ""
    promise = (public.get("promise") or recipe.summary or recipe.title).strip()
    first = promise.split("\n", 1)[0].strip()
    extra = []
    if kind:
        extra.append(f"kind={kind}")
    if facts.get("policy"):
        extra.append(f"policy={facts['policy']}")
    if facts.get("arm_mode"):
        extra.append(f"arm_mode={facts['arm_mode']}")
    passed = (manifest.get("metrics") or {}).get("passed")
    extra.append(f"passed={passed}")
    return f"{first} {' '.join(extra)}\n"


def _as_float(value: Any) -> float | None:
    try:
        if value is None or value == "N/A":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    try:
        if value is None or value == "N/A":
            return None
        return int(str(value).split(".")[0])
    except (TypeError, ValueError):
        return None


def _fps(rate: Any) -> float | None:
    text = str(rate or "")
    if "/" in text:
        num, den = text.split("/", 1)
        try:
            d = float(den)
            return float(num) / d if d else None
        except ValueError:
            return None
    return _as_float(text)
