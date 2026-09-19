from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path

import numpy as np

LogFn = Callable[[str], None]


def write_eval_video(
    frames: list[np.ndarray],
    dest: Path,
    fps: int = 30,
    log: LogFn | None = None,
) -> Path | None:
    if not frames:
        if log:
            log("no frames captured; skipping video")
        return None
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        if log:
            log("ffmpeg not on PATH; skipping eval.mp4")
        return None

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    frame_dir = dest.parent / "eval_frames"
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True)

    for i, frame in enumerate(frames):
        rgb = np.asarray(frame)
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] not in (3, 4):
            continue
        if rgb.shape[2] == 4:
            rgb = rgb[:, :, :3]
        _write_ppm(frame_dir / f"{i:05d}.ppm", rgb)

    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        str(fps),
        "-i",
        str(frame_dir / "%05d.ppm"),
        "-vf",
        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-pix_fmt",
        "yuv420p",
        str(dest),
    ]
    subprocess.run(cmd, check=True)
    if log:
        log(f"wrote {dest} ({len(frames)} frames)")
    shutil.rmtree(frame_dir, ignore_errors=True)
    return dest


def _write_ppm(path: Path, rgb: np.ndarray) -> None:
    height, width, _ = rgb.shape
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    path.write_bytes(header + rgb.tobytes())
