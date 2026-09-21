from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from humanoid_training import hardware
from humanoid_training.adapters.base import EnginePayload, EvalResult, LogFn, Support
from humanoid_training.adapters.common import ignored_scene_fields, recipe_adapter_config
from humanoid_training.errors import AdapterUnavailable


PLAYGROUND_ENV = "G1JoystickFlatTerrain"
LOGDIR_NAME = "playground_logs"


def apply_flag(argv: list[str], flag: str, value: str) -> list[str]:
    """Set `--flag value`, replacing the existing value when the flag is present."""
    out = list(argv)
    if flag in out:
        i = out.index(flag)
        if i + 1 < len(out) and not str(out[i + 1]).startswith("-"):
            out[i + 1] = str(value)
        else:
            out.insert(i + 1, str(value))
        return out
    return out + [flag, str(value)]


def harvest_rollout_mp4(logdir: Path, dest: Path) -> Path | None:
    """Copy Playground's `rollout*.mp4` to dest. Prefer `rollout0.mp4`."""
    logdir = Path(logdir)
    if not logdir.is_dir():
        return None
    videos = sorted(p for p in logdir.rglob("rollout*.mp4") if p.is_file())
    if not videos:
        return None
    preferred = [p for p in videos if p.name == "rollout0.mp4"]
    src = preferred[0] if preferred else videos[0]
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dest.resolve():
        shutil.copy2(src, dest)
    return dest


def _train_hparams(spec: dict[str, Any], payload: EnginePayload | None = None) -> tuple[str, int, int, int]:
    cfg = recipe_adapter_config(spec, "playground")
    extra = (payload.extra if payload else {}) or {}
    env_name = str(
        (payload.env_name if payload else None) or cfg.get("env_name") or PLAYGROUND_ENV
    )
    train = spec.get("train") or {}
    steps = int(train.get("steps") or extra.get("steps") or 100_000)
    seed = int(train.get("seed") or extra.get("seed") or 1)
    eval_cfg = spec.get("eval") or {}
    want_video = bool(eval_cfg.get("record_video", True))
    num_videos = 1 if want_video else 0
    return env_name, steps, seed, num_videos


def playground_train_command(
    spec: dict[str, Any],
    *,
    payload: EnginePayload | None = None,
    cli: str = "train-jax-ppo",
    logdir: str | Path = LOGDIR_NAME,
) -> list[str]:
    cfg = recipe_adapter_config(spec, "playground")
    env_name, steps, seed, num_videos = _train_hparams(spec, payload)
    argv = list(
        cfg.get("train_command")
        or (payload.command if payload and payload.command else None)
        or ["train-jax-ppo", "--env_name", env_name]
    )
    if argv:
        argv[0] = cli
    argv = apply_flag(argv, "--env_name", env_name)
    argv = apply_flag(argv, "--seed", str(seed))
    argv = apply_flag(argv, "--num_timesteps", str(steps))
    argv = apply_flag(argv, "--num_videos", str(num_videos))
    argv = apply_flag(argv, "--logdir", str(logdir))
    return argv


def _write_job(run_dir: Path, payload: dict[str, Any]) -> None:
    path = run_dir / "job.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _stream_process(command: list[str], run_dir: Path, log: LogFn) -> int:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    try:
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(run_dir),
            env=env,
            text=True,
            bufsize=1,
        )
    except OSError as err:
        raise AdapterUnavailable(
            f"Could not start train-jax-ppo ({err}). "
            "Install MuJoCo Playground: pip install playground. "
            f"Compiled payloads are in {run_dir}"
        ) from err
    _write_job(
        run_dir,
        {"command": command, "pid": proc.pid, "cwd": str(run_dir), "status": "running"},
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        log(line.rstrip("\n"))
    return int(proc.wait())


def _blocked_no_cli(run_dir: Path) -> AdapterUnavailable:
    if hardware.playground_installed():
        return AdapterUnavailable(
            "MuJoCo Playground is installed but train-jax-ppo is not on PATH. "
            "Reinstall with: pip install playground\n"
            "Then: ht train spec/examples/g1-walk.json\n"
            "Or open g1-stand for a CPU hold preview. "
            f"Compiled payloads are in {run_dir}"
        )
    return AdapterUnavailable(
        "G1 walk is blocked on this machine: MuJoCo Playground is not installed. "
        "This is expected on the CPU studio — not a silent failure. "
        "On a machine with an NVIDIA GPU:\n"
        "  pip install playground\n"
        "  ht train spec/examples/g1-walk.json\n"
        "Or open g1-stand for a CPU hold preview. "
        f"Compiled payloads are in {run_dir}"
    )


def _blocked_no_gpu(run_dir: Path) -> AdapterUnavailable:
    return AdapterUnavailable(
        "G1 walk is blocked on this machine: no NVIDIA GPU. "
        "Playground PPO needs CUDA. This is expected on the CPU studio — not a silent failure. "
        "On a machine with an NVIDIA GPU:\n"
        "  pip install playground\n"
        "  ht train spec/examples/g1-walk.json\n"
        "Or open g1-stand for a CPU hold preview. "
        f"Compiled payloads are in {run_dir}"
    )


class PlaygroundAdapter:
    """Compile and, on a GPU box with Playground, launch G1JoystickFlatTerrain PPO."""

    name = "playground"

    def support(self, spec: dict[str, Any]) -> Support:
        cfg = recipe_adapter_config(spec, self.name)
        if cfg.get("unsupported"):
            return Support(False, str(cfg["unsupported"]))
        if not cfg.get("env_name"):
            return Support(False, "recipe does not define adapters.playground.env_name")
        return Support(True)

    def compile(self, spec: dict[str, Any]) -> EnginePayload:
        env_name, steps, seed, num_videos = _train_hparams(spec)
        command = playground_train_command(spec, cli="train-jax-ppo", logdir=LOGDIR_NAME)
        script = "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "# Generated by Humanoid Training. Requires: pip install playground + NVIDIA GPU",
                f"# env={env_name} seed={seed} steps={steps} videos={num_videos}",
                "if ! command -v train-jax-ppo >/dev/null 2>&1; then",
                '  echo "MuJoCo Playground is not on PATH. Install with: pip install playground" >&2',
                "  exit 12",
                "fi",
                "exec " + " ".join(command),
                "",
            ]
        )
        payload = EnginePayload(
            adapter=self.name,
            env_name=env_name,
            command=command,
            ignored_fields=ignored_scene_fields(spec, honors_scene=False),
            notes=[
                "MuJoCo Playground / MJX. Needs an NVIDIA GPU for a real G1 train.",
                "launch() runs train-jax-ppo when Playground and a GPU are present; "
                "otherwise it compiles this payload and blocks.",
            ],
            extra={
                "steps": steps,
                "seed": seed,
                "num_videos": num_videos,
                "logdir": LOGDIR_NAME,
            },
        )
        payload.files["engine_payload.json"] = json.dumps(payload.as_dict(), indent=2)
        payload.files["train.sh"] = script
        return payload

    def launch(
        self,
        spec: dict[str, Any],
        payload: EnginePayload,
        run_dir: Path,
        log: LogFn,
    ) -> EvalResult:
        run_dir = Path(run_dir)
        log("checking for MuJoCo Playground")
        cli = hardware.playground_cli()
        if not cli:
            raise _blocked_no_cli(run_dir)
        if not hardware.gpu_available():
            raise _blocked_no_gpu(run_dir)

        logdir = run_dir / LOGDIR_NAME
        logdir.mkdir(parents=True, exist_ok=True)
        command = playground_train_command(
            spec, payload=payload, cli=cli, logdir=logdir
        )
        env_name, steps, seed, num_videos = _train_hparams(spec, payload)
        log(" ".join(command))
        _write_job(
            run_dir,
            {
                "command": command,
                "logdir": str(logdir),
                "env_name": env_name,
                "status": "starting",
            },
        )
        rc = _stream_process(command, run_dir, log)
        _write_job(
            run_dir,
            {
                "command": command,
                "logdir": str(logdir),
                "env_name": env_name,
                "returncode": rc,
                "status": "exited",
            },
        )
        facts: dict[str, Any] = {
            "kind": "rl",
            "engine": "playground",
            "device": "gpu",
            "env": env_name,
            "policy": "ppo",
            "seed": seed,
            "num_timesteps": steps,
            "num_videos": num_videos,
            "returncode": rc,
        }
        want_video = num_videos > 0
        if rc != 0:
            log(f"train-jax-ppo exited {rc}; not harvesting a walk clip")
            return EvalResult(
                success_rate=0.0,
                mean_return=0.0,
                episodes=int((spec.get("eval") or {}).get("episodes") or num_videos or 0),
                video_path=None,
                passed=False,
                notes=[
                    f"Playground train-jax-ppo exited {rc}. Not a walking eval.",
                    "Not substituting a stand clip.",
                ],
                facts=facts,
            )

        dest = run_dir / "eval.mp4"
        video = harvest_rollout_mp4(logdir, dest) if want_video else None
        if want_video and video is None:
            raise AdapterUnavailable(
                "Playground finished (exit 0) but wrote no rollout*.mp4 under "
                f"{logdir}. Not substituting a stand clip. "
                "Check run.log and playground_logs/. "
                f"Run dir: {run_dir}"
            )
        passed = True
        notes = [
            f"Playground PPO env={env_name} steps={steps} seed={seed}",
            "Eval clip is the engine rollout, not an independent upright-duration score.",
        ]
        if video:
            notes.append(f"harvested {video.name} from playground_logs")
            log(f"harvested {video}")
        else:
            notes.append("record_video=false; no eval.mp4")
        return EvalResult(
            success_rate=1.0,
            mean_return=0.0,
            episodes=int((spec.get("eval") or {}).get("episodes") or num_videos or 1),
            video_path=video,
            passed=passed,
            notes=notes,
            facts=facts,
        )
