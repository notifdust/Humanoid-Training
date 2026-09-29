from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from humanoid_training import hardware
from humanoid_training.adapters.base import EnginePayload, EvalResult, LogFn, Support
from humanoid_training.adapters.common import ignored_scene_fields, recipe_adapter_config
from humanoid_training.adapters.process import harvest_mp4, stream_process, write_job
from humanoid_training.errors import AdapterUnavailable

ISAAC_G1_FLAT = "Isaac-Velocity-Flat-G1-v0"
DEFAULT_IMAGE = "nvcr.io/nvidia/isaac-lab:2.2.0"
LOGDIR_NAME = "isaac_logs"
WORKFLOW_RSL = "rsl_rl"
WORKFLOW_ROBOMIMIC = "robomimic"


def _workflow(spec: dict[str, Any]) -> str:
    cfg = recipe_adapter_config(spec, "isaaclab")
    raw = str(cfg.get("workflow") or WORKFLOW_RSL).strip().lower()
    if raw in {"imitation", "mimic", "bc", "robomimic"}:
        return WORKFLOW_ROBOMIMIC
    return WORKFLOW_RSL


def _algo(spec: dict[str, Any]) -> str:
    cfg = recipe_adapter_config(spec, "isaaclab")
    return str(cfg.get("algo") or (spec.get("train") or {}).get("policy") or "bc")


def _task_image(spec: dict[str, Any], payload: EnginePayload | None = None) -> tuple[str, str]:
    cfg = recipe_adapter_config(spec, "isaaclab")
    extra = (payload.extra if payload else {}) or {}
    task = str((payload.env_name if payload else None) or cfg.get("task") or ISAAC_G1_FLAT)
    image = str(cfg.get("image") or extra.get("image") or DEFAULT_IMAGE)
    return task, image


def resolve_isaac_dataset(spec: dict[str, Any], run_dir: Path | None = None) -> Path | None:
    """Mimic/robomimic hdf5 — operator-supplied; never invent a dataset."""
    raw = os.environ.get("HT_ISAAC_DATASET")
    if raw and str(raw).strip() and str(raw).strip().lower() not in {"0", "false", "no", "off"}:
        path = Path(str(raw).strip()).expanduser()
        if path.is_file():
            return path
    data = spec.get("data") or {}
    for key in ("uri", "dataset", "path"):
        val = data.get(key)
        if not val:
            continue
        text = str(val).strip()
        if text.startswith("file:"):
            text = text[5:]
        path = Path(text).expanduser()
        if not path.is_absolute() and run_dir is not None:
            path = Path(run_dir) / path
        if path.is_file():
            return path
    if run_dir is not None:
        for cand in Path(run_dir).glob("*.hdf5"):
            if cand.is_file():
                return cand
    return None


def isaac_train_play_commands(task: str, cli: str, *, want_video: bool) -> tuple[list[str], list[str]]:
    train = [cli, "-p", "scripts/reinforcement_learning/rsl_rl/train.py", "--task", task, "--headless"]
    play = [cli, "-p", "scripts/reinforcement_learning/rsl_rl/play.py", "--task", task, "--headless"]
    if want_video:
        train.append("--video")
        play.extend(["--video", "--video_length", "200"])
    return train, play


def isaac_robomimic_commands(
    task: str,
    cli: str,
    dataset: Path,
    *,
    algo: str,
    checkpoint: Path | None = None,
) -> tuple[list[str], list[str] | None]:
    train = [
        cli,
        "-p",
        "scripts/imitation_learning/robomimic/train.py",
        "--task",
        task,
        "--algo",
        algo,
        "--normalize_training_actions",
        "--dataset",
        str(dataset),
    ]
    play = None
    if checkpoint is not None:
        play = [
            cli,
            "-p",
            "scripts/imitation_learning/robomimic/play.py",
            "--task",
            task,
            "--checkpoint",
            str(checkpoint),
        ]
    return train, play


def _robomimic_blocked(task: str) -> str:
    return "\n".join(
        [
            "Isaac G1 PickPlace (Robomimic) is blocked — not a silent failure.",
            f"Pinned task: {task}",
            "This is Mimic / Robomimic BC, not rsl_rl PPO.",
            "",
            "On a GPU box with Isaac Lab:",
            "  1. Record or download a Mimic hdf5 for this task",
            "  2. export HT_ISAAC_DATASET=/path/to/dataset.hdf5",
            "  3. ht train spec/examples/g1-pickplace.json",
            "",
            "Record demos (upstream):",
            f"  ./isaaclab.sh -p scripts/tools/record_demos.py --task {task} ...",
            "",
            "CPU mustard→bowl stays on the pick-and-place recipe.",
            "Do not substitute a stand clip.",
        ]
    )


class IsaacLabAdapter:
    """Compile an OSMO workflow. Launch via isaaclab.sh or GPU Docker; else block."""

    name = "isaaclab"

    def support(self, spec: dict[str, Any]) -> Support:
        cfg = recipe_adapter_config(spec, self.name)
        if cfg.get("unsupported"):
            return Support(False, str(cfg["unsupported"]))
        if not cfg.get("task"):
            return Support(False, "recipe does not define adapters.isaaclab.task")
        return Support(True)

    def compile(self, spec: dict[str, Any]) -> EnginePayload:
        task, image = _task_image(spec)
        workflow = _workflow(spec)
        algo = _algo(spec)
        want_video = bool((spec.get("eval") or {}).get("record_video", True))
        if workflow == WORKFLOW_ROBOMIMIC:
            return self._compile_robomimic(spec, task=task, image=image, algo=algo)
        video_train = " --video" if want_video else ""
        video_play = " --video --video_length 200" if want_video else ""
        name = spec.get("name") or "isaac-job"
        workflow_yaml = f"""\
# Generated by Humanoid Training. Submit with: osmo workflow submit osmo_workflow.yaml
# See https://nvidia.github.io/OSMO/main/user_guide/workflows/submission.html
# Local GPU alternative: docker run --gpus all {image}
# Phase 3d harvests /osmo/run/workspace/ht_eval/*.mp4 after COMPLETED.
workflow:
  name: {name}
  tasks:
    - name: train
      image: {image}
      environment:
        ACCEPT_EULA: "Y"
        NO_NUCLEUS: "Y"
      command: ["bash"]
      args: ["/tmp/entry.sh"]
      files:
        - path: /tmp/entry.sh
          contents: |
            set -euo pipefail
            ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \\
              --task {task} --headless{video_train}
            ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \\
              --task {task} --headless{video_play}
            mkdir -p /osmo/run/workspace/ht_eval
            find . -name '*.mp4' -type f -exec cp -n {{}} /osmo/run/workspace/ht_eval/ \\; || true
"""
        train_sh = "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "# Generated by Humanoid Training. Run inside Isaac Lab (or the NGC image).",
                f"# task={task} workflow={WORKFLOW_RSL}",
                f'./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task {task} --headless{video_train}',
                f'./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task {task} --headless{video_play}',
                "# When this script runs in GPU Docker, copy clips into the mounted run dir.",
                "if [ -d /ht_run ]; then",
                "  mkdir -p /ht_run/isaac_logs",
                "  find . -name '*.mp4' -type f -exec cp -n {} /ht_run/isaac_logs/ \\; || true",
                "fi",
                "",
            ]
        )
        train_cmd = [
            "./isaaclab.sh",
            "-p",
            "scripts/reinforcement_learning/rsl_rl/train.py",
            "--task",
            task,
            "--headless",
        ]
        if want_video:
            train_cmd.append("--video")
        payload = EnginePayload(
            adapter=self.name,
            env_name=task,
            command=train_cmd,
            ignored_fields=ignored_scene_fields(spec, honors_scene=False),
            notes=[
                "Isaac Lab GPU training. launch() uses isaaclab.sh, GPU Docker, or OSMO harvest.",
                f"Task id: {task}",
                f"Workflow: {WORKFLOW_RSL}",
                "OSMO: submit → poll COMPLETED → rsync /osmo/run/workspace/ht_eval/*.mp4",
            ],
            extra={"task": task, "image": image, "workflow": WORKFLOW_RSL},
        )
        payload.files["engine_payload.json"] = json.dumps(payload.as_dict(), indent=2)
        payload.files["osmo_workflow.yaml"] = workflow_yaml
        payload.files["train_isaac.sh"] = train_sh
        return payload

    def _compile_robomimic(
        self, spec: dict[str, Any], *, task: str, image: str, algo: str
    ) -> EnginePayload:
        train_sh = "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                "# Generated by Humanoid Training — Robomimic BC (not rsl_rl).",
                f"# task={task} workflow={WORKFLOW_ROBOMIMIC} algo={algo}",
                'DATASET="${HT_ISAAC_DATASET:?set HT_ISAAC_DATASET to a Mimic/robomimic hdf5}"',
                f'./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py --task {task} '
                f'--algo {algo} --normalize_training_actions --dataset "$DATASET"',
                "# Optional play: set HT_ISAAC_CHECKPOINT to a .pth from training.",
                'if [ -n "${HT_ISAAC_CHECKPOINT:-}" ]; then',
                f'  ./isaaclab.sh -p scripts/imitation_learning/robomimic/play.py --task {task} '
                '--checkpoint "$HT_ISAAC_CHECKPOINT"',
                "fi",
                "if [ -d /ht_run ]; then",
                "  mkdir -p /ht_run/isaac_logs",
                "  find . -name '*.mp4' -type f -exec cp -n {} /ht_run/isaac_logs/ \\; || true",
                "  find . -name '*.pth' -type f -exec cp -n {} /ht_run/isaac_logs/ \\; || true",
                "fi",
                "",
            ]
        )
        train_cmd = [
            "./isaaclab.sh",
            "-p",
            "scripts/imitation_learning/robomimic/train.py",
            "--task",
            task,
            "--algo",
            algo,
            "--normalize_training_actions",
            "--dataset",
            "${HT_ISAAC_DATASET}",
        ]
        payload = EnginePayload(
            adapter=self.name,
            env_name=task,
            command=train_cmd,
            ignored_fields=ignored_scene_fields(spec, honors_scene=False),
            notes=[
                "Isaac Lab Mimic / Robomimic BC — not rsl_rl PPO.",
                f"Task id: {task}",
                f"Workflow: {WORKFLOW_ROBOMIMIC} algo={algo}",
                "Set HT_ISAAC_DATASET to a Mimic hdf5 before launch.",
                "OSMO harvest is not wired for robomimic yet — use local GPU or GPU Docker.",
                "CPU mustard→bowl stays on pick-and-place.",
            ],
            extra={
                "task": task,
                "image": image,
                "workflow": WORKFLOW_ROBOMIMIC,
                "algo": algo,
            },
        )
        payload.files["engine_payload.json"] = json.dumps(payload.as_dict(), indent=2)
        payload.files["train_isaac.sh"] = train_sh
        # No osmo_workflow.yaml — would imply a false rsl_rl / harvest path.
        return payload

    def launch(
        self,
        spec: dict[str, Any],
        payload: EnginePayload,
        run_dir: Path,
        log: LogFn,
    ) -> EvalResult:
        run_dir = Path(run_dir)
        task, image = _task_image(spec, payload)
        want_video = bool((spec.get("eval") or {}).get("record_video", True))
        logdir = run_dir / LOGDIR_NAME
        logdir.mkdir(parents=True, exist_ok=True)
        workflow = _workflow(spec)

        if workflow == WORKFLOW_ROBOMIMIC:
            return self._launch_robomimic(
                spec, run_dir, log, task=task, image=image, logdir=logdir, want_video=want_video
            )

        cli = hardware.isaac_cli()
        if cli and hardware.gpu_available():
            return self._launch_local(spec, run_dir, log, task, cli, logdir, want_video)

        # GPU Docker only when explicitly opted in — do not steal the OSMO path
        # just because `docker` is on PATH (CI runners) and osmo_ready made
        # isaac_launch_ready true.
        if (
            hardware.gpu_available()
            and hardware.docker_gpu_requested()
            and hardware.docker_bin()
        ):
            return self._launch_docker(spec, run_dir, log, task, image, logdir, want_video)

        osmo = hardware.osmo_cli()
        yaml_path = run_dir / "osmo_workflow.yaml"
        if osmo and yaml_path.is_file():
            return self._launch_osmo(spec, run_dir, log, task, logdir, want_video, osmo, yaml_path)

        raise AdapterUnavailable(
            "Isaac Lab execution is blocked on this machine — not a silent failure. "
            "On a GPU box with Isaac Lab:\n"
            "  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py "
            f"--task {task} --headless --video\n"
            "Or submit with OSMO harvest (Phase 3d):\n"
            "  osmo workflow submit osmo_workflow.yaml\n"
            "Or GPU Docker:\n"
            f"  HT_DOCKER_GPU=1 docker run --gpus all {image}\n"
            "Do not use ht train --docker (that is the Phase 1 CPU image). "
            "Or use g1-stand / pick-and-place on the CPU studio. "
            f"Compiled payload: {run_dir}"
        )

    def _launch_robomimic(
        self,
        spec: dict[str, Any],
        run_dir: Path,
        log: LogFn,
        *,
        task: str,
        image: str,
        logdir: Path,
        want_video: bool,
    ) -> EvalResult:
        dataset = resolve_isaac_dataset(spec, run_dir)
        if dataset is None:
            raise AdapterUnavailable(_robomimic_blocked(task))
        algo = _algo(spec)
        cli = hardware.isaac_cli()
        if cli and hardware.gpu_available():
            train, _ = isaac_robomimic_commands(task, cli, dataset, algo=algo)
            log(" ".join(train))
            write_job(
                run_dir,
                {
                    "command": train,
                    "status": "starting",
                    "launch": "isaaclab.sh",
                    "workflow": WORKFLOW_ROBOMIMIC,
                    "dataset": str(dataset),
                },
            )
            rc = stream_process(train, run_dir, log, missing="Could not start isaaclab.sh robomimic train")
            ckpt = _find_checkpoint(run_dir, logdir)
            if rc == 0 and ckpt is not None:
                _, play = isaac_robomimic_commands(
                    task, cli, dataset, algo=algo, checkpoint=ckpt
                )
                if play:
                    log(" ".join(play))
                    rc_play = stream_process(
                        play, run_dir, log, missing="Could not start isaaclab.sh robomimic play"
                    )
                    if rc_play != 0:
                        rc = rc_play
            write_job(
                run_dir,
                {
                    "command": train,
                    "returncode": rc,
                    "status": "exited",
                    "launch": "isaaclab.sh",
                    "workflow": WORKFLOW_ROBOMIMIC,
                    "checkpoint": str(ckpt) if ckpt else None,
                },
            )
            return self._finish(
                spec,
                run_dir,
                log,
                task=task,
                logdir=logdir,
                want_video=want_video,
                rc=rc,
                runner="isaaclab.sh",
                workflow=WORKFLOW_ROBOMIMIC,
                policy=algo,
            )

        if (
            hardware.gpu_available()
            and hardware.docker_gpu_requested()
            and hardware.docker_bin()
        ):
            os.environ.setdefault("HT_ISAAC_DATASET", str(dataset))
            return self._launch_docker(
                spec,
                run_dir,
                log,
                task,
                image,
                logdir,
                want_video,
                workflow=WORKFLOW_ROBOMIMIC,
                policy=algo,
            )

        raise AdapterUnavailable(
            _robomimic_blocked(task)
            + f"\nDataset seen: {dataset}\nCompiled payload: {run_dir}"
        )

    def _launch_osmo(
        self,
        spec: dict[str, Any],
        run_dir: Path,
        log: LogFn,
        task: str,
        logdir: Path,
        want_video: bool,
        osmo: str,
        yaml_path: Path,
    ) -> EvalResult:
        from humanoid_training.adapters.osmo_remote import harvest_osmo_walk

        log("osmo remote harvest (Phase 3d)")
        write_job(run_dir, {"status": "submitting", "launch": "osmo", "workflow": str(yaml_path)})
        try:
            result = harvest_osmo_walk(osmo, yaml_path, run_dir, log)
        except AdapterUnavailable:
            raise
        except Exception as err:
            raise AdapterUnavailable(
                f"OSMO harvest failed: {err}. Not substituting a stand clip. "
                f"Workflow file: {yaml_path}"
            ) from err
        write_job(
            run_dir,
            {
                "status": "exited",
                "launch": "osmo",
                "workflow_id": result.get("workflow_id"),
                "osmo_status": result.get("status"),
            },
        )
        facts: dict[str, Any] = {
            "kind": "rl",
            "engine": "isaaclab",
            "device": "remote",
            "env": task,
            "policy": "ppo",
            "launch": "osmo",
            "workflow_id": result.get("workflow_id"),
            "osmo_status": result.get("status"),
        }
        dest = run_dir / "eval.mp4"
        video = None
        remote_video = result.get("video")
        if want_video and remote_video and Path(remote_video).is_file():
            video = harvest_mp4(Path(remote_video).parent, dest)
            if video is None:
                from shutil import copy2

                dest.parent.mkdir(parents=True, exist_ok=True)
                copy2(remote_video, dest)
                video = dest
        if want_video and video is None:
            staging = result.get("staging")
            if staging:
                video = harvest_mp4(Path(staging), dest)
            if video is None:
                video = harvest_mp4(logdir, dest)
        if want_video and video is None:
            raise AdapterUnavailable(
                "OSMO workflow completed but no *.mp4 was harvested. "
                "Not substituting a stand clip. "
                f"workflow_id={result.get('workflow_id')} run_dir={run_dir}"
            )
        notes = [
            f"Isaac Lab task={task} launch=osmo workflow_id={result.get('workflow_id')}",
            "Eval clip harvested from OSMO ht_eval — not a local GPU train.",
        ]
        if video:
            notes.append(f"harvested {video.name}")
            log(f"harvested {video}")
        return EvalResult(
            success_rate=1.0,
            mean_return=0.0,
            episodes=int((spec.get("eval") or {}).get("episodes") or 1),
            video_path=video,
            passed=True,
            notes=notes,
            facts=facts,
        )

    def _finish(
        self,
        spec: dict[str, Any],
        run_dir: Path,
        log: LogFn,
        *,
        task: str,
        logdir: Path,
        want_video: bool,
        rc: int,
        runner: str,
        workflow: str = WORKFLOW_RSL,
        policy: str | None = None,
    ) -> EvalResult:
        kind = "imitation" if workflow == WORKFLOW_ROBOMIMIC else "rl"
        pol = policy or ("bc" if workflow == WORKFLOW_ROBOMIMIC else "ppo")
        facts: dict[str, Any] = {
            "kind": kind,
            "engine": "isaaclab",
            "device": "gpu",
            "env": task,
            "policy": pol,
            "workflow": workflow,
            "returncode": rc,
            "launch": runner,
        }
        if rc != 0:
            return EvalResult(
                success_rate=0.0,
                mean_return=0.0,
                episodes=int((spec.get("eval") or {}).get("episodes") or 0),
                video_path=None,
                passed=False,
                notes=[f"Isaac Lab exited {rc}. Not substituting a stand clip."],
                facts=facts,
            )
        dest = run_dir / "eval.mp4"
        video = harvest_mp4(logdir, dest) if want_video else None
        if want_video and video is None:
            # Train/play write logs/rsl_rl relative to cwd (run_dir).
            video = harvest_mp4(run_dir, dest)
        if want_video and video is None:
            raise AdapterUnavailable(
                "Isaac Lab finished (exit 0) but wrote no *.mp4. "
                "Not substituting a stand clip. "
                f"Run dir: {run_dir}"
            )
        notes = [
            f"Isaac Lab task={task} launch={runner} workflow={workflow}",
            (
                "Robomimic BC eval clip — not CPU linear-BC mustard."
                if workflow == WORKFLOW_ROBOMIMIC
                else "Eval clip is the engine rollout, not an independent reach/upright score."
            ),
        ]
        if video:
            notes.append(f"harvested {video.name}")
            log(f"harvested {video}")
        return EvalResult(
            success_rate=1.0,
            mean_return=0.0,
            episodes=int((spec.get("eval") or {}).get("episodes") or 1),
            video_path=video,
            passed=True,
            notes=notes,
            facts=facts,
        )

    def _launch_local(
        self,
        spec: dict[str, Any],
        run_dir: Path,
        log: LogFn,
        task: str,
        cli: str,
        logdir: Path,
        want_video: bool,
    ) -> EvalResult:
        train, play = isaac_train_play_commands(task, cli, want_video=want_video)
        log(" ".join(train))
        write_job(run_dir, {"command": train, "status": "starting", "launch": "isaaclab.sh"})
        rc = stream_process(train, run_dir, log, missing="Could not start isaaclab.sh")
        if rc == 0:
            log(" ".join(play))
            rc_play = stream_process(play, run_dir, log, missing="Could not start isaaclab.sh play")
            if rc_play != 0:
                rc = rc_play
        write_job(run_dir, {"command": train + play, "returncode": rc, "status": "exited", "launch": "isaaclab.sh"})
        return self._finish(
            spec, run_dir, log, task=task, logdir=logdir, want_video=want_video, rc=rc, runner="isaaclab.sh"
        )

    def _launch_docker(
        self,
        spec: dict[str, Any],
        run_dir: Path,
        log: LogFn,
        task: str,
        image: str,
        logdir: Path,
        want_video: bool,
        *,
        workflow: str = WORKFLOW_RSL,
        policy: str | None = None,
    ) -> EvalResult:
        docker = hardware.docker_bin()
        if not docker:
            raise AdapterUnavailable("Docker is not on PATH for Isaac GPU launch.")
        script = run_dir / "train_isaac.sh"
        if not script.is_file():
            raise AdapterUnavailable(
                "train_isaac.sh is missing from the run dir; compile the Isaac payload first. "
                f"Run dir: {run_dir}"
            )
        cmd = [
            docker,
            "run",
            "--rm",
            "--gpus",
            "all",
            "-v",
            f"{run_dir}:/ht_run",
            "-e",
            "ACCEPT_EULA=Y",
            "-e",
            "NO_NUCLEUS=Y",
        ]
        dataset = resolve_isaac_dataset(spec, run_dir)
        if dataset is not None:
            cmd.extend(
                [
                    "-v",
                    f"{dataset}:/ht_dataset.hdf5:ro",
                    "-e",
                    "HT_ISAAC_DATASET=/ht_dataset.hdf5",
                ]
            )
        cmd.extend([image, "bash", "/ht_run/train_isaac.sh"])
        log(" ".join(cmd))
        write_job(run_dir, {"command": cmd, "image": image, "status": "starting", "launch": "docker"})
        rc = stream_process(cmd, run_dir, log, missing="Could not start docker for Isaac Lab")
        write_job(
            run_dir,
            {"command": cmd, "image": image, "returncode": rc, "status": "exited", "launch": "docker"},
        )
        return self._finish(
            spec,
            run_dir,
            log,
            task=task,
            logdir=logdir,
            want_video=want_video,
            rc=rc,
            runner="docker",
            workflow=workflow,
            policy=policy,
        )


def _find_checkpoint(run_dir: Path, logdir: Path) -> Path | None:
    candidates: list[Path] = []
    for root in (logdir, run_dir):
        if not root.is_dir():
            continue
        candidates.extend(root.rglob("*.pth"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]
