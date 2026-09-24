from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from humanoid_training.catalog import load_robot_catalog
from humanoid_training.errors import AdapterError, AdapterUnavailable, RecipeError, SpecError
from humanoid_training.recipes import english_list, expand_spec, list_recipes, public_catalog
from humanoid_training.runner import default_runs_dir, run_job
from humanoid_training.spec import load_spec, validate_spec


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ht",
        description="Humanoid Training: compile a job spec and run it on an engine adapter.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("recipes", help="List recipe packages")
    sub.add_parser("robots", help="List catalog robots")

    p_validate = sub.add_parser("validate", help="Validate a job spec against the schema")
    p_validate.add_argument("spec")

    p_expand = sub.add_parser("expand", help="Print the fully expanded spec as JSON")
    p_expand.add_argument("spec")

    p_train = sub.add_parser("train", help="Expand, compile, and launch a training run")
    p_train.add_argument("spec")
    p_train.add_argument("--out", type=Path, default=None, help="Runs directory")
    p_train.add_argument(
        "--compile-only",
        action="store_true",
        help="Write engine payload files without launching the engine",
    )
    p_train.add_argument(
        "--docker",
        action="store_true",
        help="Launch the same job inside the CPU Docker image (Phase 1). GPU recipes are refused.",
    )

    p_serve = sub.add_parser("serve", help="Run the studio API + UI")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)

    p_fetch = sub.add_parser("fetch-assets", help="Download MuJoCo Menagerie robots into the cache")
    p_fetch.add_argument("robot", nargs="?", default="unitree_g1")

    p_record = sub.add_parser("record", help="Write a LeRobot dataset of scripted demos")
    p_record.add_argument("spec")
    p_record.add_argument("--out", type=Path, required=True, help="Dataset directory")
    p_record.add_argument("--episodes", type=int, default=4)
    p_record.add_argument("--include-failure", action="store_true")

    p_gold = sub.add_parser(
        "gold",
        help="Record gold/eval.mp4 for CPU recipes (what Train should look like)",
    )
    p_gold.add_argument(
        "recipe",
        nargs="?",
        default=None,
        help="Recipe id. Omit to record every availability:cpu recipe.",
    )
    p_gold.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write under this directory instead of recipes/<id>/gold",
    )

    p_proof = sub.add_parser(
        "proof",
        help="Phase 3c: run a short G1 walk on this GPU and require eval.mp4",
    )
    p_proof.add_argument(
        "what",
        nargs="?",
        default="walk",
        choices=["walk"],
        help="What to prove (only walk today)",
    )
    p_proof.add_argument(
        "--prefer",
        nargs="+",
        default=None,
        help="Engine order, e.g. mjlab isaaclab playground",
    )
    p_proof.add_argument("--steps", type=int, default=None, help="Override train steps (default: short proof)")
    p_proof.add_argument("--out", type=Path, default=None, help="Runs directory")

    p_deploy = sub.add_parser(
        "deploy",
        help="Phase 4: attempt hardware deploy for a run (fails closed until a hardware profile passes)",
    )
    p_deploy.add_argument("run_id", help="Run id under the runs directory")
    p_deploy.add_argument("--out", type=Path, default=None, help="Runs directory")

    args = parser.parse_args(argv)
    try:
        if args.cmd == "recipes":
            return _cmd_recipes()
        if args.cmd == "robots":
            return _cmd_robots()
        if args.cmd == "validate":
            return _cmd_validate(args.spec)
        if args.cmd == "expand":
            return _cmd_expand(args.spec)
        if args.cmd == "train":
            return _cmd_train(args.spec, args.out, args.compile_only, args.docker)
        if args.cmd == "serve":
            return _cmd_serve(args.host, args.port)
        if args.cmd == "fetch-assets":
            return _cmd_fetch(args.robot)
        if args.cmd == "record":
            return _cmd_record(args.spec, args.out, args.episodes, args.include_failure)
        if args.cmd == "gold":
            return _cmd_gold(args.recipe, args.out)
        if args.cmd == "proof":
            return _cmd_proof(args.what, args.prefer, args.steps, args.out)
        if args.cmd == "deploy":
            return _cmd_deploy(args.run_id, args.out)
    except (SpecError, RecipeError, AdapterError, FileNotFoundError) as err:
        print(err, file=sys.stderr)
        return 2
    return 1


def _cmd_recipes() -> int:
    rows = list_recipes()
    if not rows:
        print("No recipes found.")
        return 0
    width = max(len(r.id) for r in rows)
    for recipe in rows:
        public = recipe.as_public_dict()
        flag = public["availability"]
        print(f"{recipe.id:<{width}}  {flag:<4}  {recipe.title} — {recipe.summary}")
    return 0


def _cmd_robots() -> int:
    for robot in load_robot_catalog():
        print(f"{robot['id']:24}  {robot.get('name')} ({robot.get('kind')})")
    return 0


def _cmd_validate(path: str) -> int:
    spec = load_spec(path)
    errors = validate_spec(spec)
    if errors:
        print("Invalid:")
        for message in errors:
            print(f"  - {message}")
        return 1
    print("valid")
    return 0


def _cmd_expand(path: str) -> int:
    spec = load_spec(path)
    expanded = expand_spec(spec)
    public = {k: v for k, v in expanded.items() if not str(k).startswith("_")}
    print(json.dumps(public, indent=2))
    return 0


def _cmd_train(path: str, out: Path | None, compile_only: bool, docker: bool = False) -> int:
    spec = load_spec(path)
    try:
        if docker:
            from humanoid_training.docker_runner import run_job_via_docker

            manifest = run_job_via_docker(
                spec,
                runs_dir=out or default_runs_dir(),
                log=print,
                compile_only=compile_only,
            )
        else:
            manifest = run_job(
                spec,
                runs_dir=out or default_runs_dir(),
                log=print,
                compile_only=compile_only,
            )
    except AdapterUnavailable as err:
        print(err, file=sys.stderr)
        return 12
    print(json.dumps({k: manifest[k] for k in ("run_id", "status", "adapter", "metrics", "error") if k in manifest}, indent=2, default=str))
    status = manifest.get("status")
    if status in {"passed", "completed", "compiled"}:
        return 0
    if status == "blocked":
        return 12
    return 1


def _cmd_fetch(robot: str) -> int:
    from humanoid_training.assets import ensure_menagerie_robot

    path = ensure_menagerie_robot(robot, log=print)
    print(path)
    return 0


def _cmd_record(path: str, out: Path, episodes: int, include_failure: bool) -> int:
    from humanoid_training.demos import record_scripted_pick_place

    spec = load_spec(path)
    expanded = expand_spec(spec)
    result = record_scripted_pick_place(
        expanded,
        out,
        episodes=episodes,
        include_failure=include_failure,
        seed=int((expanded.get("train") or {}).get("seed") or 1),
    )
    print(json.dumps({k: result.get(k) for k in ("ok", "path", "total_episodes", "total_frames", "format")}, indent=2))
    return 0 if result.get("ok") else 1


def _cmd_gold(recipe_id: str | None, out: Path | None) -> int:
    from humanoid_training.gold import cpu_recipes, record_all_gold, record_gold

    try:
        if recipe_id:
            rows = [record_gold(recipe_id, dest_root=out, log=print)]
        else:
            ids = ", ".join(r.id for r in cpu_recipes()) or "(none)"
            print(f"recording gold clips for: {ids}")
            rows = record_all_gold(dest_root=out, log=print)
    except AdapterUnavailable as err:
        print(err, file=sys.stderr)
        return 12
    print(json.dumps(rows, indent=2, default=str))
    return 0 if rows and all(r.get("ok") for r in rows) else 1


def _cmd_proof(
    what: str,
    prefer: list[str] | None,
    steps: int | None,
    out: Path | None,
) -> int:
    from humanoid_training.proof import PROOF_STEPS, run_walk_proof

    if what != "walk":
        print(f"Unknown proof target {what!r}. Use: ht proof walk", file=sys.stderr)
        return 2
    try:
        report = run_walk_proof(
            runs_dir=out or default_runs_dir(),
            prefer=prefer,
            steps=int(steps) if steps is not None else PROOF_STEPS,
            log=print,
        )
    except AdapterUnavailable as err:
        print(err, file=sys.stderr)
        return 12
    print(json.dumps(report, indent=2, default=str))
    return 0 if report.get("ok") else 1


def _cmd_deploy(run_id: str, out: Path | None) -> int:
    from humanoid_training.deploy import assess_deploy, deploy_run
    from humanoid_training.runner import load_manifest

    runs_dir = out or default_runs_dir()
    run_dir = runs_dir / run_id
    if run_dir.is_dir():
        report = assess_deploy(load_manifest(run_dir))
        print(json.dumps({k: v for k, v in report.items() if k != "error"}, indent=2, default=str))
    try:
        deploy_run(run_id, runs_dir=runs_dir)
    except AdapterUnavailable as err:
        print(err, file=sys.stderr)
        return 12
    return 0


def _cmd_serve(host: str, port: int) -> int:
    import uvicorn

    open_host = "127.0.0.1" if host in {"0.0.0.0", "::", "[::]"} else host
    catalog = public_catalog()
    ready = [r["title"] for r in catalog["recipes"] if r["id"] in catalog["ready"]]
    later = [r["title"] for r in catalog["recipes"] if r["id"] in catalog["later"]]
    print("Humanoid Training studio")
    print(f"  Open http://{open_host}:{port}")
    if ready:
        print(f"  Click {ready[0]} → Train. You should get a video.")
        if len(ready) > 1:
            print(f"  Then {english_list(ready[1:])}.")
    if later:
        print(f"  Skip {english_list(later)} on this computer (they need a GPU).")
    uvicorn.run("humanoid_training.server:app", host=host, port=port, reload=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
