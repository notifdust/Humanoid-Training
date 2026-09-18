from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from humanoid_training.catalog import load_robot_catalog
from humanoid_training.errors import AdapterError, RecipeError, SpecError
from humanoid_training.recipes import expand_spec, list_recipes
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

    p_serve = sub.add_parser("serve", help="Run the studio API + UI")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)

    p_fetch = sub.add_parser("fetch-assets", help="Download MuJoCo Menagerie robots into the cache")
    p_fetch.add_argument("robot", nargs="?", default="unitree_g1")

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
            return _cmd_train(args.spec, args.out, args.compile_only)
        if args.cmd == "serve":
            return _cmd_serve(args.host, args.port)
        if args.cmd == "fetch-assets":
            return _cmd_fetch(args.robot)
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
        flag = "run" if recipe.data.get("runnable") else "spec"
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


def _cmd_train(path: str, out: Path | None, compile_only: bool) -> int:
    spec = load_spec(path)
    manifest = run_job(
        spec,
        runs_dir=out or default_runs_dir(),
        log=print,
        compile_only=compile_only,
    )
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


def _cmd_serve(host: str, port: int) -> int:
    import uvicorn

    uvicorn.run("humanoid_training.server:app", host=host, port=port, reload=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
