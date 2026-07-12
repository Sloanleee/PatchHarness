from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks.swe_bench.models import load_case
from benchmarks.swe_bench.runner import (
    ensure_git_cache,
    load_instance,
    prepare_workspace,
    run_patchharness,
)


def main() -> None:
    args = _parse_args()
    config = load_case(args.config)
    instance = load_instance(config)
    project_root = Path(__file__).resolve().parents[2]
    cache_name = instance["repo"].replace("/", "__") + ".git"
    git_cache = project_root / "results" / "swe_bench_single" / "cache" / "git" / cache_name
    seed_workspace = (
        project_root / "results" / "swe_bench_single" / "runs"
        / "swe_single_002" / "workspace"
    )
    ensure_git_cache(instance, git_cache, seed_workspace)
    workspace = prepare_workspace(instance, args.workspace, git_cache=git_cache)
    result = run_patchharness(config, instance, workspace)
    _write_json_atomic(args.result, result.to_dict())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one PatchHarness SWE-bench worker.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    return parser.parse_args()


def _write_json_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


if __name__ == "__main__":
    main()
