from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.swe_bench.orchestrator import run_single  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one SWE-bench feasibility case.")
    parser.add_argument("--run-id")
    parser.add_argument(
        "--reuse-gold-from",
        help="Reuse a validated gold result from an earlier run ID.",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()
    run_dir = run_single(
        ROOT / "benchmarks" / "swe_bench" / "cases.json",
        ROOT / "results" / "swe_bench_single" / "runs",
        args.run_id,
        reuse_gold_from=args.reuse_gold_from,
    )
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    print(f"Summary: {run_dir / 'summary.md'}")
    if metrics["model_resolved"] is True:
        raise SystemExit(0)
    if metrics["model_resolved"] is False:
        raise SystemExit(1)
    raise SystemExit(2)


if __name__ == "__main__":
    main()
