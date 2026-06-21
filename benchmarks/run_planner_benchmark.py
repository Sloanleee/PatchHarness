from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.metrics.benchmark import (  # noqa: E402
    run_planner_benchmark,
    write_benchmark_csv,
    write_benchmark_summary,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"


def main() -> None:
    rows, summary = run_planner_benchmark()
    write_benchmark_csv(rows, RESULTS_DIR / "planner_benchmark.csv")
    write_benchmark_summary(summary, RESULTS_DIR / "planner_benchmark_summary.md")
    print(f"Wrote {len(rows)} benchmark rows to {RESULTS_DIR / 'planner_benchmark.csv'}")
    print(f"Wrote summary to {RESULTS_DIR / 'planner_benchmark_summary.md'}")
    print(
        "Saved LLM calls: "
        f"{summary.saved_llm_calls}/{summary.fixed_llm_calls} "
        f"({summary.saved_llm_call_rate:.1%})"
    )


if __name__ == "__main__":
    main()

