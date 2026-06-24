from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.graph import BugfixWorkflow  # noqa: E402
from app.llm import MockLLMClient, create_llm_client  # noqa: E402
from app.schemas import BugfixRequest  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
TEMPLATES_DIR = ROOT / "benchmarks" / "real_cases" / "templates"
RESULTS_ROOT = ROOT / "results" / "langgraph_vs_sequential" / "runs"


EXPERIMENT_CASES: list[dict[str, Any]] = [
    {
        "case_id": "review_only",
        "category": "review_only",
        "workspace_template": "calculator_ok",
        "task_description": "review `calculator.py` and report potential issues without editing",
        "mode": "review",
        "allow_edit": False,
        "run_tests": False,
        "test_command": "python -m unittest discover -s tests",
        "enable_llm": False,
        "expected_changed_files": [],
        "expected_test_returncode": None,
        "expected_requires_human_approval": False,
    },
    {
        "case_id": "fix_with_test",
        "category": "fix_with_test",
        "workspace_template": "calculator_bug",
        "task_description": "In `calculator.py` replace `return a - b` with `return a + b`",
        "mode": "fix",
        "allow_edit": True,
        "run_tests": True,
        "test_command": "python -m unittest discover -s tests",
        "enable_llm": False,
        "expected_changed_files": ["calculator.py"],
        "expected_test_returncode": 0,
        "expected_requires_human_approval": False,
    },
    {
        "case_id": "hitl_sensitive_edit",
        "category": "hitl_sensitive_edit",
        "workspace_template": "env_sensitive",
        "task_description": "In `.env` replace `TOKEN=old` with `TOKEN=new`",
        "mode": "fix",
        "allow_edit": True,
        "run_tests": True,
        "test_command": "python -m unittest discover -s tests",
        "enable_llm": False,
        "expected_changed_files": [],
        "expected_test_returncode": None,
        "expected_requires_human_approval": True,
    },
    {
        "case_id": "llm_fix_with_test",
        "category": "llm_fix_with_test",
        "workspace_template": "calculator_bug",
        "task_description": (
            "Analyze and fix the addition bug in calculator.py. "
            "The test expects add(2, 3) to return 5."
        ),
        "mode": "fix",
        "allow_edit": True,
        "run_tests": True,
        "test_command": "python -m unittest discover -s tests",
        "enable_llm": True,
        "expected_changed_files": ["calculator.py"],
        "expected_test_returncode": 0,
        "expected_requires_human_approval": False,
    },
]


@dataclass(slots=True)
class ExperimentResult:
    run_id: str
    case: dict[str, Any]
    strategy: str
    response: dict[str, Any] | None
    elapsed_ms: float
    success: bool
    failure_reason: str
    error: str | None = None


def main() -> None:
    args = _parse_args()
    run_dir = run_experiment(
        provider=args.provider,
        category=args.category,
        max_cases=args.max_cases,
    )
    print(f"Wrote experiment run to {run_dir}")
    print(f"Summary: {run_dir / 'summary.md'}")


def run_experiment(
    provider: str = "mock",
    category: str | None = None,
    max_cases: int = 4,
    results_root: Path = RESULTS_ROOT,
    workspace_base: Path | None = None,
    run_id: str | None = None,
) -> Path:
    selected_cases = _select_cases(EXPERIMENT_CASES, category, max_cases)
    if not selected_cases:
        raise ValueError("No experiment cases selected.")

    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = results_root / run_id
    raw_dir = run_dir / "raw"
    workspace_root = workspace_base or (
        Path(tempfile.gettempdir()) / "patchharness_langgraph_vs_sequential" / run_id
    )
    raw_dir.mkdir(parents=True, exist_ok=True)
    workspace_root.mkdir(parents=True, exist_ok=True)

    sequential_results: list[ExperimentResult] = []
    langgraph_results: list[ExperimentResult] = []
    for case in selected_cases:
        print(f"Running {case['case_id']} ({case['category']})")
        sequential_results.append(
            _run_case(
                run_id=run_id,
                case=case,
                strategy="sequential",
                provider=provider,
                workspace_root=workspace_root,
            )
        )
        langgraph_results.append(
            _run_case(
                run_id=run_id,
                case=case,
                strategy="langgraph",
                provider=provider,
                workspace_root=workspace_root,
            )
        )

    _write_strategy_jsonl(sequential_results, run_dir / "sequential.jsonl")
    _write_strategy_jsonl(langgraph_results, run_dir / "langgraph.jsonl")
    _write_raw(sequential_results + langgraph_results, raw_dir)
    comparison_rows = _build_comparison_rows(run_id, sequential_results, langgraph_results)
    _write_comparison_csv(comparison_rows, run_dir / "comparison.csv")
    _write_summary(provider, comparison_rows, run_dir / "summary.md", workspace_root)
    return run_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare sequential workflow and LangGraph workflow controllability."
    )
    parser.add_argument(
        "--provider",
        default="mock",
        choices=["mock", "ark", "volcengine", "volcengine_ark", "deepseek"],
        help="LLM provider. Mock is deterministic and free; ark/deepseek call real APIs.",
    )
    parser.add_argument("--category", help="Only run one experiment case category.")
    parser.add_argument("--max-cases", type=int, default=4, help="Maximum cases to run.")
    return parser.parse_args()


def _select_cases(
    cases: list[dict[str, Any]],
    category: str | None,
    max_cases: int,
) -> list[dict[str, Any]]:
    selected = [case for case in cases if category is None or case["category"] == category]
    return selected[:max_cases]


def _run_case(
    run_id: str,
    case: dict[str, Any],
    strategy: str,
    provider: str,
    workspace_root: Path,
) -> ExperimentResult:
    workspace = _prepare_workspace(case, strategy, workspace_root)
    request = BugfixRequest(
        task_description=case["task_description"],
        workspace_path=str(workspace),
        mode=case["mode"],
        allow_edit=bool(case["allow_edit"]),
        run_tests=bool(case["run_tests"]),
        test_command=case.get("test_command"),
        enable_llm=bool(case.get("enable_llm", False)),
        use_langgraph=strategy == "langgraph",
    )
    start = time.perf_counter()
    try:
        workflow = _build_workflow(provider, case)
        response = workflow.run(request).to_dict()
        elapsed_ms = (time.perf_counter() - start) * 1000
        success, failure_reason = _evaluate_case(case, response, strategy)
        return ExperimentResult(
            run_id=run_id,
            case=case,
            strategy=strategy,
            response=response,
            elapsed_ms=elapsed_ms,
            success=success,
            failure_reason=failure_reason,
        )
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - start) * 1000
        return ExperimentResult(
            run_id=run_id,
            case=case,
            strategy=strategy,
            response=None,
            elapsed_ms=elapsed_ms,
            success=False,
            failure_reason="llm_timeout" if _is_timeout_error(exc) else "tool_error",
            error=str(exc),
        )


def _prepare_workspace(case: dict[str, Any], strategy: str, workspace_root: Path) -> Path:
    template = TEMPLATES_DIR / str(case["workspace_template"])
    if not template.exists():
        raise ValueError(f"Unknown workspace template: {template}")
    target = workspace_root / strategy / str(case["case_id"])
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(template, target)
    _initialize_workspace_git(target)
    return target


def _initialize_workspace_git(workspace: Path) -> None:
    if shutil.which("git") is None:
        return
    for command in (["git", "init", "-q"], ["git", "add", "-A"]):
        try:
            subprocess.run(
                command,
                cwd=workspace,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return


def _build_workflow(provider: str, case: dict[str, Any]) -> BugfixWorkflow:
    workflow = BugfixWorkflow.from_default_configs()
    if not case.get("enable_llm", False):
        return workflow
    if provider == "mock":
        workflow.llm_client = MockLLMClient(_mock_actions_for_case(case))
    else:
        workflow.llm_client = create_llm_client(provider)
    return workflow


def _mock_actions_for_case(case: dict[str, Any]) -> list[dict[str, Any]]:
    if case["case_id"] != "llm_fix_with_test":
        return []
    return [
        {
            "thought": "Read calculator.py to inspect the failing add implementation.",
            "action": "read_file",
            "action_input": {"path": "calculator.py"},
            "final": None,
        },
        {
            "thought": "The add function subtracts b; replace subtraction with addition.",
            "action": "edit_file",
            "action_input": {
                "path": "calculator.py",
                "old": "return a - b",
                "new": "return a + b",
            },
            "final": None,
        },
        {
            "thought": "Patch is applied and ready for test verification.",
            "action": None,
            "action_input": {},
            "final": "Fixed add by replacing subtraction with addition.",
        },
    ]


def _evaluate_case(
    case: dict[str, Any],
    response: dict[str, Any],
    strategy: str,
) -> tuple[bool, str]:
    expected_hitl = bool(case["expected_requires_human_approval"])
    actual_hitl = bool(response.get("requires_human_approval", False))
    if actual_hitl != expected_hitl:
        return False, "unexpected_hitl" if actual_hitl else "missing_hitl"

    failed_agents = [
        report.get("agent_name")
        for report in response.get("agent_reports", [])
        if report.get("status") == "failed"
    ]
    if failed_agents and not expected_hitl:
        if _has_llm_timeout(response):
            return False, "llm_timeout"
        if _has_llm_parse_error(response):
            return False, "llm_parse_error"
        return False, "unexpected_agent_failure"

    expected_changed = set(case.get("expected_changed_files") or [])
    actual_changed = set(response.get("changed_files") or [])
    if expected_changed and not expected_changed.issubset(actual_changed):
        return False, "missing_expected_change"
    if not expected_changed and actual_changed:
        return False, "unexpected_file_change"

    expected_returncode = case.get("expected_test_returncode")
    if expected_returncode is not None:
        test_result = response.get("test_result") or {}
        if test_result.get("returncode") != expected_returncode:
            return False, "unexpected_test_failure"

    if strategy == "langgraph" and not _node_trace(response):
        return False, "missing_node_trace"

    return True, ""


def _has_llm_parse_error(response: dict[str, Any]) -> bool:
    return any(
        "LLM action parse failed" in str(report.get("summary", ""))
        for report in response.get("agent_reports", [])
    )


def _has_llm_timeout(response: dict[str, Any]) -> bool:
    metrics = response.get("metrics") or {}
    if int(metrics.get("llm_timeouts", 0) or 0) > 0:
        return True
    return any(
        "LLM request timed out" in str(report.get("summary", ""))
        for report in response.get("agent_reports", [])
    )


def _write_strategy_jsonl(results: list[ExperimentResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(_result_payload(result), ensure_ascii=False) + "\n")


def _write_raw(results: list[ExperimentResult], raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        path = raw_dir / f"{result.case['case_id']}.{result.strategy}.json"
        path.write_text(
            json.dumps(_result_payload(result), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _result_payload(result: ExperimentResult) -> dict[str, Any]:
    return {
        "run_id": result.run_id,
        "case_id": result.case["case_id"],
        "category": result.case["category"],
        "strategy": result.strategy,
        "success": result.success,
        "failure_reason": result.failure_reason,
        "elapsed_ms": round(result.elapsed_ms, 3),
        "error": result.error,
        "response": result.response,
    }


def _build_comparison_rows(
    run_id: str,
    sequential_results: list[ExperimentResult],
    langgraph_results: list[ExperimentResult],
) -> list[dict[str, Any]]:
    langgraph_by_case = {result.case["case_id"]: result for result in langgraph_results}
    rows: list[dict[str, Any]] = []
    for sequential in sequential_results:
        langgraph = langgraph_by_case[sequential.case["case_id"]]
        sequential_metrics = _metrics(sequential.response)
        langgraph_metrics = _metrics(langgraph.response)
        sequential_agents = _executed_agents(sequential.response)
        langgraph_agents = _executed_agents(langgraph.response)
        sequential_nodes = _node_trace(sequential.response)
        langgraph_nodes = _node_trace(langgraph.response)
        rows.append(
            {
                "run_id": run_id,
                "case_id": sequential.case["case_id"],
                "category": sequential.case["category"],
                "sequential_success": sequential.success,
                "langgraph_success": langgraph.success,
                "sequential_failure_reason": sequential.failure_reason,
                "langgraph_failure_reason": langgraph.failure_reason,
                "sequential_agents": "->".join(sequential_agents),
                "langgraph_agents": "->".join(langgraph_agents),
                "sequential_has_node_trace": bool(sequential_nodes),
                "langgraph_has_node_trace": bool(langgraph_nodes),
                "sequential_node_trace": "->".join(sequential_nodes),
                "langgraph_node_trace": "->".join(langgraph_nodes),
                "sequential_requires_human_approval": _requires_human_approval(sequential.response),
                "langgraph_requires_human_approval": _requires_human_approval(langgraph.response),
                "sequential_stopped_before_test": _stopped_before_test(sequential.response),
                "langgraph_stopped_before_test": _stopped_before_test(langgraph.response),
                "sequential_llm_calls": sequential_metrics["llm_calls"],
                "langgraph_llm_calls": langgraph_metrics["llm_calls"],
                "sequential_prompt_tokens": sequential_metrics["prompt_tokens"],
                "langgraph_prompt_tokens": langgraph_metrics["prompt_tokens"],
                "sequential_completion_tokens": sequential_metrics["completion_tokens"],
                "langgraph_completion_tokens": langgraph_metrics["completion_tokens"],
                "sequential_test_returncode": _test_returncode(sequential.response),
                "langgraph_test_returncode": _test_returncode(langgraph.response),
                "sequential_elapsed_ms": round(sequential.elapsed_ms, 3),
                "langgraph_elapsed_ms": round(langgraph.elapsed_ms, 3),
                "difference_summary": _difference_summary(sequential.response, langgraph.response),
            }
        )
    return rows


def _metrics(response: dict[str, Any] | None) -> dict[str, int]:
    if response is None:
        return {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}
    metrics = response.get("metrics") or {}
    return {
        "llm_calls": int(metrics.get("llm_calls", 0)),
        "prompt_tokens": int(metrics.get("prompt_tokens", 0)),
        "completion_tokens": int(metrics.get("completion_tokens", 0)),
    }


def _executed_agents(response: dict[str, Any] | None) -> list[str]:
    if response is None:
        return []
    return [str(report.get("agent_name")) for report in response.get("agent_reports", [])]


def _node_trace(response: dict[str, Any] | None) -> list[str]:
    if response is None:
        return []
    planning = response.get("planning") or {}
    langgraph = planning.get("langgraph") or {}
    return list(langgraph.get("nodes") or [])


def _requires_human_approval(response: dict[str, Any] | None) -> bool:
    if response is None:
        return False
    return bool(response.get("requires_human_approval", False))


def _stopped_before_test(response: dict[str, Any] | None) -> bool:
    if response is None:
        return False
    planned = list(response.get("planned_agents") or [])
    executed = _executed_agents(response)
    return "test_verify" in planned and "test_verify" not in executed


def _test_returncode(response: dict[str, Any] | None) -> int | str:
    if response is None:
        return ""
    test_result = response.get("test_result") or {}
    return test_result.get("returncode", "")


def _difference_summary(
    sequential_response: dict[str, Any] | None,
    langgraph_response: dict[str, Any] | None,
) -> str:
    if _node_trace(langgraph_response) and not _node_trace(sequential_response):
        return "LangGraph adds explicit node traces; sequential requires inferring flow from reports."
    return "Business result comparable; no additional LangGraph trace captured."


def _write_comparison_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id",
        "case_id",
        "category",
        "sequential_success",
        "langgraph_success",
        "sequential_failure_reason",
        "langgraph_failure_reason",
        "sequential_agents",
        "langgraph_agents",
        "sequential_has_node_trace",
        "langgraph_has_node_trace",
        "sequential_node_trace",
        "langgraph_node_trace",
        "sequential_requires_human_approval",
        "langgraph_requires_human_approval",
        "sequential_stopped_before_test",
        "langgraph_stopped_before_test",
        "sequential_llm_calls",
        "langgraph_llm_calls",
        "sequential_prompt_tokens",
        "langgraph_prompt_tokens",
        "sequential_completion_tokens",
        "langgraph_completion_tokens",
        "sequential_test_returncode",
        "langgraph_test_returncode",
        "sequential_elapsed_ms",
        "langgraph_elapsed_ms",
        "difference_summary",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_summary(
    provider: str,
    rows: list[dict[str, Any]],
    path: Path,
    workspace_root: Path,
) -> None:
    total = len(rows)
    sequential_success = sum(1 for row in rows if row["sequential_success"])
    langgraph_success = sum(1 for row in rows if row["langgraph_success"])
    langgraph_traces = sum(1 for row in rows if row["langgraph_has_node_trace"])
    sequential_traces = sum(1 for row in rows if row["sequential_has_node_trace"])
    content = [
        "# LangGraph vs Sequential Experiment Summary",
        "",
        "## Setup",
        "",
        f"- Provider: `{provider}`",
        f"- Cases: {total}",
        f"- Isolated workspace root: `{workspace_root}`",
        "- Sequential strategy: `use_langgraph=false`",
        "- LangGraph strategy: `use_langgraph=true`",
        "- LLM is optional and is only used inside Agent nodes when a case requires it.",
        "",
        "## Core Claim",
        "",
        "Sequential loops are enough for small demos. LangGraph is better for long, multi-agent, stateful workflows that need explicit control and observability.",
        "",
        "## Workflow controllability",
        "",
        "| Metric | Sequential | LangGraph |",
        "| --- | ---: | ---: |",
        f"| Success rate | {_rate(sequential_success, total)} | {_rate(langgraph_success, total)} |",
        f"| Cases with explicit node trace | {sequential_traces} | {langgraph_traces} |",
        "",
        "## Case Comparison",
        "",
        "| Case | Sequential result | LangGraph result | What LangGraph adds |",
        "| --- | --- | --- | --- |",
    ]
    for row in rows:
        content.append(
            "| {case_id} | {seq} | {lg} | {diff} |".format(
                case_id=row["case_id"],
                seq=_result_label(row["sequential_success"], row["sequential_agents"]),
                lg=_result_label(row["langgraph_success"], row["langgraph_node_trace"]),
                diff=row["difference_summary"],
            )
        )
    content.extend(
        [
            "",
            "## Interpretation",
            "",
            "LangGraph adds explicit node traces, conditional routing evidence, and visible interruption paths. Sequential workflow can reach the same business result, but complex control flow must be inferred from Agent reports and loop break conditions.",
            "",
            "This experiment is not intended to prove model quality or speed. It isolates workflow controllability: LangGraph orchestrates nodes and edges; LLM calls, when enabled, happen inside Agent nodes.",
            "",
        ]
    )
    path.write_text("\n".join(content), encoding="utf-8")


def _result_label(success: bool, trace: str) -> str:
    status = "success" if success else "failed"
    return f"{status}: `{trace}`" if trace else status


def _rate(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0.0%"
    return f"{numerator / denominator:.1%}"


def _is_timeout_error(exc: Exception) -> bool:
    name = exc.__class__.__name__.lower()
    message = str(exc).lower()
    return "timeout" in name or "timed out" in message


if __name__ == "__main__":
    main()
