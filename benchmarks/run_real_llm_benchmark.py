from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.graph import BugfixWorkflow  # noqa: E402
from app.llm import MockLLMClient, create_llm_client  # noqa: E402
from app.planner.rule_planner import PlanningResult  # noqa: E402
from app.schemas import BugfixRequest  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
CASES_PATH = ROOT / "benchmarks" / "real_cases" / "cases.json"
TEMPLATES_DIR = ROOT / "benchmarks" / "real_cases" / "templates"
RESULTS_ROOT = ROOT / "results" / "real_llm_benchmark" / "runs"
FIXED_PIPELINE = ["code_review", "bug_fix", "test_verify", "summary"]


@dataclass(slots=True)
class CaseResult:
    run_id: str
    case: dict[str, Any]
    strategy: str
    response: dict[str, Any] | None
    elapsed_ms: float
    success: bool
    failure_reason: str
    error: str | None = None


class ForcedPlanner:
    def __init__(self, agents: list[str]) -> None:
        self.agents = agents

    def plan(self, request: BugfixRequest) -> PlanningResult:
        return PlanningResult(
            agents=list(self.agents),
            planned_by="benchmark_fixed",
            reason="forced fixed 4-Agent benchmark baseline",
            confidence=1.0,
        )


def main() -> None:
    args = _parse_args()
    cases = _select_cases(_load_cases(), args.category, args.max_cases)
    if not cases:
        raise SystemExit("No benchmark cases selected.")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_ROOT / run_id
    raw_dir = run_dir / "raw"
    workspace_root = (
        Path(tempfile.gettempdir()) / "patchharness_real_llm_benchmark" / run_id / "workspaces"
    )
    raw_dir.mkdir(parents=True, exist_ok=True)
    workspace_root.mkdir(parents=True, exist_ok=True)

    fixed_results: list[CaseResult] = []
    dynamic_results: list[CaseResult] = []
    for case in cases:
        print(f"Running {case['case_id']} ({case['category']})")
        fixed_results.append(
            _run_case(
                run_id=run_id,
                case=case,
                strategy="fixed",
                provider=args.provider,
                workspace_root=workspace_root,
            )
        )
        dynamic_results.append(
            _run_case(
                run_id=run_id,
                case=case,
                strategy="dynamic",
                provider=args.provider,
                workspace_root=workspace_root,
            )
        )

    _write_strategy_jsonl(fixed_results, run_dir / "fixed_pipeline.jsonl")
    _write_strategy_jsonl(dynamic_results, run_dir / "dynamic_planner.jsonl")
    _write_raw(fixed_results + dynamic_results, raw_dir)
    comparison_rows = _build_comparison_rows(run_id, fixed_results, dynamic_results)
    _write_comparison_csv(comparison_rows, run_dir / "comparison.csv")
    _write_summary(args.provider, comparison_rows, run_dir / "summary.md", workspace_root)

    print(f"Wrote benchmark run to {run_dir}")
    print(f"Summary: {run_dir / 'summary.md'}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real LLM calls benchmark for PatchHarness.")
    parser.add_argument(
        "--provider",
        default="mock",
        choices=["mock", "ark", "volcengine", "volcengine_ark", "deepseek"],
        help="LLM provider. Use mock for a no-cost framework smoke test.",
    )
    parser.add_argument("--category", help="Only run one case category.")
    parser.add_argument("--max-cases", type=int, default=5, help="Maximum cases to run.")
    return parser.parse_args()


def _load_cases() -> list[dict[str, Any]]:
    data = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Benchmark cases must be a list: {CASES_PATH}")
    return data


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
) -> CaseResult:
    workspace = _prepare_workspace(case, strategy, workspace_root)
    request = BugfixRequest(
        task_description=case["task_description"],
        workspace_path=str(workspace),
        mode=case["mode"],
        allow_edit=bool(case["allow_edit"]),
        run_tests=bool(case["run_tests"]),
        test_command=case.get("test_command"),
        enable_llm=True,
    )
    start = time.perf_counter()
    try:
        workflow = _build_workflow(strategy, provider, case)
        response = workflow.run(request).to_dict()
        elapsed_ms = (time.perf_counter() - start) * 1000
        success, failure_reason = _evaluate_case(case, response)
        return CaseResult(
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
        return CaseResult(
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
    """Create an isolated git index so git_diff never reads the parent repo."""

    if shutil.which("git") is None:
        return
    commands = (
        ["git", "init", "-q"],
        ["git", "add", "-A"],
    )
    for command in commands:
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


def _build_workflow(strategy: str, provider: str, case: dict[str, Any]) -> BugfixWorkflow:
    llm_client = _build_llm_client(provider, case)
    workflow = BugfixWorkflow.from_default_configs()
    if strategy == "fixed":
        return BugfixWorkflow(
            registry=workflow.registry,
            planner=ForcedPlanner(FIXED_PIPELINE),
            llm_client=llm_client,
        )
    return BugfixWorkflow(
        registry=workflow.registry,
        llm_client=llm_client,
    )


def _build_llm_client(provider: str, case: dict[str, Any]):
    if provider != "mock":
        return create_llm_client(provider)
    return MockLLMClient(_mock_actions_for_case(case))


def _mock_actions_for_case(case: dict[str, Any]) -> list[dict[str, Any]]:
    category = str(case["category"])
    if category == "llm_fix":
        return [
            {
                "thought": "先读取 calculator.py，确认加法实现。",
                "action": "read_file",
                "action_input": {"path": "calculator.py"},
                "final": None,
            },
            {
                "thought": "发现 add 使用减法，替换为加法。",
                "action": "edit_file",
                "action_input": {
                    "path": "calculator.py",
                    "old": "return a - b",
                    "new": "return a + b",
                },
                "final": None,
            },
            {
                "thought": "代码已修改，可以交给测试 Agent 验证。",
                "action": None,
                "action_input": {},
                "final": "已将 add 的实现从减法修复为加法。",
            },
        ]
    if category == "review_only":
        return [
            {
                "thought": "固定管道仍进入 bug_fix；执行一次无差异编辑以模拟不必要 LLM 工作。",
                "action": "edit_file",
                "action_input": {
                    "path": "calculator.py",
                    "old": "return a + b",
                    "new": "return a + b",
                },
                "final": None,
            },
            {
                "thought": "没有真实 bug 需要修复。",
                "action": None,
                "action_input": {},
                "final": "未发现需要修复的加法缺陷。",
            },
        ]
    return []


def _evaluate_case(case: dict[str, Any], response: dict[str, Any]) -> tuple[bool, str]:
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
        actual_returncode = test_result.get("returncode")
        if actual_returncode != expected_returncode:
            return False, "unexpected_test_failure"

    return True, ""


def _has_llm_parse_error(response: dict[str, Any]) -> bool:
    for report in response.get("agent_reports", []):
        summary = str(report.get("summary", ""))
        if "LLM action parse failed" in summary:
            return True
    return False


def _has_llm_timeout(response: dict[str, Any]) -> bool:
    metrics = response.get("metrics") or {}
    if int(metrics.get("llm_timeouts", 0) or 0) > 0:
        return True
    for report in response.get("agent_reports", []):
        summary = str(report.get("summary", ""))
        if "LLM request timed out" in summary:
            return True
    return False


def _write_strategy_jsonl(results: list[CaseResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(_result_payload(result), ensure_ascii=False) + "\n")


def _write_raw(results: list[CaseResult], raw_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for result in results:
        path = raw_dir / f"{result.case['case_id']}.{result.strategy}.json"
        path.write_text(
            json.dumps(_result_payload(result), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _result_payload(result: CaseResult) -> dict[str, Any]:
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
    fixed_results: list[CaseResult],
    dynamic_results: list[CaseResult],
) -> list[dict[str, Any]]:
    dynamic_by_case = {result.case["case_id"]: result for result in dynamic_results}
    rows: list[dict[str, Any]] = []
    for fixed in fixed_results:
        dynamic = dynamic_by_case[fixed.case["case_id"]]
        fixed_metrics = _metrics(fixed.response)
        dynamic_metrics = _metrics(dynamic.response)
        fixed_tokens = fixed_metrics["prompt_tokens"] + fixed_metrics["completion_tokens"]
        dynamic_tokens = dynamic_metrics["prompt_tokens"] + dynamic_metrics["completion_tokens"]
        rows.append(
            {
                "run_id": run_id,
                "case_id": fixed.case["case_id"],
                "category": fixed.case["category"],
                "fixed_success": fixed.success,
                "dynamic_success": dynamic.success,
                "fixed_failure_reason": fixed.failure_reason,
                "dynamic_failure_reason": dynamic.failure_reason,
                "fixed_agents": "->".join(_planned_agents(fixed.response)),
                "dynamic_agents": "->".join(_planned_agents(dynamic.response)),
                "fixed_agent_calls": fixed_metrics["agent_calls"],
                "dynamic_agent_calls": dynamic_metrics["agent_calls"],
                "fixed_llm_calls": fixed_metrics["llm_calls"],
                "dynamic_llm_calls": dynamic_metrics["llm_calls"],
                "fixed_llm_timeouts": fixed_metrics["llm_timeouts"],
                "dynamic_llm_timeouts": dynamic_metrics["llm_timeouts"],
                "saved_llm_calls": fixed_metrics["llm_calls"] - dynamic_metrics["llm_calls"],
                "fixed_prompt_tokens": fixed_metrics["prompt_tokens"],
                "dynamic_prompt_tokens": dynamic_metrics["prompt_tokens"],
                "fixed_completion_tokens": fixed_metrics["completion_tokens"],
                "dynamic_completion_tokens": dynamic_metrics["completion_tokens"],
                "saved_total_tokens": fixed_tokens - dynamic_tokens,
                "fixed_requires_human_approval": _requires_human_approval(fixed.response),
                "dynamic_requires_human_approval": _requires_human_approval(dynamic.response),
                "fixed_test_returncode": _test_returncode(fixed.response),
                "dynamic_test_returncode": _test_returncode(dynamic.response),
                "fixed_elapsed_ms": round(fixed.elapsed_ms, 3),
                "dynamic_elapsed_ms": round(dynamic.elapsed_ms, 3),
            }
        )
    return rows


def _metrics(response: dict[str, Any] | None) -> dict[str, int]:
    if response is None:
        return {
            "agent_calls": 0,
            "llm_calls": 0,
            "llm_timeouts": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }
    metrics = response.get("metrics") or {}
    return {
        "agent_calls": int(metrics.get("agent_calls", 0)),
        "llm_calls": int(metrics.get("llm_calls", 0)),
        "llm_timeouts": int(metrics.get("llm_timeouts", 0)),
        "prompt_tokens": int(metrics.get("prompt_tokens", 0)),
        "completion_tokens": int(metrics.get("completion_tokens", 0)),
    }


def _planned_agents(response: dict[str, Any] | None) -> list[str]:
    if response is None:
        return []
    return list(response.get("planned_agents") or [])


def _requires_human_approval(response: dict[str, Any] | None) -> bool:
    if response is None:
        return False
    return bool(response.get("requires_human_approval", False))


def _test_returncode(response: dict[str, Any] | None) -> int | str:
    if response is None:
        return ""
    test_result = response.get("test_result") or {}
    return test_result.get("returncode", "")


def _write_comparison_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
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
    fixed_success = sum(1 for row in rows if row["fixed_success"])
    dynamic_success = sum(1 for row in rows if row["dynamic_success"])
    fixed_llm = sum(int(row["fixed_llm_calls"]) for row in rows)
    dynamic_llm = sum(int(row["dynamic_llm_calls"]) for row in rows)
    fixed_timeouts = sum(int(row["fixed_llm_timeouts"]) for row in rows)
    dynamic_timeouts = sum(int(row["dynamic_llm_timeouts"]) for row in rows)
    fixed_tokens = sum(int(row["fixed_prompt_tokens"]) + int(row["fixed_completion_tokens"]) for row in rows)
    dynamic_tokens = sum(
        int(row["dynamic_prompt_tokens"]) + int(row["dynamic_completion_tokens"]) for row in rows
    )
    failure_reasons = Counter(
        reason
        for row in rows
        for reason in (row["fixed_failure_reason"], row["dynamic_failure_reason"])
        if reason
    )
    by_category = _category_summary(rows)
    content = [
        "# Real LLM Benchmark Summary",
        "",
        "## Setup",
        "",
        f"- Provider: `{provider}`",
        f"- Cases: {total}",
        f"- Isolated workspace root: `{workspace_root}`",
        "- Baseline: fixed 4-Agent pipeline (`code_review -> bug_fix -> test_verify -> summary`)",
        "- Dynamic strategy: current PatchHarness planner",
        "",
        "## Results",
        "",
        "| Metric | Fixed Pipeline | Dynamic Planner |",
        "| --- | ---: | ---: |",
        f"| Success rate | {_rate(fixed_success, total)} | {_rate(dynamic_success, total)} |",
        f"| Real LLM calls | {fixed_llm} | {dynamic_llm} |",
        f"| LLM timeouts | {fixed_timeouts} | {dynamic_timeouts} |",
        f"| Total tokens | {fixed_tokens} | {dynamic_tokens} |",
        "",
        "## Savings",
        "",
        f"- Saved LLM calls: {fixed_llm - dynamic_llm}",
        f"- Saved LLM call rate: {_rate(fixed_llm - dynamic_llm, fixed_llm)}",
        f"- Saved successful calls plus timeouts: {(fixed_llm + fixed_timeouts) - (dynamic_llm + dynamic_timeouts)}",
        f"- Saved total tokens: {fixed_tokens - dynamic_tokens}",
        f"- Token saving rate: {_rate(fixed_tokens - dynamic_tokens, fixed_tokens)}",
        "",
        "## By Category",
        "",
        "| Category | Cases | Fixed success | Dynamic success | Fixed LLM calls | Dynamic LLM calls |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for category, item in by_category.items():
        content.append(
            "| {category} | {cases} | {fixed_success} | {dynamic_success} | {fixed_llm} | {dynamic_llm} |".format(
                category=category,
                cases=item["cases"],
                fixed_success=_rate(item["fixed_success"], item["cases"]),
                dynamic_success=_rate(item["dynamic_success"], item["cases"]),
                fixed_llm=item["fixed_llm"],
                dynamic_llm=item["dynamic_llm"],
            )
        )
    content.extend(["", "## Failure Reasons", ""])
    if failure_reasons:
        for reason, count in sorted(failure_reasons.items()):
            content.append(f"- `{reason}`: {count}")
    else:
        content.append("- No failures.")
    content.append("")
    path.write_text("\n".join(content), encoding="utf-8")


def _category_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    grouped: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "cases": 0,
            "fixed_success": 0,
            "dynamic_success": 0,
            "fixed_llm": 0,
            "dynamic_llm": 0,
            "fixed_timeouts": 0,
            "dynamic_timeouts": 0,
        }
    )
    for row in rows:
        item = grouped[str(row["category"])]
        item["cases"] += 1
        item["fixed_success"] += int(bool(row["fixed_success"]))
        item["dynamic_success"] += int(bool(row["dynamic_success"]))
        item["fixed_llm"] += int(row["fixed_llm_calls"])
        item["dynamic_llm"] += int(row["dynamic_llm_calls"])
        item["fixed_timeouts"] += int(row["fixed_llm_timeouts"])
        item["dynamic_timeouts"] += int(row["dynamic_llm_timeouts"])
    return dict(grouped)


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
