from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable

from app.planner import RulePlanner
from app.schemas import BugfixRequest


FIXED_PIPELINE = ["code_review", "bug_fix", "test_verify", "summary"]
LLM_CALLS_PER_AGENT = 2


@dataclass(slots=True)
class BenchmarkCase:
    case_id: int
    category: str
    task_description: str
    expected_agents: list[str]


@dataclass(slots=True)
class BenchmarkRow:
    case_id: int
    category: str
    task_description: str
    expected_agents: list[str]
    dynamic_agents: list[str]
    fixed_agent_calls: int
    dynamic_agent_calls: int
    fixed_llm_calls: int
    dynamic_llm_calls: int
    saved_llm_calls: int
    planner_hit: bool
    elapsed_ms: float


@dataclass(slots=True)
class BenchmarkSummary:
    total_cases: int
    planner_hits: int
    planner_hit_rate: float
    fixed_agent_calls: int
    dynamic_agent_calls: int
    fixed_llm_calls: int
    dynamic_llm_calls: int
    saved_llm_calls: int
    saved_llm_call_rate: float
    avg_elapsed_ms: float


def build_default_cases() -> list[BenchmarkCase]:
    review_templates = [
        "请审查登录模块的异常处理",
        "review payment service error paths",
        "检查订单状态流转是否有边界问题",
        "请审查缓存失效逻辑",
        "review API validation code",
    ]
    fix_templates = [
        "修复用户注册 bug",
        "fix payment bug when amount is zero",
        "修复接口报错：NoneType has no attribute id",
        "bug: retry counter never resets",
        "修复测试失败的解析逻辑",
    ]
    full_templates = [
        "全面检查并修复用户模块",
        "full check for billing workflow",
        "完整检查缓存模块并验证",
        "请全面检查项目中的异常路径",
        "comprehensive review and fix auth service",
    ]
    fuzzy_templates = [
        "帮我看一下这个模块",
        "这个功能感觉不太对",
        "看看最近改动有没有问题",
        "帮忙处理一下线上反馈",
        "分析一下用户反馈原因",
    ]

    cases: list[BenchmarkCase] = []
    case_id = 1
    for _ in range(5):
        for text in review_templates:
            cases.append(BenchmarkCase(case_id, "review", text, ["code_review"]))
            case_id += 1
    for _ in range(5):
        for text in fix_templates:
            cases.append(BenchmarkCase(case_id, "fix", text, ["bug_fix", "test_verify"]))
            case_id += 1
    for _ in range(5):
        for text in full_templates:
            cases.append(
                BenchmarkCase(
                    case_id,
                    "full",
                    text,
                    ["code_review", "bug_fix", "test_verify", "summary"],
                )
            )
            case_id += 1
    for _ in range(5):
        for text in fuzzy_templates:
            cases.append(BenchmarkCase(case_id, "fuzzy", text, ["code_review"]))
            case_id += 1

    return cases


def run_planner_benchmark(cases: Iterable[BenchmarkCase] | None = None) -> tuple[list[BenchmarkRow], BenchmarkSummary]:
    planner = RulePlanner()
    rows: list[BenchmarkRow] = []
    for case in cases or build_default_cases():
        start = time.perf_counter()
        result = planner.plan(BugfixRequest(task_description=case.task_description))
        elapsed_ms = (time.perf_counter() - start) * 1000
        dynamic_agents = result.agents
        fixed_agent_calls = len(FIXED_PIPELINE)
        dynamic_agent_calls = len(dynamic_agents)
        fixed_llm_calls = fixed_agent_calls * LLM_CALLS_PER_AGENT
        dynamic_llm_calls = dynamic_agent_calls * LLM_CALLS_PER_AGENT
        rows.append(
            BenchmarkRow(
                case_id=case.case_id,
                category=case.category,
                task_description=case.task_description,
                expected_agents=case.expected_agents,
                dynamic_agents=dynamic_agents,
                fixed_agent_calls=fixed_agent_calls,
                dynamic_agent_calls=dynamic_agent_calls,
                fixed_llm_calls=fixed_llm_calls,
                dynamic_llm_calls=dynamic_llm_calls,
                saved_llm_calls=fixed_llm_calls - dynamic_llm_calls,
                planner_hit=dynamic_agents == case.expected_agents,
                elapsed_ms=elapsed_ms,
            )
        )
    return rows, summarize_rows(rows)


def summarize_rows(rows: list[BenchmarkRow]) -> BenchmarkSummary:
    total_cases = len(rows)
    planner_hits = sum(1 for row in rows if row.planner_hit)
    fixed_agent_calls = sum(row.fixed_agent_calls for row in rows)
    dynamic_agent_calls = sum(row.dynamic_agent_calls for row in rows)
    fixed_llm_calls = sum(row.fixed_llm_calls for row in rows)
    dynamic_llm_calls = sum(row.dynamic_llm_calls for row in rows)
    saved_llm_calls = fixed_llm_calls - dynamic_llm_calls
    return BenchmarkSummary(
        total_cases=total_cases,
        planner_hits=planner_hits,
        planner_hit_rate=planner_hits / total_cases if total_cases else 0.0,
        fixed_agent_calls=fixed_agent_calls,
        dynamic_agent_calls=dynamic_agent_calls,
        fixed_llm_calls=fixed_llm_calls,
        dynamic_llm_calls=dynamic_llm_calls,
        saved_llm_calls=saved_llm_calls,
        saved_llm_call_rate=saved_llm_calls / fixed_llm_calls if fixed_llm_calls else 0.0,
        avg_elapsed_ms=mean(row.elapsed_ms for row in rows) if rows else 0.0,
    )


def write_benchmark_csv(rows: list[BenchmarkRow], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "case_id",
                "category",
                "task_description",
                "expected_agents",
                "dynamic_agents",
                "fixed_agent_calls",
                "dynamic_agent_calls",
                "fixed_llm_calls",
                "dynamic_llm_calls",
                "saved_llm_calls",
                "planner_hit",
                "elapsed_ms",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "case_id": row.case_id,
                    "category": row.category,
                    "task_description": row.task_description,
                    "expected_agents": "->".join(row.expected_agents),
                    "dynamic_agents": "->".join(row.dynamic_agents),
                    "fixed_agent_calls": row.fixed_agent_calls,
                    "dynamic_agent_calls": row.dynamic_agent_calls,
                    "fixed_llm_calls": row.fixed_llm_calls,
                    "dynamic_llm_calls": row.dynamic_llm_calls,
                    "saved_llm_calls": row.saved_llm_calls,
                    "planner_hit": row.planner_hit,
                    "elapsed_ms": f"{row.elapsed_ms:.4f}",
                }
            )


def write_benchmark_summary(summary: BenchmarkSummary, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = f"""# Planner Benchmark Summary

## Setup

- Cases: {summary.total_cases}
- Baseline: fixed 4-Agent pipeline (`code_review -> bug_fix -> test_verify -> summary`)
- Assumption: each Agent costs {LLM_CALLS_PER_AGENT} LLM calls
- Dynamic strategy: `RulePlanner`

## Results

| Metric | Fixed Pipeline | Dynamic Planner |
| --- | ---: | ---: |
| Agent calls | {summary.fixed_agent_calls} | {summary.dynamic_agent_calls} |
| Estimated LLM calls | {summary.fixed_llm_calls} | {summary.dynamic_llm_calls} |

## Key Numbers

- Saved LLM calls: {summary.saved_llm_calls}
- Saved LLM call rate: {summary.saved_llm_call_rate:.1%}
- Planner hit rate: {summary.planner_hit_rate:.1%}
- Average planning latency: {summary.avg_elapsed_ms:.4f} ms

## Interpretation

The benchmark makes the README optimization claim reproducible: simple review requests avoid unnecessary fix/test/summary agents, fix requests route to `bug_fix -> test_verify`, full requests keep the complete chain, and fuzzy requests safely default to review.
"""
    path.write_text(content, encoding="utf-8")
