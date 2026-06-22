# Real LLM Benchmark Summary

## Setup

- Provider: `ark`
- Cases: 9
- Isolated workspace root: `C:\Users\hp\AppData\Local\Temp\patchharness_real_llm_benchmark\20260621_214846\workspaces`
- Baseline: fixed 4-Agent pipeline (`code_review -> bug_fix -> test_verify -> summary`)
- Dynamic strategy: current PatchHarness planner

## Results

| Metric | Fixed Pipeline | Dynamic Planner |
| --- | ---: | ---: |
| Success rate | 66.7% | 100.0% |
| Real LLM calls | 7 | 3 |
| LLM timeouts | 2 | 0 |
| Total tokens | 4139 | 1398 |

## Savings

- Saved LLM calls: 4
- Saved LLM call rate: 57.1%
- Saved successful calls plus timeouts: 6
- Saved total tokens: 2741
- Token saving rate: 66.2%

## By Category

| Category | Cases | Fixed success | Dynamic success | Fixed LLM calls | Dynamic LLM calls |
| --- | ---: | ---: | ---: | ---: | ---: |
| review_only | 3 | 0.0% | 100.0% | 4 | 0 |
| deterministic_fix | 3 | 100.0% | 100.0% | 0 | 0 |
| llm_fix | 1 | 100.0% | 100.0% | 3 | 3 |
| full_workflow | 1 | 100.0% | 100.0% | 0 | 0 |
| hitl_risk | 1 | 100.0% | 100.0% | 0 | 0 |

## Failure Reasons

- `llm_timeout`: 2
- `unexpected_agent_failure`: 1
