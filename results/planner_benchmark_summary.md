# Planner Benchmark Summary

## Setup

- Cases: 100
- Baseline: fixed 4-Agent pipeline (`code_review -> bug_fix -> test_verify -> summary`)
- Assumption: each Agent costs 2 LLM calls
- Dynamic strategy: `RulePlanner`

## Results

| Metric | Fixed Pipeline | Dynamic Planner |
| --- | ---: | ---: |
| Agent calls | 400 | 200 |
| Estimated LLM calls | 800 | 400 |

## Key Numbers

- Saved LLM calls: 400
- Saved LLM call rate: 50.0%
- Planner hit rate: 100.0%
- Average planning latency: 0.0021 ms

## Interpretation

The benchmark makes the README optimization claim reproducible: simple review requests avoid unnecessary fix/test/summary agents, fix requests route to `bug_fix -> test_verify`, full requests keep the complete chain, and fuzzy requests safely default to review.
