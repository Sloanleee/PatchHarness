# Planner Benchmark Summary

## Setup

- Cases: 100
- Baseline: fixed 4-Agent pipeline (`code_review -> bug_fix -> test_verify -> summary`)
- Assumption: each Agent costs 2 LLM calls
- Dynamic strategy: `RulePlanner`

## Results

| Metric | Fixed Pipeline | Dynamic Planner |
| --- | ---: | ---: |
| Agent calls | 400 | 250 |
| Estimated LLM calls | 800 | 500 |

## Key Numbers

- Saved LLM calls: 300
- Saved LLM call rate: 37.5%
- Planner hit rate: 100.0%
- Average planning latency: 0.0024 ms

## Interpretation

The benchmark makes the README optimization claim reproducible: simple review requests avoid unnecessary fix/test/summary agents, fix requests route to `root_cause_analysis -> patch_generation -> test_verify`, full requests keep the complete chain, and fuzzy requests safely default to review.
