from __future__ import annotations

from app.schemas import WorkflowMetrics


class MetricsTracker:
    def __init__(self, planned_by: str = "rule") -> None:
        self.metrics = WorkflowMetrics(planned_by=planned_by)

    def agent_called(self) -> None:
        self.metrics.agent_calls += 1

    def tool_called(self) -> None:
        self.metrics.tool_calls += 1

    def mcp_tool_called(self) -> None:
        self.metrics.mcp_tool_calls += 1

    def llm_called(self, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        self.metrics.llm_calls += 1
        self.metrics.prompt_tokens += prompt_tokens
        self.metrics.completion_tokens += completion_tokens

    def llm_timed_out(self) -> None:
        self.metrics.llm_timeouts += 1

    def llm_fallback_used(self) -> None:
        self.metrics.llm_fallbacks += 1

    def planner_hitl_interrupted(self) -> None:
        self.metrics.planner_hitl_interruptions += 1

    def context_forked(self) -> None:
        self.metrics.context_forks += 1

    def context_merged(self) -> None:
        self.metrics.context_merges += 1

    def context_cleaned(self) -> None:
        self.metrics.context_cleanups += 1

    def skills_disclosed(self, count: int) -> None:
        self.metrics.skills_disclosed += count

    def skill_loaded(self) -> None:
        self.metrics.skills_loaded += 1

    def hitl_interrupted(self) -> None:
        self.metrics.hitl_interruptions += 1

    def compressed(self) -> None:
        self.metrics.compression_events += 1

    def snapshot(self) -> WorkflowMetrics:
        return WorkflowMetrics(
            agent_calls=self.metrics.agent_calls,
            tool_calls=self.metrics.tool_calls,
            mcp_tool_calls=self.metrics.mcp_tool_calls,
            llm_calls=self.metrics.llm_calls,
            llm_timeouts=self.metrics.llm_timeouts,
            prompt_tokens=self.metrics.prompt_tokens,
            completion_tokens=self.metrics.completion_tokens,
            llm_fallbacks=self.metrics.llm_fallbacks,
            planner_hitl_interruptions=self.metrics.planner_hitl_interruptions,
            context_forks=self.metrics.context_forks,
            context_merges=self.metrics.context_merges,
            context_cleanups=self.metrics.context_cleanups,
            skills_disclosed=self.metrics.skills_disclosed,
            skills_loaded=self.metrics.skills_loaded,
            hitl_interruptions=self.metrics.hitl_interruptions,
            compression_events=self.metrics.compression_events,
            planned_by=self.metrics.planned_by,
        )
