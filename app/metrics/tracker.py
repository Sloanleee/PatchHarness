from __future__ import annotations

from app.schemas import WorkflowMetrics


class MetricsTracker:
    def __init__(self, planned_by: str = "rule") -> None:
        self.metrics = WorkflowMetrics(planned_by=planned_by)

    def agent_called(self) -> None:
        self.metrics.agent_calls += 1

    def tool_called(self) -> None:
        self.metrics.tool_calls += 1

    def llm_called(self) -> None:
        self.metrics.llm_calls += 1

    def snapshot(self) -> WorkflowMetrics:
        return WorkflowMetrics(
            agent_calls=self.metrics.agent_calls,
            tool_calls=self.metrics.tool_calls,
            llm_calls=self.metrics.llm_calls,
            planned_by=self.metrics.planned_by,
        )

