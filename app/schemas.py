from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


Mode = Literal["auto", "review", "fix", "full"]


@dataclass(slots=True)
class BugfixRequest:
    task_description: str
    workspace_path: str = "."
    mode: Mode = "auto"
    allow_edit: bool = False
    run_tests: bool = True
    test_command: str | None = None


@dataclass(slots=True)
class ToolResult:
    tool: str
    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass(slots=True)
class AgentReport:
    agent_name: str
    status: str
    thoughts: list[str] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    observations: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""
    changed_files: list[str] = field(default_factory=list)
    test_result: dict[str, Any] | None = None


@dataclass(slots=True)
class WorkflowMetrics:
    agent_calls: int = 0
    tool_calls: int = 0
    llm_calls: int = 0
    planned_by: str = "rule"


@dataclass(slots=True)
class BugfixResponse:
    request_id: str
    planned_agents: list[str]
    agent_reports: list[AgentReport]
    changed_files: list[str]
    test_result: dict[str, Any] | None
    metrics: WorkflowMetrics
    final_summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "planned_agents": self.planned_agents,
            "agent_reports": [
                {
                    "agent_name": report.agent_name,
                    "status": report.status,
                    "thoughts": report.thoughts,
                    "actions": report.actions,
                    "observations": report.observations,
                    "summary": report.summary,
                    "changed_files": report.changed_files,
                    "test_result": report.test_result,
                }
                for report in self.agent_reports
            ],
            "changed_files": self.changed_files,
            "test_result": self.test_result,
            "metrics": {
                "agent_calls": self.metrics.agent_calls,
                "tool_calls": self.metrics.tool_calls,
                "llm_calls": self.metrics.llm_calls,
                "planned_by": self.metrics.planned_by,
            },
            "final_summary": self.final_summary,
        }

