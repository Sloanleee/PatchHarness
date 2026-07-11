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
    enable_llm: bool = False
    use_langgraph: bool = False
    planning_confidence_threshold: float = 0.65


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
    context_events: list[dict[str, Any]] = field(default_factory=list)
    skills_available: list[dict[str, Any]] = field(default_factory=list)
    skills_loaded: list[str] = field(default_factory=list)
    requires_human_approval: bool = False
    hitl_events: list[dict[str, Any]] = field(default_factory=list)
    compression_events: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str = ""
    evidence_count: int = 0
    diagnostic_evidence: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class WorkflowMetrics:
    agent_calls: int = 0
    tool_calls: int = 0
    mcp_tool_calls: int = 0
    llm_calls: int = 0
    llm_timeouts: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_fallbacks: int = 0
    planner_hitl_interruptions: int = 0
    context_forks: int = 0
    context_merges: int = 0
    context_cleanups: int = 0
    skills_disclosed: int = 0
    skills_loaded: int = 0
    hitl_interruptions: int = 0
    compression_events: int = 0
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
    requires_human_approval: bool = False
    planning: dict[str, Any] = field(default_factory=dict)
    approval_events: list[dict[str, Any]] = field(default_factory=list)
    run_id: str | None = None
    failure_reason: str = ""
    pending_approval: dict[str, Any] | None = None

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
                    "context_events": report.context_events,
                    "skills_available": report.skills_available,
                    "skills_loaded": report.skills_loaded,
                    "requires_human_approval": report.requires_human_approval,
                    "hitl_events": report.hitl_events,
                    "compression_events": report.compression_events,
                    "stop_reason": report.stop_reason,
                    "evidence_count": report.evidence_count,
                    "diagnostic_evidence": report.diagnostic_evidence,
                }
                for report in self.agent_reports
            ],
            "changed_files": self.changed_files,
            "test_result": self.test_result,
            "metrics": {
                "agent_calls": self.metrics.agent_calls,
                "tool_calls": self.metrics.tool_calls,
                "mcp_tool_calls": self.metrics.mcp_tool_calls,
                "llm_calls": self.metrics.llm_calls,
                "llm_timeouts": self.metrics.llm_timeouts,
                "prompt_tokens": self.metrics.prompt_tokens,
                "completion_tokens": self.metrics.completion_tokens,
                "llm_fallbacks": self.metrics.llm_fallbacks,
                "planner_hitl_interruptions": self.metrics.planner_hitl_interruptions,
                "context_forks": self.metrics.context_forks,
                "context_merges": self.metrics.context_merges,
                "context_cleanups": self.metrics.context_cleanups,
                "skills_disclosed": self.metrics.skills_disclosed,
                "skills_loaded": self.metrics.skills_loaded,
                "hitl_interruptions": self.metrics.hitl_interruptions,
                "compression_events": self.metrics.compression_events,
                "planned_by": self.metrics.planned_by,
            },
            "final_summary": self.final_summary,
            "requires_human_approval": self.requires_human_approval,
            "planning": self.planning,
            "approval_events": self.approval_events,
            "run_id": self.run_id,
            "failure_reason": self.failure_reason,
            "pending_approval": self.pending_approval,
        }
