from __future__ import annotations

from dataclasses import dataclass, field

from app.schemas import AgentReport, BugfixRequest


@dataclass(slots=True)
class WorkflowState:
    request: BugfixRequest
    planned_agents: list[str] = field(default_factory=list)
    reports: list[AgentReport] = field(default_factory=list)

