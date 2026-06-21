from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from app.context.visibility import Visibility
from app.schemas import AgentReport, BugfixRequest


@dataclass(slots=True)
class ContextItem:
    key: str
    value: Any
    visibility: Visibility


@dataclass(slots=True)
class AgentContext:
    agent_name: str
    items: dict[str, ContextItem]
    events: list[dict[str, Any]] = field(default_factory=list)

    def visible_payload(self) -> dict[str, Any]:
        return {
            key: deepcopy(item.value)
            for key, item in self.items.items()
            if item.visibility != Visibility.HIDDEN
        }

    def get(self, key: str, default: Any = None) -> Any:
        item = self.items.get(key)
        if item is None or item.visibility == Visibility.HIDDEN:
            return default
        return deepcopy(item.value)

    def set_writable(self, key: str, value: Any) -> None:
        item = self.items.get(key)
        if item is not None and item.visibility == Visibility.READ_ONLY:
            raise ValueError(f"Cannot write read-only context item: {key}")
        self.items[key] = ContextItem(key, deepcopy(value), Visibility.WRITABLE)


class ContextManager:
    def __init__(self, items: dict[str, ContextItem]) -> None:
        self._items = items
        self.events: list[dict[str, Any]] = []

    @classmethod
    def from_request(cls, request: BugfixRequest, planned_agents: list[str]) -> "ContextManager":
        return cls(
            {
                "task_description": ContextItem(
                    "task_description",
                    request.task_description,
                    Visibility.READ_ONLY,
                ),
                "workspace_path": ContextItem(
                    "workspace_path",
                    request.workspace_path,
                    Visibility.READ_ONLY,
                ),
                "planned_agents": ContextItem(
                    "planned_agents",
                    list(planned_agents),
                    Visibility.READ_ONLY,
                ),
                "agent_reports": ContextItem("agent_reports", [], Visibility.WRITABLE),
                "internal_secrets": ContextItem(
                    "internal_secrets",
                    {"api_key": "<hidden>"},
                    Visibility.HIDDEN,
                ),
            }
        )

    def fork(self, agent_name: str) -> AgentContext:
        forked_items = {
            key: ContextItem(key, deepcopy(item.value), item.visibility)
            for key, item in self._items.items()
        }
        visible_keys = [
            key for key, item in forked_items.items() if item.visibility != Visibility.HIDDEN
        ]
        event = {
            "event": "fork",
            "agent": agent_name,
            "visible_keys": visible_keys,
            "hidden_keys": [
                key for key, item in forked_items.items() if item.visibility == Visibility.HIDDEN
            ],
        }
        self.events.append(event)
        return AgentContext(agent_name, forked_items, events=[event])

    def merge(self, context: AgentContext, report: AgentReport) -> dict[str, Any]:
        reports_item = self._items["agent_reports"]
        merged_reports = list(reports_item.value)
        merged_reports.append(
            {
                "agent_name": report.agent_name,
                "status": report.status,
                "summary": report.summary,
                "changed_files": list(report.changed_files),
                "requires_human_approval": report.requires_human_approval,
            }
        )
        self._items["agent_reports"] = ContextItem(
            "agent_reports",
            merged_reports,
            Visibility.WRITABLE,
        )
        event = {
            "event": "merge",
            "agent": context.agent_name,
            "merged_keys": ["agent_reports"],
        }
        self.events.append(event)
        context.events.append(event)
        return event

    def cleanup(self, context: AgentContext) -> dict[str, Any]:
        event = {
            "event": "cleanup",
            "agent": context.agent_name,
            "discarded_context_items": len(context.items),
        }
        self.events.append(event)
        context.events.append(event)
        context.items.clear()
        return event

