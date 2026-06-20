from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from app.metrics import MetricsTracker
from app.schemas import ToolResult


class BaseTool(ABC):
    name: str

    @abstractmethod
    def run(self, workspace: Path, **kwargs: Any) -> ToolResult:
        raise NotImplementedError


class ToolRegistry:
    def __init__(self, tools: list[BaseTool], metrics: MetricsTracker | None = None) -> None:
        self._tools = {tool.name: tool for tool in tools}
        self._metrics = metrics

    def names(self) -> list[str]:
        return sorted(self._tools)

    def run(self, name: str, workspace: Path, **kwargs: Any) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(name, False, error=f"Unknown tool: {name}")
        if self._metrics is not None:
            self._metrics.tool_called()
        return tool.run(workspace, **kwargs)


def resolve_workspace_path(workspace: Path, relative_path: str) -> Path:
    root = workspace.resolve()
    target = (root / relative_path).resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"Path escapes workspace: {relative_path}")
    return target

