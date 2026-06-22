from __future__ import annotations

from pathlib import Path
from typing import Any

from app.mcp.server import MCPServer
from app.metrics import MetricsTracker
from app.schemas import ToolResult


class MCPClient:
    def __init__(self, server: MCPServer, metrics: MetricsTracker | None = None) -> None:
        self.server = server
        self.metrics = metrics

    def names(self) -> list[str]:
        return sorted(tool["name"] for tool in self.server.list_tools())

    def schemas(self) -> list[dict[str, Any]]:
        return self.server.list_tools()

    def run(self, name: str, workspace: Path, **kwargs: Any) -> ToolResult:
        if self.metrics is not None:
            self.metrics.mcp_tool_called()
        return self.server.call_tool(name, workspace, kwargs)

