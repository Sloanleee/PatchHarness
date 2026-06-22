from __future__ import annotations

from pathlib import Path
from typing import Any

from app.schemas import ToolResult
from app.tools.base import ToolRegistry


class MCPServer:
    """A local MCP-style tool server.

    It exposes tool schemas and a single call interface. The implementation is
    intentionally in-process for the project MVP, but the boundary mirrors the
    shape needed to replace it with a transport-based MCP server later.
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    def list_tools(self) -> list[dict[str, Any]]:
        return self.registry.schemas()

    def call_tool(self, name: str, workspace: Path, arguments: dict[str, Any]) -> ToolResult:
        return self.registry.run(name, workspace, **arguments)

