from __future__ import annotations

from pathlib import Path
from typing import Any

from app.schemas import ToolResult
from app.tools.base import BaseTool


class _SkillTool(BaseTool):
    def __init__(self, skill_manager: Any | None) -> None:
        self.skill_manager = skill_manager

    def _require_manager(self) -> ToolResult | None:
        if self.skill_manager is None:
            return ToolResult(self.name, False, error="SkillManager is not configured")
        return None


class SearchSkillTool(_SkillTool):
    name = "search_skill"
    description = "Search skills by query using the configured skill index."
    input_schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}, "limit": {"type": "integer", "default": 3}},
        "required": ["query"],
    }

    def run(self, workspace: Path, **kwargs: Any) -> ToolResult:
        error = self._require_manager()
        if error:
            return error
        results = self.skill_manager.search_skill(str(kwargs.get("query", "")), int(kwargs.get("limit", 3)))
        return ToolResult(self.name, True, {"results": results})


class DownloadSkillTool(_SkillTool):
    name = "download_skill"
    description = "Download full skill content by skill name."
    input_schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
    }

    def run(self, workspace: Path, **kwargs: Any) -> ToolResult:
        error = self._require_manager()
        if error:
            return error
        name = str(kwargs.get("name", ""))
        return ToolResult(self.name, True, {"name": name, "content": self.skill_manager.load_skill(name)})


class CreateSkillTool(_SkillTool):
    name = "create_skill"
    description = "Create a new persistent skill markdown file."
    input_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "description": {"type": "string"},
            "triggers": {"type": "array", "items": {"type": "string"}},
            "content": {"type": "string"},
        },
        "required": ["name", "description", "content"],
    }

    def run(self, workspace: Path, **kwargs: Any) -> ToolResult:
        error = self._require_manager()
        if error:
            return error
        frontmatter = self.skill_manager.create_skill(
            name=str(kwargs.get("name", "")),
            description=str(kwargs.get("description", "")),
            triggers=list(kwargs.get("triggers", [])),
            content=str(kwargs.get("content", "")),
        )
        return ToolResult(self.name, True, {"skill": frontmatter.to_public_dict()})


class UpdateSkillTool(_SkillTool):
    name = "update_skill"
    description = "Update an existing persistent skill markdown file."
    input_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "description": {"type": "string"},
            "triggers": {"type": "array", "items": {"type": "string"}},
            "content": {"type": "string"},
        },
        "required": ["name", "content"],
    }

    def run(self, workspace: Path, **kwargs: Any) -> ToolResult:
        error = self._require_manager()
        if error:
            return error
        frontmatter = self.skill_manager.update_skill(
            name=str(kwargs.get("name", "")),
            description=kwargs.get("description"),
            triggers=kwargs.get("triggers"),
            content=str(kwargs.get("content", "")),
        )
        return ToolResult(self.name, True, {"skill": frontmatter.to_public_dict()})

