from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from app.schemas import ToolResult
from app.tools.base import BaseTool, ToolRegistry, resolve_workspace_path


class GrepSearchTool(BaseTool):
    name = "grep_search"
    description = "Search text files in the workspace for a query."
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "max_results": {"type": "integer", "default": 20},
        },
        "required": ["query"],
    }

    def run(self, workspace: Path, **kwargs: Any) -> ToolResult:
        query = str(kwargs.get("query", "")).strip()
        max_results = int(kwargs.get("max_results", 20))
        if not query:
            return ToolResult(self.name, False, error="query is required")

        matches: list[dict[str, Any]] = []
        for path in workspace.rglob("*"):
            if not path.is_file() or _skip_path(path):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            except OSError as exc:
                return ToolResult(self.name, False, error=str(exc))
            for line_no, line in enumerate(text.splitlines(), start=1):
                if query.lower() in line.lower():
                    matches.append(
                        {
                            "path": str(path.relative_to(workspace)),
                            "line": line_no,
                            "text": line.strip(),
                        }
                    )
                    if len(matches) >= max_results:
                        return ToolResult(self.name, True, {"matches": matches})
        return ToolResult(self.name, True, {"matches": matches})


class ReadFileTool(BaseTool):
    name = "read_file"
    description = "Read a UTF-8 text file from the workspace."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "start_line": {"type": "integer", "minimum": 1},
            "end_line": {"type": "integer", "minimum": 1},
        },
        "required": ["path"],
    }

    def run(self, workspace: Path, **kwargs: Any) -> ToolResult:
        relative_path = str(kwargs.get("path", "")).strip()
        if not relative_path:
            return ToolResult(self.name, False, error="path is required")
        try:
            path = resolve_workspace_path(workspace, relative_path)
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            return ToolResult(self.name, False, error=str(exc))
        except ValueError as exc:
            return ToolResult(self.name, False, error=str(exc))
        lines = text.splitlines(keepends=True)
        total_lines = len(lines)
        start_line = int(kwargs.get("start_line", 1))
        end_line = int(kwargs.get("end_line", min(total_lines, start_line + 399)))
        if start_line < 1 or end_line < start_line:
            return ToolResult(
                self.name,
                False,
                error="start_line must be >= 1 and end_line must be >= start_line",
            )
        selected_end = min(end_line, total_lines)
        content = "".join(lines[start_line - 1:selected_end])
        return ToolResult(
            self.name,
            True,
            {
                "path": relative_path,
                "content": content,
                "start_line": start_line,
                "end_line": selected_end,
                "total_lines": total_lines,
                "truncated": selected_end < total_lines,
            },
        )


class EditFileTool(BaseTool):
    name = "edit_file"
    description = "Replace one exact text occurrence in a workspace file."
    input_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "old": {"type": "string"},
            "new": {"type": "string"},
            "allow_edit": {"type": "boolean"},
        },
        "required": ["path", "old", "new", "allow_edit"],
    }

    def run(self, workspace: Path, **kwargs: Any) -> ToolResult:
        relative_path = str(kwargs.get("path", "")).strip()
        old = str(kwargs.get("old", ""))
        new = str(kwargs.get("new", ""))
        allow_edit = bool(kwargs.get("allow_edit", False))

        if not allow_edit:
            return ToolResult(self.name, False, error="Editing is disabled for this request")
        if not relative_path:
            return ToolResult(self.name, False, error="path is required")
        if old == "":
            return ToolResult(self.name, False, error="old text is required")

        try:
            path = resolve_workspace_path(workspace, relative_path)
            text = path.read_text(encoding="utf-8")
            if old not in text:
                return ToolResult(
                    self.name,
                    False,
                    data={"path": relative_path},
                    error="old text not found",
                )
            path.write_text(text.replace(old, new, 1), encoding="utf-8")
        except OSError as exc:
            return ToolResult(self.name, False, error=str(exc))
        except ValueError as exc:
            return ToolResult(self.name, False, error=str(exc))

        return ToolResult(
            self.name,
            True,
            {"path": relative_path, "replacements": 1},
        )


class GitDiffTool(BaseTool):
    name = "git_diff"
    description = "Return git diff for the workspace or a path when inside a git repository."
    input_schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
    }

    def run(self, workspace: Path, **kwargs: Any) -> ToolResult:
        relative_path = str(kwargs.get("path", "")).strip()
        try:
            repo_check = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=workspace,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return ToolResult(self.name, False, error=str(exc))

        if repo_check.returncode != 0:
            return ToolResult(
                self.name,
                True,
                {
                    "returncode": 0,
                    "stdout": "",
                    "stderr": "workspace is not a git repository; diff skipped",
                },
            )

        command = ["git", "diff", "--"]
        if relative_path:
            command.append(relative_path)
        try:
            completed = subprocess.run(
                command,
                cwd=workspace,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return ToolResult(self.name, False, error=str(exc))

        return ToolResult(
            self.name,
            completed.returncode == 0,
            {
                "returncode": completed.returncode,
                "stdout": completed.stdout[-4000:],
                "stderr": completed.stderr[-2000:],
            },
            None if completed.returncode == 0 else completed.stderr,
        )


def create_default_tools(metrics: Any | None = None, skill_manager: Any | None = None) -> ToolRegistry:
    from app.tools.shell_tools import RunTestTool
    from app.skills.tools import CreateSkillTool, DownloadSkillTool, SearchSkillTool, UpdateSkillTool

    return ToolRegistry(
        [
            GrepSearchTool(),
            ReadFileTool(),
            EditFileTool(),
            RunTestTool(),
            GitDiffTool(),
            SearchSkillTool(skill_manager),
            DownloadSkillTool(skill_manager),
            CreateSkillTool(skill_manager),
            UpdateSkillTool(skill_manager),
        ],
        metrics=metrics,
    )


def _skip_path(path: Path) -> bool:
    skipped_parts = {".git", "__pycache__", ".pytest_cache", ".venv", "venv"}
    return any(part in skipped_parts for part in path.parts)
