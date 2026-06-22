from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from app.schemas import ToolResult
from app.tools.base import BaseTool


class RunTestTool(BaseTool):
    name = "run_test"
    description = "Run a test command in the workspace."
    input_schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string"},
            "timeout": {"type": "integer", "default": 60},
        },
    }

    def run(self, workspace: Path, **kwargs: Any) -> ToolResult:
        command = str(kwargs.get("command") or "python -m unittest discover -s tests")
        timeout = int(kwargs.get("timeout", 60))

        try:
            completed = subprocess.run(
                command,
                cwd=workspace,
                shell=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return ToolResult(
                self.name,
                False,
                {"command": command, "timeout": timeout},
                error=f"Test command timed out after {timeout}s: {exc}",
            )
        except OSError as exc:
            return ToolResult(self.name, False, {"command": command}, error=str(exc))

        return ToolResult(
            self.name,
            completed.returncode == 0,
            {
                "command": command,
                "returncode": completed.returncode,
                "stdout": completed.stdout[-4000:],
                "stderr": completed.stderr[-2000:],
            },
            None if completed.returncode == 0 else "test command failed",
        )
