from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any


@dataclass(slots=True)
class HitlDecision:
    requires_approval: bool
    reason: str = ""
    risk: str = "low"

    def to_event(self, tool: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "event": "human_approval_required",
            "tool": tool,
            "risk": self.risk,
            "reason": self.reason,
            "path": payload.get("path"),
        }


class HitlPolicy:
    SENSITIVE_FILENAMES = {
        ".env",
        ".env.local",
        ".env.production",
        "requirements.txt",
        "pyproject.toml",
        "poetry.lock",
        "package.json",
        "package-lock.json",
        "docker-compose.yml",
        "Dockerfile",
    }
    SENSITIVE_SUFFIXES = (".pem", ".key", ".crt", ".pfx", ".p12")

    def evaluate_tool_call(self, tool: str, payload: dict[str, Any]) -> HitlDecision:
        if tool != "edit_file":
            return HitlDecision(False)

        path = str(payload.get("path", "")).replace("\\", "/")
        old = str(payload.get("old", ""))
        new = str(payload.get("new", ""))
        filename = PurePosixPath(path).name

        if filename in self.SENSITIVE_FILENAMES or path.endswith(self.SENSITIVE_SUFFIXES):
            return HitlDecision(
                True,
                reason=f"Editing sensitive file requires approval: {path}",
                risk="high",
            )
        if len(old) > 1200 or len(new) > 1200:
            return HitlDecision(
                True,
                reason="Large replacement requires approval",
                risk="medium",
            )
        if old.count("\n") >= 20 or new.count("\n") >= 20:
            return HitlDecision(
                True,
                reason="Bulk multi-line edit requires approval",
                risk="medium",
            )
        return HitlDecision(False)

