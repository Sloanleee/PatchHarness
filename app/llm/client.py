from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(slots=True)
class LLMAction:
    thought: str
    action: str | None = None
    action_input: dict[str, Any] = field(default_factory=dict)
    final: str | None = None
    confidence: float | None = None

    @classmethod
    def from_json_text(cls, text: str) -> "LLMAction":
        data = _extract_json_object(text)
        action_input = data.get("action_input", {})
        if action_input is None:
            action_input = {}
        if not isinstance(action_input, dict):
            raise ValueError(
                "LLM field 'action_input' must be a JSON object, "
                f"got {type(action_input).__name__}"
            )
        final = data.get("final")
        if final is not None and not isinstance(final, str):
            raise ValueError(
                "LLM field 'final' must be a string or null, "
                f"got {type(final).__name__}"
            )
        action = data.get("action")
        if action is not None and not isinstance(action, str):
            raise ValueError(
                "LLM field 'action' must be a string or null, "
                f"got {type(action).__name__}"
            )
        return cls(
            thought=str(data.get("thought", "")),
            action=action,
            action_input=action_input,
            final=final,
            confidence=(
                float(data["confidence"])
                if data.get("confidence") is not None
                else None
            ),
        )


@dataclass(slots=True)
class LLMResponse:
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    raw: dict[str, Any] = field(default_factory=dict)

    def to_action(self) -> LLMAction:
        return LLMAction.from_json_text(self.content)


class LLMClient(Protocol):
    def complete_json(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        raise NotImplementedError


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"LLM response does not contain a JSON object: {text[:120]}")
    return json.loads(stripped[start : end + 1])
