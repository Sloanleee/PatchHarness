from __future__ import annotations

import json
from collections import deque
from typing import Any, Iterable

from app.llm.client import LLMAction, LLMResponse


class MockLLMClient:
    def __init__(self, actions: Iterable[LLMAction | dict[str, Any]] | None = None) -> None:
        self.actions = deque(actions or [])
        self.calls = 0

    def complete_json(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        self.calls += 1
        if self.actions:
            action = self.actions.popleft()
            if isinstance(action, LLMAction):
                payload = {
                    "thought": action.thought,
                    "action": action.action,
                    "action_input": action.action_input,
                    "final": action.final,
                    "confidence": action.confidence,
                }
            else:
                payload = action
        else:
            payload = {
                "thought": "No scripted action remains.",
                "action": None,
                "action_input": {},
                "final": "Mock LLM finished.",
                "confidence": 0.9,
            }
        content = json.dumps(payload, ensure_ascii=False)
        return LLMResponse(content=content, prompt_tokens=20, completion_tokens=10, raw=payload)

