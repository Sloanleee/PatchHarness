from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.llm.client import LLMClient, LLMResponse


class LLMBudgetExceeded(RuntimeError):
    """Base error raised when an LLM budget prevents another request."""


class LLMCallBudgetExceeded(LLMBudgetExceeded):
    """Raised before a request would exceed the configured call count."""


class LLMTokenBudgetExceeded(LLMBudgetExceeded):
    """Raised before a request after the observed token limit is reached."""


@dataclass(frozen=True, slots=True)
class BudgetSnapshot:
    calls: int
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class BudgetedLLMClient:
    def __init__(self, inner: LLMClient, max_calls: int, max_tokens: int) -> None:
        if max_calls < 0 or max_tokens < 0:
            raise ValueError("LLM budgets must be non-negative")
        self.inner = inner
        self.max_calls = max_calls
        self.max_tokens = max_tokens
        self._calls = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0

    def complete_json(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> LLMResponse:
        snapshot = self.snapshot()
        if snapshot.calls >= self.max_calls:
            raise LLMCallBudgetExceeded(
                f"LLM call budget reached: {snapshot.calls}/{self.max_calls}"
            )
        if snapshot.total_tokens >= self.max_tokens:
            raise LLMTokenBudgetExceeded(
                f"LLM token budget reached: {snapshot.total_tokens}/{self.max_tokens}"
            )

        self._calls += 1
        response = self.inner.complete_json(messages, **kwargs)
        self._prompt_tokens += response.prompt_tokens
        self._completion_tokens += response.completion_tokens
        return response

    def snapshot(self) -> BudgetSnapshot:
        return BudgetSnapshot(
            calls=self._calls,
            prompt_tokens=self._prompt_tokens,
            completion_tokens=self._completion_tokens,
        )
