from __future__ import annotations

import json
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

from app.llm.client import LLMClient, LLMResponse
from app.llm.volcengine_client import ArkAPIError


@dataclass(frozen=True, slots=True)
class RetrySnapshot:
    attempts: int
    retries: int
    client_observed_rpm: int
    client_observed_tpm: int
    last_request_id: str = ""
    last_error_code: str = ""
    last_retry_after: float | None = None
    rate_limit_headers: dict[str, str] | None = None


class RetryingLLMClient:
    def __init__(
        self,
        inner: LLMClient,
        retry_delays: tuple[float, ...] = (5, 10, 20),
        sleeper: Callable[[float], None] = time.sleep,
        event_sink: Callable[[dict[str, Any]], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        rpm_limit: int = 500,
        tpm_limit: int = 1_000_000,
    ) -> None:
        self.inner = inner
        self.retry_delays = retry_delays
        self.sleeper = sleeper
        self.event_sink = event_sink or _stderr_event
        self.clock = clock
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self._attempts: deque[dict[str, float | int]] = deque()
        self._total_attempts = 0
        self._retries = 0
        self._last_error: ArkAPIError | None = None

    def complete_json(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> LLMResponse:
        retry_index = 0
        while True:
            try:
                response = self.inner.complete_json(messages, **kwargs)
            except ArkAPIError as exc:
                self._record_attempt(0)
                self._last_error = exc
                can_retry = exc.retryable and retry_index < len(self.retry_delays)
                delay = None
                if can_retry:
                    configured = float(self.retry_delays[retry_index])
                    delay = min(60.0, max(configured, exc.retry_after or 0.0))
                    self._retries += 1
                self.event_sink(self._failure_event(exc, delay))
                if not can_retry:
                    raise
                self.sleeper(delay)
                retry_index += 1
                continue

            tokens = response.prompt_tokens + response.completion_tokens
            self._record_attempt(tokens)
            snapshot = self.snapshot()
            self.event_sink(
                {
                    "event": "ark_request_succeeded",
                    "attempt": snapshot.attempts,
                    "prompt_tokens": response.prompt_tokens,
                    "completion_tokens": response.completion_tokens,
                    "client_observed_rpm": snapshot.client_observed_rpm,
                    "client_observed_tpm": snapshot.client_observed_tpm,
                    "configured_rpm_limit": self.rpm_limit,
                    "configured_tpm_limit": self.tpm_limit,
                }
            )
            return response

    def snapshot(self) -> RetrySnapshot:
        self._purge()
        error = self._last_error
        return RetrySnapshot(
            attempts=self._total_attempts,
            retries=self._retries,
            client_observed_rpm=len(self._attempts),
            client_observed_tpm=sum(int(item["tokens"]) for item in self._attempts),
            last_request_id=error.request_id if error else "",
            last_error_code=error.error_code if error else "",
            last_retry_after=error.retry_after if error else None,
            rate_limit_headers=dict(error.rate_limit_headers) if error else {},
        )

    def _record_attempt(self, tokens: int) -> None:
        self._total_attempts += 1
        self._attempts.append({"time": self.clock(), "tokens": tokens})
        self._purge()

    def _purge(self) -> None:
        cutoff = self.clock() - 60.0
        while self._attempts and float(self._attempts[0]["time"]) < cutoff:
            self._attempts.popleft()

    def _failure_event(
        self,
        error: ArkAPIError,
        delay: float | None,
    ) -> dict[str, Any]:
        snapshot = self.snapshot()
        return {
            "event": "ark_request_failed",
            "attempt": snapshot.attempts,
            "request_id": error.request_id,
            "status_code": error.status_code,
            "error_code": error.error_code,
            "error_type": error.error_type,
            "retryable": error.retryable,
            "retry_after": error.retry_after,
            "next_delay_seconds": delay,
            "client_observed_rpm": snapshot.client_observed_rpm,
            "client_observed_tpm": snapshot.client_observed_tpm,
            "configured_rpm_limit": self.rpm_limit,
            "configured_tpm_limit": self.tpm_limit,
            "rate_limit_headers": error.rate_limit_headers,
            "response_body": error.response_body,
        }


def _stderr_event(event: dict[str, Any]) -> None:
    print(json.dumps(event, ensure_ascii=False, default=str), file=sys.stderr, flush=True)
