from __future__ import annotations

import time
from typing import Any, Callable

from app.llm import LLMClient
from app.metrics import MetricsTracker
from app.schemas import AgentReport


class ContextCompressor:
    def __init__(
        self,
        max_tokens: int = 1200,
        threshold: float = 0.8,
        keep_recent: int = 4,
        llm_client: LLMClient | None = None,
        metrics: MetricsTracker | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        network_retry_delay: float = 2.0,
    ) -> None:
        self.max_tokens = max_tokens
        self.threshold = threshold
        self.keep_recent = keep_recent
        self.llm_client = llm_client
        self.metrics = metrics
        self.sleeper = sleeper
        self.network_retry_delay = network_retry_delay
        self._fallback_events: list[dict[str, Any]] = []
        self._encoding = _load_encoding()

    def maybe_compress_report(self, report: AgentReport) -> bool:
        before_tokens = self.estimate_tokens(_report_payload(report))
        limit = int(self.max_tokens * self.threshold)
        if before_tokens <= limit:
            return False

        protected_start = max(0, len(report.observations) - self.keep_recent)
        candidates = report.observations[:protected_start]
        if not candidates:
            return False

        compressed_fields = 0
        self._fallback_events = []
        for observation in candidates:
            data = observation.get("data")
            if isinstance(data, dict):
                for key in ("content", "stdout", "stderr"):
                    value = data.get(key)
                    if isinstance(value, str) and len(value) > 300:
                        data[key] = self._summarize(value)
                        compressed_fields += 1

        after_tokens = self.estimate_tokens(_report_payload(report))
        report.compression_events.append(
            {
                "event": "compress",
                "before_tokens": before_tokens,
                "after_tokens": after_tokens,
                "max_tokens": self.max_tokens,
                "threshold": self.threshold,
                "keep_recent": self.keep_recent,
                "compressed_fields": compressed_fields,
                "tokenizer": "tiktoken:cl100k_base" if self._encoding is not None else "char_estimate",
            }
        )
        report.compression_events.extend(self._fallback_events)
        return compressed_fields > 0

    def estimate_tokens(self, value: Any) -> int:
        text = str(value)
        if self._encoding is None:
            return max(1, len(text) // 4)
        return len(self._encoding.encode(text))

    def _summarize(self, value: str) -> str:
        if self.llm_client is None:
            return value[:240] + "\n...[compressed]..."
        messages = [
            {"role": "system", "content": "Summarize this observation. Return JSON only: {\"summary\": string}."},
            {"role": "user", "content": value[:8000]},
        ]
        attempts = 0
        try:
            attempts = 1
            response = self.llm_client.complete_json(messages, temperature=0.0)
        except Exception as first_error:
            if _is_transient_network_error(first_error):
                self.sleeper(self.network_retry_delay)
                try:
                    attempts = 2
                    response = self.llm_client.complete_json(messages, temperature=0.0)
                except Exception as second_error:
                    return self._local_fallback(value, second_error, attempts)
            else:
                return self._local_fallback(value, first_error, attempts)
        if self.metrics is not None:
            self.metrics.llm_called(response.prompt_tokens, response.completion_tokens)
        import json

        try:
            data = json.loads(response.content)
            summary = str(data.get("summary", "")).strip()
        except Exception:
            summary = response.content.strip()
        return summary[:600] + "\n...[llm_summary]..."

    def _local_fallback(self, value: str, error: Exception, attempts: int) -> str:
        reason = "network_timeout" if _is_timeout_error(error) else (
            "network_connection" if _is_transient_network_error(error) else "provider_error"
        )
        self._fallback_events.append({
            "event": "compression_fallback",
            "compression_mode": "local_fallback",
            "fallback_reason": reason,
            "error_type": type(error).__name__,
            "llm_attempts": attempts,
            "fallback": "local_head_tail",
        })
        head = value[:400]
        tail = value[-200:] if len(value) > 600 else ""
        return head + "\n...[compression fallback]...\n" + tail


def _is_timeout_error(error: Exception) -> bool:
    text = f"{type(error).__name__} {error}".lower()
    return isinstance(error, TimeoutError) or "timeout" in text or "timed out" in text


def _is_transient_network_error(error: Exception) -> bool:
    text = f"{type(error).__name__} {error}".lower()
    return _is_timeout_error(error) or isinstance(error, ConnectionError) or any(
        marker in text for marker in ("connection reset", "connection aborted", "connecterror", "readerror")
    )


def _load_encoding() -> Any | None:
    try:
        import tiktoken
    except ModuleNotFoundError:
        return None
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def _report_payload(report: AgentReport) -> dict[str, Any]:
    return {
        "thoughts": report.thoughts,
        "actions": report.actions,
        "observations": report.observations,
        "summary": report.summary,
    }
