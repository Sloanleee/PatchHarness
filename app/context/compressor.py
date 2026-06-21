from __future__ import annotations

from typing import Any

from app.schemas import AgentReport


class ContextCompressor:
    def __init__(self, max_tokens: int = 1200, threshold: float = 0.8) -> None:
        self.max_tokens = max_tokens
        self.threshold = threshold
        self._encoding = _load_encoding()

    def maybe_compress_report(self, report: AgentReport) -> bool:
        before_tokens = self.estimate_tokens(_report_payload(report))
        limit = int(self.max_tokens * self.threshold)
        if before_tokens <= limit:
            return False

        for observation in report.observations[:-4] or report.observations:
            data = observation.get("data")
            if isinstance(data, dict):
                for key in ("content", "stdout", "stderr"):
                    value = data.get(key)
                    if isinstance(value, str) and len(value) > 300:
                        data[key] = value[:240] + "\n...[compressed]..."

        after_tokens = self.estimate_tokens(_report_payload(report))
        report.compression_events.append(
            {
                "event": "compress",
                "before_tokens": before_tokens,
                "after_tokens": after_tokens,
                "max_tokens": self.max_tokens,
                "threshold": self.threshold,
            }
        )
        return True

    def estimate_tokens(self, value: Any) -> int:
        text = str(value)
        if self._encoding is None:
            return max(1, len(text) // 4)
        return len(self._encoding.encode(text))


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

