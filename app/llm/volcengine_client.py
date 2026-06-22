from __future__ import annotations

import os
from typing import Any

from app.llm.client import LLMResponse


class VolcengineArkClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float = 30.0,
        http_client: Any | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("ARK_API_KEY")
        self.model = model or os.getenv("ARK_MODEL")
        self.base_url = base_url or os.getenv(
            "ARK_BASE_URL",
            "https://ark.cn-beijing.volces.com/api/v3",
        )
        if not self.api_key:
            raise RuntimeError("ARK_API_KEY is required to use VolcengineArkClient")
        if not self.model:
            raise RuntimeError("ARK_MODEL is required to use VolcengineArkClient")
        self.timeout = timeout
        if http_client is None:
            try:
                import httpx
            except ModuleNotFoundError as exc:
                raise RuntimeError("Install httpx to use VolcengineArkClient") from exc
            http_client = httpx
        self.http_client = http_client

    def complete_json(self, messages: list[dict[str, str]], **kwargs: Any) -> LLMResponse:
        payload = {
            "model": kwargs.get("model", self.model),
            "input": _messages_to_responses_input(messages),
            "temperature": kwargs.get("temperature", 0.0),
            "text": {"format": {"type": "json_object"}},
        }
        response = self.http_client.post(
            _responses_url(self.base_url),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        content = data.get("output_text") or _extract_output_text(data)
        usage = data.get("usage", {})
        return LLMResponse(
            content=content,
            prompt_tokens=_usage_value(usage, "input_tokens"),
            completion_tokens=_usage_value(usage, "output_tokens"),
            raw=data,
        )


def _messages_to_responses_input(messages: list[dict[str, str]]) -> list[dict[str, Any]]:
    return [
        {
            "role": message["role"],
            "content": [
                {
                    "type": "input_text",
                    "text": message["content"],
                }
            ],
        }
        for message in messages
    ]


def _usage_value(usage: Any, field: str) -> int:
    if usage is None:
        return 0
    if isinstance(usage, dict):
        return int(usage.get(field, 0) or 0)
    return int(getattr(usage, field, 0) or 0)


def _extract_output_text(response: Any) -> str:
    if isinstance(response, dict):
        chunks: list[str] = []
        for item in response.get("output", []) or []:
            for content in item.get("content", []) or []:
                text = content.get("text")
                if text:
                    chunks.append(text)
        return "\n".join(chunks) if chunks else str(response)

    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                chunks.append(text)
    return "\n".join(chunks) if chunks else str(response)


def _responses_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/responses"):
        return normalized
    return f"{normalized}/responses"
