from __future__ import annotations

import os
import json
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

from app.llm.client import LLMResponse


_RETRYABLE_429_CODES = {
    "RequestBurstTooFast",
    "ServerOverloaded",
    "ConcurrentOperationLimitExceeded",
    "RateLimitExceeded",
    "Throttling",
}
_SENSITIVE_KEYS = {
    "authorization",
    "api_key",
    "access_token",
    "secret",
    "token",
    "credential",
}


class ArkAPIError(RuntimeError):
    def __init__(
        self,
        *,
        status_code: int,
        error_code: str,
        error_type: str,
        message: str,
        request_id: str,
        retry_after: float | None,
        response_body: Any,
        rate_limit_headers: dict[str, str],
        retryable: bool,
    ) -> None:
        self.status_code = status_code
        self.error_code = error_code
        self.error_type = error_type
        self.message = message
        self.request_id = request_id
        self.retry_after = retry_after
        self.response_body = response_body
        self.rate_limit_headers = rate_limit_headers
        self.retryable = retryable
        details = json.dumps(response_body, ensure_ascii=False, default=str)[:4000]
        super().__init__(
            f"Ark API {status_code} {error_code or error_type}: {message}; "
            f"request_id={request_id or 'unknown'}; body={details}"
        )


class VolcengineArkClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
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
        self.timeout = _validate_ark_timeout(
            os.getenv("ARK_TIMEOUT_SECONDS", "300") if timeout is None else timeout
        )
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
        status_code = int(getattr(response, "status_code", 200))
        if status_code >= 400:
            raise _ark_error_from_response(response, self.api_key)
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


def _validate_ark_timeout(value: Any) -> float:
    try:
        timeout = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("ARK_TIMEOUT_SECONDS must be a number between 10 and 300") from exc
    if not 10 <= timeout <= 300:
        raise ValueError("ARK_TIMEOUT_SECONDS must be between 10 and 300")
    return timeout


def _ark_error_from_response(response: Any, api_key: str) -> ArkAPIError:
    try:
        raw_body = response.json()
    except Exception:
        raw_body = {"error": {"message": str(getattr(response, "text", ""))[:4000]}}
    body = _redact_value(raw_body, api_key)
    error = body.get("error", {}) if isinstance(body, dict) else {}
    if not isinstance(error, dict):
        error = {"message": str(error)}
    headers = {
        str(key).lower(): str(value)
        for key, value in dict(getattr(response, "headers", {}) or {}).items()
    }
    diagnostic_headers = {
        key: value
        for key, value in headers.items()
        if "ratelimit" in key
        or key in {"retry-after", "x-request-id", "x-tt-logid"}
    }
    code = str(error.get("code", ""))
    error_type = str(error.get("type", ""))
    request_id = str(
        error.get("request_id")
        or (body.get("request_id") if isinstance(body, dict) else "")
        or headers.get("x-request-id")
        or headers.get("x-tt-logid")
        or ""
    )
    message = str(error.get("message", ""))
    status_code = int(getattr(response, "status_code", 0))
    return ArkAPIError(
        status_code=status_code,
        error_code=code,
        error_type=error_type,
        message=message,
        request_id=request_id,
        retry_after=_parse_retry_after(headers.get("retry-after")),
        response_body=body,
        rate_limit_headers=diagnostic_headers,
        retryable=status_code == 429 and code in _RETRYABLE_429_CODES,
    )


def _redact_value(value: Any, api_key: str, key: str = "") -> Any:
    if key.lower() in _SENSITIVE_KEYS:
        return "[REDACTED]"
    if isinstance(value, dict):
        return {
            str(item_key): _redact_value(item_value, api_key, str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, list):
        return [_redact_value(item, api_key) for item in value]
    if isinstance(value, str) and api_key:
        return value.replace(api_key, "[REDACTED]")
    return value


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None
