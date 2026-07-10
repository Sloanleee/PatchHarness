from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class SingleCaseConfig:
    dataset_name: str
    split: str
    instance_id: str
    selection_reason: str
    provider: str
    max_calls: int
    max_tokens: int
    timeout_seconds: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SingleCaseConfig":
        required = {
            "dataset_name",
            "split",
            "instance_id",
            "selection_reason",
            "provider",
            "max_calls",
            "max_tokens",
            "timeout_seconds",
        }
        missing = sorted(required - data.keys())
        if missing:
            raise ValueError(
                f"Missing SWE-bench config fields: {', '.join(missing)}"
            )

        config = cls(
            dataset_name=str(data["dataset_name"]),
            split=str(data["split"]),
            instance_id=str(data["instance_id"]),
            selection_reason=str(data["selection_reason"]),
            provider=str(data["provider"]),
            max_calls=int(data["max_calls"]),
            max_tokens=int(data["max_tokens"]),
            timeout_seconds=int(data["timeout_seconds"]),
        )
        if not config.instance_id:
            raise ValueError("instance_id must not be empty")
        if config.provider != "ark":
            raise ValueError("single-instance provider must be ark")
        if min(config.max_calls, config.max_tokens, config.timeout_seconds) <= 0:
            raise ValueError("budgets and timeout must be positive")
        return config


@dataclass(slots=True)
class WorkerResult:
    instance_id: str
    patch: str
    response: dict[str, Any] | None
    llm_calls: int
    prompt_tokens: int
    completion_tokens: int
    elapsed_seconds: float
    failure_category: str = ""
    error_summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WorkerResult":
        return cls(
            instance_id=str(data["instance_id"]),
            patch=str(data.get("patch", "")),
            response=data.get("response"),
            llm_calls=int(data.get("llm_calls", 0)),
            prompt_tokens=int(data.get("prompt_tokens", 0)),
            completion_tokens=int(data.get("completion_tokens", 0)),
            elapsed_seconds=float(data.get("elapsed_seconds", 0.0)),
            failure_category=str(data.get("failure_category", "")),
            error_summary=str(data.get("error_summary", "")),
        )


def load_case(path: Path) -> SingleCaseConfig:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("SWE-bench case config must be a JSON object")
    return SingleCaseConfig.from_dict(data)
