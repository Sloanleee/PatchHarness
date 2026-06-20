from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class AgentConfig:
    name: str
    description: str
    system_prompt: str
    tools: list[str]
    max_iterations: int = 4
    temperature: float = 0.2
    output_schema: dict[str, Any] = field(default_factory=dict)


class AgentRegistry:
    def __init__(self) -> None:
        self._agents: dict[str, AgentConfig] = {}

    def register(self, config: AgentConfig) -> None:
        if config.name in self._agents:
            raise ValueError(f"Duplicate agent config: {config.name}")
        self._agents[config.name] = config

    def get(self, name: str) -> AgentConfig:
        try:
            return self._agents[name]
        except KeyError as exc:
            available = ", ".join(sorted(self._agents)) or "<none>"
            raise KeyError(f"Unknown agent '{name}'. Available agents: {available}") from exc

    def names(self) -> list[str]:
        return sorted(self._agents)

    @classmethod
    def load_from_dir(cls, config_dir: str | Path) -> "AgentRegistry":
        registry = cls()
        for path in sorted(Path(config_dir).glob("*.yaml")):
            registry.register(_load_agent_config(path))
        return registry


def _load_agent_config(path: Path) -> AgentConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Agent config must be a mapping: {path}")

    required = ("name", "description", "system_prompt", "tools")
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"Agent config {path} missing fields: {', '.join(missing)}")

    return AgentConfig(
        name=str(raw["name"]),
        description=str(raw["description"]),
        system_prompt=str(raw["system_prompt"]),
        tools=list(raw["tools"]),
        max_iterations=int(raw.get("max_iterations", 4)),
        temperature=float(raw.get("temperature", 0.2)),
        output_schema=dict(raw.get("output_schema", {})),
    )

