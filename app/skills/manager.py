from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class SkillFrontmatter:
    name: str
    description: str
    triggers: list[str]
    path: Path

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "triggers": self.triggers,
        }


class SkillManager:
    def __init__(self, skill_dir: str | Path) -> None:
        self.skill_dir = Path(skill_dir)
        self._frontmatter: dict[str, SkillFrontmatter] = {}
        self._content_cache: dict[str, str] = {}
        self._load_frontmatter()

    @classmethod
    def from_default_dir(cls) -> "SkillManager":
        return cls(Path(__file__).resolve().parent / "builtin")

    def list_frontmatter(self) -> list[SkillFrontmatter]:
        return list(self._frontmatter.values())

    def public_frontmatter(self) -> list[dict[str, Any]]:
        return [item.to_public_dict() for item in self.list_frontmatter()]

    def load_skill(self, name: str) -> str:
        if name in self._content_cache:
            return self._content_cache[name]
        frontmatter = self._frontmatter.get(name)
        if frontmatter is None:
            raise KeyError(f"Unknown skill: {name}")
        text = frontmatter.path.read_text(encoding="utf-8")
        _, content = _split_frontmatter(text)
        self._content_cache[name] = content.strip()
        return self._content_cache[name]

    def choose_for_agent(self, agent_name: str) -> str | None:
        mapping = {
            "code_review": "code_review",
            "bug_fix": "bug_fix",
            "test_verify": "testing",
        }
        skill_name = mapping.get(agent_name)
        if skill_name in self._frontmatter:
            return skill_name
        return None

    def _load_frontmatter(self) -> None:
        if not self.skill_dir.exists():
            return
        for path in sorted(self.skill_dir.glob("*.md")):
            raw_frontmatter, _ = _split_frontmatter(path.read_text(encoding="utf-8"))
            data = yaml.safe_load(raw_frontmatter) or {}
            frontmatter = SkillFrontmatter(
                name=str(data["name"]),
                description=str(data.get("description", "")),
                triggers=list(data.get("triggers", [])),
                path=path,
            )
            self._frontmatter[frontmatter.name] = frontmatter


def _split_frontmatter(text: str) -> tuple[str, str]:
    if not text.startswith("---"):
        return "", text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return "", text
    return parts[1].strip(), parts[2]

