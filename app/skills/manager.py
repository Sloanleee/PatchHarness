from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from app.skills.storage import InMemorySkillStorage, SkillRecord


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
    def __init__(self, skill_dir: str | Path, storage: InMemorySkillStorage | None = None) -> None:
        self.skill_dir = Path(skill_dir)
        self._frontmatter: dict[str, SkillFrontmatter] = {}
        self._content_cache: dict[str, str] = {}
        self.storage = storage or InMemorySkillStorage()
        self.sync_incremental()

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

    def search_skill(self, query: str, limit: int = 3) -> list[dict[str, Any]]:
        return self.storage.search(query, limit)

    def create_skill(
        self,
        name: str,
        description: str,
        triggers: list[str] | None = None,
        content: str = "",
    ) -> SkillFrontmatter:
        name = _normalize_skill_name(name)
        if not name:
            raise ValueError("skill name is required")
        path = self.skill_dir / f"{name}.md"
        if path.exists():
            raise ValueError(f"skill already exists: {name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        text = _render_skill_markdown(name, description, triggers or [], content)
        path.write_text(text, encoding="utf-8")
        frontmatter = self._register_path(path)
        self._content_cache[name] = content.strip()
        return frontmatter

    def update_skill(
        self,
        name: str,
        description: str | None = None,
        triggers: list[str] | None = None,
        content: str = "",
    ) -> SkillFrontmatter:
        name = _normalize_skill_name(name)
        frontmatter = self._frontmatter.get(name)
        if frontmatter is None:
            raise KeyError(f"Unknown skill: {name}")
        description = description if description is not None else frontmatter.description
        triggers = triggers if triggers is not None else frontmatter.triggers
        frontmatter.path.write_text(
            _render_skill_markdown(name, description, triggers, content),
            encoding="utf-8",
        )
        self._content_cache.pop(name, None)
        return self._register_path(frontmatter.path)

    def sync_incremental(self) -> None:
        self._frontmatter.clear()
        self._content_cache.clear()
        self._load_frontmatter()

    def choose_for_agent(self, agent_name: str) -> str | None:
        mapping = {
            "code_review": "code_review",
            "root_cause_analysis": "code_review",
            "patch_generation": "bug_fix",
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
            self._register_path(path)

    def _register_path(self, path: Path) -> SkillFrontmatter:
        raw_frontmatter, content = _split_frontmatter(path.read_text(encoding="utf-8"))
        data = yaml.safe_load(raw_frontmatter) or {}
        frontmatter = SkillFrontmatter(
            name=str(data["name"]),
            description=str(data.get("description", "")),
            triggers=list(data.get("triggers", [])),
            path=path,
        )
        self._frontmatter[frontmatter.name] = frontmatter
        self.storage.upsert(
            SkillRecord(
                name=frontmatter.name,
                description=frontmatter.description,
                triggers=frontmatter.triggers,
                content=content.strip(),
            )
        )
        return frontmatter


def _split_frontmatter(text: str) -> tuple[str, str]:
    if not text.startswith("---"):
        return "", text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return "", text
    return parts[1].strip(), parts[2]


def _normalize_skill_name(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def _render_skill_markdown(
    name: str,
    description: str,
    triggers: list[str],
    content: str,
) -> str:
    frontmatter = yaml.safe_dump(
        {
            "name": name,
            "description": description,
            "triggers": triggers,
        },
        allow_unicode=True,
        sort_keys=False,
    ).strip()
    return f"---\n{frontmatter}\n---\n\n{content.strip()}\n"
