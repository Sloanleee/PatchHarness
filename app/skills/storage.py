from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class SkillRecord:
    name: str
    description: str
    triggers: list[str]
    content: str


class InMemorySkillStorage:
    def __init__(self) -> None:
        self.records: dict[str, SkillRecord] = {}

    def upsert(self, record: SkillRecord) -> None:
        self.records[record.name] = record

    def get(self, name: str) -> SkillRecord | None:
        return self.records.get(name)

    def search(self, query: str, limit: int = 3) -> list[dict[str, Any]]:
        terms = set(query.lower().split())
        scored = []
        for record in self.records.values():
            haystack = " ".join([record.name, record.description, *record.triggers, record.content]).lower()
            score = sum(1 for term in terms if term and term in haystack)
            if query.lower() in haystack:
                score += 2
            if score > 0:
                scored.append((score, record))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            {
                "name": record.name,
                "description": record.description,
                "triggers": record.triggers,
                "score": score,
            }
            for score, record in scored[:limit]
        ]


class ChromaRedisSkillStorage(InMemorySkillStorage):
    """Optional storage facade.

    The class validates optional dependencies and keeps the same API as the
    in-memory storage. A production deployment can provide concrete ChromaDB and
    Redis clients without changing SkillManager call sites.
    """

    def __init__(self, chroma_client: Any | None = None, redis_client: Any | None = None) -> None:
        super().__init__()
        self.chroma_client = chroma_client
        self.redis_client = redis_client

