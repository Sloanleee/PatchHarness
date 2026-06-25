from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class CheckpointMissingError(RuntimeError):
    pass


class CheckpointInvalidError(RuntimeError):
    pass


class CheckpointConflictError(RuntimeError):
    pass


class CheckpointStore:
    def __init__(self, root: Path | str = ".storage/checkpoints") -> None:
        self.root = Path(root)

    def save(self, payload: dict[str, Any]) -> dict[str, Any]:
        run_id = str(payload.get("run_id") or "")
        if not run_id:
            raise CheckpointInvalidError("Checkpoint payload must include run_id")
        now = _utc_now()
        checkpoint = deepcopy(payload)
        checkpoint.setdefault("created_at", now)
        checkpoint["updated_at"] = now
        self.root.mkdir(parents=True, exist_ok=True)
        self._path(run_id).write_text(
            json.dumps(checkpoint, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return checkpoint

    def load(self, run_id: str) -> dict[str, Any]:
        path = self._path(run_id)
        if not path.exists():
            raise CheckpointMissingError(f"Checkpoint not found: {run_id}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise CheckpointInvalidError(f"Checkpoint is not valid JSON: {run_id}") from exc
        if not isinstance(data, dict) or data.get("run_id") != run_id:
            raise CheckpointInvalidError(f"Checkpoint payload does not match run_id: {run_id}")
        return data

    def update(self, run_id: str, **updates: Any) -> dict[str, Any]:
        checkpoint = self.load(run_id)
        checkpoint.update(updates)
        checkpoint["updated_at"] = _utc_now()
        self.root.mkdir(parents=True, exist_ok=True)
        self._path(run_id).write_text(
            json.dumps(checkpoint, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return checkpoint

    def _path(self, run_id: str) -> Path:
        safe_run_id = run_id.replace("/", "_").replace("\\", "_")
        return self.root / f"{safe_run_id}.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
