from __future__ import annotations

from dataclasses import asdict
from typing import Any

from app.graph import BugfixWorkflow
from app.schemas import BugfixRequest

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ModuleNotFoundError as exc:  # pragma: no cover - exercised when deps are missing
    FastAPI = None  # type: ignore[assignment]
    HTTPException = None  # type: ignore[assignment]
    BaseModel = object  # type: ignore[assignment]
    Field = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


if FastAPI is None:  # pragma: no cover
    app = None
else:
    app = FastAPI(title="PatchHarness")
    workflow = BugfixWorkflow.from_default_configs()


class BugfixRequestModel(BaseModel):  # type: ignore[misc, valid-type]
    if Field is not None:
        task_description: str = Field(..., min_length=1)
        workspace_path: str = "."
        mode: str = "auto"
        allow_edit: bool = False
        run_tests: bool = True
        test_command: str | None = None


if FastAPI is not None:

    @app.get("/health")  # type: ignore[union-attr]
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/bugfix")  # type: ignore[union-attr]
    def bugfix(payload: BugfixRequestModel) -> dict[str, Any]:
        try:
            request = BugfixRequest(**payload.model_dump())
            response = workflow.run(request)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return response.to_dict()


def require_fastapi() -> None:
    if _IMPORT_ERROR is not None:
        raise RuntimeError(
            "FastAPI dependencies are not installed. Run `python -m pip install -r requirements.txt`."
        ) from _IMPORT_ERROR


def run_once(payload: dict[str, Any]) -> dict[str, Any]:
    """Convenience entry point for scripts/tests without FastAPI."""

    request = BugfixRequest(**payload)
    response = BugfixWorkflow.from_default_configs().run(request)
    return response.to_dict()
