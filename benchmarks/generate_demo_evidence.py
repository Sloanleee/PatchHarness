from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import run_once  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "demo_cases"
SOURCE_WORKSPACE = ROOT / "demo" / "buggy_project"
RUNTIME_ROOT = ROOT / "demo" / ".runtime" / "evidence"


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if RUNTIME_ROOT.exists():
        shutil.rmtree(RUNTIME_ROOT)
    RUNTIME_ROOT.mkdir(parents=True, exist_ok=True)

    cases = [
        {
            "name": "review_only",
            "workspace": _copy_workspace("review_only"),
            "payload": {
                "task_description": "请审查 `calculator.py` 的加法实现",
                "mode": "review",
                "allow_edit": False,
                "run_tests": False,
            },
        },
        {
            "name": "fix_and_verify",
            "workspace": _copy_workspace("fix_and_verify"),
            "payload": {
                "task_description": "修复 bug：在 `calculator.py` 中将 `return a - b` 替换为 `return a + b`",
                "mode": "fix",
                "allow_edit": True,
                "run_tests": True,
                "test_command": "python -m unittest discover -s tests",
            },
        },
        {
            "name": "hitl_sensitive_edit",
            "workspace": _copy_workspace("hitl_sensitive_edit"),
            "prepare": _write_env_file,
            "payload": {
                "task_description": "修复配置：在 `.env` 中将 `TOKEN=old` 替换为 `TOKEN=new`",
                "mode": "fix",
                "allow_edit": True,
                "run_tests": False,
            },
        },
    ]

    for case in cases:
        workspace = case["workspace"]
        prepare = case.get("prepare")
        if prepare is not None:
            prepare(workspace)
        payload = dict(case["payload"])
        payload["workspace_path"] = str(workspace)
        response = run_once(payload)
        path = RESULTS_DIR / f"{case['name']}.json"
        path.write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {path}")


def _copy_workspace(name: str) -> Path:
    target = RUNTIME_ROOT / name
    shutil.copytree(SOURCE_WORKSPACE, target)
    return target


def _write_env_file(workspace: Path) -> None:
    (workspace / ".env").write_text("TOKEN=old\n", encoding="utf-8")


if __name__ == "__main__":
    main()

