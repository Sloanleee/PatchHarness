from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import run_once


ROOT = Path(__file__).resolve().parent
SOURCE_WORKSPACE = ROOT / "buggy_project"
WORKSPACE = ROOT / ".runtime" / "buggy_project"


if __name__ == "__main__":
    if WORKSPACE.exists():
        shutil.rmtree(WORKSPACE)
    shutil.copytree(SOURCE_WORKSPACE, WORKSPACE)

    response = run_once(
        {
            "task_description": "修复 bug：在 `calculator.py` 中将 `return a - b` 替换为 `return a + b`",
            "workspace_path": str(WORKSPACE),
            "mode": "fix",
            "allow_edit": True,
            "run_tests": True,
            "test_command": "python -m unittest discover -s tests",
        }
    )
    print(json.dumps(response, ensure_ascii=False, indent=2))
