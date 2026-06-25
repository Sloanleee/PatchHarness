from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.checkpoints import CheckpointStore  # noqa: E402
from app.graph import BugfixWorkflow  # noqa: E402
from app.llm import MockLLMClient, create_llm_client  # noqa: E402
from app.schemas import BugfixRequest  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "results" / "production_demo" / "runs"
DEMO_TEMPLATE = ROOT / "demo" / "hitl_project"


def main() -> None:
    args = _parse_args()
    run_dir = generate_evidence(provider=args.provider)
    print(f"Wrote production demo evidence to {run_dir}")
    print(f"Summary: {run_dir / 'summary.md'}")


def generate_evidence(
    provider: str = "mock",
    results_root: Path = RESULTS_ROOT,
    workspace_root: Path | None = None,
    run_id: str | None = None,
) -> Path:
    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = results_root / run_id
    workspace = workspace_root or (run_dir / "workspace")
    checkpoint_root = run_dir / "checkpoints"
    if workspace.exists():
        shutil.rmtree(workspace)
    shutil.copytree(DEMO_TEMPLATE, workspace)
    run_dir.mkdir(parents=True, exist_ok=True)

    workflow = BugfixWorkflow.from_default_configs()
    workflow.checkpoint_store = CheckpointStore(checkpoint_root)
    if provider == "mock":
        workflow.llm_client = MockLLMClient([])
    elif provider != "none":
        workflow.llm_client = create_llm_client(provider)

    request = BugfixRequest(
        task_description="In `.env` replace `FEATURE_FLAG=off` with `FEATURE_FLAG=on`",
        workspace_path=str(workspace),
        mode="fix",
        allow_edit=True,
        run_tests=True,
        test_command="python -m unittest discover -s tests",
        enable_llm=False,
        use_langgraph=True,
    )
    paused = workflow.run(request).to_dict()
    resumed = workflow.resume(
        str(paused["run_id"]),
        approved=True,
        reviewer="production-demo",
        comment="Approved controlled demo .env edit.",
    ).to_dict()
    trace = {
        "paused_nodes": ((paused.get("planning") or {}).get("langgraph") or {}).get("nodes", []),
        "resumed_nodes": ((resumed.get("planning") or {}).get("langgraph") or {}).get("nodes", []),
        "paused_events": (paused.get("planning") or {}).get("langgraph_events", []),
        "resumed_events": (resumed.get("planning") or {}).get("langgraph_events", []),
    }

    _write_json(run_dir / "paused_response.json", paused)
    _write_json(run_dir / "resumed_response.json", resumed)
    _write_json(run_dir / "trace.json", trace)
    _write_summary(run_dir / "summary.md", provider, paused, resumed, trace)
    return run_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LangGraph HITL checkpoint/resume demo evidence."
    )
    parser.add_argument(
        "--provider",
        default="mock",
        choices=["mock", "ark", "volcengine", "volcengine_ark", "deepseek", "none"],
    )
    return parser.parse_args()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _write_summary(
    path: Path,
    provider: str,
    paused: dict[str, Any],
    resumed: dict[str, Any],
    trace: dict[str, Any],
) -> None:
    paused_nodes = trace["paused_nodes"]
    resumed_nodes = trace["resumed_nodes"]
    content = [
        "# LangGraph HITL Checkpoint/Resume Demo",
        "",
        "## Setup",
        "",
        f"- Provider: `{provider}`",
        "- Demo workspace: `demo/hitl_project` copied into an isolated run workspace.",
        "- Sensitive target: `.env`",
        "- Requested edit: `FEATURE_FLAG=off` -> `FEATURE_FLAG=on`",
        "",
        "## Evidence",
        "",
        f"- Paused requires human approval: `{paused.get('requires_human_approval')}`",
        f"- Paused failure reason: `{paused.get('failure_reason')}`",
        f"- Resumed requires human approval: `{resumed.get('requires_human_approval')}`",
        f"- Test return code: `{((resumed.get('test_result') or {}).get('returncode'))}`",
        f"- Changed files: `{', '.join(resumed.get('changed_files') or [])}`",
        "",
        "## LangGraph Trace",
        "",
        f"- Paused nodes: `{' -> '.join(paused_nodes)}`",
        f"- Resumed nodes: `{' -> '.join(resumed_nodes)}`",
        "",
        "## HITL Checkpoint",
        "",
        "The first run stops at `hitl_pause` because editing `.env` is sensitive. "
        "After explicit approval, the workflow applies the pending edit and resumes at `test_verify`.",
        "",
    ]
    path.write_text("\n".join(content), encoding="utf-8")


if __name__ == "__main__":
    main()
