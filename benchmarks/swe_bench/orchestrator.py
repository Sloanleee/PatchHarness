from __future__ import annotations

import importlib.util
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping

from dotenv import load_dotenv

from benchmarks.swe_bench.models import (
    SingleCaseConfig,
    WorkerResult,
    load_case,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class PreflightError(RuntimeError):
    """Raised when the experiment cannot safely start."""


def preflight(
    project_root: Path,
    run_command: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    environ: Mapping[str, str] | None = None,
    find_spec: Callable[[str], Any] = importlib.util.find_spec,
    disk_usage: Callable[[Path], Any] = shutil.disk_usage,
) -> dict[str, Any]:
    root = project_root.resolve()
    actual_prefix = Path(sys.prefix).resolve()
    allowed_prefixes = {(root / ".venv").resolve(), (root / "venv311").resolve()}
    if actual_prefix not in allowed_prefixes:
        raise PreflightError(
            f"SWE-bench must run in a managed virtual environment: {actual_prefix}"
        )
    if find_spec("swebench") is None:
        raise PreflightError("swebench is not installed in the active environment")

    git_result = _run_probe(run_command, ["git", "--version"])
    if git_result.returncode != 0:
        raise PreflightError(f"Git is unavailable: {git_result.stderr.strip()}")
    docker_result = _run_probe(run_command, ["docker", "info"])
    if docker_result.returncode != 0:
        raise PreflightError(
            f"Docker daemon is unavailable: {docker_result.stderr.strip()}"
        )

    free_bytes = int(disk_usage(root).free)
    required_bytes = 120 * 1024**3
    if free_bytes < required_bytes:
        raise PreflightError(
            f"SWE-bench requires at least 120 GiB free; found {free_bytes / 1024**3:.1f} GiB"
        )

    if environ is None:
        load_dotenv(root / ".env", override=False)
        environ = os.environ
    missing = [name for name in ("ARK_API_KEY", "ARK_MODEL") if not environ.get(name)]
    if missing:
        raise PreflightError(f"Missing Ark configuration: {', '.join(missing)}")

    docker_version = _run_probe(
        run_command,
        ["docker", "version", "--format", "{{.Server.Version}} {{.Server.Os}} {{.Server.Arch}}"],
    )
    return {
        "interpreter": sys.executable,
        "python_version": sys.version.split()[0],
        "git_version": git_result.stdout.strip(),
        "docker_version": docker_version.stdout.strip(),
        "free_bytes": free_bytes,
    }


def run_harness(
    config: SingleCaseConfig,
    predictions: str | Path,
    phase_dir: Path,
    harness_run_id: str,
    run_command: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> bool:
    phase_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        config.dataset_name,
        "--split",
        config.split,
        "--predictions_path",
        str(predictions),
        "--instance_ids",
        config.instance_id,
        "--max_workers",
        "1",
        "--timeout",
        str(config.timeout_seconds),
        "--run_id",
        harness_run_id,
        "--cache_level",
        "env",
    ]
    completed = run_command(
        command,
        cwd=phase_dir,
        capture_output=True,
        text=True,
        timeout=config.timeout_seconds + 600,
        check=False,
    )
    (phase_dir / "harness.log").write_text(
        f"STDOUT\n{completed.stdout}\nSTDERR\n{completed.stderr}",
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"SWE-bench Harness exited {completed.returncode}: {completed.stderr[-1000:]}"
        )
    return _parse_harness_report(phase_dir, config.instance_id)


def run_worker_process(
    config_path: Path,
    workspace: Path,
    result_path: Path,
    log_path: Path,
    timeout_seconds: int,
    run_command: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> WorkerResult:
    command = [
        sys.executable,
        "-m",
        "benchmarks.swe_bench.worker",
        "--config",
        str(config_path),
        "--workspace",
        str(workspace),
        "--result",
        str(result_path),
    ]
    completed = run_command(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        f"STDOUT\n{completed.stdout}\nSTDERR\n{completed.stderr}",
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"PatchHarness worker exited {completed.returncode}: {completed.stderr[-1000:]}"
        )
    if not result_path.exists():
        raise RuntimeError("PatchHarness worker did not write its result file")
    return WorkerResult.from_dict(json.loads(result_path.read_text(encoding="utf-8")))


def run_single(
    config_path: Path,
    results_root: Path,
    run_id: str | None = None,
    reuse_gold_from: str | None = None,
    preflight_fn: Callable[[Path], dict[str, Any]] = preflight,
    harness_runner: Callable[[SingleCaseConfig, str | Path, Path, str], bool] = run_harness,
    worker_runner: Callable[..., WorkerResult] = run_worker_process,
) -> Path:
    config = load_case(config_path)
    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    _validate_run_id(run_id)
    if reuse_gold_from is not None:
        _validate_run_id(reuse_gold_from)
        if run_id == reuse_gold_from:
            raise ValueError("run_id and reuse_gold_from must be different")
    run_dir = results_root / run_id
    if run_dir.exists():
        raise FileExistsError(f"SWE-bench run already exists: {run_dir}")
    if reuse_gold_from is not None:
        validate_reusable_gold(results_root / reuse_gold_from, config)
    gold_dir = run_dir / "gold" / "evaluation"
    model_dir = run_dir / "model"
    model_evaluation_dir = model_dir / "evaluation"
    run_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "config.json", config.__dict__ if hasattr(config, "__dict__") else {
        "dataset_name": config.dataset_name,
        "split": config.split,
        "instance_id": config.instance_id,
        "selection_reason": config.selection_reason,
        "provider": config.provider,
        "max_calls": config.max_calls,
        "max_tokens": config.max_tokens,
        "timeout_seconds": config.timeout_seconds,
        "rpm_limit": config.rpm_limit,
        "tpm_limit": config.tpm_limit,
        "smoke_test_command": list(config.smoke_test_command),
        "smoke_test_timeout_seconds": config.smoke_test_timeout_seconds,
        "max_repair_attempts": config.max_repair_attempts,
    })

    metrics: dict[str, Any] = {
        "run_id": run_id,
        "instance_id": config.instance_id,
        "provider": config.provider,
        "stage": "preflight",
        "gold_resolved": None,
        "gold_reused": False,
        "gold_source_run_id": None,
        "swebench_version": _installed_package_version("swebench"),
        "model_resolved": None,
        "llm_calls": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "ark_attempts": 0,
        "ark_retries": 0,
        "ark_last_request_id": "",
        "ark_error_code": "",
        "ark_retry_after": None,
        "client_observed_rpm": 0,
        "client_observed_tpm": 0,
        "configured_rpm_limit": config.rpm_limit,
        "configured_tpm_limit": config.tpm_limit,
        "rate_limit_headers": {},
        "root_cause_status": "",
        "root_cause_evidence_count": 0,
        "root_cause_stop_reason": "",
        "validation_stage": "",
        "validation_ok": None,
        "validation_error": "",
        "repair_attempts": 0,
        "elapsed_seconds": 0.0,
        "patch_generated": False,
        "failed_stage": "",
        "failure_category": "",
        "error_summary": "",
    }

    try:
        metrics["preflight"] = preflight_fn(PROJECT_ROOT)
    except Exception as exc:
        _fail(metrics, "preflight", "preflight_error", exc)
        _write_evidence(run_dir, metrics)
        return run_dir

    if reuse_gold_from is not None:
        metrics["stage"] = "gold_reused"
        metrics["gold_resolved"] = True
        metrics["gold_reused"] = True
        metrics["gold_source_run_id"] = reuse_gold_from
    else:
        metrics["stage"] = "gold_evaluated"
        try:
            metrics["gold_resolved"] = harness_runner(
                config,
                "gold",
                gold_dir,
                f"{run_id}_gold",
            )
        except Exception as exc:
            _fail(metrics, "gold_evaluated", "harness_error", exc)
            _write_evidence(run_dir, metrics)
            return run_dir
        if not metrics["gold_resolved"]:
            _fail(
                metrics,
                "gold_evaluated",
                "gold_evaluation_failed",
                RuntimeError("Official gold patch did not resolve the selected instance"),
            )
            _write_evidence(run_dir, metrics)
            return run_dir

    worker_result_path = model_dir / "worker_result.json"
    worker_log_path = model_dir / "worker.log"
    metrics["stage"] = "model_executed"
    try:
        worker = worker_runner(
            config_path,
            run_dir / "workspace",
            worker_result_path,
            worker_log_path,
            config.timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        worker_log_path.write_text(
            _stream_text(exc.output) + _stream_text(exc.stderr),
            encoding="utf-8",
        )
        _fail(metrics, "model_executed", "run_timeout", exc)
        _write_evidence(run_dir, metrics)
        return run_dir
    except Exception as exc:
        _fail(metrics, "model_executed", "repository_setup_error", exc)
        _write_evidence(run_dir, metrics)
        return run_dir

    metrics.update(
        {
            "llm_calls": worker.llm_calls,
            "prompt_tokens": worker.prompt_tokens,
            "completion_tokens": worker.completion_tokens,
            "total_tokens": worker.prompt_tokens + worker.completion_tokens,
            "elapsed_seconds": worker.elapsed_seconds,
            "patch_generated": bool(worker.patch),
            "failure_category": worker.failure_category,
            "error_summary": _redact(worker.error_summary),
            "ark_attempts": worker.ark_attempts,
            "ark_retries": worker.ark_retries,
            "ark_last_request_id": worker.ark_last_request_id,
            "ark_error_code": worker.ark_error_code,
            "ark_retry_after": worker.ark_retry_after,
            "client_observed_rpm": worker.client_observed_rpm,
            "client_observed_tpm": worker.client_observed_tpm,
            "rate_limit_headers": worker.rate_limit_headers,
            "root_cause_status": worker.root_cause_status,
            "root_cause_evidence_count": worker.root_cause_evidence_count,
            "root_cause_stop_reason": worker.root_cause_stop_reason,
            "validation_stage": worker.validation_stage,
            "validation_ok": worker.validation_ok,
            "validation_error": _redact(worker.validation_error),
            "repair_attempts": worker.repair_attempts,
        }
    )
    patch_path = model_dir / "model.patch"
    patch_path.write_text(worker.patch, encoding="utf-8")
    prediction_path = model_dir / "predictions.jsonl"
    prediction = {
        "instance_id": config.instance_id,
        "model_name_or_path": "PatchHarness-Ark",
        "model_patch": worker.patch,
    }
    prediction_path.write_text(
        json.dumps(prediction, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    metrics["stage"] = "prediction_written"

    try:
        metrics["model_resolved"] = harness_runner(
            config,
            prediction_path,
            model_evaluation_dir,
            f"{run_id}_model",
        )
    except Exception as exc:
        _fail(metrics, "model_evaluated", "harness_error", exc)
        _write_evidence(run_dir, metrics)
        return run_dir

    if metrics["model_resolved"]:
        metrics["failure_category"] = ""
        metrics["error_summary"] = ""

    metrics["stage"] = "completed"
    _write_evidence(run_dir, metrics)
    return run_dir


def _run_probe(run_command: Callable[..., Any], command: list[str]):
    return run_command(
        command,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def _installed_package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def validate_reusable_gold(
    source_run_dir: Path,
    current_config: SingleCaseConfig,
) -> dict[str, Any]:
    if not source_run_dir.is_dir():
        raise FileNotFoundError(f"Gold source run does not exist: {source_run_dir}")
    config_path = source_run_dir / "config.json"
    metrics_path = source_run_dir / "metrics.json"
    if not config_path.is_file() or not metrics_path.is_file():
        raise ValueError("Gold source must contain config.json and metrics.json")

    source_config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = {
        "dataset_name": current_config.dataset_name,
        "split": current_config.split,
        "instance_id": current_config.instance_id,
    }
    for field, value in expected.items():
        if source_config.get(field) != value:
            raise ValueError(
                f"Gold source {field} mismatch: {source_config.get(field)!r} != {value!r}"
            )

    source_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if source_metrics.get("gold_resolved") is not True:
        raise ValueError("Gold source metrics must record gold_resolved=true")
    if source_metrics.get("failed_stage") == "gold_evaluated":
        raise ValueError("Gold source contains a gold-stage failure")

    gold_evaluation_dir = source_run_dir / "gold" / "evaluation"
    try:
        resolved = _parse_harness_report(
            gold_evaluation_dir,
            current_config.instance_id,
        )
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Gold source lacks a valid official gold artifact: {exc}") from exc
    if not resolved:
        raise ValueError("Official gold artifact is not resolved")
    return {
        "source_run_id": source_run_dir.name,
        "instance_id": current_config.instance_id,
        "gold_resolved": True,
    }


def _parse_harness_report(phase_dir: Path, instance_id: str) -> bool:
    matches = []
    for path in phase_dir.glob("logs/run_evaluation/**/report.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if instance_id in payload:
            matches.append(payload[instance_id])
    if len(matches) == 1 and "resolved" in matches[0]:
        return bool(matches[0]["resolved"])
    if len(matches) > 1:
        raise RuntimeError(
            f"Expected one Harness report for {instance_id}, found {len(matches)}"
        )

    summaries = []
    for path in phase_dir.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") == 2 and instance_id in payload.get(
            "submitted_ids", []
        ):
            summaries.append(payload)
    if len(summaries) != 1:
        raise RuntimeError(
            f"Expected one Harness report for {instance_id}, found {len(summaries)}"
        )
    summary = summaries[0]
    if instance_id in summary.get("error_ids", []):
        raise RuntimeError(f"SWE-bench Harness reported an error for {instance_id}")
    if instance_id in summary.get("resolved_ids", []):
        return True
    non_resolved_ids = set(summary.get("unresolved_ids", [])) | set(
        summary.get("empty_patch_ids", [])
    )
    if instance_id in non_resolved_ids:
        return False
    raise RuntimeError(f"SWE-bench Harness did not score {instance_id}")


def _fail(metrics: dict[str, Any], stage: str, category: str, exc: Exception) -> None:
    metrics["stage"] = stage
    metrics["failed_stage"] = stage
    metrics["failure_category"] = category
    metrics["error_summary"] = _redact(str(exc))


def _redact(text: str) -> str:
    secret = os.getenv("ARK_API_KEY", "")
    return text.replace(secret, "[REDACTED]") if secret else text


def _stream_text(value: Any) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_evidence(run_dir: Path, metrics: dict[str, Any]) -> None:
    _write_json(run_dir / "metrics.json", metrics)
    summary = "\n".join(
        [
            "# SWE-bench Single-Instance Feasibility Test",
            "",
            f"- Run ID: `{metrics['run_id']}`",
            f"- Instance: `{metrics['instance_id']}`",
            f"- Stage: `{metrics['stage']}`",
            f"- Gold resolved: `{metrics['gold_resolved']}`",
            f"- Gold reused: `{metrics['gold_reused']}`",
            f"- Gold source run: `{metrics['gold_source_run_id']}`",
            f"- Model resolved: `{metrics['model_resolved']}`",
            f"- LLM calls: `{metrics['llm_calls']}`",
            f"- Total tokens: `{metrics['total_tokens']}`",
            f"- Ark attempts: `{metrics['ark_attempts']}`",
            f"- Ark retries: `{metrics['ark_retries']}`",
            f"- Ark error code: `{metrics['ark_error_code']}`",
            f"- Ark request ID: `{metrics['ark_last_request_id']}`",
            f"- Client-observed RPM: `{metrics['client_observed_rpm']} / {metrics['configured_rpm_limit']}`",
            f"- Client-observed TPM: `{metrics['client_observed_tpm']} / {metrics['configured_tpm_limit']}`",
            f"- Root cause status: `{metrics['root_cause_status']}`",
            f"- Root cause evidence count: `{metrics['root_cause_evidence_count']}`",
            f"- Root cause stop reason: `{metrics['root_cause_stop_reason']}`",
            f"- Local validation stage: `{metrics['validation_stage']}`",
            f"- Local validation passed: `{metrics['validation_ok']}`",
            f"- Local validation error: `{metrics['validation_error']}`",
            f"- Patch repair attempts: `{metrics['repair_attempts']}`",
            f"- Patch generated: `{metrics['patch_generated']}`",
            f"- Failure category: `{metrics['failure_category']}`",
            f"- Error: `{metrics['error_summary']}`",
            "",
            "This is a single-instance feasibility result, not an aggregate SWE-bench success rate.",
        ]
    )
    (run_dir / "summary.md").write_text(summary + "\n", encoding="utf-8")


def _validate_run_id(run_id: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise ValueError(
            "run_id values may contain only letters, numbers, dot, underscore, and hyphen"
        )
