from __future__ import annotations

import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from app.graph import BugfixWorkflow
from app.llm import (
    ArkAPIError,
    BudgetedLLMClient,
    LLMCallBudgetExceeded,
    LLMTokenBudgetExceeded,
    RetryingLLMClient,
    create_llm_client,
)
from app.schemas import BugfixRequest
from benchmarks.swe_bench.models import SingleCaseConfig, WorkerResult
from benchmarks.swe_bench.validation import ValidationResult, validate_swe_patch


_REPO_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


def load_instance(
    config: SingleCaseConfig,
    dataset_loader: Callable[..., Any] | None = None,
) -> dict[str, str]:
    if dataset_loader is None:
        from datasets import load_dataset

        dataset_loader = load_dataset

    rows = dataset_loader(config.dataset_name, split=config.split)
    matches = [row for row in rows if row["instance_id"] == config.instance_id]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one SWE-bench instance, found {len(matches)}"
        )
    row = matches[0]
    return {
        "instance_id": str(row["instance_id"]),
        "repo": str(row["repo"]),
        "base_commit": str(row["base_commit"]),
        "problem_statement": str(row["problem_statement"]),
    }


def prepare_workspace(
    instance: dict[str, str],
    workspace: Path,
    git_cache: Path | None = None,
    run_command: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> Path:
    repo = instance["repo"]
    if not _REPO_PATTERN.fullmatch(repo):
        raise ValueError(f"Invalid SWE-bench repository name: {repo}")
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.parent.mkdir(parents=True, exist_ok=True)

    if git_cache is None:
        clone_command = [
            "git", "clone", "--filter=blob:none",
            f"https://github.com/{repo}.git", str(workspace),
        ]
    else:
        clone_command = [
            "git", "clone", "--shared", "--no-checkout",
            str(git_cache), str(workspace),
        ]
    run_command(
        clone_command,
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if git_cache is not None:
        run_command(
            ["git", "remote", "set-url", "origin", f"https://github.com/{repo}.git"],
            cwd=workspace,
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )
    run_command(
        ["git", "checkout", "--detach", instance["base_commit"]],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
    return workspace


def ensure_git_cache(
    instance: dict[str, str],
    cache: Path,
    seed_workspace: Path,
    run_command: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    path_exists: Callable[[Path], bool] = Path.exists,
) -> Path:
    repo = instance["repo"]
    if not _REPO_PATTERN.fullmatch(repo):
        raise ValueError(f"Invalid SWE-bench repository name: {repo}")
    remote = f"https://github.com/{repo}.git"
    if not path_exists(cache):
        cache.parent.mkdir(parents=True, exist_ok=True)
        if path_exists(seed_workspace):
            clone_command = ["git", "clone", "--bare", str(seed_workspace), str(cache)]
        else:
            clone_command = ["git", "clone", "--bare", "--filter=blob:none", remote, str(cache)]
        run_command(
            clone_command, check=True, capture_output=True, text=True, timeout=300,
        )
        run_command(
            ["git", "-C", str(cache), "remote", "set-url", "origin", remote],
            check=True, capture_output=True, text=True, timeout=60,
        )

    commit_check = run_command(
        ["git", "-C", str(cache), "cat-file", "-e", f"{instance['base_commit']}^{{commit}}"],
        capture_output=True, text=True, timeout=60, check=False,
    )
    if commit_check.returncode != 0:
        run_command(
            [
                "git", "-C", str(cache), "fetch", "--filter=blob:none",
                "origin", instance["base_commit"],
            ],
            check=True, capture_output=True, text=True, timeout=300,
        )
    return cache


def collect_patch(
    workspace: Path,
    run_command: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> str:
    completed = run_command(
        ["git", "diff", "--binary", "HEAD", "--"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return completed.stdout


def run_patchharness(
    config: SingleCaseConfig,
    instance: dict[str, str],
    workspace: Path,
    client_factory: Callable[[str], Any] = create_llm_client,
    workflow_builder: Callable[[Any], Any] | None = None,
    patch_collector: Callable[[Path], str] = collect_patch,
    retry_delays: tuple[float, ...] = (5, 10, 20),
    sleeper: Callable[[float], None] = time.sleep,
    event_sink: Callable[[dict[str, Any]], None] | None = None,
    patch_validator: Callable[[Path, SingleCaseConfig], ValidationResult] | None = None,
) -> WorkerResult:
    start = time.perf_counter()
    response_data: dict[str, Any] | None = None
    failure_category = ""
    error_summary = ""
    patch_text = ""
    validation: ValidationResult | None = None
    repair_attempts = 0

    inner = client_factory(config.provider)
    budgeted = BudgetedLLMClient(
        inner,
        max_calls=config.max_calls,
        max_tokens=config.max_tokens,
    )
    retrying = RetryingLLMClient(
        budgeted,
        retry_delays=retry_delays,
        sleeper=sleeper,
        event_sink=event_sink,
        rpm_limit=config.rpm_limit,
        tpm_limit=config.tpm_limit,
    )
    if workflow_builder is None:
        workflow_builder = _build_workflow

    try:
        workflow = workflow_builder(retrying)
        request = BugfixRequest(
                task_description=instance["problem_statement"],
                workspace_path=str(workspace),
                mode="fix",
                allow_edit=True,
                run_tests=False,
                enable_llm=True,
                use_langgraph=False,
            )
        response = workflow.run(request)
        response_data = response.to_dict()
        patch_text = patch_collector(workspace)
        if config.smoke_test_command or patch_validator is not None:
            validator = patch_validator or _validate_configured_patch
            validation = validator(workspace, config)
            response_data["swe_validation"] = validation.to_dict()
            repairable_validation_stages = {"static", "docker_smoke"}
            if (
                not validation.ok
                and validation.stage in repairable_validation_stages
                and config.max_repair_attempts
            ):
                repair_request = BugfixRequest(
                    task_description=(
                        instance["problem_statement"]
                        + "\n\nThe first patch failed local validation. Repair the existing patch once. "
                        + f"Validation stage: {validation.stage}. Error: {validation.error}. "
                        + f"stderr: {validation.stderr[-1000:]}"
                    ),
                    workspace_path=str(workspace),
                    mode="fix",
                    allow_edit=True,
                    run_tests=False,
                    enable_llm=True,
                    use_langgraph=False,
                )
                response = workflow.run(repair_request)
                response_data = response.to_dict()
                patch_text = patch_collector(workspace)
                validation = validator(workspace, config)
                response_data["swe_validation"] = validation.to_dict()
                response_data["swe_repair_attempts"] = 1
                repair_attempts = 1
                if not validation.ok:
                    failure_category = (
                        "patch_validation_failed"
                        if validation.stage in repairable_validation_stages
                        else "validation_infrastructure_error"
                    )
                    error_summary = validation.error
            elif not validation.ok:
                failure_category = "validation_infrastructure_error"
                error_summary = validation.error
    except LLMCallBudgetExceeded as exc:
        failure_category = "call_budget_exceeded"
        error_summary = str(exc)
    except LLMTokenBudgetExceeded as exc:
        failure_category = "token_budget_exceeded"
        error_summary = str(exc)
    except Exception as exc:
        if isinstance(exc, ArkAPIError):
            failure_category = _ark_failure_category(exc)
        elif _is_rate_limit_error(exc):
            failure_category = "ark_rate_limited"
        else:
            failure_category = "ark_timeout" if _is_timeout_error(exc) else "tool_error"
        error_summary = str(exc)

    try:
        if not patch_text:
            patch_text = patch_collector(workspace)
    except Exception as exc:
        if not failure_category:
            failure_category = "repository_setup_error"
            error_summary = str(exc)

    if not patch_text and not failure_category:
        failure_category = "empty_patch"

    snapshot = budgeted.snapshot()
    retry_snapshot = retrying.snapshot()
    root_cause = next(
        (
            report
            for report in (response_data or {}).get("agent_reports", [])
            if report.get("agent_name") == "root_cause_analysis"
        ),
        {},
    )
    return WorkerResult(
        instance_id=instance["instance_id"],
        patch=patch_text,
        response=response_data,
        llm_calls=snapshot.calls,
        prompt_tokens=snapshot.prompt_tokens,
        completion_tokens=snapshot.completion_tokens,
        elapsed_seconds=time.perf_counter() - start,
        failure_category=failure_category,
        error_summary=error_summary,
        ark_attempts=retry_snapshot.attempts,
        ark_retries=retry_snapshot.retries,
        ark_last_request_id=retry_snapshot.last_request_id,
        ark_error_code=retry_snapshot.last_error_code,
        ark_retry_after=retry_snapshot.last_retry_after,
        client_observed_rpm=retry_snapshot.client_observed_rpm,
        client_observed_tpm=retry_snapshot.client_observed_tpm,
        rate_limit_headers=retry_snapshot.rate_limit_headers or {},
        root_cause_status=str(root_cause.get("status", "")),
        root_cause_evidence_count=int(root_cause.get("evidence_count", 0)),
        root_cause_stop_reason=str(root_cause.get("stop_reason", "")),
        validation_stage=validation.stage if validation else "",
        validation_ok=validation.ok if validation else None,
        validation_error=validation.error if validation else "",
        repair_attempts=repair_attempts,
    )


def _build_workflow(llm_client: Any) -> BugfixWorkflow:
    defaults = BugfixWorkflow.from_default_configs()
    return BugfixWorkflow(
        registry=defaults.registry,
        planner=defaults.planner,
        llm_client=llm_client,
    )


def _validate_configured_patch(workspace: Path, config: SingleCaseConfig) -> ValidationResult:
    return validate_swe_patch(
        workspace,
        config.instance_id,
        config.smoke_test_command,
        config.smoke_test_timeout_seconds,
        dataset_name=config.dataset_name,
        split=config.split,
    )


def _is_timeout_error(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    text = str(exc).lower()
    return "timeout" in name or "timed out" in text or "timeout" in text


def _is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "429" in text or "too many requests" in text


def _ark_failure_category(error: ArkAPIError) -> str:
    categories = {
        "RequestBurstTooFast": "ark_burst_limited",
        "ServerOverloaded": "ark_server_overloaded",
        "ConcurrentOperationLimitExceeded": "ark_concurrency_limited",
        "RateLimitExceeded": "ark_rate_limited",
        "Throttling": "ark_rate_limited",
        "AccountQuotaExceeded": "ark_account_quota_exceeded",
        "QuotaExceeded": "ark_quota_exceeded",
    }
    return categories.get(error.error_code, "ark_rate_limited_unknown")
