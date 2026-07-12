from __future__ import annotations

import ast
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence


@dataclass(frozen=True, slots=True)
class ValidationResult:
    ok: bool
    stage: str
    error: str = ""
    command: tuple[str, ...] = ()
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def changed_python_files(
    workspace: Path,
    run_command: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> list[str]:
    completed = run_command(
        ["git", "diff", "--name-only", "--diff-filter=ACMR", "HEAD", "--", "*.py"],
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def validate_patch_static(workspace: Path, relative_paths: Sequence[str]) -> ValidationResult:
    for relative_path in relative_paths:
        path = (workspace / relative_path).resolve()
        try:
            path.relative_to(workspace.resolve())
        except ValueError:
            return ValidationResult(False, "static", f"path escapes workspace: {relative_path}")
        if not path.is_file() or path.suffix != ".py":
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative_path)
        except (OSError, UnicodeError, SyntaxError) as exc:
            return ValidationResult(False, "static", f"Python syntax check failed: {exc}")
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            assignments: set[str] = set()
            for statement in node.body:
                names = _assignment_names(statement)
                duplicate = assignments.intersection(names)
                if duplicate:
                    name = sorted(duplicate)[0]
                    return ValidationResult(
                        False,
                        "static",
                        f"duplicate class assignment {node.name}.{name} in {relative_path}",
                    )
                assignments.update(names)
    return ValidationResult(True, "static")


def run_docker_smoke_test(
    workspace: Path,
    instance_id: str,
    smoke_command: Sequence[str],
    timeout: int = 300,
    dataset_name: str = "SWE-bench/SWE-bench_Lite",
    split: str = "test",
    run_command: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> ValidationResult:
    image = f"sweb.eval.x86_64.{instance_id.lower()}:latest"
    inspected = run_command(
        ["docker", "image", "inspect", image],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if inspected.returncode != 0:
        prepared = run_command(
            [
                sys.executable,
                "-m", "swebench.harness.prepare_images",
                "--dataset_name", dataset_name,
                "--split", split,
                "--instance_ids", instance_id,
                "--max_workers", "1",
                "--tag", "latest",
                "--env_image_tag", "latest",
            ],
            capture_output=True,
            text=True,
            timeout=max(timeout, 900),
            check=False,
        )
        if prepared.returncode != 0:
            return ValidationResult(
                False,
                "docker_prepare",
                "SWE-bench instance image preparation failed",
                returncode=prepared.returncode,
                stdout=prepared.stdout[-4000:],
                stderr=prepared.stderr[-2000:],
            )
    command = [
        "docker", "run", "--rm",
        "--network", "none",
        "--cpus", "2",
        "--memory", "4g",
        "--mount", f"type=bind,src={workspace.resolve()},dst=/testbed",
        "--workdir", "/testbed",
        image,
        *smoke_command,
    ]
    try:
        completed = run_command(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return ValidationResult(False, "docker_smoke", f"smoke test timed out after {timeout}s", tuple(command))
    except OSError as exc:
        return ValidationResult(False, "docker_smoke", str(exc), tuple(command))
    return ValidationResult(
        completed.returncode == 0,
        "docker_smoke",
        "" if completed.returncode == 0 else "Docker smoke test failed",
        tuple(command),
        completed.returncode,
        completed.stdout[-4000:],
        completed.stderr[-2000:],
    )


def validate_swe_patch(
    workspace: Path,
    instance_id: str,
    smoke_command: Sequence[str],
    timeout: int = 300,
    dataset_name: str = "SWE-bench/SWE-bench_Lite",
    split: str = "test",
) -> ValidationResult:
    paths = changed_python_files(workspace)
    static = validate_patch_static(workspace, paths)
    if not static.ok or not smoke_command:
        return static
    return run_docker_smoke_test(
        workspace, instance_id, smoke_command, timeout,
        dataset_name=dataset_name, split=split,
    )


def _assignment_names(statement: ast.stmt) -> set[str]:
    targets: list[ast.expr] = []
    if isinstance(statement, ast.Assign):
        targets = statement.targets
    elif isinstance(statement, ast.AnnAssign):
        targets = [statement.target]
    return {target.id for target in targets if isinstance(target, ast.Name)}
