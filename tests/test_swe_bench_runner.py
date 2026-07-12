import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from benchmarks.swe_bench.models import SingleCaseConfig, WorkerResult, load_case
from benchmarks.swe_bench.runner import (
    collect_patch,
    ensure_git_cache,
    load_instance,
    prepare_workspace,
    run_patchharness,
)
from benchmarks.swe_bench.validation import (
    ValidationResult,
    validate_patch_static,
    run_docker_smoke_test,
)
from app.llm import ArkAPIError, LLMResponse


ROOT = Path(__file__).resolve().parents[1]
CASE_PATH = ROOT / "benchmarks" / "swe_bench" / "cases.json"


class SweBenchContractTests(unittest.TestCase):
    def test_loads_pinned_single_case(self):
        case = load_case(CASE_PATH)

        self.assertEqual(case.dataset_name, "SWE-bench/SWE-bench_Lite")
        self.assertEqual(case.split, "test")
        self.assertEqual(case.instance_id, "sympy__sympy-20590")
        self.assertEqual(case.provider, "ark")
        self.assertGreater(case.max_calls, 0)
        self.assertGreater(case.max_tokens, 0)
        self.assertGreater(case.timeout_seconds, 0)
        self.assertGreater(case.rpm_limit, 0)
        self.assertGreater(case.tpm_limit, 0)
        self.assertEqual(case.max_repair_attempts, 1)
        self.assertTrue(case.smoke_test_command)

    def test_rejects_invalid_single_case_values(self):
        valid = {
            "dataset_name": "SWE-bench/SWE-bench_Lite",
            "split": "test",
            "instance_id": "sympy__sympy-20590",
            "selection_reason": "official quickstart",
            "provider": "ark",
            "max_calls": 12,
            "max_tokens": 200_000,
            "timeout_seconds": 1_800,
        }
        invalid_cases = [
            {**valid, "max_calls": 0},
            {**valid, "instance_id": ""},
            {**valid, "provider": "deepseek"},
        ]

        for data in invalid_cases:
            with self.subTest(data=data):
                with self.assertRaises(ValueError):
                    SingleCaseConfig.from_dict(data)

    def test_rejects_missing_config_field(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "case.json"
            path.write_text(json.dumps({"provider": "ark"}), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Missing SWE-bench config fields"):
                load_case(path)

    def test_worker_result_round_trips_json_safe_data(self):
        result = WorkerResult(
            instance_id="sympy__sympy-20590",
            patch="diff --git a/a.py b/a.py",
            response={"final_summary": "done"},
            llm_calls=2,
            prompt_tokens=100,
            completion_tokens=20,
            elapsed_seconds=1.25,
            ark_attempts=4,
            ark_retries=3,
            ark_last_request_id="req-123",
            ark_error_code="RequestBurstTooFast",
            ark_retry_after=7.0,
            client_observed_rpm=4,
            client_observed_tpm=120,
            rate_limit_headers={"retry-after": "7"},
        )

        restored = WorkerResult.from_dict(result.to_dict())

        self.assertEqual(restored, result)


class SweBenchRunnerTests(unittest.TestCase):
    def setUp(self):
        self.config = SingleCaseConfig.from_dict(
            {
                "dataset_name": "SWE-bench/SWE-bench_Lite",
                "split": "test",
                "instance_id": "sympy__sympy-20590",
                "selection_reason": "official quickstart",
                "provider": "ark",
                "max_calls": 12,
                "max_tokens": 200_000,
                "timeout_seconds": 1_800,
            }
        )

    def test_load_instance_drops_gold_and_test_patches(self):
        row = {
            "instance_id": self.config.instance_id,
            "repo": "sympy/sympy",
            "base_commit": "abc123",
            "problem_statement": "Handle AttributeError in sympify.",
            "patch": "GOLD_SENTINEL",
            "test_patch": "TEST_SENTINEL",
        }

        instance = load_instance(
            self.config,
            dataset_loader=lambda name, split: [row],
        )

        self.assertEqual(
            set(instance),
            {"instance_id", "repo", "base_commit", "problem_statement"},
        )
        self.assertNotIn("GOLD_SENTINEL", json.dumps(instance))
        self.assertNotIn("TEST_SENTINEL", json.dumps(instance))

    def test_prepare_workspace_uses_argument_lists_and_base_commit(self):
        calls = []

        def fake_run(command, **kwargs):
            calls.append((command, kwargs))
            return subprocess.CompletedProcess(command, 0, "", "")

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            prepared = prepare_workspace(
                {
                    "repo": "sympy/sympy",
                    "base_commit": "abc123",
                },
                workspace,
                run_command=fake_run,
            )

        self.assertEqual(prepared, workspace)
        self.assertEqual(
            calls[0][0],
            [
                "git",
                "clone",
                "--filter=blob:none",
                "https://github.com/sympy/sympy.git",
                str(workspace),
            ],
        )
        self.assertEqual(calls[1][0], ["git", "checkout", "--detach", "abc123"])
        self.assertNotIn("shell", calls[0][1])

    def test_git_cache_is_seeded_from_swe_single_002_without_network_fetch(self):
        calls = []

        def fake_run(command, **kwargs):
            calls.append((command, kwargs))
            if command[0:3] == ["git", "-C", str(Path("cache.git"))] and "cat-file" in command:
                return subprocess.CompletedProcess(command, 0, "", "")
            return subprocess.CompletedProcess(command, 0, "", "")

        ensure_git_cache(
            {"repo": "sympy/sympy", "base_commit": "abc123"},
            Path("cache.git"),
            Path("swe_single_002/workspace"),
            run_command=fake_run,
            path_exists=lambda path: path == Path("swe_single_002/workspace"),
        )

        self.assertEqual(
            calls[0][0],
            ["git", "clone", "--bare", "swe_single_002/workspace", "cache.git"],
        )
        self.assertFalse(any("fetch" in command for command, _ in calls))

    def test_git_cache_fetches_only_when_base_commit_is_missing(self):
        calls = []

        def fake_run(command, **kwargs):
            calls.append(command)
            if "cat-file" in command:
                return subprocess.CompletedProcess(command, 1, "", "missing")
            return subprocess.CompletedProcess(command, 0, "", "")

        ensure_git_cache(
            {"repo": "sympy/sympy", "base_commit": "abc123"},
            Path("existing-cache.git"),
            Path("seed"),
            run_command=fake_run,
            path_exists=lambda path: True,
        )

        fetch = next(command for command in calls if "fetch" in command)
        self.assertEqual(fetch[-1], "abc123")

    def test_prepare_workspace_clones_from_cache_when_provided(self):
        calls = []

        def fake_run(command, **kwargs):
            calls.append(command)
            return subprocess.CompletedProcess(command, 0, "", "")

        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp) / "workspace"
            prepare_workspace(
                {"repo": "sympy/sympy", "base_commit": "abc123"},
                workspace,
                git_cache=Path("cache.git"),
                run_command=fake_run,
            )

        self.assertEqual(
            calls[0],
            ["git", "clone", "--shared", "--no-checkout", "cache.git", str(workspace)],
        )
        self.assertIn(["git", "remote", "set-url", "origin", "https://github.com/sympy/sympy.git"], calls)

    def test_collect_patch_returns_binary_diff_verbatim(self):
        expected = "diff --git a/a.py b/a.py\n+change\n"

        def fake_run(command, **kwargs):
            self.assertEqual(command, ["git", "diff", "--binary", "HEAD", "--"])
            return subprocess.CompletedProcess(command, 0, expected, "")

        patch_text = collect_patch(Path("workspace"), run_command=fake_run)

        self.assertEqual(patch_text, expected)

    def test_run_patchharness_passes_only_problem_statement(self):
        captured = {}

        class FakeWorkflow:
            def run(self, request):
                captured["request"] = request

                class Response:
                    def to_dict(self):
                        return {"final_summary": "done"}

                return Response()

        result = run_patchharness(
            self.config,
            {
                "instance_id": self.config.instance_id,
                "repo": "sympy/sympy",
                "base_commit": "abc123",
                "problem_statement": "Handle AttributeError in sympify.",
            },
            Path("workspace"),
            client_factory=lambda provider: object(),
            workflow_builder=lambda client: FakeWorkflow(),
            patch_collector=lambda workspace: "diff --git a/a.py b/a.py\n",
        )

        request = captured["request"]
        self.assertEqual(request.task_description, "Handle AttributeError in sympify.")
        self.assertTrue(request.allow_edit)
        self.assertTrue(request.enable_llm)
        self.assertFalse(request.run_tests)
        self.assertEqual(result.failure_category, "")

    def test_static_validation_rejects_duplicate_slots_assignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            source = workspace / "basic.py"
            source.write_text(
                "class Basic:\n    __slots__ = ()\n    __slots__ = ()\n",
                encoding="utf-8",
            )

            result = validate_patch_static(workspace, ["basic.py"])

        self.assertFalse(result.ok)
        self.assertIn("duplicate class assignment", result.error)

    def test_docker_smoke_test_uses_argument_list_and_isolation(self):
        calls = []

        def fake_run(command, **kwargs):
            calls.append((command, kwargs))
            if command[:3] == ["docker", "image", "inspect"]:
                return subprocess.CompletedProcess(command, 0, "image", "")
            return subprocess.CompletedProcess(command, 0, "ok", "")

        result = run_docker_smoke_test(
            Path("/workspace"),
            "sympy__sympy-20590",
            ["python", "-c", "from sympy import Basic; assert Basic()"],
            run_command=fake_run,
        )

        self.assertTrue(result.ok)
        command, kwargs = calls[1]
        self.assertEqual(command[:2], ["docker", "run"])
        self.assertIn("--network", command)
        self.assertIn("none", command)
        self.assertNotIn("shell", kwargs)

    def test_docker_smoke_test_prepares_missing_instance_image(self):
        calls = []

        def fake_run(command, **kwargs):
            calls.append(command)
            return subprocess.CompletedProcess(
                command,
                1 if command[:3] == ["docker", "image", "inspect"] else 0,
                "",
                "missing" if command[:3] == ["docker", "image", "inspect"] else "",
            )

        result = run_docker_smoke_test(
            Path("/workspace"),
            "sympy__sympy-20590",
            ["python", "-c", "assert True"],
            run_command=fake_run,
        )

        self.assertTrue(result.ok)
        self.assertIn("swebench.harness.prepare_images", calls[1])
        self.assertIn("--tag", calls[1])
        self.assertIn("--env_image_tag", calls[1])
        self.assertEqual(calls[2][:2], ["docker", "run"])

    def test_failed_validation_triggers_only_one_repair_workflow(self):
        requests = []

        class FakeWorkflow:
            def run(self, request):
                requests.append(request)

                class Response:
                    def to_dict(self):
                        return {"final_summary": "attempt"}

                return Response()

        patches = iter(["bad patch", "good patch"])
        validations = iter([
            ValidationResult(False, "docker_smoke", "smoke failed", stderr="assertion failed"),
            ValidationResult(True, "docker_smoke"),
        ])
        result = run_patchharness(
            self.config,
            {
                "instance_id": self.config.instance_id,
                "repo": "sympy/sympy",
                "base_commit": "abc123",
                "problem_statement": "Fix it.",
            },
            Path("workspace"),
            client_factory=lambda provider: object(),
            workflow_builder=lambda client: FakeWorkflow(),
            patch_collector=lambda workspace: next(patches),
            patch_validator=lambda workspace, config: next(validations),
        )

        self.assertEqual(len(requests), 2)
        self.assertIn("smoke failed", requests[1].task_description)
        self.assertEqual(result.patch, "good patch")
        self.assertTrue(result.validation_ok)
        self.assertEqual(result.repair_attempts, 1)

    def test_docker_prepare_failure_does_not_trigger_model_repair(self):
        requests = []

        class FakeWorkflow:
            def run(self, request):
                requests.append(request)

                class Response:
                    def to_dict(self):
                        return {"final_summary": "patched"}

                return Response()

        result = run_patchharness(
            self.config,
            {
                "instance_id": self.config.instance_id,
                "repo": "sympy/sympy",
                "base_commit": "abc123",
                "problem_statement": "Fix it.",
            },
            Path("workspace"),
            client_factory=lambda provider: object(),
            workflow_builder=lambda client: FakeWorkflow(),
            patch_collector=lambda workspace: "valid patch",
            patch_validator=lambda workspace, config: ValidationResult(
                False, "docker_prepare", "image preparation failed"
            ),
        )

        self.assertEqual(len(requests), 1)
        self.assertEqual(result.repair_attempts, 0)
        self.assertEqual(result.failure_category, "validation_infrastructure_error")

    def test_run_patchharness_classifies_empty_patch(self):
        class FakeWorkflow:
            def run(self, request):
                class Response:
                    def to_dict(self):
                        return {
                            "final_summary": "no change",
                            "agent_reports": [{
                                "agent_name": "root_cause_analysis",
                                "status": "partial",
                                "evidence_count": 3,
                                "stop_reason": "max_iterations_exhausted",
                            }],
                        }

                return Response()

        result = run_patchharness(
            self.config,
            {
                "instance_id": self.config.instance_id,
                "repo": "sympy/sympy",
                "base_commit": "abc123",
                "problem_statement": "Handle AttributeError in sympify.",
            },
            Path("workspace"),
            client_factory=lambda provider: object(),
            workflow_builder=lambda client: FakeWorkflow(),
            patch_collector=lambda workspace: "",
        )

        self.assertEqual(result.patch, "")
        self.assertEqual(result.failure_category, "empty_patch")
        self.assertEqual(result.root_cause_status, "partial")
        self.assertEqual(result.root_cause_evidence_count, 3)
        self.assertEqual(result.root_cause_stop_reason, "max_iterations_exhausted")

    def test_run_patchharness_classifies_ark_rate_limit(self):
        class RateLimitedClient:
            def complete_json(self, messages, **kwargs):
                raise RuntimeError("429 Too Many Requests")

        class FakeWorkflow:
            def __init__(self, client):
                self.client = client

            def run(self, request):
                self.client.complete_json([])

        result = run_patchharness(
            self.config,
            {
                "instance_id": self.config.instance_id,
                "repo": "sympy/sympy",
                "base_commit": "abc123",
                "problem_statement": "Handle AttributeError in sympify.",
            },
            Path("workspace"),
            client_factory=lambda provider: RateLimitedClient(),
            workflow_builder=lambda client: FakeWorkflow(client),
            patch_collector=lambda workspace: "",
            event_sink=lambda event: None,
        )

        self.assertEqual(result.llm_calls, 1)
        self.assertEqual(result.failure_category, "ark_rate_limited")

    def test_run_patchharness_retries_through_budget_and_records_diagnostics(self):
        class TransientClient:
            def __init__(self):
                self.calls = 0

            def complete_json(self, messages, **kwargs):
                self.calls += 1
                if self.calls == 1:
                    raise ArkAPIError(
                        status_code=429,
                        error_code="RequestBurstTooFast",
                        error_type="TooManyRequests",
                        message="slow down",
                        request_id="req-retry",
                        retry_after=0,
                        response_body={"error": {"code": "RequestBurstTooFast"}},
                        rate_limit_headers={"retry-after": "0"},
                        retryable=True,
                    )
                return LLMResponse(
                    '{"thought":"done","final":"done"}',
                    prompt_tokens=10,
                    completion_tokens=5,
                )

        raw_client = TransientClient()

        class FakeWorkflow:
            def __init__(self, client):
                self.client = client

            def run(self, request):
                self.client.complete_json([])

                class Response:
                    def to_dict(self):
                        return {"final_summary": "done"}

                return Response()

        result = run_patchharness(
            self.config,
            {
                "instance_id": self.config.instance_id,
                "repo": "sympy/sympy",
                "base_commit": "abc123",
                "problem_statement": "Handle AttributeError in sympify.",
            },
            Path("workspace"),
            client_factory=lambda provider: raw_client,
            workflow_builder=lambda client: FakeWorkflow(client),
            patch_collector=lambda workspace: "diff --git a/a.py b/a.py\n",
            retry_delays=(0,),
            sleeper=lambda seconds: None,
            event_sink=lambda event: None,
        )

        self.assertEqual(raw_client.calls, 2)
        self.assertEqual(result.llm_calls, 2)
        self.assertEqual(result.ark_attempts, 2)
        self.assertEqual(result.ark_retries, 1)
        self.assertEqual(result.ark_last_request_id, "req-retry")
        self.assertEqual(result.ark_error_code, "RequestBurstTooFast")
        self.assertEqual(result.client_observed_rpm, 2)
        self.assertEqual(result.client_observed_tpm, 15)


if __name__ == "__main__":
    unittest.main()
