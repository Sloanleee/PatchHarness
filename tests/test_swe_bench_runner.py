import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from benchmarks.swe_bench.models import SingleCaseConfig, WorkerResult, load_case
from benchmarks.swe_bench.runner import (
    collect_patch,
    load_instance,
    prepare_workspace,
    run_patchharness,
)


ROOT = Path(__file__).resolve().parents[1]
CASE_PATH = ROOT / "benchmarks" / "swe_bench" / "cases.json"


class SweBenchContractTests(unittest.TestCase):
    def test_loads_pinned_single_case(self):
        case = load_case(CASE_PATH)

        self.assertEqual(case.dataset_name, "SWE-bench/SWE-bench_Lite")
        self.assertEqual(case.split, "test")
        self.assertEqual(case.instance_id, "sympy__sympy-20590")
        self.assertEqual(case.provider, "ark")
        self.assertEqual(case.max_calls, 12)
        self.assertEqual(case.max_tokens, 200_000)
        self.assertEqual(case.timeout_seconds, 1_800)

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

    def test_run_patchharness_classifies_empty_patch(self):
        class FakeWorkflow:
            def run(self, request):
                class Response:
                    def to_dict(self):
                        return {"final_summary": "no change"}

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


if __name__ == "__main__":
    unittest.main()
