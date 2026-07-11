import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from benchmarks.swe_bench.models import SingleCaseConfig, WorkerResult
from benchmarks.swe_bench.orchestrator import (
    PreflightError,
    preflight,
    run_harness,
    run_single,
)


CASE = {
    "dataset_name": "SWE-bench/SWE-bench_Lite",
    "split": "test",
    "instance_id": "sympy__sympy-20590",
    "selection_reason": "official quickstart",
    "provider": "ark",
    "max_calls": 12,
    "max_tokens": 200000,
    "timeout_seconds": 1800,
}


class SweBenchPreflightTests(unittest.TestCase):
    def test_rejects_interpreter_outside_managed_environments_before_commands(self):
        calls = []
        with tempfile.TemporaryDirectory() as tmp, patch(
            "benchmarks.swe_bench.orchestrator.sys.prefix",
            str(Path(tmp) / "system-python"),
        ):
            with self.assertRaisesRegex(PreflightError, "managed virtual environment"):
                preflight(Path(tmp), run_command=lambda *args, **kwargs: calls.append(args))

        self.assertEqual(calls, [])

    def test_rejects_unavailable_docker_before_ark_use(self):
        calls = []

        def fake_run(command, **kwargs):
            calls.append(command)
            return subprocess.CompletedProcess(
                command,
                0 if command[0] == "git" else 1,
                "git version 2" if command[0] == "git" else "",
                "daemon unavailable" if command[0] == "docker" else "",
            )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "venv311").mkdir()
            with patch(
                "benchmarks.swe_bench.orchestrator.sys.prefix",
                str(root / "venv311"),
            ):
                with self.assertRaisesRegex(PreflightError, "Docker daemon"):
                    preflight(
                        root,
                        run_command=fake_run,
                        environ={"ARK_API_KEY": "secret", "ARK_MODEL": "model"},
                        find_spec=lambda name: object(),
                        disk_usage=lambda path: SimpleNamespace(free=200 * 1024**3),
                    )

        self.assertEqual(calls, [["git", "--version"], ["docker", "info"]])


class SweBenchOrchestratorTests(unittest.TestCase):
    def _config(self, root):
        path = root / "case.json"
        path.write_text(json.dumps(CASE), encoding="utf-8")
        return path

    def _gold_source(self, results_root, run_id="source_gold", **overrides):
        source = results_root / run_id
        config = {**CASE, **overrides}
        source.mkdir(parents=True)
        (source / "config.json").write_text(json.dumps(config), encoding="utf-8")
        (source / "metrics.json").write_text(
            json.dumps(
                {
                    "run_id": run_id,
                    "instance_id": config["instance_id"],
                    "gold_resolved": True,
                    "failed_stage": "",
                    "failure_category": "",
                }
            ),
            encoding="utf-8",
        )
        report = source / "gold" / "evaluation" / "logs" / "run_evaluation" / "report.json"
        report.parent.mkdir(parents=True)
        report.write_text(
            json.dumps({config["instance_id"]: {"resolved": True}}),
            encoding="utf-8",
        )
        return source

    def test_gold_failure_short_circuits_worker(self):
        worker_calls = []
        harness_calls = []

        def fake_harness(config, predictions, phase_dir, harness_run_id):
            harness_calls.append(predictions)
            return False

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = run_single(
                self._config(root),
                root / "results",
                "run_gold_fail",
                preflight_fn=lambda project_root: {"interpreter": "venv311/bin/python"},
                harness_runner=fake_harness,
                worker_runner=lambda *args: worker_calls.append(args),
            )
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))

        self.assertEqual(harness_calls, ["gold"])
        self.assertEqual(worker_calls, [])
        self.assertEqual(metrics["failure_category"], "gold_evaluation_failed")
        self.assertFalse(metrics["gold_resolved"])

    def test_worker_timeout_is_recorded_and_preserves_log(self):
        def timeout_worker(*args):
            raise subprocess.TimeoutExpired(args[0], 1800, output="partial stdout")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = run_single(
                self._config(root),
                root / "results",
                "run_timeout",
                preflight_fn=lambda project_root: {"interpreter": "venv311/bin/python"},
                harness_runner=lambda *args: True,
                worker_runner=timeout_worker,
            )
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            worker_log = (run_dir / "model" / "worker.log").read_text(encoding="utf-8")

        self.assertEqual(metrics["failure_category"], "run_timeout")
        self.assertIn("partial stdout", worker_log)

    def test_success_writes_prediction_and_official_model_score(self):
        harness_predictions = []

        def fake_harness(config, predictions, phase_dir, harness_run_id):
            harness_predictions.append(predictions)
            return True if predictions == "gold" else False

        worker = WorkerResult(
            instance_id=CASE["instance_id"],
            patch="diff --git a/a.py b/a.py\n",
            response={"final_summary": "patched"},
            llm_calls=2,
            prompt_tokens=100,
            completion_tokens=20,
            elapsed_seconds=1.5,
            ark_attempts=2,
            ark_retries=1,
            ark_last_request_id="req-123",
            ark_error_code="RequestBurstTooFast",
            ark_retry_after=5.0,
            client_observed_rpm=2,
            client_observed_tpm=120,
            rate_limit_headers={"retry-after": "5"},
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = run_single(
                self._config(root),
                root / "results",
                "run_scored",
                preflight_fn=lambda project_root: {"interpreter": "venv311/bin/python"},
                harness_runner=fake_harness,
                worker_runner=lambda *args: worker,
            )
            prediction = json.loads(
                (run_dir / "model" / "predictions.jsonl").read_text(encoding="utf-8")
            )
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))

        self.assertEqual(harness_predictions[0], "gold")
        self.assertTrue(Path(harness_predictions[1]).name == "predictions.jsonl")
        self.assertEqual(prediction["instance_id"], CASE["instance_id"])
        self.assertEqual(prediction["model_patch"], worker.patch)
        self.assertTrue(metrics["gold_resolved"])
        self.assertFalse(metrics["model_resolved"])
        self.assertEqual(metrics["llm_calls"], 2)
        self.assertEqual(metrics["ark_attempts"], 2)
        self.assertEqual(metrics["ark_retries"], 1)
        self.assertEqual(metrics["ark_last_request_id"], "req-123")
        self.assertEqual(metrics["ark_error_code"], "RequestBurstTooFast")
        self.assertEqual(metrics["client_observed_rpm"], 2)
        self.assertEqual(metrics["client_observed_tpm"], 120)

    def test_error_output_redacts_ark_key(self):
        secret = "secret-ark-key"

        def failing_preflight(project_root):
            raise PreflightError(f"request failed with {secret}")

        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            "os.environ", {"ARK_API_KEY": secret}, clear=False
        ):
            root = Path(tmp)
            run_dir = run_single(
                self._config(root),
                root / "results",
                "run_secret",
                preflight_fn=failing_preflight,
            )
            summary = (run_dir / "summary.md").read_text(encoding="utf-8")
            metrics_text = (run_dir / "metrics.json").read_text(encoding="utf-8")

        self.assertNotIn(secret, summary)
        self.assertNotIn(secret, metrics_text)
        self.assertIn("[REDACTED]", metrics_text)

    def test_harness_empty_patch_summary_is_scored_unresolved(self):
        with tempfile.TemporaryDirectory() as tmp:
            phase_dir = Path(tmp)
            summary = {
                "schema_version": 2,
                "submitted_ids": [CASE["instance_id"]],
                "resolved_ids": [],
                "unresolved_ids": [],
                "empty_patch_ids": [CASE["instance_id"]],
                "error_ids": [],
            }
            (phase_dir / "PatchHarness-Ark.run_model.json").write_text(
                json.dumps(summary),
                encoding="utf-8",
            )

            resolved = run_harness(
                SingleCaseConfig.from_dict(CASE),
                phase_dir / "predictions.jsonl",
                phase_dir,
                "run_model",
                run_command=lambda command, **kwargs: subprocess.CompletedProcess(
                    command, 0, "No instances to run.", ""
                ),
            )

        self.assertFalse(resolved)

    def test_reuses_valid_gold_without_running_gold_harness(self):
        worker = WorkerResult(
            instance_id=CASE["instance_id"],
            patch="diff --git a/a.py b/a.py\n",
            response={"final_summary": "patched"},
            llm_calls=1,
            prompt_tokens=10,
            completion_tokens=5,
            elapsed_seconds=1.0,
        )
        harness_calls = []

        def fake_harness(config, predictions, phase_dir, harness_run_id):
            harness_calls.append(predictions)
            return False

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results_root = root / "results"
            self._gold_source(results_root, "source_gold")
            run_dir = run_single(
                self._config(root),
                results_root,
                "target_run",
                reuse_gold_from="source_gold",
                preflight_fn=lambda project_root: {"interpreter": "venv311/bin/python"},
                harness_runner=fake_harness,
                worker_runner=lambda *args: worker,
            )
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))

        self.assertEqual(len(harness_calls), 1)
        self.assertNotEqual(harness_calls[0], "gold")
        self.assertTrue(metrics["gold_resolved"])
        self.assertTrue(metrics["gold_reused"])
        self.assertEqual(metrics["gold_source_run_id"], "source_gold")

    def test_rejects_same_target_and_gold_source_run_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "must be different"):
                run_single(
                    self._config(root),
                    root / "results",
                    "same_run",
                    reuse_gold_from="same_run",
                )

    def test_rejects_existing_target_run_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "results" / "target_run"
            target.mkdir(parents=True)
            with self.assertRaises(FileExistsError):
                run_single(self._config(root), root / "results", "target_run")

    def test_rejects_source_without_successful_gold(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results_root = root / "results"
            source = self._gold_source(results_root)
            metrics_path = source / "metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics["gold_resolved"] = False
            metrics_path.write_text(json.dumps(metrics), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "gold_resolved=true"):
                run_single(
                    self._config(root),
                    results_root,
                    "target_run",
                    reuse_gold_from="source_gold",
                    preflight_fn=lambda project_root: {},
                )
            self.assertFalse((results_root / "target_run").exists())

    def test_rejects_reused_gold_with_mismatched_instance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results_root = root / "results"
            self._gold_source(
                results_root,
                instance_id="astropy__astropy-14539",
            )

            with self.assertRaisesRegex(ValueError, "instance_id"):
                run_single(
                    self._config(root),
                    results_root,
                    "target_run",
                    reuse_gold_from="source_gold",
                    preflight_fn=lambda project_root: {},
                )

    def test_rejects_reused_gold_without_official_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            results_root = root / "results"
            source = self._gold_source(results_root)
            report = next(source.glob("gold/evaluation/**/report.json"))
            report.unlink()

            with self.assertRaisesRegex(ValueError, "official gold artifact"):
                run_single(
                    self._config(root),
                    results_root,
                    "target_run",
                    reuse_gold_from="source_gold",
                    preflight_fn=lambda project_root: {},
                )


if __name__ == "__main__":
    unittest.main()
