import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from benchmarks.swe_bench.models import WorkerResult
from benchmarks.swe_bench.orchestrator import PreflightError, preflight, run_single


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


if __name__ == "__main__":
    unittest.main()
