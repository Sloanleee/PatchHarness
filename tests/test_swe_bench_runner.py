import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.swe_bench.models import SingleCaseConfig, WorkerResult, load_case


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


if __name__ == "__main__":
    unittest.main()
