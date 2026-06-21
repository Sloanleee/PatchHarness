import unittest

from app.metrics.benchmark import build_default_cases, run_planner_benchmark


class BenchmarkTests(unittest.TestCase):
    def test_default_benchmark_has_100_cases_and_saves_calls(self):
        cases = build_default_cases()
        rows, summary = run_planner_benchmark(cases)

        self.assertEqual(len(cases), 100)
        self.assertEqual(summary.total_cases, 100)
        self.assertEqual(summary.fixed_llm_calls, 800)
        self.assertLess(summary.dynamic_llm_calls, summary.fixed_llm_calls)
        self.assertGreater(summary.saved_llm_call_rate, 0.0)
        self.assertTrue(all(row.planner_hit for row in rows))


if __name__ == "__main__":
    unittest.main()

