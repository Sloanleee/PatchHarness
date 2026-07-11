import unittest

from benchmarks.run_swe_bench_single import _parse_args


class SweBenchCliTests(unittest.TestCase):
    def test_parses_reuse_gold_source(self):
        args = _parse_args(
            [
                "--run-id",
                "swe_single_003",
                "--reuse-gold-from",
                "swe_single_002",
            ]
        )

        self.assertEqual(args.run_id, "swe_single_003")
        self.assertEqual(args.reuse_gold_from, "swe_single_002")


if __name__ == "__main__":
    unittest.main()
