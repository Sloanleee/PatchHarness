import importlib.util
import io
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout

from benchmarks.run_langgraph_vs_sequential_experiment import run_experiment


LANGGRAPH_AVAILABLE = importlib.util.find_spec("langgraph") is not None


@unittest.skipUnless(LANGGRAPH_AVAILABLE, "langgraph optional dependency is not installed")
class LangGraphVsSequentialExperimentTests(unittest.TestCase):
    def test_mock_experiment_writes_timestamped_report_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with redirect_stdout(io.StringIO()):
                run_dir = run_experiment(
                    provider="mock",
                    category=None,
                    max_cases=4,
                    results_root=root / "results",
                    workspace_base=root / "workspaces",
                    run_id="20990101_010203",
                )

            self.assertEqual(run_dir, root / "results" / "20990101_010203")
            self.assertTrue((run_dir / "summary.md").exists())
            self.assertTrue((run_dir / "comparison.csv").exists())
            self.assertTrue((run_dir / "sequential.jsonl").exists())
            self.assertTrue((run_dir / "langgraph.jsonl").exists())
            self.assertTrue((run_dir / "raw" / "review_only.sequential.json").exists())
            self.assertTrue((run_dir / "raw" / "review_only.langgraph.json").exists())

            summary = (run_dir / "summary.md").read_text(encoding="utf-8")
            self.assertIn("# LangGraph vs Sequential Experiment Summary", summary)
            self.assertIn("Workflow controllability", summary)
            self.assertIn("LangGraph adds explicit node traces", summary)

            comparison = (run_dir / "comparison.csv").read_text(encoding="utf-8")
            self.assertIn("sequential_has_node_trace,langgraph_has_node_trace", comparison)
            self.assertIn("False,True", comparison)


if __name__ == "__main__":
    unittest.main()
