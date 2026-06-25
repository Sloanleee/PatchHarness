import importlib.util
import tempfile
import unittest
from pathlib import Path

from benchmarks.generate_production_demo_evidence import generate_evidence


LANGGRAPH_AVAILABLE = importlib.util.find_spec("langgraph") is not None


@unittest.skipUnless(LANGGRAPH_AVAILABLE, "langgraph optional dependency is not installed")
class ProductionDemoEvidenceTests(unittest.TestCase):
    def test_generate_mock_evidence_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = generate_evidence(
                provider="mock",
                results_root=root / "results",
                workspace_root=root / "workspace",
                run_id="evidence_test",
            )

            self.assertTrue((run_dir / "summary.md").exists())
            self.assertTrue((run_dir / "paused_response.json").exists())
            self.assertTrue((run_dir / "resumed_response.json").exists())
            self.assertTrue((run_dir / "trace.json").exists())
            summary = (run_dir / "summary.md").read_text(encoding="utf-8")
            self.assertIn("LangGraph HITL Checkpoint/Resume Demo", summary)
            self.assertIn("hitl_pause", summary)


if __name__ == "__main__":
    unittest.main()
