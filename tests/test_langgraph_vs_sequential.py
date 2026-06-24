import importlib.util
import shutil
import tempfile
import unittest
from pathlib import Path

from app.graph import BugfixWorkflow
from app.schemas import BugfixRequest


LANGGRAPH_AVAILABLE = importlib.util.find_spec("langgraph") is not None


class LangGraphVsSequentialTests(unittest.TestCase):
    def _create_buggy_workspace(self, root: Path, name: str) -> Path:
        workspace = root / name
        workspace.mkdir()
        (workspace / "calc.py").write_text(
            "def add(a, b):\n    return a - b\n", encoding="utf-8"
        )
        tests_dir = workspace / "tests"
        tests_dir.mkdir()
        (tests_dir / "test_calc.py").write_text(
            "import unittest\n"
            "from calc import add\n\n"
            "class CalcTests(unittest.TestCase):\n"
            "    def test_add(self):\n"
            "        self.assertEqual(add(2, 3), 5)\n",
            encoding="utf-8",
        )
        return workspace

    def _run_case(self, workspace: Path, use_langgraph: bool):
        return BugfixWorkflow.from_default_configs().run(
            BugfixRequest(
                task_description="In `calc.py` replace `return a - b` with `return a + b`",
                workspace_path=str(workspace),
                mode="fix",
                allow_edit=True,
                run_tests=True,
                test_command="python -m unittest discover -s tests",
                use_langgraph=use_langgraph,
            )
        )

    @unittest.skipUnless(LANGGRAPH_AVAILABLE, "langgraph optional dependency is not installed")
    def test_langgraph_and_sequential_produce_same_bugfix_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sequential_workspace = self._create_buggy_workspace(root, "sequential")
            langgraph_workspace = root / "langgraph"
            shutil.copytree(sequential_workspace, langgraph_workspace)

            sequential = self._run_case(sequential_workspace, use_langgraph=False)
            langgraph = self._run_case(langgraph_workspace, use_langgraph=True)

            expected_agents = ["root_cause_analysis", "patch_generation", "test_verify"]
            self.assertEqual(sequential.planned_agents, expected_agents)
            self.assertEqual(langgraph.planned_agents, expected_agents)
            self.assertEqual(
                [report.agent_name for report in sequential.agent_reports],
                [report.agent_name for report in langgraph.agent_reports],
            )
            self.assertEqual(sequential.changed_files, ["calc.py"])
            self.assertEqual(langgraph.changed_files, ["calc.py"])
            self.assertEqual(sequential.test_result["returncode"], 0)
            self.assertEqual(langgraph.test_result["returncode"], 0)
            self.assertIn(
                "return a + b",
                (sequential_workspace / "calc.py").read_text(encoding="utf-8"),
            )
            self.assertIn(
                "return a + b",
                (langgraph_workspace / "calc.py").read_text(encoding="utf-8"),
            )

            self.assertNotIn("langgraph", sequential.planning)
            self.assertEqual(langgraph.planning["langgraph"]["enabled"], True)
            self.assertEqual(
                langgraph.planning["langgraph"]["nodes"],
                [
                    "validate_workspace",
                    "plan_agents",
                    "root_cause_analysis",
                    "patch_generation",
                    "test_verify",
                    "build_response",
                ],
            )


if __name__ == "__main__":
    unittest.main()
