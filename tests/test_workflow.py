import tempfile
import unittest
from pathlib import Path

from app.graph import BugfixWorkflow
from app.schemas import BugfixRequest


class WorkflowTests(unittest.TestCase):
    def test_fix_workflow_replaces_text_and_runs_tests(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
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

            response = BugfixWorkflow.from_default_configs().run(
                BugfixRequest(
                    task_description="修复 bug：在 `calc.py` 中将 `return a - b` 替换为 `return a + b`",
                    workspace_path=str(workspace),
                    mode="fix",
                    allow_edit=True,
                    run_tests=True,
                    test_command="python -m unittest discover -s tests",
                )
            )

            self.assertEqual(
                response.planned_agents,
                ["root_cause_analysis", "patch_generation", "test_verify"],
            )
            self.assertEqual(
                [report.agent_name for report in response.agent_reports],
                ["root_cause_analysis", "patch_generation", "test_verify"],
            )
            self.assertEqual(response.agent_reports[0].changed_files, [])
            self.assertEqual(response.agent_reports[1].changed_files, ["calc.py"])
            self.assertEqual(response.changed_files, ["calc.py"])
            self.assertEqual(response.test_result["returncode"], 0)
            self.assertIn("return a + b", (workspace / "calc.py").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
