import importlib.util
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from app.agents import AgentRegistry
from app.graph import BugfixWorkflow
from app.schemas import BugfixRequest


LANGGRAPH_AVAILABLE = importlib.util.find_spec("langgraph") is not None


class NoSequentialWorkflow(BugfixWorkflow):
    def _run_sequential(self, request):  # pragma: no cover - failure guard
        raise AssertionError("LangGraph path must not call _run_sequential")


@unittest.skipUnless(LANGGRAPH_AVAILABLE, "langgraph optional dependency is not installed")
class LangGraphWorkflowTests(unittest.TestCase):
    def _workflow(self) -> BugfixWorkflow:
        return NoSequentialWorkflow(
            AgentRegistry.load_from_dir(Path("app/agents/configs"))
        )

    def test_langgraph_fix_runs_real_agent_nodes_without_sequential_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
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

            response = self._workflow().run(
                BugfixRequest(
                    task_description="In `calc.py` replace `return a - b` with `return a + b`",
                    workspace_path=str(workspace),
                    mode="fix",
                    allow_edit=True,
                    run_tests=True,
                    test_command="python -m unittest discover -s tests",
                    use_langgraph=True,
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
            self.assertEqual(response.changed_files, ["calc.py"])
            self.assertEqual(response.test_result["returncode"], 0)
            self.assertEqual(response.planning["langgraph"]["enabled"], True)
            self.assertNotIn("execute_agents", response.planning["langgraph"]["nodes"])
            self.assertIn("root_cause_analysis", response.planning["langgraph"]["nodes"])
            self.assertIn("patch_generation", response.planning["langgraph"]["nodes"])

    def test_langgraph_review_routes_only_code_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "calc.py").write_text(
                "def add(a, b):\n    return a + b\n", encoding="utf-8"
            )

            response = self._workflow().run(
                BugfixRequest(
                    task_description="review `calc.py`",
                    workspace_path=str(workspace),
                    mode="review",
                    run_tests=False,
                    use_langgraph=True,
                )
            )

            self.assertEqual(response.planned_agents, ["code_review"])
            self.assertEqual([report.agent_name for report in response.agent_reports], ["code_review"])
            self.assertIn("code_review", response.planning["langgraph"]["nodes"])

    def test_langgraph_hitl_stops_before_following_nodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / ".env").write_text("TOKEN=old\n", encoding="utf-8")

            response = self._workflow().run(
                BugfixRequest(
                    task_description="In `.env` replace `TOKEN=old` with `TOKEN=new`",
                    workspace_path=str(workspace),
                    mode="fix",
                    allow_edit=True,
                    run_tests=True,
                    use_langgraph=True,
                )
            )

            self.assertTrue(response.requires_human_approval)
            self.assertEqual(
                [report.agent_name for report in response.agent_reports],
                ["root_cause_analysis", "patch_generation"],
            )
            self.assertEqual((workspace / ".env").read_text(encoding="utf-8"), "TOKEN=old\n")


class LangGraphRuntimeErrorTests(unittest.TestCase):
    def test_explicit_langgraph_runtime_error_is_not_sequential_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            workflow = BugfixWorkflow(
                AgentRegistry.load_from_dir(Path("app/agents/configs"))
            )

            with mock.patch(
                "app.graph.langgraph_workflow.LangGraphBugfixWorkflow"
            ) as workflow_class:
                workflow_class.return_value.run.side_effect = RuntimeError("langgraph exploded")

                with self.assertRaisesRegex(RuntimeError, "langgraph exploded"):
                    workflow.run(
                        BugfixRequest(
                            task_description="review workspace",
                            workspace_path=str(workspace),
                            mode="review",
                            run_tests=False,
                            use_langgraph=True,
                        )
                    )


if __name__ == "__main__":
    unittest.main()
