import tempfile
import unittest
from pathlib import Path

from app.graph import BugfixWorkflow
from app.llm import LLMAction, MockLLMClient
from app.mcp import MCPClient, MCPServer
from app.planner import LLMFallbackPlanner
from app.schemas import BugfixRequest
from app.skills import SkillManager
from app.tools.file_tools import create_default_tools


class TimeoutLLMClient:
    def complete_json(self, messages, **kwargs):
        raise TimeoutError("The read operation timed out")


class StageFourToNineTests(unittest.TestCase):
    def test_llm_react_can_drive_tool_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
            llm = MockLLMClient(
                [
                    LLMAction("read target", "read_file", {"path": "calc.py"}),
                    LLMAction(
                        "apply minimal patch",
                        "edit_file",
                        {"path": "calc.py", "old": "return a - b", "new": "return a + b"},
                    ),
                    LLMAction("done", final="Patch applied by LLM ReAct."),
                ]
            )

            response = BugfixWorkflow.from_default_configs()
            response.llm_client = llm
            result = response.run(
                BugfixRequest(
                    task_description="修复 calc.py 的加法实现",
                    workspace_path=str(workspace),
                    mode="fix",
                    allow_edit=True,
                    run_tests=False,
                    enable_llm=True,
                )
            )

            self.assertGreater(result.metrics.llm_calls, 0)
            self.assertEqual(result.changed_files, ["calc.py"])
            self.assertIn("return a + b", (workspace / "calc.py").read_text(encoding="utf-8"))

    def test_llm_react_rejects_non_object_action_input_and_stops_workflow(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
            llm = MockLLMClient(
                [{"thought": "bad shape", "action": "read_file", "action_input": "calc.py"}]
            )

            workflow = BugfixWorkflow.from_default_configs()
            workflow.llm_client = llm
            result = workflow.run(
                BugfixRequest(
                    task_description="修复 calc.py 的加法实现",
                    workspace_path=str(workspace),
                    mode="fix",
                    allow_edit=True,
                    run_tests=True,
                    enable_llm=True,
                )
            )

            self.assertEqual(len(result.agent_reports), 2)
            self.assertEqual(result.agent_reports[0].agent_name, "root_cause_analysis")
            self.assertEqual(result.agent_reports[0].status, "completed")
            self.assertEqual(result.agent_reports[1].agent_name, "patch_generation")
            self.assertEqual(result.agent_reports[1].status, "failed")
            self.assertIn("action_input", result.agent_reports[1].summary)

    def test_bug_fix_final_without_change_is_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
            llm = MockLLMClient([LLMAction("done too early", final="fixed")])

            workflow = BugfixWorkflow.from_default_configs()
            workflow.llm_client = llm
            result = workflow.run(
                BugfixRequest(
                    task_description="修复 calc.py 的加法实现",
                    workspace_path=str(workspace),
                    mode="fix",
                    allow_edit=True,
                    run_tests=True,
                    enable_llm=True,
                )
            )

            self.assertEqual(len(result.agent_reports), 2)
            self.assertEqual(result.agent_reports[0].agent_name, "root_cause_analysis")
            self.assertEqual(result.agent_reports[0].status, "completed")
            self.assertEqual(result.agent_reports[1].agent_name, "patch_generation")
            self.assertEqual(result.agent_reports[1].status, "failed")
            self.assertIn("before producing a code change", result.agent_reports[1].summary)

    def test_llm_timeout_returns_failed_report_and_preserves_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")

            workflow = BugfixWorkflow.from_default_configs()
            workflow.llm_client = TimeoutLLMClient()
            result = workflow.run(
                BugfixRequest(
                    task_description="修复 calc.py 的加法实现",
                    workspace_path=str(workspace),
                    mode="fix",
                    allow_edit=True,
                    run_tests=True,
                    enable_llm=True,
                )
            )

            self.assertEqual(len(result.agent_reports), 2)
            self.assertEqual(result.agent_reports[0].agent_name, "root_cause_analysis")
            self.assertEqual(result.agent_reports[0].status, "completed")
            self.assertEqual(result.agent_reports[1].agent_name, "patch_generation")
            self.assertEqual(result.agent_reports[1].status, "failed")
            self.assertIn("LLM request timed out", result.agent_reports[1].summary)
            self.assertEqual(result.metrics.agent_calls, 2)
            self.assertEqual(result.metrics.llm_timeouts, 1)

    def test_llm_fallback_planner_can_trigger_planning_hitl(self):
        llm = MockLLMClient(
            [{"thought": "ambiguous", "category": "unknown", "confidence": 0.2}]
        )
        planner = LLMFallbackPlanner(llm, confidence_threshold=0.65)

        result = planner.plan(BugfixRequest(task_description="帮我看看这个"))

        self.assertTrue(result.requires_human_approval)
        self.assertEqual(result.agents, ["code_review"])
        self.assertEqual(result.planned_by, "llm_fallback")

    def test_mcp_client_exposes_schemas_and_calls_tool(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "a.py").write_text("needle = 1\n", encoding="utf-8")
            client = MCPClient(MCPServer(create_default_tools()))

            schemas = client.schemas()
            result = client.run("grep_search", workspace, query="needle")

            self.assertTrue(any(tool["name"] == "grep_search" for tool in schemas))
            self.assertTrue(result.ok)
            self.assertEqual(result.data["matches"][0]["path"], "a.py")

    def test_skill_create_search_download_update_persists_to_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            manager = SkillManager(tmp)
            created = manager.create_skill(
                name="Django ORM",
                description="Django ORM troubleshooting",
                triggers=["django", "orm"],
                content="# Django ORM\n\nCheck migrations.",
            )
            self.assertEqual(created.name, "django_orm")

            search_results = manager.search_skill("django migrations")
            self.assertEqual(search_results[0]["name"], "django_orm")
            self.assertIn("Check migrations", manager.load_skill("django_orm"))

            manager.update_skill(
                name="django_orm",
                content="# Django ORM\n\nCheck select_related.",
            )
            reloaded = SkillManager(tmp)
            self.assertIn("select_related", reloaded.load_skill("django_orm"))


if __name__ == "__main__":
    unittest.main()
