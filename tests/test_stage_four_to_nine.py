import tempfile
import unittest
import copy
from collections import deque
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
    def test_grep_search_normalizes_model_arguments(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "src").mkdir()
            (workspace / "docs").mkdir()
            (workspace / "src" / "types.py").write_text(
                "class Alpha:\n    pass\nclass Beta:\n    pass\nA|B = 1\n", encoding="utf-8"
            )
            (workspace / "docs" / "types.py").write_text("class Alpha:\n", encoding="utf-8")
            client = MCPClient(MCPServer(create_default_tools()))

            pattern = client.run("grep_search", workspace, pattern="class Alpha", glob="src/*.py")
            preferred = client.run(
                "grep_search", workspace, query="class Beta", pattern="class Alpha", glob="src/*.py"
            )
            literal = client.run("grep_search", workspace, query="Alpha|Beta", glob="src/*.py")
            regex = client.run(
                "grep_search", workspace, pattern="class (Alpha|Beta)", regex=True, glob="src/*.py"
            )
            invalid = client.run("grep_search", workspace, pattern="[", regex=True)
            bad_limit = client.run("grep_search", workspace, query="class", max_results=0)
            scoped = client.run("grep_search", workspace, query="class Alpha", path="src")
            escaped = client.run("grep_search", workspace, query="class", path="../outside")

            self.assertEqual([m["path"] for m in pattern.data["matches"]], ["src/types.py"])
            self.assertEqual(preferred.data["search"]["query"], "class Beta")
            self.assertEqual(literal.data["matches"], [])
            self.assertEqual(len(regex.data["matches"]), 2)
            self.assertFalse(invalid.ok)
            self.assertIn("invalid regex", invalid.error)
            self.assertFalse(bad_limit.ok)
            self.assertEqual([m["path"] for m in scoped.data["matches"]], ["src/types.py"])
            self.assertFalse(escaped.ok)

    def test_root_cause_exhaustion_with_source_evidence_is_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "src.py").write_text("class Target:\n    pass\n", encoding="utf-8")
            llm = MockLLMClient(
                [LLMAction("keep investigating", "grep_search", {"query": "class Target"}) for _ in range(18)]
            )
            workflow = BugfixWorkflow.from_default_configs()
            workflow.llm_client = llm
            result = workflow.run(BugfixRequest(
                task_description="Fix Target", workspace_path=str(workspace), mode="fix",
                allow_edit=True, run_tests=False, enable_llm=True,
            ))

            report = result.agent_reports[0]
            self.assertEqual(report.status, "partial")
            self.assertEqual(report.stop_reason, "max_iterations_exhausted")
            self.assertGreater(report.evidence_count, 0)
            self.assertIn("src.py", str(report.diagnostic_evidence))

    def test_root_cause_exhaustion_with_only_docs_is_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "docs").mkdir()
            (workspace / "docs" / "guide.py").write_text("class Target:\n", encoding="utf-8")
            llm = MockLLMClient(
                [LLMAction("keep investigating", "grep_search", {"query": "class Target"}) for _ in range(18)]
            )
            workflow = BugfixWorkflow.from_default_configs()
            workflow.llm_client = llm
            result = workflow.run(BugfixRequest(
                task_description="Fix Target", workspace_path=str(workspace), mode="fix",
                allow_edit=True, run_tests=False, enable_llm=True,
            ))

            self.assertEqual(result.agent_reports[0].status, "failed")

    def test_partial_root_cause_continues_to_patch_with_compact_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "target.py").write_text(
                "SECRET_FULL_CONTENT = 'must-not-be-forwarded'\nvalue = 1\n", encoding="utf-8"
            )

            class RecordingLLM(MockLLMClient):
                def __init__(self):
                    super().__init__([
                        *[LLMAction("investigate", "grep_search", {"query": "value = 1"}) for _ in range(18)],
                        LLMAction("patch", "edit_file", {"path": "target.py", "old": "value = 1", "new": "value = 2"}),
                        LLMAction("done", final="patched"),
                    ])
                    self.batches = []
                def complete_json(self, messages, **kwargs):
                    self.batches.append(copy.deepcopy(messages))
                    return super().complete_json(messages, **kwargs)

            llm = RecordingLLM()
            workflow = BugfixWorkflow.from_default_configs()
            workflow.llm_client = llm
            result = workflow.run(BugfixRequest(
                task_description="Fix target value", workspace_path=str(workspace), mode="fix",
                allow_edit=True, run_tests=False, enable_llm=True,
            ))

            self.assertEqual(result.agent_reports[0].status, "partial")
            self.assertEqual(result.changed_files, ["target.py"])
            patch_prompt = llm.batches[18][1]["content"]
            self.assertIn("candidate_locations", patch_prompt)
            self.assertNotIn("SECRET_FULL_CONTENT", patch_prompt)

    def test_partial_root_cause_continues_when_llm_compression_falls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "target.py").write_text(
                "value = 1\n" + "\n".join("x = '" + ("a" * 100) + "'" for _ in range(450)),
                encoding="utf-8",
            )

            class OutcomeClient:
                def __init__(self):
                    self.outcomes = deque([
                        LLMAction("read source", "read_file", {"path": "target.py"}),
                        *[LLMAction("investigate", "grep_search", {"query": "value = 1"}) for _ in range(17)],
                        TimeoutError("compression timeout one"),
                        TimeoutError("compression timeout two"),
                        LLMAction("patch", "edit_file", {"path": "target.py", "old": "value = 1", "new": "value = 2"}),
                        LLMAction("done", final="patched"),
                    ])
                def complete_json(self, messages, **kwargs):
                    outcome = self.outcomes.popleft()
                    if isinstance(outcome, Exception):
                        raise outcome
                    return MockLLMClient([outcome]).complete_json(messages, **kwargs)

            workflow = BugfixWorkflow.from_default_configs()
            workflow.llm_client = OutcomeClient()
            result = workflow.run(BugfixRequest(
                task_description="Fix target value", workspace_path=str(workspace), mode="fix",
                allow_edit=True, run_tests=False, enable_llm=True,
            ))

            self.assertEqual(result.agent_reports[0].status, "partial")
            self.assertEqual(result.changed_files, ["target.py"])
            self.assertTrue(any(
                event["event"] == "compression_fallback"
                for event in result.agent_reports[0].compression_events
            ))

    def test_patch_generation_prompt_forces_progress_after_four_read_only_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "target.py").write_text("value = 1\n", encoding="utf-8")
            class RecordingLLM(MockLLMClient):
                def __init__(self):
                    super().__init__([
                        LLMAction("diagnosed", final="target.py needs value 2"),
                        *[LLMAction("inspect", "grep_search", {"query": "value", "path": "target.py"}) for _ in range(4)],
                        LLMAction("edit now", "edit_file", {"path": "target.py", "old": "value = 1", "new": "value = 2"}),
                        LLMAction("done", final="patched"),
                    ])
                    self.batches = []
                def complete_json(self, messages, **kwargs):
                    self.batches.append(copy.deepcopy(messages))
                    return super().complete_json(messages, **kwargs)
            llm = RecordingLLM()
            workflow = BugfixWorkflow.from_default_configs()
            workflow.llm_client = llm
            result = workflow.run(BugfixRequest(
                task_description="Fix target", workspace_path=str(workspace), mode="fix",
                allow_edit=True, run_tests=False, enable_llm=True,
            ))

            self.assertEqual(result.changed_files, ["target.py"])
            self.assertIn("read-only investigation budget is exhausted", str(llm.batches[5]).lower())
    def test_read_file_can_return_a_bounded_line_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "large.py").write_text(
                "".join(f"line {number}\n" for number in range(1, 21)),
                encoding="utf-8",
            )
            client = MCPClient(MCPServer(create_default_tools()))

            result = client.run(
                "read_file", workspace, path="large.py", start_line=5, end_line=7
            )

            self.assertTrue(result.ok)
            self.assertEqual(result.data["content"], "line 5\nline 6\nline 7\n")
            self.assertEqual(result.data["start_line"], 5)
            self.assertEqual(result.data["end_line"], 7)
            self.assertEqual(result.data["total_lines"], 20)

    def test_patch_generation_receives_root_cause_summary_and_edits(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "calc.py").write_text(
                "def add(a, b):\n    return a - b\n", encoding="utf-8"
            )

            class RecordingLLM(MockLLMClient):
                def __init__(self):
                    super().__init__(
                        [
                            LLMAction(
                                "diagnosed",
                                final="Root cause: calc.py returns subtraction instead of addition.",
                            ),
                            LLMAction(
                                "apply diagnosis",
                                "edit_file",
                                {
                                    "path": "calc.py",
                                    "old": "return a - b",
                                    "new": "return a + b",
                                },
                            ),
                            LLMAction("done", final="Patch applied."),
                        ]
                    )
                    self.message_batches = []

                def complete_json(self, messages, **kwargs):
                    self.message_batches.append(messages)
                    return super().complete_json(messages, **kwargs)

            llm = RecordingLLM()
            workflow = BugfixWorkflow.from_default_configs()
            workflow.llm_client = llm

            result = workflow.run(
                BugfixRequest(
                    task_description="Fix add so it performs addition.",
                    workspace_path=str(workspace),
                    mode="fix",
                    allow_edit=True,
                    run_tests=False,
                    enable_llm=True,
                )
            )

            self.assertEqual(result.changed_files, ["calc.py"])
            patch_prompt = llm.message_batches[1][1]["content"]
            self.assertIn("Root cause: calc.py returns subtraction", patch_prompt)

    def test_llm_react_can_drive_tool_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "calc.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
            llm = MockLLMClient(
                [
                    LLMAction(
                        "diagnosed",
                        final="Root cause: calc.py subtracts instead of adding.",
                    ),
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
            llm = MockLLMClient([
                LLMAction("diagnosed", final="Root cause is in calc.py."),
                {"thought": "bad shape", "action": "read_file", "action_input": "calc.py"},
            ])

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

            self.assertEqual(len(result.agent_reports), 1)
            self.assertEqual(result.agent_reports[0].agent_name, "root_cause_analysis")
            self.assertEqual(result.agent_reports[0].status, "failed")
            self.assertIn("LLM request timed out", result.agent_reports[0].summary)
            self.assertEqual(result.metrics.agent_calls, 1)
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
