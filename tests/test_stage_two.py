import tempfile
import unittest
from pathlib import Path

from app.context import ContextCompressor, ContextManager
from app.graph import BugfixWorkflow
from app.schemas import AgentReport, BugfixRequest
from app.skills import SkillManager


class StageTwoTests(unittest.TestCase):
    def test_context_isolation_hides_hidden_items_and_merges_reports(self):
        request = BugfixRequest(task_description="审查项目")
        manager = ContextManager.from_request(request, ["code_review"])

        context = manager.fork("code_review")
        self.assertIn("task_description", context.visible_payload())
        self.assertNotIn("internal_secrets", context.visible_payload())

        report = AgentReport("code_review", "completed", summary="ok")
        merge_event = manager.merge(context, report)
        cleanup_event = manager.cleanup(context)

        self.assertEqual(merge_event["event"], "merge")
        self.assertEqual(cleanup_event["event"], "cleanup")
        self.assertEqual(context.items, {})

    def test_skill_manager_loads_frontmatter_before_full_content(self):
        manager = SkillManager.from_default_dir()
        frontmatter = manager.public_frontmatter()

        self.assertGreaterEqual(len(frontmatter), 3)
        self.assertIn("description", frontmatter[0])
        self.assertNotIn("content", frontmatter[0])

        content = manager.load_skill("bug_fix")
        self.assertIn("Bug Fix Skill", content)

    def test_hitl_blocks_sensitive_file_edit(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / ".env").write_text("TOKEN=old\n", encoding="utf-8")

            response = BugfixWorkflow.from_default_configs().run(
                BugfixRequest(
                    task_description="修复 bug：在 `.env` 中将 `TOKEN=old` 替换为 `TOKEN=new`",
                    workspace_path=str(workspace),
                    mode="fix",
                    allow_edit=True,
                    run_tests=False,
                )
            )

            self.assertTrue(response.requires_human_approval)
            self.assertEqual(response.planned_agents, ["bug_fix", "test_verify"])
            self.assertEqual(len(response.agent_reports), 1)
            self.assertEqual((workspace / ".env").read_text(encoding="utf-8"), "TOKEN=old\n")
            self.assertGreater(response.metrics.hitl_interruptions, 0)

    def test_compressor_records_event_for_large_report(self):
        report = AgentReport(
            "code_review",
            "completed",
            observations=[
                {
                    "tool": "read_file",
                    "ok": True,
                    "data": {"content": "x" * 5000},
                    "error": None,
                }
            ],
        )

        compressed = ContextCompressor(max_tokens=100, threshold=0.5).maybe_compress_report(report)

        self.assertTrue(compressed)
        self.assertEqual(report.compression_events[0]["event"], "compress")
        self.assertIn("[compressed]", report.observations[0]["data"]["content"])


if __name__ == "__main__":
    unittest.main()

