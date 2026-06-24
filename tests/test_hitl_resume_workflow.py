import importlib.util
import tempfile
import unittest
from pathlib import Path

from app.agents import AgentRegistry
from app.checkpoints import CheckpointStore
from app.graph import BugfixWorkflow
from app.schemas import BugfixRequest


LANGGRAPH_AVAILABLE = importlib.util.find_spec("langgraph") is not None


class NoSequentialWorkflow(BugfixWorkflow):
    def _run_sequential(self, request):  # pragma: no cover - failure guard
        raise AssertionError("LangGraph path must not call sequential fallback")


@unittest.skipUnless(LANGGRAPH_AVAILABLE, "langgraph optional dependency is not installed")
class HitlResumeWorkflowTests(unittest.TestCase):
    def _workflow(self, checkpoint_root: Path) -> BugfixWorkflow:
        workflow = NoSequentialWorkflow(AgentRegistry.load_from_dir(Path("app/agents/configs")))
        workflow.checkpoint_store = CheckpointStore(checkpoint_root)
        return workflow

    def test_sensitive_edit_pauses_and_writes_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / ".env").write_text("TOKEN=old\n", encoding="utf-8")
            checkpoint_root = root / "checkpoints"

            response = self._workflow(checkpoint_root).run(
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
            self.assertEqual(response.failure_reason, "hitl_pause")
            self.assertIsNotNone(response.run_id)
            self.assertEqual(response.pending_approval["trigger_node"], "patch_generation")
            self.assertEqual(response.pending_approval["resume_node"], "test_verify")
            self.assertIn("hitl_pause", response.planning["langgraph"]["nodes"])
            self.assertEqual((workspace / ".env").read_text(encoding="utf-8"), "TOKEN=old\n")

            self.assertTrue((checkpoint_root / f"{response.run_id}.json").exists())
            checkpoint = CheckpointStore(checkpoint_root).load(response.run_id)
            self.assertEqual(checkpoint["status"], "paused")
            self.assertEqual(checkpoint["resume_from"], "test_verify")
            self.assertIsNone(checkpoint["approval_decision"])
            self.assertEqual(
                checkpoint["pending_approval"]["reason"],
                response.pending_approval["reason"],
            )


if __name__ == "__main__":
    unittest.main()
