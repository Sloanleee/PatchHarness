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

    def _pause_sensitive_edit(
        self,
        root: Path,
        *,
        with_tests: bool = False,
    ) -> tuple[BugfixWorkflow, Path, object]:
        workspace = root / "workspace"
        workspace.mkdir()
        (workspace / ".env").write_text("TOKEN=old\n", encoding="utf-8")
        test_command = None
        if with_tests:
            tests_dir = workspace / "tests"
            tests_dir.mkdir()
            (tests_dir / "test_env.py").write_text(
                "import unittest\n"
                "from pathlib import Path\n\n"
                "class EnvTests(unittest.TestCase):\n"
                "    def test_token_updated(self):\n"
                "        self.assertIn('TOKEN=new', Path('.env').read_text())\n",
                encoding="utf-8",
            )
            test_command = "python -m unittest discover -s tests"
        workflow = self._workflow(root / "checkpoints")
        paused = workflow.run(
            BugfixRequest(
                task_description="In `.env` replace `TOKEN=old` with `TOKEN=new`",
                workspace_path=str(workspace),
                mode="fix",
                allow_edit=True,
                run_tests=True,
                test_command=test_command,
                use_langgraph=True,
            )
        )
        return workflow, workspace, paused

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

    def test_approval_resumes_sensitive_edit_and_runs_tests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow, workspace, paused = self._pause_sensitive_edit(root, with_tests=True)
            resumed = workflow.resume(
                paused.run_id,
                approved=True,
                reviewer="unit-test",
                comment="approved",
            )

            self.assertFalse(resumed.requires_human_approval)
            self.assertEqual(resumed.failure_reason, "")
            self.assertEqual(resumed.test_result["returncode"], 0)
            self.assertEqual((workspace / ".env").read_text(encoding="utf-8"), "TOKEN=new\n")
            self.assertIn("test_verify", resumed.planning["langgraph"]["nodes"])

            checkpoint = workflow.checkpoint_store.load(paused.run_id)
            self.assertEqual(checkpoint["status"], "completed")
            self.assertIsNone(checkpoint["pending_approval"])
            self.assertIsNone(checkpoint["resume_from"])
            self.assertEqual(checkpoint["approval_decision"]["approved"], True)
            self.assertEqual(checkpoint["response"]["test_result"]["returncode"], 0)
            self.assertEqual(checkpoint["response"]["failure_reason"], "")
            self.assertFalse(checkpoint["response"]["requires_human_approval"])
            self.assertEqual(checkpoint["reports"][-1]["agent_name"], "test_verify")
            self.assertIn("build_response", checkpoint["executed_nodes"])
            self.assertIn(
                "build_response",
                [event.get("node") for event in checkpoint["events"]],
            )
            self.assertEqual(
                checkpoint["metrics"]["agent_calls"],
                checkpoint["response"]["metrics"]["agent_calls"],
            )

    def test_double_approval_resume_is_rejected_without_mutating_workspace_again(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow, workspace, paused = self._pause_sensitive_edit(root, with_tests=True)

            workflow.resume(
                paused.run_id,
                approved=True,
                reviewer="unit-test",
                comment="approved",
            )
            (workspace / ".env").write_text("TOKEN=manual-change\n", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "Cannot resume run"):
                workflow.resume(
                    paused.run_id,
                    approved=True,
                    reviewer="unit-test",
                    comment="replay",
                )

            self.assertEqual(
                (workspace / ".env").read_text(encoding="utf-8"),
                "TOKEN=manual-change\n",
            )
            checkpoint = workflow.checkpoint_store.load(paused.run_id)
            self.assertEqual(checkpoint["status"], "completed")
            self.assertEqual(checkpoint["approval_decision"]["comment"], "approved")

    def test_rejection_does_not_apply_sensitive_edit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow, workspace, paused = self._pause_sensitive_edit(root)
            rejected = workflow.resume(
                paused.run_id,
                approved=False,
                reviewer="unit-test",
                comment="rejected",
            )

            self.assertTrue(rejected.requires_human_approval)
            self.assertEqual(rejected.failure_reason, "approval_rejected")
            self.assertEqual((workspace / ".env").read_text(encoding="utf-8"), "TOKEN=old\n")

            checkpoint = workflow.checkpoint_store.load(paused.run_id)
            self.assertEqual(checkpoint["status"], "rejected")
            self.assertIsNone(checkpoint["pending_approval"])
            self.assertIsNone(checkpoint["resume_from"])
            self.assertEqual(checkpoint["approval_decision"]["approved"], False)

    def test_resume_after_rejection_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workflow, workspace, paused = self._pause_sensitive_edit(root)

            workflow.resume(
                paused.run_id,
                approved=False,
                reviewer="unit-test",
                comment="rejected",
            )

            with self.assertRaisesRegex(RuntimeError, "Cannot resume run"):
                workflow.resume(
                    paused.run_id,
                    approved=True,
                    reviewer="unit-test",
                    comment="replay",
                )

            self.assertEqual((workspace / ".env").read_text(encoding="utf-8"), "TOKEN=old\n")
            checkpoint = workflow.checkpoint_store.load(paused.run_id)
            self.assertEqual(checkpoint["status"], "rejected")
            self.assertEqual(checkpoint["approval_decision"]["comment"], "rejected")


if __name__ == "__main__":
    unittest.main()
