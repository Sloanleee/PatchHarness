import tempfile
import unittest
from pathlib import Path

from app.checkpoints import CheckpointInvalidError, CheckpointMissingError, CheckpointStore
from app.schemas import BugfixResponse, WorkflowMetrics


class CheckpointStoreTests(unittest.TestCase):
    def test_save_load_and_update_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = CheckpointStore(Path(tmp))
            saved = store.save(
                {
                    "run_id": "run_demo",
                    "status": "paused",
                    "request": {"task_description": "demo"},
                    "events": [],
                }
            )

            self.assertEqual(saved["run_id"], "run_demo")
            self.assertEqual(saved["status"], "paused")
            self.assertIn("created_at", saved)
            self.assertIn("updated_at", saved)
            self.assertTrue((Path(tmp) / "run_demo.json").exists())

            loaded = store.load("run_demo")
            self.assertEqual(loaded["request"]["task_description"], "demo")

            updated = store.update("run_demo", status="completed", response={"ok": True})
            self.assertEqual(updated["status"], "completed")
            self.assertEqual(updated["response"], {"ok": True})

    def test_missing_checkpoint_raises_named_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = CheckpointStore(Path(tmp))
            with self.assertRaises(CheckpointMissingError):
                store.load("missing")

    def test_invalid_checkpoint_raises_named_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.json"
            path.write_text("{not json", encoding="utf-8")
            store = CheckpointStore(Path(tmp))
            with self.assertRaises(CheckpointInvalidError):
                store.load("broken")


class ResponseSchemaTests(unittest.TestCase):
    def test_response_includes_optional_run_and_approval_fields(self):
        response = BugfixResponse(
            request_id="request-1",
            run_id="run-1",
            planned_agents=[],
            agent_reports=[],
            changed_files=[],
            test_result=None,
            metrics=WorkflowMetrics(),
            final_summary="paused",
            requires_human_approval=True,
            failure_reason="hitl_pause",
            pending_approval={"resume_node": "test_verify"},
        )

        payload = response.to_dict()
        self.assertEqual(payload["run_id"], "run-1")
        self.assertEqual(payload["failure_reason"], "hitl_pause")
        self.assertEqual(payload["pending_approval"], {"resume_node": "test_verify"})


if __name__ == "__main__":
    unittest.main()
