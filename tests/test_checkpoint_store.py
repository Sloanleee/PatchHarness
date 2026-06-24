import tempfile
import unittest
from pathlib import Path

from app.checkpoints import CheckpointInvalidError, CheckpointMissingError, CheckpointStore


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


if __name__ == "__main__":
    unittest.main()
