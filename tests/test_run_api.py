import importlib.util
import tempfile
import unittest
import warnings
from pathlib import Path

from app.checkpoints import CheckpointStore
from app.main import app, workflow


FASTAPI_AVAILABLE = importlib.util.find_spec("fastapi") is not None
LANGGRAPH_AVAILABLE = importlib.util.find_spec("langgraph") is not None


@unittest.skipUnless(FASTAPI_AVAILABLE and LANGGRAPH_AVAILABLE, "FastAPI and LangGraph are required")
class RunApiTests(unittest.TestCase):
    def test_get_run_and_resume_approval(self):
        warnings.filterwarnings(
            "ignore",
            message="Using `httpx` with `starlette.testclient` is deprecated.*",
        )
        from fastapi.testclient import TestClient

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / ".env").write_text("TOKEN=old\n", encoding="utf-8")
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
            workflow.checkpoint_store = CheckpointStore(root / "checkpoints")
            client = TestClient(app)

            paused = client.post(
                "/bugfix",
                json={
                    "task_description": "In `.env` replace `TOKEN=old` with `TOKEN=new`",
                    "workspace_path": str(workspace),
                    "mode": "fix",
                    "allow_edit": True,
                    "run_tests": True,
                    "test_command": "python -m unittest discover -s tests",
                    "use_langgraph": True,
                },
            )
            self.assertEqual(paused.status_code, 200)
            paused_body = paused.json()
            run_id = paused_body["run_id"]

            inspected = client.get(f"/runs/{run_id}")
            self.assertEqual(inspected.status_code, 200)
            self.assertEqual(inspected.json()["status"], "paused")

            resumed = client.post(
                f"/runs/{run_id}/resume",
                json={"approved": True, "reviewer": "api-test", "comment": "approved"},
            )
            self.assertEqual(resumed.status_code, 200)
            resumed_body = resumed.json()
            self.assertFalse(resumed_body["requires_human_approval"])
            self.assertEqual(resumed_body["test_result"]["returncode"], 0)

            replay = client.post(
                f"/runs/{run_id}/resume",
                json={"approved": True, "reviewer": "api-test", "comment": "replay"},
            )
            self.assertEqual(replay.status_code, 409)


if __name__ == "__main__":
    unittest.main()
