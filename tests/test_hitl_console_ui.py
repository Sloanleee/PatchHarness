import importlib.util
import unittest
import warnings

from app.main import app


FASTAPI_AVAILABLE = importlib.util.find_spec("fastapi") is not None


@unittest.skipUnless(FASTAPI_AVAILABLE, "FastAPI is required")
class HITLConsoleUITests(unittest.TestCase):
    def test_hitl_console_route_returns_html(self):
        warnings.filterwarnings(
            "ignore",
            message="Using `httpx` with `starlette.testclient` is deprecated.*",
        )
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.get("/ui/hitl")

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("PatchHarness HITL Console", response.text)

    def test_hitl_console_contains_required_ui_contract(self):
        warnings.filterwarnings(
            "ignore",
            message="Using `httpx` with `starlette.testclient` is deprecated.*",
        )
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.get("/ui/hitl")
        self.assertEqual(response.status_code, 200)
        html = response.text

        for phrase in [
            "/health",
            "/bugfix",
            "/runs/",
            "/resume",
            "Trigger run",
            "Inspect",
            "Approve",
            "Reject",
            "validate_workspace",
            "plan_agents",
            "root_cause_analysis",
            "patch_generation",
            "hitl_pause",
            "test_verify",
            "build_response",
            "Agent Reports",
            "Test Result",
            "Metrics",
            "Raw JSON",
        ]:
            self.assertIn(phrase, html)

    def test_hitl_console_contains_browser_interaction_logic(self):
        warnings.filterwarnings(
            "ignore",
            message="Using `httpx` with `starlette.testclient` is deprecated.*",
        )
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.get("/ui/hitl")
        self.assertEqual(response.status_code, 200)
        html = response.text

        for phrase in [
            "async function checkHealth",
            "renderError(error);",
            "async function triggerRun",
            "async function inspectRun",
            "async function resumeRun",
            "function renderResponse",
            "FEATURE_FLAG=off",
            "FEATURE_FLAG=on",
            "approved controlled .env edit",
            "rejected controlled .env edit",
            "test_result",
            "changed_files",
            "agent_reports",
        ]:
            self.assertIn(phrase, html)


if __name__ == "__main__":
    unittest.main()
