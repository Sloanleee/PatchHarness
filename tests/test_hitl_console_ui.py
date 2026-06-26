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
            "setApprovalEnabled(Boolean(state.runId));",
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

    def test_hitl_console_normalizes_checkpoint_wrappers(self):
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
            "function responsePayload(value)",
            "value.response",
            "function pendingApprovalFromResponse(value)",
            "return value.pending_approval || payload.pending_approval || null;",
            "state.rawResponse = response;",
            "state.payload = payload;",
            "renderResponse(body);",
            "elements.evidenceOutput.textContent = formatJson(state.rawResponse);",
        ]:
            self.assertIn(phrase, html)

    def test_hitl_console_uses_backend_trace_fields_in_order(self):
        warnings.filterwarnings(
            "ignore",
            message="Using `httpx` with `starlette.testclient` is deprecated.*",
        )
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.get("/ui/hitl")
        self.assertEqual(response.status_code, 200)
        html = response.text

        expected_order = [
            "appendTraceNodes(trace, normalizeTraceNodes(value && value.executed_nodes));",
            "normalizeTraceNodes(payload && payload.planning && payload.planning.langgraph && payload.planning.langgraph.nodes)",
            "appendTraceNodes(trace, traceNodesFromEvents(value && value.events));",
            "appendTraceNodes(trace, traceNodesFromEvents(payload && payload.planning && payload.planning.langgraph_events));",
        ]
        cursor = -1
        for phrase in expected_order:
            next_cursor = html.find(phrase)
            self.assertGreater(next_cursor, cursor, phrase)
            cursor = next_cursor
        self.assertIn("function appendTraceNodes(trace, nextNodes)", html)

    def test_hitl_console_treats_rejection_as_terminal(self):
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
            'if (payload && payload.failure_reason === "approval_rejected") {',
            'return "approval_rejected";',
            'const isRejected = status === "approval_rejected";',
            'setApprovalEnabled(Boolean(payload.requires_human_approval && state.runId && !isRejected));',
            'approval_rejected',
        ]:
            self.assertIn(phrase, html)

    def test_hitl_console_evidence_tabs_use_nested_payload(self):
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
            "const payload = state.payload || responsePayload(response);",
            "payload.agent_reports || []",
            "changed_files: payload.changed_files || []",
            "test_result: payload.test_result || null",
            "planned_agents: payload.planned_agents || []",
            "metrics: payload.metrics || {}",
            "approval_events: payload.approval_events || []",
        ]:
            self.assertIn(phrase, html)


if __name__ == "__main__":
    unittest.main()
