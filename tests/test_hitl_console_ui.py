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


if __name__ == "__main__":
    unittest.main()
