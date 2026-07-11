import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.llm import ArkAPIError, MockLLMClient, VolcengineArkClient, create_llm_client
from app.llm.deepseek_client import DeepSeekClient


class LLMProviderTests(unittest.TestCase):
    def test_factory_can_create_mock_provider(self):
        with patch.dict(os.environ, {"PATCHHARNESS_LLM_PROVIDER": "mock"}, clear=False):
            client = create_llm_client()

        self.assertIsInstance(client, MockLLMClient)

    def test_factory_rejects_unknown_provider(self):
        with patch.dict(os.environ, {"PATCHHARNESS_LLM_PROVIDER": "unknown"}, clear=False):
            with self.assertRaises(ValueError):
                create_llm_client()

    def test_factory_loads_provider_from_dotenv_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("PATCHHARNESS_LLM_PROVIDER=mock\n", encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True), patch(
                "app.llm.factory.Path.cwd", return_value=Path(tmp)
            ):
                client = create_llm_client()

        self.assertIsInstance(client, MockLLMClient)

    def test_deepseek_client_uses_env_model_and_base_url(self):
        with patch.dict(
            os.environ,
            {
                "DEEPSEEK_API_KEY": "deepseek-key",
                "DEEPSEEK_MODEL": "deepseek-custom",
                "DEEPSEEK_BASE_URL": "https://deepseek.example/v1",
            },
            clear=True,
        ):
            client = DeepSeekClient()

        self.assertEqual(client.api_key, "deepseek-key")
        self.assertEqual(client.model, "deepseek-custom")
        self.assertEqual(
            client.base_url,
            "https://deepseek.example/v1/chat/completions",
        )

    def test_deepseek_client_accepts_complete_chat_completions_url(self):
        client = DeepSeekClient(
            api_key="deepseek-key",
            base_url="https://deepseek.example/v1/chat/completions",
        )

        self.assertEqual(
            client.base_url,
            "https://deepseek.example/v1/chat/completions",
        )

    def test_volcengine_client_uses_httpx_responses_api_shape(self):
        calls = {}

        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "output_text": '{"thought":"ok","final":"done"}',
                    "usage": {"input_tokens": 11, "output_tokens": 7},
                }

        class FakeHttpClient:
            def post(self, url, headers, json, timeout):
                calls["url"] = url
                calls["headers"] = headers
                calls["json"] = json
                calls["timeout"] = timeout
                return FakeResponse()

        client = VolcengineArkClient(
            api_key="ark-key",
            model="ark-model",
            base_url="https://example.test/api/v3",
            http_client=FakeHttpClient(),
        )
        response = client.complete_json([{"role": "user", "content": "hello"}])

        self.assertEqual(calls["url"], "https://example.test/api/v3/responses")
        self.assertEqual(calls["headers"]["Authorization"], "Bearer ark-key")
        self.assertEqual(calls["json"]["model"], "ark-model")
        self.assertEqual(calls["json"]["input"][0]["content"][0]["type"], "input_text")
        self.assertEqual(response.prompt_tokens, 11)
        self.assertEqual(response.completion_tokens, 7)
        self.assertEqual(response.to_action().final, "done")

    def test_volcengine_client_preserves_redacted_429_diagnostics(self):
        class RateLimitedResponse:
            status_code = 429
            headers = {
                "Retry-After": "7",
                "X-Request-Id": "req-header",
                "X-RateLimit-Remaining-Requests": "496",
            }

            def json(self):
                return {
                    "error": {
                        "code": "RequestBurstTooFast",
                        "type": "TooManyRequests",
                        "message": "slow down; key=ark-secret",
                        "request_id": "req-body",
                        "api_key": "ark-secret",
                    }
                }

        class FakeHttpClient:
            def post(self, *args, **kwargs):
                return RateLimitedResponse()

        client = VolcengineArkClient(
            api_key="ark-secret",
            model="ark-model",
            http_client=FakeHttpClient(),
        )

        with self.assertRaises(ArkAPIError) as caught:
            client.complete_json([{"role": "user", "content": "hello"}])

        error = caught.exception
        self.assertEqual(error.status_code, 429)
        self.assertEqual(error.error_code, "RequestBurstTooFast")
        self.assertEqual(error.error_type, "TooManyRequests")
        self.assertEqual(error.request_id, "req-body")
        self.assertEqual(error.retry_after, 7.0)
        self.assertTrue(error.retryable)
        self.assertEqual(
            error.rate_limit_headers["x-ratelimit-remaining-requests"],
            "496",
        )
        serialized = str(error.response_body)
        self.assertNotIn("ark-secret", serialized)
        self.assertIn("[REDACTED]", serialized)


if __name__ == "__main__":
    unittest.main()
