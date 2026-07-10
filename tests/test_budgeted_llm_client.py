import unittest

from app.llm import LLMResponse
from app.llm.budgeted_client import (
    BudgetedLLMClient,
    LLMCallBudgetExceeded,
    LLMTokenBudgetExceeded,
)


class FakeClient:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def complete_json(self, messages, **kwargs):
        self.calls += 1
        return next(self.responses)


class BudgetedLLMClientTests(unittest.TestCase):
    def test_counts_calls_and_tokens(self):
        inner = FakeClient([LLMResponse("{}", prompt_tokens=11, completion_tokens=7)])
        client = BudgetedLLMClient(inner, max_calls=12, max_tokens=200_000)

        client.complete_json([{"role": "user", "content": "x"}])

        self.assertEqual(client.snapshot().calls, 1)
        self.assertEqual(client.snapshot().prompt_tokens, 11)
        self.assertEqual(client.snapshot().completion_tokens, 7)
        self.assertEqual(client.snapshot().total_tokens, 18)

    def test_rejects_thirteenth_call_before_forwarding(self):
        responses = [LLMResponse("{}") for _ in range(12)]
        inner = FakeClient(responses)
        client = BudgetedLLMClient(inner, max_calls=12, max_tokens=200_000)

        for _ in range(12):
            client.complete_json([])

        with self.assertRaises(LLMCallBudgetExceeded):
            client.complete_json([])
        self.assertEqual(inner.calls, 12)

    def test_rejects_next_call_after_observed_token_limit(self):
        inner = FakeClient(
            [LLMResponse("{}", prompt_tokens=199_000, completion_tokens=1_000)]
        )
        client = BudgetedLLMClient(inner, max_calls=12, max_tokens=200_000)

        client.complete_json([])

        with self.assertRaises(LLMTokenBudgetExceeded):
            client.complete_json([])
        self.assertEqual(inner.calls, 1)

    def test_exception_text_does_not_include_inner_client_secrets(self):
        inner = FakeClient([LLMResponse("{}")])
        inner.api_key = "secret-ark-key"
        client = BudgetedLLMClient(inner, max_calls=0, max_tokens=1)

        with self.assertRaises(LLMCallBudgetExceeded) as caught:
            client.complete_json([])

        self.assertNotIn("secret-ark-key", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
