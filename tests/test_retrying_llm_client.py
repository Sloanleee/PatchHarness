import unittest

from app.llm import (
    ArkAPIError,
    BudgetedLLMClient,
    LLMCallBudgetExceeded,
    LLMResponse,
    RetryingLLMClient,
)


def ark_error(code="RequestBurstTooFast", retryable=True, retry_after=None):
    return ArkAPIError(
        status_code=429,
        error_code=code,
        error_type="TooManyRequests",
        message="slow down",
        request_id="req-123",
        retry_after=retry_after,
        response_body={"error": {"code": code}},
        rate_limit_headers={},
        retryable=retryable,
    )


class SequenceClient:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.calls = 0

    def complete_json(self, messages, **kwargs):
        self.calls += 1
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class RetryingLLMClientTests(unittest.TestCase):
    def test_network_timeout_counts_attempt_and_logs_without_secret(self):
        events = []
        client = RetryingLLMClient(
            SequenceClient([TimeoutError("secret-request-body")]),
            event_sink=events.append,
        )

        with self.assertRaises(TimeoutError):
            client.complete_json([{"role": "user", "content": "api-key-secret"}])

        self.assertEqual(client.snapshot().attempts, 1)
        self.assertEqual(events[0]["error_code"], "network_timeout")
        self.assertNotIn("secret", str(events[0]))

    def test_connection_error_counts_attempt_without_retry(self):
        inner = SequenceClient([ConnectionError("reset")])
        client = RetryingLLMClient(inner, event_sink=lambda event: None)

        with self.assertRaises(ConnectionError):
            client.complete_json([])

        self.assertEqual(inner.calls, 1)
        self.assertEqual(client.snapshot().attempts, 1)
    def test_retries_transient_429_with_bounded_exponential_delays(self):
        inner = SequenceClient(
            [
                ark_error(),
                ark_error(),
                ark_error(),
                LLMResponse("{}", prompt_tokens=100, completion_tokens=20),
            ]
        )
        budgeted = BudgetedLLMClient(inner, max_calls=12, max_tokens=200_000)
        sleeps = []
        events = []
        client = RetryingLLMClient(
            budgeted,
            retry_delays=(5, 10, 20),
            sleeper=sleeps.append,
            event_sink=events.append,
            clock=lambda: 100.0,
            rpm_limit=500,
            tpm_limit=1_000_000,
        )

        response = client.complete_json([])

        self.assertEqual(response.prompt_tokens, 100)
        self.assertEqual(sleeps, [5, 10, 20])
        self.assertEqual(inner.calls, 4)
        self.assertEqual(budgeted.snapshot().calls, 4)
        snapshot = client.snapshot()
        self.assertEqual(snapshot.attempts, 4)
        self.assertEqual(snapshot.retries, 3)
        self.assertEqual(snapshot.client_observed_rpm, 4)
        self.assertEqual(snapshot.client_observed_tpm, 120)
        self.assertEqual(events[0]["request_id"], "req-123")
        self.assertEqual(events[0]["error_code"], "RequestBurstTooFast")
        self.assertEqual(events[0]["client_observed_rpm"], 1)

    def test_retry_after_can_increase_but_not_exceed_sixty_seconds(self):
        inner = SequenceClient(
            [ark_error(retry_after=90), LLMResponse("{}")]
        )
        sleeps = []
        client = RetryingLLMClient(
            BudgetedLLMClient(inner, 12, 200_000),
            sleeper=sleeps.append,
            event_sink=lambda event: None,
        )

        client.complete_json([])

        self.assertEqual(sleeps, [60])

    def test_does_not_retry_non_retryable_quota_error(self):
        inner = SequenceClient(
            [ark_error(code="AccountQuotaExceeded", retryable=False)]
        )
        sleeps = []
        client = RetryingLLMClient(
            BudgetedLLMClient(inner, 12, 200_000),
            sleeper=sleeps.append,
            event_sink=lambda event: None,
        )

        with self.assertRaises(ArkAPIError):
            client.complete_json([])

        self.assertEqual(inner.calls, 1)
        self.assertEqual(sleeps, [])

    def test_actual_retry_attempts_cannot_exceed_call_budget(self):
        inner = SequenceClient([ark_error(), ark_error(), ark_error()])
        budgeted = BudgetedLLMClient(inner, max_calls=2, max_tokens=200_000)
        sleeps = []
        client = RetryingLLMClient(
            budgeted,
            sleeper=sleeps.append,
            event_sink=lambda event: None,
        )

        with self.assertRaises(LLMCallBudgetExceeded):
            client.complete_json([])

        self.assertEqual(inner.calls, 2)
        self.assertEqual(budgeted.snapshot().calls, 2)


if __name__ == "__main__":
    unittest.main()
