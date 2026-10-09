from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from research.gpvp.providers import HttpResponse, ProviderAdapter, _openai_token_trace
from research.gpvp.types import AdapterError, CompletionRequest


class ProviderTests(unittest.TestCase):
    def test_missing_credentials_are_reported_without_exposing_values(self) -> None:
        with self.assertRaises(AdapterError) as caught:
            ProviderAdapter("openai", environ={})
        self.assertEqual(caught.exception.category, "missing_credential")
        self.assertNotIn("sk-", str(caught.exception))

    def test_openai_compatible_request_and_logprob_parsing(self) -> None:
        captured: dict[str, object] = {}

        def fake_post(url, headers, payload, timeout):
            captured.update(
                {"url": url, "headers": headers, "payload": payload, "timeout": timeout}
            )
            body = {
                "id": "resp-1",
                "model": "test-model-2026-01-01",
                "system_fingerprint": "fingerprint-x",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "A test response."},
                        "finish_reason": "stop",
                        "logprobs": {
                            "content": [
                                {
                                    "token": "A",
                                    "logprob": -0.1,
                                    "top_logprobs": [
                                        {"token": "A", "logprob": -0.1},
                                        {"token": "B", "logprob": -1.3},
                                    ],
                                }
                            ]
                        },
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3},
            }
            return HttpResponse(
                200, {"x-request-id": "request-1"}, json.dumps(body).encode()
            )

        adapter = ProviderAdapter("openai", api_key="test-secret", environ={})
        with patch("research.gpvp.providers._post_json", side_effect=fake_post):
            result = adapter.complete(
                CompletionRequest(
                    provider="openai",
                    model_id="test-model",
                    system_prompt="neutral system",
                    user_prompt="neutral prompt",
                    temperature=0.3,
                    seed=41,
                    max_tokens=32,
                    request_logprobs=True,
                )
            )
        self.assertEqual(result.text, "A test response.")
        self.assertEqual(result.response_id, "resp-1")
        self.assertTrue(result.seed_sent)
        self.assertEqual(result.token_trace[0]["selected_logprob"], -0.1)
        self.assertEqual(result.token_trace[0]["selected_rank_in_returned_top_k"], 1)
        self.assertEqual(captured["url"], "https://api.openai.com/v1/chat/completions")
        self.assertEqual(captured["payload"]["seed"], 41)
        self.assertNotIn("test-secret", json.dumps(captured["payload"]))
        self.assertNotIn("test-secret", json.dumps(result.raw_response))

    def test_anthropic_does_not_claim_seed_or_token_logprobs(self) -> None:
        body = {
            "id": "msg-1",
            "model": "claude-test-version",
            "content": [{"type": "text", "text": "A test response."}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 4, "output_tokens": 3},
        }
        adapter = ProviderAdapter("anthropic", api_key="test-secret", environ={})
        with patch(
            "research.gpvp.providers._post_json",
            return_value=HttpResponse(200, {}, json.dumps(body).encode()),
        ) as post:
            result = adapter.complete(
                CompletionRequest(
                    provider="anthropic",
                    model_id="claude-test",
                    system_prompt="system",
                    user_prompt="question",
                    temperature=0.7,
                    seed=88,
                    max_tokens=50,
                )
            )
        self.assertEqual(result.text, "A test response.")
        self.assertFalse(result.seed_sent)
        self.assertEqual(result.token_trace, [])
        self.assertEqual(result.token_trace_source, "unavailable")
        self.assertNotIn("seed", post.call_args.args[2])
        self.assertNotIn("test-secret", json.dumps(result.raw_response))

    def test_top_logprob_parser_does_not_invent_entropy(self) -> None:
        trace = _openai_token_trace(
            {
                "logprobs": {
                    "content": [
                        {
                            "token": "word",
                            "logprob": -0.2,
                            "top_logprobs": [{"token": "word", "logprob": -0.2}],
                        }
                    ]
                }
            }
        )
        self.assertEqual(trace[0]["entropy_nats"], None)
        self.assertEqual(trace[0]["logit_margin"], None)
        self.assertEqual(trace[0]["selected_rank_in_returned_top_k"], 1)


if __name__ == "__main__":
    unittest.main()
