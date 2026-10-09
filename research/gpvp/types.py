"""Shared request/response data types for the GPVP adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class CompletionRequest:
    provider: str
    model_id: str
    system_prompt: str
    user_prompt: str
    temperature: float
    seed: int
    max_tokens: int
    timeout_seconds: float = 90.0
    top_logprobs: int = 5
    request_logprobs: bool = True
    model_version_override: str | None = None
    intervention: dict[str, Any] | None = None
    capture_full_logits: bool = False


@dataclass
class CompletionResult:
    text: str
    raw_response: dict[str, Any]
    returned_model_id: str | None = None
    model_version: str | None = None
    model_version_source: str = "unreported"
    response_id: str | None = None
    finish_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    token_trace: list[dict[str, Any]] = field(default_factory=list)
    token_trace_source: str = "unavailable"
    seed_sent: bool = False
    full_logits: list[list[float]] | None = None
    full_logits_token_ids: list[int] | None = None
    adapter_metadata: dict[str, Any] = field(default_factory=dict)


class CompletionAdapter(Protocol):
    provider_name: str

    def complete(self, request: CompletionRequest) -> CompletionResult:
        """Return one completion or raise an adapter error."""


class AdapterError(RuntimeError):
    """Provider failure with safe, serializable diagnostic fields."""

    def __init__(
        self,
        message: str,
        *,
        category: str = "provider_error",
        status_code: int | None = None,
        retryable: bool = False,
        retry_after_seconds: float | None = None,
        response_excerpt: str | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.status_code = status_code
        self.retryable = retryable
        self.retry_after_seconds = retry_after_seconds
        self.response_excerpt = response_excerpt

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "message": str(self),
            "status_code": self.status_code,
            "retryable": self.retryable,
            "retry_after_seconds": self.retry_after_seconds,
            "response_excerpt": self.response_excerpt,
        }
