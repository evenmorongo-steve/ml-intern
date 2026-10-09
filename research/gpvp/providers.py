"""Minimal, provenance-conscious HTTP adapters for supported inference APIs.

This module has no provider SDK dependency. API-specific responses are preserved
as returned JSON; missing token probabilities remain missing rather than being
estimated from generated text.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from .types import AdapterError, CompletionRequest, CompletionResult


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    key_env: str
    base_url: str
    api_style: str
    seed_field: str | None
    version_header: str | None = None
    version_value: str | None = None


PROVIDER_SPECS: dict[str, ProviderSpec] = {
    "openai": ProviderSpec(
        "openai", "OPENAI_API_KEY", "https://api.openai.com/v1", "openai", "seed"
    ),
    "anthropic": ProviderSpec(
        "anthropic",
        "ANTHROPIC_API_KEY",
        "https://api.anthropic.com/v1",
        "anthropic",
        None,
        "anthropic-version",
        "2023-06-01",
    ),
    "google": ProviderSpec(
        "google",
        "GOOGLE_API_KEY",
        "https://generativelanguage.googleapis.com/v1beta",
        "google",
        "seed",
    ),
    "mistral": ProviderSpec(
        "mistral",
        "MISTRAL_API_KEY",
        "https://api.mistral.ai/v1",
        "openai",
        "random_seed",
    ),
    "deepseek": ProviderSpec(
        "deepseek", "DEEPSEEK_API_KEY", "https://api.deepseek.com/v1", "openai", None
    ),
    "qwen": ProviderSpec(
        "qwen",
        "DASHSCOPE_API_KEY",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "openai",
        "seed",
    ),
    "xai": ProviderSpec("xai", "XAI_API_KEY", "https://api.x.ai/v1", "openai", None),
}


@dataclass(frozen=True)
class HttpResponse:
    status: int
    headers: dict[str, str]
    body: bytes


def _safe_excerpt(text: str, limit: int = 1000) -> str:
    text = re.sub(r"(?i)bearer\s+[^\s,;]+", "Bearer [REDACTED]", text)
    text = re.sub(r"(?i)(api[_-]?key\s*[:=]\s*)[^\s,;]+", r"\1[REDACTED]", text)
    return text[:limit]


def _post_json(
    url: str, headers: dict[str, str], payload: dict[str, Any], timeout: float
) -> HttpResponse:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return HttpResponse(
                status=int(response.status),
                headers={
                    key.casefold(): value for key, value in response.headers.items()
                },
                body=response.read(),
            )
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        retry_after = _retry_after(exc.headers.get("Retry-After"))
        status = int(exc.code)
        raise AdapterError(
            f"Provider returned HTTP {status}.",
            category="rate_limited" if status == 429 else "http_error",
            status_code=status,
            retryable=status == 429 or status >= 500,
            retry_after_seconds=retry_after,
            response_excerpt=_safe_excerpt(body),
        ) from None
    except urllib.error.URLError as exc:
        reason = str(getattr(exc, "reason", "network error"))
        category = "timeout" if "timed out" in reason.casefold() else "network_error"
        raise AdapterError(
            f"Provider request failed ({category}).",
            category=category,
            retryable=True,
            response_excerpt=_safe_excerpt(reason, 300),
        ) from None
    except TimeoutError:
        raise AdapterError(
            "Provider request timed out.", category="timeout", retryable=True
        ) from None


def _retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


def _read_json(response: HttpResponse) -> dict[str, Any]:
    try:
        value = json.loads(response.body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AdapterError(
            "Provider returned a non-JSON response.",
            category="invalid_response",
            status_code=response.status,
            retryable=False,
            response_excerpt=_safe_excerpt(
                response.body.decode("utf-8", errors="replace")
            ),
        ) from exc
    if not isinstance(value, dict):
        raise AdapterError(
            "Provider response JSON was not an object.", category="invalid_response"
        )
    return value


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for part in value:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "".join(parts)
    return ""


def _openai_token_trace(choice: dict[str, Any]) -> list[dict[str, Any]]:
    logprobs = choice.get("logprobs")
    content = logprobs.get("content") if isinstance(logprobs, dict) else None
    if not isinstance(content, list):
        return []
    trace: list[dict[str, Any]] = []
    for position, item in enumerate(content):
        if not isinstance(item, dict):
            continue
        top: list[dict[str, Any]] = []
        for candidate in item.get("top_logprobs", []) or []:
            if isinstance(candidate, dict):
                top.append(
                    {
                        "token": candidate.get("token"),
                        "logprob": candidate.get("logprob"),
                    }
                )
        selected = item.get("token")
        selected_logprob = item.get("logprob")
        rank: int | None = None
        if selected is not None and top:
            ordered = sorted(
                (
                    candidate
                    for candidate in top
                    if isinstance(candidate.get("logprob"), (int, float))
                ),
                key=lambda candidate: float(candidate["logprob"]),
                reverse=True,
            )
            match = next(
                (
                    i + 1
                    for i, candidate in enumerate(ordered)
                    if candidate.get("token") == selected
                ),
                None,
            )
            rank = match
        trace.append(
            {
                "position": position,
                "token": selected,
                "token_id": None,
                "selected_logprob": selected_logprob,
                "top_logprobs": top,
                "entropy_nats": None,
                "logit_margin": None,
                "selected_rank_in_returned_top_k": rank,
                "source": "provider_logprobs",
            }
        )
    return trace


def _google_token_trace(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    result = candidate.get("logprobsResult")
    if not isinstance(result, dict):
        return []
    chosen = result.get("chosenCandidates") or []
    top_candidates = result.get("topCandidates") or []
    trace: list[dict[str, Any]] = []
    for position, chosen_item in enumerate(chosen):
        if not isinstance(chosen_item, dict):
            continue
        top: list[dict[str, Any]] = []
        if position < len(top_candidates) and isinstance(
            top_candidates[position], dict
        ):
            for item in top_candidates[position].get("candidates", []) or []:
                if isinstance(item, dict):
                    top.append(
                        {
                            "token": item.get("token"),
                            "logprob": item.get("logProbability"),
                        }
                    )
        token = chosen_item.get("token")
        rank = None
        if token is not None and top:
            ordered = sorted(
                (
                    entry
                    for entry in top
                    if isinstance(entry.get("logprob"), (int, float))
                ),
                key=lambda entry: float(entry["logprob"]),
                reverse=True,
            )
            rank = next(
                (
                    i + 1
                    for i, entry in enumerate(ordered)
                    if entry.get("token") == token
                ),
                None,
            )
        trace.append(
            {
                "position": position,
                "token": token,
                "token_id": None,
                "selected_logprob": chosen_item.get("logProbability"),
                "top_logprobs": top,
                "entropy_nats": None,
                "logit_margin": None,
                "selected_rank_in_returned_top_k": rank,
                "source": "provider_logprobs",
            }
        )
    return trace


class ProviderAdapter:
    """Single-call adapter for OpenAI, Anthropic, Google, Mistral, DeepSeek, Qwen, and xAI."""

    def __init__(
        self,
        provider_name: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        environ: dict[str, str] | None = None,
    ) -> None:
        normalized = provider_name.casefold().replace("-", "").replace("_", "")
        aliases = {"googleai": "google", "xai": "xai"}
        normalized = aliases.get(normalized, normalized)
        if normalized not in PROVIDER_SPECS:
            raise ValueError(
                f"Unsupported provider {provider_name!r}. Supported: {', '.join(PROVIDER_SPECS)}"
            )
        self.spec = PROVIDER_SPECS[normalized]
        env = os.environ if environ is None else environ
        self.api_key = api_key or env.get(self.spec.key_env)
        override_name = f"GPVP_{self.spec.name.upper()}_BASE_URL"
        self.base_url = (
            base_url or env.get(override_name) or self.spec.base_url
        ).rstrip("/")
        if not self.api_key:
            raise AdapterError(
                f"No credential found in {self.spec.key_env}.",
                category="missing_credential",
                retryable=False,
            )

    @property
    def provider_name(self) -> str:
        return self.spec.name

    def complete(self, request: CompletionRequest) -> CompletionResult:
        if request.provider.casefold().replace("-", "").replace("_", "") not in {
            self.spec.name,
            "googleai" if self.spec.name == "google" else self.spec.name,
        }:
            raise ValueError(
                f"Request provider {request.provider!r} does not match adapter {self.spec.name!r}."
            )
        if self.spec.api_style == "openai":
            return self._complete_openai_compatible(request)
        if self.spec.api_style == "anthropic":
            return self._complete_anthropic(request)
        if self.spec.api_style == "google":
            return self._complete_google(request)
        raise AssertionError(f"Unknown API style: {self.spec.api_style}")

    def _headers(
        self, *, anthropic: bool = False, google: bool = False
    ) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "hermes-gpvp/0.1.0",
        }
        if anthropic:
            headers["x-api-key"] = str(self.api_key)
            if self.spec.version_header and self.spec.version_value:
                headers[self.spec.version_header] = self.spec.version_value
        elif google:
            headers["x-goog-api-key"] = str(self.api_key)
        else:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _complete_openai_compatible(
        self, request: CompletionRequest
    ) -> CompletionResult:
        url = self.base_url
        if not url.endswith("/chat/completions"):
            if url.endswith("/v1") or "/v1/" in url:
                url += "/chat/completions"
            else:
                url += "/v1/chat/completions"
        payload: dict[str, Any] = {
            "model": request.model_id,
            "messages": [
                {"role": "system", "content": request.system_prompt},
                {"role": "user", "content": request.user_prompt},
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        seed_sent = self.spec.seed_field is not None
        if seed_sent:
            payload[self.spec.seed_field or "seed"] = request.seed
        if request.request_logprobs:
            payload["logprobs"] = True
            payload["top_logprobs"] = request.top_logprobs
        response = _post_json(url, self._headers(), payload, request.timeout_seconds)
        raw = _read_json(response)
        choices = raw.get("choices")
        if (
            not isinstance(choices, list)
            or not choices
            or not isinstance(choices[0], dict)
        ):
            raise AdapterError(
                "Provider response did not contain a chat completion choice.",
                category="invalid_response",
            )
        choice = choices[0]
        message = choice.get("message")
        text = _as_text(message.get("content") if isinstance(message, dict) else None)
        model_field = raw.get("model")
        model_version = request.model_version_override or (
            str(model_field) if model_field else None
        )
        version_source = (
            "operator_frozen"
            if request.model_version_override
            else (
                "provider-reported model field; may be an alias"
                if model_field
                else "unreported"
            )
        )
        return CompletionResult(
            text=text,
            raw_response=raw,
            returned_model_id=str(model_field) if model_field else None,
            model_version=model_version,
            model_version_source=version_source,
            response_id=str(raw.get("id")) if raw.get("id") else None,
            finish_reason=str(choice.get("finish_reason"))
            if choice.get("finish_reason") is not None
            else None,
            usage=raw.get("usage") if isinstance(raw.get("usage"), dict) else {},
            token_trace=_openai_token_trace(choice),
            token_trace_source="provider_logprobs"
            if _openai_token_trace(choice)
            else "unavailable",
            seed_sent=seed_sent,
            adapter_metadata={
                "system_fingerprint": raw.get("system_fingerprint"),
                "http_status": response.status,
                "provider_request_id": response.headers.get("x-request-id")
                or response.headers.get("request-id"),
                "seed_parameter": self.spec.seed_field,
            },
        )

    def _complete_anthropic(self, request: CompletionRequest) -> CompletionResult:
        url = self.base_url
        if not url.endswith("/messages"):
            if url.endswith("/v1") or "/v1/" in url:
                url += "/messages"
            else:
                url += "/v1/messages"
        payload = {
            "model": request.model_id,
            "system": request.system_prompt,
            "messages": [{"role": "user", "content": request.user_prompt}],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        response = _post_json(
            url, self._headers(anthropic=True), payload, request.timeout_seconds
        )
        raw = _read_json(response)
        text = _as_text(raw.get("content"))
        model_field = raw.get("model")
        model_version = request.model_version_override or (
            str(model_field) if model_field else None
        )
        version_source = (
            "operator_frozen"
            if request.model_version_override
            else (
                "provider-reported model field; may be an alias"
                if model_field
                else "unreported"
            )
        )
        return CompletionResult(
            text=text,
            raw_response=raw,
            returned_model_id=str(model_field) if model_field else None,
            model_version=model_version,
            model_version_source=version_source,
            response_id=str(raw.get("id")) if raw.get("id") else None,
            finish_reason=str(raw.get("stop_reason"))
            if raw.get("stop_reason") is not None
            else None,
            usage=raw.get("usage") if isinstance(raw.get("usage"), dict) else {},
            token_trace=[],
            token_trace_source="unavailable",
            seed_sent=False,
            adapter_metadata={
                "http_status": response.status,
                "provider_request_id": response.headers.get("request-id"),
                "seed_parameter": None,
                "logprobs_available": False,
            },
        )

    def _complete_google(self, request: CompletionRequest) -> CompletionResult:
        model = urllib.parse.quote(request.model_id, safe="")
        base = self.base_url
        if not base.endswith("/v1beta") and "/models/" not in base:
            base += "/v1beta"
        url = f"{base}/models/{model}:generateContent"
        generation_config: dict[str, Any] = {
            "temperature": request.temperature,
            "maxOutputTokens": request.max_tokens,
        }
        seed_sent = self.spec.seed_field is not None
        if seed_sent:
            generation_config["seed"] = request.seed
        if request.request_logprobs:
            generation_config["responseLogprobs"] = True
            generation_config["logprobs"] = request.top_logprobs
        payload: dict[str, Any] = {
            "systemInstruction": {"parts": [{"text": request.system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": request.user_prompt}]}],
            "generationConfig": generation_config,
        }
        response = _post_json(
            url, self._headers(google=True), payload, request.timeout_seconds
        )
        raw = _read_json(response)
        candidates = raw.get("candidates")
        if (
            not isinstance(candidates, list)
            or not candidates
            or not isinstance(candidates[0], dict)
        ):
            raise AdapterError(
                "Provider response did not contain a candidate.",
                category="invalid_response",
            )
        candidate = candidates[0]
        content = candidate.get("content")
        text = _as_text(content.get("parts") if isinstance(content, dict) else None)
        trace = _google_token_trace(candidate)
        model_version = request.model_version_override
        version_source = (
            "operator_frozen" if request.model_version_override else "unreported"
        )
        return CompletionResult(
            text=text,
            raw_response=raw,
            returned_model_id=request.model_id,
            model_version=model_version,
            model_version_source=version_source,
            response_id=str(raw.get("responseId")) if raw.get("responseId") else None,
            finish_reason=str(candidate.get("finishReason"))
            if candidate.get("finishReason") is not None
            else None,
            usage=raw.get("usageMetadata")
            if isinstance(raw.get("usageMetadata"), dict)
            else {},
            token_trace=trace,
            token_trace_source="provider_logprobs" if trace else "unavailable",
            seed_sent=seed_sent,
            adapter_metadata={
                "http_status": response.status,
                "provider_request_id": response.headers.get("x-request-id"),
                "seed_parameter": "seed" if seed_sent else None,
            },
        )


def available_providers() -> tuple[str, ...]:
    return tuple(PROVIDER_SPECS)
