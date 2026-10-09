"""Optional open-weight adapter using Hugging Face Transformers forward hooks.

This implementation is intentionally limited to decoder-only Transformers
models with a discoverable block list. Unsupported architectures/interventions
fail closed. TransformerLens, nnsight, and vLLM backends can implement the same
CompletionAdapter protocol but are not silently substituted here.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .types import AdapterError, CompletionRequest, CompletionResult

_LAYER_PATHS = (
    ("model", "layers"),
    ("transformer", "h"),
    ("gpt_neox", "layers"),
    ("model", "decoder", "layers"),
    ("decoder", "layers"),
    ("language_model", "layers"),
    ("model", "language_model", "layers"),
)


def _attribute_path(value: Any, path: tuple[str, ...]) -> Any:
    for component in path:
        if not hasattr(value, component):
            return None
        value = getattr(value, component)
    return value


def find_decoder_layers(model: Any) -> tuple[str, Any]:
    for path in _LAYER_PATHS:
        layers = _attribute_path(model, path)
        if (
            layers is not None
            and hasattr(layers, "__len__")
            and hasattr(layers, "__getitem__")
            and len(layers) > 0
        ):
            return ".".join(path), layers
    raise AdapterError(
        "Could not find a supported decoder block list; no activation intervention was applied.",
        category="unsupported_architecture",
        retryable=False,
    )


def _replace_primary_output(output: Any, hidden: Any) -> Any:
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    if isinstance(output, list):
        return [hidden, *output[1:]]
    if isinstance(output, dict):
        updated = dict(output)
        if "hidden_states" in updated:
            updated["hidden_states"] = hidden
            return updated
        raise AdapterError(
            "Layer returned a dictionary without hidden_states.",
            category="unsupported_layer_output",
        )
    return hidden


def _load_vector(intervention: dict[str, Any], hidden_size: int, torch: Any) -> Any:
    inline = intervention.get("vector")
    if isinstance(inline, list):
        values = inline
    else:
        vector_path = intervention.get("vector_path")
        if not vector_path:
            raise ValueError(
                "steer_vector intervention requires vector or vector_path."
            )
        path = Path(str(vector_path))
        if path.suffix.casefold() != ".json":
            raise ValueError(
                "Steering vectors must be JSON arrays; executable/pickle formats are not accepted."
            )
        value = json.loads(path.read_text(encoding="utf-8"))
        values = value.get("vector") if isinstance(value, dict) else value
    if not isinstance(values, list) or len(values) != hidden_size:
        raise ValueError(
            f"Steering vector length must equal hidden size {hidden_size}."
        )
    vector = torch.tensor(values, dtype=torch.float32)
    if not bool(torch.isfinite(vector).all().item()):
        raise ValueError("Steering vector contains a non-finite value.")
    return vector


def _entropy(logits: Any, torch: Any) -> float:
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    probabilities = log_probs.exp()
    return float((-(probabilities * log_probs).sum()).item())


def _token_trace(
    *,
    sequence_ids: Any,
    raw_logits: Any,
    sampling_scores: Any,
    tokenizer: Any,
    top_k: int,
    torch: Any,
) -> list[dict[str, Any]]:
    count = min(int(sequence_ids.shape[0]), len(raw_logits), len(sampling_scores))
    trace: list[dict[str, Any]] = []
    for position in range(count):
        token_id = int(sequence_ids[position].item())
        raw = raw_logits[position][0].float()
        sampled = sampling_scores[position][0].float()
        raw_log_probs = torch.log_softmax(raw, dim=-1)
        sample_log_probs = torch.log_softmax(sampled, dim=-1)
        raw_top = torch.topk(raw, k=min(top_k, int(raw.shape[-1])))
        sample_top = torch.topk(
            sample_log_probs, k=min(top_k, int(sample_log_probs.shape[-1]))
        )
        raw_ids = [int(value) for value in raw_top.indices.tolist()]
        sample_ids = [int(value) for value in sample_top.indices.tolist()]
        selected_rank = 1 + int((raw > raw[token_id]).sum().item())
        token_str = tokenizer.convert_ids_to_tokens(token_id)
        trace.append(
            {
                "position": position,
                "token": token_str,
                "token_id": token_id,
                "selected_logprob": float(sample_log_probs[token_id].item()),
                "base_logprob_temperature_1": float(raw_log_probs[token_id].item()),
                "top_logprobs": [
                    {
                        "token_id": candidate_id,
                        "token": tokenizer.convert_ids_to_tokens(candidate_id),
                        "logprob": float(sample_log_probs[candidate_id].item()),
                        "raw_logit": float(raw[candidate_id].item()),
                    }
                    for candidate_id in sample_ids
                ],
                "base_top_k": [
                    {
                        "token_id": candidate_id,
                        "token": tokenizer.convert_ids_to_tokens(candidate_id),
                        "raw_logit": float(raw[candidate_id].item()),
                        "logprob_temperature_1": float(
                            raw_log_probs[candidate_id].item()
                        ),
                    }
                    for candidate_id in raw_ids
                ],
                "entropy_nats": _entropy(raw, torch),
                "sampling_entropy_nats": _entropy(sampled, torch),
                "logit_margin": float((raw_top.values[0] - raw_top.values[1]).item())
                if raw_top.values.numel() > 1
                else None,
                "selected_rank_in_returned_top_k": selected_rank,
                "source": "local_logits",
            }
        )
    return trace


class TransformersLocalAdapter:
    provider_name = "local-transformers"

    def __init__(
        self,
        model_id: str,
        *,
        revision: str,
        device: str | None = None,
        dtype: str = "auto",
        local_files_only: bool = True,
        trust_remote_code: bool = False,
        capture_activations: bool = False,
    ) -> None:
        if not revision or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
            raise ValueError(
                "An immutable 40-character model revision is required for local evidence runs."
            )
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Local inference requires optional packages torch and transformers; "
                "install research/gpvp/requirements-open-weights.txt."
            ) from exc

        self.torch = torch
        self.model_id = model_id
        self.revision = revision.lower()
        self.capture_activations = capture_activations
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            revision=self.revision,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
        )
        load_kwargs: dict[str, Any] = {
            "revision": self.revision,
            "local_files_only": local_files_only,
            "trust_remote_code": trust_remote_code,
        }
        if dtype == "auto":
            load_kwargs["torch_dtype"] = "auto"
        elif dtype in {"float32", "float16", "bfloat16"}:
            load_kwargs["torch_dtype"] = getattr(torch, dtype)
        else:
            raise ValueError("dtype must be auto, float32, float16, or bfloat16.")
        if device:
            load_kwargs["device_map"] = {"": device}
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        self.model.eval()
        self.layer_path, self.layers = find_decoder_layers(self.model)
        self._hidden_size = int(
            getattr(self.model.config, "hidden_size", 0)
            or getattr(self.model.config, "n_embd", 0)
        )
        self._commit_hash = (
            getattr(self.model.config, "_commit_hash", None)
            or getattr(self.tokenizer, "init_kwargs", {}).get("_commit_hash")
            or self.revision
        )

    def complete(self, request: CompletionRequest) -> CompletionResult:
        if request.model_id != self.model_id:
            raise ValueError(
                f"Request model {request.model_id!r} does not match loaded model {self.model_id!r}."
            )
        torch = self.torch
        intervention = request.intervention or {"kind": "none"}
        kind = str(intervention.get("kind", "none"))
        layer_index = intervention.get("layer")
        activation_summaries: list[dict[str, Any]] = []
        handles: list[Any] = []
        active_layer: int | None = None
        vector = None

        if kind not in {"none", "skip_layer", "steer_vector"}:
            raise ValueError(f"Unsupported Transformers intervention {kind!r}.")
        if kind in {"skip_layer", "steer_vector"}:
            if not isinstance(layer_index, int) or not (
                0 <= layer_index < len(self.layers)
            ):
                raise ValueError(
                    f"Intervention layer must be an integer in [0, {len(self.layers) - 1}]."
                )
            active_layer = layer_index
            if kind == "steer_vector":
                if self._hidden_size <= 0:
                    raise ValueError(
                        "Model config does not expose a supported hidden_size/n_embd."
                    )
                vector = _load_vector(intervention, self._hidden_size, torch)
                strength = float(intervention.get("strength", 1.0))
                vector = vector.to(dtype=torch.float32)
            else:
                strength = 0.0
        else:
            strength = 0.0

        def make_hook(index: int):
            def hook(module: Any, inputs: tuple[Any, ...], output: Any) -> Any:
                if not inputs or not hasattr(inputs[0], "shape"):
                    raise AdapterError(
                        "Selected block input has an unsupported structure.",
                        category="unsupported_layer_io",
                    )
                source = inputs[0]
                hidden = output[0] if isinstance(output, (tuple, list)) else output
                if not hasattr(hidden, "shape"):
                    raise AdapterError(
                        "Selected block output has an unsupported structure.",
                        category="unsupported_layer_io",
                    )
                is_intervention_layer = index == active_layer
                should_log = self.capture_activations or is_intervention_layer
                if not should_log and not is_intervention_layer:
                    return output
                before_norm = float(source.float().norm(dim=-1).mean().item())
                after_norm = float(hidden.float().norm(dim=-1).mean().item())
                if is_intervention_layer and kind == "skip_layer":
                    changed = source
                elif is_intervention_layer and kind == "steer_vector":
                    changed = hidden + strength * vector.to(
                        device=hidden.device, dtype=hidden.dtype
                    ).view(1, 1, -1)
                else:
                    changed = hidden
                changed_norm = float(changed.float().norm(dim=-1).mean().item())
                if should_log:
                    activation_summaries.append(
                        {
                            "layer_index": index,
                            "input_mean_l2_norm": before_norm,
                            "layer_output_mean_l2_norm": after_norm,
                            "intervened_output_mean_l2_norm": changed_norm,
                            "sequence_positions": int(hidden.shape[-2])
                            if hidden.ndim >= 2
                            else None,
                            "intervention": kind if is_intervention_layer else "none",
                            "strength": strength if is_intervention_layer else 0.0,
                        }
                    )
                if is_intervention_layer:
                    return _replace_primary_output(output, changed)
                return output

            return hook

        if active_layer is not None or self.capture_activations:
            indices = (
                range(len(self.layers)) if self.capture_activations else (active_layer,)
            )
            for index in indices:
                if index is not None:
                    handles.append(
                        self.layers[index].register_forward_hook(make_hook(index))
                    )

        messages = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.user_prompt},
        ]
        chat_rendering = "plain_concat_no_chat_template"
        if getattr(self.tokenizer, "chat_template", None):
            try:
                encoded = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_tensors="pt",
                    return_dict=True,
                )
            except TypeError:
                token_tensor = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_tensors="pt",
                )
                encoded = {"input_ids": token_tensor}
            if not isinstance(encoded, dict) and hasattr(encoded, "keys"):
                encoded = {key: encoded[key] for key in encoded}
            elif not isinstance(encoded, dict):
                encoded = {"input_ids": encoded}
            if "attention_mask" not in encoded:
                encoded["attention_mask"] = torch.ones_like(encoded["input_ids"])
            chat_rendering = "tokenizer_chat_template"
        else:
            encoded = self.tokenizer(
                f"{request.system_prompt}\n\n{request.user_prompt}",
                return_tensors="pt",
            )
        input_device = getattr(self.model, "device", None)
        if input_device is not None:
            encoded = {key: value.to(input_device) for key, value in encoded.items()}
        input_length = int(encoded["input_ids"].shape[-1])
        generation_kwargs: dict[str, Any] = {
            "max_new_tokens": request.max_tokens,
            "do_sample": request.temperature > 0,
            # Disabling KV caching makes a block-skip intervention apply to
            # the full prefix at every decode step, rather than reusing stale
            # per-layer cache entries computed before the hook altered output.
            "use_cache": kind != "skip_layer",
            "return_dict_in_generate": True,
            "output_scores": True,
            "output_logits": True,
            "pad_token_id": (
                self.tokenizer.pad_token_id
                if self.tokenizer.pad_token_id is not None
                else self.tokenizer.eos_token_id
            ),
        }
        if request.temperature > 0:
            generation_kwargs["temperature"] = request.temperature
            generation_kwargs["top_p"] = 1.0
            generation_kwargs["top_k"] = 0

        # Keep PRNG changes local to this call; paired conditions use the same
        # seed without mutating process-global random state after generation.
        devices: list[int] = []
        if (
            torch.cuda.is_available()
            and input_device is not None
            and str(input_device).startswith("cuda")
        ):
            try:
                devices = [torch.device(input_device).index or 0]
            except (TypeError, RuntimeError, ValueError):
                devices = [torch.cuda.current_device()]
        try:
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(request.seed)
                if devices:
                    torch.cuda.manual_seed_all(request.seed)
                try:
                    generated = self.model.generate(**encoded, **generation_kwargs)
                except TypeError as exc:
                    # Older Transformers releases may not support output_logits.
                    if "output_logits" not in str(exc):
                        raise
                    generation_kwargs.pop("output_logits", None)
                    generated = self.model.generate(**encoded, **generation_kwargs)
        except AdapterError:
            raise
        except Exception as exc:
            raise AdapterError(
                f"Local model generation failed ({type(exc).__name__}).",
                category="local_generation_error",
                retryable=False,
                response_excerpt=str(exc)[:500],
            ) from exc
        finally:
            for handle in handles:
                handle.remove()

        sequence = generated.sequences[0]
        new_token_ids = sequence[input_length:]
        text = self.tokenizer.decode(new_token_ids, skip_special_tokens=True)
        scores = list(getattr(generated, "scores", ()) or ())
        raw_output_logits = getattr(generated, "logits", None)
        if raw_output_logits is None:
            raw_logits = scores
            raw_source = "generation_scores_processed"
        else:
            raw_logits = list(raw_output_logits)
            raw_source = "generation_logits_unprocessed"
        # `scores` are the actual processed distributions used by generate().
        # If they are absent, logits remain available but sampling entropy does not.
        sampling_scores = scores if scores else raw_logits
        token_trace: list[dict[str, Any]] = []
        if raw_logits and sampling_scores:
            token_trace = _token_trace(
                sequence_ids=new_token_ids,
                raw_logits=raw_logits,
                sampling_scores=sampling_scores,
                tokenizer=self.tokenizer,
                top_k=max(1, request.top_logprobs),
                torch=torch,
            )
            for trace_row in token_trace:
                trace_row["raw_logit_source"] = raw_source

        full_logits: list[list[float]] | None = None
        full_token_ids: list[int] | None = None
        if request.capture_full_logits:
            full_logits = []
            full_token_ids = []
            for position in range(min(len(raw_logits), int(new_token_ids.shape[0]))):
                full_logits.append(
                    [float(value) for value in raw_logits[position][0].float().tolist()]
                )
                full_token_ids.append(int(new_token_ids[position].item()))

        eos_id = self.tokenizer.eos_token_id
        reached_eos = bool(
            new_token_ids.numel()
            and eos_id is not None
            and int(new_token_ids[-1].item()) == eos_id
        )
        if reached_eos:
            local_finish_reason = "eos"
        elif int(new_token_ids.shape[0]) >= request.max_tokens:
            local_finish_reason = "max_new_tokens"
        else:
            local_finish_reason = "other_stopping_condition"
        raw_response = {
            "model_id": self.model_id,
            "revision": self._commit_hash,
            "input_token_ids": [
                int(value) for value in encoded["input_ids"][0].tolist()
            ],
            "generated_token_ids": [int(value) for value in new_token_ids.tolist()],
            "finish_reason": local_finish_reason,
            "raw_logit_source": raw_source,
            "chat_rendering": chat_rendering,
        }
        vector_hash = None
        if kind == "steer_vector" and vector is not None:
            vector_hash = hashlib.sha256(
                json.dumps(vector.tolist(), separators=(",", ":")).encode("utf-8")
            ).hexdigest()
        return CompletionResult(
            text=text,
            raw_response=raw_response,
            returned_model_id=self.model_id,
            model_version=str(self._commit_hash),
            model_version_source="local_revision",
            response_id=None,
            finish_reason=raw_response["finish_reason"],
            usage={
                "input_tokens": input_length,
                "output_tokens": int(new_token_ids.shape[0]),
            },
            token_trace=token_trace,
            token_trace_source="local_logits" if token_trace else "unavailable",
            seed_sent=True,
            full_logits=full_logits,
            full_logits_token_ids=full_token_ids,
            adapter_metadata={
                "revision": self._commit_hash,
                "layer_list_path": self.layer_path,
                "layer_count": len(self.layers),
                "activation_capture_enabled": self.capture_activations,
                "activation_summary_count": len(activation_summaries),
                "activation_summaries": activation_summaries,
                "intervention": {
                    "kind": kind,
                    "layer": active_layer,
                    "strength": strength,
                    "vector_sha256": vector_hash,
                },
                "torch_version": torch.__version__,
            },
        )
