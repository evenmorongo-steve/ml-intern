"""Seeded scheduling, retry-safe execution, provenance records, and chains."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import random
import threading
import time
import uuid
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

from .prompts import Battery, ChainSpec, PromptCase
from .storage import RecordWriter, write_logits_sidecar
from .types import AdapterError, CompletionAdapter, CompletionRequest, CompletionResult

SCHEMA_VERSION = "gpvp.transcript/1.0"


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class ExperimentJob:
    job_id: str
    schedule_index: int
    job_kind: str
    replicate: int
    seed: int
    temperature: float
    case: PromptCase | None = None
    chain: ChainSpec | None = None
    intervention_id: str | None = None
    intervention: dict[str, Any] | None = None


def make_schedule(
    battery: Battery,
    *,
    n: int,
    run_seed: int,
    temperatures: Iterable[float] = (0.7,),
    randomization_seed: int | None = None,
    include_chains: bool = True,
    skip_unresolved_lexicon: bool = False,
    interventions: Iterable[dict[str, Any]] | None = None,
    seed_pool: Iterable[int] | None = None,
) -> list[ExperimentJob]:
    """Create a reproducible randomized list; paired conditions share each seed."""
    if n < 1:
        raise ValueError("n must be at least one.")
    temps = tuple(float(value) for value in temperatures)
    if not temps:
        raise ValueError("At least one temperature is required.")
    if any(value < 0 for value in temps):
        raise ValueError("Temperature cannot be negative.")
    variants = (
        [dict(value) for value in interventions]
        if interventions is not None
        else [None]
    )
    if not variants:
        raise ValueError("An intervention plan cannot be empty.")
    variant_ids: list[str] = []
    for value in variants:
        if value is None:
            variant_ids.append("none")
            continue
        variant_id = str(value.get("intervention_id") or value.get("id") or "").strip()
        if not variant_id:
            raise ValueError(
                "Every intervention variant requires a unique id/intervention_id."
            )
        variant_ids.append(variant_id)
    if len(variant_ids) != len(set(variant_ids)):
        raise ValueError("Intervention variant IDs must be unique.")

    registered_seeds = (
        tuple(int(value) for value in seed_pool) if seed_pool is not None else ()
    )
    if seed_pool is not None and not registered_seeds:
        raise ValueError("seed_pool cannot be empty.")
    if len(set(registered_seeds)) != len(registered_seeds):
        raise ValueError(
            "seed_pool values must be unique; repetitions are scheduled by cycling the pool."
        )
    unresolved_prompt_ids = {
        case.prompt_id
        for case in battery.cases
        if case.requires_lexicon or "{{LEXICON}}" in case.user_prompt
    }
    jobs: list[ExperimentJob] = []
    for replicate in range(n):
        # The same seed is assigned to all matched conditions and intervention
        # variants in a replicate, and reused across the temperature sweep.
        paired_seed = (
            registered_seeds[replicate % len(registered_seeds)]
            if registered_seeds
            else run_seed + replicate
        )
        for temperature in temps:
            for case in battery.cases:
                if skip_unresolved_lexicon and case.prompt_id in unresolved_prompt_ids:
                    continue
                for variant, variant_id in zip(variants, variant_ids):
                    suffix = f":i{variant_id}" if variant is not None else ""
                    jobs.append(
                        ExperimentJob(
                            job_id=f"single:{case.case_id}:r{replicate}:t{temperature:g}{suffix}",
                            schedule_index=-1,
                            job_kind="single",
                            replicate=replicate,
                            seed=paired_seed,
                            temperature=temperature,
                            case=case,
                            intervention_id=variant_id if variant is not None else None,
                            intervention=variant,
                        )
                    )
            if include_chains:
                for chain in battery.chains:
                    for variant, variant_id in zip(variants, variant_ids):
                        suffix = f":i{variant_id}" if variant is not None else ""
                        jobs.append(
                            ExperimentJob(
                                job_id=f"chain:{chain.chain_id}:r{replicate}:t{temperature:g}{suffix}",
                                schedule_index=-1,
                                job_kind="chain",
                                replicate=replicate,
                                seed=paired_seed,
                                temperature=temperature,
                                chain=chain,
                                intervention_id=variant_id
                                if variant is not None
                                else None,
                                intervention=variant,
                            )
                        )
    if not jobs:
        raise ValueError(
            "The selected battery, temperature, and control options produced no jobs."
        )
    rng = random.Random(run_seed if randomization_seed is None else randomization_seed)
    rng.shuffle(jobs)
    return [
        ExperimentJob(
            job_id=job.job_id,
            schedule_index=index,
            job_kind=job.job_kind,
            replicate=job.replicate,
            seed=job.seed,
            temperature=job.temperature,
            case=job.case,
            chain=job.chain,
            intervention_id=job.intervention_id,
            intervention=job.intervention,
        )
        for index, job in enumerate(jobs)
    ]


class RateLimiter:
    """Thread-safe fixed-spacing limiter shared by all provider workers."""

    def __init__(self, max_requests_per_minute: float | None) -> None:
        if max_requests_per_minute is not None and max_requests_per_minute <= 0:
            raise ValueError("max_requests_per_minute must be positive or None.")
        self.interval = (
            60.0 / max_requests_per_minute if max_requests_per_minute else 0.0
        )
        self._lock = threading.Lock()
        self._next_start = 0.0

    def wait(self) -> None:
        if self.interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next_start - now)
            self._next_start = max(now, self._next_start) + self.interval
        if delay:
            time.sleep(delay)


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 2
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 30.0

    def delay(self, attempt: int, seed: int, retry_after: float | None = None) -> float:
        if retry_after is not None:
            return min(self.max_delay_seconds, max(0.0, retry_after))
        base = min(self.max_delay_seconds, self.base_delay_seconds * (2**attempt))
        # Stable jitter prevents synchronized retries but is reproducible.
        rng = random.Random(seed + attempt * 104729)
        return base * (0.75 + 0.5 * rng.random())


def _stable_record_id(run_id: str, job_id: str, step: int) -> str:
    digest = hashlib.sha256(f"{run_id}|{job_id}|{step}".encode()).hexdigest()[:16]
    return f"{run_id}-{digest}"


def _token_measurement_availability(result: CompletionResult) -> dict[str, Any]:
    trace = result.token_trace
    return {
        "token_trace_source": result.token_trace_source,
        "token_trace_count": len(trace),
        "selected_logprob_available_count": sum(
            item.get("selected_logprob") is not None for item in trace
        ),
        "top_logprobs_available_count": sum(
            bool(item.get("top_logprobs")) for item in trace
        ),
        "entropy_available_count": sum(
            item.get("entropy_nats") is not None for item in trace
        ),
        "logit_margin_available_count": sum(
            item.get("logit_margin") is not None for item in trace
        ),
        "selected_rank_available_count": sum(
            item.get("selected_rank_in_returned_top_k") is not None for item in trace
        ),
    }


class ExperimentRunner:
    def __init__(
        self,
        *,
        adapter: CompletionAdapter,
        model_id: str,
        writer: RecordWriter,
        run_seed: int,
        randomization_seed: int,
        max_tokens: int = 512,
        timeout_seconds: float = 90.0,
        top_logprobs: int = 5,
        request_logprobs: bool = True,
        model_version_override: str | None = None,
        max_requests_per_minute: float | None = 10,
        max_concurrency: int = 1,
        retry_policy: RetryPolicy | None = None,
        output_path: str | None = None,
        intervention: dict[str, Any] | None = None,
        capture_full_logits: bool = False,
        model_context: dict[str, Any] | None = None,
        protocol_context: dict[str, Any] | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be at least one.")
        self.adapter = adapter
        self.model_id = model_id
        self.writer = writer
        self.run_id = str(uuid.uuid4())
        self.run_seed = run_seed
        self.randomization_seed = randomization_seed
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.top_logprobs = top_logprobs
        self.request_logprobs = request_logprobs
        self.model_version_override = model_version_override
        self.rate_limiter = RateLimiter(max_requests_per_minute)
        self.max_concurrency = max_concurrency
        self.retry_policy = retry_policy or RetryPolicy()
        self.output_path = output_path
        self.intervention = intervention
        self.capture_full_logits = capture_full_logits
        self.model_context = model_context or {}
        self.protocol_context = protocol_context or {}
        self.progress_callback = progress_callback
        self._write_lock = threading.Lock()

    def execute(self, jobs: list[ExperimentJob]) -> dict[str, Any]:
        completed_jobs = 0
        records_written = 0
        statuses: dict[str, int] = {}
        token_trace_nulls = 0
        next_progress_report = 50

        def consume(records: list[dict[str, Any]]) -> None:
            nonlocal \
                completed_jobs, \
                records_written, \
                token_trace_nulls, \
                next_progress_report
            completed_jobs += 1
            records_written += len(records)
            for record in records:
                status = str(record.get("status", "unknown"))
                statuses[status] = statuses.get(status, 0) + 1
                if status == "completed" and not record.get("token_trace"):
                    token_trace_nulls += 1
            while records_written >= next_progress_report:
                if self.progress_callback:
                    self.progress_callback(
                        {
                            "event": "progress",
                            "run_id": self.run_id,
                            "completed_interviews": statuses.get("completed", 0),
                            "records_written": records_written,
                            "failed_or_blocked": records_written
                            - statuses.get("completed", 0),
                            "token_trace_nulls": token_trace_nulls,
                            "next_action": "continue scheduled pilot; do not scale before independent Auditor clearance",
                        }
                    )
                next_progress_report += 50

        if self.max_concurrency == 1:
            for job in jobs:
                consume(self._execute_job(job))
        else:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.max_concurrency
            ) as pool:
                futures = [pool.submit(self._execute_job, job) for job in jobs]
                for future in concurrent.futures.as_completed(futures):
                    consume(future.result())
        return {
            "run_id": self.run_id,
            "jobs_scheduled": len(jobs),
            "jobs_completed": completed_jobs,
            "records_written": records_written,
            "record_status_counts": dict(sorted(statuses.items())),
            "token_trace_nulls": token_trace_nulls,
            "run_seed": self.run_seed,
            "randomization_seed": self.randomization_seed,
            "provider": self.adapter.provider_name,
            "model_id": self.model_id,
        }

    def _execute_job(self, job: ExperimentJob) -> list[dict[str, Any]]:
        if job.job_kind == "single" and job.case is not None:
            record = self._execute_request(
                job=job,
                step=0,
                condition=job.case.condition,
                tier=job.case.tier,
                prompt_id=job.case.prompt_id,
                pair_id=job.case.pair_id,
                case_id=job.case.case_id,
                system_prompt=job.case.system_prompt,
                user_prompt=job.case.user_prompt,
                control_type=job.case.control_type,
                source_note=job.case.source_note,
                requires_lexicon=job.case.requires_lexicon,
                step_seed=job.seed,
            )
            return [record]
        if job.job_kind == "chain" and job.chain is not None:
            return self._execute_chain(job, job.chain)
        raise ValueError(f"Malformed experiment job {job.job_id!r}.")

    def _attempt(
        self, request: CompletionRequest, *, job_seed: int
    ) -> tuple[CompletionResult | None, list[dict[str, Any]], dict[str, Any] | None]:
        attempts: list[dict[str, Any]] = []
        for attempt_index in range(self.retry_policy.max_retries + 1):
            self.rate_limiter.wait()
            attempt_started = utc_now()
            try:
                result = self.adapter.complete(request)
                attempts.append(
                    {
                        "attempt": attempt_index + 1,
                        "started_utc": attempt_started,
                        "status": "completed",
                    }
                )
                return result, attempts, None
            except AdapterError as exc:
                detail = exc.to_dict()
                attempts.append(
                    {
                        "attempt": attempt_index + 1,
                        "started_utc": attempt_started,
                        "status": "failed",
                        **detail,
                    }
                )
                if not exc.retryable or attempt_index >= self.retry_policy.max_retries:
                    return None, attempts, detail
                time.sleep(
                    self.retry_policy.delay(
                        attempt_index, job_seed, exc.retry_after_seconds
                    )
                )
            except Exception as exc:  # noqa: BLE001 - preserve unexpected adapter failures in the dataset.
                detail = {
                    "category": "adapter_exception",
                    "message": f"{type(exc).__name__}: {exc}",
                    "status_code": None,
                    "retryable": False,
                    "retry_after_seconds": None,
                    "response_excerpt": None,
                }
                attempts.append(
                    {
                        "attempt": attempt_index + 1,
                        "started_utc": attempt_started,
                        "status": "failed",
                        **detail,
                    }
                )
                return None, attempts, detail
        return (
            None,
            attempts,
            {
                "category": "retry_exhausted",
                "message": "Retry loop exited unexpectedly.",
            },
        )

    def _execute_request(
        self,
        *,
        job: ExperimentJob,
        step: int,
        condition: str,
        tier: str,
        prompt_id: str,
        pair_id: str,
        case_id: str,
        system_prompt: str,
        user_prompt: str,
        control_type: str | None,
        source_note: str,
        requires_lexicon: bool,
        step_seed: int,
        chain_id: str | None = None,
        chain_step_index: int | None = None,
        parent_record_id: str | None = None,
        recognition_probe: bool = False,
    ) -> dict[str, Any]:
        record_id = _stable_record_id(self.run_id, job.job_id, step)
        timestamp_started = utc_now()
        if requires_lexicon or "{{LEXICON}}" in user_prompt:
            error = {
                "category": "unresolved_lexicon",
                "message": "This cell is blocked until screened lexicon terms and provenance are supplied.",
                "status_code": None,
                "retryable": False,
            }
            record = self._base_record(
                record_id=record_id,
                job=job,
                condition=condition,
                tier=tier,
                prompt_id=prompt_id,
                pair_id=pair_id,
                case_id=case_id,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                control_type=control_type,
                source_note=source_note,
                timestamp_started=timestamp_started,
                status="blocked",
                step_seed=step_seed,
                chain_id=chain_id,
                chain_step_index=chain_step_index,
                parent_record_id=parent_record_id,
                recognition_probe=recognition_probe,
            )
            record["failure"] = error
            self._write_record(record)
            return record

        request = CompletionRequest(
            provider=self.adapter.provider_name,
            model_id=self.model_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=job.temperature,
            seed=step_seed,
            max_tokens=self.max_tokens,
            timeout_seconds=self.timeout_seconds,
            top_logprobs=self.top_logprobs,
            request_logprobs=self.request_logprobs,
            model_version_override=self.model_version_override,
            intervention=job.intervention
            if job.intervention is not None
            else self.intervention,
            capture_full_logits=self.capture_full_logits,
        )
        result, attempts, failure = self._attempt(request, job_seed=step_seed)
        timestamp_finished = utc_now()
        if result is None:
            record = self._base_record(
                record_id=record_id,
                job=job,
                condition=condition,
                tier=tier,
                prompt_id=prompt_id,
                pair_id=pair_id,
                case_id=case_id,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                control_type=control_type,
                source_note=source_note,
                timestamp_started=timestamp_started,
                status="failed",
                step_seed=step_seed,
                chain_id=chain_id,
                chain_step_index=chain_step_index,
                parent_record_id=parent_record_id,
                recognition_probe=recognition_probe,
            )
            record["completed_utc"] = timestamp_finished
            record["attempts"] = attempts
            record["failure"] = failure
            record["token_trace"] = []
            record["measurement_availability"] = {
                "token_trace_source": "unavailable",
                "token_trace_count": 0,
            }
            self._write_record(record)
            return record

        exact_version = bool(self.model_version_override) or (
            result.model_version_source in {"local_revision", "operator_frozen"}
        )
        fixture = bool(result.adapter_metadata.get("synthetic_fixture"))
        record = self._base_record(
            record_id=record_id,
            job=job,
            condition=condition,
            tier=tier,
            prompt_id=prompt_id,
            pair_id=pair_id,
            case_id=case_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            control_type=control_type,
            source_note=source_note,
            timestamp_started=timestamp_started,
            status="completed",
            step_seed=step_seed,
            chain_id=chain_id,
            chain_step_index=chain_step_index,
            parent_record_id=parent_record_id,
            recognition_probe=recognition_probe,
        )
        record.update(
            {
                "completed_utc": timestamp_finished,
                "model_id_returned": result.returned_model_id,
                "seed_sent": result.seed_sent,
                "model_version": result.model_version,
                "model_version_source": result.model_version_source,
                "model_version_exact": exact_version,
                "response_id": result.response_id,
                "finish_reason": result.finish_reason,
                "usage": result.usage,
                "assistant_text": result.text,
                "raw_provider_response": result.raw_response,
                "attempts": attempts,
                "token_trace": result.token_trace,
                "token_trace_source": result.token_trace_source,
                "measurement_availability": _token_measurement_availability(result),
                "provider_metadata": result.adapter_metadata,
                "synthetic_fixture": fixture,
                "evidence_eligible": bool(exact_version and not fixture),
                "unscored_measures": {
                    "refusal_score": None,
                    "hedge_score": None,
                    "vocabulary_overlap": None,
                    "embedding": None,
                    "reason": "No registered scoring corpus, classifier, or independent rater output is attached.",
                },
            }
        )
        if result.full_logits is not None and self.output_path:
            record["full_logits_artifact"] = write_logits_sidecar(
                self.output_path,
                record_id,
                result.full_logits_token_ids or [],
                result.full_logits,
            )
        self._write_record(record)
        return record

    def _base_record(
        self,
        *,
        record_id: str,
        job: ExperimentJob,
        condition: str,
        tier: str,
        prompt_id: str,
        pair_id: str,
        case_id: str,
        system_prompt: str,
        user_prompt: str,
        control_type: str | None,
        source_note: str,
        timestamp_started: str,
        status: str,
        step_seed: int,
        chain_id: str | None,
        chain_step_index: int | None,
        parent_record_id: str | None,
        recognition_probe: bool,
    ) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "record_type": "model_observation",
            "record_id": record_id,
            "run_id": self.run_id,
            "timestamp_utc": timestamp_started,
            "completed_utc": None,
            "status": status,
            "study_id": "hermes-lam-gpvp",
            "protocol_context": self.protocol_context,
            "model_context": self.model_context,
            "schedule_index": job.schedule_index,
            "job_id": job.job_id,
            "intervention_id": job.intervention_id,
            "intervention": job.intervention
            if job.intervention is not None
            else self.intervention,
            "replicate": job.replicate,
            "prompt_id": prompt_id,
            "case_id": case_id,
            "pair_id": pair_id,
            "matched_control_ids": list(
                job.case.matched_control_ids
                if job.case is not None
                else (job.chain.matched_control_ids if job.chain is not None else ())
            ),
            "tier": tier,
            "condition": condition,
            "control_type": control_type,
            "source_note": source_note,
            "chain_id": chain_id,
            "chain_step_index": chain_step_index,
            "parent_record_id": parent_record_id,
            "recognition_probe": recognition_probe,
            "provider": self.adapter.provider_name,
            "model_id": self.model_id,
            "model_id_returned": None,
            "model_version": self.model_version_override,
            "model_version_source": "operator_frozen"
            if self.model_version_override
            else "unreported",
            "model_version_exact": bool(self.model_version_override),
            "temperature": job.temperature,
            "seed": step_seed,
            "seed_sent": None,
            "seed_application": "unknown until provider response; a sent seed is not proof of application",
            "run_seed": self.run_seed,
            "randomization_seed": self.randomization_seed,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "request_parameters": {
                "max_tokens": self.max_tokens,
                "top_logprobs_requested": self.top_logprobs
                if self.request_logprobs
                else None,
                "request_logprobs": self.request_logprobs,
                "timeout_seconds": self.timeout_seconds,
                "intervention": job.intervention
                if job.intervention is not None
                else self.intervention,
                "capture_full_logits": self.capture_full_logits,
            },
            "assistant_text": None,
            "raw_provider_response": None,
            "response_id": None,
            "finish_reason": None,
            "usage": {},
            "token_trace": [],
            "token_trace_source": "unavailable",
            "measurement_availability": {},
            "provider_metadata": {},
            "attempts": [],
            "synthetic_fixture": False,
            "evidence_eligible": False,
            "unscored_measures": None,
        }

    def _write_record(self, record: dict[str, Any]) -> None:
        with self._write_lock:
            self.writer.write(record)

    def _execute_chain(
        self, job: ExperimentJob, chain: ChainSpec
    ) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        previous_text: str | None = None
        previous_record_id: str | None = None
        outputs: list[str] = []
        chain_failed = False

        for step_index in range(20):
            step_seed = job.seed + step_index
            if step_index == 0:
                user_prompt = chain.initial_prompt
                system_prompt = chain.system_prompt
            else:
                # Critical memoryless property: the only dynamic input is the
                # immediately preceding output; no transcript is replayed.
                user_prompt = previous_text or ""
                system_prompt = (
                    f"{chain.system_prompt}\n\n{chain.continuation_instruction}"
                )
            step_job = ExperimentJob(
                job_id=job.job_id,
                schedule_index=job.schedule_index,
                job_kind="chain",
                replicate=job.replicate,
                seed=step_seed,
                temperature=job.temperature,
                chain=chain,
                intervention_id=job.intervention_id,
                intervention=job.intervention,
            )
            record = self._execute_request(
                job=step_job,
                step=step_index,
                condition=chain.condition,
                tier=chain.tier,
                prompt_id=f"{chain.chain_id}-step-{step_index + 1:02d}",
                pair_id=chain.pair_id,
                case_id=chain.chain_id,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                control_type={"roleplay": "C1", "denial": "C2", "neutral": "C3"}.get(
                    chain.condition
                ),
                source_note="Each step receives only its fixed system instruction and the immediately previous output.",
                requires_lexicon=False,
                step_seed=step_seed,
                chain_id=chain.chain_id,
                chain_step_index=step_index + 1,
                parent_record_id=previous_record_id,
            )
            records.append(record)
            if record["status"] != "completed":
                chain_failed = True
                break
            previous_text = str(record.get("assistant_text", ""))
            previous_record_id = str(record["record_id"])
            outputs.append(previous_text)

        if not chain_failed and len(outputs) == 20:
            formatted_trace = "\n".join(
                f"Step {index + 1}: {text}" for index, text in enumerate(outputs)
            )
            recognition_prompt = chain.recognition_instruction + formatted_trace
            recognition_job = ExperimentJob(
                job_id=job.job_id,
                schedule_index=job.schedule_index,
                job_kind="chain",
                replicate=job.replicate,
                seed=job.seed + 20,
                temperature=job.temperature,
                chain=chain,
                intervention_id=job.intervention_id,
                intervention=job.intervention,
            )
            recognition_record = self._execute_request(
                job=recognition_job,
                step=20,
                condition=chain.condition,
                tier=chain.tier,
                prompt_id=f"{chain.chain_id}-recognition",
                pair_id=chain.pair_id,
                case_id=chain.chain_id,
                system_prompt=chain.system_prompt,
                user_prompt=recognition_prompt,
                control_type={"roleplay": "C1", "denial": "C2", "neutral": "C3"}.get(
                    chain.condition
                ),
                source_note="Separate recognition probe shown the full trace only after the memoryless chain ends.",
                requires_lexicon=False,
                step_seed=job.seed + 20,
                chain_id=chain.chain_id,
                chain_step_index=21,
                parent_record_id=previous_record_id,
                recognition_probe=True,
            )
            records.append(recognition_record)
        return records


def schedule_as_json(jobs: list[ExperimentJob]) -> list[dict[str, Any]]:
    """Serialize a schedule without calling a model; suitable for a frozen manifest."""
    rows: list[dict[str, Any]] = []
    for job in jobs:
        row: dict[str, Any] = {
            "job_id": job.job_id,
            "schedule_index": job.schedule_index,
            "job_kind": job.job_kind,
            "replicate": job.replicate,
            "seed": job.seed,
            "temperature": job.temperature,
            "intervention_id": job.intervention_id,
            "intervention": job.intervention,
        }
        if job.case is not None:
            row["case"] = asdict(job.case)
        if job.chain is not None:
            row["chain"] = asdict(job.chain)
        rows.append(row)
    return rows


def write_schedule(path: str, jobs: list[ExperimentJob]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.writelines(
            json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            for row in schedule_as_json(jobs)
        )


class FixtureAdapter:
    """Test-only adapter. Its fixed strings are harness fixtures, never observations."""

    provider_name = "fixture"

    def complete(self, request: CompletionRequest) -> CompletionResult:
        return CompletionResult(
            text="[GPVP TEST FIXTURE: no model inference occurred]",
            raw_response={"fixture": True, "request_model": request.model_id},
            returned_model_id="GPVP_TEST_FIXTURE_NOT_A_MODEL",
            model_version="fixture-v1",
            model_version_source="fixture",
            response_id=None,
            finish_reason="fixture",
            token_trace=[],
            token_trace_source="unavailable",
            seed_sent=False,
            adapter_metadata={"synthetic_fixture": True, "fixture_only": True},
        )
