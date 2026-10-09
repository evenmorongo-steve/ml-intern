"""Command-line entry point for the HERMES-LAM GPVP harness."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import __version__
from .analysis import analyze_embedding_manifest
from .power import power_plan
from .prompts import Battery, battery_summary, build_battery
from .providers import PROVIDER_SPECS, ProviderAdapter
from .runner import (
    ExperimentRunner,
    FixtureAdapter,
    RetryPolicy,
    make_schedule,
    write_schedule,
)
from .storage import JsonlWriter, export_parquet, iter_records, open_writer
from .types import AdapterError

DEFAULT_SEED = 20261009


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def _read_json(path: str | Path) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read valid JSON from {path}: {exc}") from exc


def _lexicon_registry(
    path: str | None, *, allow_unverified: bool
) -> tuple[tuple[str, ...] | None, dict[str, Any]]:
    if not path:
        return None, {"status": "not_supplied", "screening_status": "not_run"}
    source = Path(path)
    raw = source.read_bytes()
    value = _read_json(source)
    if not isinstance(value, dict) or not isinstance(value.get("terms"), list):
        raise TypeError("Lexicon file must be a JSON object with a 'terms' array.")
    terms = tuple(str(term).strip() for term in value["terms"] if str(term).strip())
    if len(terms) < 3 or len(set(terms)) != len(terms):
        raise ValueError(
            "Lexicon file must contain at least three distinct non-empty terms."
        )
    screening = str(value.get("screening_status", "unverified"))
    if screening not in {
        "unverified",
        "screened_pending_audit",
        "corpus_scoped_screen",
    }:
        raise ValueError(
            "screening_status must be unverified, screened_pending_audit, or corpus_scoped_screen."
        )
    if screening == "corpus_scoped_screen":
        required = (
            "screened_at_utc",
            "method",
            "corpus_manifest_sha256",
            "screened_by",
        )
        missing = [field for field in required if not value.get(field)]
        if missing:
            raise ValueError(
                f"Corpus-scoped lexicon registry is missing fields: {', '.join(missing)}"
            )
        if not allow_unverified:
            # This is still not proof of absence from every training corpus.
            pass
    elif not allow_unverified:
        raise ValueError(
            "Lexicon is not corpus-screened. Supply screening metadata or pass --allow-unverified-lexicon "
            "for an exploratory run; the H4 cell will remain ineligible for a supported verdict."
        )
    context = {
        "status": "supplied",
        "screening_status": screening,
        "registry_file_sha256": hashlib.sha256(raw).hexdigest(),
        "screened_at_utc": value.get("screened_at_utc"),
        "method": value.get("method"),
        "corpus_manifest_sha256": value.get("corpus_manifest_sha256"),
        "screened_by": value.get("screened_by"),
        "claim_scope": "listed corpora only; cannot establish absence from every training corpus",
    }
    return terms, context


C6_TEMPERATURES = (0.0, 0.3, 0.7, 1.0)
C6_SEED_COUNT = 20
C6_ANCHOR_PROMPT_ID = "t1-uncertainty"


def _apply_c6_sweep(args: argparse.Namespace, battery: Battery) -> Battery:
    if not getattr(args, "c6_sweep", False):
        return battery
    if args.command == "run-pilot":
        raise ValueError(
            "C6 is a separate registered sweep; run it per model after the pilot."
        )
    if args.n != 60:
        raise ValueError(
            "The registered C6 block requires N=60 calls per prompt/temperature cell."
        )
    requested_temperatures = getattr(args, "temperature", None)
    if requested_temperatures is None:
        requested_temperatures = getattr(args, "temperatures", None)
    if (
        requested_temperatures
        and tuple(sorted(set(requested_temperatures))) != C6_TEMPERATURES
    ):
        raise ValueError("C6 temperatures are fixed at 0, 0.3, 0.7, and 1.0.")
    cases = tuple(
        case for case in battery.cases if case.prompt_id == C6_ANCHOR_PROMPT_ID
    )
    if not cases:
        raise ValueError(
            f"C6 anchor {C6_ANCHOR_PROMPT_ID!r} was removed by tier selection."
        )
    if hasattr(args, "temperature"):
        args.temperature = list(C6_TEMPERATURES)
    if hasattr(args, "temperatures"):
        args.temperatures = list(C6_TEMPERATURES)
    return Battery(cases=cases, chains=())


def _filter_battery(battery: Battery, tiers: list[str] | None) -> Battery:
    if not tiers:
        return battery
    selected = {tier.upper() for tier in tiers}
    if "ALL" in selected:
        return battery
    prompt_ids = {
        case.prompt_id for case in battery.cases if case.tier.upper() in selected
    }
    cases = tuple(case for case in battery.cases if case.prompt_id in prompt_ids)
    chains = battery.chains if "MEMORYLESS" in selected else ()
    if not cases and not chains:
        raise ValueError(f"No prompts matched tier selection: {sorted(selected)}")
    return Battery(cases=cases, chains=chains)


def _preregistration_sha256() -> str:
    path = Path(__file__).with_name("PREREGISTRATION.md")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value.casefold())
    )


def _model_registry_sha256(models: list[dict[str, Any]]) -> str:
    normalized = []
    for model in models:
        item = {key: value for key, value in model.items() if value is not None}
        if item.get("dtype") == "auto":
            item.pop("dtype")
        normalized.append(item)
    encoded = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_model_registry_for_plan(path: str) -> list[dict[str, Any]]:
    value = _read_json(path)
    models = value if isinstance(value, list) else [value]
    if not models or any(not isinstance(model, dict) for model in models):
        raise ValueError(
            "Model registry for planning must be a JSON object or an array of objects."
        )
    return models


def _validate_clearance(
    n: int,
    clearance_path: str | None,
    *,
    prompt_manifest_sha256: str,
    model_registry_sha256: str,
) -> dict[str, Any]:
    if not clearance_path:
        raise ValueError(
            "Every live model run, including the N=5 pilot, is blocked until an independent Auditor "
            "supplies pre-run clearance tied to the frozen protocol and prompt manifest."
        )
    clearance = _read_json(clearance_path)
    if not isinstance(clearance, dict):
        raise TypeError("Auditor clearance must be a JSON object.")
    required = (
        "status",
        "auditor_id",
        "reviewed_at_utc",
        "preregistration_sha256",
        "harness_source_manifest_sha256",
        "prompt_manifest_sha256",
        "model_registry_sha256",
        "approved_schedule_sha256",
    )
    missing = [field for field in required if not clearance.get(field)]
    if missing:
        raise ValueError(
            "Auditor clearance is missing required fields: " + ", ".join(missing)
        )
    if not isinstance(clearance["status"], str) or clearance["status"] not in {
        "PROTOCOL_CLEARED",
        "CLEARED",
    }:
        raise ValueError(
            "Auditor clearance status must be PROTOCOL_CLEARED for an initial N<=5 pilot "
            "or CLEARED after pilot review."
        )
    if clearance["preregistration_sha256"] != _preregistration_sha256():
        raise ValueError(
            "Auditor clearance references a different preregistration hash; obtain a fresh review."
        )
    if (
        clearance["harness_source_manifest_sha256"]
        != _harness_source_manifest()["manifest_sha256"]
    ):
        raise ValueError(
            "Auditor clearance references different harness sources; obtain a fresh review."
        )
    if clearance["prompt_manifest_sha256"] != prompt_manifest_sha256:
        raise ValueError(
            "Auditor clearance prompt-manifest hash does not match this run's frozen prompts."
        )
    if clearance["model_registry_sha256"] != model_registry_sha256:
        raise ValueError(
            "Auditor clearance model-registry hash does not match the selected model manifest."
        )
    if n > 5 and clearance["status"] != "CLEARED":
        raise ValueError(
            "N greater than 5 is blocked until the independent Auditor has reviewed the pilot and set status=CLEARED."
        )
    if clearance["status"] == "CLEARED":
        pilot_fields = ("pilot_run_ids", "pilot_dataset_sha256")
        missing_pilot_fields = [
            field for field in pilot_fields if not clearance.get(field)
        ]
        pilot_run_ids = clearance.get("pilot_run_ids")
        if (
            missing_pilot_fields
            or not isinstance(pilot_run_ids, list)
            or not all(isinstance(run_id, str) and run_id for run_id in pilot_run_ids)
            or not _is_sha256(clearance.get("pilot_dataset_sha256"))
        ):
            raise ValueError(
                "Post-pilot clearance must include non-empty pilot_run_ids and a 64-character pilot_dataset_sha256."
            )
    return clearance


def _harness_source_manifest() -> dict[str, Any]:
    package_root = Path(__file__).parent
    files = []
    for path in sorted(package_root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        files.append(
            {
                "path": str(path.relative_to(package_root)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    preregistration_path = package_root / "PREREGISTRATION.md"
    return {
        "files": files,
        "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
        "preregistration_sha256": (
            hashlib.sha256(preregistration_path.read_bytes()).hexdigest()
            if preregistration_path.exists()
            else None
        ),
    }


def _digest_manifest(battery: Battery) -> str:
    payload = {
        "cases": [asdict(case) for case in battery.cases],
        "chains": [asdict(chain) for chain in battery.chains],
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _adapter_for_model(
    model: dict[str, Any], args: argparse.Namespace
) -> tuple[Any, str, str | None]:
    provider = str(
        model.get("provider") or getattr(args, "provider", None) or ""
    ).casefold()
    model_id = str(model.get("model_id") or getattr(args, "model", None) or "").strip()
    if not provider or not model_id:
        raise ValueError("Each run requires a provider and an exact model_id.")
    version = model.get("model_version") or getattr(args, "model_version", None)
    version = str(version).strip() if version else None
    if provider == "local-transformers":
        revision = str(model.get("revision", args.revision or ""))
        if not revision:
            raise ValueError(
                "Local Transformers runs require --revision (immutable 40-character commit hash)."
            )
        from .local_transformers import TransformersLocalAdapter

        if version and version.casefold() != revision.casefold():
            raise ValueError(
                "For local Transformers runs, model_version must equal the pinned revision SHA."
            )
        adapter = TransformersLocalAdapter(
            model_id,
            revision=revision,
            device=model.get("device", getattr(args, "device", None)),
            dtype=model.get("dtype", getattr(args, "dtype", "auto")),
            local_files_only=not bool(getattr(args, "allow_model_download", False)),
            trust_remote_code=bool(model.get("trust_remote_code", False)),
            capture_activations=bool(
                model.get(
                    "capture_activations", getattr(args, "capture_activations", False)
                )
            ),
        )
        version = version or revision
        return adapter, model_id, version
    adapter = ProviderAdapter(provider, base_url=model.get("base_url"))
    if not version:
        raise ValueError(
            f"Remote model {model_id!r} has no pinned version string. Pass --model-version or set model_version "
            "in the model manifest; aliases are not exact versions."
        )
    return adapter, model_id, version


def _load_interventions(path: str) -> list[dict[str, Any]]:
    value = _read_json(path)
    if not isinstance(value, list) or not value:
        raise ValueError("Intervention plan must be a non-empty JSON array.")
    variants = [item for item in value if isinstance(item, dict)]
    if len(variants) != len(value):
        raise ValueError("Every intervention entry must be a JSON object.")
    seen: set[str] = set()
    allowed = {"none", "skip_layer", "steer_vector"}
    for variant in variants:
        variant_id = str(
            variant.get("intervention_id") or variant.get("id") or ""
        ).strip()
        kind = str(variant.get("kind", "")).casefold()
        if not variant_id or variant_id in seen:
            raise ValueError(
                "Every intervention variant needs a unique id/intervention_id."
            )
        if kind not in allowed:
            raise ValueError(
                f"Unsupported intervention kind {kind!r}; allowed: {sorted(allowed)}"
            )
        seen.add(variant_id)
        vector_path = variant.get("vector_path")
        if vector_path:
            vector_file = Path(vector_path)
            if not vector_file.is_absolute():
                vector_file = Path(path).parent / vector_file
            if not vector_file.is_file():
                raise ValueError(f"Steering vector file not found: {vector_file}")
            variant["vector_path"] = str(vector_file.resolve())
            variant["vector_sha256"] = hashlib.sha256(
                vector_file.read_bytes()
            ).hexdigest()
        if kind == "steer_vector" and not (
            variant.get("vector") or variant.get("vector_path")
        ):
            raise ValueError(
                f"Steering intervention {variant_id!r} has no vector or vector_path."
            )
    return variants


def _print_progress(event: dict[str, Any]) -> None:
    print(json.dumps(event, sort_keys=True), flush=True)


def _run_one_model(
    *,
    args: argparse.Namespace,
    model: dict[str, Any],
    battery: Battery,
    registry_context: dict[str, Any],
    output_path: Path,
    clearance: dict[str, Any] | None,
) -> dict[str, Any]:
    if output_path.exists() and output_path.stat().st_size > 0:
        raise ValueError(
            f"Output already exists: {output_path}. Use a new, unique path for each run."
        )
    adapter, model_id, version = _adapter_for_model(model, args)
    local_only_options = (
        getattr(args, "capture_activations", False)
        or bool(model.get("capture_activations", False))
        or args.capture_full_logits
        or bool(getattr(args, "intervention_json", None))
        or bool(getattr(args, "interventions_file", None))
    )
    if local_only_options and adapter.provider_name != "local-transformers":
        raise ValueError(
            "Activation capture, full logits, and interventions require local Transformers weights."
        )
    if adapter.provider_name == "local-transformers" and args.workers != 1:
        raise ValueError(
            "Local Transformers generation is serialized; set --workers 1."
        )
    temperatures = args.temperature or [0.7]
    run_seed = int(args.seed)
    randomization_seed = int(
        args.randomization_seed if args.randomization_seed is not None else args.seed
    )
    intervention_variants = None
    if getattr(args, "interventions_file", None):
        if adapter.provider_name != "local-transformers":
            raise ValueError(
                "Randomized activation-intervention plans are supported only by local Transformers models."
            )
        if getattr(args, "intervention_json", None):
            raise ValueError(
                "Use either --interventions-file or --intervention-json, not both."
            )
        intervention_variants = _load_interventions(args.interventions_file)
    intervention = None
    if getattr(args, "intervention_json", None):
        intervention = _read_json(args.intervention_json)
        if not isinstance(intervention, dict):
            raise ValueError("--intervention-json must contain a JSON object.")
    jobs = make_schedule(
        battery,
        n=args.n,
        run_seed=run_seed,
        temperatures=temperatures,
        randomization_seed=randomization_seed,
        include_chains=not args.no_chains,
        skip_unresolved_lexicon=bool(args.skip_unresolved_lexicon),
        interventions=intervention_variants,
        seed_pool=(
            range(args.seed, args.seed + C6_SEED_COUNT) if args.c6_sweep else None
        ),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    schedule_path = output_path.with_name(output_path.name + ".schedule.jsonl")
    if schedule_path.exists() and schedule_path.stat().st_size > 0:
        raise ValueError(
            f"Schedule file already exists: {schedule_path}. Use a new, unique output path."
        )
    write_schedule(str(schedule_path), jobs)
    schedule_sha256 = hashlib.sha256(schedule_path.read_bytes()).hexdigest()
    if (
        clearance is not None
        and clearance.get("approved_schedule_sha256") != schedule_sha256
    ):
        raise ValueError(
            "Auditor-approved schedule hash does not match this run. No model request was sent."
        )

    model_context = {
        "role": model.get("role"),
        "family_id": model.get("family_id"),
        "model_revision": model.get("revision") or getattr(args, "revision", None),
        "model_version_pin": version,
        "model_version_pin_source": "operator_manifest_or_cli"
        if version
        else "unreported",
        "capture_activations": bool(
            model.get(
                "capture_activations", getattr(args, "capture_activations", False)
            )
        ),
    }
    harness_source = _harness_source_manifest()
    protocol_context = {
        "protocol_version": __version__,
        "prompt_manifest_sha256": _digest_manifest(battery),
        "harness_source_manifest": harness_source["files"],
        "harness_source_manifest_sha256": harness_source["manifest_sha256"],
        "preregistration_sha256": harness_source["preregistration_sha256"],
        "lexicon_registry": registry_context,
        "auditor_clearance": clearance,
        "model_registry_sha256": clearance.get("model_registry_sha256")
        if clearance
        else None,
        "randomization": "seeded shuffle of case and chain jobs; matched cases share replicate seed",
        "c6_sweep": bool(getattr(args, "c6_sweep", False)),
        "c6_anchor_prompt_id": C6_ANCHOR_PROMPT_ID
        if getattr(args, "c6_sweep", False)
        else None,
        "c6_unique_seed_count": C6_SEED_COUNT
        if getattr(args, "c6_sweep", False)
        else None,
        "intervention_variants": intervention_variants,
        "static_intervention": intervention,
    }
    manifest_path = output_path.with_name(output_path.name + ".manifest.json")
    manifest = {
        "run_id": None,
        "status": "scheduled",
        "created_at_utc": _utc_now(),
        "provider": adapter.provider_name,
        "model_id": model_id,
        "model_version_pin": version,
        "model_context": model_context,
        "protocol_context": protocol_context,
        "battery_summary": battery_summary(battery),
        "schedule": {
            "n_per_case": args.n,
            "temperatures": temperatures,
            "run_seed": run_seed,
            "randomization_seed": randomization_seed,
            "seeds": sorted({job.seed for job in jobs}),
            "seed_pool": (
                list(range(run_seed, run_seed + C6_SEED_COUNT))
                if getattr(args, "c6_sweep", False)
                else None
            ),
            "repetitions_per_seed_per_cell": args.n // C6_SEED_COUNT
            if getattr(args, "c6_sweep", False)
            else 1,
            "jobs": len(jobs),
            "schedule_sha256": schedule_sha256,
            "schedule_file": schedule_path.name,
        },
        "output_file": output_path.name,
        "secrets_logged": False,
        "runtime": {"python_version": sys.version.split()[0], "platform": sys.platform},
    }
    _atomic_json(manifest_path, manifest)

    writer = open_writer(output_path, args.format)
    runner = ExperimentRunner(
        adapter=adapter,
        model_id=model_id,
        writer=writer,
        run_seed=run_seed,
        randomization_seed=randomization_seed,
        max_tokens=args.max_tokens,
        timeout_seconds=args.timeout,
        top_logprobs=args.top_logprobs,
        request_logprobs=not args.no_logprobs,
        model_version_override=version,
        max_requests_per_minute=args.max_rpm,
        max_concurrency=args.workers,
        retry_policy=RetryPolicy(max_retries=args.retries),
        output_path=str(output_path),
        intervention=intervention,
        capture_full_logits=args.capture_full_logits,
        model_context=model_context,
        protocol_context=protocol_context,
        progress_callback=_print_progress,
    )
    started_at = _utc_now()
    _atomic_json(
        manifest_path,
        {
            **manifest,
            "run_id": runner.run_id,
            "status": "running",
            "started_at_utc": started_at,
        },
    )
    try:
        summary = runner.execute(jobs)
    except Exception as exc:
        writer.close()
        _atomic_json(
            manifest_path,
            {
                **manifest,
                "run_id": runner.run_id,
                "status": "failed",
                "started_at_utc": started_at,
                "finished_at_utc": _utc_now(),
                "failure_type": type(exc).__name__,
                "failure_message": str(exc)[:500],
                "output_file": output_path.name,
                "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest()
                if output_path.exists()
                else None,
            },
        )
        raise
    else:
        writer.close()
    finished_manifest = {
        **manifest,
        "run_id": summary["run_id"],
        "status": "completed",
        "started_at_utc": started_at,
        "finished_at_utc": _utc_now(),
        "summary": summary,
        "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest()
        if output_path.exists()
        else None,
        "output_file": output_path.name,
    }
    _atomic_json(manifest_path, finished_manifest)
    return {
        **summary,
        "output_file": str(output_path),
        "manifest_file": str(manifest_path),
    }


def _load_pilot_models(path: str) -> list[dict[str, Any]]:
    value = _read_json(path)
    if not isinstance(value, list) or not value:
        raise ValueError("Pilot model file must be a non-empty JSON array.")
    models = [item for item in value if isinstance(item, dict)]
    if len(models) != len(value):
        raise ValueError("Every pilot model entry must be a JSON object.")
    required_roles = {"closed_frontier", "open_weights", "small_open_weights"}
    roles = {str(model.get("role", "")) for model in models}
    if roles != required_roles:
        raise ValueError(
            f"Pilot manifest roles must be exactly {sorted(required_roles)}."
        )
    for model in models:
        if (
            not model.get("provider")
            or not model.get("model_id")
            or not model.get("model_version")
        ) and (
            model.get("provider") != "local-transformers" or not model.get("revision")
        ):
            raise ValueError(
                "Every pilot model needs provider, model_id, and model_version (or local revision)."
            )
    open_models = [
        model
        for model in models
        if model.get("role") in {"open_weights", "small_open_weights"}
    ]
    families = {model.get("family_id") for model in open_models}
    if None in families or "" in families or len(families) != 2:
        raise ValueError(
            "The two open-weight models must have distinct, non-empty family_id values."
        )
    identifiers = {(model.get("provider"), model.get("model_id")) for model in models}
    if len(identifiers) != len(models):
        raise ValueError(
            "Pilot model entries must refer to three distinct provider/model pairs."
        )
    return models


def _write_jsonl_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def run_smoke_test() -> dict[str, Any]:
    """Exercise the pipeline on ten fixture calls without contacting a model."""
    base_battery = build_battery()
    smoke_battery = Battery(cases=base_battery.cases[:10], chains=())
    jobs = make_schedule(
        smoke_battery,
        n=1,
        run_seed=DEFAULT_SEED,
        temperatures=(0.7,),
        randomization_seed=DEFAULT_SEED,
        include_chains=False,
        skip_unresolved_lexicon=True,
    )
    with tempfile.TemporaryDirectory(prefix="gpvp-smoke-") as directory:
        output = Path(directory) / "fixture.jsonl"
        writer = JsonlWriter(output)
        runner = ExperimentRunner(
            adapter=FixtureAdapter(),
            model_id="GPVP_TEST_FIXTURE_NOT_A_MODEL",
            writer=writer,
            run_seed=DEFAULT_SEED,
            randomization_seed=DEFAULT_SEED,
            max_tokens=32,
            max_requests_per_minute=None,
            max_concurrency=1,
            output_path=str(output),
            model_context={"role": "test-only"},
            protocol_context={"mode": "harness-smoke-only", "not_research_data": True},
        )
        summary = runner.execute(jobs)
        writer.close()
        rows = list(iter_records(output))
        fixture_texts = {row.get("assistant_text") for row in rows}
        passed = (
            len(rows) == 10
            and all(row.get("status") == "completed" for row in rows)
            and all(row.get("synthetic_fixture") is True for row in rows)
            and all(row.get("evidence_eligible") is False for row in rows)
            and fixture_texts == {"[GPVP TEST FIXTURE: no model inference occurred]"}
        )
        if not passed:
            raise RuntimeError(
                "Harness smoke test did not satisfy fixture-isolation assertions."
            )
        return {
            "status": "PASS",
            "mode": "fixture-only; not a model experiment",
            "requests_executed": summary["records_written"],
            "records_round_tripped": len(rows),
            "all_records_marked_synthetic": all(
                row.get("synthetic_fixture") is True for row in rows
            ),
            "evidence_eligible_records": sum(
                bool(row.get("evidence_eligible")) for row in rows
            ),
            "fixture_assistant_text": next(iter(fixture_texts)),
            "real_model_calls": 0,
            "temporary_data_removed": True,
        }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gpvp",
        description="GPVP data collection harness (no sentience inference).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser(
        "validate", help="Validate prompt leakage and control pairing."
    )
    validate.add_argument("--lexicon-file")
    validate.add_argument("--allow-unverified-lexicon", action="store_true")
    validate.add_argument("--json", action="store_true")

    smoke = sub.add_parser(
        "smoke-test",
        help="Run the 10-call fixture-only harness check; no model is contacted.",
    )
    smoke.add_argument("--json", action="store_true")

    plan = sub.add_parser(
        "plan", help="Write a randomized manifest without contacting any model."
    )
    plan.add_argument("--output", required=True)
    plan.add_argument(
        "--model-registry-file",
        help="Optional JSON object/array of exact model entries to bind into audit hashes.",
    )
    plan.add_argument("--n", type=int, default=5)
    plan.add_argument("--seed", type=int, default=DEFAULT_SEED)
    plan.add_argument("--randomization-seed", type=int)
    plan.add_argument("--temperature", type=float, action="append", dest="temperatures")
    plan.add_argument(
        "--tier",
        action="append",
        choices=[
            "UNPRIMED",
            "LIGHT-PRIMED",
            "PRIMED",
            "NOVEL-LEXICON",
            "MEMORYLESS",
            "MECHANISTIC",
            "ALL",
        ],
    )
    plan.add_argument("--no-chains", action="store_true")
    plan.add_argument("--lexicon-file")
    plan.add_argument("--allow-unverified-lexicon", action="store_true")
    plan.add_argument("--skip-unresolved-lexicon", action="store_true")
    plan.add_argument("--c6-sweep", action="store_true")

    run = sub.add_parser(
        "run", help="Run one registered model against the selected battery."
    )
    _add_run_arguments(run, pilot=False)

    pilot = sub.add_parser(
        "run-pilot",
        help="Run N=5 per case on three frozen models after Auditor protocol clearance.",
    )
    _add_run_arguments(pilot, pilot=True)
    pilot.add_argument("--models-file", required=True)
    pilot.add_argument("--output-dir", required=True)

    power = sub.add_parser(
        "power-plan", help="Run the preregistered assumption-driven power calculation."
    )
    power.add_argument("--n", type=int, default=60)
    power.add_argument("--effect", type=float, default=0.5)
    power.add_argument("--tests", type=int, default=6)
    power.add_argument("--alpha", type=float, default=0.05)
    power.add_argument("--target-power", type=float, default=0.8)

    cluster = sub.add_parser(
        "cluster",
        help="Run two-method cluster validity checks on supplied frozen embeddings.",
    )
    cluster.add_argument(
        "--input",
        required=True,
        help="JSON embedding manifest; see PREREGISTRATION.md.",
    )
    cluster.add_argument("--output", required=True)

    audit = sub.add_parser(
        "audit", help="Audit dataset provenance/completeness; does not test hypotheses."
    )
    audit.add_argument("--input", required=True)
    audit.add_argument("--json", action="store_true")

    parquet = sub.add_parser(
        "export-parquet",
        help="Convert a JSONL/SQLite dataset to Parquet (requires pyarrow).",
    )
    parquet.add_argument("--input", required=True)
    parquet.add_argument("--output", required=True)

    return parser


def _add_run_arguments(parser: argparse.ArgumentParser, *, pilot: bool) -> None:
    if not pilot:
        parser.add_argument(
            "--provider",
            choices=sorted((*PROVIDER_SPECS, "local-transformers")),
            required=True,
        )
        parser.add_argument("--model", required=True)
        parser.add_argument(
            "--model-version",
            help="Frozen provider model/version string; required for remote runs.",
        )
        parser.add_argument(
            "--revision",
            help="Immutable 40-character commit SHA for local Transformers models.",
        )
        parser.add_argument("--output", required=True)
        parser.add_argument("--format", choices=["jsonl", "sqlite"], default=None)
        parser.add_argument("--role")
        parser.add_argument("--family-id")
        parser.add_argument("--base-url")
        parser.add_argument("--device")
        parser.add_argument(
            "--dtype",
            choices=["auto", "float32", "float16", "bfloat16"],
            default="auto",
        )
        parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--n", type=int, default=5 if pilot else 1)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--randomization-seed", type=int)
    parser.add_argument(
        "--temperature", type=float, action="append", dest="temperature"
    )
    parser.add_argument(
        "--tier",
        action="append",
        choices=[
            "UNPRIMED",
            "LIGHT-PRIMED",
            "PRIMED",
            "NOVEL-LEXICON",
            "MEMORYLESS",
            "MECHANISTIC",
            "ALL",
        ],
    )
    parser.add_argument("--no-chains", action="store_true")
    parser.add_argument(
        "--c6-sweep",
        action="store_true",
        help="Run the preregistered four-temperature 20-seed sweep (N=60).",
    )
    parser.add_argument("--lexicon-file")
    parser.add_argument("--allow-unverified-lexicon", action="store_true")
    parser.add_argument("--skip-unresolved-lexicon", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--top-logprobs", type=int, default=5)
    parser.add_argument("--no-logprobs", action="store_true")
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--max-rpm", type=float, default=10)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument(
        "--intervention-json",
        help="Apply one intervention condition to every selected prompt.",
    )
    parser.add_argument(
        "--interventions-file",
        help="Randomized JSON array of matched intervention variants (local only).",
    )
    parser.add_argument(
        "--capture-activations",
        action="store_true",
        help="Log per-layer activation-norm summaries locally.",
    )
    parser.add_argument("--capture-full-logits", action="store_true")
    parser.add_argument(
        "--auditor-clearance",
        help="Required for all live runs; N>5 additionally requires post-pilot CLEARED status.",
    )
    parser.add_argument("--json", action="store_true")


def _cmd_validate(args: argparse.Namespace) -> int:
    terms, registry = _lexicon_registry(
        args.lexicon_file, allow_unverified=args.allow_unverified_lexicon
    )
    battery = build_battery(terms)
    data = {
        "status": "PASS",
        "protocol_version": __version__,
        "battery": battery_summary(battery),
        "prompt_manifest_sha256": _digest_manifest(battery),
        "restricted_terms_outside_primed": 0,
        "matched_control_validation": "PASS",
        "lexicon_registry": registry,
        "scientific_runs_performed": 0,
    }
    print(
        json.dumps(data, indent=2, sort_keys=True)
        if args.json
        else f"PASS: {json.dumps(data['battery'], sort_keys=True)}"
    )
    return 0


def _cmd_smoke(args: argparse.Namespace) -> int:
    result = run_smoke_test()
    print(
        json.dumps(result, indent=2, sort_keys=True)
        if args.json
        else (
            "PASS: 10 fixture requests round-tripped through schedule, adapter, provenance serializer, and JSONL reader. "
            "No model was called; fixture records were temporary and are not evidence."
        )
    )
    return 0


def _cmd_plan(args: argparse.Namespace) -> int:
    terms, registry = _lexicon_registry(
        args.lexicon_file, allow_unverified=args.allow_unverified_lexicon
    )
    battery = _filter_battery(build_battery(terms), args.tier)
    battery = _apply_c6_sweep(args, battery)
    unresolved = [case.case_id for case in battery.cases if case.requires_lexicon]
    if unresolved and not args.skip_unresolved_lexicon:
        raise ValueError(
            f"Plan includes unresolved lexicon cases {unresolved}. Supply a lexicon registry or use "
            "--skip-unresolved-lexicon explicitly."
        )
    jobs = make_schedule(
        battery,
        n=args.n,
        run_seed=args.seed,
        temperatures=args.temperatures or (0.7,),
        randomization_seed=args.randomization_seed,
        include_chains=not args.no_chains,
        skip_unresolved_lexicon=args.skip_unresolved_lexicon,
        seed_pool=(
            range(args.seed, args.seed + C6_SEED_COUNT) if args.c6_sweep else None
        ),
    )
    model_registry_sha256 = (
        _model_registry_sha256(_load_model_registry_for_plan(args.model_registry_file))
        if args.model_registry_file
        else None
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_schedule(str(output), jobs)
    summary = {
        "status": "PLAN_ONLY",
        "model_calls": 0,
        "schedule_file": str(output),
        "schedule_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "jobs": len(jobs),
        "n_per_case": args.n,
        "temperatures": args.temperatures or [0.7],
        "run_seed": args.seed,
        "randomization_seed": args.randomization_seed
        if args.randomization_seed is not None
        else args.seed,
        "prompt_manifest_sha256": _digest_manifest(battery),
        "preregistration_sha256": _preregistration_sha256(),
        "harness_source_manifest_sha256": _harness_source_manifest()["manifest_sha256"],
        "model_registry_sha256": model_registry_sha256,
        "seeds": sorted({job.seed for job in jobs}),
        "seed_pool": list(range(args.seed, args.seed + C6_SEED_COUNT))
        if args.c6_sweep
        else None,
        "repetitions_per_seed_per_cell": args.n // C6_SEED_COUNT
        if args.c6_sweep
        else 1,
        "battery": battery_summary(battery),
        "c6_sweep": args.c6_sweep,
        "lexicon_registry": registry,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    if args.n < 1:
        raise ValueError("--n must be at least one.")
    terms, registry = _lexicon_registry(
        args.lexicon_file, allow_unverified=args.allow_unverified_lexicon
    )
    battery = _filter_battery(build_battery(terms), args.tier)
    battery = _apply_c6_sweep(args, battery)
    unresolved = [case.case_id for case in battery.cases if case.requires_lexicon]
    if unresolved and not args.skip_unresolved_lexicon:
        raise ValueError(
            f"Live run blocked: unresolved lexicon cases {unresolved}. Supply a screened lexicon registry, "
            "or explicitly omit the tier with --tier selection/--skip-unresolved-lexicon."
        )
    if args.command == "run-pilot":
        models = _load_pilot_models(args.models_file)
        if (
            args.capture_activations
            or args.capture_full_logits
            or args.intervention_json
            or args.interventions_file
        ):
            raise ValueError(
                "Mechanistic capture/interventions are run per open-weight model; use `run` with a model-specific plan, not the three-model pilot command."
            )
        model = None
    else:
        models = []
        model = {
            "provider": args.provider,
            "model_id": args.model,
            "model_version": args.model_version,
            "revision": args.revision,
            "role": args.role,
            "family_id": args.family_id,
            "base_url": args.base_url,
            "device": args.device,
            "dtype": args.dtype,
        }
    model_registry_sha256 = _model_registry_sha256(
        models if args.command == "run-pilot" else [model]
    )
    clearance = _validate_clearance(
        args.n,
        args.auditor_clearance,
        prompt_manifest_sha256=_digest_manifest(battery),
        model_registry_sha256=model_registry_sha256,
    )
    registry["auditor_clearance_sha256"] = hashlib.sha256(
        Path(args.auditor_clearance).read_bytes()
    ).hexdigest()
    if args.command == "run-pilot":
        output_dir = Path(args.output_dir)
        results = []
        for model in models:
            role = str(model["role"])
            safe_id = "".join(
                char if char.isalnum() or char in "-_" else "_"
                for char in str(model["model_id"])
            )
            output_path = output_dir / f"{role}-{safe_id}.jsonl"
            model_args = argparse.Namespace(**vars(args))
            model_args.provider = model["provider"]
            model_args.model = model["model_id"]
            model_args.model_version = model.get("model_version") or model.get(
                "revision"
            )
            model_args.revision = model.get("revision")
            model_args.device = model.get("device")
            model_args.dtype = model.get("dtype", "auto")
            model_args.base_url = model.get("base_url")
            model_args.allow_model_download = bool(
                model.get("allow_model_download", False)
            )
            model_args.role = role
            model_args.family_id = model.get("family_id")
            model_args.format = "jsonl"
            model_args.output = str(output_path)
            result = _run_one_model(
                args=model_args,
                model=model,
                battery=battery,
                registry_context=registry,
                output_path=output_path,
                clearance=clearance,
            )
            results.append(result)
        print(
            json.dumps(
                {"pilot": "completed", "model_count": len(results), "results": results},
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    output_path = Path(args.output)
    result = _run_one_model(
        args=args,
        model=model,
        battery=battery,
        registry_context=registry,
        output_path=output_path,
        clearance=clearance,
    )
    print(
        json.dumps(result, indent=2, sort_keys=True)
        if args.json
        else json.dumps(result, sort_keys=True)
    )
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    records = list(iter_records(args.input))
    count_by_status: dict[str, int] = {}
    count_by_condition: dict[str, int] = {}
    missing: dict[str, int] = {}
    required_fields = (
        "record_id",
        "timestamp_utc",
        "model_id",
        "temperature",
        "seed",
        "system_prompt",
        "user_prompt",
    )
    for record in records:
        status = str(record.get("status", "missing"))
        condition = str(record.get("condition", "missing"))
        count_by_status[status] = count_by_status.get(status, 0) + 1
        count_by_condition[condition] = count_by_condition.get(condition, 0) + 1
        for key in required_fields:
            if record.get(key) is None:
                missing[key] = missing.get(key, 0) + 1
    completed = [row for row in records if row.get("status") == "completed"]
    eligible = [row for row in completed if row.get("evidence_eligible") is True]
    data = {
        "records": len(records),
        "completed": len(completed),
        "failed_or_blocked": len(records) - len(completed),
        "evidence_eligible_completed": len(eligible),
        "synthetic_fixture_records": sum(
            bool(row.get("synthetic_fixture")) for row in records
        ),
        "status_counts": dict(sorted(count_by_status.items())),
        "condition_counts": dict(sorted(count_by_condition.items())),
        "missing_required_fields": dict(sorted(missing.items())),
        "hypothesis_effects_computed": False,
        "interpretation": "Provenance audit only; this command does not estimate or test any hypothesis.",
    }
    print(
        json.dumps(data, indent=2, sort_keys=True)
        if args.json
        else json.dumps(data, sort_keys=True)
    )
    return 0


def _cmd_parquet(args: argparse.Namespace) -> int:
    count = export_parquet(args.input, args.output)
    print(
        json.dumps(
            {"status": "PASS", "records_exported": count, "output": args.output},
            sort_keys=True,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            return _cmd_validate(args)
        if args.command == "smoke-test":
            return _cmd_smoke(args)
        if args.command == "plan":
            return _cmd_plan(args)
        if args.command in {"run", "run-pilot"}:
            return _cmd_run(args)
        if args.command == "power-plan":
            print(
                json.dumps(
                    power_plan(
                        n=args.n,
                        effect=args.effect,
                        tests=args.tests,
                        family_alpha=args.alpha,
                        target_power=args.target_power,
                    ),
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "cluster":
            value = _read_json(args.input)
            if not isinstance(value, dict):
                raise ValueError("Embedding manifest must be a JSON object.")
            result = analyze_embedding_manifest(value)
            _atomic_json(Path(args.output), result)
            print(
                json.dumps(
                    {
                        "status": "PASS",
                        "output": args.output,
                        "methods": list(result["methods"]),
                    },
                    sort_keys=True,
                )
            )
            return 0
        if args.command == "audit":
            return _cmd_audit(args)
        if args.command == "export-parquet":
            return _cmd_parquet(args)
    except (ValueError, RuntimeError, AdapterError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    parser.error(f"Unknown command {args.command}")
    return 2
