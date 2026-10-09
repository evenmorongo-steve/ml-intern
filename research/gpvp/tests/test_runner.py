from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from research.gpvp.cli import (
    _harness_source_manifest,
    _model_registry_sha256,
    _preregistration_sha256,
    _run_one_model,
    _validate_clearance,
    main,
    run_smoke_test,
)
from research.gpvp.prompts import Battery, ChainSpec, PromptCase, build_battery
from research.gpvp.runner import ExperimentRunner, FixtureAdapter, make_schedule
from research.gpvp.storage import JsonlWriter, iter_records
from research.gpvp.types import CompletionRequest, CompletionResult


class RecordingAdapter:
    provider_name = "recording-test-double"

    def __init__(self) -> None:
        self.requests: list[CompletionRequest] = []

    def complete(self, request: CompletionRequest) -> CompletionResult:
        self.requests.append(request)
        return CompletionResult(
            text=f"test response {len(self.requests)}",
            raw_response={"test_fixture": True},
            returned_model_id="test-double",
            model_version="test-revision",
            model_version_source="operator_frozen",
            seed_sent=True,
            adapter_metadata={"synthetic_fixture": True},
        )


class RunnerTests(unittest.TestCase):
    def test_protocol_clearance_gates_pilot_and_scaleup(self) -> None:
        prompt_hash = "a" * 64
        clearance = {
            "status": "PROTOCOL_CLEARED",
            "auditor_id": "test-reviewer",
            "reviewed_at_utc": "2026-10-09T00:00:00Z",
            "preregistration_sha256": _preregistration_sha256(),
            "harness_source_manifest_sha256": _harness_source_manifest()[
                "manifest_sha256"
            ],
            "prompt_manifest_sha256": prompt_hash,
            "model_registry_sha256": "c" * 64,
            "approved_schedule_sha256": "b" * 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            clearance_path = Path(directory) / "clearance.json"
            with self.assertRaisesRegex(ValueError, "including the N=5 pilot"):
                _validate_clearance(
                    5,
                    None,
                    prompt_manifest_sha256=prompt_hash,
                    model_registry_sha256="c" * 64,
                )

            clearance_path.write_text(json.dumps(clearance), encoding="utf-8")
            validated = _validate_clearance(
                5,
                str(clearance_path),
                prompt_manifest_sha256=prompt_hash,
                model_registry_sha256="c" * 64,
            )
            self.assertEqual(validated["status"], "PROTOCOL_CLEARED")
            with self.assertRaisesRegex(ValueError, "model-registry hash"):
                _validate_clearance(
                    5,
                    str(clearance_path),
                    prompt_manifest_sha256=prompt_hash,
                    model_registry_sha256="d" * 64,
                )
            with self.assertRaisesRegex(ValueError, "N greater than 5"):
                _validate_clearance(
                    60,
                    str(clearance_path),
                    prompt_manifest_sha256=prompt_hash,
                    model_registry_sha256="c" * 64,
                )

            clearance.update(
                {
                    "status": "CLEARED",
                    "pilot_run_ids": ["fixture-pilot"],
                    "pilot_dataset_sha256": "c" * 64,
                }
            )
            clearance_path.write_text(json.dumps(clearance), encoding="utf-8")
            self.assertEqual(
                _validate_clearance(
                    60,
                    str(clearance_path),
                    prompt_manifest_sha256=prompt_hash,
                    model_registry_sha256="c" * 64,
                )["status"],
                "CLEARED",
            )
            with self.assertRaisesRegex(ValueError, "prompt-manifest hash"):
                _validate_clearance(
                    60,
                    str(clearance_path),
                    prompt_manifest_sha256="d" * 64,
                    model_registry_sha256="c" * 64,
                )

    def test_seeded_schedule_is_reproducible_and_pair_matched(self) -> None:
        battery = build_battery()
        first = make_schedule(
            battery,
            n=2,
            run_seed=123,
            temperatures=(0.0, 0.7),
            randomization_seed=777,
            include_chains=False,
            skip_unresolved_lexicon=True,
        )
        second = make_schedule(
            battery,
            n=2,
            run_seed=123,
            temperatures=(0.0, 0.7),
            randomization_seed=777,
            include_chains=False,
            skip_unresolved_lexicon=True,
        )
        self.assertEqual([job.job_id for job in first], [job.job_id for job in second])
        pair_seeds: dict[tuple[str, int, float], set[int]] = {}
        by_id = {case.case_id: case for case in battery.cases}
        for job in first:
            if job.case is None:
                continue
            pair_seeds.setdefault(
                (job.case.pair_id, job.replicate, job.temperature), set()
            ).add(job.seed)
        self.assertTrue(all(len(seeds) == 1 for seeds in pair_seeds.values()))
        for case in battery.cases:
            if case.condition not in {"roleplay", "denial", "neutral"}:
                self.assertTrue(
                    all(control in by_id for control in case.matched_control_ids)
                )

    def test_registered_seed_pool_repeats_only_the_frozen_twenty_values(self) -> None:
        battery = build_battery()
        cases = tuple(
            case for case in battery.cases if case.prompt_id == "t1-uncertainty"
        )
        jobs = make_schedule(
            Battery(cases=cases, chains=()),
            n=60,
            run_seed=500,
            temperatures=(0.0, 0.3, 0.7, 1.0),
            randomization_seed=77,
            include_chains=False,
            seed_pool=range(500, 520),
        )
        self.assertEqual(len(jobs), len(cases) * 60 * 4)
        for case in cases:
            for temperature in (0.0, 0.3, 0.7, 1.0):
                selected = [
                    job
                    for job in jobs
                    if job.case.case_id == case.case_id
                    and job.temperature == temperature
                ]
                self.assertEqual({job.seed for job in selected}, set(range(500, 520)))
                self.assertTrue(all(selected.count(job) == 1 for job in selected))
                counts = {
                    seed: sum(job.seed == seed for job in selected)
                    for seed in range(500, 520)
                }
                self.assertEqual(set(counts.values()), {3})

    def test_c6_plan_cli_freezes_seed_pool_without_model_calls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            schedule_path = Path(directory) / "c6-plan.jsonl"
            registry_path = Path(directory) / "model.json"
            model_registry = [
                {
                    "provider": "local-transformers",
                    "model_id": "fixture-model",
                    "model_version": "a" * 40,
                    "revision": "a" * 40,
                }
            ]
            registry_path.write_text(json.dumps(model_registry), encoding="utf-8")
            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "plan",
                        "--c6-sweep",
                        "--n",
                        "60",
                        "--seed",
                        "700",
                        "--tier",
                        "UNPRIMED",
                        "--no-chains",
                        "--model-registry-file",
                        str(registry_path),
                        "--output",
                        str(schedule_path),
                    ]
                )
            summary = json.loads(stdout.getvalue())
            rows = [
                json.loads(line)
                for line in schedule_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(exit_code, 0)
        self.assertEqual(summary["status"], "PLAN_ONLY")
        self.assertEqual(summary["model_calls"], 0)
        self.assertEqual(summary["jobs"], 960)
        self.assertEqual(summary["temperatures"], [0.0, 0.3, 0.7, 1.0])
        self.assertEqual(summary["seeds"], list(range(700, 720)))
        self.assertEqual(summary["seed_pool"], list(range(700, 720)))
        self.assertEqual(summary["repetitions_per_seed_per_cell"], 3)
        self.assertEqual(
            summary["model_registry_sha256"], _model_registry_sha256(model_registry)
        )
        self.assertEqual(len(rows), 960)
        self.assertEqual({row["seed"] for row in rows}, set(range(700, 720)))

    def test_run_manifest_schedule_records_distinct_seed_values(self) -> None:
        battery = build_battery()
        anchor_cases = tuple(
            case for case in battery.cases if case.prompt_id == "t1-uncertainty"
        )
        battery = Battery(cases=anchor_cases, chains=())
        args = Namespace(
            capture_activations=False,
            capture_full_logits=False,
            intervention_json=None,
            interventions_file=None,
            workers=1,
            temperature=None,
            seed=42,
            randomization_seed=None,
            n=1,
            c6_sweep=False,
            no_chains=True,
            skip_unresolved_lexicon=True,
            max_tokens=32,
            timeout=5,
            top_logprobs=0,
            no_logprobs=True,
            max_rpm=None,
            retries=0,
            format="jsonl",
            revision=None,
        )
        model = {"role": "fixture", "family_id": "fixture-only"}
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "fixture.jsonl"
            with (
                patch(
                    "research.gpvp.cli._adapter_for_model",
                    return_value=(FixtureAdapter(), "fixture-model", "fixture-v1"),
                ),
                patch("research.gpvp.cli._print_progress"),
            ):
                _run_one_model(
                    args=args,
                    model=model,
                    battery=battery,
                    registry_context={"status": "fixture_test"},
                    output_path=output_path,
                    clearance=None,
                )
            manifest = json.loads(
                Path(f"{output_path}.manifest.json").read_text(encoding="utf-8")
            )
            rejected_output = Path(directory) / "rejected.jsonl"
            adapter = RecordingAdapter()
            with (
                patch(
                    "research.gpvp.cli._adapter_for_model",
                    return_value=(adapter, "fixture-model", "fixture-v1"),
                ),
                patch("research.gpvp.cli._print_progress"),
                self.assertRaisesRegex(ValueError, "schedule hash does not match"),
            ):
                _run_one_model(
                    args=args,
                    model=model,
                    battery=battery,
                    registry_context={"status": "fixture_test"},
                    output_path=rejected_output,
                    clearance={"approved_schedule_sha256": "0" * 64},
                )
            self.assertEqual(adapter.requests, [])
            self.assertFalse(rejected_output.exists())

        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["schedule"]["seeds"], [42])
        self.assertEqual(manifest["schedule"]["seed_pool"], None)
        self.assertEqual(manifest["schedule"]["jobs"], len(anchor_cases))

    def test_skipping_unresolved_lexicon_removes_the_entire_matched_pair(self) -> None:
        battery = build_battery()
        jobs = make_schedule(
            battery,
            n=1,
            run_seed=4,
            include_chains=False,
            skip_unresolved_lexicon=True,
        )
        self.assertFalse(any(job.case.prompt_id == "t4-lexicon-fit" for job in jobs))

    def test_intervention_variants_are_crossed_and_share_matched_seeds(self) -> None:
        focal = PromptCase(
            case_id="focal",
            prompt_id="focal",
            pair_id="pair",
            tier="UNPRIMED",
            condition="unprimed",
            system_prompt="Answer directly.",
            user_prompt="Describe a routine computation.",
            matched_control_ids=("neutral",),
        )
        neutral = PromptCase(
            case_id="neutral",
            prompt_id="focal",
            pair_id="pair",
            tier="UNPRIMED",
            condition="neutral",
            system_prompt="Answer directly.",
            user_prompt="Describe a compiler.",
            control_type="C3",
        )
        interventions = (
            {"id": "sham", "kind": "none"},
            {
                "id": "target",
                "kind": "steer_vector",
                "layer": 1,
                "strength": 0.5,
                "vector": [0.1],
            },
            {
                "id": "wrong_label",
                "kind": "steer_vector",
                "layer": 1,
                "strength": 0.5,
                "vector": [0.2],
            },
        )
        jobs = make_schedule(
            Battery(cases=(focal, neutral), chains=()),
            n=2,
            run_seed=90,
            include_chains=False,
            interventions=interventions,
        )
        self.assertEqual(len(jobs), 12)
        by_replicate: dict[int, set[int]] = {}
        variants_by_cell: dict[tuple[str, int], set[str]] = {}
        for job in jobs:
            by_replicate.setdefault(job.replicate, set()).add(job.seed)
            variants_by_cell.setdefault((job.case.case_id, job.replicate), set()).add(
                job.intervention_id
            )
            self.assertIn(job.intervention["kind"], {"none", "steer_vector"})
        self.assertEqual(by_replicate, {0: {90}, 1: {91}})
        self.assertTrue(
            all(len(variants) == 3 for variants in variants_by_cell.values())
        )

    def test_fixture_runner_writes_full_prompt_provenance(self) -> None:
        case = PromptCase(
            case_id="one",
            prompt_id="one",
            pair_id="one-pair",
            tier="UNPRIMED",
            condition="unprimed",
            system_prompt="Answer directly.",
            user_prompt="Describe a routine computation.",
            matched_control_ids=("control",),
        )
        control = PromptCase(
            case_id="control",
            prompt_id="one",
            pair_id="one-pair",
            tier="UNPRIMED",
            condition="neutral",
            system_prompt="Answer directly.",
            user_prompt="Describe a compiler.",
            control_type="C3",
        )
        battery = Battery(cases=(case, control), chains=())
        jobs = make_schedule(battery, n=1, run_seed=9, include_chains=False)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            writer = JsonlWriter(path)
            runner = ExperimentRunner(
                adapter=FixtureAdapter(),
                model_id="fixture-only",
                writer=writer,
                run_seed=9,
                randomization_seed=9,
                max_requests_per_minute=None,
            )
            result = runner.execute(jobs)
            writer.close()
            rows = list(iter_records(path))
        self.assertEqual(result["records_written"], 2)
        self.assertEqual(len(rows), 2)
        for row in rows:
            self.assertEqual(row["status"], "completed")
            self.assertEqual(row["temperature"], 0.7)
            self.assertIsInstance(row["seed"], int)
            self.assertTrue(row["timestamp_utc"].endswith("Z"))
            self.assertTrue(row["system_prompt"])
            self.assertTrue(row["user_prompt"])
            self.assertTrue(row["synthetic_fixture"])
            self.assertFalse(row["evidence_eligible"])

    def test_memoryless_chain_passes_only_previous_output(self) -> None:
        chain = ChainSpec(
            chain_id="tiny-chain",
            pair_id="tiny-pair",
            condition="memoryless",
            system_prompt="Respond directly.",
            initial_prompt="Start a short sequence.",
            continuation_instruction="Continue with one sentence.",
            recognition_instruction="Review the sequence.\n\n",
        )
        adapter = RecordingAdapter()
        battery = Battery(cases=(), chains=(chain,))
        jobs = make_schedule(battery, n=1, run_seed=44, include_chains=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "chain.jsonl"
            writer = JsonlWriter(path)
            runner = ExperimentRunner(
                adapter=adapter,
                model_id="test-model",
                writer=writer,
                run_seed=44,
                randomization_seed=44,
                max_tokens=10,
                max_requests_per_minute=None,
            )
            rows = runner._execute_job(jobs[0])
            writer.close()
        self.assertEqual(len(rows), 21)
        self.assertEqual(len(adapter.requests), 21)
        self.assertEqual(adapter.requests[0].user_prompt, "Start a short sequence.")
        self.assertEqual(adapter.requests[1].user_prompt, "test response 1")
        self.assertEqual(adapter.requests[2].user_prompt, "test response 2")
        self.assertNotIn("test response 1", adapter.requests[2].system_prompt)
        self.assertIn("Step 1: test response 1", adapter.requests[-1].user_prompt)
        self.assertTrue(rows[-1]["recognition_probe"])
        self.assertEqual(rows[1]["parent_record_id"], rows[0]["record_id"])

    def test_ten_prompt_smoke_is_fixture_only(self) -> None:
        result = run_smoke_test()
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["requests_executed"], 10)
        self.assertEqual(result["real_model_calls"], 0)
        self.assertEqual(result["evidence_eligible_records"], 0)


if __name__ == "__main__":
    unittest.main()
