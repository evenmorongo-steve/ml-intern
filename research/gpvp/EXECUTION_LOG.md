# Execution log

All entries are local tool/code executions in the Arena checkout on 2026-10-08 UTC. No inference API was called. The test-fixture transcript below is explicitly not a model transcript.

## Environment preflight

Executed a boolean-only credential check and `importlib.util.find_spec` check. No credential values were printed.

```text
OPENAI_API_KEY=absent
ANTHROPIC_API_KEY=absent
GOOGLE_API_KEY=absent
MISTRAL_API_KEY=absent
DEEPSEEK_API_KEY=absent
DASHSCOPE_API_KEY=absent
XAI_API_KEY=absent
HF_TOKEN=absent
torch=False
transformers=False
transformer_lens=False
nnsight=False
vllm=False
litellm=False
pytest=False
ruff=False
numpy=False
sklearn=False
pyarrow=False
```

Python reported `3.11.2`; `uv` was not installed. Ruff 0.16.10 was installed into `/tmp/gpvp-ruff` only for validation, not added to this repository.

## Prompt validation

Command: `python -m research.gpvp validate --json`

```json
{
  "battery": {
    "chain_conditions": 4,
    "prompt_cases_by_condition": {
      "denial": 13,
      "light_primed": 2,
      "mechanistic": 1,
      "neutral": 14,
      "novel_lexicon": 1,
      "primed": 5,
      "roleplay": 13,
      "sycophancy": 1,
      "unprimed": 4
    },
    "prompt_cases_by_tier_including_controls": {
      "LIGHT-PRIMED": 8,
      "MECHANISTIC": 4,
      "NOVEL-LEXICON": 4,
      "PRIMED": 20,
      "UNPRIMED": 18
    },
    "single_prompt_cases": 54,
    "unresolved_lexicon_cases": ["t4-lexicon-fit"]
  },
  "lexicon_registry": {"screening_status": "not_run", "status": "not_supplied"},
  "matched_control_validation": "PASS",
  "prompt_manifest_sha256": "91dee1874e77c5d6295e78abcc6ccb71cecc0f20e8c9a209c872ccc5f095a064",
  "protocol_version": "0.1.0",
  "restricted_terms_outside_primed": 0,
  "scientific_runs_performed": 0,
  "status": "PASS"
}
```

This is a static prompt/control validation, not a model-behavior leakage measurement.

## Offline 10-request smoke test

Command: `python -m research.gpvp smoke-test --json`

```json
{
  "all_records_marked_synthetic": true,
  "evidence_eligible_records": 0,
  "fixture_assistant_text": "[GPVP TEST FIXTURE: no model inference occurred]",
  "mode": "fixture-only; not a model experiment",
  "real_model_calls": 0,
  "records_round_tripped": 10,
  "requests_executed": 10,
  "status": "PASS",
  "temporary_data_removed": true
}
```

The fixed fixture text was used only to check scheduling, record creation, JSONL write/read, and evidence-isolation flags. It is not evidence and is not in the raw study dataset.

## Power design calculation

Command: `python -m research.gpvp power-plan`

- Calculation: two-sided normal approximation for paired standardized mean difference.
- Assumptions: N=60 paired units; assumed `d_z=0.5`; six primary tests; family alpha=.05; conservative per-test alpha=.05/6.
- Output: approximate power `0.8915337655539629`; minimum N for .80 under the same formula `49`.
- This consumes no study observations and does not account for clustering or non-normal outcomes. See `power.py` and preregistration for limits.

## Plan and live-call preflight

Command: `python -m research.gpvp plan --output /tmp/gpvp-plan-final.jsonl --n 2 --seed 7 --tier UNPRIMED --no-chains`

```text
status=PLAN_ONLY
model_calls=0
jobs=36
n_per_case=2
run_seed=7
randomization_seed=7
schedule_sha256=7ce59781d890e7fee3751ffe890cac313eca6f10bd1d947d9700e8894bf95fd2
schedule_lines=36
```

A live preflight was attempted with a provider model label but no credential. It stopped at adapter initialization with `ERROR: No credential found in OPENAI_API_KEY.` No HTTP request was issued and no output data file was created.

## C6 plan-only validation

Command: `python -m research.gpvp plan --c6-sweep --n 60 --seed 20261009 --tier UNPRIMED --no-chains --model-registry-file /tmp/gpvp-c6-model-registry.json --output /tmp/gpvp-c6-plan-final.jsonl`

```text
status=PLAN_ONLY
model_calls=0
jobs=960
prompt_cases=4 (focal, role-play, denial, neutral)
temperatures=[0.0, 0.3, 0.7, 1.0]
seeds=20261009..20261028 (20 distinct values)
repetitions_per_seed_per_cell=3
prompt_manifest_sha256=9eba045ce9e7cee8ace1ac86b9f8a6087d1c59f0cbe101ea18383fbc0a76db34
preregistration_sha256=b5a884be0e53dd9f8e9a8e2ef24fb8da3eb009247b4e095d74a81ff966f2f7d0
harness_source_manifest_sha256=959213afcc46cac3f9f8dd0d27c77c68614075a35c74ea4dca55be3ee6c55338
model_registry_sha256=927393621f84eeaf203d588a75962b6755b0c5cf2b05ae4562e8bac8bb3742ec
schedule_sha256=6fa08daf4070bbd876dc01b42390bbb747bd78f4d6334bd9ae9c3da6878789fe
```

The registry file was a temporary fixture-only hash input (`FIXTURE_ONLY_NOT_A_MODEL`), not a study model registry. This validated only a frozen schedule; no model was called.

## Pre-run Auditor gate

A provider command with no clearance was run using a placeholder model label and a `/tmp` output path. It returned exit code 2 with `Every live model run, including the N=5 pilot, is blocked until an independent Auditor supplies pre-run clearance...`; the provider adapter was not initialized, no HTTP request was sent, and no output data file was created.

## Tests and lint

Executed:

```bash
python -m compileall -q research/gpvp
python -m unittest discover -s research/gpvp/tests -v
PYTHONPATH=/tmp/gpvp-ruff python -m ruff check research/gpvp
PYTHONPATH=/tmp/gpvp-ruff python -m ruff format --check research/gpvp
```

Output: 32 unit tests passed; Ruff scoped check passed; 24 files already formatted.

A repository-wide Ruff check was also attempted. It reported 624 findings in pre-existing files outside `research/gpvp`; no unrelated files were edited. The changed package's scoped lint and formatting checks pass.

## Study data

- Model calls: 0.
- Raw model transcripts: 0.
- Pilot records: 0.
- Hypothesis effect estimates: none.
- Auditor clearance: none.
