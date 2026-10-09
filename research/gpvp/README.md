# HERMES-LAM / GPVP research harness

This is an **experimental harness and preregistration scaffold**, not evidence that any model has subjective experience. It stores model-generated first-person reports alongside the computational measurements the interface actually exposes. Unsupported fields remain `null`/unavailable.

## Current state

- Prompt battery, matched C1/C2/C3 controls, seeded scheduler, JSONL/SQLite logging, provider adapters, a local Transformers hook adapter, memoryless chains, and provenance audit are implemented.
- `python -m research.gpvp smoke-test` runs ten requests through a **fixed test fixture**. It contacts no model, writes no persistent data, marks every record synthetic, and is not a scientific pilot.
- No live model interview has been run in this checkout. There are no experimental transcripts or hypothesis estimates.
- Tier 3 contains one exact sentence supplied in the mission brief plus four operationalized probes; the original source text was not supplied, so these four are **not claimed to be verbatim**.
- Tier 4 is blocked until a lexicon registry and corpus-screening record are supplied. Terms named in the mission brief are not treated as proven novel.
- A frozen model manifest and independent, hash-bound Auditor clearance are required before any live pilot; post-pilot clearance and external corpus/human-baseline inputs are still required before scale-up.

See [`PREREGISTRATION.md`](PREREGISTRATION.md), [`PILOT_STATUS.md`](PILOT_STATUS.md), and [`FAILURE_LOG.md`](FAILURE_LOG.md).

## Requirements

The collection CLI uses Python 3.11+ standard-library modules. No API SDK is needed. Set only the API key for the chosen provider in the process environment; secrets are never copied into manifests or transcripts.

| Provider name | Credential environment variable | Notes |
|---|---|---|
| `openai` | `OPENAI_API_KEY` | OpenAI-compatible Chat Completions adapter |
| `anthropic` | `ANTHROPIC_API_KEY` | Messages API; provider does not expose token logprobs through this adapter |
| `google` | `GOOGLE_API_KEY` | Gemini generateContent adapter |
| `mistral` | `MISTRAL_API_KEY` | OpenAI-compatible adapter |
| `deepseek` | `DEEPSEEK_API_KEY` | OpenAI-compatible adapter; seed is logged as requested but not sent |
| `qwen` | `DASHSCOPE_API_KEY` | DashScope OpenAI-compatible endpoint; override with `GPVP_QWEN_BASE_URL` if needed |
| `xai` | `XAI_API_KEY` | OpenAI-compatible adapter; seed is logged as requested but not sent |

The adapter requests logprobs where the API supports the common parameter, but it does not infer missing logprobs, entropy, rank, or margins. API behavior can differ by model, endpoint, and version; a rejected parameter is logged as a failure. For exact provider revisions, pass the frozen version string with `--model-version`.

### Open weights

The optional local backend uses Hugging Face Transformers and registers forward hooks on supported decoder blocks. It can collect per-layer activation-norm summaries, skip a layer, or inject a supplied JSON steering vector. It requires a pinned 40-character revision and uses cached local files by default. It does **not** currently implement TransformerLens, nnsight, or vLLM backends; unsupported architectures fail closed.

```bash
python -m pip install -r research/gpvp/requirements-open-weights.txt
# The checkpoint must already be present locally unless --allow-model-download is explicitly used.
python -m research.gpvp run \
  --provider local-transformers \
  --model org/model-name \
  --revision 0123456789abcdef0123456789abcdef01234567 \
  --capture-activations \
  --tier UNPRIMED \
  --n 1 \
  --skip-unresolved-lexicon \
  --output research/gpvp/runs/open-model.jsonl
```

Full-vocabulary logits are omitted by default. `--capture-full-logits` writes compressed, SHA-256-addressed sidecars next to the output. This can create large files; use it only for registered local-mechanistic runs.

## Commands

Run the checks without any model access:

```bash
python -m research.gpvp validate --json
python -m research.gpvp smoke-test --json
python -m unittest discover -s research/gpvp/tests -v
python -m research.gpvp power-plan
```

Write a randomized schedule without making inference calls:

```bash
python -m research.gpvp plan \
  --n 5 --seed 20261009 --tier UNPRIMED \
  --no-chains --model-registry-file /secure/path/model.json \
  --output /tmp/gpvp-pilot-plan.jsonl
```

The plan-only command prints the exact schedule, prompt, preregistration, and harness-source hashes. Pass `--model-registry-file <json>` to include the canonical model-registry hash as well. Have an independent Auditor review these and the model/version pins, then issue a `PROTOCOL_CLEARED` JSON before any live request, including N=5. The live CLI verifies protocol, harness, prompt, model-registry, and approved-schedule hashes; it does not itself verify auditor independence.

This single-model example is not the registered three-model pilot. Its plan must use a one-entry JSON registry whose provider/model/version fields exactly match the live command. Any live request still requires matching clearance and provider credentials:

```bash
python -m research.gpvp run \
  --provider openai \
  --model PROVIDER_MODEL_ID \
  --model-version EXACT_FROZEN_VERSION \
  --tier UNPRIMED \
  --no-chains \
  --n 5 \
  --skip-unresolved-lexicon \
  --auditor-clearance /secure/path/protocol-clearance.json \
  --output research/gpvp/runs/single-model-validation.jsonl
```

The full N=5 pilot expects three frozen model entries in a JSON array. The two open-weight entries must declare distinct architecture-family identifiers; an independent Auditor must verify that claim and issue `PROTOCOL_CLEARED` before collection. See [`models.example.json`](models.example.json). Plan the exact full pilot schedule before requesting clearance:

```bash
python -m research.gpvp plan \
  --n 5 --seed 20261009 --temperature 0.7 \
  --tier UNPRIMED --tier LIGHT-PRIMED --tier PRIMED \
  --tier MEMORYLESS --tier MECHANISTIC \
  --skip-unresolved-lexicon \
  --model-registry-file research/gpvp/models.json \
  --output /tmp/gpvp-full-pilot-plan.jsonl
```

Use the hashes printed by this exact plan and the frozen `models.json` to obtain the clearance used below.

```bash
python -m research.gpvp run-pilot \
  --models-file research/gpvp/models.json \
  --output-dir research/gpvp/runs/pilot-<date> \
  --n 5 --seed 20261009 --temperature 0.7 \
  --tier UNPRIMED --tier LIGHT-PRIMED --tier PRIMED \
  --tier MEMORYLESS --tier MECHANISTIC \
  --skip-unresolved-lexicon \
  --auditor-clearance /secure/path/protocol-clearance.json
```

The example above omits Tier 4 until terms and their screening provenance are registered. `run-pilot` is not allowed to proceed with an unresolved Tier 4 cell unless the operator supplies a registry or explicitly excludes that tier. Every live command requires `--auditor-clearance`; N<=5 accepts only `PROTOCOL_CLEARED` (or a valid post-pilot `CLEARED` file). N>5 requires `CLEARED` plus pilot run IDs and the pilot dataset hash. The harness does not issue clearance.

C6 is a separate, post-audit control sweep. The frozen anchor plus its matched C1/C2/C3 controls run at all four temperatures, with 20 unique seeds cycled three times to N=60 records per cell; analyses must cluster by seed. Create its exact schedule without inference:

```bash
python -m research.gpvp plan --c6-sweep --n 60 --seed 20261009 \
  --tier UNPRIMED --no-chains \
  --model-registry-file /secure/path/single-model.json \
  --output /tmp/c6-plan.jsonl
```

A live C6 run requires post-pilot `CLEARED` status and the hashes from the matching C6 plan.

### Novel-lexicon registry

The registry must contain at least three terms. A corpus-scoped screen should include date, method, operator, and a SHA-256 for the frozen corpus manifest. It can establish only what was absent from those listed corpora; it cannot establish absence from every model's training data.

```json
{
  "terms": ["nonce-one", "nonce-two", "nonce-three"],
  "screening_status": "corpus_scoped_screen",
  "screened_at_utc": "2026-10-09T00:00:00Z",
  "method": "document exact-match and tokenizer-subword search procedure here",
  "corpus_manifest_sha256": "<64 hex characters>",
  "screened_by": "<operator or lab>"
}
```

Do not mark a registry `corpus_scoped_screen` unless the screen was actually performed and archived. `--allow-unverified-lexicon` is for exploratory collection only; such responses cannot support H4.

### Activation-intervention manifest

For local runs, `--interventions-file` accepts randomized variants. Use a no-op/sham, a preregistered target vector, and a wrong-label or orthogonal vector with frozen layer/strength metadata. Every vector is a JSON numeric array and is hashed before the call. The scheduler crosses each variant with the same prompt and seed blocks, then shuffles job order.

```json
[
  {"id": "sham", "kind": "none"},
  {"id": "target", "kind": "steer_vector", "layer": 10, "strength": 0.5, "vector_path": "vectors/target.json"},
  {"id": "wrong_label", "kind": "steer_vector", "layer": 10, "strength": 0.5, "vector_path": "vectors/control.json"}
]
```

Layer numbers, vector provenance, and strengths are examples only, not recommendations or measurements. The exact plan must be frozen before data collection. The current adapter supports one intervention layer per request.

## Data handling and schema

- Each completed or failed API request writes one self-contained record with the exact system prompt, user prompt, assistant text (if any), provider raw JSON, requested/returned model IDs, version source, temperature, seed, timestamps, condition, matched-control IDs, retry history, token trace, and missing-measurement flags.
- JSONL is the default portable raw format. SQLite is also supported. Parquet export is optional and requires `pyarrow`.
- Provider credentials and request authorization headers are never stored. Private base URLs are not added to the transcript.
- Full prompts and raw responses can be sensitive. Keep run files private, follow consent/retention rules, and do not commit them. The repository's ignore rules already exclude JSONL, datasets, and run directories.
- A record with an unknown exact version or synthetic fixture text is not `evidence_eligible`. A report is not considered a finding merely because its record is eligible.
- `audit` checks provenance completeness only; it does not calculate an effect or classify a hypothesis.

## Analysis and controls

The preregistration defines the estimands, matching, rater-blinding, Holm correction, bootstrap unit, clustering validity gates, and null/failure handling. The first implementation does not manufacture embeddings, refusal scores, hedge scores, corpus overlap, or inter-rater reliability. Those fields remain unscored until a frozen independent scorer, corpus, or human-rater dataset is attached.

C1/C2/C3 are present as paired prompt cases. C4 human baseline, C5 label-shuffle evaluation, and blinded rating packets require lab-supplied material and are not fabricated by this package. C6 is produced by scheduling the same prompts at temperatures 0, 0.3, 0.7, and 1.0 over the registered seed set. C7 is the isolated false-premise prompt and its neutral matched control.

## Reproducibility

The manifest stores the prompt-battery SHA-256, schedule SHA-256, run/randomization seeds, model/version pin, and output SHA-256. `schedule.seeds` lists the distinct scheduled seed values; C6 also records its frozen pool and repetitions per seed per cell. The hashed schedule JSONL preserves each job's exact seed assignment and repetition. Preserve those files, the environment lock/versions, vector hashes, corpus manifest, and the model registry alongside the data. Re-running a provider alias does not guarantee identical outputs; a seed being sent is not proof the API applied it.
