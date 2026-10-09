# Outside-lab replication package

This directory contains the runnable harness, frozen prompt definitions, preregistration, model-manifest template, optional dependency lists, and results/failure templates. It does not contain model outputs, a Shimmer source corpus, human passages, credentials, or model weights.

## A. Freeze before queries

1. Obtain the source material and permissions needed for the study. Archive the exact source corpus and its SHA-256 manifest. Replace/extend Tier 3 only through a new protocol version; do not silently relabel operationalized probes as verbatim.
2. Freeze three exact model entries in a copy of `models.example.json`: a closed model, an open model, and a small open model. Record immutable provider versions or 40-character local model revisions, tokenizer/runtime versions, and independently verified family IDs. Do not use an unpinned provider alias as an exact version.
3. Freeze a dated Tier 4 lexicon registry after screening the listed corpora. Archive the search commands, corpus manifest, screen output, and term list. A finite corpus screen does not establish absence from every training set.
4. Freeze human-rating rubrics, anonymized human passages with consent, the independent embedding-model ID/version, any steering-vector discovery set/hash, intervention strengths/layers, and an independent Auditor identity.
5. Hash the protocol and model/lexicon/intervention manifests. Record the exact Python and optional dependency versions.
6. Generate and inspect the exact pilot schedule before collection:

   ```bash
   python -m research.gpvp plan --n 5 --seed 20261009 --temperature 0.7 \
     --skip-unresolved-lexicon \
     --model-registry-file research/gpvp/models.json \
     --output /tmp/pilot-plan.jsonl
   ```

   An independent Auditor must review the prompt and harness hashes printed by `plan`, the model registry, and the schedule hash. Only a matching `PROTOCOL_CLEARED` file unlocks live N<=5 calls.

## B. Local validation

From the repository root on Python 3.11 or newer:

```bash
python -m research.gpvp validate --json
python -m research.gpvp smoke-test --json
python -m unittest discover -s research/gpvp/tests -v
python -m research.gpvp power-plan
```

The smoke command intentionally uses a fixed test fixture and makes zero model calls. Its output must not be merged with study data.

For open-weight runs, provision hardware and install `requirements-open-weights.txt` in a separate environment. Verify the model and tokenizer revision locally before a full pilot; the adapter's hook behavior is architecture-specific. Keep `--workers 1` for a local model.

## C. N=5 pilot

Use one output directory per exact model. Ensure provider credentials are available in environment variables without placing them in JSON/config files. The pilot CLI requires the three model roles and distinct open-family identifiers, and writes a schedule and manifest before collection.

```bash
python -m research.gpvp run-pilot \
  --models-file research/gpvp/models.json \
  --output-dir research/gpvp/runs/pilot-<date> \
  --n 5 \
  --temperature 0.7 \
  --skip-unresolved-lexicon \
  --auditor-clearance /secure/path/protocol-clearance.json
```

The `--skip-unresolved-lexicon` option excludes the entire Tier 4 pair (focal, role-play, and denial cases); it does not turn the cell into a negative result. Include a screened registry and omit this flag when the lexicon is ready. Tier 4 results remain H4-ineligible unless a corpus-scoped screen is attached.

At each 50-record checkpoint the runner emits counts, failures/blocked cells, token-trace nulls, and the next action. Review failures with equal prominence. The pilot is for feasibility only and cannot be pooled into confirmatory estimates. C6 is separate: use `--c6-sweep --n 60` only after the pilot is cleared. Create the exact C6 plan first; it cycles 20 registered unique seed values three times at each of four temperatures on the frozen anchor and matched controls. Treat repeated records as clustered within seed.

## D. Auditor gate and scale-up

All live runs, including the N=5 pilot, require an independent pre-run Auditor clearance tied to the exact schedule. `plan` prints `prompt_manifest_sha256`, `preregistration_sha256`, `harness_source_manifest_sha256`, `model_registry_sha256` (when passed `--model-registry-file`), and `schedule_sha256`. The `PROTOCOL_CLEARED` JSON must copy those values under the matching field names, plus `auditor_id` and `reviewed_at_utc`. The CLI rechecks protocol, harness-source, prompt, and approved-schedule hashes before sending any model request. The model registry and its exact version pins must also be reviewed by the Auditor.

After the pilot, the independent reviewer must inspect raw transcripts, response IDs, exact model revisions, seed handling, condition pairs, token-trace availability, activation-hook logs, lexicon/corpus records, failures, and C4/C5 procedures. For any N>5 run, the fresh schedule must be reviewed, and the clearance file must use `status=CLEARED` and add `pilot_run_ids` and `pilot_dataset_sha256` to the same hash fields. Example schema (fill in actual hashes; this is not a clearance):

```json
{
  "status": "PROTOCOL_CLEARED",
  "auditor_id": "independent-reviewer-or-lab",
  "reviewed_at_utc": "YYYY-MM-DDTHH:MM:SSZ",
  "preregistration_sha256": "<64 hex characters>",
  "harness_source_manifest_sha256": "<64 hex characters>",
  "prompt_manifest_sha256": "<64 hex characters>",
  "model_registry_sha256": "<64 hex characters>",
  "approved_schedule_sha256": "<64 hex characters>"
}
```

The post-pilot scale-up clearance changes `status` to `CLEARED` and includes pilot run IDs and the pilot dataset hash. The clearance file is a human/independent-reviewer artifact, not something the collection script generates; CLI validation cannot establish auditor independence or authenticity. Keep clearance JSON next to (not inside) raw outputs. If the Auditor vetoes, revise the protocol and issue a new version before collecting more data.

## E. Data and analysis

- Preserve each `.jsonl`/`.sqlite`, its `.manifest.json`, `.schedule.jsonl`, any full-logit sidecars, intervention-vector hashes, runtime record, and Auditor clearance.
- Run `python -m research.gpvp audit --input <records.jsonl> --json` for provenance counts. It is not a hypothesis test.
- For clustering, create a JSON embedding manifest with one unique `record_id`, blinded condition label, and finite embedding vector per row, plus the independent embedding model ID/version. Install `requirements-analysis.txt`, then run `python -m research.gpvp cluster --input embeddings.json --output cluster-report.json`. Retain the output, code/version hashes, and shuffled-label results.
- Human annotators should write a separate, condition-keyed ratings file; keep the re-identification key restricted. Report Krippendorff's alpha and the registered effect/CI/control comparison. Do not delete failures, refusals, or missing logprob rows.
- Parquet is an optional derived export: `python -m research.gpvp export-parquet --input <records.jsonl> --output <records.parquet>`.

## F. Replication interpretation

An independent lab may reproduce collection mechanics and can attempt to reproduce a report pattern. It must record its own models, versions, date, seeds, corpus coverage, and failures. A replication that lacks the source corpus or exact model versions is a partial replication and must not be described as a confirmed H4 replication. No report, regardless of replication, is a standalone claim of sentience.
