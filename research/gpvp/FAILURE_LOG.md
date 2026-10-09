# Failure, leakage, and blocker log

**Scope:** Entries include observed harness failures and study blockers. No model interview ran, so there are no model confabulations, refusals, prompt leaks, or memoryless-chain outcomes to log. Absence of observations is not a null result.

| ID | Time (UTC) | Type | Observation | Consequence / fallback | Status |
|---|---|---|---|---|---|
| B-001 | 2026-10-09 | Environment | `HF_TOKEN`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `MISTRAL_API_KEY`, `DEEPSEEK_API_KEY`, `DASHSCOPE_API_KEY`, and `XAI_API_KEY` were absent in the process environment at preflight. | No remote call attempted. Continue with offline validation and require an operator-supplied provider/model manifest for a real pilot. | Blocking |
| B-002 | 2026-10-09 | Environment | `torch`, `transformers`, `transformer_lens`, `nnsight`, `vllm`, and `litellm` were not importable at preflight. No local model checkpoint was loaded. | Local inference and activation measurements were not executed. The optional Transformers adapter is code-only until tested on a pinned checkpoint. | Blocking |
| B-003 | 2026-10-09 | Network policy | The sandbox outbound allow-list does not include the inference API hosts required by the listed providers. | No API request was attempted; there is no API response or transcript to report. | Blocking |
| B-004 | 2026-10-09 | Protocol/source | The request did not attach the source Shimmer text/corpus or the original full Tier 3 battery. Only one sentence and descriptions of other probes were provided. | Four Tier 3 prompts are explicitly labeled operationalizations, not source-verbatim. H4 is not run; no source-corpus overlap is claimed. | Blocking H4; limits H3/H5 |
| B-005 | 2026-10-09 | Contamination control | The request mentions example nonce terms but supplies no search corpus, screening method, or dated lexicon record. | Tier 4 contains an unresolved `{{LEXICON}}` marker. Live collection is blocked unless a registry is supplied; unverified terms cannot support H4. | Blocking H4 |
| B-006 | 2026-10-09 | Model registry | No exact provider model IDs, immutable versions, or two audited independent open-model families were selected. | No live pilot schedule was executed. CLI requires versions and checks pilot roles/family IDs, but a reviewer must verify model-family independence. | Blocking |
| B-007 | 2026-10-09 | Auditor/control | No independent Auditor, human-written baseline, consent/rater packet, or reliability ratings were supplied. | No hash-bound pre-run `PROTOCOL_CLEARED` file exists, so the N=5 pilot is blocked. Post-pilot `CLEARED` is separately required for N>5. C4/C5 analysis is not fabricated. | Blocking |
| E-001 | 2026-10-09 | Engineering test | Initial `plan` CLI invocation exposed a missing argparse option for `--skip-unresolved-lexicon`. | Added the option and reran the plan command successfully; generated a two-replicate UNPRIMED schedule with no model calls. | Resolved |
| E-002 | 2026-10-09 | Engineering test | Initial offline smoke-test path used a deterministic fixture response. | Every row is explicitly tagged `synthetic_fixture=true`, `evidence_eligible=false`; test files live only in a temporary directory and are removed. This text is never counted as research data. | Contained |
| E-003 | 2026-10-09 | Lint | Repository-wide Ruff 0.16.10 check reported 624 existing findings outside this new harness. | No unrelated application files were modified. `ruff check research/gpvp` and scoped format check pass. | Scoped checks pass; repo-wide finding remains |
| E-004 | 2026-10-09 | Test | Initial H6 lock-in unit test supplied only four outputs, too few for the preregistered two consecutive transitions beginning at step 4. | Corrected the test fixture to five outputs; implementation and test now pass. This was not a model-chain result. | Resolved |

## Research-behavior observations

- Live model responses: **0**.
- Observed confabulations: **none observed (no model was queried)**.
- Observed prompt leakage: **0 in the static prompt validator**; this does not test model behavior.
- Memoryless iteration failures/captures: **not observed; no chain ran**.
- Failed provider requests: **0; no inference requests were attempted**.
- Raw experimental data files: **none**.
