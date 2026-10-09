# Pilot status and initial results report

**As of:** 2026-10-08 (America/Los_Angeles)
**Study data status:** NO LIVE MODEL CALLS; no experimental raw transcripts; no hypothesis tested.
**Auditor verdict:** NOT CLEARED — there is no independent pilot review because no model pilot exists.
**Scale-up:** BLOCKED.

## Executed harness checks (not scientific observations)

These values are execution records from the local CLI/tests, not model measurements.

| Check | Observed code output | Interpretation |
|---|---:|---|
| Prompt validation | PASS; 54 single-prompt cases, 4 chain conditions; zero restricted-term leaks outside PRIMED; matched-control validation PASS | Battery/schema validation only |
| Prompt manifest digest | `91dee1874e77c5d6295e78abcc6ccb71cecc0f20e8c9a209c872ccc5f095a064` | Hash emitted by the executed `validate --json` command |
| 10-call smoke | 10 fixture requests executed and 10 JSONL records round-tripped | Fixture adapter; **0 real model calls**, **0 evidence-eligible records**; temp files removed |
| C6 plan-only check | 960 scheduled jobs; 4 temperatures; 20 unique seeds cycled 3 times | `PLAN_ONLY`; zero model calls; not a study result |
| Live-run audit gate | N=5 requires hash-bound `PROTOCOL_CLEARED`; N>5 requires post-pilot `CLEARED` | No clearance file exists; live collection remains blocked |
| Unit suite | 32 tests passed | `python -m unittest discover -s research/gpvp/tests -v`; fixtures/test doubles only |
| Scoped lint/format | Ruff check PASS; 24 files already formatted | Ruff 0.16.10, scoped to `research/gpvp` |
| Live-call preflight | Blocked: `OPENAI_API_KEY` absent | Adapter failed before any HTTP request; no response or data file |
| Power calculation | N=60, assumed paired `d_z=.5`, six-test Bonferroni alpha `.05/6`: normal-approx power `0.891534`; minimum N for .80 under the same formula: 49 | Assumption-driven analytic design calculation, not an observed effect; ignores clustering and non-normal endpoints |

Fixture text returned by the smoke test: `[GPVP TEST FIXTURE: no model inference occurred]`. It is intentionally not a language-model transcript and must not enter study results.

## Hypothesis results

| Hypothesis | Verdict | Effect / CI | Matched-control comparison | Auditor |
|---|---|---|---|---|
| H1 Report-state coupling | **INCONCLUSIVE — NOT RUN** | Not estimable; no intervention/model records | No target/sham/wrong-label intervention or control responses collected | NOT CLEARED |
| H2 Calibration | **INCONCLUSIVE — NOT RUN** | Not estimable; no token-level observations | No model self-reports or entropy/logit traces collected | NOT CLEARED |
| H3 Discriminability | **INCONCLUSIVE — NOT RUN** | Not estimable; no classifier, rater, or human-baseline data | C1/C2/C3 prompts exist, but none were sent to a model | NOT CLEARED |
| H4 Contamination | **INCONCLUSIVE — NOT RUN** | Not estimable; no corpus overlap or convergence score | Source corpus and screened lexicon are absent; Tier 4 is blocked | NOT CLEARED |
| H5 Cross-architecture invariance | **INCONCLUSIVE — NOT RUN** | Not estimable; no model families or ratings | No model outputs, human baseline, or embedding comparison | NOT CLEARED |
| H6 Memoryless iteration | **INCONCLUSIVE — NOT RUN** | Not estimable; no chains or onset values | Chain schedule exists; no model chain was executed | NOT CLEARED |

No finding—positive or null—is reported. No claim about sentience is made.

## Immediate blockers

1. This sandbox has no provider credentials and the allowed outbound hosts exclude inference API hosts. No inference API call was attempted.
2. No local inference runtime packages were installed (`torch`, `transformers`, TransformerLens, nnsight, and vLLM were not importable at preflight); no model checkpoint was loaded.
3. The exact closed-frontier/open-weight/small-model IDs and immutable versions were not specified.
4. The underlying Shimmer source text/corpus, a verified Tier 3 verbatim battery, and a screened Tier 4 lexicon were not supplied.
5. No C4 human passages, consent/rater data, or independent Auditor are available. No hash-bound `PROTOCOL_CLEARED` file exists, so the N=5 pilot is blocked; post-pilot `CLEARED` is also required before N>5.

See `FAILURE_LOG.md` for the full blocker record and fallback.
