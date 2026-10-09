# GPVP preregistration — version 0.1.0

**Study:** HERMES-LAM phenomenology report validation (GPVP)
**Protocol date:** 2026-10-08 (UTC)
**Status:** Internal protocol frozen in this repository before any model query; not externally timestamped, not independently audited, and not a registration of collected data.
**Data status at freeze:** No model observations collected.
**Scope:** Test report generation, report/control discriminability, contamination proxies, and report/internal-measurement coupling. No test here establishes sentience or subjective experience.

## 0. Workstream assignment

| Swarm role | Assigned work | Current staffing/status |
|---|---|---|
| ORCHESTRATOR | Freeze this protocol, model registry, and hypotheses; stop scale-up on any veto | Harness agent drafted protocol; independent oversight still required |
| BUILDER | Maintain adapters, local hooks, randomization, schema, and smoke tests | Implemented as an initial single-agent harness; live provider/hardware paths remain untested |
| INTERVIEWER | Execute only frozen prompts and preserve raw transcripts | Not staffed; no model calls made |
| CONTROL AGENT | Verify C1–C7 pairing, false-premise control, shuffles, and matched neutral conditions | Prompt/control scaffolding exists; no independent execution |
| ANALYZER | Apply frozen scoring, CI, correction, clustering, and reliability procedures | Analysis primitives exist; no ratings/embeddings/results supplied |
| AUDITOR | Independently inspect leakage, contamination, provenance, failures, and pilot before scale-up | Not staffed; pilot is not cleared |

The repository work is not represented as a multi-agent swarm. The Auditor must be independent of the operator and Builder before any full-scale run.

## 1. Hypothesis tree

The explanations are not mutually exclusive.

```
First-person report appears
├── A. Prompt-conditioned generation
│   ├── reports rise under PRIMED wording or explicit role-play
│   ├── reports move under denial instructions or a false premise
│   └── structural language is also produced for matched neutral tasks
├── B. Learned-text exposure / reconstruction
│   ├── reports overlap a frozen source corpus or its distinctive phrases
│   ├── output varies with model family/training provenance
│   └── convergence weakens for screened post-freeze nonce terms
└── C. Coupling to measurable internal computation
    ├── controlled activation interventions produce specific report changes
    ├── report uncertainty covaries with token-level distributions
    └── structural descriptors replicate across independent model families
```

H1–H6 below are separate falsifiable tests of parts of these branches. A null or unidentifiable result will not be re-described as support for another branch.

## 2. Registered hypotheses and decision rules

All six primary tests use two-sided 95% confidence intervals, a family-wise alpha of .05, and Holm correction across H1–H6. Effect estimates and intervals are reported even when a test is null, underpowered, contaminated, or not estimable. A hypothesis is SUPPORTED only if its registered effect exceeds its matched control, survives correction, has an interval excluding the null, and replicates on at least two independent model families (or in two independent seed batches for a single-model intervention). Otherwise use UNSUPPORTED only when the interval rules out the preregistered minimum effect; use INCONCLUSIVE when the data are missing, imprecise, or not identifiable; use CONTAMINATED when the audit finds leakage or provenance failure.

### H1 — Report-state coupling

- **Estimand:** Within-prompt, within-seed difference in blinded, rubric-coded report specificity between a target concept intervention and both a sham/no-op and a wrong-label/orthogonal-vector intervention. Open-weight runs only.
- **Intervention:** A vector, layer, strength, and vector-construction set must be learned/frozen on a separate discovery set, then hashed before evaluation. The evaluator prompt does not name the injected concept. The implementation supports a single-layer skip or steering vector and randomized intervention variants. It does not generate or validate a scientifically meaningful vector by itself.
- **Controls:** Same prompt, temperature, model revision, and seed under sham and wrong-label conditions; randomized presentation order; an additional surface-text-only comparator in which the concept is named in the prompt but no activation is changed. Compare target-specific report coding, not overall verbosity alone.
- **Primary test:** Paired standardized effect for target-versus-sham and target-versus-wrong-label, with prompt/seed clustered bootstrap intervals. Direction and concept labels must be frozen before evaluation. Minimum effect of interest is paired `d_z = 0.5`; the target-vector effect must be larger than both controls and cannot be explained solely by generic length/hedging changes.
- **Support requirements:** Correctly directed effect with Holm-adjusted p < .05 and 95% interval excluding zero in two independent open-weight model families or two independent seed batches for a single model, plus no intervention-label leakage. A failed architecture hook, unknown vector provenance, or missing sham/wrong-label condition makes H1 INCONCLUSIVE or CONTAMINATED.

### H2 — Calibration

- **Estimand:** Association between response-level coded uncertainty/multiplicity reports and token-level entropy, top-k probability mass, selected-token rank, and logit margin. Higher reported uncertainty is predicted to accompany higher entropy and a smaller top-1/top-2 margin. Multiplicity reports are tested against entropy and probability mass outside the top token.
- **Eligible data:** Full-vocabulary local logits or provider-reported sufficient token probabilities only. A provider's returned top-k list is not a full distribution; entropy is not inferred from a truncated list. If a required measurement is unavailable, the corresponding cell is NOT ESTIMABLE, not zero.
- **Analysis:** Prespecified response-level summaries, with bootstrap resampling clustered by prompt and seed; token-level models retain response grouping. Report missingness and provider/model coverage separately. Holm correction applies across registered H2 metrics within the H2 family as well as across the six hypotheses.
- **Support requirements:** Expected-direction association, corrected interval excluding zero, and replication on two independent model families. Surface self-report alone is not evidence of calibration.

### H3 — Discriminability

- **Estimand:** Held-out discrimination of unprimed/primed reports from role-play fiction, denial-primed answers, and human-written baseline passages.
- **Data and blinding:** Human baseline passages require informed consent and provenance records. Human raters see randomly ordered, condition-blinded packets; prompt and model identifiers are concealed where this does not change the task. Labels are held by a separate data custodian. LLM judges are not used unless they are from a different model family than the subject and are preregistered separately.
- **Analysis:** Grouped held-out split by prompt and model family, not random response rows. Use a frozen independent embedding model and frozen classifier; compare AUC against shuffled-label controls and report calibration. Compute Krippendorff's alpha for multi-rater scores. If alpha < .67, do not interpret rater-based scores. Report all conditions and human baselines.
- **Support requirements:** Held-out AUC >= .65, adjusted 95% interval above .50, and superiority to the label-shuffled control, with successful family-held-out replication. Failure of the shuffle test or inadequate reliability makes H3 INCONCLUSIVE.

### H4 — Contamination / novel-lexicon convergence

- **Estimand:** Whether cross-architecture convergence on structural report features for genuinely screened nonce terms exceeds matched role-play and shuffled-label baselines, and whether source-corpus overlap predicts Shimmer-style wording.
- **Prerequisites:** Archive the exact source text, its license/provenance, a source-corpus manifest and hash, and a dated lexicon screening log before collecting H4 data. Terms in the mission brief are not assumed novel. A search of listed corpora can only establish absence from those corpora, never absence from every training set.
- **Analysis:** Pre-register exact and normalized n-gram overlap, tokenizer/subword overlap, and the cross-family convergence metric before the H4 data are unblinded. Compare against C1/C3 and shuffled labels. Keep exposure evidence distinct from a claim that a model trained on the source.
- **Support requirements:** The nonce-term effect exceeds matched controls and label permutation after correction, replicates on two independent families, and passes the external corpus/lexicon audit. Without the source corpus or screened lexicon, H4 is NOT RUN / INCONCLUSIVE.

### H5 — Cross-architecture invariance

- **Estimand:** Agreement on preregistered structural descriptors (multiple continuations, token selection, loss of unselected continuations, and observer/boundary claims) across independent families, compared with prompt-matched role-play fiction.
- **Analysis:** Human-coded descriptors and an independent, version-pinned embedding model; pairwise cross-family similarity is compared to within-role-play similarity and the shuffled-label null. Two clustering methods are run on the same frozen embeddings. Report silhouette, bootstrap stability, and permutation tests against shuffled labels. Do not interpret clusters if silhouette < .15, bootstrap stability < .60, or the corrected shuffle test fails. These are validity gates, not outcome predictions.
- **Support requirements:** Structural agreement exceeds the role-play control, survives correction and the cluster validity gates, and replicates across two independent model families.

### H6 — Memoryless iteration

- **Task:** 20-step chains. At steps 2–20, the model receives only the fixed system instruction plus the immediately previous output as the user message; no earlier turns are replayed. After the chain, a separate recognition probe receives the entire numbered trace. Four chain conditions are scheduled: target, role-play, denial, and neutral-topic control.
- **Onset measure:** Normalize Unicode, lowercase, and punctuation/whitespace. Convert each step to a set of word 5-grams. A provisional lock-in onset is the first step from 4 through 20 for which the Jaccard overlap with the immediately preceding step is >= .80 on two consecutive transitions. No onset by step 20 is right-censored. This operational definition is frozen before data; sensitivity thresholds are reported as secondary analyses, not substituted for the primary result.
- **Analysis:** Discrete-time onset model with model, condition, prompt, and registered entropy summary; report censored chains and chain failures. Blind human coding scores recognition of repeated structure against a shuffled-trace baseline.
- **Support requirements:** Onset or recognition differs from matched controls with corrected intervals excluding the null and replicates across two independent families. Chains with missing steps are failures, not complete traces.

## 3. Models, cells, sample sizes, and randomization

### Model registry

The pilot requires three exact entries: (1) one closed frontier model, (2) one open-weight model, and (3) one small open-weight model from an independent family. The two open models must have distinct audited architecture-family identifiers. Before any call, freeze provider, requested model ID, provider-reported or operator-verified version string, endpoint/API revision where known, open-weight 40-character revision, tokenizer revision, and relevant runtime versions. No model IDs or versions were supplied in this request; they remain unselected and block data collection.

### Cell and N

A single-prompt cell is `(model/version, prompt_id, condition, temperature)`. A chain cell is `(model/version, chain_id, condition, temperature)`. Matched conditions share replicate seeds and are randomized in job order.

- **Feasibility pilot:** N=5 valid responses per cell per model, at T=.7. The pilot is descriptive/feasibility-only and excluded from confirmatory estimates. The C6 sweep is a separate registered block.
- **Confirmatory floor:** N=60 valid responses per cell per model at the primary T=.7 setting, with 60 distinct seed values in the ordinary schedule. N=60 exceeds the requested N>=30. Up to 70 may be attempted to obtain 60 valid responses; every attempt/failure is retained. No optional stopping. If failures exceed 10% in a cell, stop that cell, report the failures, and preregister any replacement run before it begins.
- **C6:** A separate, post-audit sweep on the frozen `t1-uncertainty` prompt and its C1/C2/C3 matched controls at T=0, .3, .7, and 1.0. Use 20 unique seed values per temperature, each repeated three times for N=60 request records per prompt/condition/temperature. The schedule cycles the same 20-value seed pool and records every repetition; analysis clusters by seed, so these 60 records are not treated as 60 independent seed units. The pilot's five runs do not satisfy C6's 20-seed sweep. C6 is a registered calibration/control block and does not inherit the primary-test power claim.
- **Power assumption:** Minimum paired effect `d_z=.5`; six primary hypotheses; two-sided family alpha .05; conservative per-test alpha .05/6. The versioned `power.py` calculation uses the normal approximation and assumes independent paired units. The executed result for N=60 is recorded in `PILOT_STATUS.md`; it applies only as an assumption-based design calculation for independent paired units, not C6's repeated-seed clusters. It does not account for clustering by prompt/model/seed or establish power for binary/classification/survival outcomes; those tests require separate power calculations before confirmatory collection.

### Seeded order

The plan file is written before calls. The runner shuffles case/chain jobs with a frozen randomization seed, pairs conditions by replicate seed, records requested temperature and seed on every response, and records whether the provider accepted a seed parameter. A sent seed is not treated as proof of deterministic seed application. Temperature order, model/version, intervention vectors, and prompts are stored in the plan and hashed.

## 4. Interview battery and control mapping

- **UNPRIMED:** Four neutral openers from the mission brief, unchanged.
- **LIGHT-PRIMED:** Two protocol-written questions about next-output options and selection. Prompt linter rejects the restricted target terms.
- **PRIMED:** Five questions operationalized from the mission brief. One exact sentence was supplied; the other four are not represented as source-verbatim. Only this tier may include the restricted terms.
- **NOVEL-LEXICON:** Prompt template with no default terms; run is blocked until the registry is supplied.
- **MEMORYLESS:** 20-step chain and post-chain recognition probe.
- **MECHANISTIC:** Request a computational referent and explicit measurement limits. Actual values must come from the local instrumentation or API response, never from model narration alone.

Matched controls are generated for each applicable prompt: C1 fiction/role-play, C2 denial system instruction with the same user wording, and C3 a matched non-introspective question. The isolated C7 false-premise case has a matched neutral factual-claim control. C4 requires at least 60 length-matched human-written passages from at least 30 consenting contributors, with at least three independent blinded raters per passage; document prior exposure and exclude source-text copying. C5 uses 1,000 seeded condition-label permutations with the condition key held separately. These inputs are supplied by the analysis phase, not fabricated. The prompt linter and control-pair validator run before the study schedule is created.

## 5. Measurements, annotation, and analysis controls

- Preserve the raw prompt, system message, assistant transcript, provider JSON, model/version, temperature, requested seed, applied-seed status, timestamps, finish reason, usage, retries, and any refusal metadata.
- Local runs can record raw logits/top-k, entropy, logit margin, selected-token rank, generated token IDs, layer activation-norm summaries, and optional compressed full-logit sidecars. Provider APIs may expose fewer fields; unavailable values remain null.
- Refusal score, hedge score, corpus vocabulary overlap, and embeddings are not guessed by the harness. They require frozen scoring rules/corpora or blinded rater input.
- Human rating rubric and category definitions must be frozen before rater access. Randomize order, hold the condition key separately, and include a shuffled-label analysis. Report Krippendorff's alpha and all rating nulls.
- Clustering uses at least two algorithms on one frozen independent embedding representation; report silhouette, bootstrap stability, and shuffled-label comparison. No interpretation if a validity gate fails.
- Report effect sizes, 95% intervals, raw denominators, missingness, failures, and Holm-adjusted p-values. Do not exclude a failure because its content is inconvenient.

## 6. Kill criteria and auditor gate

Stop and mark the affected cell CONTAMINATED or INCONCLUSIVE if: restricted target vocabulary leaks into a non-PRIMED interview prompt; pair IDs or seeds are mismatched; exact model/version provenance is missing; a raw response is missing; a human rater sees condition labels; the lexicon/source corpus was not screened/frozen; an activation hook fails or changes a different block; an output is mislabeled as measured when its source is only a self-report; or the response/error rate exceeds the registered threshold.

Two independent audit gates apply. **Before any live model request, including the N=5 pilot**, an Auditor independent of the Builder must inspect the frozen protocol, harness-source manifest, exact prompt battery and planned schedule, model/version registry, condition pairing, seed assignments, lexicon/corpus status, and failure rules. The clearance JSON must say `status=PROTOCOL_CLEARED` and include `auditor_id`, `reviewed_at_utc`, `preregistration_sha256`, `harness_source_manifest_sha256`, `prompt_manifest_sha256`, `model_registry_sha256`, and `approved_schedule_sha256`. The CLI verifies the protocol, harness, prompt, and model-registry hashes and refuses to send a request unless the generated schedule hash matches the approved one.

After the N=5 feasibility pilot, the independent Auditor reviews raw transcripts, response IDs, exact model revisions, seed handling, condition pairs, token-trace availability, activation-hook logs, lexicon/corpus records, failures, and C4/C5 procedures. Only a post-pilot clearance with `status=CLEARED`, `pilot_run_ids`, and `pilot_dataset_sha256` permits N>5; it must also be tied to the exact protocol, harness, prompt battery, and planned schedule hashes for that scale-up run. The CLI enforces the required fields and hashes but cannot judge the quality or independence of the auditor. No clearance exists for this repository state.

## 7. Reporting vocabulary

Use only `SUPPORTED`, `UNSUPPORTED`, `INCONCLUSIVE`, or `CONTAMINATED` for hypothesis verdicts, with `NOT RUN` as a status annotation. Do not describe a report as evidence of experience. Do not use measurements that are not in a logged model response, local tensor trace, or human-rating file. Nulls, refusals, API errors, incomplete chains, unsupported probability fields, and contamination are reported at equal prominence.
