# ADR: Mixing Learning deep review 2026-10-01 12:00 MSK

## Context

The 09:00 search added a DAFx-2026 paper as support for multi-axis transient evaluation. The source needed a full audit before its statistical and perceptual claims could influence experiment design.

## Options considered

1. Promote the paper's DDM result as direct support for a mix-bus processing rule.
2. Keep only the general multi-metric lesson without revisiting the listening design.
3. Audit the full method, repeated-measures structure, interaction, source rights and the cited MUSHRA screening rule, then narrow the rule and pre-register a bounded A/B.

## Decision

Choose option 3. Store DDM only as an evaluation-method source. Treat five retained listeners as the independent listener count; do not convert 30 repeated cell ratings into `n=30`. Do not store “almost indistinguishable” as equivalence because the reference-versus-DDM comparison remained significant and no equivalence test was reported. Require section-stratified reporting when model/material interaction is plausible. Do not ingest linked audio until its own licence is explicit.

## Why this won

It preserves the useful insight—spectral and time-domain residual metrics can rank systems differently—while avoiding pseudo-replication, overgeneralization from isolated acoustic sounds and accidental transfer of source-specific analysis thresholds into mixing settings.

## Rejected alternatives

- Direct DDM or peak-threshold adoption: unrelated to the current mixer and unsupported on rock mixtures.
- Treating non-significant technique effect for DDM as invariance: low retained `n` and a significant overall interaction make that too strong.
- Treating the article's CC BY notice as companion-audio permission: the sample/audio rights are not established by the article licence.

## Implementation plan

Publish one updated DAFx source card, one official ITU comparator, five knowledge cards, one limited reporting rule, one narrowing revision and one `not_run` attack-sensitivity A/B. Make no production-code or DSP change.

## Test plan

- Validate JSON syntax and cross-file IDs.
- Confirm the PR changes only the report, JSON patch, ADR and shared index.
- Require repository `Tests` and `Stem Offline Test` workflows to pass before merge.

## Risks and rollback

Risk is documentary overclaim, not runtime behavior. Roll back by reverting this documentation commit. No mixer state, audio asset, dependency, model or executable code is touched.
