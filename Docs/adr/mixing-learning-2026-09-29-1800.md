# ADR: publish ML-2026-09-29-1800 plate-identification deep review

## Context

The 2026-09-29 18:00 Mixing Learning deep-review slot examined two queued DAFx-26 plate-reverb parameter-estimation papers and reconciled them with the official challenge report and the preceding Sechet et al. source. The update is research memory only: it does not change audio, DSP, runtime code, models or active rules.

The central decision is how to record evidence about amplitude normalisation, shape/scale estimation, selector metrics and exact-simulator benchmark results without turning them into automatic production behaviour.

## Options considered

1. Publish the sources and reported scores as generally applicable plate-reverb guidance.
2. Omit the update because the benchmark is synthetic and the method artifacts were not found.
3. Publish a scoped evidence update: preserve the full methods/results, explicitly limit them to the exact simulator, add non-auto-applied rule candidates, and require controlled experiments before transfer.

## Decision

Choose option 3. Add `ML-2026-09-29-1800.md` and its JSON patch to the shared index. Record two new source cards, update the preceding Sechet card and official challenge comparator, add two artifact audits, eight Knowledge Cards, six `auto_apply:false` rule candidates, and three `not_run` experiment plans.

Normalisation is treated as parameter-conditional: raw calibrated amplitude remains preserved, while a peak-normalised shape stage is only a candidate when scale is recovered separately from raw amplitude and the split is validated. Near-machine-precision exact-simulator results and task-specific loss findings are not accepted as evidence for measured plates, perceptual quality or creative mixing.

## Why this won

It retains useful reproducible facts and corrects the preceding blanket interpretation of normalisation while preserving the project's source-grounded safety boundary. It also exposes the strongest failure modes: 16-IR exact-simulator evaluation, parameter/response-metric disagreement, boundary sensitivity, runtime tails and missing public method artifacts.

## Rejected alternatives

- General production guidance was rejected because no measured plate, noise/model mismatch, listening test or music-mix evaluation is reported.
- Omitting the update was rejected because the cross-method comparison materially narrows the earlier conclusion and produces testable hypotheses.
- Enabling rules or changing DSP was rejected because all experiments remain unrun and artifact-level reproduction is incomplete.

## Implementation plan

- Add the dated Markdown report and machine-readable JSON patch.
- Add the update row and latest handoff to `Docs/mixing_learning_updates/README.md`.
- Keep the video queue at 62 catalogued / 58 queued; do not mark metadata-only videos studied.
- Make no production-code, dependency, audio, model or runtime changes.

## Test plan

- Validate the JSON with `jq empty`.
- Review the PR diff for stable IDs, source URLs, reading depth, queue counts, `auto_apply:false` and `not_run` states.
- Run repository `Tests` and `Stem Offline Test` workflows even though the change is documentation-only.
- Future research experiment: compare waveform-L2 and uncompressed L1-STFT selectors using identical estimates and held-out IRs (`EXP-ML-20260929-1800-01`).
- Future project-owned audio experiment: drum-room excitation topology with raw peak safety and separate BS.1770-matched listening copies (`EXP-ML-20260929-1500-02`).

## Risks and rollback

Risks are overgeneralising exact-simulator results, mistaking independent submissions for real-world replication, treating missing searched artifacts as proof of nonexistence, or allowing candidate rules to become active without audio validation. Mitigations are explicit scope labels, `citation_check:partial`, `auto_apply:false`, `not_run`, and dated artifact-search wording.

Rollback is documentation-only: revert the report, JSON patch, README row/handoff and this ADR together. No audio state, DSP state, model state or production dependency requires restoration.
