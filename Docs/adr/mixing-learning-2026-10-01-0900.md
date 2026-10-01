# ADR: publish ML-2026-10-01-0900 with transient-evaluation boundaries

## Context

The 09:00 search found one unique open DAFx source and three unique topic-relevant YouTube videos. The paper provides useful but indirect evidence about evaluation of nonstationary instrument attacks. All new video pages reported captions unavailable. One scientific and three video rediscoveries were duplicates. A single allowed Scite probe after the stated reset still returned a paid-plan/free-trial requirement.

## Options considered

1. Treat all search results and video metadata as fully studied evidence.
2. Discard indirect research and all videos without transcripts.
3. Publish the read paper with explicit transfer limits, keep the three videos as metadata-only queue entries, record duplicate and access outcomes, and create only an evaluation-scoped candidate plus an unrun A/B plan.

## Decision

Use option 3. Add one source card, three queued video cards, four Knowledge Cards, one `auto_apply:false` evaluation rule candidate and one `not_run` experiment plan. Preserve the existing practical experiment priority.

## Why this won

It retains stable IDs and a useful methodological warning without claiming that isolated-sound resynthesis validates rock compression or mastering. It also makes the evidence denominator visible: only five listeners remained after screening. Metadata-only videos remain discoverable without fabricated technique extraction.

## Rejected alternatives

- Option 1 was rejected because titles, descriptions and abstracts do not establish techniques or parameter values.
- Option 2 was rejected because transparent queue entries and duplicate removals are useful project state.
- Runtime or DSP changes were rejected because no audio experiment ran and transfer confidence is low.
- Starting a Scite trial or subscription was rejected by the free-access constraint.

## Implementation plan

- Add the dated Markdown report and JSON patch under `Docs/mixing_learning_updates/`.
- Add this ADR under `Docs/adr/`.
- Update the shared index with evidence depth, queue counts, access status and next experiment.
- Do not modify production code, audio, DSP, dependencies, models or active rules.

## Test plan

- Parse the JSON patch.
- Verify stable IDs and queue arithmetic (`67→70`, `63→66`).
- Verify every candidate has `auto_apply:false` and every experiment is `not_run`.
- Review the PR changed-file list for documentation-only scope.
- Require `Tests` and `Stem Offline Test` to pass before merge.

## Risks and rollback

The main risk is overgeneralizing a five-listener isolated-instrument resynthesis study into mix-bus guidance. The report limits it to evaluation structure and labels the DSP transfer confidence low. Rollback is a documentation-only revert of the merge commit; no runtime or audio state is affected.
