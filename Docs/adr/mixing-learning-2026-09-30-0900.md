# ADR: publish ML-2026-09-30-0900 as source-grounded search evidence

## Context

The 09:00 search slot found two new preprints and two unique official educational videos. Reading depth differs materially: selected SAGE full-text sections were available, YuE2 was inspected only through metadata/abstract, and both YouTube pages reported captions unavailable. The shared queue must grow without turning metadata into technique claims or learned-representation metrics into mixing rules.

## Options considered

1. Add every search hit as fully studied evidence.
2. Add only sources with complete end-to-end full-text review and discard the rest.
3. Add stable cards with explicit reading depth, deduplicate known items, preserve metadata-only candidates in the queue, and derive only one evaluation-scoped candidate rule from actually read SAGE sections.

## Decision

Use option 3. Publish one selected-section scientific card, one abstract-only scientific card, two metadata-only video cards, one duplicate removal, four limited Knowledge Cards, one evaluation-protocol candidate and one unrun experiment plan.

## Why this won

It preserves fresh discovery and stable IDs while keeping the evidence boundary auditable. It also records a useful negative result: captions were unavailable, so no video parameters or A/B conclusions can be claimed. The SAGE evidence is useful for evaluation safety but does not justify automatic mixing behavior.

## Rejected alternatives

- Option 1 was rejected because it would mislabel metadata and abstracts as read content and invent video conclusions.
- Option 2 was rejected because it would lose a valuable priority queue and make future deep-review deduplication harder.
- Runtime rule insertion was rejected because the new rule is an unvalidated evaluation candidate with `auto_apply:false`.

## Implementation plan

- Add dated Markdown report and JSON patch under `Docs/mixing_learning_updates/`.
- Add this ADR under `Docs/adr/`.
- Update the shared index with counts, evidence depth and next experiment.
- Do not modify production code, DSP, active rules, audio, models or datasets.

## Test plan

- Parse the JSON patch.
- Verify report, patch and ADR exist on the branch.
- Review the PR diff for only the four documentation files.
- Confirm all rules are `auto_apply:false` and experiments `not_run`.
- Merge and verify the files on `master`.

## Risks and rollback

Risk: downstream readers may overgeneralize autoencoder or generation results into mix-quality claims. Mitigation: the report marks scope, reading depth, candidate status and explicit protected measurements. Rollback is a documentation-only revert of the merge commit; no runtime or audio state is affected.
