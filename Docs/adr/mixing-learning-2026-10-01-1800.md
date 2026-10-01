# ADR: Mixing Learning deep review 2026-10-01 18:00 MSK

## Context

The preceding search added an ADAC compiler paper/repository and a Scheps drum-processing video with only a third-party summary. The deep review needed to determine whether the code evidence supports a deployment gate and whether first-party video content supports bounded rock-drum experiments.

## Options considered

1. Accept repository test-count, capability and stability claims as sufficient for a future export path; retain the secondary video summary.
2. Reject both sources entirely because neither was independently reproduced with project audio.
3. Preserve the useful evidence but narrow it: audit actual CI/integration/certificate/export code, reconcile open issues, read the first-party transcript only within the playable duration, and convert findings into non-automatic controlled tests.

## Decision

Choose option 3. ADAC stays an isolated research candidate, with a stricter project gate: clean no-skip integration CI, pinned toolchain and operator manifest, target-host tests, `certified-stable` only, and no `strict=False`. Upgrade the Scheps card to transcript-complete for the 8:33 playable excerpt, but discard transcript rows outside that duration and remove earlier secondary claims beyond it.

## Why this won

It retains a reproducible deployment-safety pattern and a concrete section-dynamics experiment without confusing author tests with independent reproduction, a sufficient LTI condition with universal stability, or a single engineer's settings with defaults.

## Rejected alternatives

- Direct integration: outside the research-only scope and contradicted by skipped integration CI, machine-specific paths and unresolved capability drift.
- Total rejection: would discard useful negative evidence and a well-bounded first-party mixing demonstration.
- Accepting all transcript-panel rows: the panel extended beyond the official 8:33 player duration, so timestamp integrity failed after 8:24.

## Implementation plan

Publish one source-card revision, one repository audit, one video-card upgrade, five Knowledge Cards, one narrowed and one new limited rule, and one loudness-matched `not_run` A/B plan. Make no audio, dependency, model, DSP or runtime change.

## Test plan

- Validate the JSON patch and cross-file IDs.
- Confirm accepted video timestamps do not exceed 8:33.
- Confirm all rules are `auto_apply:false` and experiments `not_run`.
- Confirm the PR changes only report, JSON patch, ADR and shared index.
- Require repository `Tests` and `Stem Offline Test` workflows to pass before merge.

## Risks and rollback

The remaining risks are documentary overclaim, transcript-panel inconsistency and treating author-maintained issues as independent validation. Roll back by reverting the documentation commit; no executable or mixer state is touched.
