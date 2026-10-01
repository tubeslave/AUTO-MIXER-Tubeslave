# ADR: Mixing Learning search 2026-10-01 15:00 MSK

## Context

The 15:00 search needed to find non-duplicate evidence after a same-day deep review. Most recent automatic-mixing and perceptual-metric hits were already indexed. One unindexed DAFx-26 demo described a compiler from differentiable audio graphs to real-time FAUST, while three unindexed rock-drum videos had unequal content access.

## Options considered

1. Promote author-reported ADAC equivalence and speed directly into the production DSP workflow.
2. Ignore the paper because it does not prescribe a mix sound.
3. Store it as a bounded research-tool card and candidate export-verification gate, with no execution or production change; preserve exact video reading depths.

## Decision

Choose option 3. Record ADAC as an LTI deployment-verification source, not a mixing-quality rule. Treat its residual, speed and test counts as author-reported until independently reproduced. Do not apply its guarantees to nonlinear/time-varying processors. Mark the Scheps video as partial secondary-summary review and the Murphy/Herrmann videos as metadata-only.

## Why this won

It preserves a concrete, reproducible safety pattern—semantic equivalence, emitted-parameter stability and real-time deadline checks—without converting engineering validation into an audible-quality claim or overstating incomplete video access.

## Rejected alternatives

- Automatic production integration: outside research scope and unsupported by local reproduction.
- Treating the paper's `7e-5` residual as the project threshold: it is source-specific and not a negotiated safety limit.
- Extracting video settings from titles/descriptions or a secondary summary: content depth is insufficient.

## Implementation plan

Publish one source card, three Video Source Cards, four Knowledge Cards, one limited rule candidate and one `not_run` auxiliary validation plan. Carry the existing practical drum-room experiment as priority. Make no audio, dependency, model, DSP or runtime change.

## Test plan

- Validate JSON syntax and cross-file IDs.
- Confirm duplicate arXiv/DAFx/video IDs are not re-added.
- Confirm the PR changes only the report, JSON patch, ADR and shared index.
- Require repository `Tests` and `Stem Offline Test` workflows to pass before merge.

## Risks and rollback

Risk is documentary overclaim, especially confusing LTI export fidelity with mix quality or a secondary video summary with a transcript. Roll back by reverting the documentation commit. No executable or mixer state is touched.
