# ADR — Mixing Learning search 2026-10-07 09:00 MSK

## Context

The scheduled search rotation covered loudness, crest, mix-bus processing and loud-rock mastering. The shared base already contained the strongest newly surfaced DAFx-2026 compressor-model paper and three high-value YouTube IDs. A 2014 DAFx paper on statistical offline compressor parameter estimation was absent, and one queued rock-compressor video now exposed a readable transcript.

## Options considered

1. Add every rediscovered item as a new card.
2. Add only post-2026 publications.
3. Reconcile by stable ID/title/version, fill the older evidence gap, upgrade the existing video card and preserve metadata-only boundaries.

## Decision

Choose option 3. Add `SRC-ML-20261007-0900-01`, upgrade `VID-ML-20260914-1500-Q3`, add two metadata-only video candidates and one rights-recheck multitrack candidate, and create only one bounded Knowledge Card, candidate rule and unrun experiment.

## Why this won

It preserves provenance and avoids false independence. The full paper supplies a useful parameter-search mechanism but weak perceptual validation, while the complete transcript supports procedure facts but not the audible compressor ranking. The resulting rule therefore remains an initializer hypothesis rather than an automatic mastering policy.

## Rejected alternatives

- Duplicate cards would inflate evidence counts and fragment version history.
- Restricting discovery to recent publication dates would leave a relevant methodological gap.
- Promoting approximate gain reduction or long-term moment matching to acceptance criteria would confuse process controls with loudness-matched perceptual evidence.

## Implementation plan

- Publish the Markdown report and JSON patch under `Docs/mixing_learning_updates/`.
- Add this ADR under `Docs/adr/`.
- Append the update to the shared research index.
- Leave production rules, DSP, model code, audio and runtime registries unchanged.

## Test plan

- Parse the JSON patch.
- Verify report/JSON/ADR presence on the publication branch and after merge.
- Verify the index link resolves.
- Verify all new rules have `auto_apply:false` and experiments have `not_run` status.
- No production tests are required because production code is unchanged.

## Risks and rollback

Risk: readers may mistake a five-song simulation and informal listening statement for a validated mastering rule, or interpret the presenter's compressor ranking as controlled evidence. Mitigation: explicit reading-depth, listening and transfer limits; no preset range; no auto-application. Rollback is removal of the four documentation artifacts; production behavior is unaffected.
