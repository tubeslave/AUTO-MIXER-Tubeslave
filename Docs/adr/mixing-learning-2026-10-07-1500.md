# ADR — Mixing Learning search 2026-10-07 15:00 MSK

## Context

The search rotation targeted rhythm/lead-guitar placement and masking. The shared base already contained several rediscovered compressor/automatic-mixing papers and guitar videos. Two new official companion articles were readable, one JASA accessibility paper was absent, and an apparently relevant masking paper was officially withdrawn with unresolved IP ownership.

## Options considered

1. Treat cached withdrawn text, companion articles and their YouTube videos as equivalent full-content evidence.
2. Ignore all non-video text because transcripts were unavailable.
3. Preserve separate depth fields, exclude the withdrawn source from evidence, queue the JASA full text, and derive only bounded hypotheses from the fully read companion articles.

## Decision

Choose option 3. Add two literature cards with different admissibility states, two metadata-only video cards with fully read companion-article provenance, one rights-recheck multitrack candidate, three Knowledge Cards, two candidate rules and two `not_run` experiments.

## Why this won

It keeps content claims attached to what was actually read. The articles contain concrete, source-specific guitar placement and ambience techniques, while the videos' demonstrations and audio A/B were not inspected. The withdrawn paper's official status is itself useful evidence, but not evidence for its algorithm.

## Rejected alternatives

- Treating cached withdrawn content as open evidence would bypass the current official version and unresolved rights.
- Relabelling companion-article reading as video-transcript reading would overstate depth.
- Promoting panning, delay, reverb or gain-reduction numbers to defaults would confuse one production's arrangement with a validated rule.

## Implementation plan

- Publish the Markdown report and JSON patch under `Docs/mixing_learning_updates/`.
- Add this ADR under `Docs/adr/`.
- Append the update to the shared index.
- Leave audio, DSP, models, runtime rules and production code unchanged.

## Test plan

- Parse the JSON patch.
- Verify report/JSON/ADR and index links after merge.
- Verify both videos remain `content_studied:false`, rules use `auto_apply:false`, and experiments use `not_run`.
- Verify DOI, arXiv and YouTube IDs against the default branch to avoid duplicates.

## Risks and rollback

Risk: readers may treat companion-article numbers as universal settings or interpret a withdrawal notice as proof that all related AES lineage is invalid. Mitigation: source-specific range provenance, separate video/article depth, and exclusion limited to the unavailable withdrawn version. Rollback is removal of the documentation artifacts; production behaviour is unaffected.
