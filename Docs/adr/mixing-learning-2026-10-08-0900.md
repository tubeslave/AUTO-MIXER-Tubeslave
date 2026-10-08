# ADR: Mixing Learning search — 2026-10-08 09:00 MSK

## Context

The search slot rotated to mix-bus/mastering loudness and crest-factor preservation. The shared base already contains extensive scientific and video coverage, so new findings had to be reconciled by stable ID, normalized title and version family before adding cards.

## Options considered

1. Re-add prominent search hits as new evidence.
2. Treat vendor ranges, YouTube titles and chapter labels as ready-to-use mastering guidance.
3. Deduplicate existing scientific families, fully read the accessible official tutorial, queue transcript-unavailable videos at metadata depth, and add only a constrained measurement rule and unrun A/B plan.

## Decision

Choose option 3. Add one official vendor Source Card, one Knowledge Card, one candidate rule, three metadata-only Video Source Cards, one rights-recheck Reference Card and one unrun experiment. Keep the existing next executable experiment unchanged.

## Why this won

The scientific search hits were already present, and no independent new study survived deduplication. The iZotope article is useful for defining measurement context but does not validate a universal crest-factor target. The videos cannot support technique extraction without permitted content. Cambridge-MT catalogue metadata establishes availability, not project-specific rights or completed audio analysis.

## Rejected alternatives

- Numeric crest targets: rejected because the article's ranges are illustrative, programme-dependent vendor guidance.
- Video-derived settings: rejected because no transcript/content was read.
- Automatic multitrack ingestion: rejected because no download was needed and the candidate still needs a rights/host recheck.
- Promoting the new A/B ahead of the accumulated queue: rejected because `EXP-ML-20261001-1800-01` remains executable and higher priority.

## Implementation plan

Documentation only: add/update cards, queue and index. Do not change audio, DSP, runtime, models or production defaults.

## Test plan

Validate JSON syntax; verify stable IDs and exact YouTube publication metadata; check `metadata_only`, `queued_source`, `not_run` and `auto_apply:false` statuses; confirm index links and deduplication lineage.

## Risks and rollback

Risk: users may mistake illustrative crest-factor ranges or video descriptions for validated targets. Mitigation: no numeric range is promoted, the measurement contract is explicit, video claims are unaccepted, and the A/B requires loudness matching plus subjective rollback gates. Rollback is removal of this documentation-only update.
