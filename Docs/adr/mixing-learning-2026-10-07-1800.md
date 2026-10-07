# ADR: Mixing Learning deep review — 2026-10-07 18:00 MSK

## Context

The preceding search queued JASA DOI `10.1121/10.0020269` at abstract depth and conservatively described an elevated LAR preference among hearing-impaired listeners. A withdrawn masking preprint remained excluded. The deep review had to verify methods, statistics, limitations, subsequent evidence and any transferable rule without modifying audio or DSP.

## Options considered

1. Promote the reported group means into fixed accessibility presets.
2. Keep only the abstract-level summary and defer implementation details.
3. Read the full paper and its direct performance follow-up, separate preference from performance, and retain only listener-specific offline candidates.

## Decision

Choose option 3. Upgrade the JASA card to full-text depth, add the 2025 PLOS ONE follow-up, correct the abstract-level overgeneralization, and add two blocked offline A/B plans. Keep the withdrawn arXiv source excluded; a thesis duplicate is lineage, not independent evidence.

## Why this won

The full JASA results do not support a universal fixed LAR or spectral setting, and the follow-up found no main detection benefit from EQ-transform plus a Bass-specific regression. Personalization, target-class reporting and device-state documentation are the smallest evidence-compatible policy.

## Rejected alternatives

- Fixed `+2 dB` vocal preset: rejected because it is only a rounded study-group mean and experiment-1 zero-referenced means were not generally significant.
- Automatic spectral sparsity maximization: rejected because transform strength did not produce a general detection benefit and descriptors were not causally validated.
- Restoring the withdrawn masking paper through a thesis chapter: rejected because the material is the same lineage and the official withdrawal remains unresolved.

## Implementation plan

Documentation only: update Source/Knowledge/Rule/Experiment cards and the queue. Do not change audio, DSP, runtime, models or production defaults.

## Test plan

Validate JSON syntax and cross-check stable IDs, DOI deduplication, statuses, `auto_apply:false`, source-reading depth and index links. Repository code tests are not required for documentation-only changes.

## Risks and rollback

Risk: population means may still be mistaken for prescriptions. Mitigation: numeric origins, target-listener prerequisite, hearing-aid-state field, loudness-matched copies and rollback criteria are explicit. Rollback is removal of this documentation-only update.
