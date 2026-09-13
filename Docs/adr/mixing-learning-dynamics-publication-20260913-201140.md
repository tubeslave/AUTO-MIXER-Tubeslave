# ADR: publish the dynamics/compression research package

## Context

The user explicitly requested: «Оформить в общей базе». The prepared update is ML-DYNAMICS-2026-09-13-201140. The current shared research index is on master in tubeslave/AUTO-MIXER-Tubeslave. AGENTS.md, CLAUDE.md, .cursorrules and the current index were read before publication work. The baseline is ca2067fa8863f1f3dca9c5f7ced7f2e25b3b4c8a.

## Options considered

1. Leave the study in temporary chat exports.
2. Publish complete structured evidence plus an editorial guide and provenance in the existing research directory.
3. Convert all hypotheses immediately into executable runtime rules.

## Decision

Choose option 2. Keep every value and ID from the original research JSON, changing only whitespace serialization. Publish a readable Russian guide, the original export-validation record and a publication manifest containing source and published hashes. Update the shared index and latest handoff without deleting earlier entries. Use a PR targeting master; verify its changed files and checks before merge, then read back published files.

## Why this won

The user obtains durable common research data, discoverable by agents through the existing index, without changing audio behavior. Source-reading depth, categorical confidence, numerical-origin labels and unrun experiment states remain intact. Published file hashes distinguish the reformatted JSON and editorial guide from the original exports.

## Rejected alternatives

Temporary exports do not satisfy shared publication. Immediate runtime conversion would imply validation and schema adaptation that have not occurred. Numeric probabilities must not be invented from qualitative confidence; old source/rule data and feedback must not be overwritten.

## Implementation plan

Documentation/data only under Docs/mixing_learning_updates and this ADR. Preserve 14 source cards, 8 Knowledge Cards, 8 candidate rules, 4 queued videos, one metadata-only benchmark card and 4 experiment plans. The package is not a new corpus of measured genre targets. No production code, default runtime registry, mixer configuration, workflow or audio files are changed.

## Test plan

Twenty-five local artifact checks passed, including JSON semantic equality to the original, unique IDs and cross-links, bounded-rule fields, candidate/auto-apply restrictions, no fabricated video evidence or genre targets, original export hashes, analytical arithmetic and uploaded JSON/guide Git blob equality. These are not audio tests. Full local pytest was not run: public GitHub retrieval from the container failed due DNS. CI for the exact PR head must be inspected separately; pending, skipped or absent tests must not be called passed. No branch protection will be overridden.

## Risks and rollback

Original JSON and export-validation include pre-publication status fields. The guide and publication manifest explain their historical scope. Publication does not prove another chat received the update, an agent ingested it, or runtime retrieval uses it. Complete cross-registry deduplication remains separate; known overlaps are linked rather than presented as newly discovered universal rules. All candidates remain auto_apply:false and experiments not_run. Revert only this documentation commit if needed; never rewrite branch history or alter runtime settings.

Review: publishing-assistant artifact and scope review only. Independent Kimi review and human audio acceptance are not claimed.
