# ADR — Mixing Learning search 2026-10-13 15:00 MSK

- **Status:** accepted for knowledge-base documentation; no automatic processing authority
- **Decision ID:** `ADR-ML-2026-10-13-1500`
- **Context:** follow-up to `ML-2026-10-13-1200`

## Decision

Treat extrema/keyframe time stretching as a creative/embedded candidate only. It must not become a transparent vocal or full-mix timing default. Any candidate stretcher must first pass a bypass-versus-1.0x representation null on project-owned material; only a passing pinned engine may proceed to a one-factor non-unity A/B.

Keep the three newly indexed mastering videos at `queued_source_no_transcript`. Metadata and official companion-page titles do not authorize importing video-only settings or claims.

## Evidence boundary

The DAFx26 paper reports low operation-count estimates and some transient preservation, but also material-dependent spectral loss, repeated-segment failure modes, weak singing-voice preference and no inferential statistics for its 15-listener test. The Capicola repository is public and AGPL-3.0 at observed commit `8c64e855fac64054fdb38451db00866ea682d0d8`, but was not built or reproduced.

## Consequences

- Add one Source Card, one Artifact Audit, five Knowledge Cards, one bounded rule and one blocked experiment.
- Add three metadata-only Video Source Cards; remove three rediscovered duplicates.
- Promote no multitrack/reference and no video technique.
- Preserve the existing executable queue; `EXP-ML-PENA-AB15-LISTEN-20261010-1800` remains next.
- No audio, DSP, runtime, model or repository production code is changed; `auto_apply:false`.
