# Brief: M123 Instrument Space decision integration

## Goal

Turn the existing Canvas-inspired, read-only Instrument Space into a real
decision-evidence layer for Automixer. The layer must detect spatial,
spectral and level conflicts, expose deterministic evidence and feed that
evidence into the existing correction-review pipeline for every supported
console without creating a new console-write path.

## Required outcome

- Build deterministic pair relations from measured pan, spectral centroid and
  channel level.
- Produce explainable conflict and drift candidates with provenance,
  confidence, missing-data blockers and stable identifiers.
- Bind the decision support to Product Layer snapshots and soundcheck proposal
  review metadata.
- Keep WING and dLive transport-independent: the same decision evidence may
  inform both, while each action family still needs its existing approval,
  readback, rollback, emergency-stop and hardware proof.
- Cover the vertical slice with unit, integration, dependency-boundary and
  frontend contract tests.

## Non-goals

- No unattended hardware contact or mixer writes.
- No conversion of a visual drag/position into a correction.
- No claim that spectral centroid is a physical elevation or a sufficient EQ
  diagnosis.
- No live-authority promotion for WING or dLive.
- No replacement of ERB masking, mono/phase, headroom, source-identity or
  Safety Kernel evidence.

## Completion boundary

M123 software is complete when deterministic Instrument Space decision support
is visible in Product Layer, safely annotates soundcheck proposals, all scoped
tests pass, and the repository records that hardware promotion remains a
separate external gate.
