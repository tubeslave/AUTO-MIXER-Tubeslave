# WING Read-Only Artifact Root Whitespace Intake Guard

## Context

M001 is waiting for a real WING Rack read-only artifact. M095-M100 blocked
reserved roots, declared path-shape mismatches, non-canonical roots, copied
source summaries, path-like run ids and parent traversal in artifact roots. A
remaining file-intake gap allowed a self-consistent candidate to declare
`artifact_root` as `artifacts/wing_readonly_preflight ` with trailing
whitespace. The canonical-root check trimmed the value before suffix matching,
but the raw path still pointed to a different directory.

## Options Considered

1. Leave artifact-root whitespace to manual review.
2. Reject file-intake candidates whose `artifact_root` has surrounding
   whitespace.
3. Normalize whitespace and accept the candidate as canonical.

## Decision

Use option 2. Candidate intake now reports
`artifact_root_has_surrounding_whitespace` when the declared WING read-only
artifact root changes under `strip()`.

## Why This Won

The first accepted-candidate boundary should not treat a whitespace-mutated
directory as the canonical WING proof root. Rejecting surrounding whitespace
keeps path comparison exact and prevents a candidate from looking canonical
only because the checker trimmed the string for suffix matching.

## Rejected Alternatives

Manual-only review was rejected because the automated intake can cheaply catch
the path ambiguity before a bad candidate looks ready. Normalizing and
accepting whitespace was rejected because it hides that the payload points to a
different filesystem path.

## Implementation Plan

- Add a surrounding-whitespace detector for WING read-only evidence artifact
  roots.
- Add focused regression coverage for a self-consistent whitespace-root
  candidate that previously validated as an accepted candidate.
- Add rail coverage and update state/report surfaces.

## Test Plan

- Focused WING read-only evidence tests.
- M101 rail.
- Full WING MVP rail.
- JSON/JSONL validation and memory lint before commit.

## Risks and Rollback

The main risk is rejecting manually copied artifact roots with accidental
leading or trailing whitespace. That is intended for M001: the real WING proof
root must be exact. Roll back by reverting this ADR, the M101 plan/report/state
entries, the test and the artifact-root whitespace guard. No live transport or console state is affected.
