# Brief: no-WING Manifest Cross-Surface Summary Self-Check

## Task ID

wing-mvp-nowing-manifest-cross-surface-summary-self-check-20260615

## Context

The active supervised WING Rack MVP blocker remains `WING read-only connection
proof`. The no-WING release summary is an offline gate before any real WING
read-only session, so its JSON and Markdown surfaces must report manifest
contract drift from computed evidence, not static success values.

The detached `artifact-manifest.json` already computed cross-surface contract
matches from written files. The top-level no-WING `summary.json` and
operator-readable `summary.md` still initialized the cross-surface match fields
as successful before that file-based manifest existed.

## Goal

Make the no-WING summary surfaces compute and render manifest cross-surface
contract verdicts before writing `summary.json` and `summary.md`.

## Safety boundary

Offline no-WING reporting only. Do not contact WING, query a console, mutate
mixer state, close readiness, or grant live authority.

## Expected validation

- Focused manifest cross-surface tests.
- no-WING quick.
- Project Director guard.
- Standard pytest before commit.
