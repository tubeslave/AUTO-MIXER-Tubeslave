# Brief: Supervised Proof Evaluator Physical Status Gate - 2026-06-07

## Context

The WING supervised proof runner now requires physical readback status for
physical `--execute`. Downstream report/evaluator code also reviews stored
summary JSON before any safety-row review.

## Problem

A summary artifact can claim `source: physical_readback` while keeping
`readback_status: verified_from_cache`. Source-only checks are not strong enough
for promotion evidence.

## Scope

- Require `verified_from_physical_readback` status for physical pre-read, apply
  readback, and rollback readback in the evaluator.
- Keep cache-compatible status handling for dry-run and non-physical evaluation.
- Add no-WING policy coverage so this cannot silently regress.

## Out Of Scope

- Running WING hardware.
- Granting readiness or live authority.
- Widening writes beyond the existing supervised fader/gain proof path.
