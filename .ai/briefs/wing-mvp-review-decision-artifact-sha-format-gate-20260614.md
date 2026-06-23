# Brief: WING MVP Review Decision Artifact SHA Format Gate

Date: 2026-06-14
Task ID: wing-mvp-review-decision-artifact-sha-format-gate-20260614

## Context

Operator review decisions bind accepted read-only preflight candidates to the
captured summary and event-log artifacts by SHA-256. The ledger already checks
stale digest mismatches, but malformed manual digest strings should be rejected
explicitly at the decision-record boundary.

## Goal

Reject malformed `summary_sha256` and `events_sha256` values in WING read-only
review decision records before they can be accepted as operator decisions.

## Safety Boundary

- No WING hardware contact.
- No backend/WING WebSocket opening.
- No OSC/WING writes.
- No live authority granted.
- No readiness closure from offline evidence.

## Expected Validation

- Focused review-decision ledger tests.
- Full no-WING release-check module.
- Project Director/WING rails.
- Compile, memory lint, and whitespace diff.
