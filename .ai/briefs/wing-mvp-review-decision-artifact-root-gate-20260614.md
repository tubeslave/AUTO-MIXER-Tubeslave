# Brief: WING MVP Review Decision Artifact Root Gate

Date: 2026-06-14
Task ID: wing-mvp-review-decision-artifact-root-gate-20260614

## Context

Real WING read-only preflight summaries now declare the runtime
`artifact_root`, and evidence intake rejects missing or mismatched plan roots.
The operator-edited review decision template and ledger still need to bind the
same root before a manual decision can be accepted.

## Goal

Require WING read-only review decision records to include an `artifact_root`
that matches the accepted read-only preflight candidate.

## Safety Boundary

- No WING hardware contact.
- No backend/WING WebSocket opening.
- No OSC/WING writes.
- No live authority granted.
- No readiness closure from offline evidence.

## Expected Validation

- Focused review-decision and review-tool tests.
- Evidence/runner compatibility tests.
- Full no-WING release-check module.
- Project Director/WING rails.
- Compile, memory lint, and whitespace diff.
