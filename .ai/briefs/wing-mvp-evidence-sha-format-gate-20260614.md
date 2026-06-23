# Brief: WING MVP Evidence SHA Format Gate

Date: 2026-06-14
Task ID: wing-mvp-evidence-sha-format-gate-20260614

## Context

The read-only WING evidence validator already checks that plan and prestart
summary digests are present, and file-based review compares them against the
referenced files. Inline/candidate evidence still needs the same digest-format
boundary so malformed non-empty values cannot look acceptable before file
crosschecks run.

## Goal

Reject malformed `expected_plan_sha256` and
`backend_port_prestart_summary_sha256` values at the read-only evidence intake
boundary.

## Safety Boundary

- No WING hardware contact.
- No backend/WING WebSocket opening.
- No OSC/WING writes.
- No live authority granted.
- No readiness closure from offline evidence.

## Expected Validation

- Focused read-only evidence tests.
- Full no-WING release-check module.
- Project Director/WING rails.
- Compile, memory lint, and whitespace diff.
