# Brief: WING MVP Accepted-Candidate Prestart SHA Format Gate

Date: 2026-06-14
Task ID: wing-mvp-accepted-candidate-prestart-sha-format-gate-20260614

## Context

Accepted-candidate WING read-only prestart traces must bind future supervised
proof planning to a passed no-WING prestart `summary.json` by path and SHA256.
The shared contract helper already rejects missing digests, but a malformed
non-empty string should also fail closed before a candidate can unlock physical
proof steps.

## Goal

Require `backend_port_prestart_summary_sha256` in accepted-candidate prestart
rows to be a 64-character lowercase SHA-256 hex digest, and prove malformed
digests block both review-decision acceptance and supervised proof planning.

## Safety Boundary

- No WING hardware contact.
- No backend/WING listener start, stop, or reconfiguration.
- No OSC/WING writes.
- No live authority granted.
- No readiness closure from offline evidence.

## Expected Validation

- Focused accepted-candidate contract helper test.
- Focused WING read-only review-decision ledger test.
- Focused supervised proof-plan test.
- Compile, memory lint, Project Director guard, and whitespace diff.
