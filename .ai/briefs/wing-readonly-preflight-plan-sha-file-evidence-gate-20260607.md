# Brief: WING Read-Only Preflight Plan SHA File Evidence Gate

## Task ID

wing-readonly-preflight-plan-sha-file-evidence-gate-20260607

## Context

The WING read-only preflight summary records `expected_plan_path`,
`expected_plan_sha256`, and `expected_plan_match`. Evidence intake requires the
fields, but file-based artifact review should also verify that the referenced
plan file exists and hashes to the recorded SHA.

## Scope

- Keep the work read-only and offline.
- Do not contact WING Rack, WebSocket, OSC, or MIDI hardware.
- Do not widen live writes or readiness closure.
- Validate plan-file existence and SHA match only when reviewing file artifacts.

## Deliverables

- File-artifact evidence validator checks the referenced offline plan SHA.
- Focused regression tests for missing and tampered plan files.
- Review/decision test fixtures updated to include a real plan file.
- Runbook, ADR, report, and documentation index update.

## Validation

- Focused WING preflight evidence/review/decision tests.
- No-WING quick release check.
- Full pytest suite before commit.
