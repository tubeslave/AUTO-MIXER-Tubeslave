# Brief: WING Read-Only Preflight Plan Summary Consistency Gate

## Task ID

wing-readonly-preflight-plan-summary-consistency-gate-20260607

## Context

File-based WING read-only evidence review now verifies that the referenced
offline plan file exists and matches the recorded SHA. The next proof gap is
content consistency: a summary should not be accepted if its referenced plan
hashes correctly but describes different channels, value types, readback
requirements, policy flags, or request sequence.

## Scope

- Keep the change read-only and offline-testable.
- Do not contact WING Rack, WebSocket, OSC, MIDI, or any hardware transport.
- Do not widen live writes or readiness closure.
- Validate content consistency only for file-based artifact review.

## Deliverables

- Evidence validator compares expected plan content with summary content.
- Regression tests for metadata mismatch and request-sequence mismatch.
- Review/decision fixtures carry full expected-plan payloads.
- Runbook, ADR, report, and documentation index update.

## Validation

- Focused WING preflight evidence/review/decision tests.
- No-WING quick release check.
- Full pytest suite before commit.
