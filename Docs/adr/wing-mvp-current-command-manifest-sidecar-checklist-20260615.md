# ADR: WING MVP Current Command Manifest Sidecar Checklist

Date: 2026-06-15

## 1. Context

The WING read-only runbook now requires operators to inspect the no-WING
`artifact-manifest.json` summary-surface sidecar before opening a backend/WING
WebSocket. Project Director, however, routes Goal-mode work from
`## Current First Command`; that command did not yet name the sidecar checklist.

## 2. Options Considered

1. Keep the sidecar checklist only in the runbook.
2. Add a new runtime artifact field.
3. Add the sidecar checklist to Current First Command and test it.

## 3. Decision

Add the checklist to Current First Command and test it.

## 4. Why This Won

The current command is the active Project Director directive. It should carry
the same fail-closed pre-WING sidecar markers as the operator runbook without
creating new live authority or runtime behavior.

## 5. Rejected Alternatives

Runbook-only coverage was rejected because it lets active Goal-mode routing
omit a required operator gate. A new runtime field was rejected because the
sidecar verdict already exists.

## 6. Implementation Plan

- Update `.ai/project-layer/wing_mvp_blockers.md` Current First Command.
- Extend rails tests for the command markers.
- Register report evidence and regenerate latest progress.

## 7. Test Plan

- Targeted rails test.
- Project Director guard snapshot.
- Focused WING/no-WING/Director tests.
- Memory lint and `git diff --check`.

## 8. Risks and Rollback

Risk is limited to longer directive text. Roll back by removing the added
Current First Command paragraph, tests, and evidence report if the sidecar
contract is replaced.
