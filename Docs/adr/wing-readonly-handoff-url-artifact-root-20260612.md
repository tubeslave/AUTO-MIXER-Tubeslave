# ADR: WING read-only handoff URL and artifact root

## Context

The read-only preflight plan and capture commands explicitly define
`ws://localhost:8765` and `artifacts/wing_readonly_preflight`. Those fields
participate in the expected-plan contract. The handoff command used matching
defaults but did not show or validate them in no-WING evidence.

## Options Considered

1. Continue relying on handoff defaults.
2. Pass URL and artifact root explicitly and validate their match in no-WING
   evidence.

## Decision

Pass URL and artifact root explicitly through the handoff command.

## Why This Won

The real read-only run fails closed when expected-plan fields drift. The
operator handoff should show the same URL and artifact root before any backend
WebSocket session is opened.

## Rejected Alternatives

Relying only on defaults was rejected because it hides values that decide where
the real proof artifact will be captured and reviewed.

## Implementation Plan

- Update the handoff command builder.
- Surface URL and artifact-root match fields in no-WING evidence.
- Update runbook and rails tests.

## Test Plan

- Focused no-WING release-check tests.
- WING MVP rails tests.
- Full repository pytest.

## Risks And Rollback

The change remains offline/read-only. Rollback is to remove the explicit
handoff URL and artifact-root flags and summary fields.
