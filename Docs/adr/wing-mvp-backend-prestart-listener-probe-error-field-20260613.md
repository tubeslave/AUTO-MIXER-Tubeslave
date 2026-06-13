# ADR: WING MVP Backend Prestart Listener Probe Error Field

## Context

The WING MVP first blocker requires a no-WING backend-port prestart gate before
the intended backend/WING session. If the port is busy, the gate fails closed
and records listener owner diagnostics. Owner discovery can itself fail, and
that failure is captured as `listener_process_error`.

## Options considered

1. Leave `listener_process_error` as an optional implementation detail.
2. Add `listener_process_error` to the shared backend-port prestart
   failure-detail contract.
3. Create a separate owner-probe contract.

## Decision

Use option 2.

## Why this won

The existing prestart failure-detail contract is the operator-facing place for
busy-port evidence. Adding `listener_process_error` there keeps owner-probe
failures visible in the handoff packet, Project Director selected action, and
generated progress without changing live behavior.

## Rejected alternatives

- Option 1 lets a busy-port failure appear complete even when owner discovery
  failed.
- Option 3 adds a new ownership boundary for one field.

## Implementation plan

- Add `listener_process_error` to the shared WING read-only preflight contract.
- Update blocker, runbook, MVP, autoloop, and generated progress docs.
- Add focused tests for constants, busy-port detail, and Project Director/MVP
  text surfaces.
- Record validation in a report.

## Test plan

- Focused no-WING release, handoff packet, WING MVP rails, Project Director,
  and guard tests.
- Project Director JSON check.
- Direction guard JSON check.
- `git diff --check`.
- `python3 tools/memory/memory_lint.py`.
- Full backend pytest.

## Risks and rollback

Rollback removes `listener_process_error` from the exact-list contract and
related docs/tests. It does not affect live WING behavior, listener ownership,
or readiness status.
