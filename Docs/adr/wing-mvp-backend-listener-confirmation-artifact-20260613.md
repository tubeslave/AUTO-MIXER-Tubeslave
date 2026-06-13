# ADR: WING MVP Backend Listener Confirmation Artifact

## Context

The WING read-only preflight rail requires the no-WING quick gate before any
backend/WING session. When a listener already exists on `127.0.0.1:8765`, the
gate correctly fails closed and asks the operator to confirm whether that
listener is the intended Automixer backend. Before this change, there was no
file-backed confirmation path for that operator decision.

## Options considered

1. Keep every busy backend port as a hard local stop.
2. Add an operator confirmation JSON that must match the current listener.
3. Infer an intended backend from process command line.
4. Stop or reconfigure unexpected listeners automatically.

## Decision

Use option 2.

## Why this won

The operator can confirm an already-running intended backend without giving the
gate permission to stop processes or grant live authority. Requiring current PID
and command-line matching prevents a stale confirmation from silently clearing
the prestart blocker.

## Rejected alternatives

- Option 1 blocks legitimate intended-backend sessions.
- Option 3 makes Codex infer operator intent from process metadata.
- Option 4 violates the no-WING gate safety boundary.

## Implementation plan

- Add `wing_backend_port_listener_confirmation.v1` validation to
  `tools/no_wing_release_check.py`.
- Add `--backend-listener-confirmation`.
- Render accepted or invalid confirmation details in the no-WING summary.
- Update the read-only runbook.
- Add tests for accepted, mismatched, and summary-rendered confirmation.
- Add the report to first-blocker evidence.

## Test plan

- Focused no-WING confirmation tests.
- Focused WING rails, Project Director, and Direction Guard tests.
- Project Director and Direction Guard JSON checks.
- `git diff --check`.
- `tools/memory/memory_lint.py`.
- Full backend pytest.

## Risks and rollback

Rollback removes the CLI flag, confirmation validation, docs, tests, and report.
The default busy-port fail-closed behavior remains the safety baseline.
