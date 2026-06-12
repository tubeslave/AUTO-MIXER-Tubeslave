# ADR: WING Read-Only Handoff First Prestart Gate

## Context

The WING read-only handoff packet already contains `prestart_backend_port_check`
and a structured final supervised prestart gate. The first gate is just as
important: before opening the intended backend/WING session, the operator must
run the no-WING backend-port check and resolve only confirmed unexpected
listeners.

## Options considered

1. Keep the first operator step as prose only.
2. Add structured first-step fields but do not validate them.
3. Add structured first-step fields and fail closed when they drift.

## Decision

Use option 3. The handoff packet now declares the backend-port prestart check as
operator-sequence index `0`, and the no-WING validator fails closed if the
packet does not keep that gate first.

## Why this won

The current blocker is a read-only WING connection proof. A stale or ambiguous
handoff packet could lead an operator to start the backend/WING session before
the local listener check. Making the first gate machine-checkable preserves the
safe ordering without contacting hardware or granting live authority.

## Rejected alternatives

- Prose-only guidance was rejected because it cannot be checked reliably by the
  release gate.
- Validation-free fields were rejected because they could drift silently.

## Implementation plan

- Add structured first-step fields to the handoff packet.
- Render those fields in Markdown.
- Validate the fields and first operator-sequence item in
  `tools/no_wing_release_check.py`.
- Update tests and the read-only runbook.

## Test plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_handoff_packet.py tests/test_no_wing_release_check.py -k "handoff_packet"`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_mvp_rails.py -k runbooks`
- `/opt/homebrew/bin/git diff --check`
- `python3 tools/memory/memory_lint.py`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Risks and rollback

Older handoff packets may fail validation. Regenerate them with the current
tooling. Rollback is to remove the structured first-step fields and validator
checks, but that would make the operator sequence less auditable.
