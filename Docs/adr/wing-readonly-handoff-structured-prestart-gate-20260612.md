# ADR: WING Read-Only Handoff Structured Prestart Gate

## Context

The WING read-only handoff packet has two safety-critical ordering gates. The
final supervised prestart gate was already structured as actual and expected
objects. The first backend-port prestart gate had individual fields, but no
complete object comparison.

## Options considered

1. Keep the existing individual first-gate fields.
2. Add `backend_port_prestart_gate` only.
3. Add actual and expected backend-port prestart gate objects and validate both.

## Decision

Use option 3. The handoff packet now includes
`backend_port_prestart_gate` and `expected_backend_port_prestart_gate`, and the
no-WING validator fails closed if either object drifts from the derived first
operator-sequence gate.

## Why this won

The first gate must prove the backend-port check happens before any intended
backend/WING session. A structured actual/expected contract makes that ordering
auditable in the same way as the final supervised proof gate.

## Rejected alternatives

- Individual fields alone were rejected because they do not prove a coherent
  gate object.
- Actual-only gate data was rejected because there would be no declared expected
  contract to compare.

## Implementation plan

- Add structured gate objects in the handoff packet builder.
- Render both objects in Markdown.
- Derive and compare the expected gate in `tools/no_wing_release_check.py`.
- Include both objects in no-WING summaries.
- Update tests and runbook rails.

## Test plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_handoff_packet.py tests/test_no_wing_release_check.py -k "handoff_packet"`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_mvp_rails.py -k runbooks`
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --continue-on-failure`
- `/opt/homebrew/bin/git diff --check`
- `python3 tools/memory/memory_lint.py`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Risks and rollback

The stricter contract can reject stale packets. Regenerate stale handoff
packets. Rollback is to remove the structured object comparison, but that would
reduce ordering proof quality before real WING work.
