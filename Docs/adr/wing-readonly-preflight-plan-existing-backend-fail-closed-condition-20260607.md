# ADR: WING Read-Only Preflight Plan Existing Backend Fail-Closed Condition

## Context

The supervised WING Rack MVP is still blocked on the first real hardware step:
WING read-only connection proof. The preflight runner already emits
`existing_backend_connection_boundary_missing` in its offline plan, and the
runbook requires a real run to use an already-running backend/WING session
rather than opening or switching mixer transport itself.

The no-WING release gate validated the plan's boolean
`physical_preflight_requires_existing_backend_connection`, but its required
fail-closed condition list did not include the matching blocker string.

## Options considered

1. Leave the release gate as-is and trust the boolean field.
2. Add the missing fail-closed condition to the release gate contract.
3. Change the runner command shape.

## Decision

Use option 2. The no-WING release gate now requires
`existing_backend_connection_boundary_missing` in WING read-only preflight plan
artifacts.

## Why this won

The change keeps the runtime path unchanged, but makes the operator handoff more
auditable. A plan artifact can no longer pass release validation while omitting
the explicit fail-closed reason for missing already-connected backend/WING
session proof.

## Rejected alternatives

Trusting only the boolean was weaker for manual review because the fail-closed
list is what operators and reports scan before hardware work. Changing the
runner command shape was unnecessary and riskier.

## Implementation plan

- Update `tools/no_wing_release_check.py` required plan fail-closed conditions.
- Update `tests/test_no_wing_release_check.py` to prove valid plans include the
  condition and unsafe plans fail when it is missing.
- Preserve fail-closed evidence consistency by surfacing forbidden event-log
  sends as summary-level forbidden-request violations.

## Test plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_no_wing_release_check.py tests/test_wing_readonly_preflight_runner.py`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_evidence.py`
- `python3 tools/memory/memory_lint.py`

## Risks and rollback

Risk is low: this tightens offline validation only. It does not open WebSocket,
query WING, or widen live writes. Rollback is removing the condition from the
required list and the related assertions, but that would make the preflight
handoff less explicit.
