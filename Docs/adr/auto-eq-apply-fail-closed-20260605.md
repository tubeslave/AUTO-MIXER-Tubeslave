# ADR: Auto EQ Apply Fail Closed - 2026-06-05

## Context

`backend/auto_eq.py` performs spectrum analysis, target-profile matching, EQ correction calculation, local preview and multi-channel conflict resolution. This is central to the long-term AI Mixing Engineer goal.

The current proven live executor boundary remains supervised fader/gain only. EQ writes can introduce feedback, phase artifacts, unstable filters or damage existing manual/notch bands. Public server paths already quarantine EQ apply/reset, but the AutoEQ controllers still contained direct mixer-client writes.

## Options considered

1. Keep AutoEQ analysis but make apply/reset fail closed.
2. Leave controller writes because public server handlers are quarantined.
3. Remove AutoEQ apply/reset APIs.
4. Promote EQ apply into supervised auto mode now.

## Decision

Choose option 1. AutoEQ apply/reset now return structured blocked evidence and do not mutate mixer clients.

## Why this won

This preserves useful AutoEQ recommendations while removing hidden console mutations. Future promotion can reuse the blocked evidence shape after filter audit, feedback safety, readback, rollback and hardware proof exist.

## Rejected alternatives

- Leaving controller writes in place was rejected because direct Python use could bypass Product Layer safety.
- Removing apply/reset APIs was rejected because reviewable would-be EQ commands are useful for assisted mixing and validation.
- Promoting EQ writes was rejected because proof requirements are incomplete.

## Implementation plan

1. Add blocked EQ call builders and write telemetry.
2. Convert single-channel apply/reset into blocked evidence.
3. Convert multi-channel apply into blocked evidence.
4. Update registry, write-surface inventory, capability matrix, docs and tests.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_auto_eq_fail_closed.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py tests/test_server.py tests/test_write_surface_inventory_static_scan.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`
- `PYTHONPATH=backend python3 -m py_compile backend/auto_eq.py`
- `/opt/homebrew/bin/git diff --check`

## Risks and rollback

Risk: legacy callers that expected direct AutoEQ apply/reset to mutate the mixer now receive blocked evidence. This is intentional because direct EQ writes are not approved.

Rollback must not restore direct `set_eq_on`, `set_eq_low_shelf`, `set_eq_high_shelf`, `set_eq_band`, or raw EQ reset `send` calls. A live EQ path requires a separate supervised proposal/apply contract with filter audit, feedback safety, readback, rollback values and hardware validation.
