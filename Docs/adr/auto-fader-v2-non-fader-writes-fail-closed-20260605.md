# ADR: AutoFaderV2 Non-Fader Writes Fail Closed - 2026-06-05

## Context

`backend/auto_fader_v2/controller.py` is a realtime auto-fader controller. Besides fader moves, it contained side branches for adaptive noise gate, spectral masking EQ and cross-adaptive EQ.

The current proven live executor boundary remains supervised fader/gain only. Gate and EQ writes are outside that boundary. They require source identity, freshness, filter or gate safety checks, readback, rollback values and physical proof before promotion.

## Options considered

1. Keep AutoFaderV2 analysis but make gate/EQ side effects fail closed.
2. Leave the direct writes because AutoFaderV2 is not the primary Product Layer executor.
3. Remove adaptive gate and masking EQ branches.
4. Promote gate/EQ writes into supervised auto mode now.

## Decision

Choose option 1. AutoFaderV2 adaptive gate and masking EQ branches now return structured blocked evidence and do not call mixer gate, EQ or gate fallback fader methods.

## Why this won

This removes hidden mixer mutations without deleting useful realtime analysis. Future supervised promotion can reuse the evidence after source identity, freshness, false-decision replay, filter audit, readback, rollback and physical hardware proof exist.

## Rejected alternatives

- Leaving direct writes in place was rejected because direct Python use could bypass Product Layer safety.
- Removing the branches was rejected because blocked would-be commands are useful for validation and operator review.
- Promoting gate/EQ writes was rejected because proof requirements are incomplete.
- Migrating the main fader apply path was deferred because it is a separate Product Layer migration with larger blast radius.

## Implementation plan

1. Add non-fader blocked-write helpers and telemetry.
2. Convert adaptive gate close/open and fader fallback to blocked evidence.
3. Convert spectral/cross-adaptive masking EQ to blocked evidence.
4. Update write-surface inventory, capability matrix, processing inventory and tests.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_auto_fader_v2_non_fader_fail_closed.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_auto_fader_v2_non_fader_fail_closed.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py tests/test_write_surface_inventory_static_scan.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`
- `PYTHONPATH=backend python3 -m py_compile backend/auto_fader_v2/controller.py tests/test_auto_fader_v2_non_fader_fail_closed.py`
- `/opt/homebrew/bin/git diff --check`

## Risks and rollback

Risk: legacy AutoFaderV2 use that expected adaptive gate or masking EQ to mutate the mixer now receives blocked evidence. This is intentional because those writes are not approved.

Rollback must not restore direct `set_gate_on`, gate fallback `set_channel_fader(..., -120.0)` or `set_eq_band` calls. A live gate/EQ path requires a separate supervised proposal/apply contract with readback, rollback and hardware validation.
