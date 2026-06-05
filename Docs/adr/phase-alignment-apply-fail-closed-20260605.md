# ADR: Phase Alignment Apply Fail Closed - 2026-06-05

## Context

`backend/phase_alignment.py` analyzes inter-channel delay and polarity using phase and correlation measurements. This is relevant to the long-term AI Mixing Engineer goal. However, direct delay and polarity writes can alter mono compatibility and stage timing. The currently proven live executor boundary remains supervised fader/gain only.

The server's public deployment path already quarantines phase write requests, but the controller and private server helpers still contained direct calls to mixer-client delay, polarity, and raw OSC send methods.

## Options considered

1. Keep the analysis but make apply/reset fail closed.
2. Leave the controller unchanged because WebSocket handlers are deployment-quarantined.
3. Remove phase apply/reset methods.
4. Promote phase apply into supervised auto mode now.

## Decision

Choose option 1. Phase alignment apply and reset paths now return structured blocked evidence and do not mutate the mixer client.

## Why this won

This preserves useful phase-correction analysis while removing hidden console mutations. It keeps the project moving toward automatic phase correction without pretending that delay/polarity writes are proven safe. Future promotion can reuse the blocked evidence as a proposal/apply contract once readback, rollback, mono compatibility, and hardware proof exist.

## Rejected alternatives

- Leaving direct writes in the controller was rejected because direct Python use could bypass the server quarantine.
- Removing apply/reset was rejected because the autonomous roadmap needs explicit would-be commands for operator review and validation.
- Promoting phase apply was rejected because the proof requirements are not satisfied.

## Implementation plan

1. Add write-policy telemetry to `PhaseAlignmentController`.
2. Convert `apply_corrections()` into blocked evidence generation.
3. Convert `reset_all_phase_delay()` into blocked reset evidence.
4. Make private server phase helpers delegate to the fail-closed controller.
5. Update registry, write-surface inventory, capability matrix, docs, and tests.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_phase_alignment_fail_closed.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py tests/test_server.py tests/test_write_surface_inventory_static_scan.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`
- `PYTHONPATH=backend python3 -m py_compile backend/phase_alignment.py backend/server.py`
- `/opt/homebrew/bin/git diff --check`

## Risks and rollback

Risk: legacy callers that expected phase apply/reset to return `True` now receive structured blocked evidence. This is intentional because direct delay/polarity writes are not approved.

Rollback must not restore direct `set_channel_delay`, `set_channel_phase_invert`, or raw delay `send` calls. A live phase path requires a separate supervised proposal/apply contract with fresh phase context, current delay/polarity readback, mono compatibility checks, rollback values, and hardware proof.
