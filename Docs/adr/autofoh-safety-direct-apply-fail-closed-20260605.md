# ADR: AutoFOH Safety Direct Apply Fail Closed - 2026-06-05

## Context

`backend/autofoh_safety.py` is a typed correction safety layer. It applies runtime policy, bounds values and rate-limits actions before translating them into mixer calls.

The current proven live executor boundary remains Product Layer supervised fader/gain only. AutoFOH Safety still had direct translators for bus/DCA/main fader, EQ, HPF, compressor, gate, pan, send, phase/delay and feedback-notch actions.

## Options considered

1. Keep AutoFOH typed decisions but block non-channel-fader/gain direct apply.
2. Leave broad AutoFOH writes enabled because they are bounded.
3. Remove non-fader/gain typed actions.
4. Migrate every AutoFOH action through Product Layer now.

## Decision

Choose option 1. AutoFOH Safety now returns blocked evidence for every direct apply surface outside channel fader/gain.

## Why this won

It keeps the useful decision model and bounded values while removing hidden broad console mutations. Future promotion can reuse the evidence once source identity, freshness, readback, rollback and hardware proof exist.

## Rejected alternatives

- Bounds-only direct apply was rejected because bounds do not prove physical state, rollback or operator authority.
- Removing the typed actions was rejected because the proposal/evidence shape is useful.
- Full Product Layer migration was deferred because channel fader/gain migration has a wider runtime bridge impact.

## Implementation plan

1. Add AutoFOH direct-write policy constants and blocked decision fields.
2. Intercept non-channel-fader/gain actions after bounds and before translators.
3. Return blocked evidence with would-be mixer commands and bounded values.
4. Remove direct non-fader/gain mixer translator calls from `autofoh_safety.py`.
5. Update tests and inventory docs.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_autofoh_safety.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_autofoh_safety.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py tests/test_write_surface_inventory_static_scan.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`
- `PYTHONPATH=backend python3 -m py_compile backend/autofoh_safety.py tests/test_autofoh_safety.py`
- `/opt/homebrew/bin/git diff --check`

## Risks and rollback

Risk: legacy callers expecting AutoFOH non-fader/gain actions to mutate the mixer now receive blocked decisions. This is intentional.

Rollback must not restore direct broad mixer translators. A live path for those surfaces requires a separate Product Layer proposal/apply contract with readback, rollback and hardware validation.
