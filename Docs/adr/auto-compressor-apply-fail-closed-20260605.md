# ADR: Auto Compressor Apply Fail Closed - 2026-06-05

## Context

`backend/auto_compressor.py` and `backend/auto_compressor_cf.py` calculate compressor settings from channel signal features, crest factor, LUFS, target gain reduction, profile selection and makeup-gain logic. This supports the long-term AI Mixing Engineer goal.

The current proven live executor boundary remains supervised fader/gain only. Compressor writes can change dynamics, loudness, transient behavior and makeup gain, so live promotion requires fresh dynamics readback, true-peak/headroom proof, rollback values and hardware validation.

## Options considered

1. Keep compressor analysis but make direct apply paths fail closed.
2. Leave controller writes in place because deployment handlers are quarantined.
3. Remove auto-compressor apply APIs.
4. Promote compressor apply into supervised auto mode now.

## Decision

Choose option 1. AutoCompressor and AutoCFCompressor direct apply paths now return structured blocked evidence and do not mutate mixer clients.

## Why this won

This keeps automatic compression progress while avoiding a hidden live-write bypass. Future assisted or supervised compressor control can reuse the blocked evidence shape as an operator proposal contract after readback and rollback proof exist.

## Rejected alternatives

- Leaving direct writes in place was rejected because code-level callers could bypass Product Layer safety.
- Removing apply APIs was rejected because the system needs explicit would-be compressor actions for review, replay and validation.
- Promoting compressor apply was rejected because the proof requirements are not complete.

## Implementation plan

1. Add write-policy telemetry to compressor controllers.
2. Convert classic compressor smooth apply and compressor on/off writes into blocked evidence.
3. Convert CF compressor apply into blocked evidence.
4. Update registry, write-surface inventory, capability matrix, docs and tests.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_auto_compressor_fail_closed.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py tests/test_server.py tests/test_write_surface_inventory_static_scan.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`
- `PYTHONPATH=backend python3 -m py_compile backend/auto_compressor.py backend/auto_compressor_cf.py`
- `/opt/homebrew/bin/git diff --check`

## Risks and rollback

Risk: legacy callers that expected direct compressor apply now receive blocked evidence. This is intentional because direct compressor writes are not approved.

Rollback must not restore direct `set_compressor_on` or `set_compressor` calls. A live compressor path requires a separate supervised proposal/apply contract with fresh dynamics readback, bounded makeup/headroom proof, rollback values and hardware validation.
