# Automixer WING Read-Only Preflight Runner - 2026-06-06

## Summary

This slice adds a dedicated read-only preflight runner for the supervised WING
Rack MVP. It reduces the first proof blockers by making the operator's next
hardware run recordable before any live write is attempted.

The runner:

- talks only to the already-running backend WebSocket;
- sends only `get_operator_mode`, `get_state`, and `query_mixer_parameter`;
- records events and a summary artifact;
- blocks when the backend is not in `wing` mode;
- blocks when `--require-physical-readback` is set and any readback is not
  physical.

## Command

Example physical preflight command:

```bash
python3 tools/wing_readonly_preflight_runner.py \
  --url ws://localhost:8765 \
  --channels 3 \
  --value-types fader,gain \
  --require-physical-readback \
  --artifact-root artifacts/wing_readonly_preflight
```

## Safety Boundary

- No live WING/OSC action was run for this change.
- The runner does not discover or connect to WING.
- The runner does not send fader, gain, trim/preamp-gain, EQ, compressor,
  routing, snapshot, reset, mute/unmute, insert, or FX write messages. It also
  does not send `set_operator_mode`, `create_operator_proposal`,
  `apply_operator_proposal`, `rollback_last_manual_write`, `emergency_stop`,
  `connect_wing`, `disconnect_wing`, or `disconnect_mixer`.
- The output is evidence for operator review, not permission to widen live
  writes.

## Files Changed

- `tools/wing_readonly_preflight_runner.py`
- `tests/test_wing_readonly_preflight_runner.py`
- `backend/write_surface_inventory.py`
- `.ai/briefs/wing-readonly-preflight-runner-20260606.md`
- `.ai/proposals/codex-wing-readonly-preflight-runner-20260606.md`
- `Docs/adr/wing-readonly-preflight-runner-20260606.md`

## Validation

- `python3 -m py_compile tools/wing_readonly_preflight_runner.py` - PASS
- `python3 -m py_compile tools/wing_readonly_preflight_runner.py backend/write_surface_inventory.py` - PASS
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_runner.py` - `9 passed`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_runner.py tests/test_wing_operator_proposal_validation_runner.py tests/test_server.py::TestWingDeploymentWriteBoundary::test_query_mixer_parameter_reads_gain_without_writing tests/test_server.py::TestWingDeploymentWriteBoundary::test_query_mixer_parameter_blocks_unsupported_address` - `23 passed`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_write_surface_inventory_static_scan.py tests/test_processing_registry.py::test_write_surface_inventory_is_fail_closed_by_default tests/test_processing_registry.py::test_write_surface_inventory_covers_known_legacy_write_paths tests/test_processing_registry.py::test_write_surface_snapshot_status_counts_and_blocked_surfaces` - `16 passed`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q` - `2846 passed, 1 skipped, 1 warning`
- `python3 tools/memory/memory_lint.py` - PASS
- `/opt/homebrew/bin/git diff --check` - PASS

## Remaining WING MVP Blockers

This does not complete physical proof by itself. The next hardware run still
needs:

1. WING read-only connection artifact from the real backend/WING session.
2. Physical console readback artifact from the real WING.
3. Supervised fader/gain proof with explicit operator approval.
4. Rollback proof.
5. Emergency stop proof.
6. Controlled timeout/fail-closed artifact from a physical run.
7. Readiness matrix update based only on real proof.
