# Automixer WING Read-Only Preflight Handoff Forbidden Request Coverage - 2026-06-07

## Summary

This slice strengthens the first supervised WING MVP blocker:
`WING read-only connection proof`.

The offline handoff packet now lists the full forbidden live request surface for
the read-only preflight handoff, including proposal apply, rollback,
emergency-stop, operator-mode changes, direct fader/gain/trim/EQ/compressor,
routing, scene/snapshot, mute, FX, and connect/disconnect request types.

The no-WING release gate now validates that the handoff packet contains that
full forbidden request list. If a required entry such as `set_gain` is missing,
the artifact fails closed.

The plan-only artifact and no-WING release gate also now name
`event_log_wing_mode_not_proven` as a fail-closed condition, matching the
event-log validator's requirement that WING mode is proven from
`operator_mode_status` and `state_update` payloads.

Because those request names are intentionally present as strings, the handoff
builder is also listed in the write-surface inventory under lab-only proof and
release-check tooling. It is not a live executor.

## Safety Boundary

- No live WING/OSC/WebSocket command was run.
- No backend connection was opened.
- No read-only physical artifact was captured.
- No fader, gain, trim, EQ, compressor, routing, mute, FX, scene, rollback, or
  emergency-stop command was sent.
- No readiness row was closed.

## Files Changed

- `tools/build_wing_readonly_preflight_handoff_packet.py`
- `tools/no_wing_release_check.py`
- `tools/wing_readonly_preflight_runner.py`
- `backend/write_surface_inventory.py`
- `tests/test_wing_readonly_preflight_handoff_packet.py`
- `tests/test_no_wing_release_check.py`
- `tests/test_wing_readonly_preflight_runner.py`
- `.ai/project-layer/wing_mvp_blockers.md`
- `.ai/briefs/wing-readonly-preflight-handoff-forbidden-request-coverage-20260607.md`
- `.ai/proposals/codex-wing-readonly-preflight-handoff-forbidden-request-coverage-20260607.md`
- `Docs/adr/wing-readonly-preflight-handoff-forbidden-request-coverage-20260607.md`

## Validation

- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_handoff_packet.py tests/test_no_wing_release_check.py::test_no_wing_release_check_validates_wing_preflight_handoff_packet_artifact tests/test_no_wing_release_check.py::test_no_wing_release_check_rejects_unsafe_wing_preflight_handoff_packet_artifact tests/test_no_wing_release_check.py::test_no_wing_release_check_rejects_unsafe_wing_preflight_plan_artifact tests/test_wing_readonly_preflight_runner.py::test_plan_only_outputs_readonly_request_plan_without_websocket_or_artifacts`
  passed: `12 passed`.
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_handoff_packet.py tests/test_no_wing_release_check.py`
  passed: `188 passed`.
- `PYTHONPATH=backend python3 -m pytest tests/test_write_surface_inventory_static_scan.py -q --tb=short`
  passed: `13 passed`.
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_runner.py tests/test_wing_readonly_preflight_handoff_packet.py tests/test_no_wing_release_check.py tests/test_write_surface_inventory_static_scan.py`
  passed: `223 passed`.
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root artifacts/no_wing_release_check_current_cycle --command-timeout-sec 300`
  passed. The generated summary showed `wing_readonly_preflight_handoff_packet_artifact.status=passed`,
  `live_authority_granted=false`, and the expanded forbidden live request list.
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`
  passed: `3044 passed, 1 skipped`.

## Remaining Gap

The first blocker remains `tooling_ready_waits_real_wing`. The operator still
must run `Docs/runbooks/WING_READONLY_PREFLIGHT.md` against the intended
backend/WING session, review `summary.json` and `events.jsonl`, and record an
accepted decision JSON before supervised fader/gain proof can start.
