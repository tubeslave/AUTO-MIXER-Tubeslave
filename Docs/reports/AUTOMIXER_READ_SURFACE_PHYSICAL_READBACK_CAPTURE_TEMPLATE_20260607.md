# Automixer Read-Surface Physical Readback Capture Template - 2026-06-07

## Summary

This slice adds a pending capture-template artifact for read-surface physical
readback. The generated template sits between the capture packet and the
artifact-review queue: it tells an operator which fields must be replaced with
real physical readback data, while staying invalid as proof until the filled
artifact goes through the matching contract, intake, review queue, acceptance
ledger, and separate fixture-closure decision.

## Safety Boundary

- No WING, dLive, OSC, MIDI, TCP, backend WebSocket, or console query is used.
- The generated template is not physical proof.
- The template keeps placeholders for operator-edited capture fields.
- The no-WING gate rejects filled proof payloads under the pending-template
  schema.
- No physical readback artifact is accepted.
- No fixture baseline is modified.
- No physical fixture gap is closed.
- No live widening, live authority, or full autonomous readiness is granted.

## Files Changed

- `.ai/briefs/read-surface-physical-readback-capture-template-20260607.md`
- `.ai/proposals/codex-read-surface-physical-readback-capture-template-20260607.md`
- `tools/build_read_surface_physical_readback_capture_template.py`
- `tools/no_wing_release_check.py`
- `tests/test_read_surface_physical_readback_capture_template.py`
- `tests/test_no_wing_release_check.py`
- `Docs/adr/read-surface-physical-readback-capture-template-20260607.md`
- `Docs/reports/AUTOMIXER_READ_SURFACE_PHYSICAL_READBACK_CAPTURE_TEMPLATE_20260607.md`
- `Docs/INDEX.md`

## Validation

- `python3 -m py_compile tools/build_read_surface_physical_readback_capture_template.py tools/no_wing_release_check.py`
  passed.
- `PYTHONPATH=backend python3 -m pytest -q tests/test_read_surface_physical_readback_capture_template.py`
  passed: `5 passed`.
- `PYTHONPATH=backend python3 -m pytest -q tests/test_no_wing_release_check.py::test_no_wing_release_check_validates_read_surface_capture_template_artifact tests/test_no_wing_release_check.py::test_no_wing_release_check_rejects_missing_read_surface_capture_template_artifact tests/test_no_wing_release_check.py::test_no_wing_release_check_rejects_unsafe_read_surface_capture_template_artifact tests/test_no_wing_release_check.py::test_no_wing_release_check_quick_mode_still_runs_write_surface_scan tests/test_no_wing_release_check.py::test_no_wing_release_check_runs_gap_rollup_schema_after_report tests/test_no_wing_release_check.py::test_no_wing_release_check_summary_embeds_read_surface_capture_template_artifact`
  passed: `6 passed`.
- `PYTHONPATH=backend python3 -m pytest -q tests/test_read_surface_physical_readback_capture_packet.py tests/test_read_surface_physical_readback_capture_template.py tests/test_read_surface_physical_readback_artifact_review_queue.py tests/test_read_surface_physical_readback_fixture_acceptance_template.py tests/test_read_surface_physical_readback_fixture_closure_plan.py tests/test_no_wing_release_check.py`
  passed: `216 passed`.
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer-no-wing-release-check-capture-template --command-timeout-sec 300`
  passed; the generated summary included
  `read_surface_physical_readback_capture_template_artifact` with
  `status=passed`, `template_status=operator_capture_template_ready`,
  `read_only=true`, `live_write=false`, `queries_console=false`,
  `executes_tests=false`, `execution_allowed_by_template=false`,
  `accepted_as_physical_fixture_candidate=false`,
  `fixture_closure_allowed_by_template=false`,
  `live_authority_granted=false`, `full_autonomous_ready=false`,
  `all_remaining_target_count=10`, `capture_template_count=1`,
  `capture_template_surface_ids=["raw_mixer_state_cache"]`,
  `capture_template_contract_ids=["raw_mixer_state_physical_readback_contract"]`,
  `record_unsafe_flags=[]`, and `violations=[]`.
- `python3 tools/memory/memory_lint.py`
  passed.
- `/opt/homebrew/bin/git diff --check`
  passed.
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`
  passed: `2989 passed, 1 skipped, 1 warning`.

## Remaining Gap

This template makes the next operator artifact shape explicit, but it still does
not provide physical readback artifacts, accepted operator records, fixture
baseline updates, or live readiness.
