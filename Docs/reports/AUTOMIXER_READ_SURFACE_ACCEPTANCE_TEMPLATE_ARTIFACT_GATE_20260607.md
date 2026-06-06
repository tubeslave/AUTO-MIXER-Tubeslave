# Automixer Read-Surface Acceptance Template Artifact Gate - 2026-06-07

## Summary

This slice makes no-WING release checks validate the generated
read-surface physical-readback acceptance template artifact. The template is an
offline operator handoff: it helps an operator fill accepted/rejected records
after physical readback candidates exist, but it does not close fixture gaps or
grant live authority.

The new schema gate reads the generated JSON back and validates safety flags,
operator-required fields, pending record placeholders, record counts, blocked
reasons, focused validation commands, and record-level authority flags.

## Safety Boundary

- No WING, dLive, OSC, MIDI, TCP, backend WebSocket, or console query is used.
- No fixture gap is closed by the generated template.
- No readiness row is updated.
- No live authority or autonomous-ready state is granted.
- Pending template records must still be edited by an operator and then
  validated by the acceptance ledger.

## Files Changed

- `.ai/briefs/read-surface-acceptance-template-artifact-gate-20260607.md`
- `.ai/proposals/codex-read-surface-acceptance-template-artifact-gate-20260607.md`
- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `Docs/adr/read-surface-acceptance-template-artifact-gate-20260607.md`
- `Docs/reports/AUTOMIXER_READ_SURFACE_ACCEPTANCE_TEMPLATE_ARTIFACT_GATE_20260607.md`
- `Docs/INDEX.md`

## Validation

- `python3 -m py_compile tools/no_wing_release_check.py tools/build_read_surface_physical_readback_acceptance_template.py`
  passed.
- `PYTHONPATH=backend python3 -m pytest -q tests/test_read_surface_physical_readback_fixture_acceptance_template.py tests/test_no_wing_release_check.py::test_no_wing_release_check_validates_read_surface_acceptance_template_artifact tests/test_no_wing_release_check.py::test_no_wing_release_check_rejects_missing_read_surface_acceptance_template_artifact tests/test_no_wing_release_check.py::test_no_wing_release_check_rejects_unsafe_read_surface_acceptance_template_artifact tests/test_no_wing_release_check.py::test_no_wing_release_check_quick_mode_still_runs_write_surface_scan tests/test_no_wing_release_check.py::test_no_wing_release_check_runs_gap_rollup_schema_after_report tests/test_no_wing_release_check.py::test_no_wing_release_check_summary_embeds_read_surface_acceptance_template_artifact`
  passed: `12 passed`.
- `PYTHONPATH=backend python3 -m pytest -q tests/test_read_surface_physical_readback_fixture_acceptance_template.py tests/test_wing_readonly_preflight_handoff_packet.py tests/test_no_wing_release_check.py`
  passed: `170 passed`.
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer-no-wing-release-check-acceptance-template-artifact-gate --command-timeout-sec 300`
  passed; the summary included
  `read_surface_physical_readback_acceptance_template_artifact` with
  `status=passed`, `template_status=operator_candidate_missing`,
  `execution_allowed_by_template=false`,
  `fixture_closure_allowed_by_template=false`,
  `live_authority_granted=false`, and `violations=[]`.
- `python3 tools/memory/memory_lint.py` passed.
- `/opt/homebrew/bin/git diff --check` passed.
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q` passed:
  `2961 passed, 1 skipped, 1 warning`.

## Remaining Gap

This gate still does not provide physical readback. Operators must capture real
read-surface artifacts, review them, edit acceptance records, run the acceptance
ledger, and make a separate fixture-closure decision before any readiness
promotion can be considered.
