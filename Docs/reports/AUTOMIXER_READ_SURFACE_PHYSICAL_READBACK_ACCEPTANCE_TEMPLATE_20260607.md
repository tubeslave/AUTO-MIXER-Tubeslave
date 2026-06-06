# Automixer Read Surface Physical Readback Acceptance Template - 2026-06-07

## Summary

This slice adds an offline operator template for read-surface physical readback
fixture acceptance records. It supports the broader "read all console state"
goal by making the manual review handoff more concrete after a physical
readback artifact has passed intake and review-queue checks.

`tools/build_read_surface_physical_readback_acceptance_template.py` reads the
aggregate review queue and emits JSON or Markdown. It only creates template
records for valid review candidates. When no candidate exists, it reports
`operator_candidate_missing` and does not create a fake acceptance record.

## Safety Boundary

- No WING, dLive, OSC, MIDI, TCP, or WebSocket transport was added.
- No console query or mutation is performed by the template builder.
- No fixture gap is closed by the template.
- No readiness row is updated by the template.
- No live authority or autonomous-ready state is granted.
- An unedited template remains invalid for the acceptance ledger because
  decision and reviewer fields are placeholders.

## Files Changed

- `.ai/briefs/read-surface-fixture-acceptance-template-20260607.md`
- `.ai/proposals/codex-read-surface-fixture-acceptance-template-20260607.md`
- `tools/build_read_surface_physical_readback_acceptance_template.py`
- `tools/no_wing_release_check.py`
- `tests/test_read_surface_physical_readback_fixture_acceptance_template.py`
- `tests/test_no_wing_release_check.py`
- `Docs/adr/read-surface-physical-readback-acceptance-template-20260607.md`
- `Docs/reports/AUTOMIXER_READ_SURFACE_PHYSICAL_READBACK_ACCEPTANCE_TEMPLATE_20260607.md`
- `Docs/INDEX.md`

## Validation

- `python3 -m py_compile tools/build_read_surface_physical_readback_acceptance_template.py tools/no_wing_release_check.py`
  passed.
- `PYTHONPATH=backend python3 -m pytest -q tests/test_read_surface_physical_readback_fixture_acceptance_template.py tests/test_read_surface_physical_readback_fixture_acceptance_ledger.py tests/test_no_wing_release_check.py::test_no_wing_release_check_quick_mode_still_runs_write_surface_scan tests/test_no_wing_release_check.py::test_no_wing_release_check_validates_fixture_acceptance_ledger_snapshot`
  passed: `17 passed`.
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer-no-wing-release-check-acceptance-template --command-timeout-sec 300`
  passed with the acceptance-template smoke command included in the executed
  command plan.
- `PYTHONPATH=backend python3 -m pytest -q tests/test_read_surface_physical_readback_fixture_acceptance_template.py tests/test_no_wing_release_check.py`
  passed: `149 passed`.
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_runner.py tests/test_wing_readonly_preflight_handoff_packet.py tests/test_read_surface_physical_readback_fixture_acceptance_template.py tests/test_no_wing_release_check.py`
  passed after the acceptance-template and WING plan-artifact gates were both
  wired through no-WING release check: `175 passed`.
- `python3 tools/build_read_surface_physical_readback_acceptance_template.py --format json --output /tmp/automixer-read-surface-acceptance-template-smoke.json --quiet`
  passed; the template type was
  `read_surface_physical_readback_fixture_acceptance_template`, status was
  `operator_candidate_missing`, `template_record_count` was `0`, and
  `live_authority_granted` was `false`.
- `python3 tools/memory/memory_lint.py` passed.
- `/opt/homebrew/bin/git diff --check` passed.
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q` passed:
  `2952 passed, 1 skipped, 1 warning`.

## Remaining Gap

This template does not prove real console readback by itself. Operators still
must capture physical artifacts from the intended console, review them, edit the
template into an accepted or rejected record, run the acceptance ledger, and
make any fixture-closure decision separately.
