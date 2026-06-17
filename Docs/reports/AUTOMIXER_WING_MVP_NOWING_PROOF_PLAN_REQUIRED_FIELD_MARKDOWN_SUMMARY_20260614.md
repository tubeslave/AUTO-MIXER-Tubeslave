# WING MVP no-WING Proof Plan Required Field Markdown Summary

## Summary

The no-WING `summary.md` now renders the required and missing supervised
proof-plan summary field lists from `next_wing_proof_plan`.

## Changed

- `tools/no_wing_release_check.py` prints `required_supervised_proof_plan_summary_fields`
  in the markdown `## Next WING Proof Plan` section.
- `tools/no_wing_release_check.py` prints `missing_required_supervised_proof_plan_summary_fields`
  in the same section.
- `tests/test_no_wing_release_check.py` verifies the successful empty-missing
  path and the stale-summary missing-field path both render correctly.
- `Docs/INDEX.md` and the first WING blocker evidence row include this report,
  so Project Director selected evidence stays synchronized with indexed WING MVP
  proof reports.

## Safety Boundary

This is an offline report-rendering change. It does not contact WING, query the
console, grant live authority, or authorize any fader/gain movement.

## Validation

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_no_wing_release_check.py::test_no_wing_release_check_summary_embeds_next_wing_proof_plan tests/test_no_wing_release_check.py::test_no_wing_release_check_rejects_next_wing_plan_missing_required_summary_field` passed with 32 tests.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_wing_readonly_preflight_handoff_packet.py tests/test_no_wing_release_check.py -k "next_wing_proof_plan or required_supervised_proof_plan_summary_fields"` passed with 19 tests and 383 deselected.
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest -q tests/test_wing_mvp_rails.py tests/test_project_director_agent.py` passed with 12 tests.

## Remaining Blocker

The WING MVP remains blocked on a real WING read-only preflight capture and an
operator-reviewed accepted candidate before any supervised fader/gain proof run.
