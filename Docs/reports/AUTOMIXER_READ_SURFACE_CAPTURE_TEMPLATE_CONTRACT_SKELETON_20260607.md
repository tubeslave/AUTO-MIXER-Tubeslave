# Automixer Read-Surface Capture Template Contract Skeleton - 2026-06-07

## Summary

This slice adds a contract-derived `physical_readback_payload_template` to each
offline pending read-surface capture-template record. The template now shows the
operator the expected artifact type, schema id, required top-level fields,
nested required field sets, placeholder payload shape, and nested safety flags
for the target physical readback contract.

## Safety Boundary

- No WING, dLive, OSC, MIDI, TCP, backend WebSocket, or console query is used.
- No physical readback artifact is accepted.
- No fixture baseline is modified.
- No physical fixture gap is closed.
- No live widening, live authority, fixture candidacy, or full autonomous
  readiness is granted.
- The outer `physical_readback_payload` remains an operator-replaced placeholder;
  the nested skeleton is only a fill guide.

## Files Changed

- `.ai/briefs/read-surface-capture-template-contract-skeleton-20260607.md`
- `.ai/proposals/codex-read-surface-capture-template-contract-skeleton-20260607.md`
- `tools/build_read_surface_physical_readback_capture_template.py`
- `tools/no_wing_release_check.py`
- `tests/test_read_surface_physical_readback_capture_template.py`
- `tests/test_no_wing_release_check.py`
- `Docs/adr/read-surface-capture-template-contract-skeleton-20260607.md`
- `Docs/reports/AUTOMIXER_READ_SURFACE_CAPTURE_TEMPLATE_CONTRACT_SKELETON_20260607.md`
- `Docs/INDEX.md`

## Validation

- `python3 -m py_compile tools/build_read_surface_physical_readback_capture_template.py tools/no_wing_release_check.py`
  passed.
- `PYTHONPATH=backend python3 -m pytest -q tests/test_read_surface_physical_readback_capture_template.py tests/test_no_wing_release_check.py::test_no_wing_release_check_validates_read_surface_capture_template_artifact tests/test_no_wing_release_check.py::test_no_wing_release_check_rejects_unsafe_read_surface_capture_template_artifact tests/test_no_wing_release_check.py::test_no_wing_release_check_summary_embeds_read_surface_capture_template_artifact`
  passed: `8 passed`.
- `PYTHONPATH=backend python3 -m pytest -q tests/test_read_surface_physical_readback_capture_packet.py tests/test_read_surface_physical_readback_capture_template.py tests/test_read_surface_physical_readback_artifact_review_queue.py tests/test_read_surface_physical_readback_fixture_acceptance_template.py tests/test_read_surface_physical_readback_fixture_closure_plan.py tests/test_no_wing_release_check.py`
  passed: `216 passed`.
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer-no-wing-release-check-capture-template-skeleton --command-timeout-sec 300`
  passed; the generated summary included
  `read_surface_physical_readback_capture_template_artifact` with
  `status=passed`, `template_status=operator_capture_template_ready`,
  `read_only=true`, `live_write=false`, `queries_console=false`,
  `executes_tests=false`, `operator_required=true`,
  `execution_allowed_by_template=false`,
  `accepted_as_physical_fixture_candidate=false`,
  `fixture_closure_allowed_by_template=false`,
  `live_authority_granted=false`, `full_autonomous_ready=false`,
  `capture_template_surface_ids=["raw_mixer_state_cache"]`,
  `capture_template_contract_ids=["raw_mixer_state_physical_readback_contract"]`,
  `payload_template_contract_ids=["raw_mixer_state_physical_readback_contract"]`,
  `payload_template_artifact_types=["raw_mixer_state_physical_readback"]`,
  `payload_template_schema_ids=["raw_mixer_state_physical_readback.v1"]`,
  `payload_template_field_counts=[10]`,
  `required_operator_edits` including
  `physical_readback_payload_template.payload`, `record_unsafe_flags=[]`, and
  `violations=[]`.
- `python3 tools/memory/memory_lint.py` passed.
- `/opt/homebrew/bin/git diff --check` passed.
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q` passed:
  `2989 passed, 1 skipped, 1 warning`.

## Remaining Gap

The template now gives operators a contract-shaped fill guide, but it still does
not provide physical readback artifacts, accepted operator records, fixture
baseline updates, closed read-surface gaps, or live readiness.
