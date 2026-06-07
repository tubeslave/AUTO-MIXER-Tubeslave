# Automixer WING Read-Only Preflight Offline Plan Preview - 2026-06-07

## Summary

This slice adds an offline preview step before the real WING read-only preflight.
It helps operators verify the exact request sequence and channel confirmation
before opening a backend/WING session.

The preview is implemented in `tools/wing_readonly_preflight_runner.py` using
`--plan-only` and `--plan-output`. It shares the same argument validation and
read-request builder as the hardware path, but it does not open WebSocket,
query WING, or create physical proof artifacts.

The plan JSON also names the current MVP blocker, marks manual review as
required before any readiness update, lists fail-closed conditions, and includes
the artifact-review and decision-review commands operators should run after a
real read-only capture.

`tools/build_wing_readonly_preflight_handoff_packet.py` now includes
`preview_readonly_request_plan`, and `tools/no_wing_release_check.py` runs the
preview as an offline smoke command. The release check then reads the generated
plan JSON back as `wing_readonly_preflight_plan_artifact_schema` and validates
that it remains read-only, has no backend/WING contact, matches the expected
request sequence, and carries no live authority.

## Safety Boundary

- No live WING/OSC/MIDI/TCP command is sent by plan-only mode.
- No backend WebSocket is opened by plan-only mode.
- No console query is performed by plan-only mode.
- No physical proof artifact is created by plan-only mode.
- No readiness row, fixture gap, or live authority is updated.
- Real WING read-only evidence is still required before supervised proof steps.
- The plan artifact is validated as an offline preview only; it is not accepted
  as physical readback evidence.
- Listed post-run review commands do not execute during plan-only mode.

## Files Changed

- `.ai/briefs/wing-readonly-preflight-offline-plan-preview-20260607.md`
- `.ai/proposals/codex-wing-readonly-preflight-offline-plan-preview-20260607.md`
- `tools/wing_readonly_preflight_runner.py`
- `tools/build_wing_readonly_preflight_handoff_packet.py`
- `tools/no_wing_release_check.py`
- `tests/test_wing_readonly_preflight_runner.py`
- `tests/test_wing_readonly_preflight_handoff_packet.py`
- `tests/test_no_wing_release_check.py`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `Docs/adr/wing-readonly-preflight-offline-plan-preview-20260607.md`
- `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_OFFLINE_PLAN_PREVIEW_20260607.md`
- `Docs/INDEX.md`

## Validation

- `python3 -m py_compile tools/wing_readonly_preflight_runner.py tools/build_wing_readonly_preflight_handoff_packet.py tools/no_wing_release_check.py tools/build_read_surface_physical_readback_acceptance_template.py`
  passed.
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_runner.py tests/test_wing_readonly_preflight_handoff_packet.py tests/test_read_surface_physical_readback_fixture_acceptance_template.py tests/test_no_wing_release_check.py`
  passed: `175 passed`.
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer-no-wing-release-check-wing-plan-artifact-commit --command-timeout-sec 300`
  passed with the plan-only preview command included in the executed command
  plan and no-WING summary; `wing_readonly_preflight_plan_artifact` reported
  `status=passed`, `contacts_backend=false`, `queries_console=false`, and
  `request_sequence_matches_expected=true`.
- `python3 tools/wing_readonly_preflight_runner.py --url ws://localhost:8765 --channels 3 --operator-confirm-channels 3 --value-types fader,gain --require-physical-readback --artifact-root artifacts/wing_readonly_preflight --plan-only --plan-output /tmp/automixer-wing-readonly-plan-artifact-smoke-commit.json`
  passed; the plan type was `wing_readonly_preflight_offline_plan`, status was
  `ready_for_operator_review`, `plan_output` matched
  `/tmp/automixer-wing-readonly-plan-artifact-smoke-commit.json`,
  `contacts_backend=false`, `queries_console=false`, and
  `live_authority_granted=false`.
- `python3 tools/memory/memory_lint.py` passed.
- `/opt/homebrew/bin/git diff --check` passed.
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q` passed:
  `2952 passed, 1 skipped, 1 warning`.
- 2026-06-07 continuation validation passed after adding the selected MVP
  blocker, manual-review boundary, fail-closed conditions, and post-run review
  commands to the plan JSON:
  - `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_runner.py`
    passed inside the focused bundle: `39 passed`.
  - `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer-no-wing-release-check-acceptance-candidate-trace --command-timeout-sec 300`
    passed, with `wing_readonly_preflight_plan_artifact` reporting
    `status=passed`, `read_only=true`, `live_write=false`,
    `contacts_backend=false`, `queries_console=false`,
    `plan_status=ready_for_operator_review`, and
    `request_sequence_matches_expected=true`.
  - `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q` passed:
    `2994 passed, 1 skipped, 1 warning`.

## Remaining Gap

The preview is not hardware evidence. Operators still need to run the real
read-only preflight against the intended WING Rack, review the resulting
summary/events artifacts, and record an accepted review decision before any
readiness row can be updated.
