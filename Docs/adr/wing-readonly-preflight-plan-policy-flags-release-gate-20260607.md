# ADR: No-WING Release Gate Requires Preflight Plan Policy Flags

## Context

The WING read-only preflight flow uses an offline plan before a real backend
WebSocket run. The plan is the operator-facing contract for the first MVP
hardware step.

The runner emits `expected_plan_required_for_real_run: true` and
`physical_preflight_requires_existing_backend_connection: true`. Evidence
intake also requires these boundaries. The no-WING release checker should reject
plan artifacts that omit either field.

## Options Considered

1. Keep no-WING release validation focused on read-only request sequence only.
2. Require only `physical_preflight_requires_existing_backend_connection`.
3. Require both policy flags and expose them in release summaries.

## Decision

Use option 3. The no-WING release gate now requires both policy flags in the
offline preflight plan artifact and surfaces them in summary JSON and Markdown.

## Why This Won

The next real WING run should not depend on implicit assumptions. The handoff
plan must prove that real runs require the expected plan and use an
operator-prepared backend/WING connection boundary before the hardware session
opens.

## Rejected Alternatives

- Runtime-only checks catch failures too late in the operator sequence.
- Checking only one flag would leave the plan/evidence contract split across
  different tools.

## Implementation Plan

- Add no-WING validation for both fields.
- Add both fields to plan artifact summaries.
- Add the existing-backend boundary to runner fail-closed conditions.
- Update focused tests and runbook wording.

## Test Plan

- `python3 -m py_compile tools/no_wing_release_check.py tools/wing_readonly_preflight_runner.py`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_no_wing_release_check.py tests/test_wing_readonly_preflight_runner.py`
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root artifacts/no_wing_release_check_current_cycle`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Risks And Rollback

Rollback would allow stale offline plan artifacts to pass release checks without
showing the current safety policy. That should happen only if the runner can no
longer emit these fields, which current runner code contradicts.
