# ADR: Evidence Requires Existing Backend Connection Boundary

## Context

The WING read-only preflight runner must not open or close hardware mixer
connections directly. It should query the already-running backend session that
the operator prepared for the intended WING Rack.

Recent runner artifacts include
`physical_preflight_requires_existing_backend_connection: true`, but evidence
intake did not require this field before accepting a clean candidate.

## Options Considered

1. Keep the field optional and rely on forbidden request checks.
2. Require the field only in evidence intake.
3. Require the field in evidence intake and in the offline expected-plan match.

## Decision

Use option 3. A clean WING read-only preflight candidate must include
`physical_preflight_requires_existing_backend_connection: true`, and the offline
plan preview must contain the same boundary.

## Why This Won

The first MVP blocker is not just "some readback happened". It must prove a safe
operator-prepared path: backend already connected to WING, runner read-only,
no direct connect/disconnect authority, no live-write authority.

## Rejected Alternatives

- Event-log-only proof catches connect/disconnect messages after the fact but
  does not show the operator-approved plan carried the boundary.
- Backward compatibility with older summaries would keep weaker artifacts
  eligible as clean candidates.

## Implementation Plan

- Add the validation check in `backend/wing_readonly_preflight_evidence.py`.
- Add the field to the offline plan and expected-plan match keys in
  `tools/wing_readonly_preflight_runner.py`.
- Update valid fixtures and add a missing-field regression test.
- Update the WING read-only preflight runbook and documentation index.

## Test Plan

- `python3 -m py_compile backend/wing_readonly_preflight_evidence.py tools/wing_readonly_preflight_runner.py`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_runner.py tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_review_decision.py tests/test_wing_readonly_preflight_review_decision_tool.py tests/test_autonomy_promotion_readiness.py`
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer_no_wing_existing_backend_connection_evidence_gate_20260607`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Risks And Rollback

Rollback would remove the field requirement and allow older, weaker preflight
summaries again. That should happen only if current real WING runner artifacts
cannot include the boundary field, which runner code and tests currently
contradict.
