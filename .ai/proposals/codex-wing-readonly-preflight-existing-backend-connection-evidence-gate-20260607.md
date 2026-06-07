# Codex Proposal: WING Read-Only Preflight Existing Backend Connection Evidence Gate

## Short Thesis

Accepted WING read-only preflight artifacts should prove that the runner used an
already-connected backend/WING session instead of carrying any connect or
disconnect authority.

## What I Understood

The current MVP must close the WING read-only proof first. The runner already
records `physical_preflight_requires_existing_backend_connection: true`, but the
evidence validator does not yet require that field.

## Proposed Solution

Require `physical_preflight_requires_existing_backend_connection: true` in
`validate_wing_readonly_preflight_artifact`. Add the same field to the offline
plan preview and expected-plan match keys so mismatched plans fail before
WebSocket. Update valid fixtures, add a regression test for missing metadata,
and document the operator-facing runbook expectation.

## Likely Files To Touch

- `backend/wing_readonly_preflight_evidence.py`
- `tools/wing_readonly_preflight_runner.py`
- focused WING preflight tests
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `Docs/adr/wing-readonly-preflight-existing-backend-connection-evidence-gate-20260607.md`
- `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_EXISTING_BACKEND_CONNECTION_EVIDENCE_GATE_20260607.md`

## Alternatives Considered

- Leave the field as optional runner metadata. This preserves stale artifacts
  that do not prove the boundary.
- Check only the event log for absent connect/disconnect messages. That is
  necessary but does not prove the runner contract expected an existing backend
  session.

## Risks

Older captured summaries will stop being accepted as clean candidates. That is
acceptable for the current MVP because real proof quality matters more than
backward compatibility.

## Test Plan

- `python3 -m py_compile backend/wing_readonly_preflight_evidence.py tools/wing_readonly_preflight_runner.py`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_runner.py tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_review_decision.py tests/test_wing_readonly_preflight_review_decision_tool.py tests/test_autonomy_promotion_readiness.py`
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer_no_wing_existing_backend_connection_evidence_gate_20260607`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Where The Other Agent May Disagree

Kimi may argue the event-log forbidden request checks already cover this. The
stronger version is that event logs prove what happened, while this field proves
the runner policy the operator approved before the real session opened.
