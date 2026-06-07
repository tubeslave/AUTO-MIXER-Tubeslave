# Codex Proposal: WING Read-Only Preflight Plan Summary Consistency Gate

## Short Thesis

A clean WING read-only preflight artifact should prove that its offline plan
content matches the captured summary and event-backed request sequence.

## What I Understood

The first WING MVP blocker requires high-quality real read-only proof. The
artifact bundle now verifies the plan file hash, but a hash-valid plan could
still describe a different request contract than the summary claims.

## Proposed Solution

When reviewing file artifacts, load the referenced offline plan and compare
stable runner-built fields against the summary: schema, runner, channels,
operator channel confirmation, value types, physical-readback requirement,
read-only/live-write flags, policy flags, request-id policy, and request
sequence derived from readback identities. Reject mismatches fail-closed.

## Likely Files To Touch

- `backend/wing_readonly_preflight_evidence.py`
- focused WING preflight evidence/review/decision tests
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `Docs/adr/wing-readonly-preflight-plan-summary-consistency-gate-20260607.md`
- `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_PLAN_SUMMARY_CONSISTENCY_GATE_20260607.md`

## Alternatives Considered

- Trust `expected_plan_match: true` from the summary. This is too weak for
  manual artifact review because a summary can be hand-written.
- Compare every plan field. This is more brittle because some fields are
  operator handoff context, not captured runtime evidence.

## Risks

Older or hand-edited artifacts that omit full plan content will fail review.
That is acceptable for the MVP because a real clean candidate should preserve a
self-consistent plan/summary/events bundle.

## Test Plan

- `python3 -m py_compile backend/wing_readonly_preflight_evidence.py`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_review_decision.py tests/test_wing_readonly_preflight_review_decision_tool.py tests/test_autonomy_promotion_readiness.py`
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer_no_wing_plan_summary_consistency_gate_20260607`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Where The Other Agent May Disagree

Kimi may argue the runner already validates plan arguments before WebSocket.
The stricter view is that artifact review must independently detect stale,
tampered, or hand-written bundles after capture.
