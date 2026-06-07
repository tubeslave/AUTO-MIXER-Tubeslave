# ADR: Evidence Review Verifies Plan Summary Consistency

## Context

The WING read-only preflight artifact bundle contains an offline plan, a runtime
summary, and an event log. Evidence review now verifies plan-file SHA, but the
review should also prove that the referenced plan content matches the summary.

## Options Considered

1. Trust `expected_plan_match: true` in the summary.
2. Compare every plan field byte-for-byte against a regenerated plan.
3. Compare stable runner-built plan fields and request sequence against the
   captured summary.

## Decision

Use option 3. File-based evidence review compares stable plan fields and the
read-only request sequence against the summary/readback identities.

## Why This Won

This catches stale or tampered plan/summary bundles without making review depend
on local CLI arguments or unrelated handoff fields. It directly strengthens the
first WING read-only proof blocker.

## Rejected Alternatives

- Summary self-attestation is insufficient for manual proof.
- Full byte-for-byte regeneration is brittle because artifact review may run on
  a different machine or with different output paths.

## Implementation Plan

- Load the referenced expected plan during file-based evidence review.
- Compare stable plan metadata, policy flags, and request sequence against the
  summary.
- Add regression tests for metadata mismatch and request-sequence mismatch.
- Update runbook, report, and documentation index.

## Test Plan

- `python3 -m py_compile backend/wing_readonly_preflight_evidence.py`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_review_decision.py tests/test_wing_readonly_preflight_review_decision_tool.py tests/test_autonomy_promotion_readiness.py`
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer_no_wing_plan_summary_consistency_gate_20260607`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Risks And Rollback

Rollback would remove plan/summary content validation and allow weaker artifact
bundles. That should happen only if real WING artifacts cannot preserve the
offline plan content needed for review.
