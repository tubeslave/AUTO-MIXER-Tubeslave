# ADR: Evidence Review Verifies Expected Plan File SHA

## Context

Real WING read-only preflight summaries include an expected offline plan path
and SHA. Evidence intake already rejects summaries that omit the metadata, but a
file artifact review should also verify the referenced plan file.

## Options Considered

1. Keep checking only that `expected_plan_sha256` is present.
2. Verify the expected plan file exists and matches the recorded SHA for
   file-based artifacts.
3. Require plan-file verification for every inline payload as well.

## Decision

Use option 2. When a `summary.json` is loaded from disk, evidence review resolves
`expected_plan_path`, hashes the referenced file, and rejects the candidate if
the file is missing or the SHA mismatches.

## Why This Won

The current MVP needs high-quality real proof. A clean candidate should be a
self-consistent artifact bundle: offline plan, real summary, and event log.
Inline validation remains transport-free and filesystem-free for unit overlays.

## Rejected Alternatives

- Metadata-only validation lets hand-written summaries claim plan proof.
- Requiring plan files for inline payloads would over-constrain non-file test
  and overlay paths.

## Implementation Plan

- Add expected-plan file resolution and SHA hashing to
  `backend/wing_readonly_preflight_evidence.py`.
- Update file-artifact tests to write a real expected plan file.
- Add missing-plan and tampered-plan regression tests.
- Update runbook, report, and documentation index.

## Test Plan

- `python3 -m py_compile backend/wing_readonly_preflight_evidence.py`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_review_decision.py tests/test_wing_readonly_preflight_review_decision_tool.py tests/test_autonomy_promotion_readiness.py`
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer_no_wing_plan_sha_file_evidence_gate_20260607`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Risks And Rollback

Rollback would remove plan-file hashing and again allow weaker evidence bundles.
That should happen only if real WING artifact storage cannot preserve the
offline plan file next to or referenced by the summary.
