# Codex Proposal: WING Read-Only Preflight Plan SHA File Evidence Gate

## Short Thesis

A clean WING read-only preflight file artifact should prove that its
`expected_plan_sha256` matches the offline plan file named in
`expected_plan_path`.

## What I Understood

The first WING MVP blocker still waits for a real read-only hardware artifact.
Current summaries record expected-plan metadata, but evidence review can accept
a file artifact without hashing the referenced plan file.

## Proposed Solution

During file-based artifact validation, resolve `expected_plan_path`, require the
file to exist, hash it with SHA256, and reject the artifact if the hash differs
from `expected_plan_sha256`. Keep inline payload validation unchanged because
inline unit tests and overlays do not have a filesystem artifact bundle.

## Likely Files To Touch

- `backend/wing_readonly_preflight_evidence.py`
- focused WING preflight evidence/review/decision tests
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `Docs/adr/wing-readonly-preflight-plan-sha-file-evidence-gate-20260607.md`
- `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_PLAN_SHA_FILE_EVIDENCE_GATE_20260607.md`

## Alternatives Considered

- Require only non-empty SHA metadata. This is weaker and allows hand-written
  summaries to claim a plan match.
- Require plan-file hashing for inline payloads. That would make non-file
  in-memory evidence overlays depend on filesystem state they do not own.

## Risks

Old preflight artifact folders without the offline plan file will fail review.
That is acceptable because a real clean candidate should carry its plan proof
bundle.

## Test Plan

- `python3 -m py_compile backend/wing_readonly_preflight_evidence.py`
- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_readonly_preflight_evidence.py tests/test_wing_readonly_preflight_review_tool.py tests/test_wing_readonly_preflight_review_decision.py tests/test_wing_readonly_preflight_review_decision_tool.py tests/test_autonomy_promotion_readiness.py`
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer_no_wing_plan_sha_file_evidence_gate_20260607`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`

## Where The Other Agent May Disagree

Kimi may argue event-log validation is enough. The stronger version is that
event logs prove request/response behavior while the plan SHA proves the
operator-approved request plan was the same artifact the real run claims.
