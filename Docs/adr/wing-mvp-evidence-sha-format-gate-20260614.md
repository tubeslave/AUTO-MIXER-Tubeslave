# ADR: WING MVP Evidence SHA Format Gate

Date: 2026-06-14
Status: Accepted

## Context

WING read-only evidence candidates bind to an offline expected plan and a
no-WING prestart summary through SHA-256 digests. File-based review compares
referenced files against those digests, but the base evidence validator should
also reject malformed digest values before a candidate is considered valid.

## Options considered

1. Keep checking only for non-empty digest strings.
2. Reuse `is_sha256_hexdigest()` in evidence intake for expected-plan and
   prestart-summary digests.
3. Defer all digest validation to file-based review.

## Decision

Use option 2. Evidence intake now rejects malformed `expected_plan_sha256` and
`backend_port_prestart_summary_sha256` values.

## Why this won

The evidence validator is an upstream readiness gate. Enforcing the shared
digest format there keeps inline/candidate and file-based evidence on the same
contract.

## Rejected alternatives

- Non-empty-only checks: allow placeholder strings.
- File-review-only checks: leave weaker behavior for validation entrypoints
  that do not yet have file paths.

## Implementation plan

- Import `is_sha256_hexdigest()` into `wing_readonly_preflight_evidence.py`.
- Emit `expected_plan_sha256_invalid` for malformed expected-plan digests.
- Emit `backend_port_prestart_summary_sha256_invalid` for malformed prestart
  summary digests.
- Add focused evidence regression tests.

## Test plan

- `PYTHONPATH=backend python -m pytest -q
  tests/test_wing_readonly_preflight_evidence.py`
- Full no-WING release-check module.
- Project Director rails, py_compile, memory lint, and `git diff --check`.

## Risks and rollback

Rollback removes the helper import, invalid-digest checks, tests, and report.
The change is offline/review-only and does not contact hardware, mutate console
state, grant live authority, or close readiness.
