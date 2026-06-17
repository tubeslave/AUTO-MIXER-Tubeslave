# ADR: WING MVP Accepted-Candidate Prestart SHA Format Gate

Date: 2026-06-14
Status: Accepted

## Context

Accepted-candidate WING read-only review rows must prove that the future
supervised proof plan is tied to a passed no-WING prestart `summary.json` by
path and SHA256. A non-empty but malformed digest is not strong enough evidence
for this boundary.

## Options considered

1. Keep the existing non-empty digest check.
2. Add a strict 64-character lowercase SHA-256 hex check to the shared
   accepted-candidate prestart contract helper.
3. Move digest validation only into no-WING release checks.

## Decision

Use option 2. Malformed accepted-candidate prestart digests now fail the shared
contract helper with `backend_port_prestart_summary_sha256_invalid`.

## Why this won

The shared helper is consumed by review-decision acceptance and supervised
proof planning, so one narrow check protects both gates without adding any live
behavior.

## Rejected alternatives

- Non-empty only: it allows placeholder strings to masquerade as digest
  evidence.
- no-WING-only validation: it does not protect downstream ledgers or fixtures
  that are already materialized.

## Implementation plan

- Add `_is_sha256_hexdigest()` to `backend/wing_readonly_preflight_contract.py`.
- Reject malformed `backend_port_prestart_summary_sha256` in
  `wing_readonly_preflight_candidate_prestart_contract_violations()`.
- Add focused tests for the helper, review-decision ledger, and supervised
  proof plan.

## Test plan

- `PYTHONPATH=backend python -m pytest -q
  tests/test_wing_readonly_preflight_review_decision.py
  tests/test_supervised_wing_proof_plan.py --tb=short -x`
- `PYTHONPATH=backend python -m py_compile
  backend/wing_readonly_preflight_contract.py
  tests/test_wing_readonly_preflight_review_decision.py
  tests/test_supervised_wing_proof_plan.py`
- Project Director, memory lint, and `git diff --check`.

## Risks and rollback

Rollback removes the stricter helper branch and the tests. The change does not
contact hardware, mutate console state, grant live authority, or close
readiness.
