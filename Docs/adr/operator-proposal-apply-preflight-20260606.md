# ADR: operator-proposal-apply-preflight-20260606

## Context

The operator proposal queue is the supervised bridge between analysis and live
console writes. Operators can inspect pending proposals, but before this change
there is no non-mutating way to ask whether a proposal would pass the apply
gate now.

## Options considered

1. Use `apply_operator_proposal` with missing approval as a pseudo-preflight.
2. Extract a full shared apply-validation service before adding UI behavior.
3. Add a Product Layer preflight handler that reuses current handler-local
   validation and performs no live apply.

## Decision

Use option 3. Add `preflight_operator_proposal_apply` as a dry-run Product Layer
handler.

## Why this won

It improves explainability and operator confidence without widening live write
authority. It also creates a testable contract for a future shared validation
service.

## Rejected alternatives

- Pseudo-preflight through apply was rejected because apply can reserve
  idempotency keys and mark proposal results.
- Waiting for a full extraction was rejected because the current handler-local
  validation can be reused safely for a narrow dry-run.
- Direct live promotion was rejected because broader autonomy still needs
  physical readback, rollback, freshness, and safety proof.

## Implementation plan

1. Add no-mutation cumulative-delta validation support.
2. Add the preflight handler beside `apply_operator_proposal`.
3. Return ready/blocked evidence without queue mutation, idempotency
   reservation, or manual console writes.
4. Test ready, stale pre-read, and unsupported-kind paths.
5. Record the boundary in project memory.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py::test_preflight_operator_proposal_apply_reports_ready_without_live_write tests/test_operator_product_state.py::test_preflight_operator_proposal_apply_blocks_stale_pre_read_without_mutation tests/test_operator_product_state.py::test_preflight_operator_proposal_apply_blocks_unsupported_kind_without_mutation -q`
- `PYTHONPATH=backend python3 -m py_compile backend/handlers/product_state_handlers.py`
- `/opt/homebrew/bin/git diff --check`
- `python3 tools/memory/memory_lint.py`

## Risks and rollback

Risk: preflight results may be misunderstood as permission to bypass apply.
Mitigation: the payload explicitly keeps `live_write=false`, `apply_attempted=false`,
and `live_apply_entrypoint=apply_operator_proposal`.

Rollback: remove the handler and dispatch registration. Existing proposal apply
behavior remains unchanged.
