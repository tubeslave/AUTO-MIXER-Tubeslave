# ADR: operator-proposal-apply-sequence-plan-20260606

## Context

The Product Layer exposes single and batch apply preflight. A deterministic
plan is needed between preflight readiness and any future executor so operators
can see the exact supervised apply requests before writes are attempted.

## Options Considered

1. Run batch preflight automatically inside a planner.
2. Add a read-only planner that requires caller-supplied preflight results.
3. Add a batch executor that applies all ready preflight results.

## Decision

Use option 2. Add `plan_operator_proposal_apply_sequence` as a read-only handler
that converts verified ready preflight results into ordered
`apply_operator_proposal` request payloads.

## Why This Won

It advances the autonomous apply pipeline while preserving the current safety
boundary: preflight, planning, and live apply remain separate contracts.

## Rejected Alternatives

- Hidden preflight was rejected because planning should not query pre-read state
  or produce surprise latency.
- Batch execution was rejected because readiness is not proof of a successful
  live write and does not replace operator approval.
- Frontend-only planning was rejected because forged safety/idempotency fields
  need server-side validation before any later executor can trust the shape.

## Implementation Plan

1. Add a Product Layer handler and dispatch registration.
2. Require `preflight_results` input.
3. Accept only ready results with no-live-write safety flags, unreserved
   idempotency keys, and normalized `fader`/`gain` changes.
4. Return ordered plan steps with explicit `apply_operator_proposal` requests.
5. Add Decision Queue visibility for the planner.
6. Test no live writes, no queue mutation, no idempotency reservation, and
   forged-result rejection.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py::test_plan_operator_proposal_apply_sequence_builds_read_only_plan_from_batch_preflight -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py::test_plan_operator_proposal_apply_sequence_rejects_forged_ready_result -q`
- `PYTHONPATH=backend python3 -m py_compile backend/handlers/product_state_handlers.py backend/operator_product_state.py`
- `/opt/homebrew/bin/git diff --check`
- `python3 tools/memory/memory_lint.py`

## Risks And Rollback

Risk: operators may treat a plan as permission to execute. Mitigation: the
payload keeps `execution_allowed_by_plan=false`, `requires_operator_apply=true`,
and `live_apply_entrypoint=apply_operator_proposal`.

Rollback: remove the planner handler, dispatch key, and Decision Queue
visibility. Batch preflight and supervised apply remain unchanged.
