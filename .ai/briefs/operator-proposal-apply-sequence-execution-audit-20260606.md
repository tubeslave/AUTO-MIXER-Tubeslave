# Brief: operator-proposal-apply-sequence-execution-audit-20260606

## Context

The Product Layer now has a supervised sequence executor that can apply
gate-ready fader/gain proposals by delegating to `apply_operator_proposal`.
Future automation needs evidence that an executor result was produced through
the approved path and did not forge direct write authority.

## Objective

Add an audit-only handler for sequence execution results. The handler should
validate the executor payload and emit pass/block evidence without touching the
queue, idempotency store, mixer state, or manual console write path.

## Safety Constraints

- Do not call `apply_operator_proposal`.
- Do not call `_apply_manual_console_write`.
- Do not reserve idempotency keys.
- Do not mutate queue, proposal status, delta history, runtime memory, or mixer
  state.
- Require executor payload safety to say it used `apply_operator_proposal`.
- Reject direct-write claims, widened live-write kinds, missing idempotency, and
  unverified applied row evidence.

## Expected Evidence

- A valid executor result with verified fader/gain rows audits as passed.
- Forged direct-write, live-widening, missing idempotency, or unverified
  readback claims audit as blocked.
- The audit payload is clearly read-only and not an execution grant.
