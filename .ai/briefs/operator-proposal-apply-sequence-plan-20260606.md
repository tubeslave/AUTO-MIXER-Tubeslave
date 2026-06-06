# Brief: operator-proposal-apply-sequence-plan-20260606

## Context

The Product Layer can now dry-run a batch of pending apply candidates through
`preflight_operator_proposal_apply_batch`. Operators still need a deterministic
sequence view that turns ready preflight results into explicit supervised apply
requests without executing them.

## Objective

Add a read-only Product Layer sequence planner that accepts preflight results,
keeps only safe ready results, and returns the ordered `apply_operator_proposal`
requests an operator could approve one by one.

## Safety Constraints

- The planner must require caller-supplied preflight results.
- Do not run preflight from the planner.
- Do not call `_apply_manual_console_write`.
- Do not reserve idempotency keys.
- Do not mark proposal status, move queue entries, or mutate delta history.
- Do not widen live write kinds beyond the existing supervised fader/gain gate.
- The plan must not itself grant authority to execute.

## Expected Evidence

- Ready preflight results become ordered plan steps.
- Blocked or malformed preflight results remain blocked in the plan.
- Forged live-write/idempotency/unsupported-kind evidence is rejected.
- Queue state, idempotency storage, cumulative-delta history, and mixer writes
  remain unchanged by planning.
