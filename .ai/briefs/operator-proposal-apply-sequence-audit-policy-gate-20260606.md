# Brief: operator-proposal-apply-sequence-audit-policy-gate-20260606

## Context

The Product Layer can execute gate-ready fader/gain proposal sequences and
audit executor result evidence. Future Supervised AutoMix policy should not
consume raw executor output; it needs a read-only policy gate that accepts only
passed audit evidence.

## Objective

Add `gate_operator_proposal_apply_sequence_audit_policy` as a read-only handler
that converts passed execution-audit rows into policy evidence candidates for
future supervised automation.

## Safety Constraints

- Do not call `apply_operator_proposal`.
- Do not call `_apply_manual_console_write`.
- Do not reserve idempotency keys.
- Do not mutate queue, proposal status, delta history, runtime memory, models,
  policy stores, or mixer state.
- Accept only `operator_proposal_apply_sequence_execution_audit` payloads with
  `status=passed`.
- Preserve `execution_allowed_by_policy_gate=false`,
  `live_authority_granted=false`, and fader/gain-only live-write scope.

## Expected Evidence

- Passed audit rows become `policy_evidence_ready` candidates.
- Blocked audit payloads, forged authority flags, widened write kinds, or
  non-fader/gain audit rows are rejected.
- Policy gate output is not execution permission or persistence permission.
