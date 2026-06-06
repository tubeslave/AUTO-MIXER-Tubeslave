# Brief: operator-proposal-apply-sequence-policy-memory-draft-20260606

## Context

The Product Layer can turn verified sequence execution audit rows into
`policy_evidence_ready` records. The long-term AI Mixing Engineer goal needs
durable learning, but policy evidence must not be written into memory without a
separate acknowledged write contract.

## Objective

Add `draft_operator_proposal_apply_sequence_policy_memory` as a read-only
handler that converts policy-gate evidence into append-only memory draft
records.

## Safety Constraints

- Do not write `.ai/memory/project.md` or runtime memory.
- Do not mutate policy stores, models, queues, proposal status, or delta
  history.
- Do not call `apply_operator_proposal`.
- Do not call `_apply_manual_console_write`.
- Accept only `policy_evidence_ready` records from the sequence audit policy
  gate.
- Preserve `append_only_memory_write_allowed_by_draft=false`,
  `policy_mutation=false`, `live_authority_granted=false`, and
  `full_autonomous_ready=false`.

## Expected Evidence

- Valid policy evidence produces deterministic memory draft records.
- Forged policy authority, live authority, or write permission claims are
  blocked.
- The draft payload is clearly not persistence permission.
