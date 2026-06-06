# Brief: operator-proposal-apply-sequence-policy-memory-writer-20260606

## Context

The Product Layer can draft append-only memory records from verified supervised
sequence policy evidence. The long-term Automixer goal requires learning from
previous safe decisions, but writing memory must remain an explicit operator
action.

## Objective

Add `write_operator_proposal_apply_sequence_policy_memory` as an append-only
writer for sequence policy memory drafts.

## Safety Constraints

- Write only with `apply=true` and `operator_ack=true`.
- Write only append-only blocks under `.ai/memory/project.md`.
- Reject absolute paths, path traversal, forged write permission, live authority,
  queue/model/policy/runtime-memory mutation claims, or non-fader/gain evidence.
- Do not call `apply_operator_proposal`.
- Do not call `_apply_manual_console_write`.
- Do not mutate queues, models, policy stores, or live mixer state.

## Expected Evidence

- Without operator acknowledgement, the writer reports ack required and writes
  nothing.
- With valid draft records, `apply=true`, and `operator_ack=true`, the writer
  appends a deterministic block to the allowed target memory file.
- Forged draft authority flags are blocked before any file write.
