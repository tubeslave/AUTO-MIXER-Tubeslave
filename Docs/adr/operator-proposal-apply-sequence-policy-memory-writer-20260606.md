# ADR: operator-proposal-apply-sequence-policy-memory-writer-20260606

## Context

The sequence policy memory draft can produce safe learning records from verified
supervised fader/gain execution evidence, but those drafts are not durable
memory. The project needs a controlled way to persist such lessons.

## Options Considered

1. Keep drafts only.
2. Add an acknowledged append-only writer to `.ai/memory/project.md`.
3. Add a generic file writer for arbitrary memory targets.

## Decision

Use option 2. Add `write_operator_proposal_apply_sequence_policy_memory` to
append valid sequence policy memory drafts to `.ai/memory/project.md` only when
`apply=true` and `operator_ack=true`.

## Why This Won

It advances long-term learning while preserving narrow authority. The writer is
memory-only, append-only, explicit, and follows the existing source-identity
memory writer pattern.

## Rejected Alternatives

- Draft-only was rejected because it stops short of durable learning.
- Generic file writes were rejected because they widen the authority surface.
- Silent runtime-memory mutation was rejected because learning writes must be
  reviewable in repo-local project memory.

## Implementation Plan

1. Extend draft records with allowed target path and deterministic draft text.
2. Add writer request parsing for `memory_draft_payload`.
3. Validate draft envelope, summary, safety flags, and records.
4. Require `apply=true` and `operator_ack=true` before writing.
5. Append only to `.ai/memory/project.md` under the selected root.
6. Advertise the writer in Decision Queue visibility.

## Test Plan

- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py::test_write_operator_proposal_apply_sequence_policy_memory_appends_with_operator_ack -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_operator_product_state.py::test_write_operator_proposal_apply_sequence_policy_memory_requires_operator_ack -q`
- `PYTHONPATH=backend python3 -m py_compile backend/handlers/product_state_handlers.py backend/operator_product_state.py`
- `/opt/homebrew/bin/git diff --check`
- `python3 tools/memory/memory_lint.py`

## Risks And Rollback

Risk: memory write authority may be confused with live authority. Mitigation:
writer safety flags keep live writes, queue mutation, model mutation, policy
mutation, and Full Autonomous readiness false.

Rollback: remove the writer handler and Decision Queue discovery fields. Memory
draft, policy gate, audit, executor, and earlier sequence stages remain
available.
