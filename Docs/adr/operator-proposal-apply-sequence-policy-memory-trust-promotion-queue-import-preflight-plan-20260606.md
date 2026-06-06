# ADR: Operator Proposal Apply Sequence Policy Memory Trust Promotion Queue Import Preflight Plan

## Context

Sequence-policy queue import can create pending operator proposals. Existing
apply-sequence execution requires fresh batch preflight evidence before any
sequence plan, dry-run, gate, or executor path.

## Options considered

1. Let operators manually pick imported proposals from the queue.
2. Add a read-only preflight request plan for sequence-policy imports.
3. Automatically run preflight or sequence planning.

## Decision

Add a read-only sequence-policy queue import preflight plan.

## Why this won

It preserves the existing safety pipeline while making the trusted sequence
policy import path actionable and deterministic. Imported proposals are selected
in planned queue order, but approval and preflight execution remain separate.

## Rejected alternatives

Manual picking was safe but weak for autonomy and reproducibility. Automatic
preflight or sequence planning was rejected because it would blur operator
approval and existing staged gates.

## Implementation plan

- Add a module that reads pending sequence-policy imported queue actions.
- Validate import metadata and reject forged authority.
- Emit a batch-preflight request template and per-proposal request specs.
- Expose the plan in Decision Queue and via a read-only handler.
- Register the method as `shadow_only`.
- Track it in the capability matrix and project memory.

## Test plan

- Unit-test the module.
- Extend Product State tests for decision queue and handler behavior.
- Verify registry and capability matrix entries.
- Run focused pytest and memory lint.

## Risks and rollback

The main risk is implying the template has already run preflight. Rollback is to
remove the handler, decision-queue summary, registry entry, and capability row,
leaving queue import as the last active sequence-policy boundary.
