# ADR: Promotion Memory Draft Result Consumer

## Context

Promotion audit-policy-gate evidence can now be verified and converted into a memory-draft request template. The next boundary must consume the generic policy-memory draft result without writing memory or widening live authority.

## Options Considered

1. Add a promotion wrapper around the generic memory-draft result consumer.
2. Let callers use the generic consumer directly.
3. Invoke the memory writer directly after draft success.

## Decision

Add a promotion wrapper around the generic memory-draft result consumer.

## Why This Won

The wrapper preserves promotion provenance and makes the next memory-writer request visible without combining verification with persistence. It reuses the generic draft validator and keeps the promotion chain explainable in Product State, the registry, and the project-goal matrix.

## Rejected Alternatives

Using the generic consumer directly loses promotion-specific envelope and request validation. Invoking the writer directly would cross the persistence boundary and obscure whether evidence was only verified or written.

## Implementation Plan

- Add a read-only promotion memory-draft result consumer.
- Wire it into Product State dispatch.
- Register it as `shadow_only` decision evidence.
- Add focused and Product State tests.
- Record the safety rule in `.ai/memory/project.md`.

## Test Plan

- Verify not-ready, ready, forged, and blocked states.
- Verify registry and goal-matrix non-live status.
- Verify Product State dispatch emits a memory-writer request template with `apply=false` and `operator_ack=false`.
- Run `py_compile`, focused pytest, memory lint, and diff checks.

## Risks And Rollback

Risk: a malformed memory draft could be treated as writer input. Mitigation: delegate to the generic memory-draft result consumer and require all promotion provenance fields to match. Rollback: remove the new handler dispatch, registry/matrix entries, and module; prior promotion audit-policy-gate behavior remains unchanged.
