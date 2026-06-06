# ADR: Promotion Memory Writer Dry Run Result Consumer

## Context

Promotion memory-draft evidence can now be verified and converted into a memory-writer request template. The next boundary must consume the generic writer dry-run result without writing memory or widening live authority.

## Options Considered

1. Add a promotion wrapper around the generic memory-writer dry-run result consumer.
2. Let callers use the generic consumer directly.
3. Invoke the memory writer apply path directly after dry-run success.

## Decision

Add a promotion wrapper around the generic memory-writer dry-run result consumer.

## Why This Won

The wrapper preserves promotion provenance and makes the next apply request visible without combining dry-run verification with persistence. It reuses the generic dry-run validator and keeps the promotion chain explainable in Product State, the registry, and the project-goal matrix.

## Rejected Alternatives

Using the generic consumer directly loses promotion-specific envelope and request validation. Invoking the writer directly would cross the persistence boundary and obscure whether evidence was only dry-run verified or written.

## Implementation Plan

- Add a read-only promotion memory-writer dry-run result consumer.
- Wire it into Product State dispatch.
- Register it as `shadow_only` decision evidence.
- Add focused and Product State tests.
- Record the safety rule in `.ai/memory/project.md`.

## Test Plan

- Verify not-ready, ready, forged, and applied-result rejection states.
- Verify registry and goal-matrix non-live status.
- Verify Product State dispatch emits a memory-writer apply request template with `apply=false` and `operator_ack=false`.
- Run `py_compile`, focused pytest, memory lint, and diff checks.

## Risks And Rollback

Risk: a malformed dry-run could be treated as write readiness. Mitigation: delegate to the generic memory-writer dry-run result consumer and require all promotion provenance fields to match. Rollback: remove the new handler dispatch, registry/matrix entries, and module; prior promotion memory-draft behavior remains unchanged.
