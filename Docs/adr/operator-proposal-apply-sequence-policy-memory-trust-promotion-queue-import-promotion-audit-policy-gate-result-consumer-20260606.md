# ADR: Promotion Audit Policy Gate Result Consumer

## Context

Promotion queue-import execution-audit evidence can now be verified and converted into an audit-policy-gate request template. The next boundary must consume the generic audit-policy-gate result without creating memory drafts or widening live authority.

## Options Considered

1. Add a promotion wrapper around the generic audit-policy-gate result consumer.
2. Let callers use the generic consumer directly.
3. Invoke the memory-draft handler directly after audit-policy-gate success.

## Decision

Add a promotion wrapper around the generic audit-policy-gate result consumer.

## Why This Won

The wrapper preserves promotion provenance and makes the next memory-draft request visible without combining verification with memory-draft execution. It reuses the generic policy evidence validator and keeps the promotion chain explainable in Product State, the registry, and the project-goal matrix.

## Rejected Alternatives

Using the generic consumer directly loses promotion-specific envelope and request validation. Invoking memory draft directly would cross a safety boundary and obscure whether evidence was only verified or acted on.

## Implementation Plan

- Add a read-only promotion audit-policy-gate result consumer.
- Wire it into Product State dispatch.
- Register it as `shadow_only` decision evidence.
- Add focused and Product State tests.
- Record the safety rule in `.ai/memory/project.md`.

## Test Plan

- Verify not-ready, ready, forged, and blocked states.
- Verify registry and goal-matrix non-live status.
- Verify Product State dispatch emits a memory-draft request template and does not call live/manual write paths.
- Run `py_compile`, focused pytest, memory lint, and diff checks.

## Risks And Rollback

Risk: a malformed policy gate could be treated as memory-draft input. Mitigation: delegate to the generic audit-policy-gate result consumer and require all promotion provenance fields to match. Rollback: remove the new handler dispatch, registry/matrix entries, and module; prior promotion execution-audit behavior remains unchanged.
