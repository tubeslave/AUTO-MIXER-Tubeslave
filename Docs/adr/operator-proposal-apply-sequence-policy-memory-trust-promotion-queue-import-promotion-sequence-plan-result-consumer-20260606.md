# ADR: Promotion Sequence-Plan Result Consumer

## 1. Context

The sequence-policy memory trust promotion chain can now verify an approved
batch-preflight result and emit a generic sequence-plan request template. Before
any dry-run request is exposed, the generic sequence-plan result must be checked
against the promotion batch-preflight provenance.

## 2. Options considered

- Use the generic sequence-plan result consumer directly.
- Add a promotion-specific sequence-plan result consumer.
- Skip directly to dry-run execution.

## 3. Decision

Add a promotion-specific read-only sequence-plan result consumer.

## 4. Why this won

It preserves provenance between promotion batch-preflight evidence and generic
sequence planning, while still delegating the detailed sequence-plan checks to
the existing generic consumer.

## 5. Rejected alternatives

Using the generic consumer directly was rejected because it cannot prove the
promotion chain source. Skipping to dry-run execution was rejected because this
phase may only expose request templates and must not invoke handlers.

## 6. Implementation plan

- Add the transport-free consumer module.
- Register a Product Layer verification handler.
- Register the method as `shadow_only` in the processing registry and project
  goal matrix.
- Add focused and handler-chain tests.
- Record the safety lesson in project memory.

## 7. Test plan

- Focused consumer pytest.
- Product-state handler-chain pytest.
- Registry and project-goal matrix pytest.
- Full backend pytest.
- Memory lint.

## 8. Risks and rollback

Risk: future code may treat the dry-run request template as executed evidence.
Rollback: remove the consumer, handler registration, registry/matrix rows, and
tests. No runtime migration or live mixer state rollback is required because the
change is read-only and transport-free.
