# ADR: Operator Proposal Apply Sequence Policy Memory Trust Promotion Queue Importer

## Context

Sequence-policy queue-ordering plans can now produce read-only future queue
candidates. The system needs an explicit importer boundary before those
candidates become pending operator proposals.

## Options considered

1. Keep queue-ordering plans as evidence only.
2. Add an operator-gated queue importer.
3. Execute sequence-policy queue records directly.

## Decision

Add an operator-gated sequence-policy queue importer.

## Why this won

It moves sequence-memory learning closer to real operator workflow by allowing
acknowledged queue import, while Product Layer preview remains dry-run and live
console authority remains unchanged.

## Rejected alternatives

Evidence-only plans were safe but did not move the workflow toward usable
operator review. Direct sequence execution was rejected because queue import is
not execution authority and still needs separate sequence gates.

## Implementation plan

- Add the sequence-policy queue importer module.
- Validate queue-ordering plan evidence before producing import specs.
- Keep Product Layer preview dry-run only.
- Add a product-state handler that can import only with `apply=true`,
  `operator_ack=true`, and the server's explicit operator proposal queue.
- Expose sequence-policy import visibility in the decision queue.
- Register the method as `shadow_only` and add it to the capability matrix.

## Test plan

- Unit-test importer safety and ack gates.
- Extend Product Layer preview, handler, decision queue, registry, and
  capability-matrix tests.
- Run focused pytest and memory lint.

## Risks and rollback

The main risk is confusing queue import with live apply or sequence execution.
Rollback is to remove the importer from preview, registry, handlers, and
capability matrix while leaving the queue-ordering plan as the last sequence
policy evidence layer.
