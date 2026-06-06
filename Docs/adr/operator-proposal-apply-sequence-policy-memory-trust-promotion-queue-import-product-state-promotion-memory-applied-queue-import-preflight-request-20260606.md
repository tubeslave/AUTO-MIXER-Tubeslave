# ADR: ProductState Promotion-Memory Applied Queue Import Preflight Request

## Context

The ProductState promotion-memory route can produce dry-run queue-importer
specs and prove the queue-importer result consumer blocks preflight planning
until applied queue-import evidence exists. The existing queue importer may
mutate the operator proposal queue only with `apply=true` and
`operator_ack=true`; it does not touch live mixer transport.

## Options considered

1. Stop at the dry-run blocker.
2. Apply the queue importer under explicit acknowledgement and verify the
   preflight-plan request boundary.
3. Run the preflight plan in the same slice.

## Decision

Extend ProductState coverage through an explicitly acknowledged queue-importer
apply and verify that the queue-importer result consumer emits only a read-only
preflight-plan request template.

## Why this won

It proves the necessary source evidence for the next autonomous stage while
keeping the consumer boundary read-only. The applied queue import is limited to
the operator proposal queue and remains non-live; preflight planning stays a
separate downstream action.

## Rejected alternatives

Stopping at the dry-run blocker leaves the applied source-evidence path
unproven. Running the preflight plan in the same slice was rejected because the
request boundary deserves its own validation before the next step.

## Implementation plan

- Extend the ProductState promotion-memory route test to run the queue importer
  with `apply=true`, `operator_ack=true`, and a fader-shaped proposal payload.
- Verify that the importer applies only to the operator proposal queue and
  keeps live transport false.
- Verify the applied importer result into a preflight-plan request template
  while the consumer keeps mutation and authority flags false.
- Update durable memory and the promotion queue-import memory boundary report.

## Test plan

Run py_compile, focused promotion/ProductState pytest, memory lint, diff check,
and full pytest if the focused checks pass.

## Risks and rollback

The main risk is blurring source queue mutation with consumer-side authority.
Rollback is to remove the ProductState test extension and documentation lines;
no live mixer path is introduced.
