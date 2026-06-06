# ADR: ProductState Promotion-Memory Policy Consumer Request Handoff

## Context

The promotion queue-import memory chain can verify promotion memory-trust
acceptance-ledger evidence and expose a promotion-plan request template. The
existing downstream promotion consumers already cover contract acceptance,
execution planning, dry-run request templating, executor preview, and
apply-audit blocking until an applied executor result exists. The existing
executor can also apply to a caller-provided sequence-policy trust store only
with explicit `apply=true` and `operator_ack=true`.

## Options considered

1. Use only existing generic payload field names in ProductState.
2. Add memory-prefixed ProductState aliases and reuse existing downstream
   promotion consumers.
3. Add new promotion-memory-specific downstream verifier modules.

## Decision

Add memory-prefixed ProductState aliases for the downstream promotion consumer
handoffs and extend ProductState regression coverage through the executor
dry-run, preview executor-result consumer boundary, and supervised apply-audit
plus rollback-verification and policy-consumer request boundary.

## Why this won

It preserves explicit promotion-memory provenance at the handler boundary while
reusing the existing verifier builders. The change does not create a new method,
does not add dependencies, and does not widen memory, queue, proposal,
idempotency, execution, or live mixer authority. Any policy trust-store mutation
remains limited to an explicit caller-provided target store under operator
acknowledgement.

## Rejected alternatives

Generic-only payload names were rejected because they make the promotion-memory
handoff less readable in ProductState calls. New downstream verifier modules
were rejected because the existing consumers already enforce the correct
safety contract.

## Implementation plan

- Add aliases for promotion-memory contract, acceptance-ledger, execution-plan,
  executor-dry-run, executor-result, apply-audit, and rollback-verification
  prior-consumer payloads.
- Extend the ProductState route test from promotion-memory contract consumer to
  promotion acceptance, execution plan, dry-run request templating, executor
  dry-run, preview executor-result blocking, supervised target-store apply,
  apply audit, rollback verification, and sequence-policy-gate request
  templating through the policy-consumer request.
- Update the promotion queue-import memory boundary report and durable memory.

## Test plan

Run py_compile, focused promotion/ProductState pytest, memory lint, diff check,
and full pytest if the focused checks pass.

## Risks and rollback

The main risk is alias drift. Rollback is to remove the aliases, test
extension, and documentation lines; no persisted data migration or live mixer
path is introduced.
