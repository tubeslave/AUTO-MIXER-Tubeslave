# ADR: WING MVP Handoff Supervised Summary Field Contract

## Context

The WING handoff summary already exposes the canonical supervised proof-plan
summary field list and whether the packet-declared list matches that contract.
The packet itself only exposed `required_supervised_proof_plan_summary_fields`,
which made the operator-facing packet less explicit than the no-WING summary.

## Options considered

1. Keep the expected list and match verdict as no-WING-summary-only fields.
2. Add declared packet fields without validating them.
3. Add declared packet fields and make no-WING validation compare them with the
   canonical contract.

## Decision

Use option 3.

## Why this won

The handoff packet should be inspectable before the no-WING summary is read,
but no-WING must remain authoritative. Declared fields plus recomputation gives
both inspectability and fail-closed stale packet detection.

## Rejected alternatives

Summary-only fields were weaker for operator handoff. Declared-only fields
could be stale or misleading if the contract changed.

## Implementation plan

Add `expected_required_supervised_proof_plan_summary_fields` and
`required_supervised_proof_plan_summary_fields_match_contract` to the packet,
render them in Markdown, compare them inside no-WING packet validation, and
cover positive and stale-declaration cases with tests.

## Test plan

Run focused handoff packet and no-WING validator tests. Run packet generation,
Direction Guard, no-WING quick fail-closed, memory lint, `git diff --check`,
and the standard full test command.

## Risks and rollback

Risk is limited to offline handoff metadata and validation. Rollback is to
remove the declared fields and validator comparisons if the packet schema moves
to a structured contract object. This does not query or mutate a console,
update readiness, grant live authority, or close the blocker.
