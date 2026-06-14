# ADR: WING MVP handoff operator sequence cardinality

## Context

The no-WING handoff summary requires `operator_sequence_count: 10` and a final
supervised prestart gate at index `9`. The validator already checked for a
present final gate, but did not explicitly reject a sequence with an extra step
inserted before that final gate when the structured gate was shifted to match.

## Options considered

1. Keep sequence cardinality informational.
2. Permit extra operator notes in the sequence.
3. Enforce exact sequence count and exact supervised gate index.

## Decision

Use option 3.

## Why this won

The operator sequence is the bridge from no-WING checks into the real WING
read-only run. It should be stable and auditable. Extra steps can live in docs
or reports, but the packet sequence should not drift before supervised proof
planning.

## Rejected alternatives

Option 1 left the documented count and index as weak report fields. Option 2
made the hardware handoff order harder to compare across runs.

## Implementation plan

- Add a shared operator sequence count constant.
- Fail closed when `operator_sequence_count` differs from the contract.
- Fail closed when the supervised prestart gate index differs from the
  contract.
- Render expected values and match flags in summary output.

## Test plan

- Add a focused mismatch test for a coherently shifted extra-step handoff.
- Run focused no-WING tests.
- Run WING rail and Project Director tests after docs updates.
- Run no-WING quick and full pytest before commit.

## Risks and rollback

The change is offline and read-only. Rollback is reverting the new count/index
violations and summary fields if a legitimate generated handoff packet is
unexpectedly rejected.
