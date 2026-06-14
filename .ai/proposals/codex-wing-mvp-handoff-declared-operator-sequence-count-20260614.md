# Codex Proposal: WING MVP Handoff Declared Operator Sequence Count

## Short thesis

Declare the operator sequence count in the handoff packet and validate it
against both the shared contract and actual sequence length.

## What I understood

The active blocker remains WING read-only connection proof. The no-WING summary
already computes the operator sequence count, but the packet itself should also
declare the count so the operator-facing handoff metadata cannot drift.

## Proposed solution

Have `tools/build_wing_readonly_preflight_handoff_packet.py` emit
`operator_sequence_count`, `expected_operator_sequence_count`, and
`operator_sequence_count_matches_contract`. Have
`tools/no_wing_release_check.py` expose declared count fields and fail closed if
they do not match the actual sequence, the shared contract, or their own
contract flag.

## Likely files to touch

- `tools/build_wing_readonly_preflight_handoff_packet.py`
- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- WING MVP docs, autoloop docs, rails tests, ADR, and report files

## Alternatives considered

Keeping only computed summary fields is safe but less auditable from the packet.
Adding count fields without validation would create another possible source of
misleading handoff metadata.

## Risks

The main risk is wording drift between computed and declared count fields. Rails
tests should keep Project Director, autoloop docs, and MVP docs aligned.

## Test plan

Run focused no-WING validator tests, Project Director rails tests, WING MVP
rails tests, no-WING quick summary generation, direction guard, memory lint,
`git diff --check`, and the full standard pytest command.

## Where the other agent may disagree

Kimi may say computed count is enough. The counterpoint is that this handoff is
an operator artifact before a hardware session; declared packet metadata should
be validated rather than trusted implicitly.
