# ADR: WING MVP Handoff Operator Sequence SHA256

## Context

The supervised WING MVP still waits on real WING read-only connection proof.
The no-WING handoff gate already checks the operator sequence count and several
critical step indexes, but it did not bind the complete ordered checklist.

## Options considered

1. Keep count and marker validation only.
2. Add more individual marker checks for each operator sequence step.
3. Add a SHA256 digest over the complete `operator_sequence` list and validate
   the declared digest against the actual packet sequence.

## Decision

Use option 3.

## Why this won

The operator packet is a safety-critical handoff before a hardware session. A
canonical sequence digest proves the reviewed checklist has not drifted as a
whole, while preserving existing explicit checks for high-risk step positions.
It keeps the packet offline and read-only.

## Rejected alternatives

Count and marker validation can miss edits to intermediate steps. Adding many
step-specific marker checks is more fragile and still does not bind the exact
ordered sequence.

## Implementation plan

Emit `operator_sequence_sha256` from the handoff builder. Recompute the digest
inside no-WING handoff validation, expose declared and actual digests plus the
match flag, and fail closed on mismatch. Update WING MVP docs and rails tests.

## Test plan

Run focused no-WING and handoff packet tests, WING MVP rails tests, no-WING
quick, Project Director, direction guard, memory lint, `git diff --check`, and
the standard full test command.

## Risks and rollback

Risk is limited to stricter offline handoff validation. Rollback is to remove
the sequence digest field and mismatch violation if the handoff schema changes.
