# ADR: WING supervised gate expected-artifact marker

## Context

The WING read-only decision rail requires an accepted candidate source id to
carry expected-artifact binding and a valid no-WING prestart trace. The final
handoff operator-sequence step is the visible transition into supervised
fader/gain proof planning, so it must name both parts of the accepted-candidate
contract.

## Options considered

1. Keep the final gate marker as `valid accepted-candidate prestart trace`.
2. Add a new human-readable marker for expected-artifact binding.
3. Move the check later into the supervised proof runner.

## Decision

Add `expected-artifact-bound accepted source id` to the shared supervised
prestart gate required markers and final operator-sequence step.

## Why this won

The expected-artifact binding is now part of what makes the accepted-candidate
trace valid. Naming it in the final handoff gate prevents stale handoff packets
and operator notes from treating an old three-marker gate as sufficient.

## Rejected alternatives

Leaving the binding implicit was rejected because the handoff packet is
operator-facing. Moving the check later was rejected because the supervised
proof plan should not be reached from a stale read-only gate.

## Implementation plan

- Update the shared gate step and required marker tuple.
- Let handoff generation and no-WING validation consume the shared contract.
- Add regression coverage for the old marker set.
- Update Project Director and WING MVP rail text.

## Test plan

- Focused handoff/no-WING/Project Director/WING rail tests.
- no-WING quick release check.
- Full pytest before commit.

## Risks and rollback

The change invalidates older handoff packets. Rollback is to remove the new
marker from the shared contract and regenerate the progress docs, but that
would make the final supervised transition less explicit.
