# ADR: WING MVP Handoff Denial Match Flags

Date: 2026-06-14
Status: Accepted

## Context

The WING read-only handoff packet must prove that the backend-port prestart
gate grants no execution, readiness closure, safe-to-stop authority, or live
authority. The current first command now asks for explicit `*_matches_plan`
fields for those denial values.

## Options considered

1. Leave the packet unchanged and rely on raw false fields.
2. Add explicit match flags as constants.
3. Add explicit match flags derived from actual prestart summary metadata when
   available.

## Decision

Use option 3. The packet now derives denial match flags from the existing
prestart backend-check detail when a no-WING prestart summary is supplied, and
falls back to expected false values for placeholder handoff generation.

## Why this won

Derived verdicts preserve the operator-facing proof boundary without masking a
drifted prestart summary. The change stays local to the handoff packet and does
not contact WING hardware.

## Rejected alternatives

- Raw false fields only: does not satisfy the current first command.
- Constant true flags: would hide corrupted or stale prestart metadata.

## Implementation plan

- Add five handoff packet fields ending in `_matches_plan`.
- Render the new fields in markdown next to the denial evidence.
- Cover both clean and drifted prestart summaries in tests.

## Test plan

- Focused handoff packet tests.
- Focused WING MVP rails/director tests.
- Compile, memory lint, full suite where environment permits, and whitespace
  diff check.

## Risks and rollback

Risk is limited to packet/report metadata. Roll back by removing the five fields
and related tests if downstream consumers require the old schema.
