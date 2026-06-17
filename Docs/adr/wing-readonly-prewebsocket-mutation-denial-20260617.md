# ADR: WING Read-Only Pre-WebSocket Mutation Denial

## Context

The first WING MVP blocker remains real read-only connection proof. The
no-WING and artifact-manifest contracts require `mutates_console_state: false`,
but not every runner-produced offline or pre-WebSocket surface exposed that
same top-level denial.

## Options considered

1. Rely on `read_only: true` and `live_write: false`.
2. Add explicit `mutates_console_state: false` to runner pre-WebSocket surfaces.
3. Add the field only to successful real hardware artifacts.

## Decision

Use option 2. Offline plans, plan-only blocked outputs, and real-run
pre-WebSocket summaries now expose `mutates_console_state: false`.

## Why this won

The operator and downstream gates should not infer no-mutation from nearby
fields. A direct field makes blocked, validation-only, and expected-plan
artifacts auditable against the same no-WING contract before any hardware
contact.

## Rejected alternatives

- Option 1 leaves a safety property implicit.
- Option 3 delays the denial marker until after the exact phase where it is
  most useful: deciding whether it is safe to proceed toward WING contact.

## Implementation plan

- Add `mutates_console_state: false` to the offline preflight plan.
- Add the same field to plan-only blocked validation and prestart outputs.
- Add the same field to the base real-run pre-WebSocket summary.
- Assert representative plan-only, expected-plan, validation-only, and blocked
  validation surfaces.

## Test plan

- Targeted runner tests for the changed surfaces.
- no-WING quick release check.
- Focused WING preflight, no-WING, handoff, and Project Director tests.
- `git diff --check`.
- Full pytest.

## Risks and rollback

The risk is low because this only makes an existing read-only/no-live-write
property explicit. Rollback is to remove the field and assertions.
