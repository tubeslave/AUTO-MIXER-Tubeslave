# ADR: WING Read-Only Preflight Prestart Detail Fields Contract

## Context

The no-WING prestart gate emits explicit busy-port confirmation fields. The shared WING preflight contract still listed only `listener_processes` and `operator_next_step` as expected failure detail fields, which made expected plans and handoff packets less complete than the actual artifact.

## Options considered

1. Keep the shared contract unchanged.
2. Expand the shared contract to include the stable busy-port confirmation fields.
3. Require command-line diagnostics in the shared contract.

## Decision

Use option 2. The shared contract now includes listener processes, confirmation-required flags, the no-auto-stop flag, fail-closed reason, and operator next step.

## Why this won

These fields are stable safety evidence for the prestart gate. They should be visible in runner plans, handoff packets, no-WING summaries, and artifact validators.

## Rejected alternatives

Option 1 preserves drift. Option 3 would over-require optional OS diagnostics that can fail due to permissions.

## Implementation plan

- Update `backend/wing_readonly_preflight_contract.py`.
- Move test fixtures and assertions to the shared constant.
- Validate the adjacent WING/no-WING stack.

## Test plan

- Focused WING/no-WING tests.
- Full backend suite before commit.

## Risks and rollback

Risk: older generated plans or packets need regeneration. Rollback: remove the additional fields from the shared contract and restore old expectations. No live write behavior is changed.
