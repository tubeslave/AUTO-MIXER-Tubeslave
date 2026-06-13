# ADR: WING MVP Read-Only Artifact Contract Constants

## Context

The WING MVP first blocker depends on real read-only preflight evidence. The
same artifact identity appears in the runner, handoff packet, release check,
evidence intake, and supervised proof plan.

## Options considered

1. Keep per-file literals.
2. Move the WING read-only artifact identity into the existing preflight
   contract module.
3. Create a new artifact schema module.

## Decision

Use option 2.

## Why this won

`backend/wing_readonly_preflight_contract.py` already owns shared no-WING
preflight safety constants. Keeping schema and artifact names there makes the
real proof boundary explicit without adding a new module or changing runtime
behavior.

## Rejected alternatives

- Option 1 leaves runner, packet, release, and proof-plan code free to drift.
- Option 3 adds an extra ownership boundary for a small set of constants.

## Implementation plan

- Add shared constants for `wing_readonly_preflight.v1`, `summary.json`,
  `events.jsonl`, the proof summary artifact string, and the next manual
  artifact requirement.
- Use them in the preflight runner, handoff packet builder, evidence intake,
  no-WING release check, and supervised proof plan.
- Update focused tests to assert the shared values.

## Test plan

- Focused WING read-only preflight, handoff packet, no-WING release, supervised
  proof plan, Project Director, and guard tests.
- Project Director JSON check.
- Direction guard JSON check.
- `git diff --check`.
- `python3 tools/memory/memory_lint.py`.
- Full backend pytest.

## Risks and rollback

Rollback restores local literals. It does not change live WING behavior or grant
authority.
