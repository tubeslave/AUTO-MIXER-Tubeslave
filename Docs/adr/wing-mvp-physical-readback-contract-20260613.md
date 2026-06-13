# ADR: WING MVP Physical Readback Contract

## Context

The current MVP remains blocked on real `WING read-only connection proof`. A
preflight artifact can only become a readiness-update candidate when every
declared readback is verified as physical. Runner and evidence intake previously
owned local physical-readback status/source sets.

## Options considered

1. Keep local physical-readback status/source sets.
2. Move the physical-readback identity into the shared preflight contract and
   validate runner/evidence agreement in no-WING.
3. Only test the existing local sets.

## Decision

Use option 2.

## Why this won

The physical-readback identity is the proof boundary for the first WING blocker.
If runner and evidence intake disagree, a captured artifact could be accepted or
rejected under different semantics than the tool used to capture it.

## Rejected alternatives

- Option 1 leaves real-proof semantics free to drift.
- Option 3 improves tests but does not expose drift in release/prestart
  artifacts.

## Implementation plan

- Add shared physical readback statuses and sources to the preflight contract.
- Use those constants in the runner and evidence intake.
- Add no-WING detail, JSON summary, and Markdown summary fields for expected,
  runner, and evidence physical-readback identities.
- Fail closed when runner or evidence identities diverge.
- Add focused happy-path and stale-contract tests.

## Test plan

- Compile touched modules.
- Focused runner, evidence, and no-WING tests.
- no-WING quick on an unused backend port.
- Project Director and direction guard checks.
- `git diff --check`.
- `python3 tools/memory/memory_lint.py`.
- Full backend pytest.

## Risks and rollback

The main risk is stricter no-WING validation surfacing stale local proof
semantics. That is intended. Rollback would restore duplicated sets and remove
the prestart drift check, but would reopen inconsistent readback-proof handling.
