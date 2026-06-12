# WING Handoff Generator Prestart Regression

## Context

The generated WING read-only handoff packet is the operator-facing artifact
that points toward supervised proof. The release validator now checks for the
prestart gate, but generator tests should also pin that contract.

## Options considered

1. Rely only on no-WING release validation.
2. Add direct generator test assertions for the prestart-gated supervised proof
   transition.

## Decision

Use option 2. The handoff packet generator test now checks both the JSON
operator sequence and rendered Markdown for the accepted-decision plus valid
accepted-candidate prestart trace boundary before `WING_SUPERVISED_FADER_GAIN_PROOF`.

## Why this won

It catches generator regressions earlier while preserving the release validator
as the artifact-level fail-closed gate.

## Rejected alternatives

Release-only coverage was rejected because it leaves the generator's unit
contract too indirect.

## Implementation plan

- Extend `tests/test_wing_readonly_preflight_handoff_packet.py`.
- Add trace report.

## Test plan

- Focused generator tests.
- Affected no-WING handoff tests.
- Full pytest suite.
- `git diff --check`.

## Risks and rollback

Risk is test-only. Rollback is safe by removing the new assertions, but that
would weaken generator-level coverage.
