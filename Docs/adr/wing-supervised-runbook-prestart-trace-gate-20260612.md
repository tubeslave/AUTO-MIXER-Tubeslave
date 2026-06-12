# WING Supervised Runbook Prestart Trace Gate

## Context

The supervised proof plan and no-WING release gate now require accepted
read-only preflight source ids to have a valid accepted-candidate prestart trace
before physical WING proof steps can proceed. The supervised fader/gain runbook
and read-only handoff packet are the operator-facing bridge into that proof.

## Options considered

1. Keep runbook wording focused on clean preflight plus accepted decision.
2. Add the accepted-candidate prestart trace gate to the existing runbook and
   handoff sequence.
3. Create a new supervised proof runbook variant.

## Decision

Use option 2. The existing supervised fader/gain proof runbook now lists valid
accepted-candidate prestart trace as a precondition and fail-closed condition,
and the handoff packet only points to `WING_SUPERVISED_FADER_GAIN_PROOF` after
both `accepted_decision_ready` and the valid prestart trace.

## Why this won

It aligns operator instructions with the proof-plan gate without widening live
write authority or changing packet schema.

## Rejected alternatives

- Generic accepted-decision wording was rejected because it can hide the new
  candidate prestart prerequisite.
- A new runbook was rejected because the current runbook is already the
  canonical supervised fader/gain proof path.

## Implementation plan

- Update `Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md`.
- Update `tools/build_wing_readonly_preflight_handoff_packet.py`.
- Add regression assertions.
- Add documentation report and index entry.

## Test plan

- Focused runbook/handoff tests.
- Affected no-WING and MVP rails tests.
- Full test suite.
- `git diff --check`.

## Risks and rollback

Risk is limited to operator wording. Rollback is safe by restoring the previous
runbook text, operator-sequence line, and focused assertions. No runtime console
behavior changes.
