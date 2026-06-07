# Brief: WING Supervised Proof Operator Confirmation - 2026-06-07

## Context

The next WING MVP step is still a supervised fader/gain proof after a clean
read-only preflight artifact and accepted manual review decision. The proof
runner already requires `--execute`, bounded negative deltas, rollback, and
emergency-stop proof flags.

## Problem

The physical proof command did not carry a second explicit operator
confirmation that the channel, value type, and delta being executed still match
the operator-approved action. A stale copied command could therefore be easier
to miss during a live hardware session.

## Scope

- Require matching operator confirmation flags for physical `--execute` runs.
- Generate those flags in the next supervised WING proof plan.
- Fail the release gate when proof-plan commands do not carry matching
  confirmation flags.
- Keep all non-fader/gain writes and full autonomous readiness blocked.

## Out Of Scope

- Running WING hardware.
- Widening live writes beyond supervised fader/gain proof.
- Marking the MVP blocker or readiness matrix closed.
