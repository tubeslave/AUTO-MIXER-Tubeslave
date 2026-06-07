# ADR: WING Supervised Proof Physical Readback Only - 2026-06-07

## Context

The supervised WING Rack MVP depends on proving a narrow physical fader/gain
write loop: pre-read, apply, physical readback, rollback, rollback readback, and
emergency-stop block proof. Cache evidence is useful for planning, but it is not
physical proof.

## Options considered

1. Continue accepting cache-confirmed readbacks for physical proof.
2. Require physical readback for `--execute` while keeping cache available for
   dry-run.
3. Disable dry-run cache use entirely.

## Decision

Require `verified_from_physical_readback` for physical `--execute` pre-read,
apply, and rollback. Keep cache available only for non-mutating dry-run
planning.

## Why this won

The real blocker is hardware proof. A physical proof artifact should never pass
because local state cache says the value changed. Requiring physical readback
keeps readiness evidence aligned with the WING MVP contract without widening
live authority.

## Rejected alternatives

Cache-confirmed physical proof was too weak. Disabling dry-run cache use was
unnecessary because dry-run artifacts remain blocked from promotion.

## Implementation plan

1. Add a physical-only readback status set to the runner.
2. Force physical pre-read when `--execute` is present.
3. Reject apply and rollback results whose readback status is not physical.
4. Add focused tests for physical success and cache rejection.
5. Add a no-WING static policy check for the runner source.

## Test plan

- `python3 -m py_compile tools/wing_operator_proposal_validation_runner.py tools/no_wing_release_check.py`
- focused runner/no-WING pytest
- quick no-WING release check
- full pytest before landing

## Risks and rollback

The change intentionally rejects older fake or real proof artifacts that report
cache readback for physical `--execute`. Rollback would reallow weak evidence and
should not be used unless the backend cannot provide physical readback status at
all.
