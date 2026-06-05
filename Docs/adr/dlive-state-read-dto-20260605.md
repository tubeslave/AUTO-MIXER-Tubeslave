# ADR: dlive-state-read-dto-20260605

## Context

The Automixer goal includes Allen & Heath dLive support, but current Product
Layer snapshots are mostly WING-oriented. `DLiveClient` has cached MIDI/TCP
state for names, faders, mutes, and raw MIDI events, while advanced processing
and routing parity are not normalized.

## Options considered

1. Keep dLive state only in `DLiveClient.get_state`.
2. Add a cached-only Product Layer dLive DTO.
3. Add dLive transport queries or live apply support.

## Decision

Use option 2. Add `build_dlive_state_snapshot` and keep dLive apply blocked.

## Why this won

The cockpit gains a stable, display-only dLive read contract without sending
MIDI, requesting names, or implying WING parity.

## Rejected alternatives

Option 1 leaves dLive outside the Product Layer read coverage map. Option 3 is
unsafe without normalized readback, physical fixtures, stale-cache timestamps,
and explicit operator approval.

## Implementation plan

- Add cached dLive channel rows for name, fader, mute, and raw note velocity.
- Add raw MIDI cache counters and explicit WING-parity blockers.
- Register `get_dlive_state_snapshot`.
- Update read surface inventory and focused tests.

## Test plan

- `PYTHONPATH=backend python -m pytest tests/test_operator_product_state.py tests/test_read_surface_inventory.py -q`
- Broader server/registry/readiness tests.
- Full suite with `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
- `git diff --check`

## Risks and rollback

Risk: a narrow dLive DTO could be mistaken for full parity. Mitigation: the
payload includes `wing_parity_status=partial`, safety flags, and known gaps.
Rollback is removing the builder, handler, tests, and inventory updates.
