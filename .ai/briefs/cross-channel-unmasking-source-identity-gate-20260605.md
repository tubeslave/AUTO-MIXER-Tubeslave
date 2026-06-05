# Brief: cross-channel-unmasking-source-identity-gate-20260605

Task id: cross-channel-unmasking-source-identity-gate-20260605

## Context

`cross_adaptive_eq` emits linked mirror-EQ cut/boost proposals for
cross-channel unmasking. Those decisions depend on source priority and masking
pairs. If source identity or role evidence is held for operator review, blocked,
unknown, or based on stale replay calibration, the linked EQ proposal can cut or
boost the wrong source relationship.

## Scope

- Add replay-calibrated source identity trust to the proposal-only unmasking
  wrapper.
- Preserve `calculate_corrections` compatibility.
- Keep missing source identity calibration legacy-compatible.
- Block proposal output when provided source identity trust is held, blocked,
  unknown, or stale.
- Keep all output non-queueable and non-live-applicable.

## Non-goals

- No live EQ, WING, dLive, Mixing Station, or OSC writes.
- No source identity classifier mutation.
- No memory-learning writes.
- No change to legacy raw mirror-EQ calculation.

## Evidence to preserve

- `source_identity_downstream_trust` in result metadata.
- `source_identity_downstream_trust` in action safety evidence for masked and
  masker channels.
- `live_write_allowed=False`.
- `can_import_to_queue=False`.
- Gate evidence has no live, queue, model, or memory mutation.

## Validation

- Focused cross-adaptive EQ tests.
- Source identity downstream gate tests.
- Registry/contract and capability matrix tests.
- Full backend pytest.
- `git diff --check`.
- Runtime smoke for held source identity.
