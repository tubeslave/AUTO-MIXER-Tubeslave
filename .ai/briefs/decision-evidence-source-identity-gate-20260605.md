# Brief: decision-evidence-source-identity-gate-20260605

Task id: decision-evidence-source-identity-gate-20260605

## Context

`decision_evidence_packet_proposal` explains proposal and shadow decisions with
audio, psychoacoustic, statistical, history, reference, inter-channel,
arrangement, genre, and previous-outcome evidence. Those packets can influence
operator review and later memory learning, so they should not treat source-role
context as trusted when source identity replay calibration is held, blocked, or
stale.

## Scope

- Consume the shared read-only `source_identity_downstream_gate`.
- Keep legacy callers compatible when source identity calibration is missing.
- Demote packets to `analysis_only` when source identity trust is held, blocked,
  unknown, or based on stale replay context.
- Keep packets non-queueable and non-live-applicable.

## Non-goals

- No live OSC, WING, dLive, or Mixing Station writes.
- No source identity classifier mutation.
- No memory-learning writes.
- No queue import or apply authority for decision evidence packets.

## Evidence to preserve

- `source_identity_downstream_trust` in recommendation evidence.
- `can_apply_live=False`.
- `can_import_to_queue=False`.
- No `requested_change` or `requested_changes` payload fields.

## Validation

- Focused decision evidence packet tests.
- Source identity downstream gate tests.
- Registry/contract and capability matrix tests.
- Full backend pytest.
- `git diff --check`.
- Runtime smoke for held source identity.
