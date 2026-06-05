# Brief: read-surface-fixture-acceptance-ledger-20260605

## Context

The read-surface physical readback artifact review queue can aggregate valid
operator-captured candidate artifacts, invalid artifacts, unsafe authority
claims, and missing surfaces. The next safe step is a read-only ledger for
manual operator accept/reject records. Acceptance records must not be confused
with fixture closure, live authority, or autonomous readiness.

## Objective

Add a transport-free fixture acceptance ledger that validates operator
accept/reject records against queue candidates and publishes its status in
no-WING release evidence.

## Safety Constraints

- Do not close physical fixture gaps from acceptance records.
- Do not query WING, dLive, OSC, MIDI, TCP, Dante, or any live console.
- Do not execute tests, capture artifacts, or mutate console state.
- Do not grant live authority, live widening, or full autonomous readiness.
- Treat unsafe authority claims in acceptance records as invalid evidence.

## Expected Evidence

- A backend ledger module validates surface IDs, artifact source IDs, decisions,
  reviewer metadata, contract IDs, and unsafe authority flags.
- The processing registry and capability matrix expose the ledger as
  `shadow_only` read-only evidence.
- `run_release_checks()` includes the ledger gate after the artifact review
  queue and before backend pytest.
- Focused tests cover missing candidates, accepted records, rejected records,
  unsafe records, registry exposure, release ordering, and summary rendering.
