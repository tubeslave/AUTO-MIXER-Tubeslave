# Brief: read-surface-fixture-closure-plan-20260605

## Context

The artifact review queue can aggregate physical readback candidates, and the
fixture acceptance ledger can validate operator accept/reject records. The next
safe step is to expose accepted records as manual fixture-closure candidates
without actually updating fixture baselines or widening live authority.

## Objective

Add a read-only fixture closure plan that turns accepted operator review records
into manual closure candidates and publishes its status in no-WING release
evidence.

## Safety Constraints

- Do not update fixture files or test baselines.
- Do not close physical fixture gaps from the plan itself.
- Do not query WING, dLive, OSC, MIDI, TCP, Dante, or any live console.
- Do not execute tests or mutate console state.
- Do not grant live authority, live widening, or full autonomous readiness.

## Expected Evidence

- `backend/read_surface_physical_readback_fixture_closure_plan.py` builds a
  transport-free, read-only manual closure plan.
- Registry and capability matrix expose the plan as `shadow_only` analysis
  evidence for console read-parameter work.
- no-WING release checks validate and summarize the plan after the acceptance
  ledger gate.
- Focused tests cover not-ready, accepted, rejected, invalid, unreviewed,
  registry, capability, release ordering, and summary normalization paths.
