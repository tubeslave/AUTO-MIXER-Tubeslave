# Brief: read-surface-fixture-closure-plan-hardening-20260606

## Context

The read-surface fixture closure plan can surface accepted operator review
records as manual fixture-baseline update candidates. That plan is still
read-only, but it sits close to the future boundary where physical fixture gaps
may eventually be closed. It must therefore fail closed when its acceptance
ledger input is malformed, forged, or internally inconsistent.

## Objective

Harden the closure plan so only structurally valid, UTC-reviewed, accepted
records from a valid acceptance-ledger envelope can become manual closure
candidates.

## Safety Constraints

- Do not update fixture files or baselines.
- Do not close physical fixture gaps.
- Do not query WING, dLive, OSC, MIDI, TCP, Dante, or any live console.
- Do not execute tests from the closure plan.
- Preserve `fixture_update_allowed_by_plan=false`,
  `fixture_closure_allowed_by_plan=false`, `live_authority_granted=false`, and
  `full_autonomous_ready=false`.

## Expected Evidence

- The closure plan rejects malformed ledger envelopes and unsafe safety claims.
- The closure plan rejects accepted records with missing fields, non-UTC review
  timestamps, mismatched decisions, missing candidate links, or unsafe claims.
- The closure plan rejects summary count and surface-id drift before creating
  closure candidates.
- Focused closure-plan, no-WING, registry, and capability tests pass.
