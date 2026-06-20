# ADR: Master Bus PA Feed And Reference Safety Gates

## Context

The project goal requires master-bus optimization, but master/output moves are
high risk in live sound. A master-bus analyzer must not treat measurement,
capture or reference feeds as PA output, and it must not suggest master/output
changes without fresh read-only evidence for both the current main-bus state and
the PA feed affected by the change.

M001 still waits for real WING read-only evidence. This ADR does not grant live
authority or replace hardware/operator validation.

## Options Considered

1. Keep existing master-bus proposals based only on loudness, true peak and
   tonal evidence.
2. Add PA-feed and reference-feed safety gates before master/output proposals.
3. Disable master-bus proposals entirely until WING hardware proof exists.

## Decision

Use option 2. Master-bus proposal candidates require fresh read-only/no-authority
main-bus context, fresh read-only/no-authority PA-feed readback, explicit
PA-feed safety confirmation and reference-feed exclusion evidence.

## Why This Won

It advances the master-bus optimization part of the AI Mixing Engineer goal while
preserving the current live boundary. It also handles a real failure mode:
reference/capture feeds can look like master feeds in analysis, but they must
not drive PA/output proposal decisions.

## Rejected Alternatives

- Loudness-only master optimization was rejected because it can hide PA-feed,
  reference-feed and live-output safety risks.
- Disabling the layer entirely was rejected because offline proposal gating and
  tests can improve the future assisted workflow without touching hardware.

## Implementation Plan

1. Require fresh no-authority main-bus context.
2. Require PA-feed readback and explicit PA-feed safety evidence.
3. Block live-authority PA-feed context.
4. Exclude reference, measurement, capture and analysis feeds.
5. Record proposal-contract evidence and M086 rail coverage.

## Test Plan

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_master_bus_optimization_proposal.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m086_master_bus_pa_feed_reference_safety_gates_stay_readonly -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Risks and Rollback

No live transport is introduced. Rollback is to revert the M086 commit, which
restores the previous master-bus proposal contract and M086 state/report rows.
