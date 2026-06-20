# ADR: Performance Adaptation Read-Only Context Gates

## Context

The project goal requires the system to adapt the mix when musicians change
level during a concert. That behavior is safety-critical because a live
adaptation decision can affect intelligibility, feedback risk and PA headroom.

M001 still waits for real WING read-only evidence. This ADR does not grant live
authority or replace hardware/operator validation.

## Options Considered

1. Keep performance-adaptation proposals usable when current context is missing.
2. Require fresh read-only/no-authority performance context before proposal
   emission.
3. Disable performance-adaptation proposals entirely until WING hardware proof
   exists.

## Decision

Use option 2. Performance-adaptation proposal candidates require fresh
read-only/no-authority current performance context before persistent
musician-level drift can become an operator-facing proposal.

## Why This Won

It advances the concert-time adaptation part of the AI Mixing Engineer goal
while preserving the current live boundary. It also handles a real failure mode:
stale or live-authority context can make the system trust the wrong level trend
or accidentally treat a mutable console path as evidence.

## Rejected Alternatives

- Allowing missing context was rejected because live adaptation decisions need
  current readback, latency, trend, rollback and safety evidence.
- Disabling the layer entirely was rejected because offline proposal gating and
  tests can improve the future assisted workflow without touching hardware.

## Implementation Plan

1. Require fresh current performance context for proposal emission.
2. Require that context to be read-only and no-authority.
3. Block live-authority context even when marked fresh.
4. Expose no-authority evidence in the proposal-only contract.
5. Record M087 rail coverage without changing WING readiness.

## Test Plan

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_performance_adaptation_proposal.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m087_performance_adaptation_readonly_context_gates_stay_readonly -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Risks and Rollback

No live transport is introduced. Rollback is to revert the M087 commit, which
restores the previous performance-adaptation proposal contract and M087
state/report rows.
