# ADR: Routing Management Read-Only Context Gates

## Context

The project goal requires full digital-console control, including routing.
Routing is high risk because a patch or bus route can rewrite broad mixer
state, silence a source, send a source to the wrong destination, or affect many
downstream channels and buses.

M001 still waits for real WING read-only evidence. This ADR does not grant live
authority or replace hardware/operator validation.

## Options Considered

1. Keep routing proposals usable when current route context is missing.
2. Require fresh read-only/no-authority route context before proposal emission.
3. Disable routing proposals entirely until WING hardware proof exists.

## Decision

Use option 2. Routing proposal candidates require fresh read-only/no-authority
current route context before a route diff can become an operator-facing
proposal.

## Why This Won

It advances the console-routing part of the AI Mixing Engineer goal while
preserving the current live boundary. It also handles a real failure mode:
stale or live-authority context can make the system trust the wrong patch state
or accidentally treat a mutable console path as evidence.

## Rejected Alternatives

- Allowing missing context was rejected because routing decisions need current
  route readback, topology evidence, source verification, rollback and scope
  bounds.
- Disabling the layer entirely was rejected because offline proposal gating and
  tests can improve the future assisted workflow without touching hardware.

## Implementation Plan

1. Require fresh current route context for proposal emission.
2. Require that context to be read-only and no-authority.
3. Block live-authority context even when marked fresh.
4. Expose no-authority evidence in the proposal-only contract.
5. Record M088 rail coverage without changing WING readiness.

## Test Plan

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_routing_management_proposal.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m088_routing_management_readonly_context_gates_stay_readonly -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Risks and Rollback

No live transport is introduced. Rollback is to revert the M088 commit, which
restores the previous routing-management proposal contract and M088 state/report
rows.
