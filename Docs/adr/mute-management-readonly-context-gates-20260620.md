# ADR: Mute Management Read-Only Context Gates

## Context

The project goal requires full digital-console control, including mute and
unmute decisions. Mute changes are high risk because they can create dropouts,
hide performer intent, override manual safety decisions, or unmute a feedback
source.

M001 still waits for real WING read-only evidence. This ADR does not grant live
authority or replace hardware/operator validation.

## Options Considered

1. Keep mute proposals usable when current mute context is missing.
2. Require fresh read-only/no-authority mute context before proposal emission.
3. Disable mute proposals entirely until WING hardware proof exists.

## Decision

Use option 2. Mute/unmute proposal candidates require fresh
read-only/no-authority current mute context before they can become
operator-facing proposals.

## Why This Won

It advances the console-mute part of the AI Mixing Engineer goal while
preserving the current live boundary. It also handles a real failure mode:
stale or live-authority context can make the system trust the wrong mute state
or treat a mutable console path as evidence.

## Rejected Alternatives

- Allowing missing context was rejected because mute/unmute decisions need
  current state, intent, source identity, manual-mute preservation, readback and
  rollback evidence.
- Disabling the layer entirely was rejected because offline proposal gating and
  tests can improve the future assisted workflow without touching hardware.

## Implementation Plan

1. Require fresh current mute context for proposal emission.
2. Require that context to be read-only and no-authority.
3. Block live-authority context even when marked fresh.
4. Expose no-authority evidence in the proposal-only contract.
5. Record M089 rail coverage without changing WING readiness.

## Test Plan

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_mute_management_proposal.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m089_mute_management_readonly_context_gates_stay_readonly -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Risks and Rollback

No live transport is introduced. Rollback is to revert the M089 commit, which
restores the previous mute-management proposal contract and M089 state/report
rows.
