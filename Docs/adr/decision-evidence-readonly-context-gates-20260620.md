# ADR: Decision Evidence Read-Only Context Gates

## Context

The project goal requires each Automixer decision to be based on audio
analysis, psychoacoustic and statistical models, project history, reference
mixes, previous decision outcomes, inter-channel analysis, arrangement context
and genre context. Decision evidence packets are the layer that can bind those
inputs before downstream proposal families rank or fan out work.

M001 still waits for real WING read-only evidence. This ADR does not grant live
authority or replace hardware/operator validation.

## Options Considered

1. Keep decision-evidence packets usable when current decision context is
   missing.
2. Require fresh read-only/no-authority decision context, trusted source
   identity, current readback state and rollback planning before proposal
   emission.
3. Disable decision-evidence packets entirely until WING hardware proof exists.

## Decision

Use option 2. Decision-evidence packet candidates require fresh read-only/no-
authority current decision context, trusted source identity, current readback
state and rollback planning for console-affecting packets before they can
become operator-facing proposals.

## Why This Won

It advances the decision-basis part of the AI Mixing Engineer goal without
widening live scope. It also prevents stale context or live-authority-bearing
context from being laundered into downstream proposal ranking.

## Rejected Alternatives

- Allowing missing context was rejected because evidence packets become a
  shared ranking and explanation surface for downstream proposals.
- Disabling the layer entirely was rejected because offline proposal gating and
  tests can improve future assisted workflows without touching hardware.

## Implementation Plan

1. Record the decision-evidence freshness gates as M093.
2. Require fresh current decision context for proposal emission.
3. Require no-authority decision context.
4. Keep missing, stale, live-authority, source-untrusted, unreadbacked or
   rollback-missing candidates analysis-only.
5. Record M093 rail coverage without changing WING readiness.

## Test Plan

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_decision_evidence_packet_proposal.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_project_goal_capability_matrix.py::test_decision_basis_is_tracked_as_proposal_layer_without_live_widening -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m093_decision_evidence_readonly_context_gates_stay_readonly -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Risks and Rollback

No live transport is introduced. Rollback is to revert the M093 commit, which
restores the previous decision-evidence rail documentation and M093 state/report
rows.
