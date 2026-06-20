# ADR: Project Director External-Validation Scope Guard

## Context

M001 is waiting for a real WING Rack read-only artifact and operator review.
Recent offline context gates advanced separate proposal capabilities, but the
Project Director still allowed work that "stays offline/proposal-only" while
the selected blocker required external WING evidence. That creates a real risk
of reporting progress while not moving the active M001 blocker.

## Options Considered

1. Continue choosing more offline proposal/context gates until hardware is
   available.
2. Pause all Goal-mode work while M001 waits for external validation.
3. Keep Goal mode active, but restrict autonomous work to WING
   handoff/intake/control-plane state when the selected action requires real
   WING/operator evidence.

## Decision

Use option 3. Project Director now marks the current selected action as an
external-validation wait and narrows autonomous scope to
`external_validation_handoff_or_control_plane_only`.

## Why This Won

It answers the direction problem directly. The project can keep maintaining
the control plane and operator handoff without pretending that another broad
proposal-only context gate closes the current WING blocker.

## Rejected Alternatives

- Continuing broad offline proposal work was rejected because M001 already has
  a concrete next action: operator-supervised WING read-only preflight.
- Pausing all work was rejected because docs/state/rails can still preserve the
  handoff and prevent drift without touching live systems.

## Implementation Plan

1. Detect selected actions whose blocker status and action text require real
   WING/operator validation.
2. Replace the broad proposal-only allowed-scope clause with an external
   validation handoff/control-plane scope.
3. Add forbidden-scope language that blocks additional proposal-only
   context-gate milestones while M001 waits, unless the selected WING blocker
   or operator explicitly requests them.
4. Record M094 rail coverage without changing WING readiness.

## Test Plan

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_project_director_agent.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py::test_m094_project_director_external_validation_scope_guard_stays_readonly -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Risks and Rollback

No live transport is introduced. Rollback is to revert the M094 commit, which
restores the previous Project Director scope and M094 state/report rows.
