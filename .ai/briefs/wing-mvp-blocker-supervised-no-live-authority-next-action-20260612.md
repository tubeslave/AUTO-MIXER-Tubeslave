# Brief: WING MVP Blocker Supervised No-Live Authority Next Action

## Context

The WING read-only handoff validator and runbook already require
`supervised_prestart_gate.grants_live_authority: false`. The first open MVP
blocker next-action text required the structured supervised gate and accepted
candidate trace, but did not explicitly name the no-live-authority field in the
operator-facing selected action.

## Task

Promote `supervised_prestart_gate.grants_live_authority: false` into the first
open WING MVP blocker next action and generated progress, with focused tests
that keep Project Director aligned with the no-live supervised gate.

## Safety Boundary

This is documentation and test synchronization only. It does not contact WING,
does not open a backend/WING session, and does not grant live write authority.
