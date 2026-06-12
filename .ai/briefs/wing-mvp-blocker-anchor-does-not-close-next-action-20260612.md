# Brief: WING MVP Blocker Anchor Does Not Close Next Action

## Context

The WING read-only handoff packet carries `mvp_blocker_anchor` and explicitly
marks `packet_does_not_close_blocker: true`. The read-only runbook already asks
operators to verify that field, but the Project Director selected action did
not include it.

## Task

Promote `mvp_blocker_anchor.packet_does_not_close_blocker: true` into the first
open WING MVP blocker next action and generated progress.

## Safety Boundary

This is documentation and test synchronization only. It does not contact WING,
does not open a backend/WING session, and does not grant live write authority.
