# Brief: WING MVP Blocker Supervised Step Markers Next Action

## Context

The WING read-only handoff packet validator already requires the final
supervised prestart operator step to include three markers:
`accepted_decision_ready`, `valid accepted-candidate prestart trace`, and
`WING_SUPERVISED_FADER_GAIN_PROOF`.

The first open MVP blocker selected action required the supervised gate to exist
and be final, but did not explicitly require those markers in the selected
action text.

## Task

Promote the final supervised operator-step marker requirement into the first
open WING MVP blocker next action and generated progress.

## Safety Boundary

This is documentation and test synchronization only. It does not contact WING,
does not open a backend/WING session, and does not grant live write authority.
