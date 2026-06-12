# Brief: WING no-WING operator next-step summary

## Context

WING read-only evidence artifacts can carry `operator_next_step` when a run is
blocked before backend/WING contact. The artifact review report now preserves
and promotes this guidance, but the no-WING release summary still omitted it
from the WING evidence snapshot.

## Task

Carry artifact-level WING preflight next-step guidance into the no-WING release
summary without changing candidate acceptance or live authority.

## Acceptance

- The WING evidence schema check exposes unique artifact next steps.
- The no-WING JSON and Markdown summaries render those next steps.
- Invalid artifacts remain failed/blocked.
- No WING/backend/OSC contact is introduced.
