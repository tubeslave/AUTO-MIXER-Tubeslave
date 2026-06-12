# Brief: wing-project-director-handoff-required-file-20260612

## Task

Add the WING read-only handoff packet builder to Project Director required
files.

## Context

The current WING MVP next action depends on no-WING handoff summary fields such
as `backend_port_prestart_gate`, `expected_backend_port_prestart_gate`, and the
final `supervised_prestart_gate`. Project Director already requires the runner,
review tool, and no-WING release check, but not the handoff packet builder that
defines and renders those gate contracts.

## Safety Boundary

- Do not contact WING hardware.
- Do not open a backend WING session.
- Do not execute OSC/WING writes.
- Do not close any MVP blocker from this planning-only change.
