# Brief: wing-project-director-decision-proof-required-files-20260612

## Task

Add the WING read-only decision-review and supervised proof-plan files to
Project Director required files.

## Context

The current selected action requires accepted review source ids, a valid
accepted-candidate prestart trace, and the proof-plan blocker
`accepted_readonly_preflight_candidate_prestart_contract_required_before_supervised_proof`.
Project Director already points future cycles at the runner, artifact review,
handoff builder, and no-WING release check. It should also point them at the
decision ledger, decision review tool, and supervised proof plan that enforce
the accepted-candidate gate.

## Safety Boundary

- Do not contact WING hardware.
- Do not open a backend WING session.
- Do not execute OSC/WING writes.
- Do not close any MVP blocker from this planning-only change.
