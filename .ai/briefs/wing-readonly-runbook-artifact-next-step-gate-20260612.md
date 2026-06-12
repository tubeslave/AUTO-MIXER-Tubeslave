# Brief: WING read-only runbook artifact next-step gate

## Context

The no-WING release summary now carries
`wing_readonly_preflight_evidence.artifact_operator_next_steps` from blocked
read-only preflight artifacts. The WING read-only runbook should tell operators
to inspect that field before opening a backend/WING WebSocket.

## Task

Update the runbook and rails test so artifact repair guidance is part of the
operator prestart gate.

## Acceptance

- The runbook names `artifact_operator_next_steps`.
- The runbook says the guidance must be handled before backend/WING WebSocket
  opening.
- The runbook preserves the no-live-authority boundary.
- Rails tests cover the instruction.
