# Brief: WING MVP Backend Prestart First-Step Next Action

## Context

The read-only preflight runbook and no-WING handoff validator require the
backend-port prestart check to be the first operator step before any intended
backend/WING session. The first WING MVP blocker selected action required the
structured backend prestart gate but did not explicitly name the first-step
fields.

## Task

Promote `backend_port_prestart_check_operator_sequence_step_index: 0`,
`backend_port_prestart_check_must_be_first_operator_step: true`, and
`backend_port_prestart_check_operator_sequence_step_is_first: true` into the
first WING MVP blocker selected action, generated progress, and autoloop docs.

## Safety Boundary

This is documentation and test synchronization only. It does not contact WING,
does not open a backend/WING session, and does not grant live write authority.
