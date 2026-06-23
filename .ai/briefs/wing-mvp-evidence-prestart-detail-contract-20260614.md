# Brief: WING MVP Evidence Prestart Detail Contract

## Task

Make post-run WING read-only evidence intake reject captured artifacts whose
referenced no-WING prestart summary carries a stale backend-port detail contract.

## Context

The active WING MVP blocker remains `WING read-only connection proof`. The real
runner now blocks stale no-WING prestart summaries before WebSocket. The offline
reviewer must enforce the same contract after a captured artifact is reviewed,
otherwise a weaker post-run gate could accept evidence the runner is supposed to
reject.

## Scope

- Validate the referenced no-WING prestart summary's backend-port
  `backend_port_prestart_check_failure_detail_fields` list.
- Validate every required-false backend-port detail field.
- Compare recorded artifact summary values against the referenced no-WING
  prestart detail values.
- Add focused evidence tests and runbook wording.

## Non-goals

- Do not contact WING.
- Do not start, stop, or reconfigure backend/WING listeners.
- Do not grant live authority.
- Do not close the first WING blocker from offline evidence.
