# Brief: WING MVP Current Evidence Listener Probe Error Report

## Task

Keep the current first WING MVP blocker evidence aligned with the latest
validated backend-port prestart listener-probe diagnostic report.

## Context

The first open blocker remains `WING read-only connection proof`. The
`listener_process_error` contract report exists and is indexed, but Project
Director derives `selected_evidence_files` from the first blocker evidence cell.
If the report is not listed there, autonomous cycles can miss the newest
diagnostic boundary for busy backend-port owner-probe failures.

## Safety boundary

- Do not contact WING hardware.
- Do not start, stop, or reconfigure backend/WING listeners.
- Do not grant live authority.
- Do not close readiness from documentation-only evidence.

## Required outcome

- Add the listener-probe error field report to the first blocker evidence list.
- Assert WING MVP rails and Project Director expose it.
- Record the decision and validation trail.
