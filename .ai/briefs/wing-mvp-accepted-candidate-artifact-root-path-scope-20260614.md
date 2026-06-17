# Brief: WING MVP Accepted Candidate Artifact Root Path Scope

## Task

Close a fail-open gap in the accepted-candidate artifact-root contract used
before supervised WING proof planning.

## Context

The WING read-only proof chain already requires accepted candidate prestart
traces and artifact-root traces before physical supervised proof steps can be
planned. The artifact-root helper, however, only checked that the accepted
`source_id` lived under `artifact_root`. A stale or hand-built candidate row
could keep `source_id` under the declared root while moving `summary_path` or
expected summary/event artifact paths elsewhere.

## Constraints

- Stay on the first WING MVP blocker.
- Keep all changes read-only/offline.
- Do not grant live authority or close readiness from candidate evidence.
- Use one shared contract helper for planner and no-WING validation.

## Desired Outcome

Accepted read-only candidates must keep `source_id`, `summary_path`,
`events_path`, `expected_summary_artifact`, and `expected_events_artifact`
under the declared artifact root before physical proof steps can be planned.
