# ADR: WING MVP Backend Listener Template Edit Requirement Canon

Date: 2026-06-13
Status: accepted

## Context

The backend-port prestart validator rejects partially edited generated listener
confirmation templates. The WING MVP blocker rail must expose that same
requirement so future Project Director output does not plan from stale
pending-template-only criteria.

## Options considered

1. Keep the requirement only in validator tests and runbook text.
2. Add a separate schema file for listener confirmation template acceptance.
3. Add the reviewed-template requirement to the current blocker next action and
   generated progress report.

## Decision

Use option 3. The first blocker next action now names
`listener_confirmation_template_required_edits`, the reviewed template status,
the `accepted_by_template: true` marker, and the ban on `REPLACE_WITH_*`
operator placeholders in template-derived accepted confirmations.

## Why this won

Project Director and goal-mode cycles already select work from
`.ai/project-layer/wing_mvp_blockers.md`. Keeping the stricter gate there makes
the current safety boundary visible before any real WING preflight.

## Rejected alternatives

Keeping the requirement only in the runbook leaves the autonomous planning rail
weaker than the validator. A schema split could be cleaner but would be broader
than needed for this correction.

## Implementation plan

Update the blocker table, regenerate `MVP_PROGRESS_LATEST.md`, update MVP and
autoloop docs, and add rail tests that assert the stricter markers appear in
Project Director and generated progress output.

## Test plan

Run focused rail tests, Project Director/direction guard checks, memory lint,
and the standard full pytest command.

## Risks and rollback

If the next-action text becomes too hard to maintain, roll back only the added
canonical wording and replace it with a structured contract file referenced by
the blocker row.
