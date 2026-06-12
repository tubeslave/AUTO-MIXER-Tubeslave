# ADR: WING MVP Read Only Required By Packet Next Action

## Context

The supervised WING Rack MVP remains blocked on real WING read-only proof.
No-WING handoff packets can prepare operator review, but they must identify
themselves as read-only before any supervised proof planning.

## Options Considered

1. Keep only live-write denial.
2. Add prose saying the packet is read-only.
3. Require `read_only: true` in the selected action and operator-facing rails.

## Decision

Use option 3.

## Why This Won

The structured field is already emitted by the handoff packet builder and can
be asserted across the blocker table, runbook, generated progress, Project
Director, and autoloop surfaces. It prevents no-WING handoff evidence from
being misread as anything beyond read-only proof preparation.

## Rejected Alternatives

Prose-only wording was rejected because it is harder to assert and easier to
drift from generated evidence.

## Implementation Plan

1. Add the packet read-only field to the first WING blocker selected action and
   current command.
2. Mirror the requirement in MVP/runbook/autoloop docs.
3. Update tests and regenerate progress.
4. Record validation in a report.

## Test Plan

- Focused WING rail tests.
- Direction guard and Project Director checks.
- Diff check, memory lint, and full pytest.

## Risks And Rollback

Risk is limited to docs/test drift. Rollback removes the field requirement from
docs and tests. No live WING behavior changes are included.
