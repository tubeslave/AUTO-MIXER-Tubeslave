# ADR: WING MVP Operator Sequence Count Next Action

## Context

The supervised WING Rack MVP is still waiting for real read-only hardware
proof. The offline no-WING handoff packet prepares the operator sequence for
that proof, but does not grant live authority.

The release checker already verifies that the handoff operator sequence has ten
steps. The selected action did not name that total count explicitly.

## Options Considered

1. Keep the count only in the release checker.
2. Add prose saying the sequence must be complete.
3. Add the structured selected-action field `operator_sequence_count: 10`.

## Decision

Use option 3.

## Why This Won

The structured count is already machine-checkable evidence. Naming it in the
selected action and runbook lets operator review reject handoff packets with
missing or extra steps before supervised fader/gain proof planning.

## Rejected Alternatives

Prose-only completeness was rejected because it is harder to assert across the
Project Director, progress report, runbook, and autoloop surfaces.

## Implementation Plan

1. Add `operator_sequence_count: 10` to the first WING blocker next action and
   current command.
2. Mirror the field in read-only runbook, MVP rail, autoloop prompt, and README.
3. Update tests and regenerate MVP progress.
4. Record validation in a report.

## Test Plan

- Focused WING rail tests.
- Direction guard and Project Director checks.
- Diff check, memory lint, and full pytest.

## Risks And Rollback

Risk is limited to documentation/test drift. Rollback removes the field from
docs and assertions. No live mixer behavior changes are included.
