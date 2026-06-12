# ADR: WING MVP Operator Sequence Step Indexes Next Action

## Context

The supervised WING Rack MVP remains blocked on real read-only connection proof.
Before any supervised fader/gain proof, the no-WING handoff must prove it still
points at the current first open blocker and keeps the supervised proof behind a
valid accepted-candidate prestart trace.

The selected action already names the digest review step and final supervised
gate step, but did not explicitly name their operator sequence indexes.

## Options Considered

1. Leave the selected action unchanged and rely on step content.
2. Add only prose saying the steps must be ordered.
3. Add the structured fields
   `operator_sequence_mvp_blocker_anchor_digest_review_step_index: 8` and
   `operator_sequence_supervised_prestart_gate_step_index: 9`.

## Decision

Use option 3.

## Why This Won

Structured field names are easier for humans, tests, and Project Director
output to check consistently than prose-only ordering. They also keep the
digest review immediately before the final supervised prestart gate without
granting live authority.

## Rejected Alternatives

- Prose-only ordering was rejected because it is harder to assert across
  selected-action and autoloop surfaces.
- Validator changes were deferred because this slice only exposes an already
  expected handoff invariant in the current WING rail.

## Implementation Plan

1. Add the two step-index fields to the first WING blocker next action and
   current command.
2. Mirror them in the read-only runbook, MVP rail, autoloop prompt, and README.
3. Update tests for Project Director and WING MVP rail surfaces.
4. Regenerate MVP progress and record validation.

## Test Plan

- Focused Project Director/WING rail tests.
- Direction guard and Project Director JSON checks.
- Diff check, memory lint, and full pytest.

## Risks And Rollback

Risk is limited to documentation/test drift. Rollback is to remove the two
fields from the docs and tests. No live WING behavior changes are included.
