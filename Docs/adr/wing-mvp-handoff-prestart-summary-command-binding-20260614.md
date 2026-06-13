# ADR: WING handoff prestart summary command binding

## Context

The read-only WING handoff is generated and validated without touching WING.
It already carries a prestart summary placeholder and command list, but the
operator-facing no-WING summary did not explicitly prove that preview and
capture commands used the validated plan's prestart summary path.

## Options considered

1. Keep the binding implicit in whole-command equality.
2. Expose and validate preview/capture prestart-summary bindings in the
   summary.
3. Defer the check to the real hardware run.

## Decision

Use option 2. The no-WING handoff validator must compare both preview and
capture `--prestart-summary` values with the validated plan artifact path and
surface the result in JSON and Markdown summaries.

## Why this won

The operator handoff must be auditable before a real WING session starts.
Explicit match booleans make stale or hand-edited prestart summary paths fail
closed in the offline gate.

## Rejected alternatives

Implicit command equality hides the safety property in a long command list.
Deferring to hardware is too late for a prestart gate whose purpose is to
protect the hardware session boundary.

## Implementation plan

- Derive the expected prestart summary path from the validated plan artifact.
- Extract preview and capture command `--prestart-summary` values.
- Add mismatch violations and summary fields.
- Update no-WING tests and WING MVP rail docs.

## Test plan

- Focused no-WING handoff tests.
- Project Director/WING rail tests.
- no-WING quick release check.
- Full pytest before commit.

## Risks and rollback

Older hand-edited packets may fail until regenerated. Rollback removes the new
fields and mismatch checks, but that weakens prestart handoff visibility.
