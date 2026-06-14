# ADR: WING MVP Current Command Director Action Digest

## Context

The WING read-only handoff packet now carries a Project Director selected-action
anchor and validates its SHA256. Project Director uses `Current First Command`
as the selected action when that command covers the baseline marker set. Before
this ADR, the current command did not require the newly added handoff director
digest fields, so future Goal-mode cycles could pass baseline coverage without
mentioning the packet-level selected-action proof.

## Options considered

1. Leave the requirement only in the handoff packet validator.
2. Add the requirement to `Current First Command` only.
3. Add it to `Current First Command`, the baseline marker set, autoloop docs,
   and rails tests.

## Decision

Use option 3.

## Why this won

The Project Director action is the work instruction for Goal-mode cycles. If
the handoff packet now proves a selected-action digest, the action text and
coverage gate should name that proof explicitly. This keeps autonomous cycles
aligned with the packet validator without granting live authority.

## Rejected alternatives

Validator-only coverage is too easy for future planning text to omit. Command-
only wording without baseline markers would not fail when the markers drift.

## Implementation plan

Add the `project_director_selected_action_*` fields and baseline coverage
verdicts to `## Current First Command`. Add the same markers to
`CURRENT_FIRST_COMMAND_BASELINE_COVERAGE_MARKERS`. Document the no-WING handoff
summary requirement in the autoloop prompt/README and extend Project
Director/progress tests.

## Test plan

Run focused Project Director, WING rail, autoloop prompt, and autoloop contract
tests. Then run Project Director, direction guard, no-WING quick fail-closed,
memory lint, `git diff --check`, and the standard full test command.

## Risks and rollback

Risk is limited to offline planning text and baseline coverage gating. Rollback
is to remove the new markers from the current command, baseline tuple, and
tests if the handoff packet schema changes. This change does not start WING,
query a console, write to a console, update readiness, or close the blocker.
