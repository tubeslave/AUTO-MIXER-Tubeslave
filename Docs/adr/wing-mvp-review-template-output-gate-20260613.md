# ADR: WING MVP Review Template Output Gate

## Context

The current MVP remains blocked on real `WING read-only connection proof`.
Before readiness or supervised proof can proceed, a clean read-only preflight
candidate must be manually reviewed and converted into an operator decision.

The pending review decision template output path is now centralized in the
shared WING read-only preflight contract. However, the no-WING release checker
only proved that the review command contained `--decision-template-output`; it
did not prove that the value after the flag matched the shared contract.

## Options Considered

1. Keep checking only for the flag.
2. Add a no-WING comparison against the shared output-path contract.
3. Move the check into the review CLI and leave no-WING unchanged.

## Decision

Use option 2. no-WING validation now extracts the review command's
`--decision-template-output` value and fails closed unless it equals
`WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_OUTPUT`.

## Why This Won

The release checker is the safety boundary that tells the operator whether the
offline/no-hardware evidence is internally consistent. It must detect a review
command that writes the pending decision template to a stale or unexpected path,
even when the flag is present.

## Rejected Alternatives

Flag-only validation was rejected because it cannot catch path drift. Review
CLI-only validation was rejected because no-WING summaries still need to expose
the exact path used by the operator-facing review command.

## Implementation Plan

- Parse the review decision template output path from the review command.
- Add detail and summary fields for the parsed output and contract match.
- Fail closed on
  `wing_readonly_preflight_review_decision_template_output_mismatch`.
- Cover happy path, missing flag, and wrong-path cases in tests.

## Test Plan

- Compile the no-WING release checker.
- Run focused no-WING, Project Director, direction guard, and WING rails tests.
- Run no-WING quick on an unused backend port.
- Run Project Director and direction guard.
- Run `git diff --check`, memory lint, and the full pytest suite.

## Risks And Rollback

The main risk is stale report fixtures or consumers that only expected the
boolean flag field. Rollback would remove the output-path match, but that would
reopen the path where the operator review template can drift away from the
shared contract without no-WING detecting it.
