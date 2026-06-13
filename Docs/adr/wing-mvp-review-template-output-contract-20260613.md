# ADR: WING MVP Review Template Output Contract

## Context

The current WING MVP remains blocked on real `WING read-only connection proof`.
After a clean read-only preflight, the operator writes an edited review decision
JSON. The pending template output path for that JSON appeared in several tools:
the runner's post-run review commands, the handoff packet builder, the review
tool default, and the no-WING release checker.

## Options Considered

1. Keep the path duplicated.
2. Centralize only the template filename.
3. Centralize both the decision root and the pending template output path in the
   shared WING read-only preflight contract.

## Decision

Use option 3. The shared contract now owns
`WING_READONLY_PREFLIGHT_REVIEW_DECISION_ROOT` and
`WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_OUTPUT`.

## Why This Won

The decision root and template path are part of the operator handoff contract,
not one tool's private default. A shared source keeps the runner plan,
handoff packet, review CLI, and no-WING validator aligned without changing
runtime authority or contacting hardware.

## Rejected Alternatives

Duplicated literals were rejected because they can send operators and validators
to different files. Filename-only centralization was rejected because the root
directory is also part of the contract. Tool-owned constants were rejected
because this path is consumed across independent tools.

## Implementation Plan

- Add shared decision-root and pending-template-output constants.
- Replace tool-level literals with aliases to the shared contract.
- Preserve public command arguments and path values.
- Add focused tests for generated commands and defaults.

## Test Plan

- Compile touched Python modules.
- Run focused WING readonly runner, handoff, review, and no-WING tests.
- Run no-WING quick on an unused backend port.
- Run Project Director and direction guard.
- Run the full pytest suite.

## Risks And Rollback

The main risk is changing the public path accidentally. Rollback would restore
local literals, but that would reopen path drift in the manual-review handoff.
