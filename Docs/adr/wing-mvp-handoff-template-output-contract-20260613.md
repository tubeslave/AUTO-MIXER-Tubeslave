# ADR: WING MVP Handoff Template Output Contract

## Context

The current MVP remains blocked on real `WING read-only connection proof`.
Before readiness or supervised proof can proceed, a clean read-only preflight
candidate must be manually reviewed and converted into an operator decision.

The review decision template output path is contract-owned, and no-WING
evidence validation checks the review command value. The handoff packet artifact
summary still only showed whether the command wrote a template, not whether the
packet path and command path matched the shared contract.

## Options Considered

1. Keep handoff summary at flag-only visibility.
2. Add a structured handoff packet contract-match field and no-WING validation.
3. Only add Markdown text to the handoff packet.

## Decision

Use option 2. The handoff packet now declares
`decision_template_output_matches_contract`, and no-WING handoff artifact
validation fails closed when the packet path or review command path drifts from
`WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_OUTPUT`.

## Why This Won

The handoff packet is the operator-facing artifact. It should show the exact
pending decision template path and make the contract match machine-checkable
without relying on a separate evidence section.

## Rejected Alternatives

Flag-only visibility was rejected because it cannot catch path drift. Markdown
only was rejected because it gives humans a hint but does not let the release
gate fail closed on stale packet contents.

## Implementation Plan

- Add `decision_template_output_matches_contract` to the handoff packet.
- Render the match result in the packet's expected-artifacts section.
- Parse and validate packet and command template-output paths in no-WING.
- Surface packet path, command path, and match booleans in JSON and Markdown.
- Cover valid and mismatched handoff packets in tests.

## Test Plan

- Compile touched tools.
- Run focused handoff/no-WING/director/guard/rails tests.
- Run no-WING quick on an unused backend port.
- Run Project Director and direction guard.
- Run `git diff --check`, memory lint, and the full pytest suite.

## Risks And Rollback

The main risk is stale fixtures or consumers that assume the handoff artifact
summary only has the old boolean flag. Rollback would remove the structured
match fields, but that would reopen a path where stale handoff packets can point
operator review at the wrong pending decision template.
