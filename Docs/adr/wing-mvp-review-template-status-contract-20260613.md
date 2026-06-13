# ADR: WING MVP Review Template Status Contract

## Context

The current MVP remains blocked on real `WING read-only connection proof`.
Before readiness or supervised proof can proceed, a clean read-only preflight
candidate must be manually reviewed and converted into an operator decision.
The generated review decision template is deliberately pending and must not be
confused with an accepted decision.

## Options Considered

1. Leave the pending template status owned by the review tool.
2. Check only that the review command writes a decision template.
3. Move the pending template status into the shared WING read-only contract and
   make no-WING validation compare the review tool status to that contract.

## Decision

Use option 3. The pending review decision template status is now owned by
`backend/wing_readonly_preflight_contract.py` and no-WING validation fails closed
if the review tool status drifts.

## Why This Won

The status is a safety contract: it proves the generated JSON is an operator
editing template, not accepted readiness evidence. Keeping it shared aligns
review output, generated templates, and no-WING summaries without contacting
hardware or granting live authority.

## Rejected Alternatives

Tool-local status was rejected because no-WING validation would not notice
semantic drift. Command-flag validation was rejected because it proves the tool
will write a file, not that the file remains a pending operator artifact.

## Implementation Plan

- Add `WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_STATUS` to the shared
  preflight contract.
- Alias the review tool status and expose `review_decision_template_status()`.
- Add no-WING detail/summary fields for the status and contract match.
- Fail closed on `wing_readonly_preflight_review_decision_template_status_mismatch`.
- Cover the contract with focused review and no-WING tests.

## Test Plan

- Compile touched Python modules.
- Run focused review/no-WING/director/rails tests.
- Run no-WING quick on an unused backend port.
- Run Project Director and direction guard.
- Run the full test suite.

## Risks And Rollback

The main risk is a stale test fixture or downstream consumer expecting only the
required-fields contract. Rollback would remove the status check, but that would
reopen a path where a generated template could drift toward accepted-decision
semantics without no-WING detecting it.
