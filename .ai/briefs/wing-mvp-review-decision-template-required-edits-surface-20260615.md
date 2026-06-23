# Brief: WING MVP Review Decision Template Required-Edits Surface

## Task

Expose the required operator edits for generated WING read-only review decision
templates in the review report and CLI help.

## Context

The review-decision ledger now fails closed when an accepted decision still
carries the generated pending `template_status` marker. The generated review
report and CLI help still described only `decision`, `reviewed_by`, and
`reviewed_at`, which could leave the operator-facing path stale during a
physical WING session.

## Safety Boundary

This is an offline review-report guidance change. It does not contact WING,
mutate console state, update readiness rows, or grant live authority.

## Expected Evidence

- Review reports expose `review_decision_template_required_edits`.
- Markdown includes the pending-template-status removal requirement.
- CLI help names the pending `template_status` marker as something the
  operator must remove or replace.
