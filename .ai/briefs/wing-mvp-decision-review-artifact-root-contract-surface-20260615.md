# Brief: WING MVP Decision Review Artifact Root Contract Surface

Date: 2026-06-15

## Objective

Make the standalone WING read-only decision-review report enforce and expose
accepted-candidate artifact-root contract validity before it can report an
accepted decision as ready.

## Context

The review-decision ledger and no-WING summary now carry accepted-candidate
artifact-root contract fields, but `tools/review_wing_readonly_preflight_decisions.py`
still based its ready status on the accepted-candidate prestart contract alone.
That left the standalone operator report less strict than the ledger and
no-WING surfaces.

## Safety Boundary

This is offline/read-only validation. It must not contact WING, mutate console
state, update readiness, or grant live authority.

## Acceptance

- Decision-review report ready status requires accepted-candidate artifact-root
  contract validity.
- Decision-review JSON and markdown expose accepted-candidate artifact-root
  contract fields.
- Missing artifact-root blocker coverage is rendered as a fallback blocker in
  markdown.
- Regression tests cover valid and invalid artifact-root contract reports.
