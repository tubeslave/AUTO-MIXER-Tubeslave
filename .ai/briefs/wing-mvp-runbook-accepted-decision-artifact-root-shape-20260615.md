# Brief: WING MVP Runbook Accepted-Decision Artifact-Root Shape

## Task

Make the WING read-only runbook accepted-decision JSON shape include the
`artifact_root` field required by the review-decision ledger.

## Context

The shared decision-template contract and generated pending decision template
already include `artifact_root`. The runbook minimum accepted-decision example
omitted it, which could lead an operator to create a decision record that
fails closed during review.

## Safety Boundary

This is a documentation and regression-test alignment change. It does not
contact WING, mutate console state, close readiness, or grant live authority.

## Expected Evidence

- Runbook minimum JSON shape includes `artifact_root`.
- Runbook text tells the operator to keep `artifact_root` aligned with the
  template and captured artifact.
- Tests guard the runbook wording and the template/ledger artifact-root
  contract.
