# WING Proof Plan Accepted Candidate Prestart Trace

## Context

The no-WING release summary now surfaces WING read-only preflight decision `candidate_reviews` with backend-port prestart contract fields. The next supervised WING proof plan still needs to prove that an accepted decision source is tied to a candidate row carrying that prestart contract before listing any physical proof steps.

## Task

Bind `next_wing_proof_plan` acceptance to prestart-valid accepted candidate rows. If an accepted source id is missing from `candidate_reviews` or lacks the required backend-port prestart fields, the proof plan must stay blocked for manual review.

## Constraints

- Keep the plan non-executing and operator-required.
- Do not contact WING, start a backend, or run OSC/WebSocket writes.
- Reuse the shared WING read-only preflight contract constants.
- Surface the trace in no-WING release summary JSON and markdown.

## Test Plan

- Focused proof-plan and no-WING next-plan tests.
- Affected WING/no-WING tests.
- Full backend pytest suite.
- `git diff --check`.
- `tools/memory/memory_lint.py`.
