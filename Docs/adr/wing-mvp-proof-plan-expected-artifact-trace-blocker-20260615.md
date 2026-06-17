# WING MVP Proof Plan Expected Artifact Trace Blocker

## Context

The supervised proof plan already reports accepted-candidate expected artifact
trace fields, and no-WING validation compares those values against the review
decision ledger. The final handoff step also requires an
expected-artifact-bound accepted source id before
`WING_SUPERVISED_FADER_GAIN_PROOF`.

Before this change, the trace was visible in `next_wing_proof_plan.summary`,
but `review_decision_accepted` did not directly depend on trace completion.

## Options considered

1. Keep the existing no-WING value gate as the only enforcement layer.
2. Add a direct proof-plan blocker when accepted-candidate expected artifact
   trace completion is false.
3. Move all trace validation into `wing_readonly_preflight_contract.py`.

## Decision

Use option 2. The proof-plan builder now requires
`readonly_preflight_review_decision_accepted_candidate_expected_artifact_trace_complete`
before accepted review decisions can unlock physical proof steps.

## Why this won

It aligns the proof-plan execution checklist with the existing handoff contract
without changing transport behavior, backend startup, readiness closure, or
live authority. It is the smallest fail-closed change at the point where
physical proof steps are generated.

## Rejected alternatives

- Option 1 was rejected because a later no-WING failure should not be the only
  layer preventing physical proof steps from being listed.
- Option 3 was deferred because the contract already exposes the lower-level
  prestart and artifact-root helpers, and this patch only needs a builder-level
  acceptance condition.

## Implementation plan

- Add a proof-plan blocker id for incomplete accepted-candidate expected
  artifact traces.
- Include trace completion in `review_decision_accepted`.
- Surface the blocker alongside existing review blockers.
- Require the blocker marker in the Project Director Current First Command
  baseline so operator/director surfaces match the proof-plan behavior.
- Add focused regression assertions.

## Test plan

- Focused proof-plan/no-WING/director guard suite.
- Full test suite.
- no-WING quick gate.
- Memory lint.
- `git diff --check`.

## Risks and rollback

Risk is limited to stricter offline proof planning. Rollback is to remove the
new blocker, trace-completion condition, and Current First Command marker; no
WING transport or mixer state is affected.
