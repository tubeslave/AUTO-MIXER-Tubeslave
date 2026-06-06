# ADR: WING Read-Only Preflight Evidence Mixed Artifacts Fail Closed

## Context

`backend/wing_readonly_preflight_evidence.py` reviews captured
`wing_readonly_preflight.v1` summaries as read-only operator evidence. These
artifacts can inform a manual readiness-matrix review, but they must not close
proof blockers or grant live authority automatically.

## Options considered

1. Accept any valid artifact and keep invalid artifacts as warnings.
2. Select only the latest valid artifact by run id or path ordering.
3. Treat the artifact set as fail-closed when any invalid artifact is present.

## Decision

Use option 3. Mixed valid+invalid artifact sets are
`readonly_preflight_artifact_replacement_required` and cannot expose
`readiness_matrix_update_allowed=true`.

## Why this won

The next supervised WING proof depends on operator-visible evidence being
unambiguous. A corrupt, stale, or unsafe summary next to a valid summary means
the artifact root is not clean enough for readiness review.

## Rejected alternatives

Accepting one valid artifact is simpler but can hide broken evidence in the
same proof set. Timestamp arbitration is also fragile because the runner output
may be copied between directories during manual hardware sessions.

## Implementation plan

- Derive a global candidate only when `valid_candidate_count > 0` and
  `invalid_candidate_count == 0`.
- Mark mixed artifact sets as replacement-required.
- Keep matrix rows manual-review-only and closed=false.
- Update release-check validation to use the same clean-set requirement.

## Test plan

- Focused preflight evidence tests cover valid-only, invalid-only, missing, and
  mixed valid+invalid artifact sets.
- Release-check tests reject corrupted summaries that claim readiness-update
  permission while invalid artifacts are present.
- Memory lint remains green.
- The full pytest suite remains green.

## Risks and rollback

Risk: a real operator may need to move invalid artifacts out of the evidence
root before a readiness-review candidate appears. Rollback is a small revert of
the candidate calculation and tests, with no live mixer state involved.
