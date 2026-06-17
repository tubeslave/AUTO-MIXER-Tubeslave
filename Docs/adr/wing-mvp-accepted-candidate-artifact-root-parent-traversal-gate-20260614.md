# ADR: WING MVP Accepted Candidate Artifact Root Parent Traversal Gate

## 1. Context

Accepted read-only WING preflight candidates can gate the next supervised
fader/gain proof plan only after manual review and several offline contracts.
The artifact-root contract already checked that accepted source, summary,
event, and expected artifact paths appeared under the declared artifact root,
but it did that with a string prefix check.

## 2. Options considered

- Keep prefix-only path checks.
- Duplicate traversal checks in planner and release validation.
- Strengthen the shared artifact-root contract helper.

## 3. Decision

Strengthen the shared accepted-candidate artifact-root contract helper to
normalize paths and reject parent traversal segments.

## 4. Why this won

The helper is already the common authority for supervised proof planning and
no-WING validation. Updating it once prevents drift and keeps the change
offline, small, and transport-free.

## 5. Rejected alternatives

Prefix-only checks can miss paths that escape after normalization. Duplicate
planner/release validation would recreate the drift this helper was introduced
to avoid. A broad schema migration is larger than needed for the current
blocker.

## 6. Implementation plan

- Normalize candidate artifact paths with POSIX path rules.
- Add `*_contains_parent_traversal` violations when a path includes `..`.
- Keep existing `*_not_under_artifact_root` violations when the normalized path
  leaves the declared root.
- Add direct helper, proof-plan, and no-WING validator regressions.

## 7. Test plan

Run focused helper/planner/no-WING tests, then the WING-focused suite, standard
pytest, no-WING release check, diff check, memory lint, and Project Director.

## 8. Risks and rollback

Risk is limited to stricter rejection of malformed offline candidate metadata.
Rollback is reverting the helper normalization and the three regressions.
