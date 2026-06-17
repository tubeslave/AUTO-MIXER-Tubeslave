# ADR: WING MVP Accepted Candidate Artifact Root Canonical Root Gate

## 1. Context

Accepted read-only WING preflight candidates gate the transition toward
physical supervised fader/gain proof planning. The artifact-root helper checks
candidate artifact paths against the declared root, but a root value containing
parent traversal can normalize to a wider directory.

## 2. Options considered

- Keep only child path checks.
- Add local planner and no-WING validator checks.
- Add a shared root-level parent traversal violation.

## 3. Decision

Add `artifact_root_contains_parent_traversal` to the shared accepted-candidate
artifact-root contract helper.

## 4. Why this won

The helper is already the common authority for the proof plan and no-WING
validator. One root-level violation keeps behavior consistent and avoids
granting confidence from ambiguous path metadata.

## 5. Rejected alternatives

Child-only checks can miss the case where the declared root itself widens after
normalization. Duplicating checks would invite drift. Hard-coding one literal
root would be too narrow for file-based tests and operator-selected artifact
directories.

## 6. Implementation plan

- Detect `..` segments in `artifact_root`.
- Preserve existing child path checks.
- Add direct helper, supervised proof plan, and no-WING validator regressions.

## 7. Test plan

Run focused tests, WING-focused tests, standard pytest, no-WING release check,
diff check, memory lint, and Project Director.

## 8. Risks and rollback

Risk is limited to stricter rejection of malformed offline candidate metadata.
Rollback is reverting the helper violation and the three regressions.
