# ADR: WING MVP no-WING Artifact Manifest Sidecar

## 1. Context

The WING MVP remains blocked on real read-only WING proof. The no-WING bundle
now includes digest fields for the operator-readable markdown summary, but a
detached manifest is needed to hash both `summary.json` and `summary.md` without
recursive digest fields.

## 2. Options considered

- Store no sidecar manifest.
- Put `summary.json` SHA inside `summary.json`.
- Write `artifact-manifest.json` after both summary artifacts are written.

## 3. Decision

Write `artifact-manifest.json` as a sidecar manifest for each no-WING run.

## 4. Why this won

The sidecar can hash the final `summary.json` and `summary.md` bytes without
changing either artifact after hashing. It also gives reviewers one compact
integrity surface for the no-WING handoff bundle.

## 5. Rejected alternatives

No sidecar was rejected because it leaves whole-file integrity weaker.
Self-hashing `summary.json` was rejected because it is recursive.

## 6. Implementation plan

- Add a manifest constant and writer helper.
- Add manifest safety flags and summary header discovery fields.
- Write the manifest after `summary.json` and `summary.md`.
- Require manifest markers in Project Director baseline coverage.

## 7. Test plan

Run focused no-WING summary tests, focused Project Director/WING rail tests, full
pytest, `git diff --check`, and memory lint.

## 8. Risks and rollback

Risk is limited to stale marker wording or manifest schema churn. Roll back by
removing the sidecar writer and Project Director markers. This ADR does not
grant live WING authority.
