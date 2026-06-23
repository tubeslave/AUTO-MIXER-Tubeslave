# ADR: WING MVP no-WING Operator Markdown Summary SHA Match

## 1. Context

The WING MVP remains blocked on real read-only WING proof. The no-WING
`summary.json` now binds to `summary.md` through a JSON-only SHA256, but the
handoff still needs a clear machine-readable match verdict for stale digest
cases.

## 2. Options considered

- Keep only the declared digest.
- Add an actual digest and match flag in `summary.json`.
- Render the digest into `summary.md`.

## 3. Decision

Add `operator_markdown_summary_actual_sha256` and
`operator_markdown_summary_sha256_matches_actual` to `next_wing_proof_plan`, and
fail closed when the declared digest does not match the actual markdown text.

## 4. Why this won

It follows the existing handoff pattern of declared digest, actual digest, and
match flag while avoiding recursive markdown hashing.

## 5. Rejected alternatives

Declared-digest-only was rejected because it leaves stale values harder to spot.
Markdown-rendered digest was rejected because it changes the artifact being
hashed.

## 6. Implementation plan

- Add an artifact-integrity helper after the markdown SHA binding.
- Expose actual digest and match flag under `next_wing_proof_plan`.
- Add a stale-SHA fail-closed test.
- Require the match flag in Project Director baseline coverage.

## 7. Test plan

Run focused no-WING summary tests, focused Project Director/WING rail tests, full
pytest, `git diff --check`, and memory lint.

## 8. Risks and rollback

Risk is limited to marker drift in docs/tests. Roll back by removing the helper,
fields, and Project Director markers. This ADR does not grant live WING
authority.
