# ADR: WING MVP no-WING Operator Markdown Summary SHA Binding

## 1. Context

The WING MVP remains blocked on real read-only WING proof. The no-WING
`summary.md` proof-plan section is now self-checking, but `summary.json` still
needed a machine-readable binding to the exact markdown artifact bytes.

## 2. Options considered

- Store no markdown digest.
- Render the digest inside `summary.md`.
- Store the digest in `summary.json` only.

## 3. Decision

Store `operator_markdown_summary_artifact`, `operator_markdown_summary_sha256`,
and `operator_markdown_summary_sha256_available` in `summary.json` only.

## 4. Why this won

The JSON-only digest binds the reviewed markdown artifact without recursive hash
instability. It keeps the bundle self-describing and does not require another
offline validator.

## 5. Rejected alternatives

No digest was rejected because it leaves the operator surface weakly bound to the
machine-readable summary. Markdown-rendered digest was rejected because the
digest would change the artifact being hashed.

## 6. Implementation plan

- Compute the SHA256 after the final markdown status line is rendered.
- Store the artifact name and digest under `next_wing_proof_plan`.
- Require those fields in Project Director baseline markers.
- Add tests that hash the actual written markdown.

## 7. Test plan

Run focused no-WING summary tests, focused WING rail tests, full pytest,
`git diff --check`, and memory lint.

## 8. Risks and rollback

Risk is limited to stale field names in Project Director rails. Roll back by
removing the binding helper and its baseline markers. This ADR does not grant
live WING authority.
