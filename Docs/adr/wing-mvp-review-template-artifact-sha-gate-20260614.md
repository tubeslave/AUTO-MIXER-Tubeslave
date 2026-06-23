# ADR: WING MVP Review Template Artifact SHA Gate

## 1. Context

The WING read-only review decision template builder required non-empty
`summary_sha256` and `events_sha256` fields, but did not verify that they were
valid SHA-256 hex strings. A malformed report mapping could therefore produce a
pending review decision template with invalid artifact digests.

## 2. Options Considered

- Keep relying on evidence intake and review-decision ledger checks.
- Add broad schema validation to all review reports.
- Add local artifact SHA validation in the review template tool.

## 3. Decision

Use the shared SHA-256 validator before review decision template generation.

## 4. Why This Won

The decision template is a manual-review boundary and should fail closed on its
own. The change uses an existing contract helper, adds no dependency, and
preserves no-live-authority semantics.

## 5. Rejected Alternatives

Relying on intake alone is weaker because the function accepts report mappings.
Broad report schema validation is larger than needed for this artifact SHA
gate.

## 6. Implementation Plan

- Import `is_sha256_hexdigest()` into the review tool.
- Require `summary_sha256` and `events_sha256` to pass the helper before
  template generation.
- Add a regression test for malformed summary and event artifact hashes.

## 7. Test Plan

Run focused WING review/evidence/no-WING tests and the standard repository
pytest command.

## 8. Risks And Rollback

Risk is limited to rejecting malformed review data. Rollback is reverting the
helper use and regression test if a real report format unexpectedly emits
non-hex digests, which would require operator review before live use.
