# Codex Proposal: WING MVP Review Decision Artifact SHA Format Gate

## Short thesis

Manual review decisions should reject malformed artifact digests, not only
digests that mismatch a candidate.

## What I understood

The active blocker remains `WING read-only connection proof`. The evidence
intake now rejects malformed plan and prestart summary digests. The operator
review-decision ledger still needed the same explicit digest-format boundary
for the manually edited `summary_sha256` and `events_sha256` fields.

## Proposed solution

Reuse `is_sha256_hexdigest()` inside
`backend/wing_readonly_preflight_review_decision.py` and emit explicit
violations for malformed summary/event-log artifact digests. Add a focused
ledger regression test and refresh the review-decision test fixture to include
the required runtime `artifact_root`.

## Likely files to touch

- `backend/wing_readonly_preflight_review_decision.py`
- `tests/test_wing_readonly_preflight_review_decision.py`
- Council/report docs for this slice.

## Alternatives considered

- Rely only on mismatch checks. Rejected because a malformed value should be
  diagnosed as malformed, not just stale.
- Add a local regex. Rejected because the shared WING preflight contract already
  owns the digest rule.

## Risks

Old hand-authored review decisions with placeholder hashes will now fail with a
specific invalid-digest violation. That is intended for readiness gating.

## Test plan

- Focused review-decision ledger tests.
- Full no-WING release-check module.
- Project Director rails, py_compile, memory lint, and `git diff --check`.

## Where the other agent may disagree

Kimi may prefer accepting uppercase SHA-256 strings. Current generated digests
come from `hashlib.hexdigest()`, so strict lowercase keeps review decisions
consistent with evidence and handoff digest gates.
