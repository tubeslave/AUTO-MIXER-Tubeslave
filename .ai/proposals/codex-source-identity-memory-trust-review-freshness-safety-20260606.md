# Proposal: source-identity-memory-trust-review-freshness-safety-20260606

## 1. Short thesis

Keep source identity memory trust review fail-closed by validating the safety of
freshness evidence and memory-index status before exposing operator-review
candidates.

## 2. What I understood

The current dirty worktree is advancing source identity calibration from
append-only memory writer to read-only index and trust review. This is a safe
memory-learning path only if review candidates cannot be produced from forged
freshness evidence or inconsistent index envelopes.

## 3. Proposed solution

Add two narrow validations to
`backend/source_identity_calibration_memory_trust_review.py`:

- `status=fresh` context must also prove `read_only=true`,
  `live_write=false`, and `queries_console=false`.
- A memory index that is `memory_index_not_ready` while carrying records, or
  `append_only_memory_index_available` without records, is invalid. A
  `memory_index_replacement_required` status remains replacement-required even
  if an injected summary omits invalid IDs.

## 4. Likely files to touch

- `backend/source_identity_calibration_memory_trust_review.py`
- `tests/test_source_identity_calibration_memory_trust_review.py`
- `.ai/memory/project.md`
- `Docs/adr/source-identity-memory-trust-review-freshness-safety-20260606.md`

## 5. Alternatives considered

Leaving freshness as a status-only string was rejected because a forged
`fresh` value could bypass the review gate. Moving this into Product Layer only
was rejected because the trust-review builder is the reusable contract boundary.

## 6. Risks

Older test fixtures that mark context as fresh without safety flags may now be
reported as invalid. That is acceptable for trust-review candidates because
candidate creation should require explicit read-only/no-live/no-query evidence.

## 7. Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_source_identity_calibration_memory_trust_review.py -q`
- Focused source identity/Product Layer/registry/capability tests
- Standard backend pytest gate
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## 8. Where the other agent may disagree

Another agent may prefer treating unsafe freshness as simply not-ready instead
of replacement-required. I recommend replacement-required because the payload
claims freshness while carrying unsafe authority flags, which is malformed
evidence rather than merely missing evidence.
