# Codex Proposal: WING MVP Handoff Anchor Digest Review Step Index

## Short thesis

Make the anchor digest review step index a first-class no-WING contract so a
reordered handoff packet cannot pass only because the total operator sequence
count and final supervised gate still match.

## What I understood

The active Goal-mode blocker is WING read-only connection proof. The handoff
packet is offline and non-authoritative, but it is the operator checklist before
real WING read-only work. It already validates the operator step count and final
supervised prestart gate. The anchor digest review step must also be pinned.

## Proposed solution

Add `WING_READONLY_PREFLIGHT_MVP_BLOCKER_ANCHOR_DIGEST_REVIEW_STEP_INDEX` to the
shared contract. Have `tools/no_wing_release_check.py` compute and report:

- `expected_operator_sequence_mvp_blocker_anchor_digest_review_step_index`
- `operator_sequence_mvp_blocker_anchor_digest_review_step_index_matches_contract`

Fail closed with a specific violation when the review step exists at the wrong
index.

## Likely files to touch

- `backend/wing_readonly_preflight_contract.py`
- `tools/no_wing_release_check.py`
- `tools/build_wing_readonly_preflight_handoff_packet.py`
- `tests/test_no_wing_release_check.py`
- WING MVP docs, autoloop docs, rails tests, ADR, and report files

## Alternatives considered

Only documenting the expected index would preserve the existing weakness.
Checking only total sequence count and the final supervised gate still allows a
swap of the anchor review step with another existing step.

## Risks

The main risk is drift between generated handoff packets, no-WING summaries, and
Project Director text. Focused rails tests should lock the wording and fields
together.

## Test plan

Run the focused no-WING validator tests, Project Director rails tests, WING MVP
rails tests, no-WING quick summary generation, direction guard, memory lint, and
the standard full pytest command.

## Where the other agent may disagree

Kimi may argue that the review step order is operator-process detail rather than
runtime safety. The counterpoint is that this handoff is the last no-WING packet
before real hardware work, so deterministic step order is a useful fail-closed
operator safety boundary.
