# Codex Proposal: WING MVP Handoff Declared Anchor Digest Review Index

## Short thesis

Validate the packet-declared anchor digest review step index so handoff metadata
cannot drift from the actual operator sequence before real WING proof work.

## What I understood

The active blocker remains WING read-only connection proof. Recent work pinned
the actual anchor digest review step at index 8. The packet now includes a
declared `mvp_blocker_anchor_digest_review_step_index`, but no-WING validation
should also prove that declared field matches the contract and the operator
sequence.

## Proposed solution

Read the declared packet index in `tools/no_wing_release_check.py`, compare it
with `WING_READONLY_PREFLIGHT_MVP_BLOCKER_ANCHOR_DIGEST_REVIEW_STEP_INDEX`, and
compare it with the extracted operator-sequence index. Expose both match flags
in JSON/Markdown summaries.

## Likely files to touch

- `tools/no_wing_release_check.py`
- `tools/build_wing_readonly_preflight_handoff_packet.py`
- `tests/test_no_wing_release_check.py`
- WING MVP docs, autoloop docs, rails tests, ADR, and report files

## Alternatives considered

Relying only on the extracted operator-sequence index would catch the real order
but leave misleading packet metadata unvalidated. Removing the declared field
would reduce metadata, but the field is useful for operator-facing handoff
review when validated.

## Risks

The main risk is documentation drift between declared index and operator
sequence index wording. Focused Project Director and WING rails tests should
catch that drift.

## Test plan

Run focused no-WING validator tests, Project Director rails tests, WING MVP
rails tests, no-WING quick summary generation, direction guard, memory lint,
`git diff --check`, and the full standard pytest command.

## Where the other agent may disagree

Kimi may argue this is redundant because the actual operator sequence is already
validated. The counterpoint is that operator-facing handoff metadata must not be
allowed to contradict the actual checklist before a hardware session.
