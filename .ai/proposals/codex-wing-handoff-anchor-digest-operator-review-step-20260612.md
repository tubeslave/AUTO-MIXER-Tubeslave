# Proposal: WING Handoff Anchor Digest Operator Review Step

Date: 2026-06-12

## Short thesis

Make the digest-bound blocker anchor visible inside the handoff packet operator
sequence, not only in summary fields.

## What I understood

The project is still blocked on real WING read-only proof. Offline work should
make the next physical run safer and more reviewable without contacting WING or
expanding live writes.

## Proposed solution

- Add a handoff packet operator-sequence step that asks reviewers to confirm
  `mvp_blocker_anchor` and `mvp_blocker_anchor_sha256` match the current first
  open blocker.
- Make no-WING validation fail closed if the operator-sequence step is absent.
- Expose the step status and text in release summaries.
- Update runbook, tests, ADR, report, and index.

## Likely files to touch

- `tools/build_wing_readonly_preflight_handoff_packet.py`
- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `tests/test_wing_mvp_rails.py`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `Docs/INDEX.md`

## Alternatives considered

- Keep the digest requirement only as summary fields. That validates the data,
  but it does not make the review action explicit in the operator sequence.
- Add prose only to the runbook. That would not fail closed for stale generated
  packets.

## Risks

- The supervised prestart gate moves from operator-sequence index 8 to 9.
- Existing handoff packets generated before this change will fail validation
  until regenerated.

## Test plan

- Focused no-WING, WING MVP rails, Project Director, and direction guard tests.
- Build and validate a temporary handoff packet.
- Direction guard and Project Director JSON smoke checks.
- Full pytest before commit.

## Where the other agent may disagree

Kimi may argue that the digest summary is already enough. I would keep this
operator-sequence step because reviewers follow the sequence, and stale packet
failure should happen before supervised proof planning.
