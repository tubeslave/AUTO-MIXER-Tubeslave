# Proposal: WING Read-Only Handoff MVP Blocker Anchor

Date: 2026-06-12

## Short thesis

Make the offline WING handoff packet carry a current first-open blocker anchor
and make no-WING validation reject stale anchors.

## What I understood

Goal-mode work must stay on the first supervised WING MVP blocker. Handoff
packets may prepare operator evidence, but they cannot contact WING, grant live
authority, or close readiness from offline proof.

## Proposed solution

- Parse `.ai/project-layer/wing_mvp_blockers.md` for the first non-closed row.
- Emit that row as `mvp_blocker_anchor` inside the handoff packet.
- Include blocker table path, blocker name, status, next action, evidence file
  count, evidence files, and `packet_does_not_close_blocker: true`.
- Make `validate_wing_readonly_preflight_handoff_packet_artifact_snapshot`
  compare the packet anchor with the current table-derived anchor.
- Render the anchor in handoff Markdown and no-WING summary Markdown.

## Likely files to touch

- `tools/build_wing_readonly_preflight_handoff_packet.py`
- `tools/no_wing_release_check.py`
- `tests/test_wing_readonly_preflight_handoff_packet.py`
- `tests/test_no_wing_release_check.py`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `Docs/INDEX.md`

## Alternatives considered

- Rely on `blockers_not_closed_by_packet` only. This proves the packet does not
  close rows but does not prove it was generated for the current first row.
- Put the anchor only in Project Director output. That helps Codex cycles, but
  not the offline handoff artifact that operators review.

## Risks

- If the blocker table format changes, the parser can fail or produce an empty
  anchor.
- If old handoff packets are reused, validation will now fail and require a
  fresh packet.

## Test plan

- Handoff packet unit tests.
- no-WING release-check unit tests.
- WING MVP rails tests.
- Direction guard and Project Director focused checks.
- Full pytest before commit.

## Where the other agent may disagree

Kimi may prefer a shared markdown-table parser from Project Director. I would
keep this local because the no-WING gate must remain independently fail-closed
and should not depend on a planning-only module.
