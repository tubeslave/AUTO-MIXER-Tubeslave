# Codex Proposal: WING MVP Handoff Director Action Digest

## Short thesis

Make the handoff packet carry and validate the Project Director selected-action
anchor so operator review can prove it follows the same first-blocker action as
Goal-mode direction.

## What I understood

The WING MVP still waits on real read-only connection proof. The current packet
already proves the first open blocker and operator sequence, while Project
Director separately proves the `Current First Command` selected action digest.
Those two pieces should be joined in the handoff packet before hardware work.

## Proposed solution

Add a lightweight no-WING helper that mirrors Project Director selected-action
selection using the blocker table and current-first-command parser. Emit the
resulting anchor, source, SHA256, and baseline coverage fields from
`tools/build_wing_readonly_preflight_handoff_packet.py`. Validate the anchor,
source, digest, digest availability flag, and coverage fields in
`tools/no_wing_release_check.py`, then expose them in JSON and Markdown
summaries.

## Likely files to touch

- `tools/build_wing_readonly_preflight_handoff_packet.py`
- `tools/no_wing_release_check.py`
- `tests/test_wing_readonly_preflight_handoff_packet.py`
- `tests/test_no_wing_release_check.py`
- Council brief, ADR, report, and index files

## Alternatives considered

Keep the Project Director digest only in Project Director output. This is less
auditable because the operator packet can drift from the selected action without
a packet-level failure. Importing the full Project Director agent was rejected
because no-WING packet validation should stay lightweight and avoid direction
guard side effects.

## Risks

The main risk is duplicated selection logic drifting from Project Director. The
helper deliberately reuses `update_mvp_progress` parsing and selected-action
logic, and tests compare the generated packet to the helper.

## Test plan

Run focused handoff, no-WING schema, and Project Director tests. Then run
Project Director, direction guard, no-WING quick fail-closed, memory lint,
`git diff --check`, and the standard full pytest command before commit.

## Where the other agent may disagree

Kimi may prefer importing Project Director directly. The counterpoint is that
the full Project Director also collects guard evidence; a small helper that
reuses the shared selected-action parser is easier to keep side-effect free in
no-WING validation.
