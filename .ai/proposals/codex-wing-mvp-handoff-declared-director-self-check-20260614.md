# Codex Proposal: WING MVP Handoff Declared Director Self-Check

## Short thesis

Make the handoff packet self-declare Project Director digest verdicts before the
no-WING gate recomputes them.

## What I understood

The active blocker remains `WING read-only connection proof`. Recent work made
the no-WING handoff summary and `Current First Command` require the full
expected/actual Project Director digest field set. The packet builder still
emitted only the shorter digest subset, leaving the packet less inspectable than
the summary.

## Proposed solution

Add declared `expected_project_director_selected_action_*`,
`actual_project_director_selected_action_sha256`, and boolean match fields to
`tools/build_wing_readonly_preflight_handoff_packet.py`. Render the fields in
the packet Markdown and make `tools/no_wing_release_check.py` compare the
declared fields against its recomputed Project Director directive.

## Likely files to touch

- `tools/build_wing_readonly_preflight_handoff_packet.py`
- `tools/no_wing_release_check.py`
- `tests/test_wing_readonly_preflight_handoff_packet.py`
- `tests/test_no_wing_release_check.py`
- Council brief, ADR, report, and index files

## Alternatives considered

Leaving the fields as no-WING-derived-only keeps validation correct after the
gate runs but gives the operator a weaker packet before validation. Adding a
new structured digest object was deferred to avoid a broad schema migration.

## Risks

The main risk is adding redundant fields. The redundancy is intentional: the
packet declares its view, and the no-WING gate recomputes the authoritative
view and fails closed on drift.

## Test plan

Run focused handoff packet, no-WING, Project Director, and WING rail tests. Run
the packet builder, Project Director, Direction Guard, no-WING quick
fail-closed, memory lint, `git diff --check`, and the standard full pytest
command.

## Where the other agent may disagree

Kimi may prefer not to duplicate no-WING-derived fields in the packet. The
counterpoint is operator inspectability: the packet is a handoff artifact, so it
should declare the self-check fields that the gate later verifies.
