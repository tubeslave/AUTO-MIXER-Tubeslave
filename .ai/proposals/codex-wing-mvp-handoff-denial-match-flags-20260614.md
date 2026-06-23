# Codex Proposal: WING MVP Handoff Denial Match Flags

## Short thesis

Add explicit plan-match verdicts for handoff denial fields next to the existing
denial metadata so operators and automation can distinguish expected false
values from drifted prestart evidence.

## What I understood

The selected WING read-only blocker is still tooling-ready but waiting for real
hardware. While waiting, only no-live tooling work is valid. The current first
command now requires the handoff summary to show match flags for safe-to-stop,
live-authority, execution, readiness-closure, and backend-prestart authority
denials.

## Proposed solution

In `tools/build_wing_readonly_preflight_handoff_packet.py`, derive five
`*_matches_plan` booleans from the existing prestart backend-check detail and
packet denial fields. Render them in markdown alongside the source denial
fields. Add tests for the expected clean path and for a drifted prestart
summary where the match flags must become false.

## Likely files to touch

- `tools/build_wing_readonly_preflight_handoff_packet.py`
- `tests/test_wing_readonly_preflight_handoff_packet.py`
- `.ai/briefs/wing-mvp-handoff-denial-match-flags-20260614.md`
- `Docs/adr/wing-mvp-handoff-denial-match-flags-20260614.md`
- `Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_DENIAL_MATCH_FLAGS_20260614.md`

## Alternatives considered

- Only update Project Director wording. Rejected because the packet would still
  lack machine-readable verdicts.
- Always set the new flags to true. Rejected because drifted prestart metadata
  would be masked.

## Risks

- The `live_authority_granted_matches_plan` field could be confused with the
  packet's top-level live-authority constant. Mitigation: compute it from the
  actual prestart backend-check detail when an existing summary is supplied.

## Test plan

- Run focused handoff packet tests.
- Run focused no-WING/Project Director/WING rails tests.
- Run compile, memory lint, and `git diff --check`.

## Where the other agent may disagree

Kimi may prefer moving all plan-match derivation into a shared helper. I would
keep this patch local because only the handoff packet currently needs this
specific operator-facing projection.
