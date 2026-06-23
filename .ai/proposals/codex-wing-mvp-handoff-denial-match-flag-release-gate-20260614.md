# Codex Proposal: WING MVP Handoff Denial Match Flag Release Gate

## Short thesis

The no-WING release gate should reject stale handoff packet match-flag
declarations instead of merely recomputing and rendering the correct values.

## What I understood

The current selected blocker remains WING read-only connection proof. The
handoff packet now carries declared denial match flags, but a stale packet could
lie about those flags unless `no_wing_release_check.py` compares declared
values with recomputed truth.

## Proposed solution

In `tools/no_wing_release_check.py`, compute a map of declared
`<field>_matches_plan` values for all backend-port prestart required-false
fields, compare it to the recomputed map, and emit
`wing_readonly_preflight_handoff_packet_<field>_match_flag_mismatch` violations
on drift. Preserve both declared and computed values in the summary artifact
and markdown.

## Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `.ai/briefs/wing-mvp-handoff-denial-match-flag-release-gate-20260614.md`
- `Docs/adr/wing-mvp-handoff-denial-match-flag-release-gate-20260614.md`
- `Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_DENIAL_MATCH_FLAG_RELEASE_GATE_20260614.md`

## Alternatives considered

- Trust the packet builder tests alone. Rejected because release artifacts can
  be stale or edited.
- Only render computed values in the summary. Rejected because it does not
  prove the packet's declared fields are current.

## Risks

- Older handoff packets without declared match flags will now fail closed. This
  is acceptable because the current first command requires those fields before
  supervised proof planning.

## Test plan

- Focused no-WING release check tests.
- Focused WING handoff/no-WING/Project Director/rails tests.
- Compile, memory lint, `git diff --check`, and the full test suite.

## Where the other agent may disagree

Kimi may prefer a versioned schema migration for old handoff packets. I would
reject that for this gate because stale packets must not advance readiness.
