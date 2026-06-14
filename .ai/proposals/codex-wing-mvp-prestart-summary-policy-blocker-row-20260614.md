# Codex Proposal: WING MVP Prestart Summary Policy Blocker Row

## Short thesis

Make the first WING blocker table row carry the same no-WING prestart summary
connection-policy marker as the Current First Command.

## What I understood

The active blocker is still the WING read-only connection proof. Project
Director intentionally combines the Current First Command with the blocker
table baseline, so the row itself should not omit the no-hardware,
no-network, no-live-write policy that accepted-candidate prestart traces must
prove.

## Proposed solution

Add `backend_port_prestart_summary_connection_policy:
no_hardware_no_network_no_live_writes` to the first row's next-action text in
`.ai/project-layer/wing_mvp_blockers.md`. Extend the WING rail test that parses
the blocker table so it fails if the marker is removed from the baseline row.

## Likely files to touch

- `.ai/project-layer/wing_mvp_blockers.md`
- `tests/test_wing_mvp_rails.py`
- `Docs/reports/MVP_PROGRESS_LATEST.md`
- `Docs/reports/AUTOMIXER_WING_MVP_PRESTART_SUMMARY_POLICY_BLOCKER_ROW_20260614.md`
- `Docs/INDEX.md`

## Alternatives considered

Rely on the Current First Command only. Rejected because the Project Director
selected action includes the blocker table baseline, and that baseline should
not be weaker than the current command.

Generate the table row from shared constants. Rejected for this slice because
it is a broader refactor than the current blocker-first rail alignment.

## Risks

The main risk is more duplicated text in an already long blocker row. The
focused parser test keeps the safety marker visible until a generated source
replaces the duplicated documentation.

## Test plan

Run focused WING rail tests, regenerate latest MVP progress, run Project
Director and direction guard JSON output, run `git diff --check`, run memory
lint, and run the standard full pytest command.

## Where the other agent may disagree

Kimi may prefer replacing the duplicated markdown row with generated text from
the runtime contract. I agree that is a better long-term shape, but this
change keeps the current blocker baseline aligned without widening scope.
