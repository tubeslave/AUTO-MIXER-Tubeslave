# Codex Proposal: WING MVP Prestart Summary Policy Rails

## Short thesis

Expose the no-WING prestart summary connection policy in the same rails that
already require the summary type, status, and no-contact fields.

## What I understood

The WING read-only proof remains the first blocker. The prestart summary
contract is stricter than the current operator-facing text: runtime validation
requires `backend_port_prestart_summary_connection_policy:
no_hardware_no_network_no_live_writes`, but the current command and autoloop
rails only surfaced type/status/no-contact markers.

## Proposed solution

Add the connection-policy marker to `Docs/MVP_WING_SUPERVISED.md`,
`.ai/project-layer/wing_mvp_blockers.md`, `tools/codex_autoloop/README.md`, and
`tools/codex_autoloop/codex_cycle_prompt.md`. Extend Project Director and WING
rail tests so the marker stays visible in selected action text and generated
progress.

## Likely files to touch

- `Docs/MVP_WING_SUPERVISED.md`
- `.ai/project-layer/wing_mvp_blockers.md`
- `tools/codex_autoloop/README.md`
- `tools/codex_autoloop/codex_cycle_prompt.md`
- `tests/test_project_director_agent.py`
- `tests/test_wing_mvp_rails.py`
- `Docs/reports/MVP_PROGRESS_LATEST.md`
- `Docs/reports/AUTOMIXER_WING_MVP_PRESTART_SUMMARY_POLICY_RAILS_20260614.md`
- `Docs/INDEX.md`

## Alternatives considered

Keep the policy only in runtime validation. Rejected because Goal-mode work
starts from the blocker command and autoloop rails before a real artifact
exists.

Add a new runtime validator. Rejected because the runtime policy check already
exists; this slice closes the operator-facing visibility gap.

## Risks

The risk is duplicated rail text. Focused tests mitigate that by checking the
marker in Project Director, progress, and autoloop surfaces.

## Test plan

Run focused rail tests, regenerate the latest MVP report, run Project Director
and direction guard JSON output, run `git diff --check`, run memory lint, and
run the standard full pytest command.

## Where the other agent may disagree

Kimi may prefer generating all these rails directly from contract constants. I
agree that would reduce duplication, but this slice is intentionally a smaller
visibility fix for the first open WING blocker.
