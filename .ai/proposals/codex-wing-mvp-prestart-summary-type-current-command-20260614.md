# Codex Proposal: WING MVP Prestart Summary Type Current Command

## Short thesis

Expose the no-WING prestart summary type in the current blocker command and
autoloop rails so accepted-candidate prestart evidence cannot be type-ambiguous.

## What I understood

The supervised WING MVP remains blocked on read-only connection proof. The next
manual route already requires a passed no-WING prestart summary by path, SHA256,
status, and no-contact fields. A separate shared contract expects
`backend_port_prestart_summary_type: no_wing_release_check`, but the current
operator-facing command and some autoloop checklists did not make that type
visible.

## Proposed solution

Add `backend_port_prestart_summary_type: no_wing_release_check` to the
`## Current First Command` text in `.ai/project-layer/wing_mvp_blockers.md`,
the generated progress output, and the autoloop README/prompt rails. Extend
focused tests so Project Director, the generated latest progress report, and
the autoloop docs all fail if the type marker is dropped.

## Likely files to touch

- `.ai/project-layer/wing_mvp_blockers.md`
- `tools/codex_autoloop/README.md`
- `tools/codex_autoloop/codex_cycle_prompt.md`
- `tests/test_project_director_agent.py`
- `tests/test_wing_mvp_rails.py`
- `Docs/reports/MVP_PROGRESS_LATEST.md`
- `Docs/reports/AUTOMIXER_WING_MVP_PRESTART_SUMMARY_TYPE_CURRENT_COMMAND_20260614.md`
- `Docs/INDEX.md`

## Alternatives considered

Keep the type requirement only in Python constants and validation code. Rejected
because Goal-mode routing reads the blocker command and autoloop docs before a
real accepted-candidate artifact exists.

Add a new runtime validator. Rejected for this slice because runtime validation
already has the type contract; the gap is operator-facing and director-facing
visibility.

## Risks

The risk is documentation duplication across the blocker command, generated
progress, and autoloop docs. The mitigation is focused rail tests that assert
the marker in Project Director output, the progress report, and both autoloop
surfaces.

## Test plan

Run focused rail tests, regenerate the latest MVP report, run Project Director
JSON output, run the direction guard, run `git diff --check`, run memory lint,
and then run the standard full pytest command.

## Where the other agent may disagree

Kimi may prefer deriving the Current First Command entirely from structured
contract constants. That would reduce duplication, but it is a broader change
than the current blocker-first visibility fix.
