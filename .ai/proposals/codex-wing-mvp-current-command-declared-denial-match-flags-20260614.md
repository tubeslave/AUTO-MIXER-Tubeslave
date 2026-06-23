# Codex Proposal: WING MVP Current Command Declared Denial Match Flags

## Short thesis

Project Director should only count the WING read-only handoff current when the
operator-visible command requires both computed denial verdicts and declared
denial verdict self-checks.

## What I understood

The selected blocker remains WING read-only connection proof. The handoff packet
and no-WING release check now validate declared backend-port denial match flags,
but the Current First Command baseline can still pass without mentioning those
declared fields.

## Proposed solution

Add five `declared_<field>_matches_plan: true` markers to the current-command
baseline marker list, autoloop prompt, README, blocker command text, and related
rails tests. Keep the fields informational and fail-closed; they do not grant
live authority or close readiness.

## Likely files to touch

- `tools/codex_autoloop/update_mvp_progress.py`
- `tools/codex_autoloop/codex_cycle_prompt.md`
- `tools/codex_autoloop/README.md`
- `.ai/project-layer/wing_mvp_blockers.md`
- `tests/test_project_director_agent.py`
- `tests/test_wing_mvp_rails.py`
- `Docs/adr/wing-mvp-current-command-declared-denial-match-flags-20260614.md`
- `Docs/reports/AUTOMIXER_WING_MVP_CURRENT_COMMAND_DECLARED_DENIAL_MATCH_FLAGS_20260614.md`

## Alternatives considered

- Leave declared fields only in no-WING release artifacts. Rejected because
  Project Director could treat a command as current without operator-visible
  declared self-check markers.
- Add a new schema version. Rejected for this narrow rail sync; the existing
  marker contract is enough.

## Risks

Current First Command baseline coverage becomes stricter. That is acceptable
because older command text without declared self-checks should not steer
supervised proof planning.

## Test plan

- Focused Project Director and WING rails tests.
- Focused WING handoff/no-WING/Project Director/rails tests.
- Compile, memory lint, `git diff --check`, and the full test suite.

## Where the other agent may disagree

Kimi may prefer folding declared and computed checks into a structured schema
instead of marker text. I would keep this slice marker-based because it follows
the existing Project Director baseline mechanism with minimal surface area.
