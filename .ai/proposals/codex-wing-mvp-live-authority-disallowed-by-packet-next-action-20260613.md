# Proposal: WING MVP Live Authority Disallowed By Packet Next Action

## Short thesis

Require `live_authority_granted: false` in the no-WING handoff summary before
any supervised fader/gain proof planning.

## What I understood

The project is still waiting on real WING read-only proof. Offline handoff
packets may guide the operator sequence, but must not imply permission to
control the console.

## Proposed solution

- Add `live_authority_granted: false` to the first WING MVP blocker next action
  and Current First Command.
- Add the same field to MVP stop rules, read-only runbook checks, autoloop
  README, and cycle prompt.
- Add regression assertions in Project Director and WING MVP rail tests.
- Record the decision in ADR and report docs.

## Likely files to touch

- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/MVP_WING_SUPERVISED.md`
- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `tools/codex_autoloop/README.md`
- `tools/codex_autoloop/codex_cycle_prompt.md`
- `tests/test_project_director_agent.py`
- `tests/test_wing_mvp_rails.py`
- `Docs/INDEX.md`
- council brief, proposal, ADR, and report files

## Alternatives considered

- Change the packet builder. Rejected because the packet already emits the
  field.
- Jump to a real read-only preflight. Rejected because this turn must not touch
  WING hardware or backend/WING sessions.

## Risks

- The blocker row is long and easy to update incompletely.
- Autoloop docs can drift from the blocker table if tests do not cover each
  surface.

## Test plan

- Run focused Project Director and WING MVP rail tests.
- Run Project Director and direction guard command checks.
- Run `git diff --check`, memory lint, and full backend pytest.

## Where the other agent may disagree

Kimi may prefer combining `live_authority_granted: false` with
`full_autonomous_ready: false`. Keeping this change to one packet field reduces
review risk and preserves the existing incremental WING rail pattern.
