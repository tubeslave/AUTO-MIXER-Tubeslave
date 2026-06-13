# Proposal: WING MVP Full Autonomous Disallowed By Packet Next Action

## Short thesis

Require `full_autonomous_ready: false` in the no-WING handoff summary before
any supervised fader/gain proof planning.

## What I understood

The project is moving toward full autonomous mixing, but the current WING MVP
state is still a supervised proof path blocked on real read-only evidence. An
offline handoff packet must not be read as full autonomous readiness.

## Proposed solution

- Add `full_autonomous_ready: false` to the first WING MVP blocker next action
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
- Start real WING preflight. Rejected because this turn must remain no-WING and
  offline.

## Risks

- The blocker row is long and easy to update incompletely.
- Future cycles may confuse the long-term full-autonomous objective with the
  current supervised WING MVP boundary.

## Test plan

- Run focused Project Director and WING MVP rail tests.
- Run Project Director and direction guard command checks.
- Run `git diff --check`, memory lint, and full backend pytest.

## Where the other agent may disagree

Kimi may argue the field is already covered by `live_authority_granted: false`.
Keeping `full_autonomous_ready: false` explicit is useful because the long-term
goal names full autonomy, while the current blocker is only read-only proof.
