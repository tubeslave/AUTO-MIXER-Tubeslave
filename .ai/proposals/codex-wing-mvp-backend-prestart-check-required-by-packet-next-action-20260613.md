# Proposal: WING MVP Backend Prestart Check Required By Packet Next Action

## Short thesis

Require `backend_port_prestart_check_required: true` in the no-WING handoff
summary before any supervised fader/gain proof planning.

## What I understood

The project is still blocked on real WING read-only proof. The offline handoff
packet can guide a future operator sequence, but the backend listener prestart
check must be explicit and non-optional before any physical backend/WING
session.

## Proposed solution

- Add `backend_port_prestart_check_required: true` to the first WING MVP
  blocker next action and Current First Command.
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
- Run the prestart check against local listeners now. Rejected because this turn
  should not start, stop, or reconfigure backend/WING sessions.

## Risks

- The blocker row is long and easy to update incompletely.
- Future cycles may treat the detailed backend prestart gate as sufficient and
  miss the top-level required flag.

## Test plan

- Run focused Project Director and WING MVP rail tests.
- Run Project Director and direction guard command checks.
- Run `git diff --check`, memory lint, and full backend pytest.

## Where the other agent may disagree

Kimi may argue the structured `backend_port_prestart_gate` already implies this
requirement. Keeping the top-level field explicit makes packet validation easier
for operator handoff review.
