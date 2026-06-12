# Codex Proposal: wing-mvp-blocker-backend-prestart-next-action-20260612

## Short thesis

The live blocker table must carry the same backend-port prestart gate as the MVP
doc, because Project Director and progress reports choose the next action from
that table.

## What I understood

The first WING MVP blocker remains `WING read-only connection proof` with status
`tooling_ready_waits_real_wing`. Offline/docs work may only harden the path to
real WING proof and must not imply live readiness.

## Proposed solution

- Update `.ai/project-layer/wing_mvp_blockers.md` so the first blocker next
  action requires `backend_port_prestart_gate` to match
  `expected_backend_port_prestart_gate`.
- Add the same gate to the supervised fader and gain rows.
- Regenerate `Docs/reports/MVP_PROGRESS_LATEST.md`.
- Extend rails tests so the generated progress report cannot omit this gate.

## Likely files to touch

- `.ai/project-layer/wing_mvp_blockers.md`
- `Docs/reports/MVP_PROGRESS_LATEST.md`
- `tests/test_wing_mvp_rails.py`
- `Docs/adr/wing-mvp-blocker-backend-prestart-next-action-20260612.md`
- `Docs/reports/AUTOMIXER_WING_MVP_BLOCKER_BACKEND_PRESTART_NEXT_ACTION_20260612.md`

## Alternatives considered

- Leave the blocker table unchanged because the MVP doc is stricter. Rejected:
  the blocker table is the live work selector.
- Patch only the generated report. Rejected: the next regeneration would lose
  the requirement.

## Risks

The wording could be overread as proof completion. Mitigation: keep blocker
statuses unchanged and explicitly say the gate does not grant live authority.

## Test plan

- Focused WING MVP rails tests.
- Progress regeneration command.
- Direction guard.
- Diff check.

## Where the other agent may disagree

Kimi may argue that the current table already points to the runbook, which now
contains the gate. I would still keep the gate in the table because this is the
operator-facing next-action summary used by automation.
