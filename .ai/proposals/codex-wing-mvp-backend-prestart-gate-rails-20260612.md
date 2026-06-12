# Codex Proposal: WING MVP Backend Prestart Gate Rails

## Short thesis

The top-level MVP contract should require the same structured backend-port
prestart gate already enforced by the handoff validator.

## What I understood

The read-only runbook and no-WING handoff validator now require
`backend_port_prestart_gate` to match `expected_backend_port_prestart_gate`.
The MVP source of truth and autonomous cycle prompt still named only the final
`supervised_prestart_gate`, which can let future cycles skip the first gate in
planning language.

## Proposed solution

- Update `Docs/MVP_WING_SUPERVISED.md` stop-the-line and next-action sections.
- Update `tools/codex_autoloop/codex_cycle_prompt.md`.
- Add the structured first-gate report to `Docs/INDEX.md`.
- Extend `tests/test_wing_mvp_rails.py`.

## Likely files to touch

- `Docs/MVP_WING_SUPERVISED.md`
- `tools/codex_autoloop/codex_cycle_prompt.md`
- `Docs/INDEX.md`
- `tests/test_wing_mvp_rails.py`

## Alternatives considered

- Leave the requirement only in the runbook. Rejected because goal-mode cycles
  consult the MVP doc and prompt before selecting work.
- Add another runtime gate. Rejected for this slice because runtime validation
  already exists in no-WING handoff artifact validation.

## Risks

This is a documentation/rails sync and does not create real WING proof. The
wording must preserve that blockers stay open until real artifacts are reviewed.

## Test plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_wing_mvp_rails.py`
- Direction guard smoke check.
- Full repository pytest before commit.

## Where the other agent may disagree

Kimi may prefer waiting until after hardware proof to update the top-level MVP
contract. The counterpoint is that this is a pre-hardware ordering safety rule,
so it belongs in the rails before hardware work.
