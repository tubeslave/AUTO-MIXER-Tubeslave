# Codex Proposal: WING MVP Handoff Proof Plan Artifact Root Summary Fields

## Short thesis

Extend the handoff packet required proof-plan summary field contract so accepted-candidate artifact-root diagnostics are required before supervised proof planning.

## What I understood

The proof-plan artifact-root trace is now generated and validated downstream. The next safe offline tightening is to make the no-WING handoff packet advertise those fields as required, preserving the operator-visible contract before any physical proof work.

## Proposed solution

- Add the accepted-candidate artifact-root proof-plan fields to `WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_REQUIRED_PROOF_PLAN_SUMMARY_FIELDS`.
- Keep the field list centralized in `backend/wing_readonly_preflight_contract.py`.
- Add focused assertions in handoff/no-WING/rails/director tests so future contract drift fails closed.
- Update current WING blocker docs and reports to describe artifact-root fields as handoff-required.

## Likely files to touch

- `backend/wing_readonly_preflight_contract.py`
- `tests/test_wing_readonly_preflight_handoff_packet.py`
- `tests/test_no_wing_release_check.py`
- `tests/test_wing_mvp_rails.py`
- `tests/test_project_director_agent.py`
- Project memory docs under `.ai/` and `Docs/`

## Alternatives considered

- Leave artifact-root fields as proof-plan-only diagnostics. Rejected because the handoff packet is the operator contract before supervised proof and should declare all required summary fields.
- Duplicate the root field list inside handoff generation. Rejected because the existing shared constant already defines the contract and avoids drift.

## Risks

- Older generated handoff/no-WING summaries will fail contract checks until regenerated.
- The required field list is longer, so fixtures need explicit coverage.

## Test plan

- `PYTHONPATH=backend python -m pytest -q tests/test_wing_readonly_preflight_handoff_packet.py tests/test_no_wing_release_check.py`
- `PYTHONPATH=backend python -m pytest -q tests/test_wing_mvp_rails.py tests/test_project_director_agent.py`
- Compile changed modules and run repo rails before handoff.

## Where the other agent may disagree

Kimi may prefer a separate artifact-root field contract constant. I kept the fields in the existing required proof-plan summary tuple because the handoff/no-WING boundary already consumes that shared contract.
