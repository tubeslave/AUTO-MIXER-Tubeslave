# Codex Proposal: WING Handoff Prestart Gate Final-Step Validation

## Short thesis

The supervised prestart gate should be the final operator sequence step in a
WING read-only handoff packet, so no later instruction can weaken the proof
transition.

## What I understood

Same-step marker validation prevents split wording, but it does not prevent a
packet from appending a stale proof instruction after the correct gated step.
Operators follow sequence order, so the last step must be the gated transition.

## Proposed solution

- Track `operator_sequence_supervised_prestart_gate_step_index`.
- Fail release validation when the gated step is present but not final.
- Surface `operator_sequence_supervised_prestart_gate_is_final` in summaries.
- Add a regression with a correct gated step followed by a stale proof step.

## Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`

## Alternatives considered

- Keep same-step matching only. Rejected because later conflicting guidance can
  still mislead operators.
- Remove all later proof mentions by text scan. Too broad for this slice.

## Risks

Older packets with extra instructions after the gated transition now fail
release validation. That is intended fail-closed behavior.

## Test plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_no_wing_release_check.py -k "handoff_packet"`
- Affected no-WING/WING tests.
- Full pytest suite.
- `git diff --check`.

## Where the other agent may disagree

Kimi may prefer a structured packet field over final-step inference. That can
come later; this slice closes the current ordering ambiguity without schema
churn.
