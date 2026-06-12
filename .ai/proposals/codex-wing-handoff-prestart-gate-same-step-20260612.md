# Codex Proposal: WING Handoff Prestart Gate Same-Step Validation

## Short thesis

The handoff packet validator should only accept the supervised proof transition
when the accepted decision, valid prestart trace, and proof target markers are
present in one operator sequence step.

## What I understood

Searching the full operator sequence text is too loose: a stale final transition
can still say "after accepted_decision_ready, move to proof" while another line
mentions prestart trace. That is misleading at the operator handoff boundary.

## Proposed solution

- Compute `operator_sequence_supervised_prestart_gate_step` as the first
  operator sequence item containing all required markers.
- Fail validation when no single step contains all markers.
- Include the exact accepted gate step in `summary.json` and `summary.md`.
- Add a regression for split markers across multiple sequence items.

## Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`

## Alternatives considered

- Keep whole-sequence matching. Rejected because it permits misleading operator
  instructions.
- Require the gate to be the final step. Deferred because same-step matching
  closes the immediate ambiguity without overfitting sequence order.

## Risks

Older handoff packets that split the prestart marker from the proof transition
now fail. That is intended fail-closed behavior.

## Test plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_no_wing_release_check.py -k "handoff_packet"`
- Affected no-WING/WING tests.
- Full pytest suite.
- `git diff --check`.

## Where the other agent may disagree

Kimi may require the gate step to be the final sequence item. I would consider
that next if needed; this slice enforces the important invariant that the proof
transition sentence itself contains the prestart condition.
