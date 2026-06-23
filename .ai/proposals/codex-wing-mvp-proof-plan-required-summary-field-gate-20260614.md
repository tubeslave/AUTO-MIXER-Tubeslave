# Codex Proposal: WING MVP Proof Plan Required Summary Field Gate

## Short thesis

Bind the handoff-required proof-plan summary field contract to the `next_wing_proof_plan` schema validator so incomplete summaries fail closed.

## What I understood

The current WING MVP is still blocked on real read-only hardware evidence. Offline work should therefore tighten the read-only-to-supervised proof chain without granting execution authority. After making artifact-root fields handoff-required, the next gap is direct validation that the proof-plan summary actually contains every required field.

## Proposed solution

- In `validate_supervised_proof_report_next_wing_plan`, derive the required summary fields from the shared contract tuple.
- Add `missing_required_supervised_proof_plan_summary_fields` to the check detail.
- Emit `next_wing_proof_plan_required_summary_fields_missing` when any required key is absent.
- Add a regression test that removes an artifact-root summary field from an otherwise valid proof report.

## Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- Project memory docs under `.ai/` and `Docs/`

## Alternatives considered

- Rely on existing value-specific mismatch checks. Rejected because missing fields can be coerced to empty values and should produce a clear contract-level failure.
- Duplicate the required field list in the validator. Rejected because the handoff and no-WING checks must share the same contract source.

## Risks

- Older generated proof-plan summaries that lack newly required fields will fail closed until regenerated.
- The new violation is intentionally stricter and may reveal stale artifacts in local bundles.

## Test plan

- `PYTHONPATH=backend python -m pytest -q tests/test_no_wing_release_check.py::test_no_wing_release_check_validates_next_wing_proof_plan_schema tests/test_no_wing_release_check.py::test_no_wing_release_check_rejects_next_wing_plan_missing_required_summary_field`
- Run the focused no-WING/handoff and Project Director rail tests before handoff.

## Where the other agent may disagree

Kimi may prefer checking only fields referenced by physical proof steps. I kept the gate aligned to the handoff-required contract because this boundary is about summary completeness before supervised proof planning, not only step execution.
