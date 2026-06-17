# Codex Proposal: WING MVP no-WING Proof Plan Required Field Summary Embed

## Short thesis

Propagate required/missing proof-plan summary field diagnostics from `next_wing_proof_plan_schema` into the aggregated no-WING `next_wing_proof_plan` summary.

## What I understood

The WING proof path is still offline-gated until real hardware evidence exists. The validator now fails closed on missing required proof-plan summary fields, but the operator-facing no-WING summary should expose that same contract directly.

## Proposed solution

- Read `required_supervised_proof_plan_summary_fields` and `missing_required_supervised_proof_plan_summary_fields` from the schema check detail.
- Include both fields in `next_wing_proof_plan_summary()`.
- Extend the summary embedding regression test to assert both fields are present in `summary.json`.
- Update WING blocker docs and index with a short report.

## Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- Project memory docs under `.ai/` and `Docs/`

## Alternatives considered

- Leave the fields only in raw check detail. Rejected because the generated no-WING summary is the artifact operators and Project Director inspect first.
- Add a separate summary section. Rejected because the fields belong to the existing `next_wing_proof_plan` summary object.

## Risks

- Older summary fixtures may need regeneration if they assert exact `next_wing_proof_plan` content.
- This makes the generated summary more verbose, but the added fields are contract-level diagnostics.

## Test plan

- `PYTHONPATH=backend python -m pytest -q tests/test_no_wing_release_check.py::test_no_wing_release_check_summary_embeds_next_wing_proof_plan`
- Run the focused no-WING/handoff and Project Director rail tests before handoff.

## Where the other agent may disagree

Kimi may prefer keeping generated summary small. I prioritize operator-visible contract completeness here because missing required proof-plan fields are a supervised proof blocker.
