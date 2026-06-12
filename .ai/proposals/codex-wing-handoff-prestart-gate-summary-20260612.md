# Codex Proposal: WING Handoff Prestart Gate Summary

## Short thesis

The no-WING release summary should show whether the WING handoff packet includes
the supervised prestart gate, not merely hide that fact inside low-level
validator detail.

## What I understood

The validator now fails stale handoff packets that omit valid accepted-candidate
prestart trace wording. Operators and reviewers still need that pass/fail fact
and the required marker list in `summary.json` and `summary.md`.

## Proposed solution

- Extend `wing_readonly_preflight_handoff_packet_artifact_summary`.
- Render `operator_sequence_has_supervised_prestart_gate` and
  `required_supervised_prestart_gate_markers` in Markdown.
- Add regression coverage to the existing summary embedding test.

## Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`

## Alternatives considered

- Leave the fields only in `CheckResult.detail`. Rejected because the summary is
  the operator-facing release artifact.

## Risks

Low. This only adds fields to an existing summary section.

## Test plan

- `PYTHONPATH=backend python3 -m pytest -q tests/test_no_wing_release_check.py -k "handoff_packet_artifact"`
- Affected WING/no-WING tests.
- Full pytest suite.
- `git diff --check`.

## Where the other agent may disagree

Kimi may prefer keeping Markdown compact. I would keep the fields visible
because this is the exact gate between read-only review and physical proof.
