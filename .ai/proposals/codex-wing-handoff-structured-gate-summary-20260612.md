# Codex Proposal: WING Handoff Structured Gate Summary

## Short thesis

The no-WING release summary should surface the structured supervised prestart
gate that the handoff validator now checks, so reviewers can inspect the actual
machine-readable contract from summary artifacts.

## What I understood

The handoff packet now includes `supervised_prestart_gate`, and the validator
rejects missing or mismatched contracts. The summary surface still rendered only
derived `operator_sequence_*` fields, hiding the structured object from JSON and
Markdown review.

## Proposed solution

- Add `supervised_prestart_gate` and `expected_supervised_prestart_gate` to
  `wing_readonly_preflight_handoff_packet_artifact_summary`.
- Render both fields in Markdown release summaries.
- Extend the summary regression to assert the structured gate, final-step index,
  `grants_live_authority: false`, and
  `requires_accepted_candidate_prestart_trace: true`.

## Likely files to touch

- `tools/no_wing_release_check.py`
- `tests/test_no_wing_release_check.py`
- `.ai/briefs/wing-handoff-structured-gate-summary-20260612.md`
- `.ai/proposals/codex-wing-handoff-structured-gate-summary-20260612.md`
- `Docs/adr/wing-handoff-structured-gate-summary-20260612.md`
- `Docs/reports/AUTOMIXER_WING_HANDOFF_STRUCTURED_GATE_SUMMARY_20260612.md`

## Alternatives considered

- Leave summaries unchanged because validator detail already has the field.
  Rejected because release artifacts are the operator/reviewer surface.
- Render only the actual gate, not the expected gate. Rejected because the
  comparison is what proves the contract matches the final operator step.

## Risks

The Markdown line may become verbose. This is acceptable for a release-check
artifact whose job is explicit safety evidence.

## Test plan

- Focused no-WING summary test.
- no-WING handoff packet tests.
- Affected WING/no-WING tests.
- Memory lint and full pytest.

## Where the other agent may disagree

Kimi may prefer normalizing the structured gate into separate Markdown lines.
That is a reasonable future cleanup; for this pass, exact JSON rendering keeps
the summary tied to the machine-readable artifact.
