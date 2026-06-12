# Codex Proposal: WING Read-Only Runbook Final Prestart Gate

## Short thesis

Update the read-only preflight runbook so it cannot imply that an accepted
decision JSON alone unlocks supervised proof. The runbook should require the
same final handoff prestart gate that the release-check tooling already
validates.

## What I understood

Recent WING MVP work made `WING_SUPERVISED_FADER_GAIN_PROOF` depend on a valid
accepted-candidate prestart trace in the final handoff operator-sequence step.
`Docs/runbooks/WING_READONLY_PREFLIGHT.md` still had older wording that only
mentioned a clean artifact and accepted decision JSON before continuing to the
supervised runbook.

## Proposed solution

- Clarify the read-only runbook exit-code and next-step wording.
- Require the release handoff to expose
  `operator_sequence_has_supervised_prestart_gate: true` and
  `operator_sequence_supervised_prestart_gate_is_final: true`.
- Name the required final-step markers:
  `accepted_decision_ready`, `valid accepted-candidate prestart trace`, and
  `WING_SUPERVISED_FADER_GAIN_PROOF`.
- Extend the existing WING MVP rails test to assert those strings stay in the
  read-only runbook.

## Likely files to touch

- `Docs/runbooks/WING_READONLY_PREFLIGHT.md`
- `tests/test_wing_mvp_rails.py`
- `.ai/briefs/wing-readonly-runbook-final-prestart-gate-20260612.md`
- `.ai/proposals/codex-wing-readonly-runbook-final-prestart-gate-20260612.md`
- `Docs/adr/wing-readonly-runbook-final-prestart-gate-20260612.md`
- `Docs/reports/AUTOMIXER_WING_READONLY_RUNBOOK_FINAL_PRESTART_GATE_20260612.md`

## Alternatives considered

- Leave the runbook unchanged because tooling already gates the handoff. Rejected
  because operator-facing instructions would still describe a weaker transition.
- Edit historical reports. Rejected because they are point-in-time records.

## Risks

- Documentation drift if future release-check summary field names change.
- Test fragility from string checks. This is acceptable because the risk is
  precisely losing operator-visible safety wording.

## Test plan

- Focused WING MVP rails test for runbook wording.
- Affected WING MVP/release-check tests.
- `tools/memory/memory_lint.py`
- Full pytest suite.

## Where the other agent may disagree

Kimi may argue that this belongs only in release-check tests. My view is that
operator runbooks are part of the safety boundary and should carry explicit
release-check evidence names.
