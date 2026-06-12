# Codex Proposal: WING Handoff Structured Prestart Gate

## Short thesis

The handoff packet should expose the supervised prestart gate as structured JSON
so release tooling and future agents do not have to infer the active safety
boundary from operator prose alone.

## What I understood

The current no-WING validator already computes whether the operator sequence
contains a final step with `accepted_decision_ready`, valid accepted-candidate
prestart trace, and `WING_SUPERVISED_FADER_GAIN_PROOF`. The generated packet
itself did not carry a matching machine-readable gate contract.

## Proposed solution

- Add `supervised_prestart_gate` to
  `tools/build_wing_readonly_preflight_handoff_packet.py` output.
- Include required markers, final-step index, final-step requirement,
  target runbook, decision status, accepted-candidate prestart trace
  requirement, and `grants_live_authority: false`.
- Render the contract in handoff Markdown.
- Require `tools/no_wing_release_check.py` to reject missing or mismatched
  structured gate contracts.
- Add generator and validator regressions.

## Likely files to touch

- `tools/build_wing_readonly_preflight_handoff_packet.py`
- `tools/no_wing_release_check.py`
- `tests/test_wing_readonly_preflight_handoff_packet.py`
- `tests/test_no_wing_release_check.py`
- `.ai/briefs/wing-handoff-structured-prestart-gate-20260612.md`
- `.ai/proposals/codex-wing-handoff-structured-prestart-gate-20260612.md`
- `Docs/adr/wing-handoff-structured-prestart-gate-20260612.md`
- `Docs/reports/AUTOMIXER_WING_HANDOFF_STRUCTURED_PRESTART_GATE_20260612.md`

## Alternatives considered

- Keep deriving everything from `operator_sequence`. Rejected because it makes
  future automation depend on prose parsing.
- Replace the prose step with only JSON. Rejected because operators still need a
  direct sequence to follow.

## Risks

The structured block can drift from the operator sequence. The validator now
checks exact consistency and fails closed on mismatch.

## Test plan

- Handoff packet generator tests.
- no-WING handoff packet validator tests.
- Affected WING/no-WING tests.
- Memory lint and full pytest.

## Where the other agent may disagree

Kimi may ask for a schema version field. I would add that when the packet has a
second consumer or migration path; the current exact contract check is enough
for this step.
