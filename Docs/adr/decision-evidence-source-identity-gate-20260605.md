# ADR: decision-evidence-source-identity-gate-20260605

## Context

Decision evidence packets summarize why the Automixer would recommend,
withhold, or reject a proposal. They collect evidence from audio analysis,
psychoacoustic and statistical models, project history, reference comparisons,
previous outcomes, inter-channel context, arrangement, and genre. Some of that
context depends on source identity. A stale or untrusted source identity label
can make a packet look credible while pointing later operator review or learning
at the wrong source.

## Options considered

1. Keep decision evidence packets independent from source identity trust.
2. Reuse the shared source identity downstream gate and demote untrusted packets
   to `analysis_only`.
3. Require fresh trusted source identity calibration for every packet before any
   proposal-only output.

## Decision

Use option 2. `decision_evidence_packet_proposal` now consumes the shared
transport-free source identity downstream gate. Held, blocked, unknown, or stale
source identity replay context adds blockers and prevents `proposal_only`
emission. Missing calibration remains a legacy-compatible
`legacy_unverified` evidence state.

## Why this won

This keeps the explanation layer aligned with the rest of the proposal-only
surface expansion without widening live writes. It blocks known-bad source
identity context at the evidence-packet boundary, before those packets influence
operator review, replay, or memory-learning proposals.

## Rejected alternatives

Option 1 ignores a known source of downstream decision error. Option 3 is too
strict until every caller passes replay-calibrated source identity evidence.

## Implementation plan

- Add optional source identity trust fields to
  `DecisionEvidenceAnalysisInput`.
- Evaluate `source_identity_downstream_gate` with the packet target and
  normalized action family.
- Extend blockers with gate blockers and store the gate decision in evidence.
- Mark the proposal contract and capability matrix with
  `source_identity_downstream_trust`.
- Add focused tests for legacy, held, and stale replay-context behavior.

## Test plan

- `PYTHONPATH=backend python3 -m py_compile backend/decision_evidence_packet_proposal.py backend/source_identity_downstream_gate.py backend/processing_proposal_adapters.py backend/project_goal_capability_matrix.py tests/test_decision_evidence_packet_proposal.py tests/test_project_goal_capability_matrix.py`
- `PYTHONPATH=backend python3 -m pytest tests/test_decision_evidence_packet_proposal.py tests/test_source_identity_downstream_gate.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`
- `/opt/homebrew/bin/git diff --check`
- Runtime smoke for held source identity: `analysis_only`, no built proposal,
  and no live, queue, model, or memory mutation flags.

## Risks and rollback

Risk: callers with held or stale source identity evidence will produce fewer
operator-review packets. Rollback is removing the input fields, gate evaluation,
blockers, evidence key, contract/matrix metadata, tests, and this ADR. No live
write behavior is affected.
