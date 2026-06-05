# Proposal: decision-evidence-source-identity-gate-20260605

## Short thesis

Gate decision evidence packets with replay-calibrated source identity trust
before the packets can influence operator review or downstream learning.

## What I understood

Decision evidence packets are proposal-only explanation records, not live apply
requests. They still carry decision influence: later layers may rely on their
source role, arrangement, and inter-channel evidence. If source identity is held
for operator review, blocked, unknown, or based on stale replay calibration, a
complete-looking packet should not be emitted as `proposal_only`.

## Proposed solution

Extend `DecisionEvidenceAnalysisInput` with optional source identity trust
fields that mirror the existing downstream gate contract. Evaluate
`source_identity_downstream_gate` with `target_id` as source id and normalized
action family as source type. Add gate blockers to packet blockers and record the
gate decision under `source_identity_downstream_trust`. Preserve legacy
compatibility by allowing missing calibration as `legacy_unverified`.

## Likely files to touch

- `backend/decision_evidence_packet_proposal.py`
- `backend/processing_proposal_adapters.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_decision_evidence_packet_proposal.py`
- `tests/test_project_goal_capability_matrix.py`
- `.ai/briefs/decision-evidence-source-identity-gate-20260605.md`
- `Docs/adr/decision-evidence-source-identity-gate-20260605.md`

## Alternatives considered

- Require source identity calibration for every packet immediately. Rejected for
  this cycle because legacy callers still need proposal-only analysis while the
  calibration field rolls out.
- Gate only memory learning. Rejected because evidence packets are the upstream
  explanation artifact and stale identity context should be filtered before it is
  reused.
- Add a new decision-packet-specific trust evaluator. Rejected because the shared
  gate already defines the transport-free downstream trust semantics.

## Risks

- Callers that pass held or stale identity evidence will produce fewer
  proposal-only packets.
- Missing source identity calibration remains visible as legacy evidence until
  upstream callers provide replay-calibrated trust fields.

## Test plan

- `PYTHONPATH=backend python3 -m py_compile backend/decision_evidence_packet_proposal.py backend/source_identity_downstream_gate.py backend/processing_proposal_adapters.py backend/project_goal_capability_matrix.py tests/test_decision_evidence_packet_proposal.py tests/test_project_goal_capability_matrix.py`
- `PYTHONPATH=backend python3 -m pytest tests/test_decision_evidence_packet_proposal.py tests/test_source_identity_downstream_gate.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`
- `/opt/homebrew/bin/git diff --check`
- Runtime smoke that verifies held source identity yields `analysis_only`, zero
  built proposals, and no live, queue, model, or memory mutation flags.

## Where the other agent may disagree

Kimi may argue that `legacy_unverified` should block all evidence packets. That
is a reasonable promotion bar, but this patch keeps legacy proposal-only
analysis working while blocking known untrusted or stale source identity context.
