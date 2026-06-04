# Codex Proposal: Decision Evidence Packet Proposal Layer

## 1. Short thesis

Move `decision_basis` from shadow-only to Assisted proposal support by adding a
transport-free evidence packet contract.

## 2. What I understood

The Automixer should act like an expert mixing engineer only when decisions are
explainable from multiple evidence families: audio metrics, psychoacoustic
models, statistics, project history, references, previous outcomes,
inter-channel context, arrangement role, and genre. That evidence is not live
authority. It should first become a non-queueable packet for operator review and
replay learning.

## 3. Proposed solution

Create `backend/decision_evidence_packet_proposal.py` with:

- deterministic evidence family normalization and confidence checks;
- blockers for missing critical evidence, insufficient family count, low
  confidence, missing operator intent, missing readback/rollback, low true-peak
  headroom, high feedback risk, latency gaps, conflicts, and auto-apply
  requests;
- Product Layer style output with `kind="decision_evidence_packet"`,
  `can_apply_live=False`, `can_import_to_queue=False`, and no apply payload.

Register the method as proposal-only, expose a proposal contract, and attach it
to `decision_basis`.

## 4. Likely files to touch

- `backend/decision_evidence_packet_proposal.py`
- `backend/processing_registry.py`
- `backend/processing_proposal_adapters.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_decision_evidence_packet_proposal.py`
- `tests/test_project_goal_capability_matrix.py`
- `Docs/adr/decision-evidence-packet-proposal-20260604.md`

## 5. Alternatives considered

- Keep `decision_basis` shadow-only: safe, but it does not create a reviewable
  contract for proposal and rejection learning.
- Let evidence packets feed live apply directly: rejected because evidence
  explains decisions but does not prove actuator safety.
- Require every evidence family every time: rejected because some families may
  be unavailable; critical families and minimum coverage are safer and more
  practical.

## 6. Risks

- Evidence packets may be mistaken for authority if the UI does not preserve the
  non-queueable status.
- Missing evidence could bias decisions if replay does not track blockers.
- Console-affecting decisions still need separate readback, rollback, hardware,
  and operator acceptance proof.

## 7. Test plan

- Verify the module is free of mixer transport and apply calls.
- Verify complete evidence emits proposal-only packets.
- Verify missing critical evidence, low confidence, headroom, feedback,
  conflict, latency, readback, rollback, and auto-apply blockers.
- Verify registry, proposal contract, and capability matrix wiring.

## 8. Where the other agent may disagree

Kimi may argue this should wait for the long-term memory engine. The
counterpoint is that the packet is the stable contract memory and replay need;
it does not claim live authority.
