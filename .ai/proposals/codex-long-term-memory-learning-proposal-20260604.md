# Codex Proposal: Long-Term Memory Learning Proposal Layer

## 1. Short thesis

Move `long_term_memory_learning` from shadow-only to Assisted proposal support
by adding a transport-free learning-record review contract.

## 2. What I understood

The Automixer needs to improve from verified decisions, offline replay,
telemetry, accepted/rejected engineer outcomes, references, arrangement context,
genre context, and safety results. That learning should not become live console
authority. It should first become an operator-review proposal that can be
accepted into memory only after replay correlation and safety checks.

## 3. Proposed solution

Create `backend/long_term_memory_learning_proposal.py` with:

- deterministic scope and action-family normalization;
- blockers for insufficient verified/offline evidence, missing outcome labels,
  missing telemetry, missing evidence packets, low replay correlation, low
  outcome consistency, safety regressions, high conflicts, manual override
  bursts, latency gaps, active-performance writes, and auto-apply requests;
- Product Layer style output with `kind="long_term_memory_learning"`,
  `can_apply_live=False`, `can_import_to_queue=False`, `memory_mutation=False`,
  and no apply payload.

Register the method as proposal-only and attach it to
`long_term_memory_learning`.

## 4. Likely files to touch

- `backend/long_term_memory_learning_proposal.py`
- `backend/processing_registry.py`
- `backend/processing_proposal_adapters.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_long_term_memory_learning_proposal.py`
- `tests/test_project_goal_capability_matrix.py`
- `Docs/adr/long-term-memory-learning-proposal-20260604.md`

## 5. Alternatives considered

- Keep long-term learning shadow-only: safe, but it does not create an operator
  review path for memory records.
- Write directly into decision memory: rejected because memory records should
  require review and replay correlation.
- Let memory influence live actions immediately: rejected because memory can
  improve proposal priority only after separate safety proof.

## 6. Risks

- Memory learning can be mistaken for autonomous authority.
- Weak or biased replay labels can reinforce bad mix choices.
- Safety regressions must block learning records, not merely lower confidence.

## 7. Test plan

- Verify the module is free of mixer transport and apply calls.
- Verify complete verified/offline evidence emits proposal-only learning
  recommendations.
- Verify insufficient evidence, low replay correlation, safety regressions,
  conflicts, active performance, and auto-apply requests block proposals.
- Verify registry, proposal contract, and capability matrix wiring.

## 8. Where the other agent may disagree

Kimi may argue long-term memory should wait for a complete training pipeline.
The counterpoint is that this layer is the review contract a training pipeline
needs; it does not write memory automatically and does not grant live authority.
