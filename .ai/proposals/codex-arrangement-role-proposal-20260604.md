# Codex Proposal: Arrangement Role Proposal Layer

## 1. Short thesis

Move arrangement-role detection from shadow-only to Assisted proposal support by
adding a transport-free role evidence contract.

## 2. What I understood

The Automixer cannot make expert mix decisions unless it understands whether a
source is a lead, anchor, support, texture, effect, utility, or reference. That
role decision should not directly apply any console move. It should first become
an explainable proposal that the operator can accept or reject.

## 3. Proposed solution

Create `backend/arrangement_role_proposal.py` with:

- source-type and arrangement-role normalization;
- deterministic role inference from lead salience, rhythmic stability, low
  anchor, harmonic support, texture, and masking evidence;
- blockers for missing arrangement/inter-channel context, low confidence, low
  persistence, high ambiguity, recent role switching, and disabled operator
  review;
- Product Layer style proposal output with `kind="arrangement_role"`,
  `can_apply_live=False`, `can_import_to_queue=False`, and no
  `requested_change`.

Register the method as proposal-only and attach it to `identify_arrangement_roles`.

## 4. Likely files to touch

- `backend/arrangement_role_proposal.py`
- `backend/processing_registry.py`
- `backend/processing_proposal_adapters.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_arrangement_role_proposal.py`
- `tests/test_project_goal_capability_matrix.py`
- `Docs/adr/arrangement-role-proposal-20260604.md`

## 5. Alternatives considered

- Keep roles shadow-only: safe, but it does not create an Assisted-mode feedback
  loop.
- Feed roles directly into fader/EQ decisions: rejected because role authority
  needs accepted/rejected replay evidence first.
- Reuse only source-type classification: insufficient because source type and
  arrangement role are different decisions.

## 6. Risks

- Role proposals can bias downstream processing if treated as authoritative too
  early.
- Role detection may be section-dependent and unstable without persistence.
- Ambiguous roles need explicit rejection/hold behavior.

## 7. Test plan

- Verify the module is free of mixer transport and apply calls.
- Verify a lead vocal can produce a proposal-only role recommendation.
- Verify ambiguity, recent switches, missing context, and low confidence block
  proposals.
- Verify registry, proposal contract, and capability matrix wiring.

## 8. Where the other agent may disagree

Kimi may argue role proposals should wait for a full replay engine. The
counterpoint is that this layer is exactly the evidence contract replay needs;
it does not claim live authority.
