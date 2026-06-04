# Codex Proposal: mute-management-proposal-20260603

## 1. Short thesis

Move mute control from blocked metadata toward assisted automation by adding a
transport-free analyzer that emits non-queueable mute/unmute proposals.

## 2. What I understood

The Automixer should eventually control mute state, but live mute writes can
silence important sources and create audible dropouts. The current approved live
executor remains supervised fader/gain only, so mute logic must first become
explainable evidence and operator-review proposals.

## 3. Proposed solution

Create `backend/mute_management_proposal.py` with bounded recommendation
parameters: current mute state, proposed mute state, action type, source role,
source family, idle/activity evidence, auto-muted provenance, operator intent,
feedback risk, dropout risk, source confidence, confidence, blockers, and
evidence. It should allow proposal-only idle mute for safe sources, auto-unmute
for system-muted channels when signal returns, and emergency mute proposals when
operator intent flags feedback or unsafe signal. It should block critical source
idle mutes and manual unmute attempts without system provenance.

## 4. Likely files to touch

- `backend/mute_management_proposal.py`
- `backend/processing_registry.py`
- `backend/processing_proposal_adapters.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_mute_management_proposal.py`
- `tests/test_project_goal_capability_matrix.py`
- `Docs/adr/mute-management-proposal-20260603.md`

## 5. Alternatives considered

- Promote legacy mute writes: rejected because they are not in the current
  Product Layer executor and lack readback/rollback proof.
- Keep mute blocked with no proposal schema: rejected because the goal needs
  operator-visible progress toward safe mute management.

## 6. Risks

- Mute proposals can hide important sources. The module blocks critical-source
  idle mutes and manual unmute cases without provenance.
- Emergency mute can still affect a performance. The output remains
  non-queueable and requires operator review before any future promotion.

## 7. Test plan

- Unit tests for idle close-mic mute proposal, auto-unmute proposal, emergency
  feedback proposal, critical-source block, manual-mute authority block,
  proposal shape, registry status, and transport-free source scan.

## 8. Where the other agent may disagree

Kimi may prefer leaving mute entirely blocked until hardware proof. A
proposal-only surface is safer for progress because it exposes the needed
evidence while keeping all mute writes out of live execution.
