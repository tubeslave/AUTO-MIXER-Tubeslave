# Codex Proposal: compressor-dynamics-proposal-20260603

## 1. Short thesis

Close the compression capability gap by adding a transport-free dynamics
analyzer that emits compressor proposals without becoming a console executor.

## 2. What I understood

The Automixer should eventually perform compression automatically, but current
live writes are limited to supervised fader/gain. Compressor decisions need to
enter the product as explainable evidence and reviewable settings before any
future supervised dynamics write is considered.

## 3. Proposed solution

Create `backend/compressor_dynamics_proposal.py` with bounded recommendation
parameters: threshold, ratio, attack, release, knee, target gain reduction,
makeup gain, confidence, blockers, and evidence. Register it as
`proposal_only`, add a proposal-only contract, and update the project-goal
matrix so compression points at this clean surface instead of relying only on
legacy direct-controller code.

## 4. Likely files to touch

- `backend/compressor_dynamics_proposal.py`
- `backend/processing_registry.py`
- `backend/processing_proposal_adapters.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_compressor_dynamics_proposal.py`
- `tests/test_project_goal_capability_matrix.py`
- `Docs/adr/compressor-dynamics-proposal-20260603.md`

## 5. Alternatives considered

- Reuse `auto_compressor.py` directly: rejected because it owns transport and
  mixer apply behavior.
- Keep compression only as a registry contract: rejected because the goal needs
  actual analyzer output, not just roadmap metadata.

## 6. Risks

- Compressor proposals could over-compress already-controlled sources. The new
  module blocks heavy existing GR and low-need cases.
- Makeup gain could hide headroom risk. The module bounds makeup by true-peak
  headroom and remains non-queueable.

## 7. Test plan

- Unit tests for dynamic-source trigger, flat-source block, insufficient
  headroom block, proposal shape, registry status, and transport-free source
  scan.

## 8. Where the other agent may disagree

Kimi may prefer promoting the existing compressor controller. A separate
proposal-only layer is the safer first move because it keeps legacy direct
writes quarantined while making compression evidence usable by Product Layer.
