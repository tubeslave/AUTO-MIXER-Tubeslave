# Codex Proposal: deesser-proposal-layer-20260603

## 1. Short thesis

Close the explicit de-essing gap by adding a transport-free vocal sibilance
analyzer that emits proposal-only de-esser recommendations.

## 2. What I understood

The Automixer should eventually perform de-essing automatically, but current
live writes are limited to supervised fader/gain. De-essing must therefore
enter the architecture as evidence and proposals, not as a console executor.

## 3. Proposed solution

Create `backend/deesser_vocal_sibilance.py` with bounded recommendation
parameters: center frequency, threshold, ratio, confidence, blockers, and
evidence. Register it as `proposal_only` in `processing_registry.py`, add a
proposal-only contract, and update the project-goal matrix from missing to
proposal-only.

## 4. Likely files to touch

- `backend/deesser_vocal_sibilance.py`
- `backend/processing_registry.py`
- `backend/processing_proposal_adapters.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_deesser_vocal_sibilance.py`
- `tests/test_project_goal_capability_matrix.py`
- `Docs/adr/deesser-proposal-layer-20260603.md`

## 5. Alternatives considered

- Use `backend/ai/agent.py` `adjust_deesser` directly: rejected because that path
  can call mixer methods and is not the supervised proposal contract.
- Treat sibilance as ordinary AutoEQ: rejected because de-essing needs dynamic
  evidence and vocal-source checks.

## 6. Risks

- Vocal sibilance proposals could be triggered by cymbal bleed. The module
  blocks non-vocal source roles and records contribution/share evidence.
- Target parameters are estimates. They remain non-queueable until hardware
  readback and rollback exist.

## 7. Test plan

- Unit tests for vocal trigger, non-vocal block, low-sibilance block, proposal
  shape, registry status, and transport-free source scan.

## 8. Where the other agent may disagree

Kimi may prefer starting with a full dynamic-EQ processor. A proposal-only
detector is the safer first step because it creates evidence and product
visibility without widening live-write scope.
