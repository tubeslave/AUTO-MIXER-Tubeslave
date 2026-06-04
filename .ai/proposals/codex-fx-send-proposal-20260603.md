# Codex Proposal: fx-send-proposal-20260603

## 1. Short thesis

Move FX management and effect sends from lab-only toward assisted automation by
adding a transport-free analyzer that emits non-queueable FX send proposals.

## 2. What I understood

The Automixer should eventually manage effects and sends, but FX writes can
create feedback, wash out vocal intelligibility, and alter broad console state.
Current live writes are limited to supervised fader/gain, so FX must first
become explainable evidence and operator-review proposals.

## 3. Proposed solution

Create `backend/fx_send_proposal.py` with bounded recommendation parameters:
FX type, bus name, current/target send, return level, filters, predelay/decay or
tempo delay, ducking, confidence, blockers, and evidence. It should block dry
anchor sources, insufficient return headroom, high feedback risk, and high
intelligibility risk. Register it as `proposal_only`, add a proposal-only
contract, and update the project-goal matrix from lab/quarantined to
assisted/proposal-only.

## 4. Likely files to touch

- `backend/fx_send_proposal.py`
- `backend/processing_registry.py`
- `backend/processing_proposal_adapters.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_fx_send_proposal.py`
- `tests/test_project_goal_capability_matrix.py`
- `Docs/adr/fx-send-proposal-20260603.md`

## 5. Alternatives considered

- Promote `auto_fx.py`: rejected because it includes apply behavior.
- Keep FX lab-only: rejected because the goal needs product-visible progress
  toward assisted FX management.

## 6. Risks

- FX sends can trigger feedback or bury lyrics. The module blocks high-risk
  cases and records return-headroom/intelligibility evidence.
- FX changes touch broad state. The output stays non-queueable and requires
  readback/rollback proof before any future promotion.

## 7. Test plan

- Unit tests for vocal send proposal, tempo delay calculation, dry-anchor
  block, feedback/headroom block, proposal shape, registry status, and
  transport-free source scan.

## 8. Where the other agent may disagree

Kimi may prefer adapting the existing planner directly. A separate proposal-only
surface is safer because it keeps direct FX writes quarantined while making FX
evidence usable by Product Layer.
