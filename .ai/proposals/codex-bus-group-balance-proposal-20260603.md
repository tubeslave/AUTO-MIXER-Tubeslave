# Codex Proposal: bus-group-balance-proposal-20260603

## 1. Short thesis

Move bus, group, DCA, and matrix balance from blocked metadata toward assisted
automation by adding a transport-free analyzer that emits non-queueable
bus/group balance proposals.

## 2. What I understood

The Automixer should build balance across channels and summed stems, not only
isolated sources. DCA, group, and matrix writes can affect many signals at once,
so the current supervised fader/gain live boundary must not widen without
separate proof.

## 3. Proposed solution

Create `backend/bus_group_balance_proposal.py` with bounded recommendation
parameters: bus id/kind, Ayaic target plane, measured integrated LUFS, summed
child LUFS, current fader readback, suggested fader delta, target fader, true
peak headroom, main-bus headroom, assignment confidence, feedback risk,
confidence, blockers, and evidence. It should block bus-up moves when true-peak
or main-bus headroom is insufficient, when assignment confidence is low, when
the current bus fader is missing, or when the fader ceiling would be exceeded.
Register it as `proposal_only`, add a proposal-only contract, and update the
project-goal matrix for mix balance and DCA/group/matrix control.

## 4. Likely files to touch

- `backend/bus_group_balance_proposal.py`
- `backend/processing_registry.py`
- `backend/processing_proposal_adapters.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_bus_group_balance_proposal.py`
- `tests/test_project_goal_capability_matrix.py`
- `Docs/adr/bus-group-balance-proposal-20260603.md`

## 5. Alternatives considered

- Promote existing DCA safety actions: rejected because they are not the current
  Product Layer executor and need separate physical proof.
- Keep DCA/group/matrix blocked with no proposal schema: rejected because it
  leaves an explicit goal capability without operator-facing progress.

## 6. Risks

- Group/DCA moves can alter many channels. The output stays non-queueable and
  requires assignment readback, rollback, and hardware proof before promotion.
- Bus-up moves can violate true-peak or feedback safety. The analyzer blocks
  insufficient headroom and high feedback-risk cases.

## 7. Test plan

- Unit tests for safe group proposal, peak-limited bus-up block, low assignment
  confidence block, Ayaic target helpers, proposal shape, registry status, and
  transport-free source scan.

## 8. Where the other agent may disagree

Kimi may prefer keeping DCA/group/matrix entirely blocked until hardware proof.
A proposal-only surface is safer for product progress because it exposes the
needed evidence without creating any DCA/group/matrix apply path.
