# Proposal: bus-group-balance-source-identity-gate-20260605

## Short thesis

Use replay-calibrated source identity trust as a bus/group balance blocker so
grouped balance moves cannot rely on explicitly untrusted source-role labels.

## What I understood

The bus/group layer already checks Ayaic target planes, integrated and
short-term LUFS, summed child loudness, assignment confidence, fader readback,
true-peak headroom, main-bus headroom, feedback risk, persistence, and current
bus/group/DCA/matrix freshness. It is explicitly non-queueable and
transport-free. Because grouped balance decisions depend on whether the group
really represents drums, bass, backing vocals, FX returns, support stems, or a
DCA/matrix target, source identity trust should become part of the safety
evidence before any grouped balance proposal is shown.

## Proposed solution

Add optional source identity trust fields to `BusGroupBalanceAnalysisInput` and
evaluate them through `source_identity_downstream_gate`. Add the resulting
blockers and evidence to `BusGroupBalanceRecommendation`. Keep legacy missing
calibration allowed so existing callers remain compatible, but block explicit
`hold`, `blocked`, stale, or unrecognized trust states.

Update the proposal-only contract and capability matrix support paths, then add
focused tests for legacy compatibility, held trust, stale replay context, and
contract/matrix evidence.

## Likely files to touch

- `backend/bus_group_balance_proposal.py`
- `backend/processing_proposal_adapters.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_bus_group_balance_proposal.py`
- `.ai/briefs/bus-group-balance-source-identity-gate-20260605.md`
- `Docs/adr/bus-group-balance-source-identity-gate-20260605.md`

## Alternatives considered

- Require source identity calibration for all bus/group balance inputs
  immediately. Rejected because legacy callers do not yet pass replay
  artifacts.
- Ignore replay trust for grouped balance. Rejected because a wrong source role
  can choose the wrong Ayaic plane or imply a different DCA/group balance move.
- Duplicate trust-state parsing inside bus/group balance. Rejected because the
  shared gate keeps blocker names and no-mutation evidence consistent across
  downstream modules.

## Risks

- Conservative blocking may hold useful group-balance proposals until replay
  calibration is consistently available.
- The gate currently relies on upstream trust-state strings; future DTO wiring
  should pass structured calibration records.

## Test plan

```bash
PYTHONPATH=backend python3 -m py_compile \
  backend/bus_group_balance_proposal.py \
  backend/source_identity_downstream_gate.py \
  backend/processing_proposal_adapters.py \
  backend/project_goal_capability_matrix.py \
  tests/test_bus_group_balance_proposal.py
PYTHONPATH=backend python3 -m pytest \
  tests/test_bus_group_balance_proposal.py \
  tests/test_source_identity_downstream_gate.py \
  tests/test_processing_registry.py \
  tests/test_project_goal_capability_matrix.py -q
PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q
/opt/homebrew/bin/git diff --check
```

## Where the other agent may disagree

Kimi may argue that grouped balance can rely on assignment confidence alone.
I would keep the source identity gate because assignment confidence says the
bus membership looks coherent, while identity trust says the role labels should
be allowed to influence downstream mix balance.
