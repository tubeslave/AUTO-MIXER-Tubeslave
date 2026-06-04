# bus-group-balance-proposal-20260603

## 1. Context

The long-term project goal requires automatic mix balance and control of DCA,
groups, and matrices. Current live WING deployment is limited to supervised
channel fader/gain writes. Group, DCA, matrix, and bus writes affect broad
console state and have no approved live proof path. The project canon also
requires Ayaic Mix Monolith level planes to be applied to both individual
sources and summed stems/groups.

## 2. Options considered

- Promote existing DCA safety actions into the main assisted pipeline.
- Keep DCA/group/matrix control blocked with no proposal schema.
- Add a separate transport-free bus/group balance proposal layer.

## 3. Decision

Add `backend/bus_group_balance_proposal.py` as a proposal-only analyzer. It
estimates bus/group/DCA/matrix balance recommendations from Ayaic target plane,
integrated LUFS, short-term LUFS, true peak, summed child LUFS, child count,
dominant child share, current fader readback, main-bus headroom, feedback risk,
assignment confidence, and persistence. It does not create `requested_change`
payloads for live apply.

## 4. Why this won

This gives the Product Layer a concrete path for summed bus balance while
preserving the current supervised fader/gain boundary. It turns a blocked goal
area into explainable evidence and operator-review proposals without promoting
any DCA, group, or matrix executor.

## 5. Rejected alternatives

Direct use of existing DCA actions would bypass the current Product Layer proof
requirements. Keeping the area blocked with no proposal schema leaves no
testable evidence packet for future review.

## 6. Implementation plan

1. Build bounded bus/group balance analysis and recommendation objects.
2. Resolve Ayaic target planes for lead, strong groups, backing vocals, support
   buses, and FX returns.
3. Block bus-up moves when true-peak headroom, main-bus headroom, fader ceiling,
   feedback risk, or assignment confidence is unsafe.
4. Emit proposal-only payloads with no live apply fields.
5. Register the method and proposal contract.
6. Update the project-goal capability matrix to assisted/proposal-only for
   DCA/group/matrix control.

## 7. Test plan

- `PYTHONPATH=backend python -m pytest tests/test_bus_group_balance_proposal.py -q`
- `PYTHONPATH=backend python -m pytest tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- Compile/smoke checks with the available Python runtime if full test
  dependencies are unavailable.

## 8. Risks and rollback

The main risk is overstating live readiness for broad bus controls. The method
is registered as proposal-only, emits no queueable live changes, requires
readback/rollback proof before promotion, and keeps DCA/group/matrix writes
outside the live executor. Rollback is removal of the module, registry entry,
proposal contract, matrix update, tests, brief, proposal, and ADR; no live
transport code is touched.
