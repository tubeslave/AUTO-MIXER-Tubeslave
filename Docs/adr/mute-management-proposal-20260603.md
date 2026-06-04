# mute-management-proposal-20260603

## 1. Context

The long-term project goal requires console mute control and real-time
corrections. Current live WING deployment is limited to supervised channel
fader/gain writes. Mute writes can silence critical sources, create audible
dropouts, or mask routing/DCA mistakes, and existing legacy mute write paths are
not approved live executors.

## 2. Options considered

- Promote existing mute write paths into the main assisted pipeline.
- Keep mute control blocked with no proposal schema.
- Add a separate transport-free mute management proposal layer.

## 3. Decision

Add `backend/mute_management_proposal.py` as a proposal-only analyzer. It
estimates mute/unmute recommendations from source role, current mute state,
auto-muted provenance, RMS/peak signal activity, noise floor, idle duration,
recent signal cycles, expected cue timing, source confidence, bleed risk,
feedback risk, dropout risk, operator intent, and persistence. It does not
create `requested_change` payloads for live apply.

## 4. Why this won

This gives the Product Layer a concrete mute-management recommendation surface
while preserving the current supervised fader/gain boundary. It makes manual
mute authority, signal provenance, and feedback/dropout risks visible before
any future promotion.

## 5. Rejected alternatives

Direct use of legacy mute writes would bypass current Product Layer proof
requirements. Keeping the area blocked with no proposal schema leaves no
testable evidence packet for future review.

## 6. Implementation plan

1. Build bounded mute analysis and recommendation objects.
2. Allow idle mute proposals only for safe or explicitly allowed source roles.
3. Allow unmute proposals only for system-auto-muted channels when signal
   returns or an upcoming cue is close.
4. Allow emergency mute proposals only from operator unsafe/feedback intent.
5. Emit proposal-only payloads with no live apply fields.
6. Register the method and proposal contract.
7. Update the project-goal capability matrix to assisted/proposal-only for mute
   control.

## 7. Test plan

- `PYTHONPATH=backend python -m pytest tests/test_mute_management_proposal.py -q`
- `PYTHONPATH=backend python -m pytest tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- Compile/smoke checks with the available Python runtime if full test
  dependencies are unavailable.

## 8. Risks and rollback

The main risk is overstating live readiness for mute. The method is registered
as proposal-only, emits no queueable live changes, requires readback/rollback
proof before promotion, and keeps mute writes outside the live executor.
Rollback is removal of the module, registry entry, proposal contract, matrix
update, tests, brief, proposal, and ADR; no live transport code is touched.
