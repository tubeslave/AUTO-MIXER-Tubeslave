# pan-spatial-proposal-20260603

## 1. Context

The long-term project goal requires automatic panning and pan control on the
console. Existing panner modules contain useful heuristics but also include
apply paths. The current live runtime must remain supervised fader/gain only.

## 2. Options considered

- Promote `auto_panner.py` into the main assisted pipeline.
- Keep panning lab-only until supervised pan writes are proven.
- Add a separate transport-free pan proposal layer.

## 3. Decision

Add `backend/pan_spatial_proposal.py` as a proposal-only analyzer. It estimates
target pan from source role, current pan, stereo-pair side, stage position,
low-frequency share, contribution, and mono-compatibility risk. It does not
create `requested_change` payloads for live apply.

## 4. Why this won

This moves panning from lab-only metadata into a usable Product Layer proposal
surface while preserving the current live-write boundary. It separates spatial
reasoning from legacy transport/apply code.

## 5. Rejected alternatives

Direct reuse of `auto_panner.py` risks mixing proposal logic with console
writes. Keeping panning lab-only leaves an explicit project-goal capability
without a product-visible proposal surface.

## 6. Implementation plan

1. Build bounded pan analysis and recommendation objects.
2. Allow center-anchor corrections and stereo/stage-backed non-center pan
   proposals.
3. Block high mono-risk and missing-context cases.
4. Emit proposal-only payloads with no live apply fields.
5. Register the method and proposal contract.
6. Update the project-goal capability matrix to assisted/proposal-only.

## 7. Test plan

- `PYTHONPATH=backend python -m pytest tests/test_pan_spatial_proposal.py -q`
- `PYTHONPATH=backend python -m pytest tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- Compile/smoke checks with the available Python runtime if full test
  dependencies are unavailable.

## 8. Risks and rollback

The main risk is overstating live readiness. The method is registered as
proposal-only, emits no queueable live changes, and includes blockers for
unsupported pan value type and missing hardware proof. Rollback is removal of
the module, registry entry, proposal contract, matrix update, tests, brief,
proposal, and ADR; no live transport code is touched.
