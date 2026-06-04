# master-bus-optimization-proposal-20260603

## 1. Context

The long-term project goal requires automatic master bus optimization. Existing
offline renderers and quality scoring include loudness and master-stage
concepts, but live WING deployment is currently limited to supervised
fader/gain writes. Main bus, matrix, limiter, EQ, and compressor changes affect
the whole PA and are not approved live write surfaces.

## 2. Options considered

- Promote existing offline mastering logic into the main assisted pipeline.
- Keep master bus optimization offline/shadow only.
- Add a separate transport-free master bus proposal layer.

## 3. Decision

Add `backend/master_bus_optimization_proposal.py` as a proposal-only analyzer.
It estimates master-bus recommendations from integrated LUFS, short-term LUFS,
true peak, crest factor, loudness range, spectral tilt, low-end excess,
harshness, stereo correlation, limiter ceiling, reference/profile target, and
limiter/dynamics proof. It does not create `requested_change` payloads for live
apply.

## 4. Why this won

This gives the Product Layer a concrete master-bus recommendation surface while
preserving the current supervised fader/gain boundary. It makes loudness-vs-
true-peak blockers visible instead of hiding master decisions inside offline
render behavior.

## 5. Rejected alternatives

Direct reuse of offline mastering or limiter code risks conflating render-time
processing with live main-bus control. Keeping master bus offline-only leaves an
explicit goal capability without assisted, operator-facing evidence.

## 6. Implementation plan

1. Build bounded master-bus analysis and recommendation objects.
2. Recommend safe static trim, ceiling, broad EQ, and glue-compression settings
   only as proposal payloads.
3. Block loudness-up cases when true-peak headroom or approved limiter/dynamics
   proof is missing.
4. Emit proposal-only payloads with no live apply fields.
5. Register the method and proposal contract.
6. Update the project-goal capability matrix to assisted/proposal-only.

## 7. Test plan

- `PYTHONPATH=backend python -m pytest tests/test_master_bus_optimization_proposal.py -q`
- `PYTHONPATH=backend python -m pytest tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- Compile/smoke checks with the available Python runtime if full test
  dependencies are unavailable.

## 8. Risks and rollback

The main risk is overstating live readiness or treating LUFS-only gain as a
finished listening master. The method is registered as proposal-only, emits no
queueable live changes, includes explicit blockers for missing limiter/dynamics
proof, and keeps master-bus writes outside the live executor. Rollback is
removal of the module, registry entry, proposal contract, matrix update, tests,
brief, proposal, and ADR; no live transport code is touched.
