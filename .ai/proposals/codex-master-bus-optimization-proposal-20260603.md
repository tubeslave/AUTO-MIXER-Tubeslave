# Codex Proposal: master-bus-optimization-proposal-20260603

## 1. Short thesis

Move master bus optimization from offline/shadow metadata toward assisted
automation by adding a transport-free analyzer that emits non-queueable master
bus proposals.

## 2. What I understood

The Automixer should eventually optimize the main mix bus, but master bus moves
are broad PA-affecting changes. The current approved live executor is still
limited to supervised fader/gain, so master processing must first become
explainable evidence and operator-review proposals.

## 3. Proposed solution

Create `backend/master_bus_optimization_proposal.py` with bounded recommendation
parameters: target integrated LUFS, true-peak ceiling, safe static trim, limiter
ceiling, broad tonal tilt, low/high shelf corrections, glue-compression ratio,
threshold, and target gain reduction. It should block peak-limited loudness-up
cases without approved limiter/dynamics proof and should state that LUFS-only
gain staging is not a listening master. Register it as `proposal_only`, add a
proposal-only contract, and update the project-goal matrix to
assisted/proposal-only.

## 4. Likely files to touch

- `backend/master_bus_optimization_proposal.py`
- `backend/processing_registry.py`
- `backend/processing_proposal_adapters.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_master_bus_optimization_proposal.py`
- `tests/test_project_goal_capability_matrix.py`
- `Docs/adr/master-bus-optimization-proposal-20260603.md`

## 5. Alternatives considered

- Promote offline mastering or limiter logic: rejected because render code is
  not a live apply path and can hide broad state changes.
- Keep master bus offline-only: rejected because the goal needs product-visible
  progress while preserving live safety.

## 6. Risks

- Master bus changes can affect every listener. The output stays non-queueable,
  requires readback/rollback proof before promotion, and never widens the live
  write kinds.
- Loudness targets can tempt unsafe gain-up. The analyzer blocks loudness-up
  when true-peak headroom or limiter/dynamics proof is insufficient.

## 7. Test plan

- Unit tests for healthy proposal, peak-limited loudness-up block, tonal
  recommendations, proposal shape, registry status, and transport-free source
  scan.
- Focused registry and project-goal matrix tests.

## 8. Where the other agent may disagree

Kimi may prefer keeping master-bus work in offline render code until a limiter
implementation is proven. A separate proposal-only surface is safer for product
progress because it makes the safety blockers visible without creating any
main-bus apply path.
