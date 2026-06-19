# ADR: Offline Replay Evaluation Contract

## Context

The Project Completion Gate requires end-to-end soundcheck, studio and concert
evaluations, approved quality thresholds, recovery/failure evidence and no
bypass around the deterministic safety kernel. Existing offline and no-WING
artifacts are useful, but they must not be confused with hardware proof, live
authority or Full Autonomous readiness.

## Options Considered

1. Keep relying on narrative documentation and scattered offline reports.
2. Add a fail-closed offline replay evaluation contract.
3. Wire replay evidence directly into readiness promotion.

## Decision

Add `backend/offline_replay_evaluation_contract.py` as a deterministic,
offline-only contract. It validates soundcheck, studio and concert scenario
coverage, offline/replay/fault-injection evidence, safety checks, quality axes
and approved threshold policy before reporting `ready_for_replay_promotion`.

The contract always returns `live_authority_granted: false` and
`readiness_closure_allowed: false`.

## Why This Won

This creates a concrete gate that moves Automixer toward replay-level evaluation
without needing WING hardware and without widening live authority. It also makes
threshold approval and human/listening evidence explicit instead of allowing a
green unit test or offline artifact to imply completion.

## Rejected Alternatives

- Narrative-only tracking was rejected because it does not give tests a stable
  authority boundary.
- Direct readiness promotion was rejected because offline replay evidence is not
  hardware proof and thresholds are not currently approved.

## Implementation Plan

1. Add the offline replay evaluation contract module.
2. Add unit tests for incomplete evidence, complete offline evidence, live or
   hardware claims, missing quality axes and forbidden live transport imports.
3. Bind the contract to `Docs/ROADMAP.md`, `Docs/CAPABILITY_MATRIX.md`,
   `Docs/CURRENT_STATE.md`, `ops/autopilot/LAST_REPORT.md` and the M011 rail.

## Test Plan

- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_offline_replay_evaluation_contract.py -q`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/test_wing_mvp_rails.py -q`
- `/opt/homebrew/bin/python3 tools/memory/memory_lint.py --root .`
- `PYTHONPATH=backend /opt/homebrew/bin/python3 -m pytest tests/ -x --tb=short -q`

## Risks and Rollback

The main risk is treating a complete offline replay bundle as stronger evidence
than it is. The contract mitigates this by keeping live authority and readiness
closure false even when replay promotion is ready.

Rollback is to revert the M011 commit. No live console path imports or calls
this contract.

