# Codex Proposal: read-surface-physical-readback-capture-plan-20260605

## Short thesis

Promote the capture plan only as read-only/operator-required planning evidence,
and harden release summaries so they explicitly deny live authority.

## What I understood

Automixer needs a practical next step for physical readback gaps, but the system
must not let a checklist become a console executor. The current gap is not
another live path; it is evidence clarity for the next operator-captured
hardware artifact.

## Proposed solution

- Keep `backend/read_surface_physical_readback_capture_plan.py` transport-free
  and non-executing.
- Expose the capture plan through registry, no-WING summaries, Product Layer
  proof quality, and frontend read-only status rows.
- Serialize and validate `live_authority_granted=false` alongside
  `live_write=false`, `execution_allowed_by_plan=false`, and
  `full_autonomous_ready=false`.
- Record an ADR so later agents do not treat the checklist as a hardware runner.

## Likely files to touch

- `backend/read_surface_physical_readback_capture_plan.py`
- `tools/no_wing_release_check.py`
- `tests/test_read_surface_physical_readback_capture_plan.py`
- `tests/test_no_wing_release_check.py`
- `backend/operator_product_state.py`
- `frontend/src/services/operatorAdapters.ts`
- `Docs/adr/read-surface-physical-readback-capture-plan-20260605.md`

## Alternatives considered

- Add a real capture runner now: rejected because it would require operator
  tooling, hardware scope, readback storage rules, and a separate supervised
  proof plan.
- Keep the target only in remediation summaries: rejected because operators and
  release artifacts need a concrete checklist and explicit forbidden actions.
- Treat cached read state as proof: rejected because cached state cannot close a
  physical readback fixture gap.

## Risks

- Operators could mistake a checklist for permission to run hardware actions.
  Mitigation: explicit `execution_allowed_by_plan=false`,
  `live_authority_granted=false`, and forbidden-action text.
- Stale no-WING artifacts could display an old next target. Mitigation:
  target/action consistency checks and Product Layer normalization.
- The worktree is dirty from adjacent read-surface work. Mitigation: keep edits
  narrow and do not revert unrelated changes.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_read_surface_physical_readback_capture_plan.py tests/test_no_wing_release_check.py tests/test_operator_product_state.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py -q`
- `node --test frontend/src/services/__tests__/operatorAdaptersContract.test.mjs`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root artifacts/no_wing_release_check_current_cycle`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Where the other agent may disagree

Another agent may argue for building hardware capture automation immediately.
That should stay out of this slice because the current evidence gap is contract
clarity, not live or read transport execution.
