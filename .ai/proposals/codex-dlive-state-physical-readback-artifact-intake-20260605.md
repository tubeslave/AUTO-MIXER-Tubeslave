# Codex Proposal: dlive-state-physical-readback-artifact-intake-20260605

## Short thesis

Expose dLive-state physical readback artifacts as read-only manual-review
candidates, and make invalid or authority-claiming artifacts fail closed.

## What I understood

The project needs a path to inspect future operator-captured dLive state and
WING parity artifacts, but dLive MIDI/TCP sends and dLive writes are live
control surfaces. The intake can help reviewers see artifact quality; it cannot
close fixture gaps or grant dLive authority.

## Proposed solution

- Keep `backend/dlive_state_physical_readback_artifact_intake.py`
  transport-free.
- Register the method as `shadow_only` and non-executor in the processing
  registry and capability matrix.
- Add no-WING validation and summary output.
- Surface normalized Product Layer and frontend module status rows.
- Treat any invalid artifact in a candidate set as a failed intake.

## Likely files to touch

- `backend/dlive_state_physical_readback_artifact_intake.py`
- `backend/processing_registry.py`
- `backend/operator_product_state.py`
- `backend/project_goal_capability_matrix.py`
- `tools/no_wing_release_check.py`
- `frontend/src/services/operatorAdapters.ts`
- `tests/test_dlive_state_physical_readback_artifact_intake.py`
- `tests/test_no_wing_release_check.py`
- `tests/test_operator_product_state.py`
- `frontend/src/services/__tests__/operatorAdaptersContract.test.mjs`

## Alternatives considered

- Build a dLive capture runner now. Rejected: that requires separate
  operator-approved hardware procedure, passive capture design, rollback
  proof, and emergency-stop handling.
- Treat any valid artifact as fixture closure. Rejected: manual review,
  freshness proof, WING parity review, and physical fixture governance are
  still required.
- Keep only the schema contract. Rejected: operators need normalized artifact
  intake visibility once captured evidence appears.

## Risks

- A valid-looking JSON artifact could be mistaken for dLive readiness.
  Mitigation: backend, release summary, Product Layer, and frontend rows
  preserve fixture gap, live authority denial, and MIDI/TCP/dLive write denial.
- A mixed artifact directory could hide a bad artifact next to a good one.
  Mitigation: invalid artifacts dominate status and prevent candidate
  acceptance.
- dLive parity could be overgeneralized to WING. Mitigation: artifact intake
  remains manual-review evidence; parity gaps remain explicit.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_dlive_state_physical_readback_artifact_intake.py tests/test_no_wing_release_check.py tests/test_operator_product_state.py -q`
- `node --test frontend/src/services/__tests__/operatorAdaptersContract.test.mjs`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Where the other agent may disagree

Another agent may want to promote valid dLive artifacts directly into dLive or
WING parity readiness. That should stay blocked: intake only creates a
manual-review candidate, not live authority or fixture closure.
