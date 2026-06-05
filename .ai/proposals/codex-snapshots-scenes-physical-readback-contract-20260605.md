# Codex Proposal: Snapshots/Scenes Physical Readback Contract

## Short thesis

Add a shadow-only physical readback contract for snapshots/scenes so scene
automation cannot advance from cached catalog data alone.

## What I understood

The current Product Layer can show cached scene names and active-scene markers,
but it cannot prove scene contents, scoped diffs, rollback evidence, or physical
readback parity. Scene load/save remains destructive and must stay blocked.

## Proposed solution

- Add `backend/snapshots_scenes_physical_readback_contract.py` with required
  active-scene fields, snapshot row fields, coverage flags, physical sources,
  forbidden mutation evidence, and false safety flags for scene load/save.
- Register the contract as `shadow_only` and include it in the console-read goal.
- Add it to read-surface fixture evidence, remediation expected artifacts,
  no-WING schema validation/summary, Product Layer proof quality, and frontend
  Module Status.
- Add tests for valid future artifacts, cached/live claim rejection, registry
  shadow-only behavior, no-WING ordering, and Product Layer/frontend visibility.

## Likely files to touch

- `backend/snapshots_scenes_physical_readback_contract.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `backend/read_surface_fixture_evidence.py`
- `backend/read_surface_remediation_plan.py`
- `backend/operator_product_state.py`
- `tools/no_wing_release_check.py`
- `frontend/src/services/operatorAdapters.ts`
- Focused tests and council docs.

## Alternatives considered

Trust cached scene catalog rows. Rejected because names and active markers do not
prove scene contents, diffs, rollback, or physical console parity.

Add live scene query/recall support now. Rejected for this slice because any
scene transport work needs its own operator-supervised safety proof.

## Risks

- Contract visibility could be mistaken for scene automation readiness.
- Summary artifacts could claim scene load/save authority.
- Future scene artifacts could omit rollback/diff evidence unless this schema
  names those fields now.

## Test plan

- Compile touched Python modules.
- Run focused backend tests for the contract, registry, fixture evidence,
  remediation plan, no-WING, and Product Layer proof quality.
- Run frontend adapter tests and build.
- Run no-WING quick check, memory lint, diff check, and the standard backend
  pytest command before commit.

## Where the other agent may disagree

Kimi may prefer a combined scene/preset contract. That is reasonable later, but
separate surface contracts match the current read-surface inventory and keep
scene recall safety easier to audit.
