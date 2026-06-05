# Codex Proposal: User Presets Physical Readback Contract

## Short thesis

Add a schema-only physical readback contract for `user_presets` so preset
catalog, target, content fingerprint, scoped diff, and rollback evidence can be
validated before any preset automation is considered.

## What I understood

The system already publishes cached user preset catalog rows, but this is not
enough evidence for scene/preset promotion. Preset recall and store are
destructive operations, so the next step must be read-only proof structure, not
execution.

## Proposed solution

Create `backend/user_presets_physical_readback_contract.py` with constants,
contract builder, and artifact validator. Wire it through the processing
registry, project goal capability matrix, read-surface fixture evidence,
remediation plan, no-WING release summary, Product Layer proof quality, and the
frontend Module Status adapter.

## Likely files to touch

- `backend/user_presets_physical_readback_contract.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `backend/read_surface_fixture_evidence.py`
- `backend/read_surface_remediation_plan.py`
- `backend/operator_product_state.py`
- `tools/no_wing_release_check.py`
- `frontend/src/services/operatorAdapters.ts`
- Related backend/frontend tests and council docs.

## Alternatives considered

Keep presets covered only by the cached DTO. Rejected because cached catalog
metadata cannot prove preset content, scoped diff, rollback evidence, or physical
parity.

## Risks

The contract may look like readiness if surfaced poorly. Mitigation: keep
`physical_fixture_gap_remains=true`, `accepted_as_physical_fixture_candidate=false`
for contract summaries, and force recall/store/live authority to false in
Product Layer and frontend.

## Test plan

Run focused backend tests for the new contract, fixture evidence, remediation,
registry, no-WING, and Product Layer. Run frontend adapter tests and build.
Run no-WING quick validation and full backend pytest before commit.

## Where the other agent may disagree

Kimi may argue for waiting until a real physical preset artifact exists. The
counterpoint is that the schema contract is needed first so the future artifact
has a stable acceptance gate and cannot silently claim live authority.
