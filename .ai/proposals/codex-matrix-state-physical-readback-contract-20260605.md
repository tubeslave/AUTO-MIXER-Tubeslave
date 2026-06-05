# Codex Proposal: Matrix State Physical Readback Contract

## Short thesis

Add a schema-only matrix-state physical readback contract and expose it through
existing proof surfaces without querying hardware or widening live writes.

## What I understood

The project goal requires complete console awareness before autonomous matrix,
routing, FX-send, and master-bus decisions are credible. Current `matrix_state`
DTOs expose cached state, but cached evidence is not enough for physical proof.

## Proposed solution

Define `matrix_state_physical_readback_contract` with required matrix controls,
8-band EQ, dynamics, insert, metering, coverage, source, and safety fields.
Wire the contract into registry snapshots, project-goal capability rows,
read-surface fixture evidence, remediation expected artifacts, no-WING release
summary validation, Product Layer proof-quality normalization, and frontend
module status rows.

## Likely files to touch

- `backend/matrix_state_physical_readback_contract.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `backend/read_surface_fixture_evidence.py`
- `backend/read_surface_remediation_plan.py`
- `backend/operator_product_state.py`
- `tools/no_wing_release_check.py`
- `frontend/src/services/operatorAdapters.ts`
- focused backend/frontend tests

## Alternatives considered

- Reuse bus/main/DCA contract for matrices. Rejected because matrix EQ,
  dynamics, inserts, and matrix-specific promotion blockers need their own
  schema.
- Wait for hardware readback runner first. Rejected because the runner needs a
  stable artifact contract before captured evidence can be trusted.

## Risks

- Operators could mistake contract readiness for completed physical proof.
  Mitigation: every surface preserves `physical_fixture_gap_remains=true`,
  `live_authority_granted=false`, and `full_autonomous_ready=false`.
- Schema may miss future matrix send evidence. Mitigation: this contract covers
  matrix output state; routing/matrix-send proof remains a separate
  `routing_and_sends` surface until promoted with diff/rollback evidence.

## Test plan

- Validate artifact accepts only physical readback sources/statuses.
- Reject cached/live/full-autonomous claims.
- Verify registry method is `shadow_only` and excluded from live executors.
- Verify no-WING summaries and frontend rows surface the new gap.

## Where the other agent may disagree

Kimi may prefer bundling matrix sends into this contract. The safer split is to
keep matrix output state here and leave send/routing diffs to
`routing_and_sends`, because send writes can re-route the console.
