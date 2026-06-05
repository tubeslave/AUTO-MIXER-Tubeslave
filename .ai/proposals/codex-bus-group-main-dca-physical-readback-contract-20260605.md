# Proposal: bus-group-main-dca-physical-readback-contract-20260605

## 1. Short thesis

The bus/group/main/DCA read surface needs its own physical readback artifact
contract before grouped balance, DCA, or main-bus promotion evidence can be
trusted.

## 2. What I understood

The current Product Layer already exposes cached `bus_main_dca_snapshot` rows
and proposal-only bus/group balance logic. That is useful planning evidence,
but it is not physical proof. Future promotion needs a schema that separates
cached DTO parity from actual console readback across buses, mains, and DCAs.

## 3. Proposed solution

Add a new transport-free backend contract module that defines and validates a
future operator-run bus/group/main/DCA physical readback artifact. Register it
as `shadow_only`, include it in the console-read goal, declare it in fixture and
remediation summaries, validate it in no-WING release checks, and expose the
normalized status through Product Layer/frontend module-status rows.

## 4. Likely files to touch

- `backend/bus_group_main_dca_physical_readback_contract.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `backend/read_surface_fixture_evidence.py`
- `backend/read_surface_remediation_plan.py`
- `backend/operator_product_state.py`
- `tools/no_wing_release_check.py`
- focused backend/frontend tests
- `.ai/memory/project.md`

## 5. Alternatives considered

- Reuse channel control contracts. Rejected because bus/main/DCA proof needs
  bus assignments, DCA membership, main-send state, and metering/headroom
  fields.
- Wait for real hardware proof first. Rejected because without a contract, a
  future artifact can drift or imply unsafe authority.
- Hide the contract in docs only. Rejected because release and Product Layer
  summaries need machine-readable proof metadata.

## 6. Risks

Operators may mistake schema readiness for completed readback proof. The
mitigation is to keep `physical_fixture_gap_remains=true`,
`accepted_as_physical_fixture_candidate=false`, `live_authority_granted=false`,
and `full_autonomous_ready=false` across summaries and UI adapters.

## 7. Test plan

- `PYTHONPATH=backend python3 -m py_compile backend/bus_group_main_dca_physical_readback_contract.py tools/no_wing_release_check.py`
- `PYTHONPATH=backend python3 -m pytest tests/test_bus_group_main_dca_physical_readback_contract.py tests/test_read_surface_fixture_evidence.py tests/test_read_surface_remediation_plan.py tests/test_processing_registry.py tests/test_no_wing_release_check.py tests/test_operator_product_state.py -q`
- `npm --prefix frontend test -- --watch=false`
- `npm --prefix frontend run build`
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## 8. Where the other agent may disagree

Kimi may prefer a generic physical readback contract framework shared by all
surfaces. I would keep this surface-specific for now because bus/main/DCA proof
needs different fields from raw state, channels, and processing.
