# Proposal: channel-processing-physical-readback-contract-20260605

## 1. Short thesis

The channel EQ/dynamics/gate read surface needs its own physical readback
artifact contract before EQ, compressor, gate, de-esser, or unmasking promotion
evidence can be trusted.

## 2. What I understood

The read-surface pipeline already tracks cached DTO fixtures plus raw
mixer-state and channel basic-control readback contracts. The next
promotion-critical surface is `channel_eq_dynamics_gate`, because it supplies
the observed HPF, EQ, compressor, gate, and coverage state used by future
processing write families.

## 3. Proposed solution

Add a new transport-free backend contract module that defines and validates a
future operator-run channel processing physical readback artifact. Register it
as `shadow_only`, include it in the console-read goal, declare it in fixture and
remediation summaries, validate it in no-WING release checks, and expose the
normalized status through Product Layer/frontend module-status rows.

## 4. Likely files to touch

- `backend/channel_processing_physical_readback_contract.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `backend/read_surface_fixture_evidence.py`
- `backend/read_surface_remediation_plan.py`
- `backend/operator_product_state.py`
- `tools/no_wing_release_check.py`
- focused backend/frontend tests
- `.ai/memory/project.md`

## 5. Alternatives considered

- Reuse the channel basic-control contract. Rejected because processing proof
  needs structured HPF, EQ-band, compressor, gate, and coverage fields.
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

- `PYTHONPATH=backend python3 -m py_compile backend/channel_processing_physical_readback_contract.py tools/no_wing_release_check.py`
- `PYTHONPATH=backend python3 -m pytest tests/test_channel_processing_physical_readback_contract.py tests/test_read_surface_fixture_evidence.py tests/test_read_surface_remediation_plan.py tests/test_processing_registry.py tests/test_no_wing_release_check.py tests/test_operator_product_state.py -q`
- `npm --prefix frontend test -- --watch=false`
- `npm --prefix frontend run build`
- `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick --skip-frontend`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## 8. Where the other agent may disagree

Kimi may prefer a generic channel physical readback framework instead of a
surface-specific module. I would keep this surface-specific for now because the
processing field shape is promotion-critical and materially different from
basic controls.
