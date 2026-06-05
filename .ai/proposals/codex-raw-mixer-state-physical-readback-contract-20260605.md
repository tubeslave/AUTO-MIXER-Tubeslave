# Proposal: raw-mixer-state-physical-readback-contract-20260605

## 1. Short thesis

Make the next raw mixer-state physical readback action concrete by defining the
artifact contract now, while keeping proof and live authority separate.

## 2. What I understood

The read-surface remediation summary points operators at
`raw_mixer_state_cache`, but the repository does not yet state what a future
physical readback artifact must contain. Without a schema, the target is visible
but hard to verify consistently.

## 3. Proposed solution

Add `raw_mixer_state_physical_readback_contract` as a shadow/evaluation method.
It declares required top-level fields, per-address row fields, accepted physical
sources, safety flags, and forbidden mutation evidence. Add a validator for
future artifacts that accepts physical readback rows and rejects target drift,
cached sources, live-write flags, mutation records, and Full Autonomous claims.

## 4. Likely files to touch

- `backend/raw_mixer_state_physical_readback_contract.py`
- `backend/processing_registry.py`
- `backend/project_goal_capability_matrix.py`
- `backend/read_surface_fixture_evidence.py`
- `backend/read_surface_remediation_plan.py`
- `tools/no_wing_release_check.py`
- `tests/test_raw_mixer_state_physical_readback_contract.py`
- `tests/test_read_surface_fixture_evidence.py`
- `tests/test_read_surface_remediation_plan.py`
- `tests/test_processing_registry.py`
- `tests/test_no_wing_release_check.py`

## 5. Alternatives considered

- Leave the artifact shape implicit. Rejected because remediation action IDs
  should map to concrete evidence.
- Mark the contract as a physical fixture. Rejected because a schema is not
  readback proof.
- Put validation in the no-WING runner only. Rejected because Product Layer and
  registry snapshots need the same read-only contract.

## 6. Risks

Operators may mistake a valid schema for completed hardware proof. Mitigation:
the contract status keeps `physical_fixture_gap_remains=true`, fixture evidence
does not add `physical_readback_fixture`, and every safety payload keeps live
authority disabled.

## 7. Test plan

- `PYTHONPATH=backend python3 -m py_compile backend/raw_mixer_state_physical_readback_contract.py tests/test_raw_mixer_state_physical_readback_contract.py`
- `PYTHONPATH=backend python3 -m pytest tests/test_raw_mixer_state_physical_readback_contract.py tests/test_read_surface_fixture_evidence.py tests/test_read_surface_remediation_plan.py tests/test_processing_registry.py tests/test_project_goal_capability_matrix.py tests/test_no_wing_release_check.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## 8. Where the other agent may disagree

Kimi may prefer waiting until a physical runner exists. I would still land the
contract first because it prevents future artifacts from inventing incompatible
fields or smuggling live readiness claims.
