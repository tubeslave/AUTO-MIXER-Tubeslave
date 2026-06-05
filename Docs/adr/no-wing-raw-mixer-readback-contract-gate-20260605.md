# ADR: no-wing-raw-mixer-readback-contract-gate-20260605

## Context

The raw mixer-state physical readback contract now defines the expected schema
for the next read-surface proof target. no-WING release checks validate fixture
evidence and remediation alignment, but they do not yet validate the contract as
a first-class schema gate.

## Options considered

1. Add a dedicated no-WING schema gate and summary for the contract.
2. Rely on focused unit tests only.
3. Fold the contract fields into the fixture evidence summary.

## Decision

Use option 1. no-WING release checks validate the raw mixer-state physical
readback contract after read-surface remediation validation and before backend
pytest. The release summary exposes target consistency, physical gap status, and
no-live-authority flags.

## Why this won

It keeps the evidence chain explicit: fixture evidence names the gap,
remediation names the action, and the contract names the future artifact shape.
Each layer can fail independently without implying physical proof or live
authority.

## Rejected alternatives

Unit tests alone do not make release artifacts self-checking. Folding contract
fields into fixture evidence would blur the difference between fixture coverage
and artifact schema readiness.

## Implementation plan

1. Import the raw mixer contract builder and constants into the release check.
2. Add a contract schema validator that cross-checks fixture/remediation targets.
3. Add JSON/Markdown summary rendering.
4. Add tests for pass, drift rejection, run order, and summary output.

## Test plan

- `PYTHONPATH=backend python3 -m py_compile tools/no_wing_release_check.py tests/test_no_wing_release_check.py`
- `PYTHONPATH=backend python3 -m pytest tests/test_no_wing_release_check.py tests/test_raw_mixer_state_physical_readback_contract.py tests/test_read_surface_fixture_evidence.py tests/test_read_surface_remediation_plan.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`
- `python3 tools/memory/memory_lint.py`
- `/opt/homebrew/bin/git diff --check`

## Risks and rollback

Risk: schema readiness could be mistaken for hardware proof. Mitigation: the
summary preserves `physical_fixture_gap_remains=true`,
`accepted_as_physical_fixture_candidate=false`,
`live_authority_granted=false`, and `full_autonomous_ready=false`.

Rollback is removing the validator, summary block, tests, and this ADR. No live
transport path is added.
