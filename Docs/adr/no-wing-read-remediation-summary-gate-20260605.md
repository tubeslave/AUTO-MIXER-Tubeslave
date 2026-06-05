# ADR: no-wing-read-remediation-summary-gate-20260605

## Context

`read_surface_fixture_evidence` identifies the next physical readback target.
`read_surface_remediation_plan` turns that target into an action. The no-WING
release summary validated fixture evidence but did not validate the remediation
plan/action alignment.

## Options considered

1. Add a read-surface remediation schema gate to no-WING release checks.
2. Keep remediation validation only in unit tests.
3. Merge remediation fields into fixture evidence validation.

## Decision

Use option 1. Add a dedicated `read_surface_remediation_plan_schema` validation
and summary block. The gate also requires the plan to have exactly one
`next_physical_readback_target` action while physical fixture gaps remain.

## Why this won

It keeps proof coverage and remediation planning separate while making release
artifacts self-checking.

## Rejected alternatives

Unit tests alone do not prove release artifacts are current. Merging summaries
would blur fixture evidence and action planning.

## Implementation plan

1. Build fixture evidence and remediation plan inside the no-WING schema gate.
2. Validate read-only safety flags and next target/action alignment.
3. Validate that only the expected remediation action is flagged as the next
   physical readback target.
4. Embed compact remediation summary in JSON and Markdown.
5. Add tests for pass, mismatch rejection, duplicate target flags, and summary output.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_no_wing_release_check.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_no_wing_release_check.py tests/test_read_surface_remediation_plan.py tests/test_processing_registry.py -q`
- `python3 tools/memory/memory_lint.py`

## Risks and rollback

Risk: remediation data may be interpreted as executable. The summary reports
`read_only=true`, `live_write=false`, `mutates_console_state=false`, and
`full_autonomous_ready=false`.

Rollback is removing the schema gate, summary block, tests, and this ADR.
