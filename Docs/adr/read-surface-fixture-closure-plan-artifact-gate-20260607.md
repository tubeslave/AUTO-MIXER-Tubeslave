# ADR: Read-Surface Fixture Closure Plan Artifact Gate

## Context

The read-surface fixture closure plan converts accepted operator review records
into manual closure candidates. The existing release check validated the
in-memory plan, but no-WING did not generate and validate a closure-plan
artifact file for operator handoff.

## Options considered

1. Keep the in-memory schema check only.
2. Let acceptance-ledger summaries stand in for closure planning.
3. Generate a closure-plan artifact and validate it inside no-WING.

## Decision

Use option 3. no-WING now writes
`read_surface_physical_readback_fixture_closure_plan.json` and validates the
generated artifact immediately after the builder command succeeds.

## Why this won

Fixture closure is the step where accepted physical-readback evidence could be
mistaken for permission to update baselines. A generated artifact gate makes the
boundary explicit: the plan can expose manual candidates, but fixture update,
fixture closure, live widening, live authority, and full autonomous readiness
remain false.

## Rejected alternatives

- In-memory validation alone does not prove the release run produced a safe
  operator artifact.
- Acceptance-ledger summaries do not describe the separate fixture-closure
  decision.
- Automatic fixture closure is unsafe and outside this no-WING gate.

## Implementation plan

1. Add an offline closure-plan builder with Markdown and JSON output.
2. Add the builder to no-WING release checks.
3. Validate the generated artifact for manual-only safety semantics.
4. Add JSON/Markdown summary output for the artifact gate.
5. Cover valid, missing, unsafe, ordering, and summary cases in tests.

## Test plan

- `python3 -m py_compile tools/no_wing_release_check.py tools/build_read_surface_physical_readback_capture_packet.py tools/build_read_surface_physical_readback_fixture_closure_plan.py`
- Focused no-WING pytest for closure-plan artifact validation.
- Broader read-surface/no-WING pytest.
- no-WING quick run.
- Memory lint and diff check.
- Full backend pytest.

## Risks and rollback

This gate does not close fixtures or prove live readiness. Operators still need
real physical artifacts, acceptance records, and a separate manual fixture
baseline update before any readiness promotion. Rollback is to remove the
builder, validator, tests, and docs; no mixer transport path is affected.
