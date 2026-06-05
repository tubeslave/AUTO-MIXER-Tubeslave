# ADR: read-surface-fixture-drift-guard-20260605

## Context

`backend/read_surface_fixture_evidence.py` is a declarative read-only map from
console read surfaces to cached DTO fixture tests. It intentionally does not
execute tests or contact hardware. The known risk is drift: a fixture test can
be renamed while the evidence map still claims it exists.

## Options considered

1. Keep review-only discipline for fixture map updates.
2. Add a focused pytest guard for declared fixture files and test names.
3. Add runtime validation to the Product Layer snapshot builder.

## Decision

Use option 2. Add a pytest that parses declared fixture files with `ast`, fails
when a declared pytest function is missing, and runs inside the no-WING release
check before proof reports are generated. The no-WING summary should also
publish the read-only fixture evidence counts and remaining physical readback
fixture gaps.

## Why this won

The evidence snapshot remains side-effect free and transport-free, while CI can
catch stale evidence before it reaches readiness reports.

## Rejected alternatives

Review-only discipline is too easy to miss. Runtime validation would make the
Product Layer snapshot inspect test files, which is not appropriate for the
application path.

## Implementation plan

1. Build the read-surface fixture evidence snapshot in the existing focused test
   module.
2. For every non-missing row, require at least one `tests/...` fixture path and
   at least one declared test name.
3. Parse each fixture file with Python `ast`.
4. Fail if any declared test name is absent.
5. Keep missing-evidence rows explicit with empty fixture and test lists.
6. Add the focused guard to `tools/no_wing_release_check.py`.
7. Add a read-surface fixture evidence section to no-WING summary JSON/Markdown.

## Test plan

- `PYTHONPATH=backend python3 -m pytest tests/test_read_surface_fixture_evidence.py -q`
- `PYTHONPATH=backend python3 -m pytest tests/test_read_surface_fixture_evidence.py tests/test_read_surface_inventory.py tests/test_processing_registry.py -q`
- `python3 tools/no_wing_release_check.py --quick --skip-frontend --artifact-root /tmp/automixer-no-wing-release-check-read-surface --command-timeout-sec 300`
- `PYTHONPATH=backend python3 -m pytest tests/ -x --tb=short -q`
- `python3 tools/memory/memory_lint.py`

## Risks and rollback

Risk: the test fails after legitimate fixture renames until the declarative map
is updated. That is acceptable because stale evidence is worse than a focused
test failure. Rollback is removing the guard test and this ADR. No live OSC/WING
write path is added.
