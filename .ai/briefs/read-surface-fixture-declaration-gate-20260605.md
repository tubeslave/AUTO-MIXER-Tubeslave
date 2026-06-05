# read-surface-fixture-declaration-gate-20260605

## Context

`read_surface_fixture_evidence` declares fixture paths and pytest function names
that support cached read DTO coverage. A release report should not trust those
declarations unless the named files and tests still exist.

## Goal

Make the no-WING read-surface fixture schema fail closed when a fixture path is
missing, points outside `tests/`, or names a nonexistent pytest function.

## Non-goals

- Do not execute fixture tests from the schema validator.
- Do not query mixer hardware.
- Do not mark physical readback proof complete.
- Do not widen live write authority.

## Safety constraints

- The validator may read test source files but must remain non-mutating.
- Missing declaration evidence should keep Full Autonomous blocked.
- `missing_fixture_evidence` rows should still be allowed only with empty
  fixture/test declarations.

## Expected evidence

- no-WING schema validates fixture paths and declared test names.
- Corrupted declarations fail with explicit violation IDs.
- Existing read-only/no-live-write release checks continue to pass.
