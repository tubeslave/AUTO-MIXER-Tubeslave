# Brief: read-surface-fixture-evidence-20260605

## Context

The project goal requires full console-state awareness before broad autonomous
control. The repository already has read-surface inventory rows and a
remediation plan, plus cached Product Layer DTO fixture tests for channels,
processing, buses, matrices, FX, routing/sends, scenes, presets, and dLive.

What is missing is an explicit evidence layer that separates "cached DTO fixture
exists" from "physical readback and parity proof are complete." Without that
distinction, read coverage can look more mature than it is.

## Task

Add a transport-free `read_surface_fixture_evidence` snapshot. It should map
read-surface IDs to known fixture tests, covered contracts, and remaining gaps.
It must not execute tests, query hardware, or mutate console state.

## Safety constraints

- Do not add live OSC, WING, dLive, or Mixing Station writes.
- Do not execute tests from the snapshot builder.
- Do not claim full autonomous readiness from cached DTO fixtures.
- Preserve physical readback fixtures, stale-cache evidence, scene/preset diffs,
  and dLive/WING parity as explicit blockers.

## Acceptance criteria

- The snapshot is read-only and exposes `live_write=False`.
- Every current read surface gets fixture-evidence status.
- Cached DTO fixtures are visible, but physical readback gaps remain blockers.
- Processing registry includes the evidence snapshot as shadow/evaluation only.
- `console_read_parameters` references the fixture evidence layer.
- Focused tests and the standard test command pass.
