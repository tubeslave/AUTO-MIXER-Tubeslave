# snapshots-scenes-physical-readback-artifact-intake-20260605

## Context

Snapshots/scenes already has a physical readback schema contract, but the
project also needs a read-only intake surface for separately operator-captured
JSON artifacts. Scene recall/save operations are destructive, so an intake row
must never look like permission to load, save, recall, or close a physical
fixture gap automatically.

The current worktree contains a snapshots/scenes artifact-intake slice wired
through registry, Product Layer proof quality, no-WING release summaries, and
frontend module status rows.

## Goal

Finish the slice as manual-review evidence only:

- accept valid operator-captured artifacts as review candidates;
- reject invalid or mixed valid+invalid artifact sets;
- preserve `fixture_closure_allowed_by_intake=false`;
- preserve `scene_load_allowed=false` and `scene_save_allowed=false`;
- keep live authority and Full Autonomous readiness false.

## Non-goals

- Do not contact WING or dLive hardware.
- Do not load, save, recall, or mutate snapshots/scenes.
- Do not close physical fixture gaps from intake alone.
- Do not add a new executor or live-write path.

## Safety constraints

- Intake must remain transport-free and read-only.
- Any source claiming scene load/save authority must fail validation or be
  normalized back to false in summaries.
- Mixed candidate sets must fail closed when any invalid artifact is present.
- Frontend rows must render the intake as read-only status evidence.

## Expected evidence

- Focused backend tests for snapshots/scenes intake and no-WING release checks.
- Frontend adapter contract test for the read-only module row.
- Memory lint and diff hygiene.
