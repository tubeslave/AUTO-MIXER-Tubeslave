# Brief: read-surface-artifact-review-queue-release-gate-20260605

## Context

The repository now has per-surface physical readback artifact-intake gates for
raw mixer state, channel controls, processing, bus/group/main/DCA, matrix, FX,
routing/sends, snapshots/scenes, user presets, and dLive state. Each intake gate
is read-only and can only produce manual-review evidence. Without an aggregate
queue in no-WING release evidence, operators must inspect many separate blocks
to understand which artifacts are missing, invalid, unsafe, or ready for manual
review.

## Objective

Make the read-surface physical readback artifact review queue visible and
required in no-WING release evidence after all per-surface intake gates have
run.

## Safety Constraints

- Do not close physical fixture gaps from the queue.
- Do not query WING, dLive, OSC, MIDI, TCP, Dante, or any live console.
- Do not execute tests or artifact capture from the queue.
- Do not grant live authority, live widening, or autonomous readiness.
- Keep invalid or unsafe artifact claims as release-check failures.

## Expected Evidence

- `run_release_checks()` includes
  `read_surface_physical_readback_artifact_review_queue_schema` after
  `dlive_state_physical_readback_artifact_intake_schema` and before backend
  pytest.
- `summary.json` and `summary.md` expose queue safety flags, counts, candidate
  surface IDs, missing surface IDs, and blockers.
- Focused tests cover direct validation, run ordering, summary rendering, and
  normalization of unsafe queue authority claims.
