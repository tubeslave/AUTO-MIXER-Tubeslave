# AutoPanner Apply Fail-Closed Boundary

## Task

Close the legacy AutoPanner and controller pan-separation direct write paths while preserving pan decisions as reviewable evidence for assisted mixing.

## Context

- `backend/auto_panner.py` computes useful pan decisions from instrument role, spectral centroid, dominant band, and symmetric distribution rules.
- `AutoPanner.apply_to_mixer()` historically called `mixer_client.set_channel_pan`.
- `backend/auto_fader_v2/controller.py` also had a simple pan-separation branch that directly called `set_channel_pan`.
- Pan writes are part of the quarantined pan/phase/delay surface; current live-approved execution remains supervised fader/gain only.

## Constraints

- Do not add pan writes to the live executor.
- Do not mutate a mixer client even if one is supplied.
- Preserve pan decision output and would-be command evidence for operator review.
- Keep panning proposal-only until current-pan readback, mono compatibility, and rollback proof exist.

## Acceptance

- `AutoPanner.apply_to_mixer()` returns `applied=False`.
- Controller pan separation returns blocked evidence instead of calling `set_channel_pan`.
- Fake mixer methods are not called.
- `set_channel_pan` is included in static write-token coverage and write-surface inventory.
- Registry, capability matrix, processing inventory, tests, and ADR record the fail-closed boundary.
