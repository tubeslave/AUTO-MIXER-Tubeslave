# AutoReverb Apply Fail-Closed Boundary

## Task

Close the legacy `AutoReverb.apply_to_mixer()` direct reverb send/parameter write path while preserving reverb decisions as reviewable evidence for assisted mixing.

## Context

- `backend/auto_reverb.py` computes useful per-channel reverb decisions from instrument profiles, spectral centroid, spectral flux, dry-side margins, predelay, and filtering rules.
- `apply_to_mixer()` historically called `mixer_client.set_reverb_send` and `mixer_client.set_reverb_params`.
- `backend/auto_fader_v2/controller.py` can call this apply method when auto reverb is enabled.
- FX, insert, send, and reverb writes are quarantined; current live-approved execution remains supervised fader/gain only.

## Constraints

- Do not add reverb/FX/send writes to the live executor.
- Do not mutate a mixer client even if one is supplied.
- Preserve reverb decision output and would-be command evidence for operator review.
- Keep useful AutoReverb behavior proposal-only until send/return readback, feedback checks, and rollback proof exist.

## Acceptance

- `AutoReverb.apply_to_mixer()` returns `applied=False`.
- Fake mixer methods are not called.
- Would-be `set_reverb_send` and `set_reverb_params` commands are returned as `blocked_calls`.
- `auto_fader_v2` logs blocked reverb evidence instead of claiming live apply.
- Registry, write-surface inventory, capability matrix, processing inventory, tests, and ADR record the fail-closed boundary.
