# AutoSoundcheck legacy direct writes fail closed

Task id: `auto-soundcheck-legacy-direct-writes-fail-closed-20260605`

## Context

The current project goal keeps the full autonomous mixing target, but live console writes must remain bounded and evidence-driven. The previous AutoFOH safety pass blocked typed non-fader/gain actions, but static scan still shows legacy direct writes inside `backend/auto_soundcheck_engine.py`:

- `set_eq_on`
- `set_compressor_threshold`
- `set_mute`

`set_fader` is also present in restore logic and must be inspected separately because channel fader movement is the only current live-write family with a supervised path.

## Requirement

Convert legacy non-fader/gain AutoSoundcheck direct writes into blocked evidence without mutating the mixer client. Preserve the analyzer decisions, expected values, and test observability.

## Safety Boundary

- Do not widen live write scope.
- Do not add dependencies.
- Do not touch secrets or hardware credentials.
- Keep direct EQ, compressor, and mute automation quarantined until explicit readback, rollback, cooldown, and physical proof exist.

## Validation

- Focused AutoSoundcheck tests for EQ enable, compressor threshold, and restore mute evidence.
- Static scan proving the remaining direct writes in `auto_soundcheck_engine.py` are limited to channel fader restoration.
- Inventory/matrix tests proving the policy is documented.
- Full standard pytest before considering the slice complete.
