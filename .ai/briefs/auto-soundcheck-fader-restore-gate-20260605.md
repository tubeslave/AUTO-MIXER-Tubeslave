# AutoSoundcheck fader restore gate

Task id: `auto-soundcheck-fader-restore-gate-20260605`

## Context

After blocking AutoSoundcheck legacy EQ, compressor-threshold, and mute writes, static scan still shows one direct mixer write in `backend/auto_soundcheck_engine.py`: `self.mixer_client.set_fader(...)` inside `restore_original_settings`.

Channel fader is a current live-write kind, but the approved route is still the Product Layer supervised gate: `apply_operator_proposal -> _apply_manual_console_write`. AutoSoundcheck restore should not bypass that route.

## Requirement

Convert AutoSoundcheck fader restore into blocked restore evidence. Preserve the target fader value and channel identity so the operator/Product Layer can apply it later through the supervised path.

## Safety Boundary

- No direct AutoSoundcheck mixer writes.
- No new live write authority.
- Keep fader/gain live scope tied to Product Layer/operator approval only.

## Validation

- Static scan shows no `self.mixer_client.set_*` calls in `auto_soundcheck_engine.py`.
- Restore test asserts fader and mute restore are evidence-only.
- Inventory/matrix docs state AutoSoundcheck restore evidence must go through the supervised manual gate before promotion.
- Full standard pytest.
