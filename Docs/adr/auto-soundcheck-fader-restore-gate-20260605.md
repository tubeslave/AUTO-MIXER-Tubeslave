# ADR: AutoSoundcheck fader restore gate

## 1. Context

AutoSoundcheck had one remaining direct mixer write after the legacy EQ/compressor/mute quarantine pass: fader restore in `restore_original_settings`. The current canonical live path for fader/gain is Product Layer `apply_operator_proposal` into the supervised manual write gate.

## 2. Options considered

- Leave direct `set_fader` restore in place.
- Route restore through `AutoFOHSafetyController`.
- Record blocked restore evidence and require Product Layer/manual gate for live application.

## 3. Decision

Record AutoSoundcheck fader restore as blocked evidence. Do not mutate the mixer from `restore_original_settings`.

## 4. Why this won

This keeps rollback intent explicit while avoiding a second fader executor. It aligns AutoSoundcheck with the same evidence-first promotion ladder used for EQ, compressor, mute, pan, FX, and routing surfaces.

## 5. Rejected alternatives

Direct restore was rejected because it bypasses the current supervised proof route. `AutoFOHSafetyController` rollback was rejected because it still writes directly to the mixer and does not provide operator approval or Product Layer readback proof.

## 6. Implementation plan

- Replace direct `set_fader` with `_record_blocked_console_write`.
- Update restore tests to expect no mixer writes and exact blocked calls for fader/mute.
- Update inventory and capability matrix language.
- Add static scan coverage for zero direct `self.mixer_client.set_*` calls in AutoSoundcheck.

## 7. Test plan

- Focused restore/static tests.
- Registry and project-goal matrix tests.
- Full standard pytest.

## 8. Risks and rollback

The risk is slower recovery if an operator expects immediate restore. Rollback would reintroduce direct fader mutation, so it should only happen through a separate emergency restore proof path.
