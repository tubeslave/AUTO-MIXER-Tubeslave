# Preserve Live Mixer State

## 1. Context

AUTO-MIXER can send OSC commands directly to a live WING Rack. The previous auto-soundcheck flow reset eligible channel processing before analysis. That is unsafe when an operator has already set EQ, HPF, dynamics, polarity, delay, trim, pan, or fader positions.

## 2. Options considered

- Preserve console state by default and apply bounded corrections.
- Keep reset-by-default and rely on operator caution.
- Run observe-only by default and require a separate apply step.

## 3. Decision

Preserve existing channel processing by default. Read channel state before any automatic correction, and make destructive reset an explicit opt-in path.

## 4. Why this won

This matches live-sound safety: existing operator work is the baseline, not disposable state. It also reduces the blast radius of a wrong source classification or an incomplete analysis pass.

## 5. Rejected alternatives

Reset-by-default was rejected because it can erase live channel processing. Observe-only by default was rejected because the operator asked for the backend to keep applying corrections, not merely observe.

## 6. Implementation plan

- Extend WING state reads for fader, mute, trim, pan, HPF, EQ, compressor, gate, polarity, and delay.
- Preserve current processing in `_reset_channels()` unless preservation is disabled and destructive reset is explicitly allowed.
- Apply input trim and fader corrections relative to the current snapshot.
- Preserve pan when a snapshot exists.
- Move existing HPF, EQ, and compressor settings toward analysis targets in bounded steps.
- Prevent safety fader bounds from raising parked faders.

## 7. Test plan

Run:

`PYTHONPATH=backend python -m pytest tests/test_auto_soundcheck_engine.py tests/test_autofoh_safety.py -q`

Then, before a live run, restart the backend so the patched preservation behavior is loaded.

## 8. Risks and rollback

Startup reads are heavier because each selected WING channel queries more OSC addresses. If a read is incomplete, the safety layer still bounds outgoing moves, but some controls may use conservative defaults. Rollback is to stop the auto engine and restore the previous code path, but destructive reset should remain disabled for live use.
