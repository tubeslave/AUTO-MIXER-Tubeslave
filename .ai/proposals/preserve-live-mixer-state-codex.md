# Preserve Live Mixer State - Codex Proposal

## 1. Short thesis

Make preservation the default live behavior: snapshot console state first, then apply bounded corrections without resetting channel strips.

## 2. What I understood

The operator may already have useful channel processing on WING Rack. Automatic analysis may improve that processing, but it must not erase it. The backend should be safe to run against a live console.

## 3. Proposed solution

- Extend WING channel state reads to include fader, mute, trim, pan, HPF, EQ, compressor, gate, polarity, and delay.
- Keep `preserve_existing_processing=True` by default.
- Gate destructive `reset_channel_processing()` behind `allow_destructive_reset=True` plus preservation disabled.
- Treat gain and fader moves as relative corrections from the snapshot.
- Preserve pan by default when a snapshot exists.
- For existing HPF, EQ, and compressor settings, move toward analysis targets in bounded steps instead of replacing the whole shape.
- Prevent safety fader bounds from raising parked faders to `-30 dB`.

## 4. Likely files to touch

- `backend/auto_soundcheck_engine.py`
- `backend/wing_client.py`
- `backend/autofoh_safety.py`
- `tests/test_auto_soundcheck_engine.py`
- `tests/test_autofoh_safety.py`

## 5. Alternatives considered

- Keep destructive reset but add a warning: rejected because the request requires reading and preserving live state before actions.
- Add a separate observe-only mode: useful for rehearsal, but it does not solve the live write path.

## 6. Risks

- More OSC read queries increase startup latency by a few seconds on large channel sets.
- Preserving existing processing may reduce how aggressively auto-soundcheck converges.
- Existing state snapshots may be incomplete if the console does not reply before the read window expires.

## 7. Test plan

- Unit-test that reset is skipped by default.
- Unit-test that destructive reset requires explicit opt-in.
- Unit-test input trim correction is current trim plus correction, not an absolute overwrite.
- Unit-test fader safety does not lift parked channels.

## 8. Where the other agent may disagree

An opposing review may prefer an explicit "operator lock" per channel instead of preserving all channels by default. The safer default for live sound is still preservation, with destructive reset available only by explicit opt-in.
