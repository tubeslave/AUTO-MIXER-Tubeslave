# Preserve Live Mixer State

## Task

Before automatic soundcheck or monitoring writes to a live console, read the current channel state and treat it as the operator baseline.

## Context

The WING Rack is connected and can receive OSC commands. The previous auto-soundcheck path reset channel processing before analysis, which can erase EQ, HPF, dynamics, polarity, and delay that are already set on the console.

## Safety Requirements

- Read each selected channel's fader, mute, trim, pan, HPF, EQ, dynamics, gate, polarity, and delay before applying corrections.
- Preserve existing processing by default.
- Destructive reset must require explicit operator opt-in.
- Apply gain, fader, HPF, EQ, compressor, pan, and phase corrections as bounded changes relative to the snapshot.
- Never lift parked faders during safety bounding.

## Test Command

`PYTHONPATH=backend python -m pytest tests/test_auto_soundcheck_engine.py tests/test_autofoh_safety.py -q`
