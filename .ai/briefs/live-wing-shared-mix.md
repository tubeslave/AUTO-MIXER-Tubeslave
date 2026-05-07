# Live WING Shared Mix Soundcheck

## Task

Port the previous offline folder-mixing workflow into the live WING Rack
soundcheck path. The live engine should read the console state first, preserve
existing processing, use the same anchor-order mix analysis, evaluate actions
with the configured perceptual/MERT path, and apply only bounded safe OSC
corrections.

## Constraints

- Live sound safety wins over completeness.
- Read WING channel state before deciding actions.
- Preserve existing EQ, HPF, dynamics, faders, routing, and manual intent.
- Exclude Dante master-reference channels 23/24 from source correction.
- Treat master spectrum as a balance meter; fix sources first.
- Never raise Main above its current value or above 0 dB.
- Routing and naming writes need an expected patch/name map before enabling.

## Test Gate

`PYTHONPATH=backend ./backend/venv/bin/python -m pytest tests/ -x --tb=short -q`
