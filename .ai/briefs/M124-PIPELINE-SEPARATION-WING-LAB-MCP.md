# Brief — M124 pipeline separation and WING lab MCP

## Goal

Separate supported pipeline entrypoints from retained experiments and provide a
local WING Rack test MCP that can apply accepted Automixer decisions after one
bounded lab-session arm.

## Boundaries

- Production `main_live_assist` safety and approval policy stays unchanged.
- No public HTTP MCP, arbitrary commands, Git mutation, tokens or messaging.
- No routing, scenes/snapshots/presets, mute or unmute.
- Lab writes require fresh preimage and post-write readback, durable trace,
  global writer exclusion, reverse rollback and emergency stop.
- De-esser parameters require an already loaded DS902 model.
- Real hardware execution is attended work and is not part of software tests.
