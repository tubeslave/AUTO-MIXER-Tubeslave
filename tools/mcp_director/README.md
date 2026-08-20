# Automixer WING Lab MCP

Local stdio MCP server for pipeline inspection and bounded WING Rack testing.
It contains no remote HTTP listener, arbitrary command execution, Git mutation,
Telegram integration, tokens, routing, scene/snapshot, or mute tools.

## Enable

The backend is fail-closed unless the MCP process has:

```bash
AUTOMIXER_WING_LAB_AUTOWRITE=1 node tools/mcp_director/server.js
```

`./tools/mcp_director/install_mcp_director.sh` validates the server and prints a
Codex-compatible local configuration. It does not edit global configuration.

## Session flow

1. `wing_lab_status`
2. `wing_lab_arm` once with the exact `WING Rack` model and endpoint
3. `wing_lab_apply_decisions` as the Automixer produces accepted corrections
4. `wing_lab_trace` to inspect every prepared/write/readback event
5. `wing_lab_stop` to restore all preimages and disconnect

After the one arm call, accepted decisions do not ask for individual approval.
The session is capped by time and correction count. A readback mismatch triggers
reverse rollback and emergency stop. The lab keeps the per-user global console
writer lock for the whole session.

Supported correction families are channel/bus/master faders, input trim, EQ,
gate, compressor, an already loaded DS902 de-esser, pan, phase, delay, FX values,
and channel send levels. Routing, scenes/snapshots/presets, mute and unmute are
not exposed.

This profile does not modify production `main_live_assist` authority. Run real
hardware testing only while attended, with PA/monitor outputs physically
contained and the WING Rack visible to the operator.
