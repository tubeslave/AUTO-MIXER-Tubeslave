# ADR — M124 pipeline separation and WING lab MCP

## 1. Context

Supported and experimental entrypoints coexist under `tools/`, including
several duplicate WING soundcheck generations. The legacy MCP was quarantined
because it mixed arbitrary process execution, Git operations and token-bearing
messaging. A local, attended WING Rack algorithm test needs a narrower surface.

## 2. Options considered

1. Physically move every script and update all callers.
2. Restore the legacy Director and remove its write confirmations.
3. Add a validated catalog and a new fixed-command, opt-in lab MCP.

## 3. Decision

Choose option 3. Existing entrypoint paths remain stable. The catalog identifies
working, experimental and duplicate families. The MCP is stdio-only and starts
one fixed Python bridge. One explicit arm grants a bounded lab session; accepted
decisions then require no individual confirmation.

## 4. Why this won

It makes the separation machine-checkable without changing runtime imports and
creates the smallest real write path needed for algorithm observation while
keeping production policy intact.

## 5. Rejected alternatives

- No filesystem reshuffle in this milestone.
- No public/remote MCP gateway.
- No caller-selected commands, Git operations, tokens or notifications.
- No mute, routing, scenes, snapshots or presets.
- No de-esser model replacement; only parameters of a read-back DS902 instance.

## 6. Implementation plan

1. Add entrypoint catalog and validation.
2. Add opt-in lab session with exact mappings, bounds and shared writer lock.
3. Add durable hash-chained trace, readback and reverse rollback.
4. Add fixed-command JSON-lines bridge and local MCP framing.
5. Validate with fake transport and no-WING tests.

## 7. Test plan

- Working/experimental groups are disjoint and paths exist.
- Excluded families fail before transport.
- One arm permits multiple accepted corrections without confirmation.
- Every correction has prepared/write/verified events.
- Readback mismatch rolls prior writes back and emergency-stops.
- MCP smoke works while autowrite is disabled and contacts no console.
- Full backend and no-WING gates pass in a clean tree.

## 8. Risks and rollback

UDP readback does not authenticate the physical device; the arm requires the
operator to identify the WING Rack and the transport pins replies to its peer.
Stop testing on any identity, readback or physical mismatch. Rollback is removal
of the isolated lab module/MCP and catalog; production runtime is not modified.
