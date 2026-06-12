# Brief: WING Autoloop Current Handoff Gates Docs

## Context

The first WING MVP blocker selected action now requires
`mvp_blocker_anchor.packet_does_not_close_blocker: true` and the final
`operator_sequence_supervised_prestart_gate_step` marker set. The autoloop
prompt and README still documented the previous gate list.

## Task

Synchronize `tools/codex_autoloop/codex_cycle_prompt.md` and
`tools/codex_autoloop/README.md` with the current WING selected-action gates.

## Safety Boundary

This is documentation and test synchronization only. It does not contact WING,
does not open a backend/WING session, and does not grant live write authority.
