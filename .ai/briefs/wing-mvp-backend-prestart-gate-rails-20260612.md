# WING MVP Backend Prestart Gate Rails Brief

## Task

Promote the structured backend-port prestart gate from the read-only runbook and
handoff validator into the top-level WING MVP rails.

## Constraints

- No WING hardware contact.
- No backend WING WebSocket session.
- No OSC/WING writes.
- Do not stop or reconfigure local listener processes.
- Do not close MVP blockers from documentation or offline evidence.

## Expected Output

- `Docs/MVP_WING_SUPERVISED.md` requires
  `backend_port_prestart_gate == expected_backend_port_prestart_gate` before
  any supervised fader/gain proof.
- `tools/codex_autoloop/codex_cycle_prompt.md` tells autonomous cycles to check
  the structured backend-port prestart gate before supervised proof work.
- `Docs/INDEX.md` links the structured prestart gate report from current WING
  MVP rails.
- Rails tests enforce the new source-of-truth wording.
