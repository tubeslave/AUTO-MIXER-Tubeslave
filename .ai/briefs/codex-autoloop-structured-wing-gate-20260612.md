# Codex Autoloop Structured WING Gate Brief

## Task

Update Codex autoloop guidance so autonomous cycles require the structured WING
handoff gate before any supervised fader/gain proof.

## Constraints

- Prompt/readme guidance only.
- No hardware contact.
- No backend WING WebSocket session.
- No OSC/WING writes.
- No live authority widening.

## Expected Output

- `codex_cycle_prompt.md` requires `supervised_prestart_gate` to match
  `expected_supervised_prestart_gate`.
- `tools/codex_autoloop/README.md` documents the same structured gate.
- Tests protect both surfaces.
