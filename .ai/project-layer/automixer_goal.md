# Automixer Goal

## Main Goal

Automixer is an automatic live mixer for soundchecks and concerts. It analyzes
multi-channel audio, makes safe recommendations or decisions, and can send
bounded corrections to a Behringer WING Rack through OSC only when the operator
explicitly approves the exact live action.

## Product Direction

- Help an FOH operator get faster, safer soundchecks.
- Keep live concert operation conservative and supervised.
- Use audio analysis, source knowledge, DSP heuristics, and decision traces to
  explain why a change is suggested.
- Keep offline, shadow, proposal, supervised, and live-control modes separate.

## Non Goals

- Do not create a fully autonomous live FOH executor without a separate ADR and
  hardware proof.
- Do not bypass the operator with hidden OSC writes.
- Do not treat Paperclip, Codex, or LLM output as authority to mutate the
  console.
- Do not optimize loudness at the expense of feedback safety, true-peak
  headroom, or fader ceilings.

## Definition Of Useful Progress

Useful progress makes Automixer safer, more explainable, better tested, or more
accurate at soundcheck/live recommendations while preserving dry-run and
supervised boundaries.

## Current Execution Priority

The long-term goal is a fully capable AI mixing engineer, but the current
execution priority is not broad autonomy. The current priority is a supervised
WING Rack MVP that proves the first safe hardware loop end-to-end.

A development cycle counts as useful only when it closes or measurably reduces
one of these blockers:

1. WING read-only connection proof.
2. Physical console readback proof.
3. Supervised fader write proof with explicit operator approval.
4. Supervised gain-down / trim proof with explicit operator approval.
5. Rollback proof.
6. Emergency stop proof.
7. Connection-loss / timeout fail-closed proof.
8. Operator-visible evidence report.
9. Readiness matrix update based on real proof.

Do not add new shadow-only methods, proposal-only surfaces, memory-trust flows,
queue-import bridges, result consumers, broad report layers, or read models
unless the work directly closes one of the blockers above.

When WING Rack is connected, work in this order:

1. Read-only preflight.
2. Capture current console state.
3. Verify physical readback.
4. Run dry-run recommendation.
5. Require explicit operator approval for the exact action.
6. Apply only bounded supervised fader/gain action.
7. Read back the result.
8. Test rollback.
9. Record proof artifact.

Forbidden live writes until a separate ADR and hardware proof:

- EQ.
- Compressor.
- Gate.
- Routing.
- Mute/unmute.
- FX sends.
- Scenes.
- Matrices.
- DCA changes.
- Full autonomous mode.

Daily success is measured by closed WING proof blockers, not by commit count,
method count, report count, or new abstractions.
