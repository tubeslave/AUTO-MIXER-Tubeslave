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
