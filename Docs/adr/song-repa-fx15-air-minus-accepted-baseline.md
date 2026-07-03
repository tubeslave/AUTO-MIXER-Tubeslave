# SONG REPA FX15 Air -0.80 Accepted Baseline

## Context

The operator reported that `/Users/dmitrijvolkov/Desktop/Ai MIXING/SONG_REPA_MIXING_STATION_VIS_FX15_AIR_MINUS_0_80_20260427.wav` sounded ideal. Later decision-layer renders that started from a raw or weakly adjusted mix did not preserve that balance and exposed two issues: default pymixconsole FX busses could add unwanted ambience, and micro-candidates without the accepted baseline did not create meaningful balance changes.

## Options Considered

1. Continue optimizing from raw stems only.
2. Treat the accepted FX15/Air print as a reference file but not as a candidate.
3. Record the accepted FX15/Air print as the golden `no_change` candidate and require future candidates to beat it.

## Decision

Use the accepted FX15/Air -0.80 SONG REPA render as the golden baseline for future SONG REPA passes. Future runs may add AYAIC balance, optimizer candidates, critics, and Safety Governor checks, but they must keep the accepted recipe as `candidate_000_no_change` or reproduce it before applying additional candidate corrections.

## Why This Won

The accepted file is supported by operator feedback, prior MuQ-A1 searches, and a 2026-04-28 postcheck where the new critics selected `no_change` over extra mirror-EQ, glue, mud/harsh cleanup, and trim candidates. It also preserves safe headroom around `-4.979 dBFS` sample peak.

## Rejected Alternatives

- Raw unity-sum starts were rejected because they lost the accepted balance.
- Heavy postmaster EQ, limiting, and compression were rejected because previous logs show they underperformed or chased metrics at the expense of the mix.
- Live OSC replay of EQ/FX/compressor changes was rejected because the accepted visualization report only verified fader and pan paths.

## Implementation Plan

- Add a durable accepted recipe file under `configs/accepted_mix_recipes/`.
- Add a documentation page describing the pipeline and logs.
- Update project memory and accepted-decision JSONL.
- Ensure the decision-layer runner passes AYAIC balance config into the mixer and renderer.
- Keep pymixconsole FX busses disabled by default unless FX are explicitly modeled.

## Test Plan

- Run targeted decision-layer tests for candidate generation, fallback virtual mixer, sandbox rendering, and offline correction runner.
- Compile the edited decision-layer modules.
- Do not run live OSC/MIDI commands.

## Risks And Rollback

Risk: the recipe is SONG REPA-specific and should not be blindly copied to unrelated genres. Rollback is to disable this accepted recipe and fall back to generic AYAIC input balance plus normal critic/decision flow.
