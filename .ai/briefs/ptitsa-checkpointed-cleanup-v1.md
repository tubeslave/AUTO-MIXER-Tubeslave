# Ptitsa checkpointed model cleanup v1

## Problem

The first full-song model-cleanup workflow (`35828150231`) ran the eleven model-processed sources sequentially on a CPU runner. The job had a 180-minute timeout and was cancelled during `Neural cleanup selected sources`. Because the workflow only uploaded artifacts after every source finished, the final render, MP3 and all intermediate work were lost when the timeout was reached.

## Decision

Model cleanup is now split into independent per-source matrix jobs. The current source set is:

- BASS
- GTR
- FLOOR
- KICK_IN
- KICK_OUT
- NIKITA_VOX
- SN_B
- SN_T
- TOM_1
- TOM_2
- VALERA_VOX

The matrix uses `fail-fast: false` and `max-parallel: 6`. A slow or failed source must no longer erase already completed source work.

## Checkpoint contract

Every successfully processed source uploads its own `ptitsa-clean-<STEM>` artifact containing:

1. the cleaned full-length WAV used by the final render;
2. 192 kbps listening previews for RAW, CLEAN and REMOVED;
3. a JSON report with source name, processing strength, frame/sample-rate metadata, raw/clean/removed RMS and removed-to-raw dB.

The REMOVED signal is defined as `RAW - CLEAN`. It is evidence for artifact/over-cleaning review, not a musical stem to be used in the mix.

## Processing policy

Bleed-suppressor sources use the previously selected conservative 50% blend. BASS retains the previously tested 100% model result because its measured residual was weak in the earlier candidate test. GTR uses the previously selected Aufr33 denoise candidate at 50% blend.

These settings are experiment candidates, not accepted production processing.

## Acceptance gate

No model-cleaned source is accepted automatically. For every processed source:

- compare RAW versus CLEAN at matched listening level;
- listen to REMOVED for wanted source content, transient loss, tonal damage and modulation artifacts;
- inspect the residual metrics as supporting evidence only;
- accept, revise strength/model, or reject the cleanup for that source.

Human listening remains mandatory for subjective acceptance.

Only after the per-source decisions are made should the final cleaned full mix be treated as an accepted editing input. A successful CI render alone is not evidence that the cleanup sounds better.

## Current execution

- Refactor commit: `705b6f92c1bf202657e115ee330afcefd7fbddcb`
- Replacement cleanup run: `35846566268`
- Status at launch: matrix accepted by GitHub; up to six cleanup jobs started concurrently, remaining jobs queued behind the matrix concurrency limit.
