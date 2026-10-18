# Live Soundcheck / Concert Pipeline v2

## Decision

Studio and live are separate products sharing analysis/director code.

### Studio pipeline
Offline files -> Editing/Cleanup -> Mixing Directors -> Perceptual Critic -> Autonomous Iteration -> Mastering.

### Live pipeline
Audio I/O -> realtime feature engine -> live context -> channel/group/main directors -> safety governor -> WING control.

Editing, pitch correction, destructive cleanup and mastering are not part of the live control loop.

## Two planes

### Audio plane
Runs locally next to the console. It must never depend on cloud LLM latency.

Inputs:
- WING USB: up to 48x48 channels.
- Dante/AoIP: up to 64x64 with the WING Dante/AES67 module.
- future: AES50/other capture adapters.

Responsibilities:
- enumerate/open audio device;
- stable channel map and ring buffers;
- per-channel meters, LUFS/RMS/peak/crest and spectrum;
- masking, feedback, correlation/phase, transient/dynamics features;
- group/main features and optional short audition capture.

### Control plane
- WING state readback;
- OSC/native protocol adapter;
- proposal -> write -> readback -> verify;
- rollback snapshot and audit log;
- production modes add rate limiting, confidence, max-step limits and manual-touch priority.

## Operating modes

OBSERVE: audio + console read only.
PROPOSE: produce actions, never write.
BENCH_TEST: explicit development/test mode. Normal production write restrictions are bypassed so EQ, dynamics, faders, buses, routing and other WING decisions can be observed directly on the console. Every write is still logged and read back so the test is inspectable.
SUPERVISED: operator-approved bounded write batches.
AUTO_SAFE: allowlisted reversible low-risk autonomous writes.
EMERGENCY: deterministic feedback/clip protection.
FREEZE: no writes.

Mode is never inferred merely from the presence of a console.

### User mode contract

When the operator explicitly says this is a TEST / development session, BENCH_TEST may be selected and broad WING writes are expected.

When the operator explicitly says "we are doing a soundcheck", "soundcheck mode", "we are running the concert/show", or equivalent, production protections are enabled. BENCH_TEST must not carry across into a soundcheck or concert.

Default after a fresh connection or uncertain context: OBSERVE.

## Live Directors

1. Input/Preamp Director.
2. Phase Director.
3. Channel EQ Director.
4. Dynamics Director.
5. Feedback Director.
6. Balance Director.
7. Group Director.
8. Masking Director.
9. Space/FX Director.
10. Main Director.
11. Monitor Director.
12. Show/Scene Director.

## GPT / voice commands

GPT is supervisory, not the realtime DSP loop. Commands compile into explicit local actions; free-form LLM text is never sent directly to OSC.

Examples:
- "test mode" -> BENCH_TEST
- "start soundcheck" -> leave BENCH_TEST and enter production soundcheck policy
- "concert mode" -> leave BENCH_TEST and enter production show policy
- "analyze drums only"
- "freeze vocal 1"
- "show proposed EQ changes"
- "undo last pass"
- "stop all writes"

## Audio routing

### USB MVP
WING 48x48 USB -> local Live Bridge -> feature engine.

### Dante
WING AoIP-Dante 64x64 -> Dante network -> host receiver -> Live Bridge.
Dante network subscriptions are a separate routing adapter from WING internal routing.

## First MVP

- WING discovery + state snapshot.
- USB 48-channel capture adapter.
- channel map from WING names/sources.
- 20 Hz feature stream.
- channel/group/main analysis.
- Proposal API.
- BENCH_TEST for visible engineering tests.
- production safety governor.
- OSC write/readback/rollback.
- soundcheck FSM: DISCOVER -> PATCH_VERIFY -> LISTEN -> PROPOSE -> APPLY -> VERIFY -> HOLD.
