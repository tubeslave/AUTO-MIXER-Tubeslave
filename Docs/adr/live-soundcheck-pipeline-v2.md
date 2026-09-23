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
- stable channel map;
- ring buffers;
- per-channel meters, LUFS/RMS/peak/crest;
- spectrum, masking, feedback candidates;
- correlation/phase;
- transient/dynamics features;
- group/main features;
- optional short audition capture for higher-level analysis.

### Control plane
- WING state readback first;
- OSC/native protocol adapter;
- proposal -> safety -> write -> readback -> verify;
- rollback snapshot;
- rate limiting, hysteresis, confidence and max-step limits;
- manual-touch freeze: operator action wins.

Dante network subscriptions are a separate adapter. WING internal MOD/USB routing is not the same thing as Dante network subscription routing.

## Live Directors

1. Input/Preamp Director: clipping/headroom; conservative trim only when explicitly enabled.
2. Phase Director: polarity/correlation proposals; no blind delay writes.
3. Channel EQ Director: HPF/PEQ, resonance/harshness, bounded dynamic corrections.
4. Dynamics Director: compression/gate/expander proposals appropriate to live sources.
5. Feedback Director: fast narrow-band detection; emergency path has deterministic local rules.
6. Balance Director: faders/VCAs/DCAs and vocal-anchor relationship.
7. Group Director: drums/music/vocals/subgroups, buses, DCAs, mute groups.
8. Masking Director: kick/bass, vocal/music, snare/guitar relationships.
9. Space/FX Director: sends/returns and section-aware FX within safe bounds.
10. Main Director: whole-mix spectrum, crest, loudness trend and stereo image. Prefer source/group fixes before Main EQ.
11. Monitor Director: separate policy; never reuse FOH targets blindly.
12. Show/Scene Director: snapshots/snippets, song/section context and recall boundaries.

## GPT / voice commands

GPT is the supervisory interface, not the realtime DSP loop.

Examples:
- "start soundcheck"
- "analyze drums only"
- "freeze vocal 1"
- "make the lead vocal slightly more forward"
- "show proposed EQ changes"
- "apply only high-confidence changes"
- "undo last pass"
- "stop all writes"

Commands compile into explicit local actions. No free-form LLM output is sent directly to OSC.

## Audio routing

### USB MVP
WING 48x48 USB -> local Live Bridge -> feature engine. This is the first implementation target because it avoids Dante network-control dependencies.

### Dante
WING AoIP-Dante 64x64 -> Dante network -> host receiver (DVS/interface) -> Live Bridge.
Dante subscriptions are configured through a dedicated Dante routing layer. If Dante Managed API/DDM is unavailable, routing remains a verified preset/manual operation rather than GUI automation during a show.

## Safety states

OBSERVE: audio + console read only.
PROPOSE: produce actions, never write.
SUPERVISED: user approves batches; bounded writes enabled.
AUTO_SAFE: only allowlisted reversible low-risk writes.
EMERGENCY: deterministic feedback/clip protection only.
FREEZE: no writes.

Default at startup: OBSERVE.

## First MVP

- WING discovery + state snapshot.
- USB 48-channel capture adapter.
- channel map from WING names/sources.
- 20 Hz feature stream.
- channel/group/main analysis.
- Proposal API.
- safety governor.
- OSC write/readback/rollback.
- soundcheck FSM: DISCOVER -> PATCH_VERIFY -> LISTEN -> PROPOSE -> APPLY -> VERIFY -> HOLD.
- no routing writes until expected patch map is verified.
