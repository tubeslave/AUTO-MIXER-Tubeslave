# PENA-KVMS-GTR-DOSE-02 — new audio candidate v0.5 (2026-11-18)

**State:** Rendered, DSP/sample/level checks passed, **blind listening pending**. This experiment must not be promoted to a production mixing rule. `auto_apply:false`. No OSC/mixer/production changes.

## Previously observed listener choice

In v0.4 blind A/B the listener selected **X**. The stored blind key verifies **X = V04_REBALANCED**, **Y = CONTROL_Y**. The v0.4 candidate applied Crunch −3 dB, Rhythm −3 dB, Lead −2 dB, Lead2 −4 dB (vs control). This supports *only* a one-listener, one-passage preference for the combined rebalance. The earlier v0.3 Y preference concerned the no-dynamic-EQ control, so there is no contradiction.

## New falsifiable test

v0.5 re-renders the same 48-FLAC song with one controlled adjustment relative to preferred v0.4: **additional −0.8 dB** on **Crunch and Rhythm**, Lead and Lead2 unchanged. It does NOT re-enable vocal-conditional dynamic guitar EQ, add master processing, delete instruments, change panning or modify timing. Keep v0.4 as baseline. Test whether further reducing continuous guitar energy is preferable to the already-chosen mix. If no clear preference, retain v0.4.

## Reproducible engineering results

- 48 kHz / 24-bit / stereo; 195.81 sec; 42 selected live audio stems; no mastering and no clipping.
- EBU R128 FFmpeg integrated: v0.4 −22.6 LUFS, v0.5 −22.8 LUFS; LRA 7.6 vs 7.5 LU; true peak −5.6 vs −5.7 dBTP.
- 51.28–77.28 sec: guitar-group sum / lead-vocal RMS **+3.45 dB** (v0.4) vs **+2.94 dB** (v0.5), delta **−0.51 dB**. This is an engineering proxy, **not** listener preference.
- A/B comparison WAV: both −20.000 LUFS measured with pyloudnorm; difference <0.01 LU. Hidden X/Y mapping is stored with local handoff, **not revealed in this research note**.
- WAV SHA256: `59509ea91ba4d9c7709e8b34b7b9a4386e7e5d796e3b293e82d589295a23daa1`.
- MP3 SHA256: `efabb47977fc8a2c8828f993e5f846f263bf90dfa9c720b10d17de598061f948`.
- Full local deliverables: `Pena_KVMS_v05_Mix_Package.zip` (WAV, MP3, matched blind clips, render script, QC script, recipe, evidence, local blind key, SHA256 manifest); audio **not** uploaded to GitHub.

## Theory update

H7 (role-density / relative musical group balance): **supportive_single_listener_observation**, NOT independently proven or generalized. The next evidence should measure preferred guitar-balance *dose response* with matched loudness and separate tests across song sections. No automatic DSP coefficient or live mixer action is authorized.

The test only isolates a combined Crunch/Rhythm gain adjustment; it cannot determine the individual causal contribution of either group, or distinguish perceptual source segregation from musical-aesthetic preference.