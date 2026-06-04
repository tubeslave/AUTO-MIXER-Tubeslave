# Offline SmartEQ-Style EQ Method

This folder contains a separate offline-only equalization method. It is not part
of autogain and does not send mixer, WING, OSC, or live writes.

The method reuses the observable smartEQ4 architecture at a high level:

- block/FFT spectrum analysis;
- long-term average spectrum per source;
- source-profile target curves;
- constrained smart-weight-style correction regions;
- minimum-phase biquad EQ for offline render;
- conservative caps, protected fundamental regions, spaced bands, and full JSON reporting.
- AIAK-style Integrated LUFS level-plane correction per source before stereo summing.

It does not copy Sonible's proprietary model weights or hidden ONNX/profile
logic. The implementation is an own automixer EQ method built from the observed
DSP structure.

## Render SONG REPA

```bash
PYTHONPATH=. python3 -m offline_smarteq_eq.render_mix \
  --input-dir "$HOME/Desktop/SONG  REPA" \
  --output "$HOME/Desktop/SONG_REPA_OFFLINE_SMARTEQ_V2_AIAK_LUFS/SONG_REPA_OFFLINE_SMARTEQ_V2_AIAK_LUFS.wav" \
  --report "$HOME/Desktop/SONG_REPA_OFFLINE_SMARTEQ_V2_AIAK_LUFS/SONG_REPA_OFFLINE_SMARTEQ_V2_AIAK_LUFS_report.json" \
  --eq-max-active-bands 5 \
  --aiak-level-plane-base-lufs -24.0 \
  --aiak-channel-fader-ceiling-db 6.0
```

The render chain is:

1. Current canonical input-trim gain-staging method.
2. Offline spectral EQ v2 from this folder.
3. AIAK Integrated LUFS level-plane correction per source.
4. Stereo pan/sum.
5. Safety master trim.
