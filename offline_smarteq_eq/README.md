# Offline SmartEQ-Style EQ Method

This folder contains a separate offline-only equalization method. It is not part
of autogain and does not send mixer, WING, OSC, or live writes.

The method reuses the observable smartEQ4 architecture at a high level:

- block/FFT spectrum analysis;
- long-term average spectrum per source;
- source-profile target curves;
- constrained smart-weight-style correction regions;
- minimum-phase biquad EQ for offline render;
- conservative caps and full JSON reporting.

It does not copy Sonible's proprietary model weights or hidden ONNX/profile
logic. The implementation is an own automixer EQ method built from the observed
DSP structure.

## Render SONG REPA

```bash
PYTHONPATH=. python3 -m offline_smarteq_eq.render_mix \
  --input-dir "$HOME/Desktop/SONG  REPA" \
  --output "$HOME/Desktop/SONG_REPA_OFFLINE_SMARTEQ_GAINSTAGED/SONG_REPA_OFFLINE_SMARTEQ_GAINSTAGED.wav" \
  --report "$HOME/Desktop/SONG_REPA_OFFLINE_SMARTEQ_GAINSTAGED/SONG_REPA_OFFLINE_SMARTEQ_GAINSTAGED_report.json"
```

The render chain is:

1. Current canonical input-trim gain-staging method.
2. Offline spectral EQ method from this folder.
3. Stereo pan/sum and safety master trim.

