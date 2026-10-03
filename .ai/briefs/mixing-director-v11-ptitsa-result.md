# Mixing Director v1.1 Dynamics: Ptitsa experiment

Trigger from human evaluation of v1.0:
- static balance / pan foundation was judged very good;
- main weakness was source-level dynamics causing instruments to rise/fall and continuously change balance.

Experiment:
- inserted Dynamics Director before Masking/EQ;
- no EQ, reverb, saturation, limiting;
- processed buses: drums, bass, lead vocal;
- guitar/keys/playback bypassed because measured spread did not justify processing;
- secondary vocal compressor candidate was rejected because its stability metric regressed; defer that source to phrase-aware Automation Director.

Observed:
- drums: max GR ~2.0 dB, p95 ~1.64 dB;
- bass: max GR ~3.50 dB, p95 ~1.54 dB;
- lead vocal: max GR 5.0 dB, p95 ~4.48 dB;
- vocal/music p10-p90 ratio spread reduced by ~1.27 dB;
- drums/bass ratio spread reduced by ~0.58 dB.

Decision:
- technical regression gate PASS;
- human level-matched A/B is required before promoting v1.1 as the new mixing baseline.
