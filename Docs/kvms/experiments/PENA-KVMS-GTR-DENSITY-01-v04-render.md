# PENA-KVMS-GTR-DENSITY-01 — audio render v0.4 (2026-11-18)

**Mode:** local offline mix, listening pending, research-only. **Auto-apply: false.** Production DSP, mixer OSC, firmware and master branch unchanged.

## Listener-provided observation

Previous blinded Pena v0.3 test: Y (no vocal-conditional dynamic guitar EQ) preferred over X (with candidate guitar band attenuation), comment “Гитар слишком много” (“Too many guitars”).

## v0.4 test intervention

Use the previously preferred **Y** (no dynamic EQ) as baseline. All original drum/bass/vocal/backing processing, vocal reverb, original performances and pans preserved. **Only** guitar musical-group gains adjusted:

- crunch −3.0 dB
- rhythm −3.0 dB
- lead −2.0 dB
- lead2 −4.0 dB

Do not confuse 24 close microphone tracks with 24 independent guitar performances. No guitar track removal or new timing/pitch editing. The dynamic guitar-band EQ candidate of v0.3 was disabled.

## Measured, not subjective, outcomes

- Full mix length 195.81 s; 48 kHz PCM24 stereo; 42 non-empty source channels used (48 original FLAC, 4 empty); no mastering/limiting.
- Full v0.4: −22.69 LUFS-I, −5.59 dBTP, 7.60 LU LRA (FFmpeg). MP3: −22.69 LUFS-I, −5.55 dBTP.
- On 51.28–77.28-second excerpt, summed guitar-group RMS relative to lead-vocal RMS: 6.14 dB baseline → 3.45 dB v0.4, change −2.68 dB. **This is not a psychoacoustic quality metric.**
- Prior Y control reconstruction cross-correlates with previous Y excerpt at >0.9999999999 after level normalization.
- Four 26s pairwise experiment stems (Y control, Crunch+Rhythm −2 dB, Lead+Lead2 −2 dB, combined v0.4) were normalized as **listening copies only** to −20.00 LUFS-I PCM (pyloudnorm); MP3 approx −20.02 to −20.03 LUFS (FFmpeg). Original full premaster remained unchanged.
- No sample clipping; source ZIP CRC pass; full render exported and checked. SHA256 full WAV: 6713423d0aafd2b83b7a608b91e1a6f984f9632758ffb1ad1559220a637866ba. SHA256 full MP3: d551ad4de8399ab27b5cb3d72b0c14d5a899b06a4132f6736f6563c93194fc67.

## Hypothesis update

H7 (role density): **candidate / unvalidated**. The v0.4 balanced mix is a proposed response to the single listener comment, **not an established perceptual win**. Main listening comparison: old Y control vs the v0.4 combined rebalance at matched loudness. Additional probes independently isolate rhythm-group vs lead-group gain, no frequency carve.

**Decision rule:** promote no rules and send no console changes until blind listening confirms preference without unwanted reduction in musical energy, vocal intelligibility, or musical roles across several song sections and listeners.

No audio uploaded to GitHub; only this experimental record. All audio files remain in local handoff package Pena_KVMS_v04_Mix_Package.zip.
