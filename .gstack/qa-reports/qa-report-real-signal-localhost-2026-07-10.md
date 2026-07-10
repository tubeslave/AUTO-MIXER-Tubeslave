# Real-signal QA — 2026-07-10

Status: `DONE_WITH_CONCERNS`

## Test setup

- Ableton Live set: `/Users/dmitrijvolkov/Music/REPA 09.04.26 Project/REPA 09.04.26.als`
- Transport: playback enabled, arrangement loop enabled.
- WING: `192.168.1.102`, connected through `EnhancedOSCClient`.
- Audio input: WING USB, 48 channels at 48 kHz.
- Frontend: `http://localhost:3000/#/dashboard`.
- Backend: `http://127.0.0.1:8765`.

## Backend evidence

- 47 real meter frames observed over 5 seconds.
- 43 USB channels exceeded -60 dBFS.
- Example peaks: ch1 -9.1 dBFS, ch3 -9.2 dBFS, ch9/ch10 -12.0 dBFS, ch47/ch48 -12.9 dBFS.
- Audio capture remained active; no synthetic meter source was used.

## Frontend evidence

- WING connection indicator: connected.
- Audio indicator: `48ch · 48k`.
- Spectrum source: `Real audio`.
- Channel energy source: `Real meters`.
- Master and channel values changed after backend refresh.
- Browser console warnings/errors: none.
- Screenshot: `screenshots/real-signal-dashboard-20260710.png`.

## 60-second master capture

- Source: WING USB 47/48 (Main L/R capture path).
- Duration: exactly 60.000 s, 2,880,000 frames, 48 kHz, stereo.
- Overflow blocks: 0.
- Integrated loudness: -28.08 LUFS.
- True peak: -9.33 dBTP.
- L/R balance: +0.01 dB.
- Correlation: 0.998.
- MP3: 320 kbps, 2,401,965 bytes.
- SHA-256: `f63cc17c92ce3acd27d3d03eb846cc46782ffbfe09749c79dc7b97931c124394`.
- Share page: https://tempfile.org/e84K35UPdmW/
- Direct download: https://tempfile.org/e84K35UPdmW/download
- Link expiry: 2026-07-12 12:00:33 MSK.
- Downloaded copy hash matched the local MP3 exactly.

## Concern

The capture is about 2.1 LU below the lower bound of the current -20 ± 6 LUFS target. Peak headroom is safe, so no live console gain change was made without a separate operator request.
