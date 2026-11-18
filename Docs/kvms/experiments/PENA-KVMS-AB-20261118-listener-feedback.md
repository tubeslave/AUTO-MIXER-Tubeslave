# PENA-KVMS-AB-20261118 — listener feedback (2026-11-18)

Status: **one observed preference / one listener**, research evidence only. **No production rule promotion or console actions**.

## Original blind protocol

- Excerpt: Pena, 51.28–77.28 s, 26 s.
- Integrated loudness: X −19.61 LUFS, Y −19.60 LUFS (0.01 LU difference).
- Blind key checked against `BLIND_REVEAL_AFTER_LISTENING.json` and `mix_kvms_v03.py`:
  - **X = candidate_vocal_priority_dynamic_guitar_band_Eq**: Crunch/Rhythm attenuated in approximately 1350–3700 Hz during detected vocals, maximum approximately 1.1 dB.
  - **Y = control_no_dynamic_guitar_EQ**.
- Listener selected **Y**, comment in Russian: **«Гитар слишком много»** ("Too many guitars").
- Playback environment, repeats, and randomization details beyond the initial mapping are unknown.

## Evidence-based interpretation

1. The control **Y was preferred** to the experimental dynamic-EQ variant **X** in this single 26-second A/B observation. This rejects treating this particular candidate as a *validated* improvement.
2. The complaint may involve guitar role competition, density, or relative group level, **not proven to be a masking problem**. A/B only varied dynamic EQ, not guitar gain or active arrangement layers.
3. The source mix has four guitar groups (crunch, rhythm, lead, lead2), each with independent L/R performances recorded with several microphones: **24 microphone tracks are not 24 independent performances**.
4. Full v0.3 WAV/MP3 rendered with the **X** candidate. It must not be labelled as the user-preferred Y control. Original files are preserved.
5. This single listener report has no significance estimate and is not generalizable to other songs, song sections, or playback systems.

## Hypothesis / Automixer decision

- KVMS H1/H3: `negative_single_listener_observation`, `auto_apply=false`. Do not automatically deploy vocal-conditional guitar-band EQ.
- KVMS H7 (new): **role-density hypothesis**, `proposed_not_tested`, `auto_apply=false`: guitar perceived-overcrowding may depend on role competition and relative group balance more than a small single-band EQ change.
- Keep `masking_proxy`, `group_level`, `role_overlap`, `event_detectability`, and `listener_preference` as distinct fields, not interchangeable labels.
- For next A/B take **Y** as control, then independently test (a) Crunch+Rhythm −2 dB, (b) Lead+Lead2 −2 dB, (c) section-conditioned attenuation of a nonlead *role*. Hold unchanged components fixed; loudness-match copies and blind-review. No winner claimed in advance.

**Follow-up ID:** `PENA-KVMS-GTR-DENSITY-01`.

Source artifacts: Pena_KVMS_v03 / `Pena_KVMS_v03_1_Experiment_Evidence.json`, `BLIND_REVEAL_AFTER_LISTENING.json`, `Pena_KVMS_v03_verification.json`, `mix_kvms_v03.py`. These local source artifacts are not claimed as GitHub-tracked files.

Research-only entry on `research/kvms-v0.2-20261118`; master/OSC/DSP unchanged.
