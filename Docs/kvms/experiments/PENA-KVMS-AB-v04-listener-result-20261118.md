# PENA-KVMS-AB-v04 — blinded listener result (2026-11-18)

Status: **one observed preference, research-only**, not production validation.

## Ground truth

- Source song: "Пена" / Логопед, edited 48-FLAC multitrack, mix target 48 kHz.
- Stored blind key: `Pena_KVMS_v04_blind_key.json`: **X = V04_REBALANCED**, **Y = CONTROL_Y**.
- Listener feedback, recorded verbatim: **"предпочитаю X"** (user supplied no additional subjective notes on this test).
- X intervention vs Y: broadband musical-group gains Crunch −3.0 dB, Rhythm −3.0 dB, Lead −2.0 dB, Lead2 −4.0 dB. Both versions use no vocal-conditional guitar band EQ. All other mix processing held fixed in the 26-second blind comparison.
- The 26-second versions were loudness-matched around −20 LUFS. 
- Earlier v0.3 result was Y (no vocal-conditional guitar band EQ) preferred to X with the experimental EQ, comment "Гитар слишком много".

## What the result supports / does NOT support

Supports: for this single listener and passage, a **combined guitar-group rebalance** was preferred to the higher-guitar-level control, consistent with the earlier complaint. Prior preference Y from the v0.3 test and current preference X in the v0.4 test refer to DIFFERENT pairwise comparisons, not a contradiction.

Does NOT independently establish optimal gain for any one group, causal attribution to guitar role masking vs overall balance, improved intelligibility, population generalization, full-song preference, or benefit of *further* reduction. There are no independent listener ratings of the component-isolated RHYTHM_MINUS2 and LEADS_MINUS2 tests yet.

## Hypothesis and engineering consequences

- H7 **role-density / relative group balance**: status **supportive_single_listener_observation**, still `candidate`; `auto_apply:false`.
- H1/H3 candidate vocal-conditional guitar-band EQ from v0.3: `not_preferred_in_single_AB`; `auto_apply:false`.
- Add a separate objective feature `guitar_group_to_vocal_rms_db`, but do **not** treat it as artistic-quality score or sufficient optimization target.
- Distinguish 24 guitar microphones from 8 independent L/R performances in 4 musical groups.
- Preserve previous chosen **v0.4 X** as the listening reference. All future changes must be evaluated against it at matched loudness.

## Follow-up PENA-KVMS-GTR-DOSE-02

Create an offline, *experimental* full-song v0.5 with **one isolated manipulation**: further −0.8 dB for Crunch and Rhythm vs preferred v0.4; keep Lead, Lead2 and all non-guitar buses unchanged. Compare **v0.4 control vs v0.5 proposal**, using 26-second excerpts and full mixes with matched loudness for blind listening. Revert to v0.4 if not preferred. No mixer/OSC writes, no production DSP update, no rule promotion.

This document records the observed choice before v0.5 listening; it does not claim v0.5 is better.