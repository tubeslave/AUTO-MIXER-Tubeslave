# AI Mixing Offline Test Chain

This offline chain renders candidate mixes from local files, scores them with
role-specific critics, and accepts a result only after the Safety Governor
passes it. It does not send OSC/MIDI and does not modify a real console.

## Module roles

- MuQ-Eval / MuQ-A1: chief music critic for master or stem before/after scores.
  It only evaluates renders and never controls EQ, gain, compression, or faders.
- Audiobox Aesthetics: secondary quality critic for pleasantness, cleanliness,
  and naturalness. If it disagrees with MuQ, the report makes that visible.
- MERT: embedding encoder for future stem heads such as vocal clarity, kick
  punch, bass definition, drum cleanliness, guitar harshness, mix cleanliness,
  and live mix readiness.
- CLAP / LAION-CLAP: semantic audio-text critic for prompts such as clean vocal,
  muddy low mids, punchy kick, harsh guitar, and overcompressed mix.
- Essentia: technical analyzer role. When Essentia is absent, the adapter uses
  existing project metrics for spectral centroid, rolloff, loudness, brightness,
  low-mid mud, harshness, dynamics, and stereo safety.
- PANNs / BEATs: channel identity, event, bleed, noise, and silence detector
  role. The fallback uses filename role inference and signal statistics.
- Demucs / Open-Unmix: offline reference source separation role. It is never
  allowed in the real-time audio thread.
- automix-toolkit / FxNorm-Automix / Diff-MST / DeepAFx ideas: action planner
  inspiration. The planner only proposes candidate actions.
- Sandbox Renderer: applies candidate actions to local buffers, loudness-matches
  variants, and writes WAVs with a conservative pre-safety peak margin before
  critics and the Safety Governor evaluate them.
- Decision Engine: combines critic deltas and safety score with weight
  renormalization.
- Safety Governor: final protection layer for clipping, headroom, excessive
  gain/EQ/compression, vocal clarity, phase, harshness, mud, and identity/bleed.

## Input layout

Put files here:

```text
offline_test_input/
  multitrack/
    01_kick.wav
    02_snare.wav
    03_bass.wav
    04_vocal.wav
  reference/
    reference_mix.wav
  config/
    channel_map.json
    test_config.yaml
```

`channel_map.json` is optional. It can map file names or stem names to roles.
If it is missing, roles are inferred from filenames.

## Run

```bash
python -m ai_mixing_pipeline.offline_test_runner \
  --input offline_test_input \
  --output offline_test_output \
  --config configs/ai_mixing_roles.yaml \
  --mode offline_test
```

By default `offline_test.safe_render_peak_margin_db` is `0.6`. With the
standard Safety Governor limit of `-1.0 dBTP`, candidate renders are printed
against a `-1.6 dB` render ceiling before they are scored. This prevents tiny
true-peak or headroom rounding misses from blocking otherwise safe candidates;
the Safety Governor still enforces the original `-1.0 dBTP` and `1.0 dB`
headroom rules.

The main mode is `offline_test`. Other modes are available:

- `observe`: analyze only.
- `suggest`: analysis plus candidate actions, without a final console-style
  state application.
- `offline_test`: full load, analyze, candidates, render, critics, safety, and
  best mix.
- `shadow_mix`: compare variants and write reports without treating the result
  as applied state.
- `assisted_offline`: same offline safety path, intended for future approved
  safe actions.

## Output layout

```text
offline_test_output/
  renders/
    000_initial_mix.wav
    001_candidate_gain_balance.wav
    002_candidate_eq_cleanup.wav
    003_candidate_compression.wav
    004_candidate_fx.wav
    005_best_mix.wav
  reports/
    decision_log.jsonl
    summary_report.md
    critic_scores.csv
    accepted_actions.json
    rejected_actions.json
  snapshots/
    mixer_state_before.json
    mixer_state_after.json
```

Additional candidate renders may appear for vocal clarity, low-mid cleanup, and
bass/kick balance.

## Reading the summary

`summary_report.md` shows module availability, candidates, actions, critic
scores, the selected candidate, rejection reasons, and Safety Governor notes.
`critic_scores.csv` is the compact table for comparing deltas across critics.
`decision_log.jsonl` has one JSON row per candidate and is suitable for later
training or audit.

## Optional models and fallback

Every proposed layer participates on every run. If MuQ-Eval, Audiobox, MERT,
CLAP, Essentia, PANNs/BEATs, Demucs, Open-Unmix, or external research planners
are not installed, that layer still writes an explicit status row and warning.
Critic/analyzer roles continue with deterministic fallback metrics where
possible; source separation records a skipped fallback when there is no
reference or no separator backend.

No heavy model is mandatory for the CLI to run. The mandatory pieces are local
Python, NumPy, soundfile, and the repository's existing lightweight analysis
code.

`module_status` in the CLI JSON and `summary_report.md` is the participation
ledger. It must include:

- `muq_eval`
- `audiobox_aesthetics`
- `mert`
- `clap`
- `essentia`
- `panns_or_beats`
- `demucs_or_openunmix`
- `automix_toolkit_fxnorm_diffmst_deepafx`
- Safety Governor output per candidate

## Why critics do not control parameters

Critics answer whether a render is better. They do not know operator intent,
hardware state, headroom limits, feedback risk, channel identity confidence, or
whether a change is safe to apply. For that reason MuQ-Eval and every other
critic only produce scores and explanations.

## Why Safety Governor is final

Live-sound safety rules remain non-negotiable even offline. A candidate can
score well musically and still be rejected for clipping, low headroom, phase
collapse, excessive compression, vocal clarity loss, harshness, mud, or a
channel identity/bleed risk. The Sandbox Renderer may apply a transparent
global trim before evaluation to create headroom, but the Safety Governor is
still the last gate before `best_mix.wav` is accepted.
