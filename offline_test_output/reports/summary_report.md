# AI Mixing Offline Test Summary

- Mode: `assisted_offline`
- Selected candidate: `000_initial_mix`
- No change selected: `True`
- Best render: `offline_test_output/renders/000_initial_mix.wav`

## Module Availability

- `audiobox_aesthetics`: fallback/unavailable Audiobox Aesthetics model unavailable; using deterministic aesthetic proxy.
- `automix_toolkit_fxnorm_diffmst_deepafx`: available Research planners participate through bounded candidate-action templates; no external planner dependency is required.; Action planner proposes actions only; Sandbox Renderer and Safety Governor decide whether any candidate is accepted.
- `clap`: fallback/unavailable CLAP/LAION-CLAP unavailable; using prompt-specific technical proxy.
- `demucs_or_openunmix`: fallback/unavailable No reference mix supplied; Demucs/Open-Unmix stage recorded a skipped fallback.
- `essentia`: fallback/unavailable Essentia unavailable; using project technical metrics fallback.
- `mert`: fallback/unavailable MERT backend unavailable: No module named 'perceptual.embedding_backend'
- `muq_eval`: fallback/unavailable MuQ-Eval service unavailable; using local technical fallback.; cannot import name 'MuQEvalService' from 'evaluation' (unknown location)
- `panns_or_beats`: fallback/unavailable PANNs/BEATs unavailable; using filename and signal-statistics fallback.

## Source Separation

- Status: `False`
- No reference mix supplied; Demucs/Open-Unmix stage recorded a skipped fallback.

## Candidates

### 000_initial_mix

- Label: `no_change`
- Render: `offline_test_output/renders/000_initial_mix.wav`
- Final score: `0.050000`
- Safety: `passed`
- Safety level: true_peak=`-1.600 dBTP`, headroom=`1.600 dB`, clips=`0`
- Actions: none

### 001_candidate_gain_balance

- Label: `gain_balance`
- Render: `offline_test_output/renders/001_candidate_gain_balance.wav`
- Final score: `0.050000`
- Safety: `passed`
- Safety level: true_peak=`-1.600 dBTP`, headroom=`1.600 dB`, clips=`0`
- Actions:
  - `gain_change` on `Accordion`: Conservative gain-balance probe for unknown.
  - `gain_change` on `Back Vox L`: Conservative gain-balance probe for unknown.
  - `gain_change` on `Back Vox R`: Conservative gain-balance probe for unknown.
  - `gain_change` on `Bass`: Conservative gain-balance probe for unknown.
  - `gain_change` on `F Tom`: Conservative gain-balance probe for unknown.
  - `gain_change` on `Guitar L`: Conservative gain-balance probe for unknown.
  - `gain_change` on `Guitar R`: Conservative gain-balance probe for unknown.
  - `gain_change` on `KICK`: Conservative gain-balance probe for unknown.
  - `gain_change` on `OH L`: Conservative gain-balance probe for unknown.
  - `gain_change` on `OH R`: Conservative gain-balance probe for unknown.
  - `gain_change` on `Playback L`: Conservative gain-balance probe for unknown.
  - `gain_change` on `Playback R`: Conservative gain-balance probe for unknown.
  - `gain_change` on `SNARE T`: Conservative gain-balance probe for unknown.
  - `gain_change` on `Snare B`: Conservative gain-balance probe for unknown.
  - `gain_change` on `Tom`: Conservative gain-balance probe for unknown.
  - `gain_change` on `Vocal`: Conservative gain-balance probe for unknown.

### 002_candidate_eq_cleanup

- Label: `eq_cleanup`
- Render: `offline_test_output/renders/002_candidate_eq_cleanup.wav`
- Final score: `0.050000`
- Safety: `passed`
- Safety level: true_peak=`-1.600 dBTP`, headroom=`1.600 dB`, clips=`0`
- Actions:
  - `high_pass_filter` on `Accordion`: Clean unnecessary low-frequency energy before mix-bus decisions.
  - `high_pass_filter` on `Back Vox L`: Clean unnecessary low-frequency energy before mix-bus decisions.
  - `high_pass_filter` on `Back Vox R`: Clean unnecessary low-frequency energy before mix-bus decisions.
  - `high_pass_filter` on `Bass`: Clean unnecessary low-frequency energy before mix-bus decisions.
  - `high_pass_filter` on `F Tom`: Clean unnecessary low-frequency energy before mix-bus decisions.
  - `high_pass_filter` on `Guitar L`: Clean unnecessary low-frequency energy before mix-bus decisions.
  - `high_pass_filter` on `Guitar R`: Clean unnecessary low-frequency energy before mix-bus decisions.
  - `high_pass_filter` on `KICK`: Clean unnecessary low-frequency energy before mix-bus decisions.
  - `high_pass_filter` on `OH L`: Clean unnecessary low-frequency energy before mix-bus decisions.
  - `high_pass_filter` on `OH R`: Clean unnecessary low-frequency energy before mix-bus decisions.
  - `high_pass_filter` on `Playback L`: Clean unnecessary low-frequency energy before mix-bus decisions.
  - `high_pass_filter` on `Playback R`: Clean unnecessary low-frequency energy before mix-bus decisions.
  - `high_pass_filter` on `SNARE T`: Clean unnecessary low-frequency energy before mix-bus decisions.
  - `high_pass_filter` on `Snare B`: Clean unnecessary low-frequency energy before mix-bus decisions.
  - `high_pass_filter` on `Tom`: Clean unnecessary low-frequency energy before mix-bus decisions.
  - `high_pass_filter` on `Vocal`: Clean unnecessary low-frequency energy before mix-bus decisions.

### 003_candidate_compression

- Label: `compression`
- Render: `offline_test_output/renders/003_candidate_compression.wav`
- Final score: `0.050000`
- Safety: `passed`
- Safety level: true_peak=`-1.600 dBTP`, headroom=`1.600 dB`, clips=`0`
- Actions: none

### 004_candidate_fx

- Label: `fx_space`
- Render: `offline_test_output/renders/004_candidate_fx.wav`
- Final score: `0.047500`
- Safety: `passed`
- Safety level: true_peak=`-1.600 dBTP`, headroom=`1.600 dB`, clips=`0`
- Actions:
  - `fx_send_change` on `all`: Report-only FX wash reduction candidate.

### 005_candidate_vocal_clarity

- Label: `vocal_clarity`
- Render: `offline_test_output/renders/005_candidate_vocal_clarity.wav`
- Final score: `0.060000`
- Safety: `passed`
- Safety level: true_peak=`-1.600 dBTP`, headroom=`1.600 dB`, clips=`0`
- Actions: none

### 006_candidate_low_mid_mud_reduction

- Label: `low_mid_mud_reduction`
- Render: `offline_test_output/renders/006_candidate_low_mid_mud_reduction.wav`
- Final score: `0.050000`
- Safety: `passed`
- Safety level: true_peak=`-1.600 dBTP`, headroom=`1.600 dB`, clips=`0`
- Actions:
  - `eq_correction` on `Accordion`: Dedicated low-mid mud reduction candidate.
  - `eq_correction` on `Back Vox L`: Dedicated low-mid mud reduction candidate.
  - `eq_correction` on `Back Vox R`: Dedicated low-mid mud reduction candidate.
  - `eq_correction` on `Bass`: Dedicated low-mid mud reduction candidate.
  - `eq_correction` on `F Tom`: Dedicated low-mid mud reduction candidate.
  - `eq_correction` on `Guitar L`: Dedicated low-mid mud reduction candidate.
  - `eq_correction` on `Guitar R`: Dedicated low-mid mud reduction candidate.
  - `eq_correction` on `KICK`: Dedicated low-mid mud reduction candidate.
  - `eq_correction` on `OH L`: Dedicated low-mid mud reduction candidate.
  - `eq_correction` on `OH R`: Dedicated low-mid mud reduction candidate.
  - `eq_correction` on `Playback L`: Dedicated low-mid mud reduction candidate.
  - `eq_correction` on `Playback R`: Dedicated low-mid mud reduction candidate.
  - `eq_correction` on `SNARE T`: Dedicated low-mid mud reduction candidate.
  - `eq_correction` on `Snare B`: Dedicated low-mid mud reduction candidate.
  - `eq_correction` on `Tom`: Dedicated low-mid mud reduction candidate.
  - `eq_correction` on `Vocal`: Dedicated low-mid mud reduction candidate.

### 007_candidate_bass_kick_balance

- Label: `bass_kick_balance`
- Render: `offline_test_output/renders/007_candidate_bass_kick_balance.wav`
- Final score: `0.060000`
- Safety: `passed`
- Safety level: true_peak=`-1.600 dBTP`, headroom=`1.600 dB`, clips=`0`
- Actions: none

## Critic Scores

### 000_initial_mix

- `muq_eval` delta=0.0000 score=1.0000 confidence=0.25
- `audiobox_aesthetics` delta=0.0000 score=1.0000 confidence=0.35
- `mert` delta=0.0000 score=1.0000 confidence=0.30
- `clap` delta=0.0000 score=0.7363 confidence=0.30
- `essentia` delta=0.0000 score=1.0000 confidence=0.45
- `panns_or_beats` delta=0.0000 score=0.8000 confidence=0.35

### 001_candidate_gain_balance

- `muq_eval` delta=0.0000 score=1.0000 confidence=0.25
- `audiobox_aesthetics` delta=0.0000 score=1.0000 confidence=0.35
- `mert` delta=0.0000 score=1.0000 confidence=0.30
- `clap` delta=0.0000 score=0.7363 confidence=0.30
- `essentia` delta=0.0000 score=1.0000 confidence=0.45
- `panns_or_beats` delta=0.0000 score=0.8000 confidence=0.35

### 002_candidate_eq_cleanup

- `muq_eval` delta=0.0000 score=1.0000 confidence=0.25
- `audiobox_aesthetics` delta=0.0000 score=1.0000 confidence=0.35
- `mert` delta=0.0000 score=1.0000 confidence=0.30
- `clap` delta=0.0000 score=0.7363 confidence=0.30
- `essentia` delta=0.0000 score=1.0000 confidence=0.45
- `panns_or_beats` delta=0.0000 score=0.8000 confidence=0.35

### 003_candidate_compression

- `muq_eval` delta=0.0000 score=1.0000 confidence=0.25
- `audiobox_aesthetics` delta=0.0000 score=1.0000 confidence=0.35
- `mert` delta=0.0000 score=1.0000 confidence=0.30
- `clap` delta=0.0000 score=0.7363 confidence=0.30
- `essentia` delta=0.0000 score=1.0000 confidence=0.45
- `panns_or_beats` delta=0.0000 score=0.8000 confidence=0.35

### 004_candidate_fx

- `muq_eval` delta=0.0000 score=1.0000 confidence=0.25
- `audiobox_aesthetics` delta=0.0000 score=1.0000 confidence=0.35
- `mert` delta=0.0000 score=1.0000 confidence=0.30
- `clap` delta=0.0000 score=0.7363 confidence=0.30
- `essentia` delta=0.0000 score=1.0000 confidence=0.45
- `panns_or_beats` delta=0.0000 score=0.8000 confidence=0.35

### 005_candidate_vocal_clarity

- `muq_eval` delta=0.0000 score=1.0000 confidence=0.25
- `audiobox_aesthetics` delta=0.0000 score=1.0000 confidence=0.35
- `mert` delta=0.0000 score=1.0000 confidence=0.30
- `clap` delta=0.0000 score=0.7363 confidence=0.30
- `essentia` delta=0.0000 score=1.0000 confidence=0.45
- `panns_or_beats` delta=0.1000 score=0.9000 confidence=0.35

### 006_candidate_low_mid_mud_reduction

- `muq_eval` delta=0.0000 score=1.0000 confidence=0.25
- `audiobox_aesthetics` delta=0.0000 score=1.0000 confidence=0.35
- `mert` delta=0.0000 score=1.0000 confidence=0.30
- `clap` delta=0.0000 score=0.7363 confidence=0.30
- `essentia` delta=0.0000 score=1.0000 confidence=0.45
- `panns_or_beats` delta=0.0000 score=0.8000 confidence=0.35

### 007_candidate_bass_kick_balance

- `muq_eval` delta=0.0000 score=1.0000 confidence=0.25
- `audiobox_aesthetics` delta=0.0000 score=1.0000 confidence=0.35
- `mert` delta=0.0000 score=1.0000 confidence=0.30
- `clap` delta=0.0000 score=0.7363 confidence=0.30
- `essentia` delta=0.0000 score=1.0000 confidence=0.45
- `panns_or_beats` delta=0.1000 score=0.9000 confidence=0.35

## Decision

clap=0.0000*0.100; panns_or_beats=0.0000*0.100; mert=0.0000*0.150; audiobox_aesthetics=0.0000*0.200; muq_eval=0.0000*0.300; essentia=0.0000*0.100; safety=1.0000*0.050; selected because best improvement 0.0100 < 0.0300

## Safety Governor

The selected candidate is accepted only after action bounds, clipping, headroom, phase, vocal clarity, compression, and identity/bleed checks pass. MuQ-Eval and other critics never write mixer parameters directly.
