# Contextual Compression

`contextual_compression` is the offline compressor stage used by the
`ayaic-clean` render pipeline. It runs after corrective EQ and before the final
channel/pair level matching.

## Goal

The method calculates compression per channel from measured signal behavior,
instrument role, style, and tempo. It does not simply load a fixed preset.

It is designed for:

- vocal leveling without pushing every word equally loud;
- kick, snare, and tom control while preserving transient punch;
- bass stability;
- gentle guitar, keys, pad, playback, overhead, and room control;
- explainable reports for every automatic decision.

## Pipeline Position

```text
arrangement_input_levels_pre_phase
snare_bleed_phase_alignment
offline_hpf_lpf_correction
corrective_eq_contextual_deep_eq
contextual_compression
musical_panning
musical_output_balance
ayaic_master_level
```

Musical panning is applied after compression so dynamics changes do not distort
the intended stereo scene. Musical output balance is applied after panning so
pan-related L/R and mono changes do not become the final mix balance by
accident.

## Metrics

For each channel, the module measures:

- integrated loudness;
- RMS and peak;
- crest factor;
- frame loudness range;
- activity ratio;
- transient density and transient strength;
- spectral flux.

The tempo is either supplied with `--compression-bpm` or estimated from the
summed transient envelope. If the tempo estimate is weak, the stage uses
`120 BPM` and records that fallback in the report.

## Parameter Solver

The method chooses a role intent, then adapts it with style and tempo:

- `target_gr_db`;
- `max_gr_db`;
- ratio;
- attack;
- release;
- knee;
- detector type.

Threshold is not guessed from a preset. It is solved by searching for the
threshold that reaches the target average gain reduction on active frames, while
respecting role-specific peak gain-reduction limits.

Release is tempo-aware:

```text
beat_ms = 60000 / bpm
release = blend(role_base_release, beat_ms / role_release_divisor)
```

Style presets:

- `live_pop_rock`;
- `rock`;
- `dense`;
- `ballad`;
- `transparent`.

## Critic

After simulation, a critic checks:

- excessive gain reduction;
- transient loss;
- pumping risk;
- peak growth.

If the critic finds a risky move, the plan is softened by lowering target gain
reduction and ratio, lengthening attack, and lengthening release. The revision
is logged instead of silently hiding the decision.

## CLI

Enabled by default for `ayaic-clean`.

```bash
python -m automixer.production_mix_v1 \
  --scheme ayaic-clean \
  --input-dir "/path/to/stems" \
  --output-dir "/path/to/out" \
  --compression-style live_pop_rock \
  --compression-bpm 120
```

Useful switches:

```text
--disable-contextual-compression
--compression-report-only
--compression-style {ballad,dense,live_pop_rock,rock,transparent}
--compression-bpm <number>
```

## Report Fields

The render report includes:

- `contextual_compression`;
- `compression`;
- `post_compression_channel_levels`;
- `post_compression_pair_levels`;
- per-stem `compression_notes`.

Each channel report includes metrics, intent, final parameters, gain-reduction
statistics, critic result, and pre/post loudness/peak values.

## Evidence Base

The compressor model follows the feed-forward dynamic range compressor family
described in AES literature. The target-GR and feature-driven adaptation are
based on established automatic mixing research, but exact style/tempo tables are
engineering heuristics. For a universal proof that one compressor setting table
is optimal across styles, there is no strong evidence.

References:

- D. Giannoulis, M. Massberg, and J. D. Reiss, "Digital Dynamic Range
  Compressor Design - A Tutorial and Analysis", AES, 2012.
- D. Giannoulis, M. Massberg, and J. D. Reiss, "Dynamic range compression
  automation", Journal of the Audio Engineering Society, 2013.
- M. Maddams, S. Finn, and J. D. Reiss, "An autonomous method for multi-track
  dynamic range compression", DAFx, 2012.
