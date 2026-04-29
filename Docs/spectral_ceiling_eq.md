# Spectral Ceiling EQ

`spectral_ceiling_eq` is a safe, explainable EQ proposal layer for AutoFOH and
the offline mix-agent. It uses role-aware spectral slopes, broad musical zones,
roll-off guides, and foreground/background intent to suggest small EQ moves.
It is part of the main no-reference offline mixing pipeline by default.

It is inspired by public ideas around spectral ceilings and noise-slope guides.
It does not copy Ayaic/COS Pro proprietary logic, does not run aggressive match
EQ, and does not replace the existing AutoFOH EQ presets, masking detectors,
phase guards, safety controller, or offline evaluation.

## How It Works

1. Normalize the channel or stem role.
2. Select a profile from `configs/spectral_ceiling_profiles.yaml`.
3. Measure a smoothed long-term spectrum on log-frequency bins.
4. Build a white, pink, brown, or custom dB/oct target curve around 1 kHz.
5. Compare broad zones such as mud, presence, sibilance, air, and vocal conflict.
6. Propose small EQ moves and roll-off guides.
7. In live AutoFOH, merge compatible moves into the existing `_apply_eq` target
   before the normal safety controller sends anything.
8. In offline mix-agent, emit `MixAction` suggestions and render only actions
   whose mode is `safe_apply`.

## Config

Main config:

```yaml
spectral_ceiling_eq:
  enabled: true
  dry_run: false
  correction_strength: 0.4
  smoothing_octaves: 0.333
  reference_freq_hz: 1000
  max_bands_per_track: 5
  min_confidence_to_apply: 0.65
  allow_master_bus_eq: true
  master_bus_max_abs_gain_db: 1.0
  max_abs_gain_db: 3.0
  log_verbose: true
```

Disable it:

```yaml
spectral_ceiling_eq:
  enabled: false
```

Log only:

```yaml
spectral_ceiling_eq:
  enabled: true
  dry_run: true
```

## Supported Roles

Profiles currently cover `lead_vocal`, `backing_vocal`, `kick`, `snare`,
`tom`, `floor_tom`, `hihat`, `ride`, `bass`, `electric_guitar`,
`acoustic_guitar`, `keys`, `synth`, `percussion`, `overheads`, `drums_bus`,
`room`, `fx_return`, `playback`, `mix_bus`, and `unknown`.

Use `overheads` only for OH/cymbal microphones. Use `drums_bus` for the summed
kit or drum group so the spectral ceiling preserves kick/body instead of
applying the overhead low-cut profile.

`playback` is the support profile for combined synth/percussion/pad or tracks
stems. The offline loader maps filenames containing synth + percussion/perc +
pad/pads to this role, and pad-only stems are treated as playback support for
this heuristic.

Each profile can define:

- `guide`: `white`, `pink`, `brown`, or `custom`
- `slope_db_per_oct`
- `front_back_position`: `foreground`, `midground`, or `background`
- `low_cut_hz` and `high_cut_hz`
- `important_zones`
- `avoid`
- `max_eq_gain_db`
- `max_eq_cut_db`

## Reading Logs

Human logs look like:

```text
[SPECTRAL_CEILING_EQ]
track: Lead Vocal
role: lead_vocal
profile: lead_vocal_pop_presence
measured_tilt: -1.8 dB/oct
target_tilt: -2.0 dB/oct
decision:
  - high-pass guide around 95 Hz
  - cut -1.2 dB at 268 Hz (mud) because energy exceeds target ceiling
  - no change: harshness: existing EQ decision already addresses zone
confidence: 0.78
applied: true
```

Structured JSONL events use event type `spectral_ceiling_eq` and include the
proposal, skipped reasons, merge report, confidence, and safety metadata.

## Safety Limits

- Broad smoothed spectrum only; no raw FFT peak chasing.
- Default correction strength is 0.4.
- Per-band boosts and cuts are profile-limited.
- Master bus correction is limited to ±1 dB by default.
- `dry_run` logs without changing audio.
- Low confidence or unknown role logs only.
- Live changes still pass through AutoFOH runtime policy, phase target guards,
  broad EQ rate limits, existing-processing preservation, and mixer bounds.
- Vocal demasking only activates when a lead vocal is present with enough
  confidence.

## Debug CLI

```bash
python -m automixer.tools.inspect_spectral_ceiling \
  --input path/to/audio.wav \
  --role lead_vocal
```

Use `--dry-run` to force log-only inspection and `--json` for machine-readable
output.

## Adding Profiles

Add a new role under `configs/spectral_ceiling_profiles.yaml`. Prefer broad
zones over narrow notch frequencies. Keep boosts smaller than cuts, define
`front_back_position`, and add avoid zones for masking or harshness before
adding presence/air boosts.
