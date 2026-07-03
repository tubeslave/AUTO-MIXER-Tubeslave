# Arrangement-Aware Level Automation

Arrangement-Aware Level Automation is a safe fader-offset planner for live mixing.
In the offline `ayaic-clean` render pipeline it is also used as the first
relative input-level stage, replacing the old first Ayaic per-channel LUFS match.
It does not normalize every source to one RMS/LUFS target. The module estimates
what is musically important in the current arrangement and calculates small
relative offsets.

## What It Analyzes

- Channel role from mixer channel names, including English and Russian aliases.
- Channel activity with hysteresis so noise and bleed do not count as performance.
- Arrangement Density Index (ADI), from active channel ratio, broadband level,
  midrange congestion, transient density, low-end occupancy, and masking pressure.
- Probable song section: intro, verse, pre-chorus, chorus, solo, breakdown, outro,
  or unknown.
- Current foreground source: lead vocal, solo instrument, lead guitar, or fallback.
- Masking sources in vocal/solo presence zones.

## ADI Formula

```text
ADI =
0.25 * active_channel_ratio
+ 0.20 * broadband_loudness_density
+ 0.20 * midrange_congestion
+ 0.15 * transient_density
+ 0.10 * low_end_occupancy
+ 0.10 * masking_pressure_on_lead
```

Labels:

- `sparse`: 0.00-0.25
- `light`: 0.25-0.45
- `medium`: 0.45-0.70
- `dense`: 0.70-0.90
- `very_dense`: 0.90-1.00

## Offset Planning

The planner calculates relative offsets:

```text
target_offset =
base_mix_position
+ section_offset
+ density_offset
+ masking_offset
+ role_priority_offset
- safety_penalty
```

Examples:

- Verse: protect lead vocal, keep guitars/keys slightly behind if they mask.
- Chorus: small vocal protection, backing vocals can rise but should stay under lead.
- Solo: lift solo source and gently reduce masking rhythm sources.
- Unknown channels: strict limits and no aggressive automation.

## Safety Limits

The safety limiter applies before OSC:

- `max_step_db_per_tick`: default `0.25 dB`.
- `deadband_db`: default `0.1 dB`.
- background analysis loop: default `250 ms`.
- live apply requires `min_confidence_for_live_apply`, default `0.65`.
- very low confidence is blocked.
- role-specific total offset limits.
- fader ceiling defaults to `0.0 dB` for automation.
- `analysis_only_mode` is enabled by default.

For offline renders, positive input offsets are additionally limited by a
4x-oversampled true-peak headroom check. If a stem cannot be raised without
crossing `-1.0 dBTP`, the report marks `true_peak_headroom_limited` and applies
no boost to that stem.

## Offline Pipeline

`automixer.production_mix_v1.ayaic_offline_pipeline` now runs:

```text
bleed_primary_signal_detection
arrangement_input_levels_pre_phase
snare_bleed_phase_alignment
offline_hpf_lpf_correction
corrective_eq_contextual_deep_eq
contextual_compression
musical_panning
musical_output_balance
ayaic_master_level
```

The report stores the first-stage decisions in `pre_phase_channel_levels`.
Each entry includes role, section, ADI, masking score, requested offset, safe
offset, applied gain, and safety reasons. Later EQ, compression, musical output
balance, and master stages remain explainable in their own report blocks.

The phase stage is documented separately in
`Docs/GLOBAL_PHASE_ALIGNMENT.md`. It now uses snare-bleed arrival alignment
first, with a global bleed/correlation graph fallback when snare evidence is
not reliable.

The compression stage is documented separately in
`Docs/CONTEXTUAL_COMPRESSION.md`. It uses a style/tempo-aware target-GR solver
and a critic pass before musical output balancing.

The panning stage is documented separately in `Docs/MUSICAL_PANNING.md`. It
places center anchors, stereo pairs, drums, support vocals, and music-bed
sources before final output balancing.

The output balance stage is documented separately in
`Docs/MUSICAL_OUTPUT_BALANCE.md`. It replaces final Ayaic channel/pair LUFS
matching with a mix-anchor output solver for foreground, foundation, support,
and image/air sources.

## Contextual Deep EQ

The default offline corrective EQ method is `contextual_deep_eq`.
It replaces the previous default `project_corrective_hybrid`, while keeping the
old method selectable through `--corrective-eq-method project_corrective_hybrid`.

The method has two decision layers:

- role correction: vocal mud/sibilance, kick boxiness/attack, bass mud/definition,
  snare boxiness/attack, tom bleed, overhead wash/harshness, music-bed mud/fizz;
- contextual masking: lead vocal presence and body are protected by cutting
  guitars, accordion, keys, playback, and backing vocals where they mask it.

Unlike the older hybrid method, `contextual_deep_eq` allows larger offline moves:

- cuts up to about `-12 dB`;
- boosts up to about `+9 dB`;
- up to seven filters per channel;
- Q up to `6.0` after stability limiting.

The method does not silently block every bold move. Instead, each EQ band can
carry:

- `regret_score`: risk score from `0.0` to `1.0`;
- `critic_notes`: warnings for large moves, high-Q filters, headroom risk, or
  overhead/room boosts;
- `reason`, `source`, `target`, `metric`, and `confidence`.

Hard blocks are reserved for invalid frequency, unstable Q, and extreme boost
headroom risk. This keeps the algorithm capable of strong corrections while
still making questionable decisions visible in the report.

## WebSocket API

Messages:

- `start_arrangement_automation`
- `stop_arrangement_automation`
- `get_arrangement_automation_status`
- `run_arrangement_automation_tick`
- `set_arrangement_automation_live_apply`

State is returned under `arrangement_automation_state` and includes:

- current section and confidence
- ADI and density label
- primary source
- active channel count
- masking sources
- planned offsets
- safe offsets
- applied offsets
- blocked offsets
- decision reasons

## Enabling Live Apply

Default config is analysis-only:

```json
"arrangement_automation": {
  "enabled": true,
  "live_apply_enabled": false,
  "analysis_only_mode": true
}
```

To apply to the mixer, both `live_apply_enabled` must be true and
`analysis_only_mode` must be false. Safety limiting still cannot be bypassed by
this setting.

## Logs

Each decision includes timestamp, channel, role, section, ADI, activity,
masking score, requested offset, final safe offset, OSC address, applied/blocked
status, and reason. This makes every fader move explainable after the show.
