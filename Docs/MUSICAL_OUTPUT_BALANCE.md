# Musical Output Balance

`musical_output_balance` is the offline output-level stage used by the
`ayaic-clean` render pipeline. It replaces the previous final
channel/pair LUFS matching pass.

## Goal

The method builds a musical mix balance instead of normalizing every channel to
a fixed loudness target.

It balances around mix anchors:

- foreground: lead vocal or the current melodic source;
- foundation: kick and bass;
- rhythm foreground: snare and toms;
- music bed: guitars, keys, pads, playback, accordion;
- image/air: overheads and room;
- support vocals: backing vocals below the lead.

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

The panning stage runs before output balance because it changes L/R energy,
mono fold-down, and peak summing. This stage therefore balances the already
panoramic mix.

## Method

1. Measure each stem after EQ and compression:
   - primary LUFS;
   - full-track LUFS;
   - peak;
   - activity ratio;
   - low-mid and presence-band energy.
2. Select the primary source:
   - lead vocal when active;
   - otherwise a melodic foreground candidate;
   - otherwise the highest-priority active source.
3. Estimate a broad section label:
   - `verse`;
   - `chorus_or_pre_chorus`;
   - `solo_or_instrumental`;
   - `unknown`.
4. Build role-relative output offsets around the primary source.
5. Add density, section, style, masking, and bus corrections.
6. Link stereo pairs before applying offsets.
7. Apply safety:
   - role-specific min/max offsets;
   - strict unknown-channel limits;
   - per-channel peak headroom for boosts;
   - global mix-peak growth guard that scales all offsets toward zero when the
     proposed balance would raise the summed peak too much;
   - deadband for negligible moves.
8. Run a critic:
   - mix peak growth;
   - foreground margin;
   - large output boosts.

## Important Behavior

The solver prefers lowering masking or overly loud bed sources instead of
boosting the lead vocal into the master ceiling. This is intentional: it leaves
more headroom and reduces the amount of gain the master stage must remove.

Stereo pairs such as guitars, playback, and overheads receive a linked median
offset so the stereo image does not tilt.

## Report Fields

The render report includes:

- `musical_output_balance`;
- `output_balance`;
- `post_compression_channel_levels`;
- `post_compression_pair_levels`;
- per-stem `output_balance_notes`.

The main report contains:

- `primary_source`;
- `arrangement_density_index`;
- `arrangement_density_label`;
- `channel_offsets`;
- `bus_offsets`;
- `blocked_offsets`;
- `critic`.

## CLI

Enabled by default for `ayaic-clean`.

```bash
python -m automixer.production_mix_v1 \
  --scheme ayaic-clean \
  --input-dir "/path/to/stems" \
  --output-dir "/path/to/out" \
  --output-balance-style live_pop_rock
```

Useful switches:

```text
--disable-musical-output-balance
--output-balance-report-only
--output-balance-style {ballad,dense,live_pop_rock,rock,transparent}
```

## Evidence Base

Automatic fader and level balancing is supported by automatic mixing literature,
but fixed universal level tables for every instrument/style are not strongly
supported. This implementation therefore uses a solver plus critic rather than a
hard-coded "ideal mix" table.

References:

- E. Perez-Gonzalez and J. D. Reiss, "Automatic gain and fader control for live
  mixing", WASPAA, 2009:
  https://www.eecs.qmul.ac.uk/~josh/documents/2009/PerezReiss-WASPAA2009-AutomaticGainandFaderControlForLiveMixing_001.pdf
- J. Mansbridge, S. Finn, and J. D. Reiss, "Implementation and evaluation of
  autonomous multi-track fader control", AES, 2012:
  https://www.eecs.qmul.ac.uk/~josh/documents/2012/MansbridgeFinnReiss-AES1322012-AutoMultitrackFaders.pdf
- J. D. Reiss and O. Brandtsegg, "Applications of Cross-Adaptive Audio Effects:
  Automatic Mixing, Live Performance and Everything in Between", Frontiers in
  Digital Humanities, 2018:
  https://www.frontiersin.org/journals/digital-humanities/articles/10.3389/fdigh.2018.00017/full

For universal instrument/style balance targets, there is no strong evidence.
