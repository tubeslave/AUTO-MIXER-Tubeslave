# Musical Panning

`musical_panning` is the offline panorama stage used by the `ayaic-clean`
render pipeline. It places instruments before final output-level balancing, so
the balance solver hears the post-pan stereo image.

## Goal

The method builds a stable stereo scene instead of applying a fixed pan table to
every channel.

It protects mix anchors:

- lead vocal: center;
- kick and bass: center;
- snare: near center;
- stereo pairs: linked left/right positions;
- guitars, keys, pads, playback, and backing vocals: spatial separation;
- overheads and room: image/air, limited by mono and L/R safety.

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

Panning is placed after compression because dynamics can change the apparent
foreground. It is placed before output balance because pan decisions change L/R
energy, summed peak, and mono fold-down.

## Method

1. Measure each stem after EQ and compression:
   - LUFS;
   - peak;
   - activity;
   - low, low-mid, and presence energy;
   - stereo correlation;
   - dual-mono detection.
2. Detect stereo pairs from channel names ending in `L/R` or `Left/Right`.
3. Assign a role-aware target scene:
   - center locks for vocal/kick/bass;
   - linked stereo pairs;
   - alternating support sources;
   - moderate drum and overhead positions.
4. Apply spatial unmasking hints:
   - presence-heavy bed sources can move slightly farther from center when lead
     vocal is active.
5. Apply safety:
   - role-specific max pan;
   - low-end center guard;
   - strict unknown-channel pan limit;
   - global mix-peak growth guard;
   - L/R imbalance guard;
   - mono fold-down loss guard.
6. Run a critic:
   - center anchors must not move;
   - L/R imbalance must remain bounded;
   - mono fold-down loss must remain bounded;
   - mix peak must not grow too much.

## Report Fields

The render report includes:

- `musical_panning`;
- `panning`;
- `channel_pans`;
- `pair_pans`;
- `blocked_pans`;
- per-stem `pan_notes`.

Each channel report includes:

- role;
- before/requested/final pan;
- WING-style `pan_100`;
- stereo pair id;
- activity and spectral metrics;
- applied/blocked status;
- explanation.

## CLI

Enabled by default for `ayaic-clean`.

```bash
python -m automixer.production_mix_v1 \
  --scheme ayaic-clean \
  --input-dir "/path/to/stems" \
  --output-dir "/path/to/out" \
  --panning-style live_pop_rock
```

Useful switches:

```text
--disable-musical-panning
--panning-report-only
--panning-style {live_pop_rock,narrow,rock,transparent,wide}
```

## Evidence Base

Automatic panning has support in automatic mixing literature, especially around
source balance, spatial balance, spectral balance, and unmasking. Fixed
universal pan layouts are not strongly supported, so this implementation uses a
role scene plus critic rather than a single hard-coded pan map.

References:

- E. Perez-Gonzalez and J. D. Reiss, "A Real-Time Semiautonomous Audio Panning
  System for Music Mixing", EURASIP Journal on Advances in Signal Processing,
  2010:
  https://asp-eurasipjournals.springeropen.com/articles/10.1155/2010/436895
- J. Mansbridge, S. Finn, and J. D. Reiss, "An Autonomous System for
  Multi-track Stereo Pan Positioning", AES, 2012:
  https://www.eecs.qmul.ac.uk/~josh/documents/2012/MansbridgeFinnReiss-AES133-Autonomoussystemformultitrackstereopositioning.pdf
- S. Tom, J. D. Reiss, and P. Depalle, "An automatic mixing system for
  multitrack spatialization for stereo based on unmasking and best panning
  practices", AES, 2019:
  https://eecs.qmul.ac.uk/~josh/documents/2019/20311.pdf

For universal instrument/style pan targets, there is no strong evidence.
