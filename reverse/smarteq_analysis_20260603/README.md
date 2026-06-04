# smartEQ4 Read-Only Reverse Analysis - 2026-06-03

## Scope

This folder records a read-only, observational analysis of the locally installed
Sonible smartEQ4 plug-in for Automixer method design. It does not patch,
decompile for source recreation, bypass licensing, or copy proprietary model
weights. The goal is to extract reusable architecture and DSP ideas that can be
implemented as an original Automixer method.

The user asked for "Smart EQ from Sony"; the installed plug-in found in the
standard macOS Audio plug-in folders is Sonible `smartEQ4`, not Sony.

## Located Plug-In

- VST3: `/Library/Audio/Plug-Ins/VST3/smartEQ4.vst3`
- VST: `/Library/Audio/Plug-Ins/VST/smartEQ4.vst`
- AU: `/Library/Audio/Plug-Ins/Components/smartEQ4.component`
- Executable in each bundle: `Contents/MacOS/smartEQ4`
- Bundle identifier: `com.sonible.smarteq4`
- Version: `1.0.2`
- Format: universal Mach-O bundle, `x86_64` and `arm64`

Binary Ninja GUI was opened with the VST3 executable:

```bash
open -a "/Applications/Binary Ninja.app" \
  /Library/Audio/Plug-Ins/VST3/smartEQ4.vst3/Contents/MacOS/smartEQ4
```

The GUI process started successfully. A headless Binary Ninja API check was
also attempted, but the process did not return useful output and was stopped.
The reproducible evidence in this folder is therefore based on macOS static
tools: `file`, `lipo`, `plutil`, `codesign`, `otool`, `nm`, and `strings`.

## Observable Architecture

The plug-in is a large JUCE-style audio plug-in with Apple audio/UI framework
links, Accelerate/vDSP DSP imports, ONNX Runtime linkage, and Sonible support
files under `/Library/Application Support/sonible`.

Observed dependencies:

- `Accelerate.framework`
- `CoreAudio.framework`, `AudioToolbox.framework`, `CoreMIDI.framework`
- `WebKit.framework`, `Cocoa.framework`, UI-related Apple frameworks
- `/Library/Application Support/sonible/lib/libonnxruntime.1.15.1.dylib`
- `libcurl`, `libresolv`, `libc++`, `libSystem`

Observed DSP/ML imports include:

- `_OrtGetApiBase`
- `_vDSP_create_fftsetup`, `_vDSP_destroy_fftsetup`
- `_vDSP_fft_zop`, `_vDSP_fft_zrip`
- `_vDSP_zvmags`, `_vDSP_meanv`, `_vDSP_maxv`, `_vDSP_minv`, `_vDSP_sve`
- `_vDSP_vadd`, `_vDSP_vsub`, `_vDSP_vmul`, `_vDSP_vsma`, `_vDSP_vsmul`
- `_vDSP_vthr`, `_vvlog10f`, `_vvsqrtf`, `_vvtanhf`

The installed Sonible support directory is large (`1.6G`) and includes
instrument-specific `smartEQ4` profile modules:

- `Acousticguitar.nn`
- `Bass.nn`
- `Cello.nn`
- `Contrabass.nn`
- `Drums.nn`
- `Hihat.nn`
- `Piano.nn`
- `Snare.nn`
- `Speechhigh.nn`
- `Speechlow.nn`
- `Universal.nn`
- `Vocalshigh.nn`
- `Vocalslow.nn`

These `.nn` files are not plain data files. `file` reports them as universal
Mach-O dynamically linked shared libraries. This suggests an architecture where
the plug-in loads separate instrument/profile modules rather than one small
embedded static curve.

## Observable Parameter Surface

Filtered strings reveal a parameter schema with:

- analyzer controls: `analyzer_enabled`, `analyzer_mode`, `analyzer_range`,
  `analyzer_speed`, `analyzer_tilt`
- learning/profile controls: `learner_learn`, `learner_learn_all`,
  `learner_profile_id`
- group controls: `xcf_group_enabled`, `xcf_group_impact`
- global controls: `general_auto_gain_enabled`, `general_auto_gain_mode`,
  `general_gain_output_external`, `general_gain_output_internal`,
  `general_ms_side_pan`, `general_ms_side_solo`
- linear phase switch: `eq_use_linear_phase`
- 24 EQ band slots: `eq_band00` through `eq_band23`
- per-band fields:
  - `active`
  - `attack`
  - `channelmode`
  - `dynamic`
  - `dynamic_gain_range`
  - `enabled`
  - `frequency`
  - `gain`
  - `quality`
  - `ratio`
  - `release`
  - `threshold`
  - `type`
- two smart-weight regions: `eq_smartweight0` and `eq_smartweight1`, with
  `active`, `adaptive`, `enabled`, `frequency`, `mode`, `slope_hi`,
  `slope_lo`, `smoothing`, `strength`, and `width`

## Method Inference

The following is an inference from observed imports, strings, and installed
support files, not a claim of full source-level equivalence.

The practical architecture appears to be:

1. Analyze incoming audio with block FFT / spectrum statistics.
2. Select or learn an instrument profile (`learner_profile_id`) from
   instrument-specific modules.
3. Produce a target spectral correction using 24 possible EQ bands.
4. Apply static and dynamic band behavior with threshold, ratio, attack,
   release, and dynamic gain range.
5. Use one or two smart-weight regions to bias correction toward musically
   important ranges.
6. Support group-aware behavior through `xcf_group_enabled` and
   `xcf_group_impact`.
7. Optionally apply auto gain and linear-phase behavior.

## Limits

- Binary Ninja GUI was opened, but this report does not include BN IL/HLIL
  evidence.
- The headless BN API attempt did not produce useful output before it was
  stopped.
- The `.nn` modules were inspected only at metadata level. Their internals are
  proprietary and should not be copied.
- No live plug-in execution, DAW automation capture, patching, license bypass,
  or model-weight extraction was performed.
- The implementation target for AutoMixer must remain an original method:
  profile-inspired curves, bounded dynamic EQ proposals, and shadow/audit-first
  behavior.

## Logs

Primary logs:

- `logs/plist_vst3.log`
- `logs/file_all_formats.log`
- `logs/lipo_archs.log`
- `logs/codesign_vst3.log`
- `logs/otool_L_vst3.log`
- `logs/otool_imports_vst3.log`
- `logs/nm_arm64_dsp_filtered.log`
- `logs/strings_dsp_filtered.log`
- `logs/sonible_smarteq4_support_files.log`
- `logs/sonible_smarteq4_nn_file_types.log`
- `logs/sonible_smarteq4_nn_sizes.log`
- `logs/smarteq4_nn_otool_L_short.log`
- `logs/onnxruntime_file.log`
- `logs/onnxruntime_otool_L.log`

Focused extracts:

- `extracts/plugin_paths_and_archs.txt`
- `extracts/dsp_ml_imports_extract.txt`
- `extracts/parameter_surface_extract.txt`
- `extracts/smarteq4_profile_families.txt`
