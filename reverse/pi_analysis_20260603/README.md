# Pi Read-Only Reverse Analysis - 2026-06-03

## Scope

This folder records a read-only, observational analysis of the locally installed
Sound Radix Pi plug-in for Automixer method design. It does not patch,
decompile for source recreation, bypass licensing, inspect PACE internals, or
copy proprietary implementation. The goal is to extract observable architecture
and DSP ideas that can be implemented as original Automixer behavior.

## Located Plug-In

- VST3: `/Library/Audio/Plug-Ins/VST3/Pi.vst3`
- VST: `/Library/Audio/Plug-Ins/VST/Pi.vst`
- AU: `/Library/Audio/Plug-Ins/Components/Pi.component`
- Executable in each bundle: `Contents/MacOS/Pi`
- Bundle identifier: `com.SoundRadix.Pi`
- Version: `1.1.3`
- Manufacturer: `Sound Radix`
- AU name: `Sound Radix: Pi`
- AU type/subtype: `aufx` / `kase`
- Format: universal Mach-O bundle, `x86_64` and `arm64`
- Signature: Developer ID Application `SOUND RADIX LTD (9ZTHPRB4MV)`

Binary Ninja GUI was opened with the VST3 executable:

```bash
open -a "/Applications/Binary Ninja.app" \
  /Library/Audio/Plug-Ins/VST3/Pi.vst3/Contents/MacOS/Pi
```

The GUI process was active after launch. Reproducible evidence in this folder
comes from read-only macOS static tools: `file`, `lipo`, `plutil`, `codesign`,
`otool`, `nm`, `strings`, and XML inspection.

## Bundle Notes

The bundle contains PACE Eden components:

- `Contents/__Pace_Eden.bundle`
- `Contents/PlugIns/__Pace_Eden/libWrapHelper.dylib`
- `Contents/Resources/__Pace_Eden/...`

Those files were only listed as bundle metadata. They were not analyzed or
modified.

## Observable Architecture

Pi is a smaller, non-ML plug-in compared with smartEQ4. It does not show ONNX
Runtime linkage or external instrument model files. It links normal audio/UI
frameworks plus Accelerate:

- `Accelerate.framework`
- `CoreAudio.framework`
- `CoreAudioKit.framework`
- `AudioToolbox.framework`
- `CoreMIDI.framework`
- `Cocoa.framework`, `WebKit.framework`, `OpenGL.framework`
- `libc++`, `libSystem`

The filtered import list exposes only light vDSP imports (`_vDSP_vclr`,
`_vDSP_vclrD`) rather than obvious public FFT imports. That does not mean the
plug-in has no phase DSP; it means the exported import surface does not reveal a
large FFT/ONNX stack the way smartEQ4 did.

The binary exposes JUCE/VST3/AU class names and an internal protobuf-like schema:

- `Pi.proto`
- `.Pi.ParamSet`
- `.Pi.GuiSettings`
- `.Pi.GlobalConfig`
- `.Pi.GlobalParams`
- `.Pi.GroupParams`
- `.Pi.GroupEnableMode`

## Observable Parameter Surface

The Avid page-table XML and binary strings reveal this control surface:

- `Master Bypass`
- `Global Enabled`
- `Global Low Freq Focus`
- `Group`
- `Group Enable Mode`
- `Group Gain`
- `Gain`
- `Weight`

The internal string/protobuf surface adds:

- `lookahead_buffer_ms`
- `report_latency`
- `logic_latency`
- `global_config`
- `global_params`
- `group_params`
- `is_low_freq_focus`
- `is_group_graph`
- `is_freq_graph`
- `group`
- `gain`
- `weight`

Group modes observed:

- `GROUP_FULL`
- `GROUP_INTERNAL_ONLY`
- `GROUP_PHASE_LOCK`
- `Phase-Lock`
- `GROUP_OFF`

## Method Inference

The following is an inference from observed bundle metadata, parameter names,
protobuf strings, and UI XML. It is not source-level equivalence.

Pi appears to be built around group-based phase interaction control:

1. Multiple plug-in instances can be assigned to groups.
2. Each source has gain and weight parameters inside the group.
3. A global low-frequency focus mode can bias processing toward low-end phase
   interaction, where kick, bass, toms, and multi-mic drum bleed are most
   sensitive.
4. Group modes distinguish normal group behavior, internal-only group behavior,
   phase-lock behavior, and disabled behavior.
5. The plug-in has lookahead/latency reporting, so it may buffer audio and tell
   the host about processing latency.
6. The UI can show group and frequency graphs.

Simple reading: Pi is less like a classic EQ and more like a phase-aware group
coordinator. It tries to make channels in a group combine better, especially
where phase cancellation would make EQ decisions misleading.

## Automixer Relevance

For AutoMixer, the useful idea is not to clone Pi. The useful idea is to add a
phase-before-EQ guard:

1. Before boosting/cutting EQ, check whether the problem is really phase
   cancellation between related channels.
2. For drum, bass, guitar mic/DI, snare top/bottom, and stereo/overhead groups,
   compute group coherence and delay confidence.
3. Use source weight to decide which channel is the anchor and which channels
   are allowed to move.
4. If the problem is low-frequency cancellation, propose polarity/delay/phase
   alignment first, not EQ boost.
5. Keep live mode shadow/proposal-only unless a supervised hardware proof is
   completed.

## Limits

- No PACE internals were inspected.
- No plug-in runtime automation capture was performed.
- No Binary Ninja IL/HLIL is included in this report.
- No live DAW session or audio processing test was run.
- This report does not claim the full Pi algorithm. It records observable
  structure and safe Automixer design takeaways.

## Logs

Primary logs:

- `logs/plist_vst3.log`
- `logs/plist_component.log`
- `logs/pi_xml_head.log`
- `logs/pi_xml_parameter_refs.log`
- `logs/file_all_formats.log`
- `logs/lipo_archs.log`
- `logs/codesign_vst3.log`
- `logs/otool_L_vst3.log`
- `logs/otool_imports_vst3.log`
- `logs/otool_load_commands_vst3.log`
- `logs/otool_cstrings_vst3.log`
- `logs/nm_arm64_dsp_filtered.log`
- `logs/strings_dsp_filtered.log`
- `logs/strings_proto_surface.log`
- `logs/bundle_files_vst3.log`
- `logs/pace_bundle_files_vst3.log`

Focused extracts:

- `extracts/plugin_paths_and_archs.txt`
- `extracts/dependencies_extract.txt`
- `extracts/dsp_imports_extract.txt`
- `extracts/pi_xml_parameter_ids.txt`
- `extracts/pi_proto_param_surface.txt`
- `extracts/parameter_surface_extract.txt`
- `extracts/disassembly_relevant_refs.txt`
