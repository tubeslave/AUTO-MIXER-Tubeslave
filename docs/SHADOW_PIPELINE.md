# Guarded / shadow mixing

## Scope and default

This adds an opt-in safety pipeline to the existing AutoSoundcheckEngine. The
existing installation and `mode: off` behavior remain compatible. It does not
connect a console, deploy settings, train on a user's mixes, or enable writes
merely by being installed. No approved-mix dataset is bundled or fabricated.

The actual production path is:

`analysis -> typed proposal -> runtime permissions -> numeric bounds -> virtual
render -> objective -> DecisionGuard -> readback recheck -> write permit ->
transport submission -> guard-history commit`

All typed tonal decisions are reviewed, not only shared-mix EQ. A separate
GuardedMixerClient denies engine writes that bypass that path. Existing
emergency cuts and rollback have explicit review exemptions, remain bounded,
and retain priority. In shadow mode **even emergency actions never write**.
Shadow mode is an observer, not an active feedback-protection system.

## Install

```bash
# DSP/live runtime, no torch/transformers/nnAudio/RAG requirement:
python -m pip install -r backend/requirements-core.txt
# Optional learned features:
python -m pip install -r backend/requirements-ml.txt
# Optional RAG and matchering integration:
python -m pip install -r backend/requirements-extras.txt
# Historical full install still installs the same package set:
python -m pip install -r backend/requirements.txt
```

The full and core stacks are tested separately on Python 3.10, 3.11 and 3.12.
Optional ML tests may skip on the core stack. CPU torch is installed explicitly
in full-stack CI; this is not a test of GPU hardware or downloaded model quality.
The unrelated existing optional research gitlinks are preserved. The confirmed
orphan `external/AutomaticMixingPapers` was removed without touching runtime DSP.

## Target Corridor

`target_corridor.py` accepts explicitly approved, context-matched records with
`mix_id`, `approved: true`, `context`, `feature_version: shadow-mix-v1`, and a
finite, complete `features` mapping. Each mix ID has one vote per context.
At least five distinct approved mixes must survive validation and row-wise
outlier rejection. Median/MAD estimation prevents one extreme row from defining
the target; there is no guarantee against a majority of bad training examples.

The objective is zero inside each interval and increases outside it. Features
are relative compensated spectral density, LUFS, crest and, when separate
sources exist, a lead/masker power ratio. This is not a universal ideal spectrum
or a perceptual-quality score. Genre/song-section context is supplied explicitly.
Do not pool unrelated arrangements into one target. Automatic live decisions
are never re-labelled as approved training examples.

An approved WAV manifest is a JSON array, for example:

```json
[
  {"mix_id": "song-01-chorus", "approved": true,
   "context": "rock:chorus", "audio_path": "approved/song-01-chorus.wav"}
]
```

Provide at least five actual, distinct approved mixes, not five aliases of one
file. WAV content duplicates are excluded by the CLI. Paths are relative to the
manifest. For manually supplied features, identity/provenance is the caller's
responsibility. Stereo references cannot provide a separated-source masking
feature; use source-aware records for that feature.

```bash
PYTHONPATH=backend python -m shadow_mix_cli fit \
  --manifest approved.json --context rock:chorus --output targets.json
```

The target JSON is versioned and validates context, finite ordered intervals,
and extractor compatibility on load. An absent target blocks tonal approval;
the old hand-written display corridor is not passed off as learned data.

## Renderer and model contract

`shadow_renderer.py` performs actual RBJ bell filtering, Butterworth HPF,
reference linked-peak compression, fader multiplication, constant-power mono
panning and stereo summation. Each candidate replaces the selected physical
setting, then renders from the original capture. It does not append a fictitious
delta filter, normalize the result, clip it into compliance, or silently bypass
failed processing. EQ bypass is preserved because `set_eq_band` does not enable
EQ on WING/dLive. Mono/stereo signal layout is explicit and filter state is
independent across columns.

The required live tap is **after input trim, input polarity and input delay,
before HPF/EQ/dynamics**. Buffers must be aligned, complete, finite, sample-major,
and of equal length. A verified direct-to-main sum is required. Current state
must include all four EQ bands, enable flags, faders, pan, main routing and
processing state. Unknown readback, unsupported routing, gates, parallel
compression, active bus/DCA routing or protected feedback bands block approval.
Inserts, shelves, FX and bus/master processing must be verified bypassed. HPF
order and the compressor model must be explicitly specified before changing
them. dLive's minimal MIDI readback is not sufficient for a complete tonal
simulation; it is rejected rather than filled with assumed defaults.

The compressor is a reference implementation, **not a bit-exact model of any
console**. Filter/envelope histories start at rest on the captured window. The
4x polyphase true-peak estimate includes the original sample peak; it is not a
certified meter. Default headroom is -1.25 dBTP, with a separate per-channel
ceiling, a 1 LU change bound and a 1.5 dB crest-loss bound. These tests cannot
establish ADC clipping recovery, real room acoustics, calibrated console model
accuracy, or listening preference.

Live guarded operation requires `model_validated: true` only AFTER a measured
hardware-in-the-loop validation of the tap, routing, pan law, filter responses
and dynamics against the reference model. Setting the flag is an operator
acknowledgement, not automatic certification. No hardware validation is claimed
by this change. Without it, use offline audit or shadow observation.

## Decision Guard and write boundary

`decision_guard.py` subtracts an uncertainty penalty from the objective gain,
requires a minimum improvement, applies per-parameter deadbands and requires
fresh-frame confirmations. Replayed frames do not count twice. Reversals have
a hold time and a higher improvement threshold afterward. Changes, proposals
and successful submissions have separate state. A rejected decision or a
transport failure does not advance applied history or the rate-limit budget.
History is bounded and a regressing/non-finite clock fails closed.

Temporal dispersion of paired subwindow improvements is an extra uncertainty
penalty, not a statistically calibrated confidence interval. A model uncertainty
floor is configurable. Default maximum render age is two seconds; stale reviews
fail closed. Analysis runs in the engine's control path, never an audio callback.
A successful transport method return is a submission, **not a hardware readback
acknowledgement**; existing post-action evaluation/rollback remains in place.

`AutoFOHSafetyController` serializes bounds/review/send/history, rejects NaN/Inf
before clamping, reviews the final bounded action, and rechecks bounds after
rendering. `GuardedMixerClient` permits a typed approved submission only inside
that safety boundary; other engine writes return false. Shadow mode additionally
wraps the engine client with its observation proxy, including raw MIDI/packet
write paths. Reads/subscriptions/keepalives remain possible. This does not block
manual operators or unrelated clients from operating the console.

Modes in `autofoh.shadow`:

- `off`: backward-compatible behavior.
- `shadow`: review/log, simulated decisions have `sent=false`, no engine writes.
- `guarded`: only approved modelled actions may submit; unknown models fail closed.

Use `config/autofoh-shadow.example.yaml` as an override template. Do not mark
unknown capture/routing facts as verified merely to remove a rejection. The
reviewer is installed in the engine's real connection path, not only in tests.

## Offline audit command

A session JSON contains `sample_rate`, `channels` (each with `audio_path` and a
RenderChannel `state`) and an `action`. For example, a mono fader proposal:

```json
{
  "sample_rate": 48000,
  "channels": [{"audio_path": "bass.wav", "state": {
    "channel_id": 1, "fader_db": -6, "eq_enabled": false,
    "eq_bands": {}, "hpf_hz": null, "compressor": null,
    "muted": false, "role": "bass", "pan": 0
  }}],
  "action": {"type": "ChannelFaderMove", "channel_id": 1,
             "target_db": -6.5, "reason": "offline trial"}
}
```

```bash
PYTHONPATH=backend python -m shadow_mix_cli audit \
  --session session.json --target targets.json --report report.json \
  --preview candidate.wav
```

An accepted preview is 32-bit float WAV at the input sample rate. Denied
proposals do not produce a preview. Exit codes: 0 approved, 2 rejected, 1 invalid
input/configuration. The offline CLI deliberately uses one observation and does
not constitute live guard calibration or permission to send commands.

## Regression audit

Reproductions were run against the unchanged source and then against the fixes:
HPF left-to-right filter-state leakage, HPF stereo chunk inconsistency,
non-finite EQ-limiter poisoning/false moves, mutation of live-planner readback,
and observation-proxy raw-write bypass. The old sources fail all seven
parameterized reproductions; the fixed sources pass them. Existing EQ/masking
and dynamics invariants remain part of the DSP audit. No additional masking
algorithm rewrite is claimed without a reproduced defect.

Additional tests cover learned-data outliers, missing approval/context,
non-finite state/actions/audio, clipping, actual rendered EQ/HPF/compression,
slot replacement, bypass semantics, anti-phase stereo measurement, stale
reviews, objective regression, reversal oscillation, failed transport, raw-write
blocking, production connection wiring and zero shadow writes.

## Primary implementation references

- RBJ audio EQ equations: https://www.w3.org/TR/audio-eq-cookbook/
- Independent SOS state/axis contract: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.sosfilt.html
- Polyphase reconstruction: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample_poly.html

The reference equations and API contracts do not validate the console-specific
model assumptions listed above.
