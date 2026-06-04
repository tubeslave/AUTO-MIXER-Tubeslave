# Dynamic EQ Playbook Bridge

## Why This Exists

Diff-DEQ and similar research models point in the right direction for Automixer:
instead of treating EQ and compression separately, they predict dynamic
frequency-band behavior from the signal itself.

For this project, the safe first step is not to import a neural model or make
the WING change EQ live. The safe first step is to accept external PEQ+DRC
predictions as offline research data, validate them, and store them as dry-run
playbook entries.

## Current Scope

`backend/dynamic_eq_playbook.py` supports JSON-like payloads with six dynamic EQ
bands. Each band may contain:

- center frequency;
- Q;
- PEQ gain;
- DRC threshold;
- DRC ratio;
- attack;
- release;
- optional parameter ramp/smoothing time;
- dynamic gain range;
- makeup gain.

The bridge then checks:

- channel id and sample rate are valid;
- expected band count is present;
- model confidence is high enough for review;
- PEQ gain and Q stay within safety limits;
- DRC threshold, ratio, attack and release stay inside sane ranges;
- attack, release and optional parameter ramps are not faster than configured
  audio-buffer guards;
- dynamic gain range and makeup gain do not exceed the configured budget.

## Buffer Timing Guard

The default safety config assumes a `256` sample buffer and uses the
prediction sample rate to derive timing floors:

- attack must be at least one buffer;
- release must be at least two buffers;
- optional parameter ramps must be at least five buffers.

Invalid buffer guard config fails closed: a non-positive buffer size or ramp
multiplier blocks the playbook entry instead of silently disabling the timing
check.

This is intentionally a shadow/offline validation rule. Failing it blocks the
playbook entry from being treated as usable research evidence, but it does not
create any live EQ or dynamics write path.

## Safety Boundary

The bridge is registered as `diff_deq_playbook_bridge` with status
`shadow_only`.

It never writes to WING. It never imports OSC clients. It does not run the
external model. It only validates already-produced predictions.

## Example Payload

```json
{
  "channel_id": 12,
  "source_model": "Diff-DEQ",
  "model_version": "research_import_v0",
  "instrument_role": "lead_vocal",
  "sample_rate": 24000,
  "confidence": 0.82,
  "bands": [
    {
      "band_index": 1,
      "center_hz": 120,
      "q": 1.2,
      "peq_gain_db": 1.0,
      "threshold_db": -24,
      "ratio": 2.0,
      "attack_ms": 12,
      "release_ms": 120,
      "parameter_ramp_ms": 60,
      "dynamic_gain_range_db": 3,
      "makeup_gain_db": 0.5
    }
  ]
}
```

## Next Safe Step

Use exported predictions from an external Diff-DEQ-style experiment as fixtures:

1. run the model outside Automixer;
2. export predicted PEQ+DRC parameters to JSON;
3. feed JSON into `build_dynamic_eq_playbook(...)`;
4. compare the playbook against Automixer telemetry and AutoEQ audit results;
5. only then decide whether a proposal-only cockpit view is useful.

Live EQ/dynamics control remains blocked until a separate supervised hardware
proof exists for EQ readback, dynamics readback and rollback.
