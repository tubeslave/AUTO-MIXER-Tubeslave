# AutoEQ Audit Validation

## Purpose

`backend/autoeq_audit.py` is a dry-run validation layer for AutoEQ and EQ
filter packages. It does not apply filters to WING, does not import OSC clients,
and does not create live write paths.

The goal is to catch common EQ implementation mistakes before an EQ target,
IIR/FIR export, or future recommendation is allowed into a supervised review
workflow.

## What It Checks

- Unit safety: dB and linear magnitude round trips.
- Phase contract: target phase and filter phase must be explicit and compatible.
- IIR safety: biquad poles must stay inside the unit circle.
- Parametric EQ limits: max gain and Q are bounded.
- Gain budget: excessive total positive boost produces a warning and asks for
  global output trim.
- Frequency response shape: obvious narrow high-gain needles are blocked.
- Dataset split safety: train and held-out measurement IDs must not overlap.
- Export reproducibility: exported responses must match the reference response
  within tolerance.
- Level safety: reference signals can be normalized to true-peak headroom.

## Live Safety

The validator is registered as `autoeq_audit_validation` in the processing
registry with status `shadow_only`.

It can analyze and report. It cannot write live. It is intentionally outside the
`main_live_assist` executor path. The only live executor remains the supervised
manual fader/gain gate.

## Typical Use

```python
from autoeq_audit import (
    PhaseContract,
    ParametricEQBand,
    build_autoeq_audit_report,
)

report = build_autoeq_audit_report(
    phase_contract=PhaseContract(
        target_phase="minimum",
        filter_phase="minimum",
        target_source="held_out_measurement",
        filter_design="iir_fit",
        minimum_phase_reconstruction=True,
    ),
    peq_bands=[
        ParametricEQBand(frequency_hz=300.0, gain_db=-3.0, q=2.0),
    ],
    train_ids=["unit_a_position_1.wav"],
    test_ids=["unit_b_position_2.wav"],
)
assert report["passed"]
```

## Next Steps

1. Feed this audit into future AutoEQ proposal builders before any operator
   proposal is shown.
2. Add offline report artifacts with target curve version, microphone
   calibration, sample rate, phase type, filter order and headroom.
3. Keep EQ live writes blocked until a separate supervised hardware proof plan
   exists for EQ readback and rollback.
