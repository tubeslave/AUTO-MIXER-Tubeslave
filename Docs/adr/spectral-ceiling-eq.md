# Spectral Ceiling EQ Heuristic

## Context

The automixer already has fixed instrument EQ presets, adaptive EQ metrics,
masking detectors, live shared-mix planning, and a safety controller. The new
requirement is to add a role-aware spectral-shape heuristic inspired by public
spectral ceiling and noise-slope guide concepts, without copying proprietary
logic or replacing the approved pipeline.

## Options considered

1. Replace existing AutoFOH EQ with a spectral matching system.
2. Add a separate safety-executed action pass after existing EQ.
3. Add a spectral ceiling proposal module and merge compatible moves into the
   current EQ decision before safety execution, with offline proposals using the
   same module.

## Decision

Use option 3. The spectral ceiling module is independent and configurable. It
returns structured proposals, logs its reasoning, and leaves application to the
existing live safety controller or conservative offline renderer.

## Why this won

It preserves known AutoFOH behavior and avoids double-applying mud, presence or
air moves. It also avoids immediate broad-EQ rate-limit collisions in live mode
because the spectral proposals are merged into the same four-band EQ decision
that the current pipeline already sends through safety bounds.

## Rejected alternatives

- Proprietary COS Pro reproduction: rejected for licensing, safety, and
  correctness reasons.
- Aggressive match EQ: rejected because narrow or full-spectrum matching can
  overfit noise, bleed and short analysis windows.
- Master-bus-first correction: rejected because source/stem fixes should be
  preferred and master bus movement is limited to ±1 dB by default.

## Implementation plan

- Load role profiles from `configs/spectral_ceiling_profiles.yaml`.
- Generate white/pink/brown/custom slope targets around 1 kHz.
- Analyze smoothed log-frequency spectra and broad musical zones.
- Propose bounded EQ bands, HPF/LPF suggestions, skipped reasons and safety
  metadata.
- Merge compatible live moves into `AutoSoundcheckEngine._apply_eq`.
- Add offline `MixAction` proposals in `mix_agent`.
- Add a debug CLI and documentation.

## Test plan

- Focused pytest module for spectral guide math, profile selection, clamps,
  dry-run, foreground/background behavior, vocal demasking, and master bus
  limits.
- Offline pipeline regression through existing mix-agent tests.
- Standard repository command:
  `PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`

## Risks and rollback

Wrong role detection can suggest the wrong tonal shape. Roll back by setting
`spectral_ceiling_eq.enabled: false` or run in `dry_run: true`. Live safety
guards, confidence thresholds, max bands, broad EQ rate limits and console EQ
bounds remain in force.
