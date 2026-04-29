# Task spectral-ceiling-eq: Spectral Ceiling EQ Heuristic

## Problem

Add an explainable, safe EQ recommendation layer inspired by public spectral
ceiling/noise-slope guide concepts. The new layer must shape instruments and
vocals toward broad target spectral forms instead of hard-coded magic
frequencies or aggressive match EQ.

## Constraints

- Preserve the current AutoFOH and offline mix-agent pipelines.
- Keep existing EQ rules, metrics, masking detectors, safety limits, and logs.
- Add the feature behind config with dry-run support.
- Do not copy proprietary Ayaic/COS Pro internals.
- No new production dependencies.
- Never bypass AutoFOH safety, phase guards, true-peak/headroom policy, or
  existing processing preservation.

## Definition of Done

- Spectral profiles exist for lead vocal, backing vocal, kick, snare, bass,
  guitars, keys, overheads, room, FX/mix bus style targets.
- The module produces structured proposals with reasons, skipped moves, safety
  metadata, and confidence.
- Live AutoFOH merges proposals into the existing EQ stage without replacing
  legacy rules.
- Offline mix-agent can emit and render bounded conservative actions.
- Debug CLI prints measured tilt, selected profile, moves, confidence and
  warnings.
- Tests cover target curve generation, clamps, profile selection, dry-run,
  front/back roles, vocal demasking, master bus limits, disable config, and
  existing pipeline regression.

## Test Command

`PYTHONPATH=backend python -m pytest tests/ -x --tb=short -q`
