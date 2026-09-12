# MuQ EWMA Drift

## 1. Context

The project already has a MuQ-Eval quality layer for master A/B validation.
The new need is per-stem temporal drift detection so noisy MuQ observations do
not cause twitchy live decisions, while sustained quality drops can freeze ML
corrections and restore the last known good parameters.

## 2. Options considered

- Add EWMA drift in the AutoFOH engine only.
- Add EWMA drift in the MuQ evaluation layer with an engine bridge.
- Add a separate OSC daemon for visualization.

## 3. Decision

Add `backend/ewma_metrics.py` and integrate it through
`MuQEvalService.update_stem_score_batch(...)`. AutoFOH consumes that result for
structured logs, OSC-style UI payloads, and correction freeze checks.

## 4. Why this won

The evaluation layer owns MuQ semantics and can be reused by offline, live, and
future batch scoring paths. The engine remains responsible for live actions and
does not bypass existing safety gates.

## 5. Rejected alternatives

- Engine-only drift tracking would make non-engine MuQ callers duplicate logic.
- A separate OSC daemon would add operational complexity and could be confused
  with console OSC control.

## 6. Implementation plan

- Implement `EwmaDrift` and a per-stem monitor with group defaults and optional
  frequency masks.
- Add disabled-by-default config in YAML/JSON.
- Add a soundcheck handler endpoint for batch score submission.
- Emit concise console logs and observation payloads for UI visualization.

## 7. Test plan

- Unit-test EWMA delay, false positives, hysteresis, and debounce.
- Unit-test MuQ service batch integration.
- Run focused tests plus the standard pytest command.

## 8. Risks and rollback

If the monitor causes unwanted freeze decisions, set
`muq_eval.stem_drift.enabled: false`. No mixer write path depends on this module
when disabled.
