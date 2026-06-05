# Brief: manual-write-cache-freshness-gate-20260605

## Context

The recommendation and operator-analysis producers now gate applyable fader/gain
proposals on fresh runtime cache. The central supervised manual write gate still
accepted cache-backed pre-read values without checking cache age. That leaves a
gap: a direct approved write or a queue producer outside the freshness-aware
bridges could use stale cache as the rollback baseline.

## Task

Add an apply-time cache freshness gate to supervised manual fader/gain writes.
When pre-read comes from physical readback, allow it. When pre-read falls back
to mixer state cache, require fresh `state_timestamps` for the exact address
before writing.

## Safety Boundaries

- No live OSC/WING, dLive, or Mixing Station writes during implementation.
- Do not widen live authority beyond supervised `fader` and `gain`.
- EQ, dynamics, routing, FX, scenes, and presets remain quarantined.
- Keep physical readback as the strongest pre-read evidence.

## Expected Evidence

- Direct approved manual fader/gain writes block stale cache pre-read before
  calling the mixer write method.
- Operator proposal apply blocks stale cache pre-read before invoking the manual
  write gate.
- Fresh cache and physical readback paths still work.
- Apply results and decision traces expose cache freshness evidence.
