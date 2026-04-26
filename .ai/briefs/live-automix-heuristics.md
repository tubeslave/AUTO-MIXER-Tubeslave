# Live automix heuristics

## Task

Embed the reviewed live-sound automixing heuristics into the project in a safe,
incremental way.

## Constraints

- Preserve feedback and true-peak safety.
- Do not raise faders above unity without explicit operator intent.
- Prefer deterministic control-plane logic before ML or high-latency DSP.
- Keep existing AutoFader V2 behavior available for rollback.
- The working tree is dirty; avoid unrelated files.

## Scope for first implementation

- Add a true Dugan-style NOM/gain-sharing controller with Last Hold, Gain
  Limiting, and Auto Mix Depth.
- Wire it into AutoFader V2 as an explicit controller mode.
- Tighten existing gate defaults/presets to match live-sound heuristics.
- Extend auto-pan with Perez/Reiss-style band accumulation and 22 ms smoothing.
- Add focused unit tests for the new deterministic DSP/control logic.

## Deferred scope

- Dynamic Soothe/BalancEQ-style EQ.
- CNN-based EQ parameter prediction.
- Spectral phase correction with all-pass/FIR filters.
- Full continuous phase/vector-scope UI.
