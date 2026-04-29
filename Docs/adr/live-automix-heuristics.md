# Live automix heuristics

## 1. Context

The project review identified several recommended live-sound automixing
heuristics that are missing or only partially implemented: true Dugan/NOM
automix, music-oriented Last Hold/Gain Limiting/Auto Mix Depth, production gate
presets, Perez/Reiss-style spectral panning, dynamic EQ, and spectral phase
correction.

## 2. Options considered

- Implement every reviewed heuristic in one large pipeline refactor.
- Start with deterministic control-plane heuristics and keep existing modules
  available for rollback.
- Leave the implementation unchanged and document the gaps only.

## 3. Decision

Implement the deterministic, low-latency control-plane heuristics first:
Dugan/NOM automix, Last Hold, Gain Limiting, Auto Mix Depth, live gate presets,
and spectral-band panning. Defer dynamic EQ and spectral phase correction to
separate ADRs because they need deeper DSP validation and UI observability.

## 4. Why this won

Dugan-style gain sharing is mature, mathematically simple, and directly reduces
NOM/feedback risk. The implementation can be isolated behind a new AutoFader V2
mode, preserving the current LUFS balancing behavior.

## 5. Rejected alternatives

The one-shot pipeline rewrite was rejected because it would mix control logic,
ML, EQ, phase, UI, and mixer commands into a high-risk diff. Replacing the
current gain-sharing controller in place was rejected because the current
cross-adaptive LUFS balancing is a different useful mode.

## 6. Implementation plan

- Add a Dugan automix controller with NOM, Last Hold, Gain Limiting, and Auto
  Mix Depth.
- Wire `controller_mode: "dugan"` into AutoFader V2.
- Add safer gate preset helpers for drums and non-drum instruments.
- Extend AutoPanner with a band accumulator and smooth pan transitions.
- Keep dynamic EQ and spectral phase as documented future work.

## 7. Test plan

- Focused unit tests for Dugan/NOM math, Last Hold, depth clamp, and gain
  limiting.
- Focused unit tests for gate defaults/presets.
- Focused unit tests for panner band grouping and low-frequency centering.
- Run the standard pytest command before merge when the broader dirty tree is
  ready.

## 8. Risks and rollback

Risk: Dugan attenuation can be too strong for music. Rollback: set
`controller_mode` back to `gain_sharing` or reduce `auto_mix_depth_db`.

Risk: gate presets may close sustained sources too quickly. Rollback: override
per-channel settings or disable adaptive gate.

Risk: panning changes can surprise operators. Rollback: keep
`auto_panner_enabled` false or disable spectral band mode.
