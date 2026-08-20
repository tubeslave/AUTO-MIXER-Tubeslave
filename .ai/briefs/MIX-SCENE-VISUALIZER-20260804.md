# Brief — MIX-SCENE-VISUALIZER-20260804

## Goal

Add the smallest real read-only visualization of each input as an object in a
mix space, informed by ART DSP Canvas but implemented from Automixer-owned
runtime evidence.

## Confirmed reference semantics

Canvas publicly describes a visual mix where left/right represents pan,
up/down represents frequency and front/back represents volume. Its support page
explicitly says this is not a 3D binaural panner. Automixer adopts only that
public interaction metaphor and does not copy proprietary implementation.

## Boundaries

- No installation or launch of the supplied package.
- No reverse engineering or reuse of proprietary assets/code.
- No mixer transport, proposal queue, FSM mutation or actuation dependency.
- No drag-to-apply gesture.
- Unknown measurements remain unknown; instrument names do not generate fake
  spectral coordinates.
- This work does not change dLive/WING authority or hardware readiness.

## Acceptance

1. Backend publishes a versioned read-only scene inside channel inventory.
2. Frontend renders measured pan, level and spectral centroid when present.
3. Missing spectral evidence is shown explicitly outside the semantic Y axis.
4. Unsafe/write-capable scene contracts are rejected.
5. Unit, contract and frontend build checks pass.

Owner: Codex implementation workstation. Independent patch review: pending.
