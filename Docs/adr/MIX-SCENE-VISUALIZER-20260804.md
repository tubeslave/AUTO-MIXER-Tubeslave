# ADR — Read-only Instrument Space vertical slice

Date: 2026-08-04
Status: accepted for read-only prototype; independent patch review pending

## 1. Context

Automixer needs an operator-readable view of each instrument in the mix. ART
DSP Canvas provides a useful public reference metaphor: pan on the horizontal
axis, frequency on the vertical axis and volume in depth. ART DSP also states
that this is not literal binaural 3D localization.

Automixer already publishes cached channel pan, input identity, meter level and
freshness through the Product Layer. This slice adds a bounded read-only
spectral cache to `AudioCapture`, so FFT work runs outside the realtime callback
and outside Product Layer snapshot construction. The cache constructor flag
defaults off; only the canonical server-owned Product Layer capture opts in.
The live-sound control plane is safety-sensitive and the dLive write proof
remains unresolved.

## 2. Options considered

1. Editable spatial mixer connected to console writes.
2. Frontend-only visualization derived from mock or role defaults.
3. Versioned backend read-only projection with an inspect-only UI.
4. Direct integration of the supplied Canvas plug-in.

## 3. Decision

Choose option 3.

`backend/instrument_space.py` publishes `automixer.instrument_space.v1` inside
the existing `channel_inventory`. Every axis is nullable and carries evidence.
The frontend validates that `read_only=true`, `live_write=false`,
`can_apply_to_console=false` and `physical_localization=false` before rendering.

Axis meanings:

- X: cached pan readback.
- Y: measured per-channel spectral centroid on a log scale.
- Z: measured channel level; louder objects appear closer/larger.

When audio is offline, below threshold, pending or stale, the centroid remains
null and the object is placed on a labelled unknown-Y rail. That rail is layout,
not an acoustic coordinate.

## 4. Why this won

It delivers a real end-to-end UI path using existing Product Layer evidence,
keeps missing data honest, and remains removable. Embedding the projection in
channel inventory gives channels and scene one generation timestamp without a
new WebSocket protocol.

## 5. Rejected alternatives

- Write-capable drag control: no verified scene-to-action/readback/rollback
  contract exists.
- Role-based spectral position: would present a heuristic as measurement.
- Canvas package integration: its DAW plug-in runtime is not an Automixer truth
  source, and copying internal behavior is outside scope.
- Literal 3D localization: the axes describe mix attributes, not acoustic source
  position around a listener.

## 6. Implementation plan

1. Completed: nullable DTO and read-only frontend scene.
2. Completed: bounded 0.5-second per-channel spectral cache outside the realtime
   callback and snapshot builder.
3. Add object selection, filtering and longer time/freshness inspection without
   apply.
4. Evaluate on replay fixtures and actual read-only capture.
5. Require a separate ADR before generating proposals or writes from scene
   gestures.

## 7. Test plan

- Builder normalization and unknown preservation.
- Product Layer integration regression.
- Forbidden dependency/import scan.
- Frontend adapter fail-closed contract.
- No drag/apply gesture contract.
- Frontend production build and standard repository suite before merge.

## 8. Risks and rollback

- Pan has legacy native/normalized representations. Preserve raw evidence and
  introduce an explicit protocol/unit field before broad mixer parity.
- Offline, silent, pending or stale spectral data falls back to a two-axis view
  plus an explicit unknown rail.
- Dense channel counts need grouping/zoom later.

Rollback is deletion of the `instrument_space` inventory field, its pure
builder, adapter and dashboard tab. No console state or authority is affected.

## 9. Implementation evidence

- Focused backend integration: 671 tests passed.
- Frontend read-only contracts: 91 tests passed; production Vite build passed.
- Spectral worker lifecycle: opt-in default is false; enabled worker becomes
  fresh and terminates with `AudioCapture.stop()`.
- Offline end-to-end replay produced fresh 1000 Hz and 6000 Hz centroids and
  distinct Y coordinates while `read_only=true` and `live_write=false`.
- Indicative local 64-channel/2048-point benchmark at a 500 ms interval:
  3.236 ms mean, 10.419 ms p95 per full refresh. This is diagnostic evidence,
  not a production latency guarantee.
- The repository-wide suite remains gated by the intentional CP05 clean-tree
  check in the already dirty worktree; no clean-source or hardware claim is
  made from the focused suite.
