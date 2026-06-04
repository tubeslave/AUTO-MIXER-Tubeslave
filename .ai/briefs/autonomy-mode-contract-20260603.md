# autonomy-mode-contract-20260603

## Context

The project goal now names four target operating modes: Analysis Mode, Assisted
Mixing Mode, Supervised AutoMix Mode, and Full Autonomous Mixing Mode. The
backend already exposes stable UI keys (`analyze`, `assist`, `supervised`,
`auto`), but the product contract did not explicitly tie those keys to the
target architecture or distinguish future autonomous write surfaces from the
small current proven live write scope.

## Scope

- Keep the existing mode keys for UI compatibility.
- Expose the four-mode architecture in the backend mode status.
- Keep WING full autonomous apply locked.
- Keep current live writes limited to the proven fader/gain scope.
- Add focused tests for aliases, policy status, and fail-closed auto behavior.

## Non-goals

- No live OSC/WING writes.
- No new production dependencies.
- No promotion of EQ, compressor, routing, FX, scene, or preset live apply.
