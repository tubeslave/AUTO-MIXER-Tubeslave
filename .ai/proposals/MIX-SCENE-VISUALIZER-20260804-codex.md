# Proposal — read-only Instrument Space

## 1. Short thesis

Ship a transport-free Product Layer projection and SVG inspector before any
editable spatial mixer or scene-to-proposal integration.

## 2. What I understood

The desired product should let an operator see every instrument as a distinct
object in the mix. The useful Canvas reference is a perceptual control map, not
literal physical localization: pan, spectral position and level are the axes.

## 3. Proposed solution

- Build `automixer.instrument_space.v1` from the existing channel inventory.
- Keep X/Y/Z nullable and retain measurement provenance alongside coordinates.
- Put objects without spectral evidence on a visibly separate unknown rail.
- Reject any frontend scene DTO that advertises write authority.
- Keep all interactions inspect-only in this slice.

## 4. Likely files to touch

- `backend/instrument_space.py`
- `backend/operator_product_state.py`
- `frontend/src/services/operatorAdapters.ts`
- `frontend/src/components/dashboard/AutomixerDashboard.tsx`
- shared frontend types and focused tests

## 5. Alternatives considered

- Role-derived Y coordinates: rejected because they would look measured.
- Direct Canvas plug-in integration: rejected because it is DAW plug-in state,
  not Automixer Product Layer evidence.
- Editable drag surface: deferred until a separate proposal/safety ADR.
- New WebSocket event: deferred; embedding in the existing channel inventory is
  smaller and keeps one generation timestamp.

## 6. Risks

- Current inventory does not yet publish per-channel spectral centroid.
- Dense unknown-Y rows may overlap; cap the initial visible set and retain a
  textual evidence list.
- Pan readback has legacy normalized/native conventions that require a future
  explicit unit contract.

## 7. Test plan

- Deterministic builder and unknown-axis tests.
- Static dependency denial for transport/actuation modules.
- Adapter rejection of write-capable DTOs.
- UI source contract denying drag/apply gestures.
- Existing Product Layer tests and production frontend build.

## 8. Where another reviewer may disagree

A reviewer may prefer a dedicated WebSocket snapshot or require the pan unit
contract to be normalized before exposing X. Those are valid follow-ups; neither
requires broadening authority in this slice.
