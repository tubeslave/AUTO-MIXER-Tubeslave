# Brief: product-layer-mode-gate-status-20260605

## Context

`autonomy_mode_transition_gate` is registered as a shadow-only evaluation
method and is now used by the supervised proof artifact report. The Product
Layer processing snapshot still does not expose the gate directly, so the
frontend Settings module table cannot show the four operating modes and Full
Auto blockers from the canonical backend gate.

## Goal

Expose `autonomy_mode_transition_gate` in the read-only processing snapshot and
map its rows into frontend Settings as status-only `mode_gate:*` rows.
Use those rows in the Settings Mode Policy summary so Full Auto blocked status
and the current Supervised policy boundary are visible without filtering the
module table.

## Constraints

- Do not add live mixer reads or writes.
- Do not place `autonomy_mode_transition_gate` into the main live executor.
- UI rows must use `canWriteLive=false` and `liveWriteKinds=[]`.
- Supervised current write kinds may be displayed as policy evidence, not as a
  new UI apply surface.

## Likely files

- `backend/processing_registry.py`
- `frontend/src/services/operatorAdapters.ts`
- `frontend/src/views/SettingsView.tsx`
- `tests/test_processing_registry.py`
- `frontend/src/services/__tests__/operatorAdaptersContract.test.mjs`
- `Docs/adr/product-layer-mode-gate-status-20260605.md`

## Validation

- `PYTHONPATH=backend python3 -m pytest tests/test_processing_registry.py tests/test_autonomy_mode_transition_gate.py -q`
- `npm --prefix frontend test`
- `cd frontend && npx tsc --noEmit`
- `python3 -m py_compile backend/processing_registry.py backend/autonomy_mode_transition_gate.py`
