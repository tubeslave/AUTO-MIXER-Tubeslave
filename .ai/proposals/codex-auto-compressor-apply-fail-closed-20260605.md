# Codex Proposal: Auto Compressor Apply Fail Closed - 2026-06-05

## Short thesis

Keep automatic compressor analysis and parameter planning, but make direct compressor enable/apply paths fail closed so they emit blocked evidence instead of mutating a mixer.

## What I understood

The long-term AI Mixing Engineer needs automatic compression. The current repository safety boundary does not yet prove compressor writes. Direct calls in `auto_compressor.py` and `auto_compressor_cf.py` could bypass Product Layer review even though server deployment handlers quarantine compressor surfaces.

## Proposed solution

- Add write-policy telemetry to `AutoCompressorController`.
- Convert `_apply_compressor_smooth()` into a blocked evidence builder for `set_compressor_on` and `set_compressor`.
- Replace soundcheck compressor disable/restore and live-start enable writes with blocked evidence records.
- Convert `AutoCFCompressorController.apply_params()` into blocked evidence.
- Update registry, write-surface inventory, project-goal matrix, processing inventory, focused tests, and ADR.

## Likely files to touch

- `backend/auto_compressor.py`
- `backend/auto_compressor_cf.py`
- `backend/processing_registry.py`
- `backend/write_surface_inventory.py`
- `backend/project_goal_capability_matrix.py`
- `Docs/PROCESSING_METHODS_INVENTORY.md`
- `tests/test_auto_compressor_fail_closed.py`
- `tests/test_processing_registry.py`
- `tests/test_project_goal_capability_matrix.py`
- `Docs/adr/auto-compressor-apply-fail-closed-20260605.md`

## Alternatives considered

- Leave the direct writes because server deployment routes are quarantined. Rejected because direct Python use could still mutate a mixer.
- Remove compressor automation modules. Rejected because the project goal needs automatic compression evidence and planning.
- Promote compressor apply now. Rejected because dynamics readback, makeup/headroom proof, rollback and hardware validation are incomplete.

## Risks

- Callers that assumed compressor apply mutates the console now receive structured blocked evidence. This is intentional.
- `current_params` remains planned evidence state, not proof of console state. Promotion must add readback before treating it as applied state.

## Test plan

- Add direct tests for classic and CF compressor fail-closed behavior.
- Add static tests proving no direct compressor mixer calls remain in the two modules.
- Run compressor, registry, matrix, server quarantine and static write-surface tests.
- Run the full backend suite.

## Where the other agent may disagree

Kimi may prefer building a generic write-gate first. That is a good next architectural step, but this targeted patch closes real hidden compressor mutations immediately and preserves evidence for future promotion.
