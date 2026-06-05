# Codex Proposal: Phase Alignment Apply Fail Closed - 2026-06-05

## Short thesis

Keep phase alignment analysis, but make delay/polarity apply and reset paths fail closed so they emit blocked evidence instead of mutating the console.

## What I understood

The target system needs phase correction as part of an autonomous AI mixing engineer. The current proven live boundary is narrower: supervised fader/gain only. Phase delay and polarity writes are high-risk because a bad change can damage mono compatibility and alter stage timing. The server already blocks deployment-level phase write messages, but the underlying controller and private direct helpers still contained mixer-client writes.

## Proposed solution

- Add explicit write-policy telemetry to `PhaseAlignmentController`.
- Convert `apply_corrections()` into an evidence builder:
  - keep measurement parsing, bleed checks, delay thresholds, and local `corrections`;
  - produce would-be `set_channel_delay`, delay-disable, and `set_channel_phase_invert` calls;
  - return `applied=False`, `blocked=True`, `mutates_mixer=False`.
- Convert `reset_all_phase_delay()` into blocked reset evidence without clearing local corrections as if reset succeeded.
- Make server private phase helpers delegate to the fail-closed controller.
- Update registry, write-surface inventory, capability matrix, processing inventory, tests, and ADR.

## Likely files to touch

- `backend/phase_alignment.py`
- `backend/server.py`
- `backend/processing_registry.py`
- `backend/write_surface_inventory.py`
- `backend/project_goal_capability_matrix.py`
- `Docs/PROCESSING_METHODS_INVENTORY.md`
- `tests/test_phase_alignment_fail_closed.py`
- `tests/test_processing_registry.py`
- `tests/test_project_goal_capability_matrix.py`
- `Docs/adr/phase-alignment-apply-fail-closed-20260605.md`

## Alternatives considered

- Leave the controller unchanged because the server handler is quarantined. Rejected because direct controller use could still mutate a mixer.
- Delete phase apply/reset support entirely. Rejected because the future autonomous system needs reviewable phase-correction evidence.
- Promote supervised phase apply now. Rejected because fresh readback, rollback, mono proof, and physical validation are not complete.

## Risks

- Existing direct callers that expected a boolean `True` now receive structured blocked evidence. This is intentional and safer than silently mutating delay/polarity.
- UI code that assumes success from direct phase apply must read the result object before future promotion.

## Test plan

- Add direct controller tests proving apply/reset do not call mixer methods.
- Add static tests proving `backend/phase_alignment.py` has no direct mixer delay/polarity calls.
- Run phase, registry, capability matrix, server quarantine, and static write-surface tests.
- Run the full backend test suite before considering the slice ready.

## Where the other agent may disagree

Kimi may argue for a generic write-gate abstraction instead of per-module fail-closed patches. That is valid longer-term, but this targeted change removes an existing hidden write path now while preserving a future proposal/apply contract.
