# Codex Proposal: Auto EQ Apply Fail Closed - 2026-06-05

## Short thesis

Keep AutoEQ analysis and correction planning, but make direct EQ apply/reset paths fail closed so they emit blocked evidence instead of mutating the mixer.

## What I understood

The project needs automatic EQ as part of the AI Mixing Engineer. The current proof boundary does not include EQ writes. Direct controller methods in `auto_eq.py` can still call mixer EQ setters or raw reset OSC, even though public server handlers quarantine these surfaces.

## Proposed solution

- Add shared blocked-EQ result helpers in `auto_eq.py`.
- Add write telemetry to single-channel and multi-channel AutoEQ controllers.
- Convert single-channel apply/reset to blocked evidence.
- Convert multi-channel apply to blocked evidence while preserving conflict resolution.
- Update registry, write-surface inventory, capability matrix, processing inventory, focused tests and ADR.

## Likely files to touch

- `backend/auto_eq.py`
- `backend/processing_registry.py`
- `backend/write_surface_inventory.py`
- `backend/project_goal_capability_matrix.py`
- `Docs/PROCESSING_METHODS_INVENTORY.md`
- `tests/test_auto_eq_fail_closed.py`
- `tests/test_processing_registry.py`
- `tests/test_project_goal_capability_matrix.py`
- `Docs/adr/auto-eq-apply-fail-closed-20260605.md`

## Alternatives considered

- Leave controller writes because server handlers are quarantined. Rejected because direct controller calls can bypass Product Layer safety.
- Remove AutoEQ apply methods. Rejected because the future assisted/supervised path needs explicit would-be EQ commands.
- Promote EQ apply now. Rejected because filter audit, feedback safety, readback, rollback and hardware proof are incomplete.

## Risks

- Callers that expected boolean apply success now receive structured blocked evidence. This is intentional.
- Applied correction history must not be updated because the mixer was not mutated.

## Test plan

- Add single-channel apply/reset and multi-channel apply fail-closed tests.
- Add static test proving no direct mixer EQ writes remain in `auto_eq.py`.
- Run AutoEQ, registry, matrix, server quarantine and write-surface tests.
- Run the full backend suite.

## Where the other agent may disagree

Kimi may prefer introducing a reusable write gate first. That is a good follow-up, but this removes a real hidden EQ mutation path now and preserves evidence for future promotion.
