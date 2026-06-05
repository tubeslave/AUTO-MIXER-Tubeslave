# Codex Proposal: Server Auto Compressor Reset Fail Closed - 2026-06-05

## Short thesis

Keep Auto Compressor startup/reset intent as reviewable evidence, but make the server reset helper impossible to use as a hidden direct compressor writer.

## What I understood

The repository already quarantines the public WING Auto Compressor entrypoint and the compressor controller apply methods. The remaining risk is a private server helper that still performs direct compressor reset writes if invoked from a non-quarantined path or future refactor.

## Proposed solution

- Add a small blocked-evidence builder for compressor reset intent.
- Change `_reset_compressor_on_channel` to return that evidence and log the block instead of calling `mixer_client`.
- Add a direct unit test against the helper.
- Update write-surface inventory, project-goal matrix, processing inventory, and ADR.

## Likely files to touch

- `backend/server.py`
- `backend/write_surface_inventory.py`
- `backend/project_goal_capability_matrix.py`
- `Docs/PROCESSING_METHODS_INVENTORY.md`
- `tests/test_server.py`
- `tests/test_processing_registry.py`
- `tests/test_project_goal_capability_matrix.py`
- `Docs/adr/server-auto-compressor-reset-fail-closed-20260605.md`

## Alternatives considered

- Rely on the existing public WING quarantine. Rejected because direct helper calls could still mutate a mixer.
- Delete the helper. Rejected because the would-be reset intent is useful evidence for future operator proposals.
- Promote compressor reset into supervised apply. Rejected because compressor readback, makeup/headroom proof, rollback and hardware validation are not complete.

## Risks

- A legacy non-WING caller that expected compressor reset mutation now receives blocked evidence. This matches the current "no live compressor writes" policy.

## Test plan

- Add focused `tests/test_server.py` coverage for `_reset_compressor_on_channel`.
- Run server, registry, matrix and static write-surface tests.
- Run `py_compile`, `git diff --check`, and the full backend suite.

## Where the other agent may disagree

Kimi may ask for a generic server write-gate abstraction. That is reasonable later, but this patch closes a concrete hidden compressor mutation with a smaller diff.
