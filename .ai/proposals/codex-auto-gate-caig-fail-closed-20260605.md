# Codex Proposal: Auto Gate CAIG Fail-Closed Write Policy

## Short thesis

Keep CAIG useful as a gate analysis/recommendation source, but make every legacy direct gate send path fail closed until gate control has replay evidence, physical readback, rollback proof, and an approved supervised apply contract.

## What I understood

`AutoGateController` can compute gate decisions from audio features, group influence, adaptive thresholds, and drum rules. That analysis is valuable. The unsafe part is the historical `_send_osc()` branch, which can be reached when a caller passes `mixer_client`. Gate writes are not in the current live-approved executor; only supervised fader/gain apply is approved.

The worktree already contains many uncommitted changes. This proposal should touch only CAIG write policy, tests, and the council metadata that records the boundary.

## Proposed solution

- Preserve `AutoGateController.process()` and its returned `Dict[int, GateDecision]`.
- Reset write telemetry at the start of each `process()` call.
- Keep `_send_osc()` as a compatibility method, but make it a quarantine sink:
  - do not call `mixer_client.set_channel_gate`;
  - record `last_write_attempted`;
  - record `last_write_blocked`;
  - keep `last_write_count` at `0`;
  - return `0`.
- Expose `live_write_allowed=False`, `write_policy="proposal_only_quarantined"`, and latest write telemetry through `get_status()`.
- Update the processing registry, write-surface inventory, and capability matrix so CAIG remains proposal-only and direct gate sends remain quarantined.
- Add fake-mixer tests proving both `process()` and direct `_send_osc()` calls do not mutate mixer state.

## Likely files to touch

- `backend/auto_gate_caig.py`
- `backend/processing_registry.py`
- `backend/write_surface_inventory.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_gate.py`
- `tests/test_processing_registry.py`
- `.ai/briefs/auto-gate-caig-fail-closed-20260605.md`
- `Docs/adr/auto-gate-caig-fail-closed-20260605.md`

## Alternatives considered

- Remove `_send_osc()` entirely. Rejected because compatibility removal could break legacy analysis callers and obscure the safety boundary.
- Add a feature flag to allow gate writes. Rejected because there is no approved gate apply contract, no physical proof, and no rollback/readback safety path.
- Move CAIG into quarantine status. Rejected because the analyzer itself is useful; only the direct write behavior is quarantined.

## Risks

- Legacy callers may have assumed passing `mixer_client` applies gate changes. That behavior is now intentionally blocked.
- The method name `_send_osc()` is historical and misleading. The registry and tests now pin the safer interpretation.
- Future contributors could attempt to re-enable writes unless tests and inventory remain explicit.

## Test plan

- Compile touched Python modules.
- Run `tests/test_gate.py` and registry/inventory tests.
- Run the full backend suite with the project standard command.
- Run a fake-mixer smoke test proving no mixer calls occur when `process()` is invoked with a mixer client.

## Where the other agent may disagree

Kimi may argue for deleting or renaming `_send_osc()` to remove ambiguity. I would keep the method for compatibility in this cycle and rely on tests/registry to make the quarantine policy explicit. A later cleanup can rename it if all callers are inventoried.
