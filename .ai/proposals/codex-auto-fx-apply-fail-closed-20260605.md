# Codex Proposal: AutoFX Apply Fail-Closed Boundary

## Short thesis

Keep `AutoFXPlanner` as a useful FX planning layer, but make its legacy apply method fail closed so FX/send commands become blocked evidence instead of mixer mutations.

## What I understood

The project goal includes automatic FX management and send tuning. The repo already has `fx_send_proposal.py` for transport-free send proposals, while `auto_fx.py` still contains a legacy `apply_to_mixer()` path that directly calls mixer-client FX/send methods. Current safety policy does not allow FX, insert, or send writes in the live executor.

## Proposed solution

- Add `live_write_allowed=False` and `write_policy="proposal_only_quarantined"` to `AutoFXPlanner`.
- Change `apply_to_mixer()` to return a fail-closed result:
  - no mixer-client methods are called;
  - `applied=False`;
  - `calls=[]`;
  - intended commands are exposed as `blocked_calls`;
  - `mutates_mixer=False`.
- Move command construction into a helper that builds reviewable blocked evidence.
- Update registry, write-surface inventory, capability matrix, and processing inventory so the boundary is explicit.
- Update tests to prove fake mixer no-mutation and blocked evidence.

## Likely files to touch

- `backend/auto_fx.py`
- `tests/test_auto_fx.py`
- `backend/processing_registry.py`
- `backend/write_surface_inventory.py`
- `backend/project_goal_capability_matrix.py`
- `tests/test_processing_registry.py`
- `tests/test_project_goal_capability_matrix.py`
- `Docs/PROCESSING_METHODS_INVENTORY.md`
- `Docs/adr/auto-fx-apply-fail-closed-20260605.md`

## Alternatives considered

- Delete `apply_to_mixer()`. Rejected because a compatibility method returning blocked evidence is safer for unknown legacy callers.
- Keep direct apply for lab mode. Rejected because lab/offline code must not become an accidental live write path.
- Promote AutoFX into main assisted pipeline. Rejected because `fx_send_proposal.py` already owns the transport-free assisted contract, and broad FX rack writes need stronger proof.

## Risks

- Legacy tests or scripts that expected fake mixer calls must now assert blocked evidence.
- The method name remains misleading until all callers are inventoried.
- Blocked evidence still contains write command names, so inventory/static-scan coverage must remain explicit.

## Test plan

- Compile touched Python modules.
- Run `tests/test_auto_fx.py`, registry tests, project goal matrix tests, and static write-surface scan.
- Run the full backend suite.
- Run a runtime fake-mixer smoke test proving `apply_to_mixer()` returns blocked calls but makes zero mixer calls.

## Where the other agent may disagree

Kimi may prefer deleting or renaming `apply_to_mixer()`. I would keep it as fail-closed compatibility in this slice because that makes legacy calls safe immediately and leaves a clean later cleanup path.
