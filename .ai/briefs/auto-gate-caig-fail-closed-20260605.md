# Auto Gate CAIG Fail-Closed Write Policy

## Task

Close the legacy CAIG direct gate write path so `AutoGateController` can keep producing gate decisions without mutating mixer state when a `mixer_client` is present.

## Context

- `backend/auto_gate_caig.py` is useful as gate analysis and recommendation logic.
- Gate, HPF, and send writes are not part of the current live-approved executor.
- The current live-approved surface remains the supervised Product Layer fader/gain apply path only.
- `backend/write_surface_inventory.py` already classifies gate/HPF/send direct apply as quarantined.

## Constraints

- Do not add gate writes to the live executor.
- Do not introduce console transport, queue mutation, or mixer state mutation.
- Preserve the existing `process()` return shape for callers that use CAIG as analysis.
- If a legacy caller passes a mixer client, the controller must fail closed.
- The worktree is dirty, so keep the diff scoped and do not overwrite unrelated work.

## Acceptance

- `AutoGateController.process()` returns decisions but does not call `mixer_client.set_channel_gate`.
- `_send_osc()` remains callable for compatibility but returns `0` and records a blocked write attempt.
- `get_status()` exposes the fail-closed policy and latest blocked-write telemetry.
- Registry and write-surface inventory describe CAIG direct gate sends as quarantined.
- Focused tests prove no fake mixer calls occur.
