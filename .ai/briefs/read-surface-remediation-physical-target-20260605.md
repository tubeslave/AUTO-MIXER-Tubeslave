# read-surface-remediation-physical-target-20260605

## Context

`read_surface_fixture_evidence` now identifies
`next_physical_readback_surface_id`. The remediation plan still builds actions
only from inventory statuses, so a cached surface can remain absent from the
action list even when it is the next physical readback gap.

## Goal

Connect fixture-evidence physical readback gaps to
`read_surface_remediation_plan` actions without querying or mutating a console.

## Non-goals

- Do not run physical readback.
- Do not query WING or dLive.
- Do not add live write authority or Full Autonomous promotion.

## Safety constraints

- Physical gap actions are planning evidence only.
- `raw_mixer_state_cache` may get a remediation action even though its inventory
  status is `cached`; the action means fixture proof is missing, not cache
  coverage is broken.
- Summary must expose the next physical readback target and matching action ID.

## Expected evidence

- Remediation plan includes fixture physical gap IDs.
- Remediation plan includes an action for the next physical readback target.
- Existing incomplete surface actions keep their current remediation contracts.
- Tests prove the module remains transport-free and non-mutating.
