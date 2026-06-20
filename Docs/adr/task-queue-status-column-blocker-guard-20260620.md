# ADR: Task Queue Status Column Blocker Guard

## Context

M001 is blocked on external WING/operator validation. The project can continue
with offline, fail-closed presentation hardening that prevents operator surfaces
from implying promotion readiness while the integration bundle blocks planning.

Task Queue already blocks apply actions and adds row badges while the integration
bundle is blocked. The Status column still renders the raw task status, which
can make a row look applied or currently actionable even though apply is blocked.

## Options Considered

1. Leave the Task Queue Status column unchanged.
2. Hide the Status column while the integration bundle is blocked.
3. Keep the Status column visible and override only its local display status.

## Decision

Use option 3. `TaskQueuePanels` derives local Task Queue status display values
through `taskStatusDisplay(task, integrationBundleApplyBlocked)`. While blocked,
the Status column shows `Blocked` with a red tone. Otherwise it preserves the
raw task status and normal status tone.

This does not mutate task status input, task data, proposal data, backend status
input, backend result input, safety monitor input or readiness input.

## Why This Won

The operator keeps useful row status context while the UI makes the global
promotion blocker explicit at the status point. It is local presentation logic
only and does not alter task records, backend contracts or apply paths.

## Rejected Alternatives

Leaving the column unchanged keeps a misleading current-status gap. Hiding the
column removes useful queue context and is not required for safety because the
badge can fail closed locally.

## Implementation Plan

- Add local task status display derivation in `TaskQueuePanels`.
- Render the Status column from the local display values.
- Add frontend and WING rails.
- Record the change in roadmap, capability matrix, completion gate, current
  state, heartbeat, report and ledger.

## Test Plan

- Frontend contract test verifies the local status-column derivation.
- WING MVP rail verifies docs, state, reports and live-authority denial remain
  aligned.
- Full frontend build and full backend test suite remain green.

## Risks and Rollback

Risk is limited to Task Queue presentation. Rollback removes the local
status-column derivation, the M064 rails and the M064 docs/state entries. M001
remains blocked on real WING read-only evidence either way.
