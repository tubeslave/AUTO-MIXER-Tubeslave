# TUB-61 Operator Supervision Spec

## Status

Slice 1 in this spec is implemented.

Implemented files:
- `frontend/src/components/GainStagingTab.js`
- `frontend/src/components/GainStagingTab.css`
- `frontend/src/components/AutoGainSupervisionCard.js`
- `frontend/src/components/autoGainSupervision.js`
- `frontend/src/components/autoGainSupervision.test.js`
- `frontend/src/services/websocket.js`

Verification completed:
- `CI=true npm test -- --runInBand --watch=false src/components/autoGainSupervision.test.js`
- `npm run build`

Verification result:
- focused supervision tests passed
- frontend production build succeeded
- slice-local warnings were cleared from `GainStagingTab.js` and `websocket.js`
- remaining frontend warnings are outside this issue's slice

## Goal

Add the smallest safe operator-supervision surface for AutoGain in Control Center.

The surface must let the operator see:
- whether AutoGain is in `dry-run` or `live-ready`
- whether an apply is pending or blocked on confirmation
- the last blocked reason when apply did not happen

The UI must not imply that live apply is active before confirmation and safety gates pass.

## Chosen Backend Source Of Truth

Primary source:
- `live_input_trim_status`

Secondary source:
- `live_apply_result`
- `ready_for_live`
- `freeze_status`

Rationale:
- `live_input_trim_status` already exposes per-channel and global supervision state:
  - `live_apply_enabled`
  - `analysis_only_mode`
  - `meter_status`
  - `blocked`
  - `applied`
  - `waiting_channels`
  - `ready_channels`
  - channel-level `blocked_reason`
- `live_apply_result` already exposes:
  - `dry_run_only`
  - `blocked_reason`
  - `send_status`
  - `readback_status`
  - `message_for_user`
- current frontend does not consume these messages yet

## Narrow Surface

Implement one passive card inside `GainStagingTab` only.

Working name:
- `AutoGainSupervisionCard`

Placement:
- top of `GainStagingTab`, directly below `SignalHint`
- above start/stop buttons and existing settings/table

Why this slice:
- keeps scope inside the existing AutoGain surface
- avoids broad App shell changes
- exposes safety state where operators already look for gain behavior

## Exact UI States To Surface

### 1. Mode State

Render exactly one mode badge:
- `Dry Run`
- `Live Ready`
- `Live Applying`
- `Idle`

Derivation:
- `Idle`:
  - no active `live_input_trim_status`
- `Dry Run`:
  - `analysis_only_mode === true`
  - or `live_apply_enabled === false`
- `Live Ready`:
  - controller active
  - `analysis_only_mode === false`
  - `live_apply_enabled === true`
  - but no successful apply observed in current session
- `Live Applying`:
  - controller active
  - `live_apply_enabled === true`
  - and at least one applied adjustment has been observed

Safety rule:
- do not show `Live Applying` from operator intent alone
- only show it after actual applied activity is present

### 2. Pending Apply State

Render one concise status row:
- `Pending apply`
- `Waiting for signal`
- `Waiting for confirmation`
- `Blocked`
- `Applied`
- `Monitoring`

Derivation order:
- `Waiting for confirmation`:
  - latest `live_apply_result.blocked_reason === "confirm_live_apply_required"`
- `Blocked`:
  - latest blocked reason exists
  - or `live_input_trim_status.blocked` is non-empty
- `Applied`:
  - `live_input_trim_status.applied.length > 0`
- `Pending apply`:
  - one or more channels are ready
  - no apply yet
  - no blocking reason
- `Waiting for signal`:
  - `waiting_channels.length > 0`
  - and `ready_channels.length === 0`
- `Monitoring`:
  - fallback while active and not matching cases above

### 3. Confirmation Required State

Render an explicit warning banner when:
- latest `live_apply_result.blocked_reason === "confirm_live_apply_required"`

Copy intent:
- make it clear the system wants to apply
- make it equally clear nothing was sent

Required wording behavior:
- mention `confirmation required`
- mention `no console write sent`

### 4. Last Blocked Reason

Render a compact read-only field:
- label: `Last blocked reason`
- value: newest readable reason from either:
  - latest `live_apply_result.message_for_user`
  - first global blocked reason from `live_input_trim_status.blocked`
  - first channel `blocked_reason`

If none exists:
- show `None`

## Minimum Data Model In Frontend

Add local frontend state in `GainStagingTab` for:
- `liveInputTrimState`
- `latestLiveApplyResult`
- `readyForLive`
- `freezeStatus`

Subscribe to:
- `live_input_trim_status`
- `live_apply_result`
- `ready_for_live`
- `freeze_status`

Request on mount:
- `get_live_input_trim_status`
- `get_ready_for_live`
- `get_freeze_status`

Do not add controls in this slice for:
- toggling live apply
- acknowledging confirmations
- panic stop

This slice is status-only.

## Visual Rules

Badges:
- `Dry Run`: neutral blue
- `Live Ready`: amber
- `Live Applying`: green
- `Blocked`: red

Status hierarchy:
- top line: mode badge + concise current status
- second line: last blocked reason or last operator-safe message
- third line: small metadata chips
  - `ready N`
  - `waiting N`
  - `blocked N`
  - `meter trusted` or `meter untrusted`

Copy rules:
- never say `live` without also saying whether it is `ready` or `applying`
- never say `applied` if the latest result is dry-run

## First Implementation Slice

Implement only this:
1. Add websocket listeners and request methods usage in `GainStagingTab`.
2. Add `AutoGainSupervisionCard` in `GainStagingTab`.
3. Render the four required supervision outputs:
   - `dry-run/live-ready/live-applying/idle`
   - `pending apply`
   - `confirmation required`
   - `last blocked reason`

Do not do in slice 1:
- global topbar indicators
- cross-tab supervision
- per-channel detail tables
- correction history timeline
- new backend endpoints

## Follow-On Slice After This One

If slice 1 is accepted, slice 2 should add:
- per-channel blocked/apply rows from `live_input_trim_status.channels`
- last applied delta per channel
- correction history from backend decision logs

## Next Action Boundary

`TUB-61` does not need more scope to satisfy its minimum operator-supervision goal.

Reasonable next actions are:
- review and land this slice as-is
- or explicitly open a follow-on issue for slice 2 history/detail work

This issue should not silently expand into:
- app-shell indicators
- new live-apply controls
- per-channel tables
- broader dashboard redesign
