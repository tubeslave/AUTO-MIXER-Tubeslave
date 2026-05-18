# TUB-336: Shadow-mode analysis for WING Rack telemetry and replay/live mismatches

Date: 2026-05-14

## Scope

This report is a repo-level analysis of the current WING Rack observation path,
telemetry probe path, and likely replay/live mismatch sources.

Important boundary:
- No real WING telemetry dump, JSON capture, or replay corpus was found in this
  repository during this pass.
- Conclusions below are therefore limited to code-path analysis and safety
  readiness, not to verified live-console telemetry behavior.

Related existing note:
- `Docs/reports/tub/TUB-335-WING-Telemetry-Notes.md`

## Current shadow-mode path

### 1. WebSocket entry point

The server already accepts an explicit observation flag on WING connect:
- `backend/handlers/connection_handlers.py`
- `backend/server.py`

Current flow:
1. Frontend or test client sends `connect_wing`.
2. Handler forwards `observation_only`.
3. `AutoMixerServer.connect_wing()` creates `EnhancedOSCClient`.
4. The server applies `set_observation_mode(...)` before use.

Relevant code:
- `backend/server.py:481-487`

### 2. Transport-level write block

`WingClient` already has a transport-level observation gate:
- Environment-driven default: `AUTOMIXER_WING_OBSERVATION_MODE`
- Runtime toggle: `set_observation_mode(enabled, reason=...)`
- Write commands are blocked in `send()` when values are present
- Queries without values are still allowed

Relevant code:
- `backend/wing_client.py:70-76`
- `backend/wing_client.py:129-149`
- `backend/wing_client.py:313-335`

### 3. Read-only telemetry probe

There is already a repo path for safe telemetry probing:
- WebSocket message: `probe_wing_level_telemetry`
- Probe helper: `WingClient.probe_channel_levels(...)`
- Candidate paths include meter-like addresses and fallback control addresses

Relevant code:
- `backend/wing_client.py:84-127`
- `backend/server.py:1470-1556`

### 4. Existing tests around observation mode

Observation behavior already has targeted tests:
- blocked writes in observation mode
- structured blocked-write event emission
- env-driven observation mode
- server-level application of observation mode on connect

Relevant tests:
- `tests/test_wing_client.py:104-190`
- `tests/test_server.py:180-240`

## Safety mechanisms already present

### Present and useful

- Transport-level write blocking in observation mode
- Per-address OSC rate throttle, default `10 Hz`
- Blocked write counter via `get_stats()`
- Read-only candidate telemetry probe path
- Structured logging hooks are available at the application level

Relevant code:
- `backend/wing_client.py:65-68`
- `backend/wing_client.py:301-311`
- `backend/wing_client.py:386-392`
- `backend/logging_config.py:10-70`

### Present but incomplete for live telemetry forensics

- Blocked writes are logged, but there is no guaranteed persistent JSON log sink
  for WING telemetry sessions by default.
- Telemetry probe returns a summary payload, but does not persist a raw capture
  timeline for later replay diffing.

## Main replay/live mismatch risks

### 1. Observation mode is not global across the repo

The server path is guarded, but many other repo paths create `WingClient`
directly. Those flows can bypass the server-managed observation policy.

Most important example:
- `backend/auto_soundcheck_engine.py:511-521`

There are also many direct utility scripts creating `WingClient(...)` for reset,
routing, snapshot, and ad-hoc tests.

Why this matters:
- A session may be called "shadow-mode" at the server layer while another code
  path still has the ability to write to the console.
- Replay/live mismatch investigation becomes untrustworthy if the same run can
  mix observation and mutation paths.

### 2. No durable raw telemetry capture for diffing

`probe_channel_levels()` sends candidate read queries, waits, and then returns
the subset of `self.state` that became populated.

What is missing:
- timestamps per observation
- source tagging for each value
- session/run id on the capture output
- raw packet transcript
- persisted JSONL or similar replay artifact

Why this matters:
- A live mismatch cannot be reproduced from a final snapshot alone.
- Replay analysis needs a timeline, not only final observed values.

### 3. Fallback addresses can produce false confidence

The probe can return fallback control values such as:
- `/ch/{channel}/fdr`
- `/ch/{channel}/mute`
- `/ch/{channel}/name`

This confirms callback traffic, but not actual level telemetry.

The server does flag fallback-only responses via:
- `no_non_meter_control_feedback`

Relevant code:
- `backend/server.py:1521-1537`

Risk:
- `num_observed > 0` can still mean "no usable meters".
- A replay pipeline built on fallback-only responses will not validate real
  live-level telemetry behavior.

### 4. Local state cache conflates sent values and received values

When `WingClient.send()` is called with values, it immediately updates the local
state cache before any separate console echo or confirmation path is proven.

Relevant code:
- `backend/wing_client.py:367-372`

Why this matters:
- In a real write-enabled run, local state may look correct even if the console
  did not apply or echo the value yet.
- Replay/live diffing should compare "command attempted" versus "console
  observed", but the current state cache can blur those two concepts.

### 5. Keepalive timing differs between docs and code

Repo docs repeatedly describe a `5 sec` WING keepalive expectation, but the
current renewal loop sleeps `8` seconds.

Relevant code:
- `backend/wing_client.py:408-415`

Why this matters:
- If live subscriptions are marginal, dropped or delayed callback traffic can
  look like telemetry mismatch when the real cause is subscription stability.

This is not yet proven as a bug here, but it is a live-observability risk.

### 6. Runtime config source is currently unstable

`config/automixer.yaml` contains unresolved merge markers.

Relevant file:
- `config/automixer.yaml:18-32`

Why this matters:
- Any runtime path reading this file may get invalid or unintended config.
- Shadow-mode experiments should not rely on a config file that is currently in
  a conflicted state.

### 7. Existing state-diff infrastructure is not wired into telemetry replay

There is state synchronization infrastructure in `MixerStateManager`:
- `update_from_osc_state(...)`

Relevant code:
- `backend/mixer_state.py:88-150`

But there is no integrated pipeline yet that:
1. ingests live telemetry into a durable state timeline
2. replays the same timeline offline
3. emits normalized diffs by address/channel/value/timestamp

## What can be concluded safely today

- The repo already has a valid first-layer shadow-mode gate for the main
  WebSocket WING path.
- The repo already has a safe first telemetry probe path for firmware discovery.
- The repo does not yet have enough durable telemetry capture and replay tooling
  to claim a completed live-vs-replay comparison.
- The biggest current risk is not DSP logic; it is control-path inconsistency:
  some execution paths are observation-safe, others are direct-client scripts or
  engines that can bypass the same policy.

## Safe next actions

### Highest priority

1. Capture one real observation-only session from the actual WING Rack and save
   the full raw payload stream to a durable artifact.
2. Separate `command attempted`, `blocked write`, and `console observed`
   channels in the logging model.
3. Build a small JSONL replay artifact format for telemetry sessions.
4. Compare replay against live using address-level diffs, not only final state.

### Safety-first capture procedure

Recommended minimal live capture path:
1. Start backend with `AUTOMIXER_WING_OBSERVATION_MODE=true`
2. Connect through `connect_wing(... observation_only=true)`
3. Run `probe_wing_level_telemetry`
4. Record returned payload plus raw observed callback stream
5. Do not use standalone scripts that create `WingClient` directly during the
   same capture unless they also explicitly run in observation mode

### Follow-up implementation tasks

1. Add persistent telemetry session recording for WING observation runs.
2. Add source tags to state entries: `query_response`, `subscription_update`,
   `local_write_attempt`, `blocked_write`.
3. Route `AutoSoundcheckEngine` through the same observation policy surface as
   `AutoMixerServer`.
4. Add a replay comparator built on `MixerStateManager`.
5. Resolve `config/automixer.yaml` merge conflict before relying on it for live
   capture settings.

## Blocker for closing TUB-336

Actual replay/live mismatch analysis remains blocked until at least one real
WING Rack observation capture is attached or produced from this workspace.
