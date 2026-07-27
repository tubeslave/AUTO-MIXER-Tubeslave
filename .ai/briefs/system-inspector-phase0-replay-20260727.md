# System Inspector Phase 0 and Snapshot Replay Brief

## Goal

Add an independent, evidence-backed `System Inspector` vertical slice that
operates on validated offline snapshots only. It must load a show profile,
macOS snapshot, WING snapshot and imported Dante snapshot, normalize them into
a topology, compare the observed route with the expected route, emit a
deterministic finding with evidence, and render CLI plus JSON output.

## Safety boundary

- No WING, OSC, CoreAudio, Dante Controller or network contact.
- No write-client imports from the Inspector package.
- `INSPECTOR_READ_ONLY` is fail-closed and cannot be disabled in v1.
- Any attempted Inspector write raises, appends a redacted security audit
  event, and performs no action.
- Snapshot/profile inputs are bounded and schema-validated.
- YAML uses `yaml.safe_load`.
- Missing evidence becomes `UNKNOWN`, never `OK`.
- Replay evidence grants no live authority and does not alter M001/CP05.

## Required first vertical slice

1. Load `config/show_profile.yaml` or an explicit fixture profile.
2. Load simulated `macos.json`, `wing.json`, `dante.json`.
3. Build stable-ID devices, endpoints, subscriptions, console inputs and
   observed routes.
4. Compare Playback L/R roles with the profile.
5. Detect `routing.playback-right-duplicate`.
6. Emit expected, observed, difference, evidence, confidence and impact.
7. Print a human-readable CLI result and write deterministic JSON.
8. Pass offline unit/integration/security tests.

## Manual checkpoints

Stop before real WING contact, active discovery, Dante Controller reads,
CoreAudio capture, macOS permission changes, licensed SDK use, helpers, or any
write path.

## Existing-state constraints

- Preserve unrelated changes in `frontend/package.json`,
  `reports/loop_soundcheck_pipeline.jsonl`, generated artifacts and the nested
  untracked directory.
- Reuse existing WING protocol knowledge but do not depend on the mixed
  read/write `WingClient` in Phase 0.
- Keep the existing live milestone and hardware evidence unchanged.

