# TUB-323 Deployment-Path Live-Write Boundary Map

Scope: `backend/server.py` and the WebSocket handlers that expose WING mutation paths.

Deployment rule:
- Approved live writes on WING are limited to the supervised manual path for channel `fader` and `gain`.
- Everything else on the deployment path must be either already transport-blocked or explicitly quarantined before it reaches the mixer.

## Classified WING mutation surfaces

### Gated and still allowed

- `set_fader` -> approved manual write path
- `set_gain` -> approved manual write path

### Already blocked before this change

- WING headless auto-engine `auto_apply` -> already forced off in `backend/handlers/soundcheck_handlers.py`

### Quarantined in this change

- Safe gain apply path: `start_realtime_correction`
- Soundcheck orchestration that would fan out into resets/applies: `start_auto_soundcheck`
- Snapshot mutation paths: `load_snap`, `save_snap`, `restore_snapshot`
- Broad reset helpers: `reset_trim`, `bypass_mixer`, `reset_eq`, `reset_all_eq`, `reset_phase_delay`
- Broad apply helpers: `apply_eq_correction`, `apply_channel_correction`, `apply_all_corrections`, `apply_phase_corrections`, `apply_auto_balance`
- Continuous or automatic automation writes: `start_realtime_fader`, `start_auto_compressor`, `start_auto_compressor_soundcheck`, `start_auto_compressor_live`
- Direct compressor editing helpers: `set_auto_compressor_profile`, `set_auto_compressor_manual`
- `start_auto_eq(auto_apply=true)` -> analysis still allowed, but `auto_apply` is forced off on WING while the deployment boundary is active

## Remaining out-of-scope risk

- Legacy scripts under `backend/` can still attempt direct write calls, but WING transport-level write blocking should stop unsupervised writes.
- This pass does not refactor those scripts onto the supervised path; it only tightens the live deployment server boundary.
