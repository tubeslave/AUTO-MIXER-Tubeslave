# TUB-364 Analyzer-to-Control Runtime Consolidation

Date: 2026-05-17

Scope:
- gain/fader analyzer-to-control runtime on the backend deployment surface
- duplicate controller generations still present in the repo
- integration boundaries for audio capture, bleed compensation, replay intents, and supervised writes

## Decision

The active WING deployment runtime for gain/fader automation is:

1. `backend/auto_soundcheck_engine.py`
   - owns analyzer-side signal capture, channel metrics, and recommendation production
   - on WING it stays dry-run for gain/fader proposals and emits replay/write-intent artifacts first
2. `backend/replay_write_intent_adapter.py`
   - turns analyzer output into replay-safe proposal artifacts
   - blocks automatic promotion to live writes on WING
3. `backend/server.py` supervised manual write path
   - `AutoMixerServer._apply_manual_console_write(...)`
   - this is the only approved live gain/fader mutation path on the WING deployment surface

That gives one deployment-approved path:

`AudioCapture -> AutoSoundcheckEngine -> ReplayWriteIntentAdapter -> supervised manual fader/gain write gate`

## Runtime Classification

| Runtime | Current role | Deployment status | Notes |
|---|---|---|---|
| `AutoSoundcheckEngine` | analyzer + recommendation runtime | Active on WING deployment | `soundcheck_handlers.py` forces `auto_apply=False` on WING and gain/fader writes emit replay intents first |
| Supervised manual write gate | live write application path | Active on WING deployment | Only approved live `fader`/`gain` path; readback + rollback aware |
| `LUFSGainStagingController` | legacy realtime gain control | Quarantined on WING deployment | Still exposed through `start_realtime_correction`, but blocked by deployment boundary on WING |
| `SafeGainCalibrator` | one-shot legacy gain helper | Quarantined on WING deployment | Uses legacy gain staging buffers and direct apply behavior outside deployment mode |
| `AutoFaderController` | legacy realtime fader control | Quarantined on WING deployment | Still exposed through `start_realtime_fader`, but blocked by deployment boundary on WING |
| `AutoFaderControllerV2` | experimental analyzer/fader runtime | Experimental and unwired | Test-covered, but not instantiated by `backend/server.py` |

## Integration Boundaries

### Audio input and analyzer ownership

- `AudioCapture` remains the shared capture primitive for deployment-safe analyzer work.
- `AutoSoundcheckEngine` owns the deployment analyzer loop for gain/fader recommendations.
- `BleedService` remains the shared bleed-analysis dependency when legacy gain/fader paths are used in lab mode.

### Control and safety boundary

- `ReplayWriteIntentAdapter` is the mandatory proposal boundary for WING gain/fader automation.
- `WingClient` transport blocking and `AutoMixerServer` supervision still fail closed if a legacy path attempts a live WING write.
- Automatic promotion from analyzer output to live WING control is still disallowed.

### Readback and rollback boundary

- Readback-confirmed live changes remain scoped to supervised manual writes.
- Replay/shadow evidence is the deployment-safe proof surface for analyzer recommendations.

## Module Ownership

- Analyzer runtime owner: `AutoSoundcheckEngine`
- Recommendation transport owner: `ReplayWriteIntentAdapter`
- Live write safety owner: `AutoMixerServer` supervised manual gate plus `WingClient` write gate
- Legacy lab-only gain owner: `LUFSGainStagingController` and `SafeGainCalibrator`
- Legacy lab-only fader owner: `AutoFaderController`
- Experimental branch owner: `AutoFaderControllerV2`

## Migration Order

1. Keep WING deployment pinned to `AutoSoundcheckEngine -> ReplayWriteIntentAdapter -> supervised manual writes`.
2. Treat `LUFSGainStagingController`, `SafeGainCalibrator`, and `AutoFaderController` as lab-only until they either emit the same intent artifacts everywhere or are removed from the WING surface entirely.
3. Keep `AutoFaderControllerV2` unwired until it can replace both legacy gain/fader controllers behind the same replay-intent and supervision contract.
4. Only after step 3, decide whether the repo still needs separate legacy realtime gain and fader controllers.

## Safe Next Tasks

1. Add one status/report endpoint consumer that displays the selected gain/fader runtime to the operator UI.
2. Remove dead UI affordances for WING paths that are permanently quarantined.
3. Port any remaining non-fader `AutoSoundcheckEngine` writes onto replay-intent artifacts before reviewing them for supervised promotion.
4. Decide whether `SafeGainCalibrator` should be folded into `AutoSoundcheckEngine` analysis or explicitly marked lab-only in the UI.
