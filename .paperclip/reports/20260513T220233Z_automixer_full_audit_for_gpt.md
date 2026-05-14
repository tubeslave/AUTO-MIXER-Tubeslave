# Полный аудит AUTO-MIXER для GPT

Дата проверки: 2026-05-14 MSK / 2026-05-13T22:02:33Z  
Проект: `/Users/dmitrijvolkov/AUTO-MIXER-Tubeslave-main`  
Ветка: `feature/mcp-director`  
Режим: анализ без исправления production-кода

## Короткий вердикт

Проект сейчас нельзя считать чисто готовым live-runtime для реального пульта. В репозитории есть рабочие и хорошо защищённые части, но рядом с ними остаются красные тесты, прямые обходы live-write gate, runtime mismatch в Multi-Channel AutoEQ, UI-команды без backend handlers, tracked секрет в `.env`, а также несколько параллельных/legacy архитектур.

Практически: offline `automixer.production_mix_v1` и часть dry-run/report-only поверхностей выглядят пригодными для дальнейшей работы; live-консоль безопасно считать подтверждённой только в узком смысле: guarded/manual `set_fader` / `set_gain` для Wing через `LiveApplyService` при explicit confirm и readback. Всё остальное требует отдельного review перед реальным применением.

## Проверки, которые были выполнены

- `git status --short --branch`: рабочее дерево грязное, 54 modified файла и много untracked `.paperclip`, `Docs`, backend/frontend/tests артефактов.
- `python3 -m compileall -q backend automixer ai_mixing_pipeline tools scripts tests`: OK.
- `pytest -q`: 797 passed, 6 skipped, 6 failed.
- Targeted backend pytest set: 138 passed, 3 failed.
- `cd frontend && npm test -- --watchAll=false --runInBand`: OK, 5 tests passed.
- `cd frontend && npm run build`: OK, production build compiled.
- `node --check tools/mcp_remote_gateway/server.js`: OK.
- `node --check tools/mcp_director/server.js`: OK.
- `python3 -m automixer.production_mix_v1 --help`: OK from repo root.
- `python3 -c "import backend.server"` from repo root: FAIL, absolute imports expect backend cwd/PYTHONPATH.
- `cd backend && python3 -c "import server"`: OK.

## Critical Findings

### 1. Tracked `.env` contains a real-looking OpenAI API key

Evidence:
- `.env:1` contains `OPENAI_API_KEY=...`.
- `git ls-files .env` returns `.env`, so the file is tracked.
- `.gitignore` does not ignore repo-root `.env`; it only ignores frontend env variants and `config/user_config.json`.

Impact:
- This is the highest priority issue. The key should be treated as compromised if this repo was ever pushed, copied, archived, or shared.

Recommended fix:
- Rotate/revoke the key immediately.
- Remove `.env` from git tracking.
- Add root `.env` and similar local secret files to `.gitignore`.
- Add a sanitized `.env.example`.
- If the secret was committed, purge it from git history before sharing the repository.

### 2. Full pytest is red

Command:

```text
pytest -q
```

Result:

```text
797 passed, 6 skipped, 6 failed
```

Failures:
- `tests/test_direct_write_inventory.py::test_arrangement_automation_inventory_live_enabled_enforces_fader_ceiling`
- `tests/test_direct_write_inventory.py::test_arrangement_automation_replay_safe_offset_plan_applies_through_live_apply_gate`
- `tests/test_direct_write_inventory.py::test_auto_fader_inventory_apply_auto_balance_enforces_min_max_fader_clip`
- `tests/test_replay_intelligence_benchmark_suite.py::test_replay_intelligence_benchmark_matches_canonical_metrics`
- `tests/test_room_compensation_replay.py::test_room_compensation_catalog_replay_paths[case0]`
- `tests/test_room_compensation_replay.py::test_room_compensation_catalog_replay_paths[case1]`

Impact:
- Safety/replay inventory and room compensation baseline are not stable. This blocks any honest “all green” readiness claim.

## High-Severity Runtime Bugs

### 3. Multi-Channel AutoEQ has a live runtime TypeError

Evidence:
- `backend/server.py:2731-2738` calls:

```python
self.multi_channel_auto_eq_controller.start_multi_channel(
    device_id=int(device_id),
    channels_config=channels_config,
    mode=mode,
    ...
)
```

- Actual signature in `backend/auto_eq.py:2772-2777` is:

```python
def start_multi_channel(self, device_id, channels_config, on_spectrum_callback=None, on_corrections_callback=None, on_status_callback=None)
```

There is no `mode` kwarg.

Impact:
- Frontend `startMultiChannelAutoEQ(..., mode)` can reach server and fail at runtime with `unexpected keyword argument 'mode'`.

Recommended fix:
- Either add `mode` to `MultiChannelAutoEQController.start_multi_channel()` and implement semantics, or stop passing it from `server.py`.
- Add a direct test around `AutoMixerServer.start_multi_channel_auto_eq()`.

### 4. Live-write safety gate is not consistently applied

The repo has a good central gate in `backend/live_apply.py`, but several active paths still write directly to the mixer.

Evidence:
- AutoEQ direct apply/reset:
  - `backend/server.py:2409-2425`
  - `backend/server.py:2468-2495`
  - `backend/auto_eq.py:2484-2516`
  - `backend/auto_eq.py:2550-2570`
- Phase correction direct apply:
  - `backend/server.py:3094-3167`
  - `backend/server.py:3185-3244`
  - `backend/phase_alignment.py:821-842`
- AutoSoundcheck direct writes:
  - `backend/auto_soundcheck_engine.py:1006`
  - `backend/auto_soundcheck_engine.py:1093`
  - `backend/auto_soundcheck_engine.py:1236`
  - `backend/auto_soundcheck_engine.py:1297`
  - `backend/auto_soundcheck_engine.py:1518`
  - `backend/auto_soundcheck_engine.py:1608`
  - `backend/auto_soundcheck_engine.py:1706-1716`
- System measurement apply:
  - `backend/system_measurement.py:1044-1072`
- Operator reset/restore scripts have broad direct writes:
  - `backend/reset_all_channels.py:23-165`
  - `backend/restore_channels.py:23-187`

Impact:
- `LiveApplyService` cannot be treated as system-wide protection.
- `panic_stop` is not system-wide.
- Some UI/API actions can mutate console state without the same confirmation/readback/audit semantics as manual fader/gain.

Recommended fix:
- Inventory every real mutation path.
- Route all non-emergency writes through a unified gate or explicitly mark them as blocked/dry-run.
- Keep feedback emergency policy separate from broad AutoSoundcheck apply.

### 5. `LiveApplyService` is Wing-shaped and breaks/contradicts other mixer clients

Evidence:
- `backend/live_apply.py:60-64` maps operations to `set_channel_fader` and `set_channel_gain`.
- `backend/wing_client.py:441-460` and `backend/wing_client.py:489-501` return send result and `backend/wing_client.py:1204-1224` implements `confirm_manual_write`.
- `backend/dlive_client.py:501-505` and `backend/dlive_client.py:530-534` expose `set_fader` / `set_gain`, not `set_channel_fader` / `set_channel_gain`, and have no `confirm_manual_write`.
- `backend/mixing_station_client.py:344-359` exposes `set_channel_fader` / `set_channel_gain`, but those methods do not return the send result and there is no `confirm_manual_write`.

Impact:
- dLive manual UI fader/gain through `LiveApplyService` likely blocks as `operation_not_supported_by_client`.
- Mixing Station can send a real command and then be reported as failed/blocked because the setter returns `None`; readback is also unsupported.
- Operator-visible safety status can diverge from actual transport side effects.

Recommended fix:
- Define a real `MixerClientBase` live-write contract: method names, return values, readback support.
- Add client-specific adapters or implement aliases/readback consistently.
- Do not call a sender before confirming that readback verification is supported for a real-write path, unless policy explicitly allows “sent but unconfirmed”.

## Failed Test Findings

### 6. Arrangement automation no longer writes where inventory test expects it

Failure:

```text
assert len(mixer.event_log) == 1
got 0
```

Evidence:
- Test expects one event in `tests/test_direct_write_inventory.py:422-451`.
- Apply code exits early unless `analysis_only_mode=false`, `live_apply_enabled=true`, `enabled=true`, mixer connected, and live apply confirms: `backend/arrangement_automation/controller.py:289-338`.

Likely cause:
- Default confirmation resolution or fake mixer/readback assumptions changed. The test still expects a confirmed write for the live-enabled/fader-ceiling case.

Impact:
- Direct-write inventory proof is no longer stable for arrangement automation.

### 7. Arrangement automation replay applies `{1: 0.0}` instead of recorded `-4.75`

Failure:

```text
assert {1: 0.0} == {1: -4.75}
```

Evidence:
- Test path: `tests/test_direct_write_inventory.py:478-522`.
- `_apply_safe_offsets()` computes `target = min(self.fader_ceiling_db, base + offset)` at `backend/arrangement_automation/controller.py:303-304`.

Impact:
- Replay-safe automation is not reproducing the recorded apply result.

### 8. AutoFader auto-balance clipping uses mixer readback over state and misses expected floor

Failure:

```text
expected -60.0, got -25.0
```

Evidence:
- Test sets `state.current_fader = -55.0`, correction `-20.0`, expects clamp to `-60.0`: `tests/test_direct_write_inventory.py:608-640`.
- Implementation reads current fader from mixer first at `backend/auto_fader.py:2011-2022`, then applies correction and clips at `backend/auto_fader.py:2031-2035`.

Likely cause:
- Fake mixer readback and controller state disagree. Implementation trusts mixer readback, so it computes from a different starting fader.

Impact:
- Auto-balance replay/inventory semantics are ambiguous: source of truth is unclear.

### 9. Replay intelligence canonical benchmark drifted

Failure:

```text
expected trace_signature dd59e257...
actual   trace_signature 2bb403dc...
```

Evidence:
- Assertion in `tests/test_replay_intelligence_benchmark_suite.py:266-281`.
- Baseline fixture is `tests/fixtures/replay_intelligence_benchmark_v1.json`.

Impact:
- Replay executor output changed without fixture update or compatibility explanation.
- This can be real behavioral drift or a stale fixture; either way the benchmark currently fails.

### 10. Room compensation replay blocks cases that tests expect to produce filters

Failures:
- `tests/test_room_compensation_replay.py:122-147`, case0 and case1.

Observed:
- `plan.blocked is True`
- `blocked_reason == "no_safe_filters_generated"`

Evidence:
- Planner blocks if no bounded filters survive: `backend/room_compensation_planner.py:187-200`.
- Fallback candidates require `abs(deviation) >= 2.5`: `backend/room_compensation_planner.py:432-456`.
- The failing “reflective_room” and “conference_hall” fixtures have relatively small magnitude deviations.

Impact:
- The room compensation contract is inconsistent: tests expect a plan with filters, current planner says no safe filters.

## UI/API Integration Gaps And Stubs

### 11. Frontend exposes modules that backend does not implement

Frontend sends:
- `start_auto_panner`, `stop_auto_panner`, `calculate_auto_panning`, `apply_auto_panning`: `frontend/src/services/websocket.js:736-763`
- `start_auto_reverb`, `stop_auto_reverb`, `calculate_auto_reverb`, `apply_auto_reverb`: `frontend/src/services/websocket.js:765-793`
- `start_auto_gate`, `stop_auto_gate`, `configure_gate_channel`: `frontend/src/services/websocket.js:795-817`
- `start_auto_effects`, `stop_auto_effects`: `frontend/src/services/websocket.js:819-833`
- `start_cross_adaptive_eq`, `stop_cross_adaptive_eq`: `frontend/src/services/websocket.js:835-849`

Backend only registers status handlers for those modules:
- `backend/handlers/compressor_handlers.py:24-37`
- `backend/handlers/compressor_handlers.py:64-72`

Server status responses explicitly say not implemented:
- `backend/server.py:4855-4898`

Unknown commands are only logged, not sent back to UI:
- `backend/server.py:469-473`

Impact:
- UI can show full tabs/buttons for modules that do nothing or silently fail on start/apply.

Recommended fix:
- Either hide/disable those tabs until handlers exist, or implement backend handlers returning explicit `unsupported_operation` responses for every command.

### 12. Stub / placeholder inventory

Confirmed stubs/placeholders:
- `backend/auto_fader_v2/ml/data_collector.py:6-14`: `MLDataCollector.add_sample()` is a no-op.
- `backend/auto_fader_v2/core/channel_classifier.py:28-40`: spectral-centroid fallback stub with only 4 drum-ish labels.
- `ai_mixing_pipeline/stem_critics/mert.py:94`: custom trained heads are placeholders.
- `frontend/public/icon.png.placeholder:1`: Electron package expects `frontend/public/icon.png`.
- `backend/server.py:4855-4898`: Auto Gate/Panner/Reverb/Effects/Cross-Adaptive EQ status-only stubs.
- `tools/mcp_remote_gateway/server.js:400-413` and `760-817`: `run_director` is deterministic, in-process, dry-run and does not run Codex/external LLM/real agents. This is safe by design, but should not be marketed as actual multi-agent execution.

## Architecture And Configuration Drift

### 13. Multiple overlapping runtimes remain active

The current docs are accurate about the main issue:
- `backend/server.py` is canonical live coordinator.
- `automixer/production_mix_v1` is offline-first.
- `ai_mixing_pipeline` is offline/research.
- `backend/auto_soundcheck_engine.py`, `backend/auto_fader.py`, arrangement automation, reset/restore tools overlap with live mutation surfaces.

Impact:
- “Which path is source of truth?” is still unresolved for many features.

### 14. Configuration sources conflict

Evidence:
- `README.md` presents Behringer Wing Rack as primary.
- `config/default_config.json` contains Wing and dLive defaults, but server explicitly loads only `config/default_config.json`: `backend/server.py:343-365`.
- `config/automixer.yaml` says target setup is dLive + SoundGrid and sets `mixer.type: dlive`, but server does not load this file by default.
- Frontend connection UI still labels fields as Wing and defaults to `192.168.1.102:2223`: `frontend/src/App.js:55-56`, `frontend/src/App.js:341-352`.
- `config/automixer.yaml` live input trim uses `level_source: wing_meter` even though mixer type is dLive.

Impact:
- Operator can believe they are running dLive/SoundGrid config while backend live server is actually using `default_config.json` and Wing-shaped UI/assumptions.

Recommended fix:
- Pick one canonical runtime config and document which process reads it.
- Make frontend mixer selection explicit: Wing vs dLive vs Mixing Station.
- Remove or clearly mark obsolete config/docs.

### 15. Backend package imports are cwd-dependent

Evidence:
- From repo root, `python3 -c "import backend.server"` fails with `ModuleNotFoundError: No module named 'wing_client'`.
- From `backend/`, `python3 -c "import server"` succeeds.
- `backend/server.py:21-60` uses many absolute local imports.
- Tests compensate by inserting `backend` into `sys.path`, e.g. `tests/conftest.py:8`.

Impact:
- Packaging, external tools, IDEs, and `python -m backend.server` style launches are fragile.

Recommended fix:
- Convert backend to a package with relative imports, or document and enforce `PYTHONPATH=backend`/`cd backend` launch contract consistently.

## Security And Operational Risks

### 16. MCP gateway host policy is broad for Cloudflare tunnels

Evidence:
- Hardcoded public URL and path: `tools/mcp_remote_gateway/server.js:15-18`.
- Default allowed hosts include specific Cloudflare tunnel domains: `tools/mcp_remote_gateway/server.js:53-59`.
- `isAllowedHost()` allows any hostname ending in `.trycloudflare.com`: `tools/mcp_remote_gateway/server.js:179-180`.

Impact:
- This may be intentional for ChatGPT Actions testing, but it is broader than the explicit allowlist and should be reviewed before public exposure.

### 17. Old MCP gateway remains in the tree

Evidence:
- `tools/mcp_remote_gateway_old/server.js` is a one-line old server.
- It binds `0.0.0.0`, writes `.secret`, and exposes a minimal JSON-RPC-like tool surface.

Impact:
- If started accidentally, it creates another public-ish control surface with different behavior from the current gateway.

Recommended fix:
- Archive clearly, remove from active docs, or add a README warning that it must not be launched.

### 18. Full WebSocket messages are logged

Evidence:
- `backend/server.py:462-467` logs first 200 chars and full message payload.

Impact:
- Useful for debugging, but unsafe if future messages include credentials, personal data, or large payloads.

## Audio/Device Reliability Risks

### 19. Two audio device discovery paths behave differently

Evidence:
- `backend/audio_devices.py:51-58` falls back to mock devices if PyAudio is unavailable or device enumeration errors.
- `backend/audio_capture.py:48-67` uses sounddevice and returns an empty list if unavailable/error.

Impact:
- UI may show fake `Wing USB (48 channels)` even when no real capture device exists, while actual capture service may have no device.

Recommended fix:
- Make fake devices opt-in test mode.
- Use one canonical device discovery service for UI and capture.

## Positive Findings

- Python syntax compiles across backend, automixer, ai pipeline, tools, scripts, and tests.
- Frontend unit tests and production build pass.
- MCP remote gateway and MCP director parse with `node --check`.
- `automixer.production_mix_v1` CLI help works from repo root now.
- Config defaults for key new live automation surfaces are conservative:
  - `config/default_config.json`: `live_apply_enabled=false`, `analysis_only_mode=true`.
  - `config/automixer.yaml`: `soundcheck.auto_apply=false`, live input trim apply disabled, AutoEQ OSC dry-run enabled.
- The documentation now correctly warns that `LiveApplyService` is not system-wide and panic stop is not global.

## GPT Action Plan

Recommended order:

1. Security first: rotate the tracked OpenAI key, remove `.env` from git, add `.env.example`.
2. Make tests green before live claims:
   - fix or intentionally update direct-write inventory expectations;
   - decide whether replay benchmark hash drift is intended;
   - align room compensation tests with planner threshold behavior or adjust planner fallback generation.
3. Fix Multi-Channel AutoEQ `mode` signature mismatch.
4. Normalize `LiveApplyService` client contract across Wing, dLive, and Mixing Station.
5. Add explicit unsupported responses for frontend modules without backend implementation, or hide those tabs.
6. Inventory and gate direct live-write bypasses:
   - AutoEQ apply/reset;
   - phase apply;
   - AutoSoundcheck apply and feedback reaction;
   - system measurement apply;
   - restore/reset scripts.
7. Decide canonical config source and launch method.
8. Mark Paperclip `run_director` as deterministic dry-run director, not real agent execution, unless a true executor is added.

## Readiness Statement

- Offline/report-only: usable for continued development and dry-run analysis.
- Frontend: builds, but several visible modules are status-only or unhandled.
- Backend live runtime: partially usable, but not globally safe.
- Real console: only guarded/manual Wing `set_fader` / `set_gain` has a defensible gate/readback story. All other live mutation paths require explicit review before use.
