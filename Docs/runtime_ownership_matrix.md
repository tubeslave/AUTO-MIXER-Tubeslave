# Runtime Ownership Matrix

## Purpose And Reading Rules

This document is the first-pass runtime ownership ledger for console-control and adjacent runtime paths in this repository.

Reading rules:

- One row represents one runtime path, not one Python file.
- `owner_path / canonical_module` is the proposed owner after current board guidance, not a blanket approval for live use.
- `current_safety_gate` describes what exists in code today, not what should exist later.
- `panic_stop_coverage` is intentionally strict:
  - `manual_only` means only the manual `LiveApplyService` path is documented as covered.
  - `none` means no current panic-stop guarantee was found for that path.
  - `unclear` means code and product semantics do not line up cleanly enough to certify.
- Any doubtful area is marked `unclear` instead of guessed.
- This file does not change policy. It records the current state and the current recommended direction.

## Column Definitions

- `entrypoint`: How the path is reached at runtime.
- `module/file`: Primary implementation file or file chain for the path.
- `purpose`: What the path is intended to do.
- `classification`: `live`, `offline`, `research`, `legacy`, or `unclear`.
- `can_send_osc`: Whether the path can send real console writes today.
- `current_safety_gate`: Existing gate or local limiter, if any.
- `dry_run_default`: Whether dry-run is the default behavior for this path.
- `confirm_required`: Whether an explicit confirm gate is required for real writes.
- `readback_required`: Whether readback confirmation is required for real writes.
- `panic_stop_coverage`: Current panic-stop guarantee for the path.
- `owner_path / canonical_module`: Proposed owner path after current board-confirmed decisions.
- `risk_level`: `low`, `medium`, `high`, or `unclear`.
- `recommendation`: Current recommended action.
- `evidence`: File and line evidence for the row.

## First Revision Rows

| entrypoint | module/file | purpose | classification | can_send_osc | current_safety_gate | dry_run_default | confirm_required | readback_required | panic_stop_coverage | owner_path / canonical_module | risk_level | recommendation | evidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `python backend/server.py` | `backend/server.py` | Main live runtime coordinator. Owns mixer client, handler dispatch, controllers, audio capture, and several direct-write flows. | `live` | `yes` | `mixed`: some flows use `LiveApplyService`, others bypass it directly | `unclear` | `unclear` | `unclear` | `manual_only` | `backend/server.py` | `high` | `keep` | `backend/server.py:104-176`; `backend/server.py:5024-5058` |
| WebSocket manual UI `set_fader` / `set_gain` | `backend/handlers/mixer_handlers.py` + `backend/live_apply.py` | Manual fader and gain writes from UI. | `live` | `yes` | `LiveApplyService` | `yes` | `yes` | `yes` | `manual_only` | `backend/server.py -> backend/handlers/mixer_handlers.py -> backend/live_apply.py` | `medium` | `keep` | `backend/handlers/mixer_handlers.py:80-112`; `backend/live_apply.py:93-176` |
| WebSocket manual UI `set_eq` / `set_compressor` | `backend/handlers/mixer_handlers.py` + `backend/live_apply.py` | Manual EQ and compressor requests from UI. Public handler exists, but real write path is not approved/open. | `unclear` | `unclear` | `blocked_at_gate`: `LiveApplyService` returns `unsupported_operation` for these operations | `unclear` | `unclear` | `no` | `manual_only` | `unclear` | `medium` | `document` | `backend/handlers/mixer_handlers.py:114-158`; `backend/live_apply.py:101-108` |
| Auto fader automation loop | `backend/auto_fader.py` | Automated fader updates during auto-fader processing. | `live` | `yes` | local freeze / channel-freeze checks only | `no` | `no` | `no` | `none` | `unclear`; board-confirmed as direct bypass, not approved gate | `high` | `migrate_later` | `backend/auto_fader.py:1525-1543`; `backend/auto_fader.py:1966-1973` |
| Arrangement automation controller | `backend/arrangement_automation/controller.py` | Arrangement-aware fader offsets based on activity, density, masking, and section logic. | `live` | `yes` | controller-local `SafetyLimiter` plus `analysis_only_mode` / `live_apply_enabled` | `yes` | `no` | `no` | `none` | `backend/server.py -> backend/arrangement_automation/controller.py` | `high` | `migrate_later` | `backend/arrangement_automation/controller.py:267-288` |
| Live input trim | `backend/arrangement_automation/live_input_trim_controller.py` | Canonical live input trim path. Role-aware, bleed-aware, meter-aware gain changes. | `live` | `yes` | `LiveApplyService` plus controller-local bounds, cooldown, deadband, and role limits | `yes` | `yes` | `yes` | `manual_only` | `backend/server.py -> backend/arrangement_automation/live_input_trim_controller.py` | `medium` | `keep` | `backend/arrangement_automation/live_input_trim_controller.py:651-690` |
| Headless auto soundcheck engine | `backend/auto_soundcheck_engine.py` | Soundcheck flow that can analyze channels and optionally apply EQ, compressor, pan, fader, and gain-related changes. Board-confirmed as active but overlapping, not canonical. | `live` | `yes` | local `auto_apply` flag; no unified gate for all write types | `yes` | `yes` | `no` | `none` | `backend/server.py -> backend/arrangement_automation/live_input_trim_controller.py` is canonical live trim owner; this path remains overlapping | `high` | `document` | `backend/auto_soundcheck_engine.py:1077-1090`; `backend/auto_soundcheck_engine.py:1514-1524`; `backend/auto_soundcheck_engine.py:1881-1883`; `backend/auto_soundcheck_engine.py:1934-1973` |
| Feedback reaction path | `backend/feedback_detector.py` -> `backend/auto_soundcheck_engine.py::_handle_feedback_event` | Detect feedback and react by notch EQ or fader reduction. Board-confirmed only as a safety-critical emergency exception candidate, not an approved exception. | `unclear` | `unclear` | detector-side analysis exists, but write-side gate is direct client calls | `unclear` | `no` | `no` | `none` | `unclear`; requires dedicated policy review | `high` | `document` | `backend/feedback_detector.py:1-8`; `backend/auto_soundcheck_engine.py:1686-1708` |
| Voice-command writes | `backend/server.py` voice command handling + `backend/live_apply.py` | Voice path for `set_fader` and `set_gain`. Unsupported voice writes like snapshot load are blocked. | `live` | `yes` for fader/gain; `no` for blocked operations | `LiveApplyService` for fader/gain; explicit block for unsupported voice operations | `yes` | `yes` | `yes` | `manual_only` | `backend/server.py -> backend/live_apply.py` | `medium` | `keep` | `backend/server.py:5006-5059` |
| Snapshot / reset / restore utility scripts | `backend/restore_channels.py`, `backend/load_snap_final.py`, `backend/find_and_load_snap.py` | Operator utility scripts for restore, snapshot load, and direct console state changes. Board-confirmed as operator-only legacy tools / archive candidates. | `legacy` | `yes` | none beyond ad hoc sleeps / manual verification patterns | `no` | `no` | `unclear` | `none` | `unclear`; not a canonical runtime path | `high` | `archive` | `backend/restore_channels.py:21-46`; `backend/restore_channels.py:70-127`; `backend/load_snap_final.py:52-104`; `backend/find_and_load_snap.py:53-92`; `backend/find_and_load_snap.py:124-182` |
| `production_mix_v1` OSC queue | `automixer/production_mix_v1/osc_queue.py` | Offline-first queue that records would-send commands and only transmits when a real sender is attached and `dry_run=False`. | `offline` | `yes`, but only when a sender is attached and dry-run is disabled | `ProductionSafetyGovernor` before queueing; queue-local dry-run behavior; no `LiveApplyService` | `yes` | `no` | `no` | `none` | `python -m automixer.production_mix_v1 -> automixer/production_mix_v1/pipeline.py` | `medium` | `document` | `automixer/production_mix_v1/osc_queue.py:13-36` |
| `ai_mixing_pipeline` offline pipeline | `ai_mixing_pipeline/__init__.py` + `ai_mixing_pipeline/decision_layer/virtual_mixer_base.py` + `ai_mixing_pipeline/decision_layer/safety_governor.py` | Offline-only AI mixing pipeline for local candidate rendering and reports. Included because the board asked to track it if it could ever attach to a real sender. Current evidence says it is intentionally separate from live OSC/MIDI. | `research` | `no` | offline-only contract and report-only warnings for FX send behavior | `yes` | `no` | `no` | `none` | `ai_mixing_pipeline` remains non-canonical for live console control | `low` | `document` | `ai_mixing_pipeline/__init__.py:1-5`; `ai_mixing_pipeline/decision_layer/virtual_mixer_base.py:12-14`; `ai_mixing_pipeline/decision_layer/safety_governor.py:46-50` |

## Notes On Unclear Rows

- Manual `set_eq` / `set_compressor` are exposed at the handler layer but currently blocked by `LiveApplyService` because those operations are not in the supported operation set. They are not approved real-write paths, but the underlying mixer client layer does contain direct EQ and compressor write methods.
- The feedback reaction path can clearly call direct client methods for EQ notch writes. The fader-reduction branch uses `self.mixer_client.set_fader(...)`, which should be treated as implementation-risky until audited against the actual client interface.
- The feedback reaction row must stay policy-separated from the headless AutoSoundcheck row. `AutoSoundcheck` is a broad correction surface; feedback reaction is a narrow emergency exception candidate. See `Docs/TUB-194_AUTOSOUNDCHECK_FEEDBACK_POLICY_SPLIT.md`.
- `panic_stop_coverage` is intentionally conservative. Even where `LiveApplyService` is reused outside manual UI, the current board instruction is to document panic-stop semantics as `not system-wide` and `manual LiveApplyService paths only`.

## Open Decisions Remaining After This Revision

- Whether the feedback reaction path should later become an approved emergency exception or be migrated into a separate gated model.
- Whether utility scripts should remain `operator-only` permanently or move from `archive candidates` to actual archival/removal work.
- Whether any offline sender attachment path should ever be allowed to bypass a unified console-write gate in the future.
