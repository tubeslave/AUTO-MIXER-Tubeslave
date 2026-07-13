# WebSocket-протокол AUTO-MIXER (frontend ↔ backend)

Сгенерировано из регистров `backend/handlers/*.py` (входящие) и
grep по `"type": "..."` (исходящие). Обновлять при изменении handlers.

Транспорт: WebSocket, JSON-сообщения вида `{"type": "<имя>", ...поля}`.
Backend слушает только loopback. Packaged Electron и удалённый HTTPS/WSS
клиент передают 43-символьный base64url token и
`client_id="automixer-ui"` первым WebSocket frame. Token в URL, заголовках
страницы или browser storage не поддерживается. Ошибка/истечение
аутентификации закрывает socket кодом `1008`. Удалённый браузер допускается
только через trusted same-origin `wss://<host>/ws` reverse proxy к
`127.0.0.1:8765`; это ещё не release-ready deployment.

Heartbeat: клиент шлёт `{"type": "ping", "ts": ...}`, сервер отвечает
`{"type": "pong", "ts": ..., "mixer_connected": bool}`.

## Входящие сообщения (137 типов)

### agent (10)
- `approve_action`
- `approve_all_actions`
- `dismiss_action`
- `dismiss_all_actions`
- `get_action_history`
- `get_agent_status`
- `get_audio_capture_status`
- `get_channel_meters`
- `get_pending_actions`
- `set_agent_mode`

### audio (4)
- `get_audio_devices`
- `get_dante_routing`
- `scan_audio_devices`
- `select_audio_device`

### automation (15)
- `emergency_stop`
- `get_freeze_status`
- `get_manual_write_session`
- `get_manual_write_supervision`
- `get_operator_mode`
- `get_ready_for_live`
- `get_supervised_pilot_status`
- `rollback_all`
- `rollback_last`
- `set_automation_frozen`
- `set_channel_frozen`
- `set_live_mode`
- `set_manual_write_supervision`
- `set_operator_mode`
- `start_manual_write_session`

### arrangement_automation (5; fail-closed)
- `get_arrangement_automation_status`
- `run_arrangement_automation_tick`
- `set_arrangement_automation_live_apply`
- `start_arrangement_automation`
- `stop_arrangement_automation`

Все пять возвращают `reason=arrangement_automation_not_promoted`;
`live_apply_enabled=false` и `live_authority_granted=false`.

### channel_scan (5)
- `bypass_mixer`
- `probe_wing_level_telemetry`
- `reset_trim`
- `scan_channel_names`
- `scan_mixer_channel_names`

### compressor (14)
- `get_auto_compressor_status`
- `get_auto_effects_status`
- `get_auto_gate_status`
- `get_auto_panner_status`
- `get_auto_reverb_status`
- `get_cross_adaptive_eq_status`
- `set_auto_compressor_manual`
- `set_auto_compressor_profile`
- `start_auto_compressor`
- `start_auto_compressor_live`
- `start_auto_compressor_soundcheck`
- `stop_auto_compressor`
- `stop_auto_compressor_live`
- `stop_auto_compressor_soundcheck`

### connection (11)
- `auto_connect`
- `connect_dlive`
- `connect_mixing_station`
- `connect_wing`
- `disconnect_mixer`
- `disconnect_wing`
- `discover_mixing_station`
- `ping`
- `scan_mixers`
- `start_wing_telemetry_capture`
- `stop_wing_telemetry_capture`

### eq (13)
- `apply_all_corrections`
- `apply_channel_correction`
- `apply_eq_correction`
- `get_auto_eq_status`
- `get_eq_profiles`
- `reset_all_eq`
- `reset_eq`
- `set_channel_profile`
- `set_eq_profile`
- `start_auto_eq`
- `start_multi_channel_auto_eq`
- `stop_auto_eq`
- `stop_multi_channel_auto_eq`

### fader (14)
- `apply_auto_balance`
- `cancel_auto_balance`
- `get_auto_fader_status`
- `load_all_settings`
- `load_auto_fader_defaults`
- `lock_channel`
- `save_all_settings`
- `save_auto_fader_defaults`
- `set_auto_fader_profile`
- `start_auto_balance`
- `start_realtime_fader`
- `stop_realtime_fader`
- `unlock_channel`
- `update_auto_fader_settings`

### gain_staging (4)
- `get_gain_staging_status`
- `start_realtime_correction`
- `stop_realtime_correction`
- `update_safe_gain_settings`

### mixer (5)
- `get_state`
- `set_compressor`
- `set_eq`
- `set_fader`
- `set_gain`

### phase (5)
- `apply_phase_corrections`
- `get_phase_alignment_status`
- `reset_phase_delay`
- `start_phase_alignment`
- `stop_phase_alignment`

### product_state (11)
- `accept_operator_proposal`
- `apply_operator_proposal`
- `create_operator_proposal`
- `dismiss_operator_proposal`
- `get_channel_inventory`
- `get_connection_topology`
- `get_dashboard_snapshot`
- `get_decision_queue`
- `import_safe_gain_suggestions`
- `import_soundcheck_recommendations`
- `run_operator_analysis`

### routing (6)
- `get_channel_input_routing`
- `get_output_routing`
- `route_multiple_outputs`
- `route_output`
- `set_channel_alt_input`
- `set_channel_input`

### snapshot (5)
- `create_snapshot`
- `load_snap`
- `save_snap`
- `undo_restore_snapshot`
- `undo_snapshot_create`

### soundcheck (6)
- `get_auto_engine_status`
- `get_auto_soundcheck_status`
- `start_auto_engine`
- `start_auto_soundcheck`
- `stop_auto_engine`
- `stop_auto_soundcheck`

### voice (4)
- `get_voice_control_status`
- `rescan_channel_names`
- `start_voice_control`
- `stop_voice_control`

## Исходящие сообщения (156 типов)

Статусы/результаты, рассылаемые сервером (broadcast или ответ):

- `action_approved`
- `action_dismissed`
- `action_history`
- `agent_mode_changed`
- `agent_status`
- `all_actions_approved`
- `all_actions_dismissed`
- `all_settings_loaded`
- `all_settings_saved`
- `apply_gain_adjustments`
- `audio_capture_status`
- `audio_device_scan`
- `audio_device_selected`
- `audio_devices`
- `auto_compressor_status`
- `auto_connect_status`
- `auto_effects_status`
- `auto_engine_channel`
- `auto_engine_state`
- `auto_engine_status`
- `auto_eq_apply_result`
- `auto_eq_corrections`
- `auto_eq_reset_result`
- `auto_eq_spectrum`
- `auto_eq_status`
- `auto_fader_defaults_loaded`
- `auto_fader_defaults_saved`
- `auto_fader_status`
- `auto_gate_status`
- `auto_panner_status`
- `auto_reverb_status`
- `auto_soundcheck_channel_recognizer`
- `auto_soundcheck_status`
- `automation_frozen_set`
- `bypass_result`
- `channel_frozen_set`
- `channel_input_routing`
- `channel_inventory`
- `channel_meters`
- `channel_scan_result`
- `compressor_gain_down`
- `compressor_gain_up`
- `compressor_on`
- `compressor_threshold`
- `config_reloaded`
- `connect_wing`
- `connection_status`
- `connection_topology`
- `cooldown_wait`
- `cross_adaptive_eq_status`
- `dante_routing`
- `dashboard_snapshot`
- `decision_queue`
- `dry_run_reference`
- `emergency_stop`
- `emergency_stop_applied`
- `eq_band_down`
- `eq_band_up`
- `eq_on`
- `eq_profiles`
- `error`
- `freeze_status`
- `gain_down`
- `gain_staging_status`
- `gain_up`
- `get_audio_devices`
- `get_auto_compressor_status`
- `get_auto_effects_status`
- `get_auto_eq_status`
- `get_auto_fader_status`
- `get_auto_gate_status`
- `get_auto_panner_status`
- `get_auto_reverb_status`
- `get_gain_staging_status`
- `get_manual_write_session`
- `get_manual_write_supervision`
- `get_voice_control_status`
- `high_shelf`
- `live_mode_set`
- `load_snap`
- `load_snap_result`
- `low_shelf`
- `manual_write_blocked`
- `manual_write_error`
- `manual_write_rollback`
- `manual_write_rollback_applied`
- `manual_write_rollback_blocked`
- `manual_write_session`
- `manual_write_supervision`
- `mixer_channel_names`
- `mixer_scan_results`
- `mixer_update`
- `mixing_station_discovered`
- `multi_channel_apply_result`
- `multi_channel_auto_eq_status`
- `multi_channel_corrections`
- `multi_channel_spectrum`
- `multi_channel_status`
- `mute_channel`
- `operator_analysis_report`
- `operator_mode_status`
- `operator_proposal_accepted`
- `operator_proposal_apply_blocked`
- `operator_proposal_apply_result`
- `operator_proposal_created`
- `operator_proposal_dismissed`
- `operator_proposals_imported`
- `output_routing`
- `peak`
- `pending_actions`
- `phase_alignment_apply_result`
- `phase_alignment_measurements`
- `phase_alignment_reset_result`
- `phase_alignment_status`
- `ping`
- `pong`
- `probe_wing_level_telemetry`
- `ready_for_live`
- `rescan_complete`
- `reset_all_eq_result`
- `reset_trim_result`
- `restore_result`
- `rollback_all`
- `rollback_last`
- `route_multiple_outputs_result`
- `route_output_result`
- `safe_gain_settings_updated`
- `save_snap_result`
- `scan_mixer_channel_names`
- `set_channel_alt_input_result`
- `set_channel_input_result`
- `set_compressor`
- `set_eq`
- `set_fader`
- `set_gain`
- `set_manual_write_supervision`
- `snapshot_result`
- `solo_channel`
- `start_gain_staging`
- `start_manual_write_session`
- `start_realtime_correction`
- `start_voice_control`
- `state_update`
- `stop_gain_staging`
- `stop_realtime_correction`
- `stop_voice_control`
- `stop_wing_telemetry_capture`
- `supervised_pilot_status`
- `voice_command_blocked`
- `voice_command_error`
- `voice_command_executed`
- `voice_control_status`
- `volume_down`
- `volume_up`
- `wing_level_telemetry`
- `wing_telemetry_capture_status`
