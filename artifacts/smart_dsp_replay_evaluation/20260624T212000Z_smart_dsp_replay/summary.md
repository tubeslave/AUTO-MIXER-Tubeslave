# Smart DSP Replay Evaluation

- schema: `automixer.smart_dsp_replay_evaluation.v1`
- status: `passed`
- method_id: `smart_dsp_strategy_engine`
- scenario_count: `15`
- passed_scenario_count: `15`
- failed_scenario_count: `0`
- pass_rate: `1.0`
- read_only_or_dry_run_only: `True`
- live_authority_granted: `False`
- live_apply_allowed: `False`
- queue_import_allowed: `False`

## Scenario Results

### soundcheck_muddy_vocal

- status: `passed`
- expected_selected_strategy_id: `low_mid_corrective_cut`
- actual_selected_strategy_id: `low_mid_corrective_cut`
- actual_method_family: `eq`
- candidate_count: `2`
- selected_total_score: `0.842`
- global_blockers: `none`

### live_sibilant_vocal

- status: `passed`
- expected_selected_strategy_id: `vocal_de_esser`
- actual_selected_strategy_id: `vocal_de_esser`
- actual_method_family: `de_esser`
- candidate_count: `3`
- selected_total_score: `0.878`
- global_blockers: `none`

### soundcheck_dynamic_vocal_compression

- status: `passed`
- expected_selected_strategy_id: `gentle_channel_compression`
- actual_selected_strategy_id: `gentle_channel_compression`
- actual_method_family: `dynamics`
- candidate_count: `3`
- selected_total_score: `0.813`
- global_blockers: `none`

### soundcheck_quiet_vocal_level_plane_gain_up

- status: `passed`
- expected_selected_strategy_id: `soundcheck_bounded_gain_up`
- actual_selected_strategy_id: `soundcheck_bounded_gain_up`
- actual_method_family: `gain_fader`
- candidate_count: `3`
- selected_total_score: `0.832`
- global_blockers: `none`

### live_loud_bass_level_plane_gain_down

- status: `passed`
- expected_selected_strategy_id: `level_plane_gain_down`
- actual_selected_strategy_id: `level_plane_gain_down`
- actual_method_family: `gain_fader`
- candidate_count: `3`
- selected_total_score: `0.882`
- global_blockers: `none`

### kick_bass_masking_low_headroom

- status: `passed`
- expected_selected_strategy_id: `kick_bass_pocket_eq_split`
- actual_selected_strategy_id: `kick_bass_pocket_eq_split`
- actual_method_family: `eq`
- candidate_count: `7`
- selected_total_score: `0.858`
- global_blockers: `true_peak_headroom_low`

### soundcheck_kick_bass_dynamic_pocket

- status: `passed`
- expected_selected_strategy_id: `kick_bass_dynamic_pocket_ducking`
- actual_selected_strategy_id: `kick_bass_dynamic_pocket_ducking`
- actual_method_family: `dynamic_eq`
- candidate_count: `3`
- selected_total_score: `0.87`
- global_blockers: `none`

### live_kick_bass_conservative_eq_split

- status: `passed`
- expected_selected_strategy_id: `kick_bass_pocket_eq_split`
- actual_selected_strategy_id: `kick_bass_pocket_eq_split`
- actual_method_family: `eq`
- candidate_count: `3`
- selected_total_score: `0.878`
- global_blockers: `none`

### noisy_tom_tail_risk

- status: `passed`
- expected_selected_strategy_id: `gentle_expander_bleed_control`
- actual_selected_strategy_id: `gentle_expander_bleed_control`
- actual_method_family: `expander`
- candidate_count: `2`
- selected_total_score: `0.802`
- global_blockers: `none`

### wet_fx_monitor_risk

- status: `passed`
- expected_selected_strategy_id: `reduce_fx_send_or_return`
- actual_selected_strategy_id: `reduce_fx_send_or_return`
- actual_method_family: `fx`
- candidate_count: `5`
- selected_total_score: `0.833`
- global_blockers: `none`

### soundcheck_long_fx_tail_decay_filter

- status: `passed`
- expected_selected_strategy_id: `shorten_fx_decay_filter`
- actual_selected_strategy_id: `shorten_fx_decay_filter`
- actual_method_family: `fx`
- candidate_count: `5`
- selected_total_score: `0.841`
- global_blockers: `none`

### pan_phase_mono_failure

- status: `passed`
- expected_selected_strategy_id: `center_hold_for_phase_safety`
- actual_selected_strategy_id: `center_hold_for_phase_safety`
- actual_method_family: `pan`
- candidate_count: `2`
- selected_total_score: `0.822`
- global_blockers: `none`

### master_true_peak_overload

- status: `passed`
- expected_selected_strategy_id: `master_trim_for_true_peak_headroom`
- actual_selected_strategy_id: `master_trim_for_true_peak_headroom`
- actual_method_family: `master_bus`
- candidate_count: `2`
- selected_total_score: `0.848`
- global_blockers: `none`

### stale_context_blocks_strategy

- status: `passed`
- expected_selected_strategy_id: `high_pass_cleanup`
- actual_selected_strategy_id: `high_pass_cleanup`
- actual_method_family: `eq`
- candidate_count: `2`
- selected_total_score: `0.799`
- global_blockers: `smart_dsp_context_cache_not_fresh`

### live_authority_context_blocks_strategy

- status: `passed`
- expected_selected_strategy_id: `high_pass_cleanup`
- actual_selected_strategy_id: `high_pass_cleanup`
- actual_method_family: `eq`
- candidate_count: `2`
- selected_total_score: `0.799`
- global_blockers: `smart_dsp_context_cache_live_authority_present`
