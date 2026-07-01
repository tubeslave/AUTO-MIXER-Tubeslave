# Smart DSP Replay Evaluation

- schema: `automixer.smart_dsp_replay_evaluation.v1`
- status: `failed`
- method_id: `smart_dsp_strategy_engine`
- scenario_count: `146`
- passed_scenario_count: `145`
- failed_scenario_count: `1`
- pass_rate: `0.993`
- selected_unsupported_wing_osc_count: `0`
- selected_unsupported_wing_osc_by_family: `{}`
- selected_review_only_wing_osc_parameter_count: `13`
- selected_review_only_wing_osc_by_family: `{'expander': 4, 'dynamics': 8, 'gate': 1}`
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

### soundcheck_rumble_vocal_hpf

- status: `passed`
- expected_selected_strategy_id: `rumble_dominant_high_pass_cleanup`
- actual_selected_strategy_id: `rumble_dominant_high_pass_cleanup`
- actual_method_family: `eq`
- candidate_count: `3`
- selected_total_score: `0.883`
- global_blockers: `none`

### soundcheck_plosive_vocal_dynamic_hpf

- status: `passed`
- expected_selected_strategy_id: `vocal_plosive_dynamic_hpf_control`
- actual_selected_strategy_id: `vocal_plosive_dynamic_hpf_control`
- actual_method_family: `dynamic_eq`
- candidate_count: `11`
- selected_total_score: `0.901`
- global_blockers: `none`

### live_plosive_vocal_hold_before_hpf

- status: `passed`
- expected_selected_strategy_id: `manual_plosive_mic_technique_review`
- actual_selected_strategy_id: `manual_plosive_mic_technique_review`
- actual_method_family: `analysis_only`
- candidate_count: `5`
- selected_total_score: `0.912`
- global_blockers: `none`

### soundcheck_plosive_vocal_deq_gate_slot_conflict_hpf

- status: `passed`
- expected_selected_strategy_id: `static_plosive_high_pass_cleanup`
- actual_selected_strategy_id: `static_plosive_high_pass_cleanup`
- actual_method_family: `eq`
- candidate_count: `11`
- selected_total_score: `0.803`
- global_blockers: `none`

### soundcheck_plosive_vocal_eq_model_conflict_hold

- status: `passed`
- expected_selected_strategy_id: `manual_plosive_mic_technique_review`
- actual_selected_strategy_id: `manual_plosive_mic_technique_review`
- actual_method_family: `analysis_only`
- candidate_count: `5`
- selected_total_score: `0.809`
- global_blockers: `none`

### soundcheck_vocal_noise_floor_expander

- status: `passed`
- expected_selected_strategy_id: `vocal_phrase_noise_downward_expander`
- actual_selected_strategy_id: `vocal_phrase_noise_downward_expander`
- actual_method_family: `expander`
- candidate_count: `8`
- selected_total_score: `0.878`
- global_blockers: `none`

### soundcheck_vocal_phrase_noise_before_de_esser_compressor

- status: `passed`
- expected_selected_strategy_id: `vocal_phrase_noise_downward_expander`
- actual_selected_strategy_id: `vocal_phrase_noise_downward_expander`
- actual_method_family: `expander`
- candidate_count: `14`
- selected_total_score: `0.878`
- global_blockers: `none`

### soundcheck_quiet_noisy_vocal_gain_stage_review

- status: `passed`
- expected_selected_strategy_id: `mic_gain_stage_noise_floor_review`
- actual_selected_strategy_id: `mic_gain_stage_noise_floor_review`
- actual_method_family: `analysis_only`
- candidate_count: `7`
- selected_total_score: `0.771`
- global_blockers: `none`

### live_vocal_noise_floor_review_before_expander

- status: `passed`
- expected_selected_strategy_id: `mic_gain_stage_noise_floor_review`
- actual_selected_strategy_id: `mic_gain_stage_noise_floor_review`
- actual_method_family: `analysis_only`
- candidate_count: `8`
- selected_total_score: `0.849`
- global_blockers: `none`

### soundcheck_line_noise_hum_notch

- status: `passed`
- expected_selected_strategy_id: `mains_hum_notch_stack`
- actual_selected_strategy_id: `mains_hum_notch_stack`
- actual_method_family: `eq`
- candidate_count: `6`
- selected_total_score: `0.893`
- global_blockers: `none`

### soundcheck_bass_di_line_noise_before_gain_up

- status: `passed`
- expected_selected_strategy_id: `mains_hum_notch_stack`
- actual_selected_strategy_id: `mains_hum_notch_stack`
- actual_method_family: `eq`
- candidate_count: `7`
- selected_total_score: `0.893`
- global_blockers: `none`

### live_line_noise_physical_path_before_notch_or_dynamics

- status: `passed`
- expected_selected_strategy_id: `di_ground_lift_line_check`
- actual_selected_strategy_id: `di_ground_lift_line_check`
- actual_method_family: `analysis_only`
- candidate_count: `8`
- selected_total_score: `0.887`
- global_blockers: `none`

### live_vocal_plosive_line_noise_mic_review_before_hum

- status: `passed`
- expected_selected_strategy_id: `manual_plosive_mic_technique_review`
- actual_selected_strategy_id: `manual_plosive_mic_technique_review`
- actual_method_family: `analysis_only`
- candidate_count: `7`
- selected_total_score: `0.912`
- global_blockers: `none`

### soundcheck_dynamic_muddy_vocal

- status: `passed`
- expected_selected_strategy_id: `vocal_dynamic_low_mid_de_mud`
- actual_selected_strategy_id: `vocal_dynamic_low_mid_de_mud`
- actual_method_family: `dynamic_eq`
- candidate_count: `8`
- selected_total_score: `0.859`
- global_blockers: `none`

### live_dynamic_muddy_vocal_static_low_mid_priority

- status: `passed`
- expected_selected_strategy_id: `low_mid_corrective_cut`
- actual_selected_strategy_id: `low_mid_corrective_cut`
- actual_method_family: `eq`
- candidate_count: `8`
- selected_total_score: `0.849`
- global_blockers: `none`

### soundcheck_dynamic_muddy_guitar_low_mid_bloom

- status: `passed`
- expected_selected_strategy_id: `instrument_dynamic_low_mid_de_mud`
- actual_selected_strategy_id: `instrument_dynamic_low_mid_de_mud`
- actual_method_family: `dynamic_eq`
- candidate_count: `7`
- selected_total_score: `0.859`
- global_blockers: `none`

### live_dynamic_muddy_guitar_static_low_mid_priority

- status: `passed`
- expected_selected_strategy_id: `low_mid_corrective_cut`
- actual_selected_strategy_id: `low_mid_corrective_cut`
- actual_method_family: `eq`
- candidate_count: `7`
- selected_total_score: `0.849`
- global_blockers: `none`

### soundcheck_fx_wash_muddy_vocal_cleanup

- status: `passed`
- expected_selected_strategy_id: `reduce_fx_send_or_return`
- actual_selected_strategy_id: `reduce_fx_send_or_return`
- actual_method_family: `fx`
- candidate_count: `7`
- selected_total_score: `0.782`
- global_blockers: `none`

### soundcheck_dull_vocal_presence_lift

- status: `passed`
- expected_selected_strategy_id: `soundcheck_presence_air_lift`
- actual_selected_strategy_id: `soundcheck_presence_air_lift`
- actual_method_family: `eq`
- candidate_count: `3`
- selected_total_score: `0.801`
- global_blockers: `none`

### soundcheck_dull_vocal_monitor_feedback_hold

- status: `passed`
- expected_selected_strategy_id: `hold_presence_boost_for_live_safety`
- actual_selected_strategy_id: `hold_presence_boost_for_live_safety`
- actual_method_family: `analysis_only`
- candidate_count: `3`
- selected_total_score: `0.689`
- global_blockers: `none`

### soundcheck_dull_vocal_elevated_feedback_path_hold

- status: `passed`
- expected_selected_strategy_id: `hold_presence_boost_for_live_safety`
- actual_selected_strategy_id: `hold_presence_boost_for_live_safety`
- actual_method_family: `analysis_only`
- candidate_count: `3`
- selected_total_score: `0.689`
- global_blockers: `none`

### soundcheck_dull_vocal_summed_bus_headroom_hold

- status: `passed`
- expected_selected_strategy_id: `hold_presence_boost_for_live_safety`
- actual_selected_strategy_id: `hold_presence_boost_for_live_safety`
- actual_method_family: `analysis_only`
- candidate_count: `3`
- selected_total_score: `0.689`
- global_blockers: `none`

### soundcheck_backing_vocal_role_aware_presence_lift

- status: `passed`
- expected_selected_strategy_id: `soundcheck_presence_air_lift`
- actual_selected_strategy_id: `soundcheck_presence_air_lift`
- actual_method_family: `eq`
- candidate_count: `3`
- selected_total_score: `0.788`
- global_blockers: `none`

### live_dull_vocal_presence_hold

- status: `passed`
- expected_selected_strategy_id: `hold_presence_boost_for_live_safety`
- actual_selected_strategy_id: `hold_presence_boost_for_live_safety`
- actual_method_family: `analysis_only`
- candidate_count: `3`
- selected_total_score: `0.786`
- global_blockers: `none`

### live_vocal_feedback_gain_pullback

- status: `passed`
- expected_selected_strategy_id: `feedback_gain_pullback`
- actual_selected_strategy_id: `feedback_gain_pullback`
- actual_method_family: `gain_fader`
- candidate_count: `3`
- selected_total_score: `0.838`
- global_blockers: `none`

### live_feedback_low_headroom_gain_pullback

- status: `passed`
- expected_selected_strategy_id: `feedback_gain_pullback`
- actual_selected_strategy_id: `feedback_gain_pullback`
- actual_method_family: `gain_fader`
- candidate_count: `4`
- selected_total_score: `0.838`
- global_blockers: `true_peak_headroom_low`

### soundcheck_feedback_blocks_presence_boost

- status: `passed`
- expected_selected_strategy_id: `feedback_gain_pullback`
- actual_selected_strategy_id: `feedback_gain_pullback`
- actual_method_family: `gain_fader`
- candidate_count: `6`
- selected_total_score: `0.79`
- global_blockers: `none`

### soundcheck_measured_feedback_ring_notch

- status: `passed`
- expected_selected_strategy_id: `narrow_feedback_notch_review`
- actual_selected_strategy_id: `narrow_feedback_notch_review`
- actual_method_family: `eq`
- candidate_count: `3`
- selected_total_score: `0.866`
- global_blockers: `none`

### soundcheck_measured_feedback_low_headroom_pullback_before_notch

- status: `passed`
- expected_selected_strategy_id: `feedback_gain_pullback`
- actual_selected_strategy_id: `feedback_gain_pullback`
- actual_method_family: `gain_fader`
- candidate_count: `4`
- selected_total_score: `0.79`
- global_blockers: `true_peak_headroom_low`

### soundcheck_harsh_sibilant_vocal_split_chain

- status: `passed`
- expected_selected_strategy_id: `vocal_de_esser`
- actual_selected_strategy_id: `vocal_de_esser`
- actual_method_family: `de_esser`
- candidate_count: `5`
- selected_total_score: `0.874`
- global_blockers: `none`

### soundcheck_sibilant_dynamic_vocal_order_review

- status: `passed`
- expected_selected_strategy_id: `vocal_de_esser_before_compression_recheck`
- actual_selected_strategy_id: `vocal_de_esser_before_compression_recheck`
- actual_method_family: `de_esser`
- candidate_count: `12`
- selected_total_score: `0.912`
- global_blockers: `none`

### live_sibilant_vocal

- status: `passed`
- expected_selected_strategy_id: `vocal_de_esser`
- actual_selected_strategy_id: `vocal_de_esser`
- actual_method_family: `de_esser`
- candidate_count: `4`
- selected_total_score: `0.878`
- global_blockers: `none`

### soundcheck_vocal_level_pressure_before_de_essing

- status: `passed`
- expected_selected_strategy_id: `level_plane_gain_down`
- actual_selected_strategy_id: `level_plane_gain_down`
- actual_method_family: `gain_fader`
- candidate_count: `7`
- selected_total_score: `0.868`
- global_blockers: `none`

### live_harsh_vocal_static_presence_cut

- status: `passed`
- expected_selected_strategy_id: `broad_presence_cut`
- actual_selected_strategy_id: `broad_presence_cut`
- actual_method_family: `eq`
- candidate_count: `4`
- selected_total_score: `0.802`
- global_blockers: `none`

### live_feedback_risk_beats_sibilance_correction

- status: `passed`
- expected_selected_strategy_id: `feedback_gain_pullback`
- actual_selected_strategy_id: `feedback_gain_pullback`
- actual_method_family: `gain_fader`
- candidate_count: `7`
- selected_total_score: `0.838`
- global_blockers: `none`

### soundcheck_harsh_dynamic_vocal_envelope

- status: `passed`
- expected_selected_strategy_id: `vocal_compressor_envelope_smoothing`
- actual_selected_strategy_id: `vocal_compressor_envelope_smoothing`
- actual_method_family: `dynamics`
- candidate_count: `9`
- selected_total_score: `0.849`
- global_blockers: `none`

### soundcheck_harsh_guitar_dynamic_presence_tame

- status: `passed`
- expected_selected_strategy_id: `instrument_dynamic_presence_tame`
- actual_selected_strategy_id: `instrument_dynamic_presence_tame`
- actual_method_family: `dynamic_eq`
- candidate_count: `3`
- selected_total_score: `0.843`
- global_blockers: `none`

### soundcheck_harsh_overhead_cymbal_dynamic_de_edge

- status: `passed`
- expected_selected_strategy_id: `drum_cymbal_static_air_cut`
- actual_selected_strategy_id: `drum_cymbal_static_air_cut`
- actual_method_family: `eq`
- candidate_count: `3`
- selected_total_score: `0.833`
- global_blockers: `none`

### soundcheck_harsh_overhead_top_octave_high_cut

- status: `passed`
- expected_selected_strategy_id: `drum_cymbal_high_cut_air_limit`
- actual_selected_strategy_id: `drum_cymbal_high_cut_air_limit`
- actual_method_family: `eq`
- candidate_count: `4`
- selected_total_score: `0.85`
- global_blockers: `none`

### live_harsh_overhead_cymbal_mic_level_review

- status: `passed`
- expected_selected_strategy_id: `drum_cymbal_mic_level_review`
- actual_selected_strategy_id: `drum_cymbal_mic_level_review`
- actual_method_family: `analysis_only`
- candidate_count: `3`
- selected_total_score: `0.851`
- global_blockers: `none`

### inferred_vocal_identity_requires_confirmation

- status: `passed`
- expected_selected_strategy_id: `hold_for_source_identity_confirmation`
- actual_selected_strategy_id: `hold_for_source_identity_confirmation`
- actual_method_family: `analysis_only`
- candidate_count: `5`
- selected_total_score: `0.888`
- global_blockers: `source_identity_not_confirmed, source_identity_confidence_low`

### low_confidence_source_identity_blocks_eq

- status: `passed`
- expected_selected_strategy_id: `hold_for_source_identity_confirmation`
- actual_selected_strategy_id: `hold_for_source_identity_confirmation`
- actual_method_family: `analysis_only`
- candidate_count: `3`
- selected_total_score: `0.888`
- global_blockers: `source_identity_confidence_low`

### unconfirmed_source_identity_blocks_eq_without_inference

- status: `passed`
- expected_selected_strategy_id: `hold_for_source_identity_confirmation`
- actual_selected_strategy_id: `hold_for_source_identity_confirmation`
- actual_method_family: `analysis_only`
- candidate_count: `3`
- selected_total_score: `0.888`
- global_blockers: `source_identity_not_confirmed, source_identity_confidence_low`

### low_confidence_audio_evidence_holds_before_eq

- status: `passed`
- expected_selected_strategy_id: `hold_for_audio_evidence_recheck`
- actual_selected_strategy_id: `hold_for_audio_evidence_recheck`
- actual_method_family: `analysis_only`
- candidate_count: `3`
- selected_total_score: `0.887`
- global_blockers: `audio_evidence_confidence_low`

### stale_live_audio_evidence_holds_before_eq

- status: `passed`
- expected_selected_strategy_id: `hold_for_audio_evidence_recheck`
- actual_selected_strategy_id: `hold_for_audio_evidence_recheck`
- actual_method_family: `analysis_only`
- candidate_count: `3`
- selected_total_score: `0.912`
- global_blockers: `audio_evidence_stale`

### auto_apply_request_holds_before_dsp

- status: `passed`
- expected_selected_strategy_id: `hold_for_operator_review_no_auto_apply`
- actual_selected_strategy_id: `hold_for_operator_review_no_auto_apply`
- actual_method_family: `analysis_only`
- candidate_count: `3`
- selected_total_score: `0.9`
- global_blockers: `auto_apply_not_supported`

### unsupported_mode_holds_before_dsp

- status: `passed`
- expected_selected_strategy_id: `hold_for_supported_mode_selection`
- actual_selected_strategy_id: `hold_for_supported_mode_selection`
- actual_method_family: `analysis_only`
- candidate_count: `3`
- selected_total_score: `0.882`
- global_blockers: `unsupported_smart_dsp_mode`

### routing_scene_snapshot_intent_holds_before_dsp

- status: `passed`
- expected_selected_strategy_id: `hold_for_routing_scene_snapshot_review`
- actual_selected_strategy_id: `hold_for_routing_scene_snapshot_review`
- actual_method_family: `analysis_only`
- candidate_count: `3`
- selected_total_score: `0.897`
- global_blockers: `forbidden_routing_scene_snapshot_intent`

### live_master_output_intent_holds_before_channel_eq

- status: `passed`
- expected_selected_strategy_id: `hold_for_live_master_output_intent_review`
- actual_selected_strategy_id: `hold_for_live_master_output_intent_review`
- actual_method_family: `analysis_only`
- candidate_count: `3`
- selected_total_score: `0.917`
- global_blockers: `live_master_output_intent_requires_review`

### live_eq_intent_holds_before_channel_eq_or_gain_up_guard

- status: `passed`
- expected_selected_strategy_id: `hold_for_live_eq_intent_review`
- actual_selected_strategy_id: `hold_for_live_eq_intent_review`
- actual_method_family: `analysis_only`
- candidate_count: `3`
- selected_total_score: `0.919`
- global_blockers: `live_eq_intent_requires_review`

### live_feedback_risk_beats_eq_intent_hold

- status: `passed`
- expected_selected_strategy_id: `feedback_gain_pullback`
- actual_selected_strategy_id: `feedback_gain_pullback`
- actual_method_family: `gain_fader`
- candidate_count: `6`
- selected_total_score: `0.838`
- global_blockers: `live_eq_intent_requires_review`

### live_source_gain_up_intent_holds_before_channel_eq

- status: `passed`
- expected_selected_strategy_id: `hold_for_live_source_gain_up_intent_review`
- actual_selected_strategy_id: `hold_for_live_source_gain_up_intent_review`
- actual_method_family: `analysis_only`
- candidate_count: `3`
- selected_total_score: `0.92`
- global_blockers: `live_source_gain_up_intent_requires_review`

### live_processing_change_intent_holds_before_channel_eq

- status: `passed`
- expected_selected_strategy_id: `hold_for_live_processing_intent_review`
- actual_selected_strategy_id: `hold_for_live_processing_intent_review`
- actual_method_family: `analysis_only`
- candidate_count: `3`
- selected_total_score: `0.911`
- global_blockers: `live_processing_intent_requires_review`

### live_pan_width_intent_holds_before_channel_eq

- status: `passed`
- expected_selected_strategy_id: `hold_for_live_pan_width_intent_review`
- actual_selected_strategy_id: `hold_for_live_pan_width_intent_review`
- actual_method_family: `analysis_only`
- candidate_count: `3`
- selected_total_score: `0.911`
- global_blockers: `live_pan_width_intent_requires_review`

### declared_vocal_audio_identity_conflict_holds

- status: `passed`
- expected_selected_strategy_id: `hold_for_source_identity_recheck`
- actual_selected_strategy_id: `hold_for_source_identity_recheck`
- actual_method_family: `analysis_only`
- candidate_count: `3`
- selected_total_score: `0.883`
- global_blockers: `source_identity_evidence_conflict`

### soundcheck_dynamic_vocal_compression

- status: `passed`
- expected_selected_strategy_id: `vocal_phrase_leveling_compression`
- actual_selected_strategy_id: `vocal_phrase_leveling_compression`
- actual_method_family: `dynamics`
- candidate_count: `4`
- selected_total_score: `0.879`
- global_blockers: `none`

### soundcheck_aux_playback_pse_la_dynamics

- status: `failed`
- expected_selected_strategy_id: `group_bus_compression_review`
- actual_selected_strategy_id: `group_bus_compression_review`
- actual_method_family: `dynamics`
- candidate_count: `2`
- selected_total_score: `0.756`
- global_blockers: `none`

### soundcheck_dynamic_vocal_dynamics_model_conflict_ride

- status: `passed`
- expected_selected_strategy_id: `manual_level_ride_before_compression`
- actual_selected_strategy_id: `manual_level_ride_before_compression`
- actual_method_family: `analysis_only`
- candidate_count: `4`
- selected_total_score: `0.769`
- global_blockers: `none`

### soundcheck_dynamic_vocal_active_dynamics_model_readback_ride

- status: `passed`
- expected_selected_strategy_id: `manual_level_ride_before_compression`
- actual_selected_strategy_id: `manual_level_ride_before_compression`
- actual_method_family: `analysis_only`
- candidate_count: `4`
- selected_total_score: `0.769`
- global_blockers: `none`

### soundcheck_dynamic_vocal_summed_bus_makeup_hold

- status: `passed`
- expected_selected_strategy_id: `hold_compressor_makeup_for_summed_bus_headroom`
- actual_selected_strategy_id: `hold_compressor_makeup_for_summed_bus_headroom`
- actual_method_family: `analysis_only`
- candidate_count: `5`
- selected_total_score: `0.884`
- global_blockers: `none`

### soundcheck_dynamic_vocal_feedback_path_hold

- status: `passed`
- expected_selected_strategy_id: `hold_dynamics_for_feedback_path_review`
- actual_selected_strategy_id: `hold_dynamics_for_feedback_path_review`
- actual_method_family: `analysis_only`
- candidate_count: `5`
- selected_total_score: `0.878`
- global_blockers: `none`

### soundcheck_master_true_peak_beats_source_vocal_compression

- status: `passed`
- expected_selected_strategy_id: `master_trim_for_true_peak_headroom`
- actual_selected_strategy_id: `master_trim_for_true_peak_headroom`
- actual_method_family: `gain_fader`
- candidate_count: `7`
- selected_total_score: `0.878`
- global_blockers: `none`

### soundcheck_master_true_peak_beats_source_headroom_pullback

- status: `passed`
- expected_selected_strategy_id: `master_trim_for_true_peak_headroom`
- actual_selected_strategy_id: `master_trim_for_true_peak_headroom`
- actual_method_family: `gain_fader`
- candidate_count: `8`
- selected_total_score: `0.878`
- global_blockers: `true_peak_headroom_low`

### live_vocal_dynamics_ride_before_compression

- status: `passed`
- expected_selected_strategy_id: `manual_level_ride_before_compression`
- actual_selected_strategy_id: `manual_level_ride_before_compression`
- actual_method_family: `analysis_only`
- candidate_count: `4`
- selected_total_score: `0.864`
- global_blockers: `none`

### soundcheck_dynamic_vocal_level_plane_before_compression

- status: `passed`
- expected_selected_strategy_id: `soundcheck_bounded_gain_up`
- actual_selected_strategy_id: `soundcheck_bounded_gain_up`
- actual_method_family: `gain_fader`
- candidate_count: `7`
- selected_total_score: `0.832`
- global_blockers: `none`

### soundcheck_bass_sustain_leveling_compression

- status: `passed`
- expected_selected_strategy_id: `bass_sustain_leveling_compression`
- actual_selected_strategy_id: `bass_sustain_leveling_compression`
- actual_method_family: `dynamics`
- candidate_count: `4`
- selected_total_score: `0.878`
- global_blockers: `none`

### soundcheck_bass_hot_bus_rebalance_before_compression

- status: `passed`
- expected_selected_strategy_id: `bass_sustain_summed_bus_rebalance_before_compression`
- actual_selected_strategy_id: `bass_sustain_summed_bus_rebalance_before_compression`
- actual_method_family: `gain_fader`
- candidate_count: `5`
- selected_total_score: `0.891`
- global_blockers: `none`

### live_guitar_pick_sustain_compression

- status: `passed`
- expected_selected_strategy_id: `guitar_pick_sustain_compression`
- actual_selected_strategy_id: `guitar_pick_sustain_compression`
- actual_method_family: `dynamics`
- candidate_count: `4`
- selected_total_score: `0.872`
- global_blockers: `none`

### live_lead_guitar_ride_before_compression

- status: `passed`
- expected_selected_strategy_id: `manual_level_ride_before_compression`
- actual_selected_strategy_id: `manual_level_ride_before_compression`
- actual_method_family: `analysis_only`
- candidate_count: `4`
- selected_total_score: `0.87`
- global_blockers: `none`

### live_keys_swell_ride_before_compression

- status: `passed`
- expected_selected_strategy_id: `manual_level_ride_before_compression`
- actual_selected_strategy_id: `manual_level_ride_before_compression`
- actual_method_family: `analysis_only`
- candidate_count: `4`
- selected_total_score: `0.831`
- global_blockers: `none`

### soundcheck_quiet_vocal_level_plane_gain_up

- status: `passed`
- expected_selected_strategy_id: `soundcheck_bounded_gain_up`
- actual_selected_strategy_id: `soundcheck_bounded_gain_up`
- actual_method_family: `gain_fader`
- candidate_count: `3`
- selected_total_score: `0.832`
- global_blockers: `none`

### soundcheck_quiet_vocal_feedback_path_holds_gain_up

- status: `passed`
- expected_selected_strategy_id: `hold_gain_up_for_feedback_path_review`
- actual_selected_strategy_id: `hold_gain_up_for_feedback_path_review`
- actual_method_family: `analysis_only`
- candidate_count: `4`
- selected_total_score: `0.729`
- global_blockers: `none`

### soundcheck_quiet_muddy_vocal_level_first

- status: `passed`
- expected_selected_strategy_id: `soundcheck_bounded_gain_up`
- actual_selected_strategy_id: `soundcheck_bounded_gain_up`
- actual_method_family: `gain_fader`
- candidate_count: `6`
- selected_total_score: `0.832`
- global_blockers: `none`

### soundcheck_hot_muddy_vocal_level_down_before_eq

- status: `passed`
- expected_selected_strategy_id: `level_plane_gain_down`
- actual_selected_strategy_id: `level_plane_gain_down`
- actual_method_family: `gain_fader`
- candidate_count: `6`
- selected_total_score: `0.868`
- global_blockers: `none`

### soundcheck_quiet_muddy_vocal_hot_bus_rebalance_before_eq

- status: `passed`
- expected_selected_strategy_id: `summed_bus_rebalance_before_source_gain`
- actual_selected_strategy_id: `summed_bus_rebalance_before_source_gain`
- actual_method_family: `gain_fader`
- candidate_count: `7`
- selected_total_score: `0.838`
- global_blockers: `none`

### soundcheck_quiet_vocal_hot_bus_rebalance

- status: `passed`
- expected_selected_strategy_id: `summed_bus_rebalance_before_source_gain`
- actual_selected_strategy_id: `summed_bus_rebalance_before_source_gain`
- actual_method_family: `gain_fader`
- candidate_count: `4`
- selected_total_score: `0.838`
- global_blockers: `none`

### live_loud_bass_level_plane_gain_down

- status: `passed`
- expected_selected_strategy_id: `level_plane_gain_down`
- actual_selected_strategy_id: `level_plane_gain_down`
- actual_method_family: `gain_fader`
- candidate_count: `3`
- selected_total_score: `0.882`
- global_blockers: `none`

### soundcheck_hot_drum_bus_group_trim

- status: `passed`
- expected_selected_strategy_id: `group_bus_trim_for_headroom`
- actual_selected_strategy_id: `group_bus_trim_for_headroom`
- actual_method_family: `gain_fader`
- candidate_count: `5`
- selected_total_score: `0.855`
- global_blockers: `none`

### soundcheck_dynamic_vocal_bus_group_compression

- status: `passed`
- expected_selected_strategy_id: `group_bus_compression_review`
- actual_selected_strategy_id: `group_bus_compression_review`
- actual_method_family: `dynamics`
- candidate_count: `2`
- selected_total_score: `0.756`
- global_blockers: `none`

### live_quiet_vocal_bus_group_gain_hold

- status: `passed`
- expected_selected_strategy_id: `hold_group_bus_for_member_evidence`
- actual_selected_strategy_id: `hold_group_bus_for_member_evidence`
- actual_method_family: `analysis_only`
- candidate_count: `3`
- selected_total_score: `0.813`
- global_blockers: `none`

### soundcheck_quiet_vocal_bus_feedback_member_review

- status: `passed`
- expected_selected_strategy_id: `hold_group_bus_feedback_member_review`
- actual_selected_strategy_id: `hold_group_bus_feedback_member_review`
- actual_method_family: `analysis_only`
- candidate_count: `4`
- selected_total_score: `0.889`
- global_blockers: `none`

### live_group_feedback_member_review_before_bus_dsp

- status: `passed`
- expected_selected_strategy_id: `hold_group_bus_feedback_member_review`
- actual_selected_strategy_id: `hold_group_bus_feedback_member_review`
- actual_method_family: `analysis_only`
- candidate_count: `6`
- selected_total_score: `0.927`
- global_blockers: `none`

### kick_bass_masking_low_headroom

- status: `passed`
- expected_selected_strategy_id: `source_true_peak_headroom_pullback`
- actual_selected_strategy_id: `source_true_peak_headroom_pullback`
- actual_method_family: `gain_fader`
- candidate_count: `8`
- selected_total_score: `0.85`
- global_blockers: `true_peak_headroom_low`

### live_source_true_peak_headroom_pullback_before_eq

- status: `passed`
- expected_selected_strategy_id: `source_true_peak_headroom_pullback`
- actual_selected_strategy_id: `source_true_peak_headroom_pullback`
- actual_method_family: `gain_fader`
- candidate_count: `3`
- selected_total_score: `0.884`
- global_blockers: `true_peak_headroom_low`

### soundcheck_general_masking_static_unmasking_eq

- status: `passed`
- expected_selected_strategy_id: `subtractive_unmasking_eq_split`
- actual_selected_strategy_id: `subtractive_unmasking_eq_split`
- actual_method_family: `eq`
- candidate_count: `2`
- selected_total_score: `0.829`
- global_blockers: `none`

### soundcheck_vocal_intelligibility_dynamic_unmasking

- status: `passed`
- expected_selected_strategy_id: `vocal_intelligibility_static_pocket_eq`
- actual_selected_strategy_id: `vocal_intelligibility_static_pocket_eq`
- actual_method_family: `eq`
- candidate_count: `3`
- selected_total_score: `0.867`
- global_blockers: `none`

### live_vocal_intelligibility_static_pocket

- status: `passed`
- expected_selected_strategy_id: `vocal_intelligibility_static_pocket_eq`
- actual_selected_strategy_id: `vocal_intelligibility_static_pocket_eq`
- actual_method_family: `eq`
- candidate_count: `3`
- selected_total_score: `0.894`
- global_blockers: `none`

### soundcheck_vocal_masking_over_level_rebalance

- status: `passed`
- expected_selected_strategy_id: `vocal_masking_level_plane_rebalance`
- actual_selected_strategy_id: `vocal_masking_level_plane_rebalance`
- actual_method_family: `gain_fader`
- candidate_count: `7`
- selected_total_score: `0.878`
- global_blockers: `none`

### live_vocal_masking_over_level_rebalance

- status: `passed`
- expected_selected_strategy_id: `vocal_masking_level_plane_rebalance`
- actual_selected_strategy_id: `vocal_masking_level_plane_rebalance`
- actual_method_family: `gain_fader`
- candidate_count: `7`
- selected_total_score: `0.892`
- global_blockers: `none`

### soundcheck_kick_bass_dynamic_pocket

- status: `passed`
- expected_selected_strategy_id: `kick_bass_pocket_eq_split`
- actual_selected_strategy_id: `kick_bass_pocket_eq_split`
- actual_method_family: `eq`
- candidate_count: `3`
- selected_total_score: `0.858`
- global_blockers: `none`

### soundcheck_kick_bass_hot_bus_rebalance_before_ducking

- status: `passed`
- expected_selected_strategy_id: `kick_bass_summed_bus_rebalance_before_ducking`
- actual_selected_strategy_id: `kick_bass_summed_bus_rebalance_before_ducking`
- actual_method_family: `gain_fader`
- candidate_count: `4`
- selected_total_score: `0.879`
- global_blockers: `none`

### soundcheck_kick_target_avoids_ducking_kick_from_bass

- status: `passed`
- expected_selected_strategy_id: `kick_bass_pocket_eq_split`
- actual_selected_strategy_id: `kick_bass_pocket_eq_split`
- actual_method_family: `eq`
- candidate_count: `3`
- selected_total_score: `0.858`
- global_blockers: `none`

### soundcheck_kick_bass_arrangement_overlap_hold

- status: `passed`
- expected_selected_strategy_id: `arrangement_aware_low_end_separation_hold`
- actual_selected_strategy_id: `arrangement_aware_low_end_separation_hold`
- actual_method_family: `analysis_only`
- candidate_count: `3`
- selected_total_score: `0.938`
- global_blockers: `none`

### live_kick_bass_conservative_eq_split

- status: `passed`
- expected_selected_strategy_id: `kick_bass_pocket_eq_split`
- actual_selected_strategy_id: `kick_bass_pocket_eq_split`
- actual_method_family: `eq`
- candidate_count: `3`
- selected_total_score: `0.878`
- global_blockers: `none`

### soundcheck_bass_low_end_dynamic_resonance

- status: `passed`
- expected_selected_strategy_id: `low_end_dynamic_resonance_control`
- actual_selected_strategy_id: `low_end_dynamic_resonance_control`
- actual_method_family: `dynamic_eq`
- candidate_count: `10`
- selected_total_score: `0.882`
- global_blockers: `none`

### live_bass_low_end_static_pocket_cut

- status: `passed`
- expected_selected_strategy_id: `low_end_static_pocket_cut`
- actual_selected_strategy_id: `low_end_static_pocket_cut`
- actual_method_family: `eq`
- candidate_count: `10`
- selected_total_score: `0.905`
- global_blockers: `none`

### soundcheck_snare_ring_dynamic_notch

- status: `passed`
- expected_selected_strategy_id: `drum_dynamic_ring_notch_control`
- actual_selected_strategy_id: `drum_dynamic_ring_notch_control`
- actual_method_family: `dynamic_eq`
- candidate_count: `6`
- selected_total_score: `0.885`
- global_blockers: `none`

### live_snare_ring_static_notch

- status: `passed`
- expected_selected_strategy_id: `drum_static_ring_notch_cut`
- actual_selected_strategy_id: `drum_static_ring_notch_cut`
- actual_method_family: `eq`
- candidate_count: `6`
- selected_total_score: `0.908`
- global_blockers: `none`

### soundcheck_snare_transient_tamer

- status: `passed`
- expected_selected_strategy_id: `drum_transient_tamer_compression`
- actual_selected_strategy_id: `drum_transient_tamer_compression`
- actual_method_family: `dynamics`
- candidate_count: `3`
- selected_total_score: `0.856`
- global_blockers: `none`

### live_snare_low_headroom_transient_hold

- status: `passed`
- expected_selected_strategy_id: `source_true_peak_headroom_pullback`
- actual_selected_strategy_id: `source_true_peak_headroom_pullback`
- actual_method_family: `gain_fader`
- candidate_count: `4`
- selected_total_score: `0.884`
- global_blockers: `true_peak_headroom_low`

### noisy_tom_tail_risk

- status: `passed`
- expected_selected_strategy_id: `gentle_expander_bleed_control`
- actual_selected_strategy_id: `gentle_expander_bleed_control`
- actual_method_family: `expander`
- candidate_count: `3`
- selected_total_score: `0.802`
- global_blockers: `none`

### measured_tom_tail_risk_expander

- status: `passed`
- expected_selected_strategy_id: `gentle_expander_bleed_control`
- actual_selected_strategy_id: `gentle_expander_bleed_control`
- actual_method_family: `expander`
- candidate_count: `3`
- selected_total_score: `0.802`
- global_blockers: `none`

### clean_tom_bleed_aware_gate

- status: `passed`
- expected_selected_strategy_id: `bleed_aware_clean_hit_gate`
- actual_selected_strategy_id: `bleed_aware_clean_hit_gate`
- actual_method_family: `gate`
- candidate_count: `3`
- selected_total_score: `0.88`
- global_blockers: `none`

### low_margin_tom_clean_hit_evidence_hold

- status: `passed`
- expected_selected_strategy_id: `hold_noisy_drum_for_clean_hit_bleed_evidence`
- actual_selected_strategy_id: `hold_noisy_drum_for_clean_hit_bleed_evidence`
- actual_method_family: `analysis_only`
- candidate_count: `4`
- selected_total_score: `0.655`
- global_blockers: `clean_hit_bleed_margin_required_before_drum_dynamics`

### extreme_noisy_tom_manual_mute_review

- status: `passed`
- expected_selected_strategy_id: `manual_tom_mute_between_hits`
- actual_selected_strategy_id: `manual_tom_mute_between_hits`
- actual_method_family: `mute`
- candidate_count: `5`
- selected_total_score: `0.846`
- global_blockers: `none`

### live_noisy_tom_hold_before_gate_or_expander

- status: `passed`
- expected_selected_strategy_id: `hold_live_noisy_drum_processing_for_operator`
- actual_selected_strategy_id: `hold_live_noisy_drum_processing_for_operator`
- actual_method_family: `analysis_only`
- candidate_count: `4`
- selected_total_score: `0.905`
- global_blockers: `none`

### live_extreme_noisy_tom_hold_before_manual_mute

- status: `passed`
- expected_selected_strategy_id: `hold_live_noisy_drum_processing_for_operator`
- actual_selected_strategy_id: `hold_live_noisy_drum_processing_for_operator`
- actual_method_family: `analysis_only`
- candidate_count: `6`
- selected_total_score: `0.905`
- global_blockers: `none`

### wet_fx_monitor_risk

- status: `passed`
- expected_selected_strategy_id: `fx_monitor_feedback_safety_pullback`
- actual_selected_strategy_id: `fx_monitor_feedback_safety_pullback`
- actual_method_family: `fx`
- candidate_count: `6`
- selected_total_score: `0.85`
- global_blockers: `none`

### soundcheck_long_fx_tail_decay_filter

- status: `passed`
- expected_selected_strategy_id: `reduce_fx_send_or_return`
- actual_selected_strategy_id: `reduce_fx_send_or_return`
- actual_method_family: `fx`
- candidate_count: `5`
- selected_total_score: `0.773`
- global_blockers: `none`

### soundcheck_delay_smear_fx_ducking_cleanup

- status: `passed`
- expected_selected_strategy_id: `reduce_fx_send_or_return`
- actual_selected_strategy_id: `reduce_fx_send_or_return`
- actual_method_family: `fx`
- candidate_count: `5`
- selected_total_score: `0.741`
- global_blockers: `none`

### soundcheck_fx_monitor_risk_pullback_before_time_edits

- status: `passed`
- expected_selected_strategy_id: `reduce_fx_send_or_return`
- actual_selected_strategy_id: `reduce_fx_send_or_return`
- actual_method_family: `fx`
- candidate_count: `5`
- selected_total_score: `0.78`
- global_blockers: `none`

### soundcheck_fx_return_headroom_only_pullback

- status: `passed`
- expected_selected_strategy_id: `fx_return_headroom_pullback_before_time_edits`
- actual_selected_strategy_id: `fx_return_headroom_pullback_before_time_edits`
- actual_method_family: `fx`
- candidate_count: `6`
- selected_total_score: `0.865`
- global_blockers: `none`

### soundcheck_source_fx_return_headroom_send_only_until_slot_bound

- status: `passed`
- expected_selected_strategy_id: `fx_return_headroom_pullback_before_time_edits`
- actual_selected_strategy_id: `fx_return_headroom_pullback_before_time_edits`
- actual_method_family: `fx`
- candidate_count: `6`
- selected_total_score: `0.865`
- global_blockers: `none`

### soundcheck_source_fx_return_headroom_bound_slot_uses_send_and_return

- status: `passed`
- expected_selected_strategy_id: `fx_return_headroom_pullback_before_time_edits`
- actual_selected_strategy_id: `fx_return_headroom_pullback_before_time_edits`
- actual_method_family: `fx`
- candidate_count: `6`
- selected_total_score: `0.865`
- global_blockers: `none`

### soundcheck_aux_playback_fx_send_pullback

- status: `passed`
- expected_selected_strategy_id: `reduce_fx_send_or_return`
- actual_selected_strategy_id: `reduce_fx_send_or_return`
- actual_method_family: `fx`
- candidate_count: `5`
- selected_total_score: `0.786`
- global_blockers: `none`

### soundcheck_fx_return_headroom_before_delay_ducking

- status: `passed`
- expected_selected_strategy_id: `fx_return_headroom_pullback_before_time_edits`
- actual_selected_strategy_id: `fx_return_headroom_pullback_before_time_edits`
- actual_method_family: `fx`
- candidate_count: `6`
- selected_total_score: `0.865`
- global_blockers: `none`

### live_delay_smear_fx_pullback_before_ducking

- status: `passed`
- expected_selected_strategy_id: `reduce_fx_send_or_return`
- actual_selected_strategy_id: `reduce_fx_send_or_return`
- actual_method_family: `fx`
- candidate_count: `5`
- selected_total_score: `0.781`
- global_blockers: `none`

### pan_phase_mono_failure

- status: `passed`
- expected_selected_strategy_id: `mono_compatibility_width_reduction`
- actual_selected_strategy_id: `mono_compatibility_width_reduction`
- actual_method_family: `pan`
- candidate_count: `6`
- selected_total_score: `0.831`
- global_blockers: `none`

### soundcheck_phase_safe_pan_delta

- status: `passed`
- expected_selected_strategy_id: `phase_safe_pan_target`
- actual_selected_strategy_id: `phase_safe_pan_target`
- actual_method_family: `pan`
- candidate_count: `3`
- selected_total_score: `0.799`
- global_blockers: `none`

### soundcheck_measured_stereo_balance_pan_target

- status: `passed`
- expected_selected_strategy_id: `phase_safe_pan_target`
- actual_selected_strategy_id: `phase_safe_pan_target`
- actual_method_family: `pan`
- candidate_count: `3`
- selected_total_score: `0.875`
- global_blockers: `none`

### live_measured_stereo_balance_pan_review_hold

- status: `passed`
- expected_selected_strategy_id: `hold_live_pan_change_for_operator_review`
- actual_selected_strategy_id: `hold_live_pan_change_for_operator_review`
- actual_method_family: `analysis_only`
- candidate_count: `3`
- selected_total_score: `0.918`
- global_blockers: `none`

### soundcheck_center_anchor_vocal_recenter

- status: `passed`
- expected_selected_strategy_id: `center_anchor_recenter_review`
- actual_selected_strategy_id: `center_anchor_recenter_review`
- actual_method_family: `pan`
- candidate_count: `4`
- selected_total_score: `0.924`
- global_blockers: `none`

### soundcheck_phase_mono_beats_measured_pan_target

- status: `passed`
- expected_selected_strategy_id: `mono_compatibility_width_reduction`
- actual_selected_strategy_id: `mono_compatibility_width_reduction`
- actual_method_family: `pan`
- candidate_count: `6`
- selected_total_score: `0.831`
- global_blockers: `none`

### live_pan_change_holds_for_operator_review

- status: `passed`
- expected_selected_strategy_id: `hold_live_pan_change_for_operator_review`
- actual_selected_strategy_id: `hold_live_pan_change_for_operator_review`
- actual_method_family: `analysis_only`
- candidate_count: `3`
- selected_total_score: `0.894`
- global_blockers: `none`

### soundcheck_phase_mono_risk_width_reduction

- status: `passed`
- expected_selected_strategy_id: `mono_compatibility_width_reduction`
- actual_selected_strategy_id: `mono_compatibility_width_reduction`
- actual_method_family: `pan`
- candidate_count: `3`
- selected_total_score: `0.831`
- global_blockers: `none`

### live_phase_mono_risk_hold

- status: `passed`
- expected_selected_strategy_id: `hold_phase_mono_processing_for_operator`
- actual_selected_strategy_id: `hold_phase_mono_processing_for_operator`
- actual_method_family: `analysis_only`
- candidate_count: `3`
- selected_total_score: `0.818`
- global_blockers: `none`

### soundcheck_phase_mono_risk_beats_harshness_eq

- status: `passed`
- expected_selected_strategy_id: `hold_phase_mono_processing_for_operator`
- actual_selected_strategy_id: `hold_phase_mono_processing_for_operator`
- actual_method_family: `analysis_only`
- candidate_count: `6`
- selected_total_score: `0.859`
- global_blockers: `none`

### live_phase_mono_risk_beats_harshness_eq

- status: `passed`
- expected_selected_strategy_id: `hold_phase_mono_processing_for_operator`
- actual_selected_strategy_id: `hold_phase_mono_processing_for_operator`
- actual_method_family: `analysis_only`
- candidate_count: `6`
- selected_total_score: `0.896`
- global_blockers: `none`

### master_true_peak_overload

- status: `passed`
- expected_selected_strategy_id: `master_trim_for_true_peak_headroom`
- actual_selected_strategy_id: `master_trim_for_true_peak_headroom`
- actual_method_family: `gain_fader`
- candidate_count: `3`
- selected_total_score: `0.878`
- global_blockers: `none`

### master_low_headroom_trim_before_tone_or_compression

- status: `passed`
- expected_selected_strategy_id: `master_trim_for_true_peak_headroom`
- actual_selected_strategy_id: `master_trim_for_true_peak_headroom`
- actual_method_family: `gain_fader`
- candidate_count: `5`
- selected_total_score: `0.878`
- global_blockers: `true_peak_headroom_low`

### master_reference_feed_holds_processing

- status: `passed`
- expected_selected_strategy_id: `hold_master_reference_feed_no_pa_write`
- actual_selected_strategy_id: `hold_master_reference_feed_no_pa_write`
- actual_method_family: `analysis_only`
- candidate_count: `1`
- selected_total_score: `0.619`
- global_blockers: `true_peak_headroom_low, master_reference_or_capture_feed_not_pa_output`

### master_low_end_broad_eq_cleanup

- status: `passed`
- expected_selected_strategy_id: `master_low_end_broad_eq_cleanup`
- actual_selected_strategy_id: `master_low_end_broad_eq_cleanup`
- actual_method_family: `eq`
- candidate_count: `1`
- selected_total_score: `0.831`
- global_blockers: `none`

### master_presence_broad_eq_cleanup

- status: `passed`
- expected_selected_strategy_id: `master_presence_broad_eq_cleanup`
- actual_selected_strategy_id: `master_presence_broad_eq_cleanup`
- actual_method_family: `eq`
- candidate_count: `1`
- selected_total_score: `0.838`
- global_blockers: `none`

### master_static_loudness_trim_before_bus_compression

- status: `passed`
- expected_selected_strategy_id: `master_level_trim_for_static_loudness`
- actual_selected_strategy_id: `master_level_trim_for_static_loudness`
- actual_method_family: `gain_fader`
- candidate_count: `2`
- selected_total_score: `0.869`
- global_blockers: `none`

### master_static_loudness_trim_before_low_end_recheck

- status: `passed`
- expected_selected_strategy_id: `master_level_trim_for_static_loudness`
- actual_selected_strategy_id: `master_level_trim_for_static_loudness`
- actual_method_family: `gain_fader`
- candidate_count: `3`
- selected_total_score: `0.869`
- global_blockers: `none`

### master_loud_dynamic_trim_before_bus_compression

- status: `passed`
- expected_selected_strategy_id: `master_loud_dynamic_trim_before_compression`
- actual_selected_strategy_id: `master_loud_dynamic_trim_before_compression`
- actual_method_family: `gain_fader`
- candidate_count: `2`
- selected_total_score: `0.884`
- global_blockers: `none`

### master_quiet_loudness_lift_before_hold_or_processing

- status: `passed`
- expected_selected_strategy_id: `master_output_lift_for_quiet_soundcheck`
- actual_selected_strategy_id: `master_output_lift_for_quiet_soundcheck`
- actual_method_family: `gain_fader`
- candidate_count: `2`
- selected_total_score: `0.856`
- global_blockers: `none`

### master_quiet_loudness_feedback_hold_before_lift

- status: `passed`
- expected_selected_strategy_id: `hold_master_feedback_or_monitor_risk_before_loudness_lift`
- actual_selected_strategy_id: `hold_master_feedback_or_monitor_risk_before_loudness_lift`
- actual_method_family: `analysis_only`
- candidate_count: `3`
- selected_total_score: `0.847`
- global_blockers: `none`

### live_master_presence_holds_before_master_eq

- status: `passed`
- expected_selected_strategy_id: `hold_live_master_output_processing_for_operator`
- actual_selected_strategy_id: `hold_live_master_output_processing_for_operator`
- actual_method_family: `analysis_only`
- candidate_count: `2`
- selected_total_score: `0.866`
- global_blockers: `none`

### master_dynamic_range_bus_compression

- status: `passed`
- expected_selected_strategy_id: `gentle_bus_compression`
- actual_selected_strategy_id: `gentle_bus_compression`
- actual_method_family: `dynamics`
- candidate_count: `1`
- selected_total_score: `0.836`
- global_blockers: `none`

### live_master_dynamics_holds_before_bus_compression

- status: `passed`
- expected_selected_strategy_id: `hold_live_master_output_processing_for_operator`
- actual_selected_strategy_id: `hold_live_master_output_processing_for_operator`
- actual_method_family: `analysis_only`
- candidate_count: `2`
- selected_total_score: `0.866`
- global_blockers: `none`

### stale_context_blocks_strategy

- status: `passed`
- expected_selected_strategy_id: `hold_for_context_readback_or_rollback_recheck`
- actual_selected_strategy_id: `hold_for_context_readback_or_rollback_recheck`
- actual_method_family: `analysis_only`
- candidate_count: `3`
- selected_total_score: `0.874`
- global_blockers: `smart_dsp_context_cache_not_fresh`

### live_authority_context_blocks_strategy

- status: `passed`
- expected_selected_strategy_id: `hold_for_context_readback_or_rollback_recheck`
- actual_selected_strategy_id: `hold_for_context_readback_or_rollback_recheck`
- actual_method_family: `analysis_only`
- candidate_count: `3`
- selected_total_score: `0.874`
- global_blockers: `smart_dsp_context_cache_live_authority_present`

### missing_readback_context_blocks_strategy

- status: `passed`
- expected_selected_strategy_id: `hold_for_context_readback_or_rollback_recheck`
- actual_selected_strategy_id: `hold_for_context_readback_or_rollback_recheck`
- actual_method_family: `analysis_only`
- candidate_count: `3`
- selected_total_score: `0.874`
- global_blockers: `current_state_readback_missing`

### missing_rollback_context_blocks_strategy

- status: `passed`
- expected_selected_strategy_id: `hold_for_context_readback_or_rollback_recheck`
- actual_selected_strategy_id: `hold_for_context_readback_or_rollback_recheck`
- actual_method_family: `analysis_only`
- candidate_count: `3`
- selected_total_score: `0.874`
- global_blockers: `rollback_plan_missing`

## Failed Checks

- `soundcheck_aux_playback_pse_la_dynamics_candidate_count_low` (soundcheck_aux_playback_pse_la_dynamics): 2 < 4
- `soundcheck_aux_playback_pse_la_dynamics_readback_contract_wing_threshold_readback_missing` (soundcheck_aux_playback_pse_la_dynamics): {'required': True, 'wing_osc_path_templates': ['/aux/{aux}/dyn/on', '/aux/{aux}/dyn/thr', '/aux/{aux}/dyn/depth', '/aux/{aux}/dyn/fast', '/aux/{aux}/dyn/peak', '/aux/{aux}/dyn/ingain', '/aux/{aux}/dyn/cpeak', '/aux/{aux}/dyn/cmode'], 'wing_rack_fw_3_0_5_native_osc_supported': True, 'dynamics_model_readback_required': False, 'dynamics_model_readback_must_equal': None, 'wing_compressor_model_readback_required': False, 'wing_compressor_model_readback_must_equal': None, 'wing_compressor_model_osc_path_suffix': None, 'wing_compressor_model_parameters_apply_only_when_model_matches': True, 'wing_aux_dynamics_pse_la_combo': True, 'wing_aux_dynamics_no_dyn_mdl_path': True, 'wing_aux_dynamics_no_ratio_attack_release_paths': True, 'wing_aux_dynamics_write_key_selection_policy': 'apply_only_selected_wing_fw_3_0_5_aux_pse_la_dynamics_value_keys', 'wing_aux_dynamics_apply_only_selected_write_keys': True, 'wing_aux_dynamics_selected_write_keys': ['wing_aux_dynamics_on_osc_value', 'wing_aux_dynamics_threshold_osc_value', 'wing_aux_dynamics_depth_osc_value', 'wing_aux_dynamics_fast_osc_value'], 'wing_aux_dynamics_forbidden_write_keys': ['wing_aux_dynamics_peak_osc_value'], 'wing_dynamics_required_model': None, 'wing_dynamics_current_model': None, 'wing_dynamics_current_enabled': None, 'wing_dynamics_current_operator_owned': None, 'wing_dynamics_model_readback_available': None, 'wing_dynamics_model_conflict': None, 'wing_dynamics_model_readback_required_for_active_slot': None, 'wing_dynamics_model_conflict_policy': None, 'wing_dynamics_model_conflict_blocker': None, 'wing_dynamics_model_readback_blocker': None, 'wing_dynamics_model_change_requires_operator_review': None, 'wing_dynamics_model_read_before_path': None, 'wing_dynamics_on_read_before_path': None, 'wing_dynamics_model_apply_only_when_current_model_matches': None, 'wing_dynamics_write_key_selection_policy': None, 'wing_dynamics_apply_only_selected_write_keys': None, 'wing_dynamics_selected_write_keys': None, 'wing_dynamics_forbidden_write_keys': None, 'read_before_paths': ['dyn/on', 'dyn/thr', 'dyn/depth', 'dyn/fast', 'dyn/peak', 'dyn/ingain', 'dyn/cpeak', 'dyn/cmode'], 'read_after_paths': ['dyn/on', 'dyn/thr', 'dyn/depth', 'dyn/fast', 'dyn/peak', 'dyn/ingain', 'dyn/cpeak', 'dyn/cmode'], 'makeup_gain_must_respect_headroom': True, 'soundcheck_dynamics_bounded_compression_proposal': True, 'soundcheck_dynamics_native_compressor_osc_only_required': True, 'soundcheck_dynamics_max_target_gain_reduction_db': 4.0, 'soundcheck_dynamics_max_makeup_gain_db': 1.0, 'soundcheck_dynamics_min_headroom_margin_db': 1.0, 'soundcheck_dynamics_required_true_peak_headroom_db': None, 'soundcheck_dynamics_required_summed_bus_true_peak_headroom_db': None, 'soundcheck_dynamics_target_gr_review_only': True, 'soundcheck_dynamics_target_gr_readback_required': True, 'soundcheck_dynamics_makeup_gain_headroom_required': True, 'soundcheck_dynamics_summed_bus_headroom_required': True, 'soundcheck_dynamics_compressor_model_readback_required': True, 'soundcheck_dynamics_gain_reduction_meter_readback_required': True, 'soundcheck_dynamics_feedback_recheck_required': True, 'soundcheck_dynamics_operator_confirmation_required': True, 'soundcheck_dynamics_readback_required': True, 'soundcheck_dynamics_rollback_ready': True, 'soundcheck_dynamics_live_promotion_status': 'blocked_until_real_wing_compressor_readback_gr_meter_headroom_feedback_recheck_and_operator_approval', 'soundcheck_dynamics_live_apply_blocked_without_wing_proof': True, 'soundcheck_master_processing_review_only_proposal': None, 'soundcheck_master_processing_native_osc_only_required': None, 'soundcheck_master_processing_target_surface': None, 'soundcheck_master_processing_types': None, 'soundcheck_master_processing_wing_apply_parameter_scope': None, 'soundcheck_master_processing_pa_feed_identity_required': None, 'soundcheck_master_processing_true_peak_headroom_recheck_required': None, 'soundcheck_master_processing_feedback_recheck_required': None, 'soundcheck_master_processing_bypass_level_match_required': None, 'soundcheck_master_processing_readback_required': None, 'soundcheck_master_processing_operator_confirmation_required': None, 'soundcheck_master_processing_rollback_ready': None, 'soundcheck_master_processing_no_routing_scene_snapshot_write_required': None, 'soundcheck_master_processing_bus_compression_max_target_gr_db': None, 'soundcheck_master_processing_limiter_max_target_gr_db': None, 'soundcheck_master_processing_live_promotion_status': None, 'soundcheck_master_processing_live_apply_blocked_without_wing_proof': None, 'live_dynamics_conservative_proposal': None, 'live_dynamics_stepwise_threshold_review_required': None, 'live_dynamics_max_threshold_step_db': None, 'live_dynamics_max_target_gain_reduction_db': None, 'live_dynamics_no_makeup_gain_required': None, 'live_dynamics_makeup_gain_db_limit': None, 'live_dynamics_gain_reduction_meter_readback_after_each_step_required': None, 'live_dynamics_previous_comp_model_readback_required': None, 'live_dynamics_operator_rehearsal_required': None, 'live_dynamics_live_promotion_status': None, 'live_dynamics_live_apply_blocked_without_wing_proof': None, 'review_only_parameter_keys': ['target_gain_reduction_db'], 'wing_dynamics_review_only_values': {'target_gain_reduction_db': 2.9}, 'wing_dynamics_review_only_parameter_reason_by_key': {'target_gain_reduction_db': 'target gain reduction is an operator/readback objective; WING Rack fw3.0.5 exposes compressor threshold/ratio/timing/knee/makeup/mix/envelope curve via OSC but no direct target gain-reduction write path'}, 'wing_dynamics_review_only_no_direct_osc_path': True, 'wing_dynamics_review_only_not_apply_ready': True, 'wing_dynamics_gain_reduction_objective_required': True, 'wing_dynamics_gain_reduction_objective_parameter_keys': ['target_gain_reduction_db'], 'wing_dynamics_gain_reduction_objective_values': {'target_gain_reduction_db': 2.9}, 'wing_dynamics_gain_reduction_target_db': 2.9, 'wing_dynamics_gain_reduction_max_db': None, 'wing_dynamics_gain_reduction_objective_tolerance_db': 1.0, 'wing_dynamics_gain_reduction_acceptance_min_db': 1.9, 'wing_dynamics_gain_reduction_acceptance_max_db': 3.9, 'wing_dynamics_gain_reduction_must_not_exceed_db': None, 'wing_dynamics_gain_reduction_objective_is_review_only': True, 'wing_dynamics_gain_reduction_objective_is_not_osc_write': True, 'wing_dynamics_gain_reduction_objective_no_direct_osc_write_path': True, 'wing_dynamics_gain_reduction_objective_apply_ready': False, 'wing_dynamics_gain_reduction_objective_requires_post_apply_meter_readback': True, 'wing_dynamics_gain_reduction_meter_readback_path_suffix': 'dyn/cpeak', 'wing_dynamics_gain_reduction_meter_readback_path_templates': ['/aux/{aux}/dyn/cpeak', '/aux/{aux}/dyn/cmode'], 'wing_dynamics_gain_reduction_meter_fallback_path_templates': [], 'wing_dynamics_gain_reduction_meter_value_type': 'read_only_aux_compressor_peak_percent_and_mode', 'wing_dynamics_gain_reduction_meter_is_not_apply_value': True, 'wing_dynamics_gain_reduction_meter_path_source': 'Docs/manuals/wing/WING Remote Protocols v3.0.5.pdf', 'wing_dynamics_gain_reduction_aux_depth_proxy': True, 'wing_dynamics_gain_reduction_objective_requires_operator_confirmation': True, 'wing_dynamics_gain_reduction_objective_live_promotion_status': 'blocked_until_actual_gain_reduction_meter_readback_and_operator_review', 'wing_dynamics_gain_reduction_objective_operator_check': 'confirm WING aux PSE/LA cpeak activity and cmode after the bounded depth/threshold proposal; aux exposes no dyn/gr meter', 'wing_dynamics_gain_reduction_objective_direct_osc_write_keys_excluded': ['target_gain_reduction_db'], 'wing_aux_dynamics_on_mapping_required': True, 'wing_aux_dynamics_on_osc_path_suffix': 'dyn/on', 'wing_aux_dynamics_on_osc_value': 1, 'wing_aux_dynamics_on_value_type': 'integer_bool', 'wing_aux_dynamics_on_readback_must_match_osc_value': True, 'wing_aux_dynamics_threshold_mapping_required': True, 'wing_aux_dynamics_threshold_osc_path_suffix': 'dyn/thr', 'wing_aux_dynamics_threshold_osc_value': -20.3, 'wing_aux_dynamics_threshold_value_type': 'float', 'wing_aux_dynamics_threshold_readback_must_match_osc_value': True, 'wing_aux_dynamics_depth_mapping_required': True, 'wing_aux_dynamics_depth_osc_path_suffix': 'dyn/depth', 'wing_aux_dynamics_depth_osc_value': 2.9, 'wing_aux_dynamics_depth_value_type': 'float', 'wing_aux_dynamics_depth_readback_must_match_osc_value': True, 'wing_aux_dynamics_fast_mapping_required': True, 'wing_aux_dynamics_fast_osc_path_suffix': 'dyn/fast', 'wing_aux_dynamics_fast_osc_value': 0, 'wing_aux_dynamics_fast_value_type': 'integer_bool', 'wing_aux_dynamics_fast_readback_must_match_osc_value': True}
- `soundcheck_aux_playback_pse_la_dynamics_selected_dynamics_wing_attack_mapping_missing` (soundcheck_aux_playback_pse_la_dynamics): ['wing_compressor_attack_max', 'wing_compressor_attack_min', 'wing_compressor_attack_osc_path_suffix', 'wing_compressor_attack_osc_value', 'wing_compressor_attack_quantization_error', 'wing_compressor_attack_quantized_from', 'wing_compressor_attack_source', 'wing_compressor_attack_value_type']
- `soundcheck_aux_playback_pse_la_dynamics_readback_contract_wing_attack_mapping_not_required` (soundcheck_aux_playback_pse_la_dynamics): {'required': True, 'wing_osc_path_templates': ['/aux/{aux}/dyn/on', '/aux/{aux}/dyn/thr', '/aux/{aux}/dyn/depth', '/aux/{aux}/dyn/fast', '/aux/{aux}/dyn/peak', '/aux/{aux}/dyn/ingain', '/aux/{aux}/dyn/cpeak', '/aux/{aux}/dyn/cmode'], 'wing_rack_fw_3_0_5_native_osc_supported': True, 'dynamics_model_readback_required': False, 'dynamics_model_readback_must_equal': None, 'wing_compressor_model_readback_required': False, 'wing_compressor_model_readback_must_equal': None, 'wing_compressor_model_osc_path_suffix': None, 'wing_compressor_model_parameters_apply_only_when_model_matches': True, 'wing_aux_dynamics_pse_la_combo': True, 'wing_aux_dynamics_no_dyn_mdl_path': True, 'wing_aux_dynamics_no_ratio_attack_release_paths': True, 'wing_aux_dynamics_write_key_selection_policy': 'apply_only_selected_wing_fw_3_0_5_aux_pse_la_dynamics_value_keys', 'wing_aux_dynamics_apply_only_selected_write_keys': True, 'wing_aux_dynamics_selected_write_keys': ['wing_aux_dynamics_on_osc_value', 'wing_aux_dynamics_threshold_osc_value', 'wing_aux_dynamics_depth_osc_value', 'wing_aux_dynamics_fast_osc_value'], 'wing_aux_dynamics_forbidden_write_keys': ['wing_aux_dynamics_peak_osc_value'], 'wing_dynamics_required_model': None, 'wing_dynamics_current_model': None, 'wing_dynamics_current_enabled': None, 'wing_dynamics_current_operator_owned': None, 'wing_dynamics_model_readback_available': None, 'wing_dynamics_model_conflict': None, 'wing_dynamics_model_readback_required_for_active_slot': None, 'wing_dynamics_model_conflict_policy': None, 'wing_dynamics_model_conflict_blocker': None, 'wing_dynamics_model_readback_blocker': None, 'wing_dynamics_model_change_requires_operator_review': None, 'wing_dynamics_model_read_before_path': None, 'wing_dynamics_on_read_before_path': None, 'wing_dynamics_model_apply_only_when_current_model_matches': None, 'wing_dynamics_write_key_selection_policy': None, 'wing_dynamics_apply_only_selected_write_keys': None, 'wing_dynamics_selected_write_keys': None, 'wing_dynamics_forbidden_write_keys': None, 'read_before_paths': ['dyn/on', 'dyn/thr', 'dyn/depth', 'dyn/fast', 'dyn/peak', 'dyn/ingain', 'dyn/cpeak', 'dyn/cmode'], 'read_after_paths': ['dyn/on', 'dyn/thr', 'dyn/depth', 'dyn/fast', 'dyn/peak', 'dyn/ingain', 'dyn/cpeak', 'dyn/cmode'], 'makeup_gain_must_respect_headroom': True, 'soundcheck_dynamics_bounded_compression_proposal': True, 'soundcheck_dynamics_native_compressor_osc_only_required': True, 'soundcheck_dynamics_max_target_gain_reduction_db': 4.0, 'soundcheck_dynamics_max_makeup_gain_db': 1.0, 'soundcheck_dynamics_min_headroom_margin_db': 1.0, 'soundcheck_dynamics_required_true_peak_headroom_db': None, 'soundcheck_dynamics_required_summed_bus_true_peak_headroom_db': None, 'soundcheck_dynamics_target_gr_review_only': True, 'soundcheck_dynamics_target_gr_readback_required': True, 'soundcheck_dynamics_makeup_gain_headroom_required': True, 'soundcheck_dynamics_summed_bus_headroom_required': True, 'soundcheck_dynamics_compressor_model_readback_required': True, 'soundcheck_dynamics_gain_reduction_meter_readback_required': True, 'soundcheck_dynamics_feedback_recheck_required': True, 'soundcheck_dynamics_operator_confirmation_required': True, 'soundcheck_dynamics_readback_required': True, 'soundcheck_dynamics_rollback_ready': True, 'soundcheck_dynamics_live_promotion_status': 'blocked_until_real_wing_compressor_readback_gr_meter_headroom_feedback_recheck_and_operator_approval', 'soundcheck_dynamics_live_apply_blocked_without_wing_proof': True, 'soundcheck_master_processing_review_only_proposal': None, 'soundcheck_master_processing_native_osc_only_required': None, 'soundcheck_master_processing_target_surface': None, 'soundcheck_master_processing_types': None, 'soundcheck_master_processing_wing_apply_parameter_scope': None, 'soundcheck_master_processing_pa_feed_identity_required': None, 'soundcheck_master_processing_true_peak_headroom_recheck_required': None, 'soundcheck_master_processing_feedback_recheck_required': None, 'soundcheck_master_processing_bypass_level_match_required': None, 'soundcheck_master_processing_readback_required': None, 'soundcheck_master_processing_operator_confirmation_required': None, 'soundcheck_master_processing_rollback_ready': None, 'soundcheck_master_processing_no_routing_scene_snapshot_write_required': None, 'soundcheck_master_processing_bus_compression_max_target_gr_db': None, 'soundcheck_master_processing_limiter_max_target_gr_db': None, 'soundcheck_master_processing_live_promotion_status': None, 'soundcheck_master_processing_live_apply_blocked_without_wing_proof': None, 'live_dynamics_conservative_proposal': None, 'live_dynamics_stepwise_threshold_review_required': None, 'live_dynamics_max_threshold_step_db': None, 'live_dynamics_max_target_gain_reduction_db': None, 'live_dynamics_no_makeup_gain_required': None, 'live_dynamics_makeup_gain_db_limit': None, 'live_dynamics_gain_reduction_meter_readback_after_each_step_required': None, 'live_dynamics_previous_comp_model_readback_required': None, 'live_dynamics_operator_rehearsal_required': None, 'live_dynamics_live_promotion_status': None, 'live_dynamics_live_apply_blocked_without_wing_proof': None, 'review_only_parameter_keys': ['target_gain_reduction_db'], 'wing_dynamics_review_only_values': {'target_gain_reduction_db': 2.9}, 'wing_dynamics_review_only_parameter_reason_by_key': {'target_gain_reduction_db': 'target gain reduction is an operator/readback objective; WING Rack fw3.0.5 exposes compressor threshold/ratio/timing/knee/makeup/mix/envelope curve via OSC but no direct target gain-reduction write path'}, 'wing_dynamics_review_only_no_direct_osc_path': True, 'wing_dynamics_review_only_not_apply_ready': True, 'wing_dynamics_gain_reduction_objective_required': True, 'wing_dynamics_gain_reduction_objective_parameter_keys': ['target_gain_reduction_db'], 'wing_dynamics_gain_reduction_objective_values': {'target_gain_reduction_db': 2.9}, 'wing_dynamics_gain_reduction_target_db': 2.9, 'wing_dynamics_gain_reduction_max_db': None, 'wing_dynamics_gain_reduction_objective_tolerance_db': 1.0, 'wing_dynamics_gain_reduction_acceptance_min_db': 1.9, 'wing_dynamics_gain_reduction_acceptance_max_db': 3.9, 'wing_dynamics_gain_reduction_must_not_exceed_db': None, 'wing_dynamics_gain_reduction_objective_is_review_only': True, 'wing_dynamics_gain_reduction_objective_is_not_osc_write': True, 'wing_dynamics_gain_reduction_objective_no_direct_osc_write_path': True, 'wing_dynamics_gain_reduction_objective_apply_ready': False, 'wing_dynamics_gain_reduction_objective_requires_post_apply_meter_readback': True, 'wing_dynamics_gain_reduction_meter_readback_path_suffix': 'dyn/cpeak', 'wing_dynamics_gain_reduction_meter_readback_path_templates': ['/aux/{aux}/dyn/cpeak', '/aux/{aux}/dyn/cmode'], 'wing_dynamics_gain_reduction_meter_fallback_path_templates': [], 'wing_dynamics_gain_reduction_meter_value_type': 'read_only_aux_compressor_peak_percent_and_mode', 'wing_dynamics_gain_reduction_meter_is_not_apply_value': True, 'wing_dynamics_gain_reduction_meter_path_source': 'Docs/manuals/wing/WING Remote Protocols v3.0.5.pdf', 'wing_dynamics_gain_reduction_aux_depth_proxy': True, 'wing_dynamics_gain_reduction_objective_requires_operator_confirmation': True, 'wing_dynamics_gain_reduction_objective_live_promotion_status': 'blocked_until_actual_gain_reduction_meter_readback_and_operator_review', 'wing_dynamics_gain_reduction_objective_operator_check': 'confirm WING aux PSE/LA cpeak activity and cmode after the bounded depth/threshold proposal; aux exposes no dyn/gr meter', 'wing_dynamics_gain_reduction_objective_direct_osc_write_keys_excluded': ['target_gain_reduction_db'], 'wing_aux_dynamics_on_mapping_required': True, 'wing_aux_dynamics_on_osc_path_suffix': 'dyn/on', 'wing_aux_dynamics_on_osc_value': 1, 'wing_aux_dynamics_on_value_type': 'integer_bool', 'wing_aux_dynamics_on_readback_must_match_osc_value': True, 'wing_aux_dynamics_threshold_mapping_required': True, 'wing_aux_dynamics_threshold_osc_path_suffix': 'dyn/thr', 'wing_aux_dynamics_threshold_osc_value': -20.3, 'wing_aux_dynamics_threshold_value_type': 'float', 'wing_aux_dynamics_threshold_readback_must_match_osc_value': True, 'wing_aux_dynamics_depth_mapping_required': True, 'wing_aux_dynamics_depth_osc_path_suffix': 'dyn/depth', 'wing_aux_dynamics_depth_osc_value': 2.9, 'wing_aux_dynamics_depth_value_type': 'float', 'wing_aux_dynamics_depth_readback_must_match_osc_value': True, 'wing_aux_dynamics_fast_mapping_required': True, 'wing_aux_dynamics_fast_osc_path_suffix': 'dyn/fast', 'wing_aux_dynamics_fast_osc_value': 0, 'wing_aux_dynamics_fast_value_type': 'integer_bool', 'wing_aux_dynamics_fast_readback_must_match_osc_value': True}
- `soundcheck_aux_playback_pse_la_dynamics_selected_dynamics_wing_attack_path_mismatch` (soundcheck_aux_playback_pse_la_dynamics): None
- `soundcheck_aux_playback_pse_la_dynamics_readback_contract_wing_attack_path_mismatch` (soundcheck_aux_playback_pse_la_dynamics): {'required': True, 'wing_osc_path_templates': ['/aux/{aux}/dyn/on', '/aux/{aux}/dyn/thr', '/aux/{aux}/dyn/depth', '/aux/{aux}/dyn/fast', '/aux/{aux}/dyn/peak', '/aux/{aux}/dyn/ingain', '/aux/{aux}/dyn/cpeak', '/aux/{aux}/dyn/cmode'], 'wing_rack_fw_3_0_5_native_osc_supported': True, 'dynamics_model_readback_required': False, 'dynamics_model_readback_must_equal': None, 'wing_compressor_model_readback_required': False, 'wing_compressor_model_readback_must_equal': None, 'wing_compressor_model_osc_path_suffix': None, 'wing_compressor_model_parameters_apply_only_when_model_matches': True, 'wing_aux_dynamics_pse_la_combo': True, 'wing_aux_dynamics_no_dyn_mdl_path': True, 'wing_aux_dynamics_no_ratio_attack_release_paths': True, 'wing_aux_dynamics_write_key_selection_policy': 'apply_only_selected_wing_fw_3_0_5_aux_pse_la_dynamics_value_keys', 'wing_aux_dynamics_apply_only_selected_write_keys': True, 'wing_aux_dynamics_selected_write_keys': ['wing_aux_dynamics_on_osc_value', 'wing_aux_dynamics_threshold_osc_value', 'wing_aux_dynamics_depth_osc_value', 'wing_aux_dynamics_fast_osc_value'], 'wing_aux_dynamics_forbidden_write_keys': ['wing_aux_dynamics_peak_osc_value'], 'wing_dynamics_required_model': None, 'wing_dynamics_current_model': None, 'wing_dynamics_current_enabled': None, 'wing_dynamics_current_operator_owned': None, 'wing_dynamics_model_readback_available': None, 'wing_dynamics_model_conflict': None, 'wing_dynamics_model_readback_required_for_active_slot': None, 'wing_dynamics_model_conflict_policy': None, 'wing_dynamics_model_conflict_blocker': None, 'wing_dynamics_model_readback_blocker': None, 'wing_dynamics_model_change_requires_operator_review': None, 'wing_dynamics_model_read_before_path': None, 'wing_dynamics_on_read_before_path': None, 'wing_dynamics_model_apply_only_when_current_model_matches': None, 'wing_dynamics_write_key_selection_policy': None, 'wing_dynamics_apply_only_selected_write_keys': None, 'wing_dynamics_selected_write_keys': None, 'wing_dynamics_forbidden_write_keys': None, 'read_before_paths': ['dyn/on', 'dyn/thr', 'dyn/depth', 'dyn/fast', 'dyn/peak', 'dyn/ingain', 'dyn/cpeak', 'dyn/cmode'], 'read_after_paths': ['dyn/on', 'dyn/thr', 'dyn/depth', 'dyn/fast', 'dyn/peak', 'dyn/ingain', 'dyn/cpeak', 'dyn/cmode'], 'makeup_gain_must_respect_headroom': True, 'soundcheck_dynamics_bounded_compression_proposal': True, 'soundcheck_dynamics_native_compressor_osc_only_required': True, 'soundcheck_dynamics_max_target_gain_reduction_db': 4.0, 'soundcheck_dynamics_max_makeup_gain_db': 1.0, 'soundcheck_dynamics_min_headroom_margin_db': 1.0, 'soundcheck_dynamics_required_true_peak_headroom_db': None, 'soundcheck_dynamics_required_summed_bus_true_peak_headroom_db': None, 'soundcheck_dynamics_target_gr_review_only': True, 'soundcheck_dynamics_target_gr_readback_required': True, 'soundcheck_dynamics_makeup_gain_headroom_required': True, 'soundcheck_dynamics_summed_bus_headroom_required': True, 'soundcheck_dynamics_compressor_model_readback_required': True, 'soundcheck_dynamics_gain_reduction_meter_readback_required': True, 'soundcheck_dynamics_feedback_recheck_required': True, 'soundcheck_dynamics_operator_confirmation_required': True, 'soundcheck_dynamics_readback_required': True, 'soundcheck_dynamics_rollback_ready': True, 'soundcheck_dynamics_live_promotion_status': 'blocked_until_real_wing_compressor_readback_gr_meter_headroom_feedback_recheck_and_operator_approval', 'soundcheck_dynamics_live_apply_blocked_without_wing_proof': True, 'soundcheck_master_processing_review_only_proposal': None, 'soundcheck_master_processing_native_osc_only_required': None, 'soundcheck_master_processing_target_surface': None, 'soundcheck_master_processing_types': None, 'soundcheck_master_processing_wing_apply_parameter_scope': None, 'soundcheck_master_processing_pa_feed_identity_required': None, 'soundcheck_master_processing_true_peak_headroom_recheck_required': None, 'soundcheck_master_processing_feedback_recheck_required': None, 'soundcheck_master_processing_bypass_level_match_required': None, 'soundcheck_master_processing_readback_required': None, 'soundcheck_master_processing_operator_confirmation_required': None, 'soundcheck_master_processing_rollback_ready': None, 'soundcheck_master_processing_no_routing_scene_snapshot_write_required': None, 'soundcheck_master_processing_bus_compression_max_target_gr_db': None, 'soundcheck_master_processing_limiter_max_target_gr_db': None, 'soundcheck_master_processing_live_promotion_status': None, 'soundcheck_master_processing_live_apply_blocked_without_wing_proof': None, 'live_dynamics_conservative_proposal': None, 'live_dynamics_stepwise_threshold_review_required': None, 'live_dynamics_max_threshold_step_db': None, 'live_dynamics_max_target_gain_reduction_db': None, 'live_dynamics_no_makeup_gain_required': None, 'live_dynamics_makeup_gain_db_limit': None, 'live_dynamics_gain_reduction_meter_readback_after_each_step_required': None, 'live_dynamics_previous_comp_model_readback_required': None, 'live_dynamics_operator_rehearsal_required': None, 'live_dynamics_live_promotion_status': None, 'live_dynamics_live_apply_blocked_without_wing_proof': None, 'review_only_parameter_keys': ['target_gain_reduction_db'], 'wing_dynamics_review_only_values': {'target_gain_reduction_db': 2.9}, 'wing_dynamics_review_only_parameter_reason_by_key': {'target_gain_reduction_db': 'target gain reduction is an operator/readback objective; WING Rack fw3.0.5 exposes compressor threshold/ratio/timing/knee/makeup/mix/envelope curve via OSC but no direct target gain-reduction write path'}, 'wing_dynamics_review_only_no_direct_osc_path': True, 'wing_dynamics_review_only_not_apply_ready': True, 'wing_dynamics_gain_reduction_objective_required': True, 'wing_dynamics_gain_reduction_objective_parameter_keys': ['target_gain_reduction_db'], 'wing_dynamics_gain_reduction_objective_values': {'target_gain_reduction_db': 2.9}, 'wing_dynamics_gain_reduction_target_db': 2.9, 'wing_dynamics_gain_reduction_max_db': None, 'wing_dynamics_gain_reduction_objective_tolerance_db': 1.0, 'wing_dynamics_gain_reduction_acceptance_min_db': 1.9, 'wing_dynamics_gain_reduction_acceptance_max_db': 3.9, 'wing_dynamics_gain_reduction_must_not_exceed_db': None, 'wing_dynamics_gain_reduction_objective_is_review_only': True, 'wing_dynamics_gain_reduction_objective_is_not_osc_write': True, 'wing_dynamics_gain_reduction_objective_no_direct_osc_write_path': True, 'wing_dynamics_gain_reduction_objective_apply_ready': False, 'wing_dynamics_gain_reduction_objective_requires_post_apply_meter_readback': True, 'wing_dynamics_gain_reduction_meter_readback_path_suffix': 'dyn/cpeak', 'wing_dynamics_gain_reduction_meter_readback_path_templates': ['/aux/{aux}/dyn/cpeak', '/aux/{aux}/dyn/cmode'], 'wing_dynamics_gain_reduction_meter_fallback_path_templates': [], 'wing_dynamics_gain_reduction_meter_value_type': 'read_only_aux_compressor_peak_percent_and_mode', 'wing_dynamics_gain_reduction_meter_is_not_apply_value': True, 'wing_dynamics_gain_reduction_meter_path_source': 'Docs/manuals/wing/WING Remote Protocols v3.0.5.pdf', 'wing_dynamics_gain_reduction_aux_depth_proxy': True, 'wing_dynamics_gain_reduction_objective_requires_operator_confirmation': True, 'wing_dynamics_gain_reduction_objective_live_promotion_status': 'blocked_until_actual_gain_reduction_meter_readback_and_operator_review', 'wing_dynamics_gain_reduction_objective_operator_check': 'confirm WING aux PSE/LA cpeak activity and cmode after the bounded depth/threshold proposal; aux exposes no dyn/gr meter', 'wing_dynamics_gain_reduction_objective_direct_osc_write_keys_excluded': ['target_gain_reduction_db'], 'wing_aux_dynamics_on_mapping_required': True, 'wing_aux_dynamics_on_osc_path_suffix': 'dyn/on', 'wing_aux_dynamics_on_osc_value': 1, 'wing_aux_dynamics_on_value_type': 'integer_bool', 'wing_aux_dynamics_on_readback_must_match_osc_value': True, 'wing_aux_dynamics_threshold_mapping_required': True, 'wing_aux_dynamics_threshold_osc_path_suffix': 'dyn/thr', 'wing_aux_dynamics_threshold_osc_value': -20.3, 'wing_aux_dynamics_threshold_value_type': 'float', 'wing_aux_dynamics_threshold_readback_must_match_osc_value': True, 'wing_aux_dynamics_depth_mapping_required': True, 'wing_aux_dynamics_depth_osc_path_suffix': 'dyn/depth', 'wing_aux_dynamics_depth_osc_value': 2.9, 'wing_aux_dynamics_depth_value_type': 'float', 'wing_aux_dynamics_depth_readback_must_match_osc_value': True, 'wing_aux_dynamics_fast_mapping_required': True, 'wing_aux_dynamics_fast_osc_path_suffix': 'dyn/fast', 'wing_aux_dynamics_fast_osc_value': 0, 'wing_aux_dynamics_fast_value_type': 'integer_bool', 'wing_aux_dynamics_fast_readback_must_match_osc_value': True}
- `soundcheck_aux_playback_pse_la_dynamics_readback_contract_wing_attack_native_path_missing` (soundcheck_aux_playback_pse_la_dynamics): []
- `soundcheck_aux_playback_pse_la_dynamics_readback_contract_wing_attack_contract_path_missing` (soundcheck_aux_playback_pse_la_dynamics): []
- `soundcheck_aux_playback_pse_la_dynamics_readback_contract_wing_attack_readback_missing` (soundcheck_aux_playback_pse_la_dynamics): {'required': True, 'wing_osc_path_templates': ['/aux/{aux}/dyn/on', '/aux/{aux}/dyn/thr', '/aux/{aux}/dyn/depth', '/aux/{aux}/dyn/fast', '/aux/{aux}/dyn/peak', '/aux/{aux}/dyn/ingain', '/aux/{aux}/dyn/cpeak', '/aux/{aux}/dyn/cmode'], 'wing_rack_fw_3_0_5_native_osc_supported': True, 'dynamics_model_readback_required': False, 'dynamics_model_readback_must_equal': None, 'wing_compressor_model_readback_required': False, 'wing_compressor_model_readback_must_equal': None, 'wing_compressor_model_osc_path_suffix': None, 'wing_compressor_model_parameters_apply_only_when_model_matches': True, 'wing_aux_dynamics_pse_la_combo': True, 'wing_aux_dynamics_no_dyn_mdl_path': True, 'wing_aux_dynamics_no_ratio_attack_release_paths': True, 'wing_aux_dynamics_write_key_selection_policy': 'apply_only_selected_wing_fw_3_0_5_aux_pse_la_dynamics_value_keys', 'wing_aux_dynamics_apply_only_selected_write_keys': True, 'wing_aux_dynamics_selected_write_keys': ['wing_aux_dynamics_on_osc_value', 'wing_aux_dynamics_threshold_osc_value', 'wing_aux_dynamics_depth_osc_value', 'wing_aux_dynamics_fast_osc_value'], 'wing_aux_dynamics_forbidden_write_keys': ['wing_aux_dynamics_peak_osc_value'], 'wing_dynamics_required_model': None, 'wing_dynamics_current_model': None, 'wing_dynamics_current_enabled': None, 'wing_dynamics_current_operator_owned': None, 'wing_dynamics_model_readback_available': None, 'wing_dynamics_model_conflict': None, 'wing_dynamics_model_readback_required_for_active_slot': None, 'wing_dynamics_model_conflict_policy': None, 'wing_dynamics_model_conflict_blocker': None, 'wing_dynamics_model_readback_blocker': None, 'wing_dynamics_model_change_requires_operator_review': None, 'wing_dynamics_model_read_before_path': None, 'wing_dynamics_on_read_before_path': None, 'wing_dynamics_model_apply_only_when_current_model_matches': None, 'wing_dynamics_write_key_selection_policy': None, 'wing_dynamics_apply_only_selected_write_keys': None, 'wing_dynamics_selected_write_keys': None, 'wing_dynamics_forbidden_write_keys': None, 'read_before_paths': ['dyn/on', 'dyn/thr', 'dyn/depth', 'dyn/fast', 'dyn/peak', 'dyn/ingain', 'dyn/cpeak', 'dyn/cmode'], 'read_after_paths': ['dyn/on', 'dyn/thr', 'dyn/depth', 'dyn/fast', 'dyn/peak', 'dyn/ingain', 'dyn/cpeak', 'dyn/cmode'], 'makeup_gain_must_respect_headroom': True, 'soundcheck_dynamics_bounded_compression_proposal': True, 'soundcheck_dynamics_native_compressor_osc_only_required': True, 'soundcheck_dynamics_max_target_gain_reduction_db': 4.0, 'soundcheck_dynamics_max_makeup_gain_db': 1.0, 'soundcheck_dynamics_min_headroom_margin_db': 1.0, 'soundcheck_dynamics_required_true_peak_headroom_db': None, 'soundcheck_dynamics_required_summed_bus_true_peak_headroom_db': None, 'soundcheck_dynamics_target_gr_review_only': True, 'soundcheck_dynamics_target_gr_readback_required': True, 'soundcheck_dynamics_makeup_gain_headroom_required': True, 'soundcheck_dynamics_summed_bus_headroom_required': True, 'soundcheck_dynamics_compressor_model_readback_required': True, 'soundcheck_dynamics_gain_reduction_meter_readback_required': True, 'soundcheck_dynamics_feedback_recheck_required': True, 'soundcheck_dynamics_operator_confirmation_required': True, 'soundcheck_dynamics_readback_required': True, 'soundcheck_dynamics_rollback_ready': True, 'soundcheck_dynamics_live_promotion_status': 'blocked_until_real_wing_compressor_readback_gr_meter_headroom_feedback_recheck_and_operator_approval', 'soundcheck_dynamics_live_apply_blocked_without_wing_proof': True, 'soundcheck_master_processing_review_only_proposal': None, 'soundcheck_master_processing_native_osc_only_required': None, 'soundcheck_master_processing_target_surface': None, 'soundcheck_master_processing_types': None, 'soundcheck_master_processing_wing_apply_parameter_scope': None, 'soundcheck_master_processing_pa_feed_identity_required': None, 'soundcheck_master_processing_true_peak_headroom_recheck_required': None, 'soundcheck_master_processing_feedback_recheck_required': None, 'soundcheck_master_processing_bypass_level_match_required': None, 'soundcheck_master_processing_readback_required': None, 'soundcheck_master_processing_operator_confirmation_required': None, 'soundcheck_master_processing_rollback_ready': None, 'soundcheck_master_processing_no_routing_scene_snapshot_write_required': None, 'soundcheck_master_processing_bus_compression_max_target_gr_db': None, 'soundcheck_master_processing_limiter_max_target_gr_db': None, 'soundcheck_master_processing_live_promotion_status': None, 'soundcheck_master_processing_live_apply_blocked_without_wing_proof': None, 'live_dynamics_conservative_proposal': None, 'live_dynamics_stepwise_threshold_review_required': None, 'live_dynamics_max_threshold_step_db': None, 'live_dynamics_max_target_gain_reduction_db': None, 'live_dynamics_no_makeup_gain_required': None, 'live_dynamics_makeup_gain_db_limit': None, 'live_dynamics_gain_reduction_meter_readback_after_each_step_required': None, 'live_dynamics_previous_comp_model_readback_required': None, 'live_dynamics_operator_rehearsal_required': None, 'live_dynamics_live_promotion_status': None, 'live_dynamics_live_apply_blocked_without_wing_proof': None, 'review_only_parameter_keys': ['target_gain_reduction_db'], 'wing_dynamics_review_only_values': {'target_gain_reduction_db': 2.9}, 'wing_dynamics_review_only_parameter_reason_by_key': {'target_gain_reduction_db': 'target gain reduction is an operator/readback objective; WING Rack fw3.0.5 exposes compressor threshold/ratio/timing/knee/makeup/mix/envelope curve via OSC but no direct target gain-reduction write path'}, 'wing_dynamics_review_only_no_direct_osc_path': True, 'wing_dynamics_review_only_not_apply_ready': True, 'wing_dynamics_gain_reduction_objective_required': True, 'wing_dynamics_gain_reduction_objective_parameter_keys': ['target_gain_reduction_db'], 'wing_dynamics_gain_reduction_objective_values': {'target_gain_reduction_db': 2.9}, 'wing_dynamics_gain_reduction_target_db': 2.9, 'wing_dynamics_gain_reduction_max_db': None, 'wing_dynamics_gain_reduction_objective_tolerance_db': 1.0, 'wing_dynamics_gain_reduction_acceptance_min_db': 1.9, 'wing_dynamics_gain_reduction_acceptance_max_db': 3.9, 'wing_dynamics_gain_reduction_must_not_exceed_db': None, 'wing_dynamics_gain_reduction_objective_is_review_only': True, 'wing_dynamics_gain_reduction_objective_is_not_osc_write': True, 'wing_dynamics_gain_reduction_objective_no_direct_osc_write_path': True, 'wing_dynamics_gain_reduction_objective_apply_ready': False, 'wing_dynamics_gain_reduction_objective_requires_post_apply_meter_readback': True, 'wing_dynamics_gain_reduction_meter_readback_path_suffix': 'dyn/cpeak', 'wing_dynamics_gain_reduction_meter_readback_path_templates': ['/aux/{aux}/dyn/cpeak', '/aux/{aux}/dyn/cmode'], 'wing_dynamics_gain_reduction_meter_fallback_path_templates': [], 'wing_dynamics_gain_reduction_meter_value_type': 'read_only_aux_compressor_peak_percent_and_mode', 'wing_dynamics_gain_reduction_meter_is_not_apply_value': True, 'wing_dynamics_gain_reduction_meter_path_source': 'Docs/manuals/wing/WING Remote Protocols v3.0.5.pdf', 'wing_dynamics_gain_reduction_aux_depth_proxy': True, 'wing_dynamics_gain_reduction_objective_requires_operator_confirmation': True, 'wing_dynamics_gain_reduction_objective_live_promotion_status': 'blocked_until_actual_gain_reduction_meter_readback_and_operator_review', 'wing_dynamics_gain_reduction_objective_operator_check': 'confirm WING aux PSE/LA cpeak activity and cmode after the bounded depth/threshold proposal; aux exposes no dyn/gr meter', 'wing_dynamics_gain_reduction_objective_direct_osc_write_keys_excluded': ['target_gain_reduction_db'], 'wing_aux_dynamics_on_mapping_required': True, 'wing_aux_dynamics_on_osc_path_suffix': 'dyn/on', 'wing_aux_dynamics_on_osc_value': 1, 'wing_aux_dynamics_on_value_type': 'integer_bool', 'wing_aux_dynamics_on_readback_must_match_osc_value': True, 'wing_aux_dynamics_threshold_mapping_required': True, 'wing_aux_dynamics_threshold_osc_path_suffix': 'dyn/thr', 'wing_aux_dynamics_threshold_osc_value': -20.3, 'wing_aux_dynamics_threshold_value_type': 'float', 'wing_aux_dynamics_threshold_readback_must_match_osc_value': True, 'wing_aux_dynamics_depth_mapping_required': True, 'wing_aux_dynamics_depth_osc_path_suffix': 'dyn/depth', 'wing_aux_dynamics_depth_osc_value': 2.9, 'wing_aux_dynamics_depth_value_type': 'float', 'wing_aux_dynamics_depth_readback_must_match_osc_value': True, 'wing_aux_dynamics_fast_mapping_required': True, 'wing_aux_dynamics_fast_osc_path_suffix': 'dyn/fast', 'wing_aux_dynamics_fast_osc_value': 0, 'wing_aux_dynamics_fast_value_type': 'integer_bool', 'wing_aux_dynamics_fast_readback_must_match_osc_value': True}
- `soundcheck_aux_playback_pse_la_dynamics_selected_dynamics_wing_attack_value_out_of_range` (soundcheck_aux_playback_pse_la_dynamics): {'value': None, 'minimum': 0.0, 'maximum': 120.0}
- `soundcheck_aux_playback_pse_la_dynamics_selected_dynamics_wing_release_mapping_missing` (soundcheck_aux_playback_pse_la_dynamics): ['wing_compressor_release_max', 'wing_compressor_release_min', 'wing_compressor_release_osc_path_suffix', 'wing_compressor_release_osc_value', 'wing_compressor_release_quantization_error', 'wing_compressor_release_quantized_from', 'wing_compressor_release_source', 'wing_compressor_release_value_type']
- `soundcheck_aux_playback_pse_la_dynamics_readback_contract_wing_release_mapping_not_required` (soundcheck_aux_playback_pse_la_dynamics): {'required': True, 'wing_osc_path_templates': ['/aux/{aux}/dyn/on', '/aux/{aux}/dyn/thr', '/aux/{aux}/dyn/depth', '/aux/{aux}/dyn/fast', '/aux/{aux}/dyn/peak', '/aux/{aux}/dyn/ingain', '/aux/{aux}/dyn/cpeak', '/aux/{aux}/dyn/cmode'], 'wing_rack_fw_3_0_5_native_osc_supported': True, 'dynamics_model_readback_required': False, 'dynamics_model_readback_must_equal': None, 'wing_compressor_model_readback_required': False, 'wing_compressor_model_readback_must_equal': None, 'wing_compressor_model_osc_path_suffix': None, 'wing_compressor_model_parameters_apply_only_when_model_matches': True, 'wing_aux_dynamics_pse_la_combo': True, 'wing_aux_dynamics_no_dyn_mdl_path': True, 'wing_aux_dynamics_no_ratio_attack_release_paths': True, 'wing_aux_dynamics_write_key_selection_policy': 'apply_only_selected_wing_fw_3_0_5_aux_pse_la_dynamics_value_keys', 'wing_aux_dynamics_apply_only_selected_write_keys': True, 'wing_aux_dynamics_selected_write_keys': ['wing_aux_dynamics_on_osc_value', 'wing_aux_dynamics_threshold_osc_value', 'wing_aux_dynamics_depth_osc_value', 'wing_aux_dynamics_fast_osc_value'], 'wing_aux_dynamics_forbidden_write_keys': ['wing_aux_dynamics_peak_osc_value'], 'wing_dynamics_required_model': None, 'wing_dynamics_current_model': None, 'wing_dynamics_current_enabled': None, 'wing_dynamics_current_operator_owned': None, 'wing_dynamics_model_readback_available': None, 'wing_dynamics_model_conflict': None, 'wing_dynamics_model_readback_required_for_active_slot': None, 'wing_dynamics_model_conflict_policy': None, 'wing_dynamics_model_conflict_blocker': None, 'wing_dynamics_model_readback_blocker': None, 'wing_dynamics_model_change_requires_operator_review': None, 'wing_dynamics_model_read_before_path': None, 'wing_dynamics_on_read_before_path': None, 'wing_dynamics_model_apply_only_when_current_model_matches': None, 'wing_dynamics_write_key_selection_policy': None, 'wing_dynamics_apply_only_selected_write_keys': None, 'wing_dynamics_selected_write_keys': None, 'wing_dynamics_forbidden_write_keys': None, 'read_before_paths': ['dyn/on', 'dyn/thr', 'dyn/depth', 'dyn/fast', 'dyn/peak', 'dyn/ingain', 'dyn/cpeak', 'dyn/cmode'], 'read_after_paths': ['dyn/on', 'dyn/thr', 'dyn/depth', 'dyn/fast', 'dyn/peak', 'dyn/ingain', 'dyn/cpeak', 'dyn/cmode'], 'makeup_gain_must_respect_headroom': True, 'soundcheck_dynamics_bounded_compression_proposal': True, 'soundcheck_dynamics_native_compressor_osc_only_required': True, 'soundcheck_dynamics_max_target_gain_reduction_db': 4.0, 'soundcheck_dynamics_max_makeup_gain_db': 1.0, 'soundcheck_dynamics_min_headroom_margin_db': 1.0, 'soundcheck_dynamics_required_true_peak_headroom_db': None, 'soundcheck_dynamics_required_summed_bus_true_peak_headroom_db': None, 'soundcheck_dynamics_target_gr_review_only': True, 'soundcheck_dynamics_target_gr_readback_required': True, 'soundcheck_dynamics_makeup_gain_headroom_required': True, 'soundcheck_dynamics_summed_bus_headroom_required': True, 'soundcheck_dynamics_compressor_model_readback_required': True, 'soundcheck_dynamics_gain_reduction_meter_readback_required': True, 'soundcheck_dynamics_feedback_recheck_required': True, 'soundcheck_dynamics_operator_confirmation_required': True, 'soundcheck_dynamics_readback_required': True, 'soundcheck_dynamics_rollback_ready': True, 'soundcheck_dynamics_live_promotion_status': 'blocked_until_real_wing_compressor_readback_gr_meter_headroom_feedback_recheck_and_operator_approval', 'soundcheck_dynamics_live_apply_blocked_without_wing_proof': True, 'soundcheck_master_processing_review_only_proposal': None, 'soundcheck_master_processing_native_osc_only_required': None, 'soundcheck_master_processing_target_surface': None, 'soundcheck_master_processing_types': None, 'soundcheck_master_processing_wing_apply_parameter_scope': None, 'soundcheck_master_processing_pa_feed_identity_required': None, 'soundcheck_master_processing_true_peak_headroom_recheck_required': None, 'soundcheck_master_processing_feedback_recheck_required': None, 'soundcheck_master_processing_bypass_level_match_required': None, 'soundcheck_master_processing_readback_required': None, 'soundcheck_master_processing_operator_confirmation_required': None, 'soundcheck_master_processing_rollback_ready': None, 'soundcheck_master_processing_no_routing_scene_snapshot_write_required': None, 'soundcheck_master_processing_bus_compression_max_target_gr_db': None, 'soundcheck_master_processing_limiter_max_target_gr_db': None, 'soundcheck_master_processing_live_promotion_status': None, 'soundcheck_master_processing_live_apply_blocked_without_wing_proof': None, 'live_dynamics_conservative_proposal': None, 'live_dynamics_stepwise_threshold_review_required': None, 'live_dynamics_max_threshold_step_db': None, 'live_dynamics_max_target_gain_reduction_db': None, 'live_dynamics_no_makeup_gain_required': None, 'live_dynamics_makeup_gain_db_limit': None, 'live_dynamics_gain_reduction_meter_readback_after_each_step_required': None, 'live_dynamics_previous_comp_model_readback_required': None, 'live_dynamics_operator_rehearsal_required': None, 'live_dynamics_live_promotion_status': None, 'live_dynamics_live_apply_blocked_without_wing_proof': None, 'review_only_parameter_keys': ['target_gain_reduction_db'], 'wing_dynamics_review_only_values': {'target_gain_reduction_db': 2.9}, 'wing_dynamics_review_only_parameter_reason_by_key': {'target_gain_reduction_db': 'target gain reduction is an operator/readback objective; WING Rack fw3.0.5 exposes compressor threshold/ratio/timing/knee/makeup/mix/envelope curve via OSC but no direct target gain-reduction write path'}, 'wing_dynamics_review_only_no_direct_osc_path': True, 'wing_dynamics_review_only_not_apply_ready': True, 'wing_dynamics_gain_reduction_objective_required': True, 'wing_dynamics_gain_reduction_objective_parameter_keys': ['target_gain_reduction_db'], 'wing_dynamics_gain_reduction_objective_values': {'target_gain_reduction_db': 2.9}, 'wing_dynamics_gain_reduction_target_db': 2.9, 'wing_dynamics_gain_reduction_max_db': None, 'wing_dynamics_gain_reduction_objective_tolerance_db': 1.0, 'wing_dynamics_gain_reduction_acceptance_min_db': 1.9, 'wing_dynamics_gain_reduction_acceptance_max_db': 3.9, 'wing_dynamics_gain_reduction_must_not_exceed_db': None, 'wing_dynamics_gain_reduction_objective_is_review_only': True, 'wing_dynamics_gain_reduction_objective_is_not_osc_write': True, 'wing_dynamics_gain_reduction_objective_no_direct_osc_write_path': True, 'wing_dynamics_gain_reduction_objective_apply_ready': False, 'wing_dynamics_gain_reduction_objective_requires_post_apply_meter_readback': True, 'wing_dynamics_gain_reduction_meter_readback_path_suffix': 'dyn/cpeak', 'wing_dynamics_gain_reduction_meter_readback_path_templates': ['/aux/{aux}/dyn/cpeak', '/aux/{aux}/dyn/cmode'], 'wing_dynamics_gain_reduction_meter_fallback_path_templates': [], 'wing_dynamics_gain_reduction_meter_value_type': 'read_only_aux_compressor_peak_percent_and_mode', 'wing_dynamics_gain_reduction_meter_is_not_apply_value': True, 'wing_dynamics_gain_reduction_meter_path_source': 'Docs/manuals/wing/WING Remote Protocols v3.0.5.pdf', 'wing_dynamics_gain_reduction_aux_depth_proxy': True, 'wing_dynamics_gain_reduction_objective_requires_operator_confirmation': True, 'wing_dynamics_gain_reduction_objective_live_promotion_status': 'blocked_until_actual_gain_reduction_meter_readback_and_operator_review', 'wing_dynamics_gain_reduction_objective_operator_check': 'confirm WING aux PSE/LA cpeak activity and cmode after the bounded depth/threshold proposal; aux exposes no dyn/gr meter', 'wing_dynamics_gain_reduction_objective_direct_osc_write_keys_excluded': ['target_gain_reduction_db'], 'wing_aux_dynamics_on_mapping_required': True, 'wing_aux_dynamics_on_osc_path_suffix': 'dyn/on', 'wing_aux_dynamics_on_osc_value': 1, 'wing_aux_dynamics_on_value_type': 'integer_bool', 'wing_aux_dynamics_on_readback_must_match_osc_value': True, 'wing_aux_dynamics_threshold_mapping_required': True, 'wing_aux_dynamics_threshold_osc_path_suffix': 'dyn/thr', 'wing_aux_dynamics_threshold_osc_value': -20.3, 'wing_aux_dynamics_threshold_value_type': 'float', 'wing_aux_dynamics_threshold_readback_must_match_osc_value': True, 'wing_aux_dynamics_depth_mapping_required': True, 'wing_aux_dynamics_depth_osc_path_suffix': 'dyn/depth', 'wing_aux_dynamics_depth_osc_value': 2.9, 'wing_aux_dynamics_depth_value_type': 'float', 'wing_aux_dynamics_depth_readback_must_match_osc_value': True, 'wing_aux_dynamics_fast_mapping_required': True, 'wing_aux_dynamics_fast_osc_path_suffix': 'dyn/fast', 'wing_aux_dynamics_fast_osc_value': 0, 'wing_aux_dynamics_fast_value_type': 'integer_bool', 'wing_aux_dynamics_fast_readback_must_match_osc_value': True}
- `soundcheck_aux_playback_pse_la_dynamics_selected_dynamics_wing_release_path_mismatch` (soundcheck_aux_playback_pse_la_dynamics): None
- `soundcheck_aux_playback_pse_la_dynamics_readback_contract_wing_release_path_mismatch` (soundcheck_aux_playback_pse_la_dynamics): {'required': True, 'wing_osc_path_templates': ['/aux/{aux}/dyn/on', '/aux/{aux}/dyn/thr', '/aux/{aux}/dyn/depth', '/aux/{aux}/dyn/fast', '/aux/{aux}/dyn/peak', '/aux/{aux}/dyn/ingain', '/aux/{aux}/dyn/cpeak', '/aux/{aux}/dyn/cmode'], 'wing_rack_fw_3_0_5_native_osc_supported': True, 'dynamics_model_readback_required': False, 'dynamics_model_readback_must_equal': None, 'wing_compressor_model_readback_required': False, 'wing_compressor_model_readback_must_equal': None, 'wing_compressor_model_osc_path_suffix': None, 'wing_compressor_model_parameters_apply_only_when_model_matches': True, 'wing_aux_dynamics_pse_la_combo': True, 'wing_aux_dynamics_no_dyn_mdl_path': True, 'wing_aux_dynamics_no_ratio_attack_release_paths': True, 'wing_aux_dynamics_write_key_selection_policy': 'apply_only_selected_wing_fw_3_0_5_aux_pse_la_dynamics_value_keys', 'wing_aux_dynamics_apply_only_selected_write_keys': True, 'wing_aux_dynamics_selected_write_keys': ['wing_aux_dynamics_on_osc_value', 'wing_aux_dynamics_threshold_osc_value', 'wing_aux_dynamics_depth_osc_value', 'wing_aux_dynamics_fast_osc_value'], 'wing_aux_dynamics_forbidden_write_keys': ['wing_aux_dynamics_peak_osc_value'], 'wing_dynamics_required_model': None, 'wing_dynamics_current_model': None, 'wing_dynamics_current_enabled': None, 'wing_dynamics_current_operator_owned': None, 'wing_dynamics_model_readback_available': None, 'wing_dynamics_model_conflict': None, 'wing_dynamics_model_readback_required_for_active_slot': None, 'wing_dynamics_model_conflict_policy': None, 'wing_dynamics_model_conflict_blocker': None, 'wing_dynamics_model_readback_blocker': None, 'wing_dynamics_model_change_requires_operator_review': None, 'wing_dynamics_model_read_before_path': None, 'wing_dynamics_on_read_before_path': None, 'wing_dynamics_model_apply_only_when_current_model_matches': None, 'wing_dynamics_write_key_selection_policy': None, 'wing_dynamics_apply_only_selected_write_keys': None, 'wing_dynamics_selected_write_keys': None, 'wing_dynamics_forbidden_write_keys': None, 'read_before_paths': ['dyn/on', 'dyn/thr', 'dyn/depth', 'dyn/fast', 'dyn/peak', 'dyn/ingain', 'dyn/cpeak', 'dyn/cmode'], 'read_after_paths': ['dyn/on', 'dyn/thr', 'dyn/depth', 'dyn/fast', 'dyn/peak', 'dyn/ingain', 'dyn/cpeak', 'dyn/cmode'], 'makeup_gain_must_respect_headroom': True, 'soundcheck_dynamics_bounded_compression_proposal': True, 'soundcheck_dynamics_native_compressor_osc_only_required': True, 'soundcheck_dynamics_max_target_gain_reduction_db': 4.0, 'soundcheck_dynamics_max_makeup_gain_db': 1.0, 'soundcheck_dynamics_min_headroom_margin_db': 1.0, 'soundcheck_dynamics_required_true_peak_headroom_db': None, 'soundcheck_dynamics_required_summed_bus_true_peak_headroom_db': None, 'soundcheck_dynamics_target_gr_review_only': True, 'soundcheck_dynamics_target_gr_readback_required': True, 'soundcheck_dynamics_makeup_gain_headroom_required': True, 'soundcheck_dynamics_summed_bus_headroom_required': True, 'soundcheck_dynamics_compressor_model_readback_required': True, 'soundcheck_dynamics_gain_reduction_meter_readback_required': True, 'soundcheck_dynamics_feedback_recheck_required': True, 'soundcheck_dynamics_operator_confirmation_required': True, 'soundcheck_dynamics_readback_required': True, 'soundcheck_dynamics_rollback_ready': True, 'soundcheck_dynamics_live_promotion_status': 'blocked_until_real_wing_compressor_readback_gr_meter_headroom_feedback_recheck_and_operator_approval', 'soundcheck_dynamics_live_apply_blocked_without_wing_proof': True, 'soundcheck_master_processing_review_only_proposal': None, 'soundcheck_master_processing_native_osc_only_required': None, 'soundcheck_master_processing_target_surface': None, 'soundcheck_master_processing_types': None, 'soundcheck_master_processing_wing_apply_parameter_scope': None, 'soundcheck_master_processing_pa_feed_identity_required': None, 'soundcheck_master_processing_true_peak_headroom_recheck_required': None, 'soundcheck_master_processing_feedback_recheck_required': None, 'soundcheck_master_processing_bypass_level_match_required': None, 'soundcheck_master_processing_readback_required': None, 'soundcheck_master_processing_operator_confirmation_required': None, 'soundcheck_master_processing_rollback_ready': None, 'soundcheck_master_processing_no_routing_scene_snapshot_write_required': None, 'soundcheck_master_processing_bus_compression_max_target_gr_db': None, 'soundcheck_master_processing_limiter_max_target_gr_db': None, 'soundcheck_master_processing_live_promotion_status': None, 'soundcheck_master_processing_live_apply_blocked_without_wing_proof': None, 'live_dynamics_conservative_proposal': None, 'live_dynamics_stepwise_threshold_review_required': None, 'live_dynamics_max_threshold_step_db': None, 'live_dynamics_max_target_gain_reduction_db': None, 'live_dynamics_no_makeup_gain_required': None, 'live_dynamics_makeup_gain_db_limit': None, 'live_dynamics_gain_reduction_meter_readback_after_each_step_required': None, 'live_dynamics_previous_comp_model_readback_required': None, 'live_dynamics_operator_rehearsal_required': None, 'live_dynamics_live_promotion_status': None, 'live_dynamics_live_apply_blocked_without_wing_proof': None, 'review_only_parameter_keys': ['target_gain_reduction_db'], 'wing_dynamics_review_only_values': {'target_gain_reduction_db': 2.9}, 'wing_dynamics_review_only_parameter_reason_by_key': {'target_gain_reduction_db': 'target gain reduction is an operator/readback objective; WING Rack fw3.0.5 exposes compressor threshold/ratio/timing/knee/makeup/mix/envelope curve via OSC but no direct target gain-reduction write path'}, 'wing_dynamics_review_only_no_direct_osc_path': True, 'wing_dynamics_review_only_not_apply_ready': True, 'wing_dynamics_gain_reduction_objective_required': True, 'wing_dynamics_gain_reduction_objective_parameter_keys': ['target_gain_reduction_db'], 'wing_dynamics_gain_reduction_objective_values': {'target_gain_reduction_db': 2.9}, 'wing_dynamics_gain_reduction_target_db': 2.9, 'wing_dynamics_gain_reduction_max_db': None, 'wing_dynamics_gain_reduction_objective_tolerance_db': 1.0, 'wing_dynamics_gain_reduction_acceptance_min_db': 1.9, 'wing_dynamics_gain_reduction_acceptance_max_db': 3.9, 'wing_dynamics_gain_reduction_must_not_exceed_db': None, 'wing_dynamics_gain_reduction_objective_is_review_only': True, 'wing_dynamics_gain_reduction_objective_is_not_osc_write': True, 'wing_dynamics_gain_reduction_objective_no_direct_osc_write_path': True, 'wing_dynamics_gain_reduction_objective_apply_ready': False, 'wing_dynamics_gain_reduction_objective_requires_post_apply_meter_readback': True, 'wing_dynamics_gain_reduction_meter_readback_path_suffix': 'dyn/cpeak', 'wing_dynamics_gain_reduction_meter_readback_path_templates': ['/aux/{aux}/dyn/cpeak', '/aux/{aux}/dyn/cmode'], 'wing_dynamics_gain_reduction_meter_fallback_path_templates': [], 'wing_dynamics_gain_reduction_meter_value_type': 'read_only_aux_compressor_peak_percent_and_mode', 'wing_dynamics_gain_reduction_meter_is_not_apply_value': True, 'wing_dynamics_gain_reduction_meter_path_source': 'Docs/manuals/wing/WING Remote Protocols v3.0.5.pdf', 'wing_dynamics_gain_reduction_aux_depth_proxy': True, 'wing_dynamics_gain_reduction_objective_requires_operator_confirmation': True, 'wing_dynamics_gain_reduction_objective_live_promotion_status': 'blocked_until_actual_gain_reduction_meter_readback_and_operator_review', 'wing_dynamics_gain_reduction_objective_operator_check': 'confirm WING aux PSE/LA cpeak activity and cmode after the bounded depth/threshold proposal; aux exposes no dyn/gr meter', 'wing_dynamics_gain_reduction_objective_direct_osc_write_keys_excluded': ['target_gain_reduction_db'], 'wing_aux_dynamics_on_mapping_required': True, 'wing_aux_dynamics_on_osc_path_suffix': 'dyn/on', 'wing_aux_dynamics_on_osc_value': 1, 'wing_aux_dynamics_on_value_type': 'integer_bool', 'wing_aux_dynamics_on_readback_must_match_osc_value': True, 'wing_aux_dynamics_threshold_mapping_required': True, 'wing_aux_dynamics_threshold_osc_path_suffix': 'dyn/thr', 'wing_aux_dynamics_threshold_osc_value': -20.3, 'wing_aux_dynamics_threshold_value_type': 'float', 'wing_aux_dynamics_threshold_readback_must_match_osc_value': True, 'wing_aux_dynamics_depth_mapping_required': True, 'wing_aux_dynamics_depth_osc_path_suffix': 'dyn/depth', 'wing_aux_dynamics_depth_osc_value': 2.9, 'wing_aux_dynamics_depth_value_type': 'float', 'wing_aux_dynamics_depth_readback_must_match_osc_value': True, 'wing_aux_dynamics_fast_mapping_required': True, 'wing_aux_dynamics_fast_osc_path_suffix': 'dyn/fast', 'wing_aux_dynamics_fast_osc_value': 0, 'wing_aux_dynamics_fast_value_type': 'integer_bool', 'wing_aux_dynamics_fast_readback_must_match_osc_value': True}
- `soundcheck_aux_playback_pse_la_dynamics_readback_contract_wing_release_native_path_missing` (soundcheck_aux_playback_pse_la_dynamics): []
- `soundcheck_aux_playback_pse_la_dynamics_readback_contract_wing_release_contract_path_missing` (soundcheck_aux_playback_pse_la_dynamics): []
- `soundcheck_aux_playback_pse_la_dynamics_readback_contract_wing_release_readback_missing` (soundcheck_aux_playback_pse_la_dynamics): {'required': True, 'wing_osc_path_templates': ['/aux/{aux}/dyn/on', '/aux/{aux}/dyn/thr', '/aux/{aux}/dyn/depth', '/aux/{aux}/dyn/fast', '/aux/{aux}/dyn/peak', '/aux/{aux}/dyn/ingain', '/aux/{aux}/dyn/cpeak', '/aux/{aux}/dyn/cmode'], 'wing_rack_fw_3_0_5_native_osc_supported': True, 'dynamics_model_readback_required': False, 'dynamics_model_readback_must_equal': None, 'wing_compressor_model_readback_required': False, 'wing_compressor_model_readback_must_equal': None, 'wing_compressor_model_osc_path_suffix': None, 'wing_compressor_model_parameters_apply_only_when_model_matches': True, 'wing_aux_dynamics_pse_la_combo': True, 'wing_aux_dynamics_no_dyn_mdl_path': True, 'wing_aux_dynamics_no_ratio_attack_release_paths': True, 'wing_aux_dynamics_write_key_selection_policy': 'apply_only_selected_wing_fw_3_0_5_aux_pse_la_dynamics_value_keys', 'wing_aux_dynamics_apply_only_selected_write_keys': True, 'wing_aux_dynamics_selected_write_keys': ['wing_aux_dynamics_on_osc_value', 'wing_aux_dynamics_threshold_osc_value', 'wing_aux_dynamics_depth_osc_value', 'wing_aux_dynamics_fast_osc_value'], 'wing_aux_dynamics_forbidden_write_keys': ['wing_aux_dynamics_peak_osc_value'], 'wing_dynamics_required_model': None, 'wing_dynamics_current_model': None, 'wing_dynamics_current_enabled': None, 'wing_dynamics_current_operator_owned': None, 'wing_dynamics_model_readback_available': None, 'wing_dynamics_model_conflict': None, 'wing_dynamics_model_readback_required_for_active_slot': None, 'wing_dynamics_model_conflict_policy': None, 'wing_dynamics_model_conflict_blocker': None, 'wing_dynamics_model_readback_blocker': None, 'wing_dynamics_model_change_requires_operator_review': None, 'wing_dynamics_model_read_before_path': None, 'wing_dynamics_on_read_before_path': None, 'wing_dynamics_model_apply_only_when_current_model_matches': None, 'wing_dynamics_write_key_selection_policy': None, 'wing_dynamics_apply_only_selected_write_keys': None, 'wing_dynamics_selected_write_keys': None, 'wing_dynamics_forbidden_write_keys': None, 'read_before_paths': ['dyn/on', 'dyn/thr', 'dyn/depth', 'dyn/fast', 'dyn/peak', 'dyn/ingain', 'dyn/cpeak', 'dyn/cmode'], 'read_after_paths': ['dyn/on', 'dyn/thr', 'dyn/depth', 'dyn/fast', 'dyn/peak', 'dyn/ingain', 'dyn/cpeak', 'dyn/cmode'], 'makeup_gain_must_respect_headroom': True, 'soundcheck_dynamics_bounded_compression_proposal': True, 'soundcheck_dynamics_native_compressor_osc_only_required': True, 'soundcheck_dynamics_max_target_gain_reduction_db': 4.0, 'soundcheck_dynamics_max_makeup_gain_db': 1.0, 'soundcheck_dynamics_min_headroom_margin_db': 1.0, 'soundcheck_dynamics_required_true_peak_headroom_db': None, 'soundcheck_dynamics_required_summed_bus_true_peak_headroom_db': None, 'soundcheck_dynamics_target_gr_review_only': True, 'soundcheck_dynamics_target_gr_readback_required': True, 'soundcheck_dynamics_makeup_gain_headroom_required': True, 'soundcheck_dynamics_summed_bus_headroom_required': True, 'soundcheck_dynamics_compressor_model_readback_required': True, 'soundcheck_dynamics_gain_reduction_meter_readback_required': True, 'soundcheck_dynamics_feedback_recheck_required': True, 'soundcheck_dynamics_operator_confirmation_required': True, 'soundcheck_dynamics_readback_required': True, 'soundcheck_dynamics_rollback_ready': True, 'soundcheck_dynamics_live_promotion_status': 'blocked_until_real_wing_compressor_readback_gr_meter_headroom_feedback_recheck_and_operator_approval', 'soundcheck_dynamics_live_apply_blocked_without_wing_proof': True, 'soundcheck_master_processing_review_only_proposal': None, 'soundcheck_master_processing_native_osc_only_required': None, 'soundcheck_master_processing_target_surface': None, 'soundcheck_master_processing_types': None, 'soundcheck_master_processing_wing_apply_parameter_scope': None, 'soundcheck_master_processing_pa_feed_identity_required': None, 'soundcheck_master_processing_true_peak_headroom_recheck_required': None, 'soundcheck_master_processing_feedback_recheck_required': None, 'soundcheck_master_processing_bypass_level_match_required': None, 'soundcheck_master_processing_readback_required': None, 'soundcheck_master_processing_operator_confirmation_required': None, 'soundcheck_master_processing_rollback_ready': None, 'soundcheck_master_processing_no_routing_scene_snapshot_write_required': None, 'soundcheck_master_processing_bus_compression_max_target_gr_db': None, 'soundcheck_master_processing_limiter_max_target_gr_db': None, 'soundcheck_master_processing_live_promotion_status': None, 'soundcheck_master_processing_live_apply_blocked_without_wing_proof': None, 'live_dynamics_conservative_proposal': None, 'live_dynamics_stepwise_threshold_review_required': None, 'live_dynamics_max_threshold_step_db': None, 'live_dynamics_max_target_gain_reduction_db': None, 'live_dynamics_no_makeup_gain_required': None, 'live_dynamics_makeup_gain_db_limit': None, 'live_dynamics_gain_reduction_meter_readback_after_each_step_required': None, 'live_dynamics_previous_comp_model_readback_required': None, 'live_dynamics_operator_rehearsal_required': None, 'live_dynamics_live_promotion_status': None, 'live_dynamics_live_apply_blocked_without_wing_proof': None, 'review_only_parameter_keys': ['target_gain_reduction_db'], 'wing_dynamics_review_only_values': {'target_gain_reduction_db': 2.9}, 'wing_dynamics_review_only_parameter_reason_by_key': {'target_gain_reduction_db': 'target gain reduction is an operator/readback objective; WING Rack fw3.0.5 exposes compressor threshold/ratio/timing/knee/makeup/mix/envelope curve via OSC but no direct target gain-reduction write path'}, 'wing_dynamics_review_only_no_direct_osc_path': True, 'wing_dynamics_review_only_not_apply_ready': True, 'wing_dynamics_gain_reduction_objective_required': True, 'wing_dynamics_gain_reduction_objective_parameter_keys': ['target_gain_reduction_db'], 'wing_dynamics_gain_reduction_objective_values': {'target_gain_reduction_db': 2.9}, 'wing_dynamics_gain_reduction_target_db': 2.9, 'wing_dynamics_gain_reduction_max_db': None, 'wing_dynamics_gain_reduction_objective_tolerance_db': 1.0, 'wing_dynamics_gain_reduction_acceptance_min_db': 1.9, 'wing_dynamics_gain_reduction_acceptance_max_db': 3.9, 'wing_dynamics_gain_reduction_must_not_exceed_db': None, 'wing_dynamics_gain_reduction_objective_is_review_only': True, 'wing_dynamics_gain_reduction_objective_is_not_osc_write': True, 'wing_dynamics_gain_reduction_objective_no_direct_osc_write_path': True, 'wing_dynamics_gain_reduction_objective_apply_ready': False, 'wing_dynamics_gain_reduction_objective_requires_post_apply_meter_readback': True, 'wing_dynamics_gain_reduction_meter_readback_path_suffix': 'dyn/cpeak', 'wing_dynamics_gain_reduction_meter_readback_path_templates': ['/aux/{aux}/dyn/cpeak', '/aux/{aux}/dyn/cmode'], 'wing_dynamics_gain_reduction_meter_fallback_path_templates': [], 'wing_dynamics_gain_reduction_meter_value_type': 'read_only_aux_compressor_peak_percent_and_mode', 'wing_dynamics_gain_reduction_meter_is_not_apply_value': True, 'wing_dynamics_gain_reduction_meter_path_source': 'Docs/manuals/wing/WING Remote Protocols v3.0.5.pdf', 'wing_dynamics_gain_reduction_aux_depth_proxy': True, 'wing_dynamics_gain_reduction_objective_requires_operator_confirmation': True, 'wing_dynamics_gain_reduction_objective_live_promotion_status': 'blocked_until_actual_gain_reduction_meter_readback_and_operator_review', 'wing_dynamics_gain_reduction_objective_operator_check': 'confirm WING aux PSE/LA cpeak activity and cmode after the bounded depth/threshold proposal; aux exposes no dyn/gr meter', 'wing_dynamics_gain_reduction_objective_direct_osc_write_keys_excluded': ['target_gain_reduction_db'], 'wing_aux_dynamics_on_mapping_required': True, 'wing_aux_dynamics_on_osc_path_suffix': 'dyn/on', 'wing_aux_dynamics_on_osc_value': 1, 'wing_aux_dynamics_on_value_type': 'integer_bool', 'wing_aux_dynamics_on_readback_must_match_osc_value': True, 'wing_aux_dynamics_threshold_mapping_required': True, 'wing_aux_dynamics_threshold_osc_path_suffix': 'dyn/thr', 'wing_aux_dynamics_threshold_osc_value': -20.3, 'wing_aux_dynamics_threshold_value_type': 'float', 'wing_aux_dynamics_threshold_readback_must_match_osc_value': True, 'wing_aux_dynamics_depth_mapping_required': True, 'wing_aux_dynamics_depth_osc_path_suffix': 'dyn/depth', 'wing_aux_dynamics_depth_osc_value': 2.9, 'wing_aux_dynamics_depth_value_type': 'float', 'wing_aux_dynamics_depth_readback_must_match_osc_value': True, 'wing_aux_dynamics_fast_mapping_required': True, 'wing_aux_dynamics_fast_osc_path_suffix': 'dyn/fast', 'wing_aux_dynamics_fast_osc_value': 0, 'wing_aux_dynamics_fast_value_type': 'integer_bool', 'wing_aux_dynamics_fast_readback_must_match_osc_value': True}
- `soundcheck_aux_playback_pse_la_dynamics_selected_dynamics_wing_release_value_out_of_range` (soundcheck_aux_playback_pse_la_dynamics): {'value': None, 'minimum': 4.0, 'maximum': 4000.0}
- `soundcheck_aux_playback_pse_la_dynamics_selected_dynamics_wing_makeup_gain_mapping_missing` (soundcheck_aux_playback_pse_la_dynamics): ['wing_compressor_makeup_gain_max', 'wing_compressor_makeup_gain_min', 'wing_compressor_makeup_gain_osc_path_suffix', 'wing_compressor_makeup_gain_osc_value', 'wing_compressor_makeup_gain_quantization_error', 'wing_compressor_makeup_gain_quantized_from', 'wing_compressor_makeup_gain_source', 'wing_compressor_makeup_gain_value_type']
- `soundcheck_aux_playback_pse_la_dynamics_readback_contract_wing_makeup_gain_mapping_not_required` (soundcheck_aux_playback_pse_la_dynamics): {'required': True, 'wing_osc_path_templates': ['/aux/{aux}/dyn/on', '/aux/{aux}/dyn/thr', '/aux/{aux}/dyn/depth', '/aux/{aux}/dyn/fast', '/aux/{aux}/dyn/peak', '/aux/{aux}/dyn/ingain', '/aux/{aux}/dyn/cpeak', '/aux/{aux}/dyn/cmode'], 'wing_rack_fw_3_0_5_native_osc_supported': True, 'dynamics_model_readback_required': False, 'dynamics_model_readback_must_equal': None, 'wing_compressor_model_readback_required': False, 'wing_compressor_model_readback_must_equal': None, 'wing_compressor_model_osc_path_suffix': None, 'wing_compressor_model_parameters_apply_only_when_model_matches': True, 'wing_aux_dynamics_pse_la_combo': True, 'wing_aux_dynamics_no_dyn_mdl_path': True, 'wing_aux_dynamics_no_ratio_attack_release_paths': True, 'wing_aux_dynamics_write_key_selection_policy': 'apply_only_selected_wing_fw_3_0_5_aux_pse_la_dynamics_value_keys', 'wing_aux_dynamics_apply_only_selected_write_keys': True, 'wing_aux_dynamics_selected_write_keys': ['wing_aux_dynamics_on_osc_value', 'wing_aux_dynamics_threshold_osc_value', 'wing_aux_dynamics_depth_osc_value', 'wing_aux_dynamics_fast_osc_value'], 'wing_aux_dynamics_forbidden_write_keys': ['wing_aux_dynamics_peak_osc_value'], 'wing_dynamics_required_model': None, 'wing_dynamics_current_model': None, 'wing_dynamics_current_enabled': None, 'wing_dynamics_current_operator_owned': None, 'wing_dynamics_model_readback_available': None, 'wing_dynamics_model_conflict': None, 'wing_dynamics_model_readback_required_for_active_slot': None, 'wing_dynamics_model_conflict_policy': None, 'wing_dynamics_model_conflict_blocker': None, 'wing_dynamics_model_readback_blocker': None, 'wing_dynamics_model_change_requires_operator_review': None, 'wing_dynamics_model_read_before_path': None, 'wing_dynamics_on_read_before_path': None, 'wing_dynamics_model_apply_only_when_current_model_matches': None, 'wing_dynamics_write_key_selection_policy': None, 'wing_dynamics_apply_only_selected_write_keys': None, 'wing_dynamics_selected_write_keys': None, 'wing_dynamics_forbidden_write_keys': None, 'read_before_paths': ['dyn/on', 'dyn/thr', 'dyn/depth', 'dyn/fast', 'dyn/peak', 'dyn/ingain', 'dyn/cpeak', 'dyn/cmode'], 'read_after_paths': ['dyn/on', 'dyn/thr', 'dyn/depth', 'dyn/fast', 'dyn/peak', 'dyn/ingain', 'dyn/cpeak', 'dyn/cmode'], 'makeup_gain_must_respect_headroom': True, 'soundcheck_dynamics_bounded_compression_proposal': True, 'soundcheck_dynamics_native_compressor_osc_only_required': True, 'soundcheck_dynamics_max_target_gain_reduction_db': 4.0, 'soundcheck_dynamics_max_makeup_gain_db': 1.0, 'soundcheck_dynamics_min_headroom_margin_db': 1.0, 'soundcheck_dynamics_required_true_peak_headroom_db': None, 'soundcheck_dynamics_required_summed_bus_true_peak_headroom_db': None, 'soundcheck_dynamics_target_gr_review_only': True, 'soundcheck_dynamics_target_gr_readback_required': True, 'soundcheck_dynamics_makeup_gain_headroom_required': True, 'soundcheck_dynamics_summed_bus_headroom_required': True, 'soundcheck_dynamics_compressor_model_readback_required': True, 'soundcheck_dynamics_gain_reduction_meter_readback_required': True, 'soundcheck_dynamics_feedback_recheck_required': True, 'soundcheck_dynamics_operator_confirmation_required': True, 'soundcheck_dynamics_readback_required': True, 'soundcheck_dynamics_rollback_ready': True, 'soundcheck_dynamics_live_promotion_status': 'blocked_until_real_wing_compressor_readback_gr_meter_headroom_feedback_recheck_and_operator_approval', 'soundcheck_dynamics_live_apply_blocked_without_wing_proof': True, 'soundcheck_master_processing_review_only_proposal': None, 'soundcheck_master_processing_native_osc_only_required': None, 'soundcheck_master_processing_target_surface': None, 'soundcheck_master_processing_types': None, 'soundcheck_master_processing_wing_apply_parameter_scope': None, 'soundcheck_master_processing_pa_feed_identity_required': None, 'soundcheck_master_processing_true_peak_headroom_recheck_required': None, 'soundcheck_master_processing_feedback_recheck_required': None, 'soundcheck_master_processing_bypass_level_match_required': None, 'soundcheck_master_processing_readback_required': None, 'soundcheck_master_processing_operator_confirmation_required': None, 'soundcheck_master_processing_rollback_ready': None, 'soundcheck_master_processing_no_routing_scene_snapshot_write_required': None, 'soundcheck_master_processing_bus_compression_max_target_gr_db': None, 'soundcheck_master_processing_limiter_max_target_gr_db': None, 'soundcheck_master_processing_live_promotion_status': None, 'soundcheck_master_processing_live_apply_blocked_without_wing_proof': None, 'live_dynamics_conservative_proposal': None, 'live_dynamics_stepwise_threshold_review_required': None, 'live_dynamics_max_threshold_step_db': None, 'live_dynamics_max_target_gain_reduction_db': None, 'live_dynamics_no_makeup_gain_required': None, 'live_dynamics_makeup_gain_db_limit': None, 'live_dynamics_gain_reduction_meter_readback_after_each_step_required': None, 'live_dynamics_previous_comp_model_readback_required': None, 'live_dynamics_operator_rehearsal_required': None, 'live_dynamics_live_promotion_status': None, 'live_dynamics_live_apply_blocked_without_wing_proof': None, 'review_only_parameter_keys': ['target_gain_reduction_db'], 'wing_dynamics_review_only_values': {'target_gain_reduction_db': 2.9}, 'wing_dynamics_review_only_parameter_reason_by_key': {'target_gain_reduction_db': 'target gain reduction is an operator/readback objective; WING Rack fw3.0.5 exposes compressor threshold/ratio/timing/knee/makeup/mix/envelope curve via OSC but no direct target gain-reduction write path'}, 'wing_dynamics_review_only_no_direct_osc_path': True, 'wing_dynamics_review_only_not_apply_ready': True, 'wing_dynamics_gain_reduction_objective_required': True, 'wing_dynamics_gain_reduction_objective_parameter_keys': ['target_gain_reduction_db'], 'wing_dynamics_gain_reduction_objective_values': {'target_gain_reduction_db': 2.9}, 'wing_dynamics_gain_reduction_target_db': 2.9, 'wing_dynamics_gain_reduction_max_db': None, 'wing_dynamics_gain_reduction_objective_tolerance_db': 1.0, 'wing_dynamics_gain_reduction_acceptance_min_db': 1.9, 'wing_dynamics_gain_reduction_acceptance_max_db': 3.9, 'wing_dynamics_gain_reduction_must_not_exceed_db': None, 'wing_dynamics_gain_reduction_objective_is_review_only': True, 'wing_dynamics_gain_reduction_objective_is_not_osc_write': True, 'wing_dynamics_gain_reduction_objective_no_direct_osc_write_path': True, 'wing_dynamics_gain_reduction_objective_apply_ready': False, 'wing_dynamics_gain_reduction_objective_requires_post_apply_meter_readback': True, 'wing_dynamics_gain_reduction_meter_readback_path_suffix': 'dyn/cpeak', 'wing_dynamics_gain_reduction_meter_readback_path_templates': ['/aux/{aux}/dyn/cpeak', '/aux/{aux}/dyn/cmode'], 'wing_dynamics_gain_reduction_meter_fallback_path_templates': [], 'wing_dynamics_gain_reduction_meter_value_type': 'read_only_aux_compressor_peak_percent_and_mode', 'wing_dynamics_gain_reduction_meter_is_not_apply_value': True, 'wing_dynamics_gain_reduction_meter_path_source': 'Docs/manuals/wing/WING Remote Protocols v3.0.5.pdf', 'wing_dynamics_gain_reduction_aux_depth_proxy': True, 'wing_dynamics_gain_reduction_objective_requires_operator_confirmation': True, 'wing_dynamics_gain_reduction_objective_live_promotion_status': 'blocked_until_actual_gain_reduction_meter_readback_and_operator_review', 'wing_dynamics_gain_reduction_objective_operator_check': 'confirm WING aux PSE/LA cpeak activity and cmode after the bounded depth/threshold proposal; aux exposes no dyn/gr meter', 'wing_dynamics_gain_reduction_objective_direct_osc_write_keys_excluded': ['target_gain_reduction_db'], 'wing_aux_dynamics_on_mapping_required': True, 'wing_aux_dynamics_on_osc_path_suffix': 'dyn/on', 'wing_aux_dynamics_on_osc_value': 1, 'wing_aux_dynamics_on_value_type': 'integer_bool', 'wing_aux_dynamics_on_readback_must_match_osc_value': True, 'wing_aux_dynamics_threshold_mapping_required': True, 'wing_aux_dynamics_threshold_osc_path_suffix': 'dyn/thr', 'wing_aux_dynamics_threshold_osc_value': -20.3, 'wing_aux_dynamics_threshold_value_type': 'float', 'wing_aux_dynamics_threshold_readback_must_match_osc_value': True, 'wing_aux_dynamics_depth_mapping_required': True, 'wing_aux_dynamics_depth_osc_path_suffix': 'dyn/depth', 'wing_aux_dynamics_depth_osc_value': 2.9, 'wing_aux_dynamics_depth_value_type': 'float', 'wing_aux_dynamics_depth_readback_must_match_osc_value': True, 'wing_aux_dynamics_fast_mapping_required': True, 'wing_aux_dynamics_fast_osc_path_suffix': 'dyn/fast', 'wing_aux_dynamics_fast_osc_value': 0, 'wing_aux_dynamics_fast_value_type': 'integer_bool', 'wing_aux_dynamics_fast_readback_must_match_osc_value': True}
- `soundcheck_aux_playback_pse_la_dynamics_selected_dynamics_wing_makeup_gain_path_mismatch` (soundcheck_aux_playback_pse_la_dynamics): None
- `soundcheck_aux_playback_pse_la_dynamics_readback_contract_wing_makeup_gain_path_mismatch` (soundcheck_aux_playback_pse_la_dynamics): {'required': True, 'wing_osc_path_templates': ['/aux/{aux}/dyn/on', '/aux/{aux}/dyn/thr', '/aux/{aux}/dyn/depth', '/aux/{aux}/dyn/fast', '/aux/{aux}/dyn/peak', '/aux/{aux}/dyn/ingain', '/aux/{aux}/dyn/cpeak', '/aux/{aux}/dyn/cmode'], 'wing_rack_fw_3_0_5_native_osc_supported': True, 'dynamics_model_readback_required': False, 'dynamics_model_readback_must_equal': None, 'wing_compressor_model_readback_required': False, 'wing_compressor_model_readback_must_equal': None, 'wing_compressor_model_osc_path_suffix': None, 'wing_compressor_model_parameters_apply_only_when_model_matches': True, 'wing_aux_dynamics_pse_la_combo': True, 'wing_aux_dynamics_no_dyn_mdl_path': True, 'wing_aux_dynamics_no_ratio_attack_release_paths': True, 'wing_aux_dynamics_write_key_selection_policy': 'apply_only_selected_wing_fw_3_0_5_aux_pse_la_dynamics_value_keys', 'wing_aux_dynamics_apply_only_selected_write_keys': True, 'wing_aux_dynamics_selected_write_keys': ['wing_aux_dynamics_on_osc_value', 'wing_aux_dynamics_threshold_osc_value', 'wing_aux_dynamics_depth_osc_value', 'wing_aux_dynamics_fast_osc_value'], 'wing_aux_dynamics_forbidden_write_keys': ['wing_aux_dynamics_peak_osc_value'], 'wing_dynamics_required_model': None, 'wing_dynamics_current_model': None, 'wing_dynamics_current_enabled': None, 'wing_dynamics_current_operator_owned': None, 'wing_dynamics_model_readback_available': None, 'wing_dynamics_model_conflict': None, 'wing_dynamics_model_readback_required_for_active_slot': None, 'wing_dynamics_model_conflict_policy': None, 'wing_dynamics_model_conflict_blocker': None, 'wing_dynamics_model_readback_blocker': None, 'wing_dynamics_model_change_requires_operator_review': None, 'wing_dynamics_model_read_before_path': None, 'wing_dynamics_on_read_before_path': None, 'wing_dynamics_model_apply_only_when_current_model_matches': None, 'wing_dynamics_write_key_selection_policy': None, 'wing_dynamics_apply_only_selected_write_keys': None, 'wing_dynamics_selected_write_keys': None, 'wing_dynamics_forbidden_write_keys': None, 'read_before_paths': ['dyn/on', 'dyn/thr', 'dyn/depth', 'dyn/fast', 'dyn/peak', 'dyn/ingain', 'dyn/cpeak', 'dyn/cmode'], 'read_after_paths': ['dyn/on', 'dyn/thr', 'dyn/depth', 'dyn/fast', 'dyn/peak', 'dyn/ingain', 'dyn/cpeak', 'dyn/cmode'], 'makeup_gain_must_respect_headroom': True, 'soundcheck_dynamics_bounded_compression_proposal': True, 'soundcheck_dynamics_native_compressor_osc_only_required': True, 'soundcheck_dynamics_max_target_gain_reduction_db': 4.0, 'soundcheck_dynamics_max_makeup_gain_db': 1.0, 'soundcheck_dynamics_min_headroom_margin_db': 1.0, 'soundcheck_dynamics_required_true_peak_headroom_db': None, 'soundcheck_dynamics_required_summed_bus_true_peak_headroom_db': None, 'soundcheck_dynamics_target_gr_review_only': True, 'soundcheck_dynamics_target_gr_readback_required': True, 'soundcheck_dynamics_makeup_gain_headroom_required': True, 'soundcheck_dynamics_summed_bus_headroom_required': True, 'soundcheck_dynamics_compressor_model_readback_required': True, 'soundcheck_dynamics_gain_reduction_meter_readback_required': True, 'soundcheck_dynamics_feedback_recheck_required': True, 'soundcheck_dynamics_operator_confirmation_required': True, 'soundcheck_dynamics_readback_required': True, 'soundcheck_dynamics_rollback_ready': True, 'soundcheck_dynamics_live_promotion_status': 'blocked_until_real_wing_compressor_readback_gr_meter_headroom_feedback_recheck_and_operator_approval', 'soundcheck_dynamics_live_apply_blocked_without_wing_proof': True, 'soundcheck_master_processing_review_only_proposal': None, 'soundcheck_master_processing_native_osc_only_required': None, 'soundcheck_master_processing_target_surface': None, 'soundcheck_master_processing_types': None, 'soundcheck_master_processing_wing_apply_parameter_scope': None, 'soundcheck_master_processing_pa_feed_identity_required': None, 'soundcheck_master_processing_true_peak_headroom_recheck_required': None, 'soundcheck_master_processing_feedback_recheck_required': None, 'soundcheck_master_processing_bypass_level_match_required': None, 'soundcheck_master_processing_readback_required': None, 'soundcheck_master_processing_operator_confirmation_required': None, 'soundcheck_master_processing_rollback_ready': None, 'soundcheck_master_processing_no_routing_scene_snapshot_write_required': None, 'soundcheck_master_processing_bus_compression_max_target_gr_db': None, 'soundcheck_master_processing_limiter_max_target_gr_db': None, 'soundcheck_master_processing_live_promotion_status': None, 'soundcheck_master_processing_live_apply_blocked_without_wing_proof': None, 'live_dynamics_conservative_proposal': None, 'live_dynamics_stepwise_threshold_review_required': None, 'live_dynamics_max_threshold_step_db': None, 'live_dynamics_max_target_gain_reduction_db': None, 'live_dynamics_no_makeup_gain_required': None, 'live_dynamics_makeup_gain_db_limit': None, 'live_dynamics_gain_reduction_meter_readback_after_each_step_required': None, 'live_dynamics_previous_comp_model_readback_required': None, 'live_dynamics_operator_rehearsal_required': None, 'live_dynamics_live_promotion_status': None, 'live_dynamics_live_apply_blocked_without_wing_proof': None, 'review_only_parameter_keys': ['target_gain_reduction_db'], 'wing_dynamics_review_only_values': {'target_gain_reduction_db': 2.9}, 'wing_dynamics_review_only_parameter_reason_by_key': {'target_gain_reduction_db': 'target gain reduction is an operator/readback objective; WING Rack fw3.0.5 exposes compressor threshold/ratio/timing/knee/makeup/mix/envelope curve via OSC but no direct target gain-reduction write path'}, 'wing_dynamics_review_only_no_direct_osc_path': True, 'wing_dynamics_review_only_not_apply_ready': True, 'wing_dynamics_gain_reduction_objective_required': True, 'wing_dynamics_gain_reduction_objective_parameter_keys': ['target_gain_reduction_db'], 'wing_dynamics_gain_reduction_objective_values': {'target_gain_reduction_db': 2.9}, 'wing_dynamics_gain_reduction_target_db': 2.9, 'wing_dynamics_gain_reduction_max_db': None, 'wing_dynamics_gain_reduction_objective_tolerance_db': 1.0, 'wing_dynamics_gain_reduction_acceptance_min_db': 1.9, 'wing_dynamics_gain_reduction_acceptance_max_db': 3.9, 'wing_dynamics_gain_reduction_must_not_exceed_db': None, 'wing_dynamics_gain_reduction_objective_is_review_only': True, 'wing_dynamics_gain_reduction_objective_is_not_osc_write': True, 'wing_dynamics_gain_reduction_objective_no_direct_osc_write_path': True, 'wing_dynamics_gain_reduction_objective_apply_ready': False, 'wing_dynamics_gain_reduction_objective_requires_post_apply_meter_readback': True, 'wing_dynamics_gain_reduction_meter_readback_path_suffix': 'dyn/cpeak', 'wing_dynamics_gain_reduction_meter_readback_path_templates': ['/aux/{aux}/dyn/cpeak', '/aux/{aux}/dyn/cmode'], 'wing_dynamics_gain_reduction_meter_fallback_path_templates': [], 'wing_dynamics_gain_reduction_meter_value_type': 'read_only_aux_compressor_peak_percent_and_mode', 'wing_dynamics_gain_reduction_meter_is_not_apply_value': True, 'wing_dynamics_gain_reduction_meter_path_source': 'Docs/manuals/wing/WING Remote Protocols v3.0.5.pdf', 'wing_dynamics_gain_reduction_aux_depth_proxy': True, 'wing_dynamics_gain_reduction_objective_requires_operator_confirmation': True, 'wing_dynamics_gain_reduction_objective_live_promotion_status': 'blocked_until_actual_gain_reduction_meter_readback_and_operator_review', 'wing_dynamics_gain_reduction_objective_operator_check': 'confirm WING aux PSE/LA cpeak activity and cmode after the bounded depth/threshold proposal; aux exposes no dyn/gr meter', 'wing_dynamics_gain_reduction_objective_direct_osc_write_keys_excluded': ['target_gain_reduction_db'], 'wing_aux_dynamics_on_mapping_required': True, 'wing_aux_dynamics_on_osc_path_suffix': 'dyn/on', 'wing_aux_dynamics_on_osc_value': 1, 'wing_aux_dynamics_on_value_type': 'integer_bool', 'wing_aux_dynamics_on_readback_must_match_osc_value': True, 'wing_aux_dynamics_threshold_mapping_required': True, 'wing_aux_dynamics_threshold_osc_path_suffix': 'dyn/thr', 'wing_aux_dynamics_threshold_osc_value': -20.3, 'wing_aux_dynamics_threshold_value_type': 'float', 'wing_aux_dynamics_threshold_readback_must_match_osc_value': True, 'wing_aux_dynamics_depth_mapping_required': True, 'wing_aux_dynamics_depth_osc_path_suffix': 'dyn/depth', 'wing_aux_dynamics_depth_osc_value': 2.9, 'wing_aux_dynamics_depth_value_type': 'float', 'wing_aux_dynamics_depth_readback_must_match_osc_value': True, 'wing_aux_dynamics_fast_mapping_required': True, 'wing_aux_dynamics_fast_osc_path_suffix': 'dyn/fast', 'wing_aux_dynamics_fast_osc_value': 0, 'wing_aux_dynamics_fast_value_type': 'integer_bool', 'wing_aux_dynamics_fast_readback_must_match_osc_value': True}
- `soundcheck_aux_playback_pse_la_dynamics_readback_contract_wing_makeup_gain_native_path_missing` (soundcheck_aux_playback_pse_la_dynamics): []
- `soundcheck_aux_playback_pse_la_dynamics_readback_contract_wing_makeup_gain_contract_path_missing` (soundcheck_aux_playback_pse_la_dynamics): []
- `soundcheck_aux_playback_pse_la_dynamics_readback_contract_wing_makeup_gain_readback_missing` (soundcheck_aux_playback_pse_la_dynamics): {'required': True, 'wing_osc_path_templates': ['/aux/{aux}/dyn/on', '/aux/{aux}/dyn/thr', '/aux/{aux}/dyn/depth', '/aux/{aux}/dyn/fast', '/aux/{aux}/dyn/peak', '/aux/{aux}/dyn/ingain', '/aux/{aux}/dyn/cpeak', '/aux/{aux}/dyn/cmode'], 'wing_rack_fw_3_0_5_native_osc_supported': True, 'dynamics_model_readback_required': False, 'dynamics_model_readback_must_equal': None, 'wing_compressor_model_readback_required': False, 'wing_compressor_model_readback_must_equal': None, 'wing_compressor_model_osc_path_suffix': None, 'wing_compressor_model_parameters_apply_only_when_model_matches': True, 'wing_aux_dynamics_pse_la_combo': True, 'wing_aux_dynamics_no_dyn_mdl_path': True, 'wing_aux_dynamics_no_ratio_attack_release_paths': True, 'wing_aux_dynamics_write_key_selection_policy': 'apply_only_selected_wing_fw_3_0_5_aux_pse_la_dynamics_value_keys', 'wing_aux_dynamics_apply_only_selected_write_keys': True, 'wing_aux_dynamics_selected_write_keys': ['wing_aux_dynamics_on_osc_value', 'wing_aux_dynamics_threshold_osc_value', 'wing_aux_dynamics_depth_osc_value', 'wing_aux_dynamics_fast_osc_value'], 'wing_aux_dynamics_forbidden_write_keys': ['wing_aux_dynamics_peak_osc_value'], 'wing_dynamics_required_model': None, 'wing_dynamics_current_model': None, 'wing_dynamics_current_enabled': None, 'wing_dynamics_current_operator_owned': None, 'wing_dynamics_model_readback_available': None, 'wing_dynamics_model_conflict': None, 'wing_dynamics_model_readback_required_for_active_slot': None, 'wing_dynamics_model_conflict_policy': None, 'wing_dynamics_model_conflict_blocker': None, 'wing_dynamics_model_readback_blocker': None, 'wing_dynamics_model_change_requires_operator_review': None, 'wing_dynamics_model_read_before_path': None, 'wing_dynamics_on_read_before_path': None, 'wing_dynamics_model_apply_only_when_current_model_matches': None, 'wing_dynamics_write_key_selection_policy': None, 'wing_dynamics_apply_only_selected_write_keys': None, 'wing_dynamics_selected_write_keys': None, 'wing_dynamics_forbidden_write_keys': None, 'read_before_paths': ['dyn/on', 'dyn/thr', 'dyn/depth', 'dyn/fast', 'dyn/peak', 'dyn/ingain', 'dyn/cpeak', 'dyn/cmode'], 'read_after_paths': ['dyn/on', 'dyn/thr', 'dyn/depth', 'dyn/fast', 'dyn/peak', 'dyn/ingain', 'dyn/cpeak', 'dyn/cmode'], 'makeup_gain_must_respect_headroom': True, 'soundcheck_dynamics_bounded_compression_proposal': True, 'soundcheck_dynamics_native_compressor_osc_only_required': True, 'soundcheck_dynamics_max_target_gain_reduction_db': 4.0, 'soundcheck_dynamics_max_makeup_gain_db': 1.0, 'soundcheck_dynamics_min_headroom_margin_db': 1.0, 'soundcheck_dynamics_required_true_peak_headroom_db': None, 'soundcheck_dynamics_required_summed_bus_true_peak_headroom_db': None, 'soundcheck_dynamics_target_gr_review_only': True, 'soundcheck_dynamics_target_gr_readback_required': True, 'soundcheck_dynamics_makeup_gain_headroom_required': True, 'soundcheck_dynamics_summed_bus_headroom_required': True, 'soundcheck_dynamics_compressor_model_readback_required': True, 'soundcheck_dynamics_gain_reduction_meter_readback_required': True, 'soundcheck_dynamics_feedback_recheck_required': True, 'soundcheck_dynamics_operator_confirmation_required': True, 'soundcheck_dynamics_readback_required': True, 'soundcheck_dynamics_rollback_ready': True, 'soundcheck_dynamics_live_promotion_status': 'blocked_until_real_wing_compressor_readback_gr_meter_headroom_feedback_recheck_and_operator_approval', 'soundcheck_dynamics_live_apply_blocked_without_wing_proof': True, 'soundcheck_master_processing_review_only_proposal': None, 'soundcheck_master_processing_native_osc_only_required': None, 'soundcheck_master_processing_target_surface': None, 'soundcheck_master_processing_types': None, 'soundcheck_master_processing_wing_apply_parameter_scope': None, 'soundcheck_master_processing_pa_feed_identity_required': None, 'soundcheck_master_processing_true_peak_headroom_recheck_required': None, 'soundcheck_master_processing_feedback_recheck_required': None, 'soundcheck_master_processing_bypass_level_match_required': None, 'soundcheck_master_processing_readback_required': None, 'soundcheck_master_processing_operator_confirmation_required': None, 'soundcheck_master_processing_rollback_ready': None, 'soundcheck_master_processing_no_routing_scene_snapshot_write_required': None, 'soundcheck_master_processing_bus_compression_max_target_gr_db': None, 'soundcheck_master_processing_limiter_max_target_gr_db': None, 'soundcheck_master_processing_live_promotion_status': None, 'soundcheck_master_processing_live_apply_blocked_without_wing_proof': None, 'live_dynamics_conservative_proposal': None, 'live_dynamics_stepwise_threshold_review_required': None, 'live_dynamics_max_threshold_step_db': None, 'live_dynamics_max_target_gain_reduction_db': None, 'live_dynamics_no_makeup_gain_required': None, 'live_dynamics_makeup_gain_db_limit': None, 'live_dynamics_gain_reduction_meter_readback_after_each_step_required': None, 'live_dynamics_previous_comp_model_readback_required': None, 'live_dynamics_operator_rehearsal_required': None, 'live_dynamics_live_promotion_status': None, 'live_dynamics_live_apply_blocked_without_wing_proof': None, 'review_only_parameter_keys': ['target_gain_reduction_db'], 'wing_dynamics_review_only_values': {'target_gain_reduction_db': 2.9}, 'wing_dynamics_review_only_parameter_reason_by_key': {'target_gain_reduction_db': 'target gain reduction is an operator/readback objective; WING Rack fw3.0.5 exposes compressor threshold/ratio/timing/knee/makeup/mix/envelope curve via OSC but no direct target gain-reduction write path'}, 'wing_dynamics_review_only_no_direct_osc_path': True, 'wing_dynamics_review_only_not_apply_ready': True, 'wing_dynamics_gain_reduction_objective_required': True, 'wing_dynamics_gain_reduction_objective_parameter_keys': ['target_gain_reduction_db'], 'wing_dynamics_gain_reduction_objective_values': {'target_gain_reduction_db': 2.9}, 'wing_dynamics_gain_reduction_target_db': 2.9, 'wing_dynamics_gain_reduction_max_db': None, 'wing_dynamics_gain_reduction_objective_tolerance_db': 1.0, 'wing_dynamics_gain_reduction_acceptance_min_db': 1.9, 'wing_dynamics_gain_reduction_acceptance_max_db': 3.9, 'wing_dynamics_gain_reduction_must_not_exceed_db': None, 'wing_dynamics_gain_reduction_objective_is_review_only': True, 'wing_dynamics_gain_reduction_objective_is_not_osc_write': True, 'wing_dynamics_gain_reduction_objective_no_direct_osc_write_path': True, 'wing_dynamics_gain_reduction_objective_apply_ready': False, 'wing_dynamics_gain_reduction_objective_requires_post_apply_meter_readback': True, 'wing_dynamics_gain_reduction_meter_readback_path_suffix': 'dyn/cpeak', 'wing_dynamics_gain_reduction_meter_readback_path_templates': ['/aux/{aux}/dyn/cpeak', '/aux/{aux}/dyn/cmode'], 'wing_dynamics_gain_reduction_meter_fallback_path_templates': [], 'wing_dynamics_gain_reduction_meter_value_type': 'read_only_aux_compressor_peak_percent_and_mode', 'wing_dynamics_gain_reduction_meter_is_not_apply_value': True, 'wing_dynamics_gain_reduction_meter_path_source': 'Docs/manuals/wing/WING Remote Protocols v3.0.5.pdf', 'wing_dynamics_gain_reduction_aux_depth_proxy': True, 'wing_dynamics_gain_reduction_objective_requires_operator_confirmation': True, 'wing_dynamics_gain_reduction_objective_live_promotion_status': 'blocked_until_actual_gain_reduction_meter_readback_and_operator_review', 'wing_dynamics_gain_reduction_objective_operator_check': 'confirm WING aux PSE/LA cpeak activity and cmode after the bounded depth/threshold proposal; aux exposes no dyn/gr meter', 'wing_dynamics_gain_reduction_objective_direct_osc_write_keys_excluded': ['target_gain_reduction_db'], 'wing_aux_dynamics_on_mapping_required': True, 'wing_aux_dynamics_on_osc_path_suffix': 'dyn/on', 'wing_aux_dynamics_on_osc_value': 1, 'wing_aux_dynamics_on_value_type': 'integer_bool', 'wing_aux_dynamics_on_readback_must_match_osc_value': True, 'wing_aux_dynamics_threshold_mapping_required': True, 'wing_aux_dynamics_threshold_osc_path_suffix': 'dyn/thr', 'wing_aux_dynamics_threshold_osc_value': -20.3, 'wing_aux_dynamics_threshold_value_type': 'float', 'wing_aux_dynamics_threshold_readback_must_match_osc_value': True, 'wing_aux_dynamics_depth_mapping_required': True, 'wing_aux_dynamics_depth_osc_path_suffix': 'dyn/depth', 'wing_aux_dynamics_depth_osc_value': 2.9, 'wing_aux_dynamics_depth_value_type': 'float', 'wing_aux_dynamics_depth_readback_must_match_osc_value': True, 'wing_aux_dynamics_fast_mapping_required': True, 'wing_aux_dynamics_fast_osc_path_suffix': 'dyn/fast', 'wing_aux_dynamics_fast_osc_value': 0, 'wing_aux_dynamics_fast_value_type': 'integer_bool', 'wing_aux_dynamics_fast_readback_must_match_osc_value': True}
- `soundcheck_aux_playback_pse_la_dynamics_selected_dynamics_wing_makeup_gain_value_out_of_range` (soundcheck_aux_playback_pse_la_dynamics): {'value': None, 'minimum': -6.0, 'maximum': 12.0}
- `soundcheck_aux_playback_pse_la_dynamics_selected_dynamics_wing_mix_mapping_missing` (soundcheck_aux_playback_pse_la_dynamics): ['wing_compressor_mix_max', 'wing_compressor_mix_min', 'wing_compressor_mix_osc_path_suffix', 'wing_compressor_mix_osc_value', 'wing_compressor_mix_quantization_error', 'wing_compressor_mix_quantized_from', 'wing_compressor_mix_source', 'wing_compressor_mix_value_type']
- `soundcheck_aux_playback_pse_la_dynamics_readback_contract_wing_mix_mapping_not_required` (soundcheck_aux_playback_pse_la_dynamics): {'required': True, 'wing_osc_path_templates': ['/aux/{aux}/dyn/on', '/aux/{aux}/dyn/thr', '/aux/{aux}/dyn/depth', '/aux/{aux}/dyn/fast', '/aux/{aux}/dyn/peak', '/aux/{aux}/dyn/ingain', '/aux/{aux}/dyn/cpeak', '/aux/{aux}/dyn/cmode'], 'wing_rack_fw_3_0_5_native_osc_supported': True, 'dynamics_model_readback_required': False, 'dynamics_model_readback_must_equal': None, 'wing_compressor_model_readback_required': False, 'wing_compressor_model_readback_must_equal': None, 'wing_compressor_model_osc_path_suffix': None, 'wing_compressor_model_parameters_apply_only_when_model_matches': True, 'wing_aux_dynamics_pse_la_combo': True, 'wing_aux_dynamics_no_dyn_mdl_path': True, 'wing_aux_dynamics_no_ratio_attack_release_paths': True, 'wing_aux_dynamics_write_key_selection_policy': 'apply_only_selected_wing_fw_3_0_5_aux_pse_la_dynamics_value_keys', 'wing_aux_dynamics_apply_only_selected_write_keys': True, 'wing_aux_dynamics_selected_write_keys': ['wing_aux_dynamics_on_osc_value', 'wing_aux_dynamics_threshold_osc_value', 'wing_aux_dynamics_depth_osc_value', 'wing_aux_dynamics_fast_osc_value'], 'wing_aux_dynamics_forbidden_write_keys': ['wing_aux_dynamics_peak_osc_value'], 'wing_dynamics_required_model': None, 'wing_dynamics_current_model': None, 'wing_dynamics_current_enabled': None, 'wing_dynamics_current_operator_owned': None, 'wing_dynamics_model_readback_available': None, 'wing_dynamics_model_conflict': None, 'wing_dynamics_model_readback_required_for_active_slot': None, 'wing_dynamics_model_conflict_policy': None, 'wing_dynamics_model_conflict_blocker': None, 'wing_dynamics_model_readback_blocker': None, 'wing_dynamics_model_change_requires_operator_review': None, 'wing_dynamics_model_read_before_path': None, 'wing_dynamics_on_read_before_path': None, 'wing_dynamics_model_apply_only_when_current_model_matches': None, 'wing_dynamics_write_key_selection_policy': None, 'wing_dynamics_apply_only_selected_write_keys': None, 'wing_dynamics_selected_write_keys': None, 'wing_dynamics_forbidden_write_keys': None, 'read_before_paths': ['dyn/on', 'dyn/thr', 'dyn/depth', 'dyn/fast', 'dyn/peak', 'dyn/ingain', 'dyn/cpeak', 'dyn/cmode'], 'read_after_paths': ['dyn/on', 'dyn/thr', 'dyn/depth', 'dyn/fast', 'dyn/peak', 'dyn/ingain', 'dyn/cpeak', 'dyn/cmode'], 'makeup_gain_must_respect_headroom': True, 'soundcheck_dynamics_bounded_compression_proposal': True, 'soundcheck_dynamics_native_compressor_osc_only_required': True, 'soundcheck_dynamics_max_target_gain_reduction_db': 4.0, 'soundcheck_dynamics_max_makeup_gain_db': 1.0, 'soundcheck_dynamics_min_headroom_margin_db': 1.0, 'soundcheck_dynamics_required_true_peak_headroom_db': None, 'soundcheck_dynamics_required_summed_bus_true_peak_headroom_db': None, 'soundcheck_dynamics_target_gr_review_only': True, 'soundcheck_dynamics_target_gr_readback_required': True, 'soundcheck_dynamics_makeup_gain_headroom_required': True, 'soundcheck_dynamics_summed_bus_headroom_required': True, 'soundcheck_dynamics_compressor_model_readback_required': True, 'soundcheck_dynamics_gain_reduction_meter_readback_required': True, 'soundcheck_dynamics_feedback_recheck_required': True, 'soundcheck_dynamics_operator_confirmation_required': True, 'soundcheck_dynamics_readback_required': True, 'soundcheck_dynamics_rollback_ready': True, 'soundcheck_dynamics_live_promotion_status': 'blocked_until_real_wing_compressor_readback_gr_meter_headroom_feedback_recheck_and_operator_approval', 'soundcheck_dynamics_live_apply_blocked_without_wing_proof': True, 'soundcheck_master_processing_review_only_proposal': None, 'soundcheck_master_processing_native_osc_only_required': None, 'soundcheck_master_processing_target_surface': None, 'soundcheck_master_processing_types': None, 'soundcheck_master_processing_wing_apply_parameter_scope': None, 'soundcheck_master_processing_pa_feed_identity_required': None, 'soundcheck_master_processing_true_peak_headroom_recheck_required': None, 'soundcheck_master_processing_feedback_recheck_required': None, 'soundcheck_master_processing_bypass_level_match_required': None, 'soundcheck_master_processing_readback_required': None, 'soundcheck_master_processing_operator_confirmation_required': None, 'soundcheck_master_processing_rollback_ready': None, 'soundcheck_master_processing_no_routing_scene_snapshot_write_required': None, 'soundcheck_master_processing_bus_compression_max_target_gr_db': None, 'soundcheck_master_processing_limiter_max_target_gr_db': None, 'soundcheck_master_processing_live_promotion_status': None, 'soundcheck_master_processing_live_apply_blocked_without_wing_proof': None, 'live_dynamics_conservative_proposal': None, 'live_dynamics_stepwise_threshold_review_required': None, 'live_dynamics_max_threshold_step_db': None, 'live_dynamics_max_target_gain_reduction_db': None, 'live_dynamics_no_makeup_gain_required': None, 'live_dynamics_makeup_gain_db_limit': None, 'live_dynamics_gain_reduction_meter_readback_after_each_step_required': None, 'live_dynamics_previous_comp_model_readback_required': None, 'live_dynamics_operator_rehearsal_required': None, 'live_dynamics_live_promotion_status': None, 'live_dynamics_live_apply_blocked_without_wing_proof': None, 'review_only_parameter_keys': ['target_gain_reduction_db'], 'wing_dynamics_review_only_values': {'target_gain_reduction_db': 2.9}, 'wing_dynamics_review_only_parameter_reason_by_key': {'target_gain_reduction_db': 'target gain reduction is an operator/readback objective; WING Rack fw3.0.5 exposes compressor threshold/ratio/timing/knee/makeup/mix/envelope curve via OSC but no direct target gain-reduction write path'}, 'wing_dynamics_review_only_no_direct_osc_path': True, 'wing_dynamics_review_only_not_apply_ready': True, 'wing_dynamics_gain_reduction_objective_required': True, 'wing_dynamics_gain_reduction_objective_parameter_keys': ['target_gain_reduction_db'], 'wing_dynamics_gain_reduction_objective_values': {'target_gain_reduction_db': 2.9}, 'wing_dynamics_gain_reduction_target_db': 2.9, 'wing_dynamics_gain_reduction_max_db': None, 'wing_dynamics_gain_reduction_objective_tolerance_db': 1.0, 'wing_dynamics_gain_reduction_acceptance_min_db': 1.9, 'wing_dynamics_gain_reduction_acceptance_max_db': 3.9, 'wing_dynamics_gain_reduction_must_not_exceed_db': None, 'wing_dynamics_gain_reduction_objective_is_review_only': True, 'wing_dynamics_gain_reduction_objective_is_not_osc_write': True, 'wing_dynamics_gain_reduction_objective_no_direct_osc_write_path': True, 'wing_dynamics_gain_reduction_objective_apply_ready': False, 'wing_dynamics_gain_reduction_objective_requires_post_apply_meter_readback': True, 'wing_dynamics_gain_reduction_meter_readback_path_suffix': 'dyn/cpeak', 'wing_dynamics_gain_reduction_meter_readback_path_templates': ['/aux/{aux}/dyn/cpeak', '/aux/{aux}/dyn/cmode'], 'wing_dynamics_gain_reduction_meter_fallback_path_templates': [], 'wing_dynamics_gain_reduction_meter_value_type': 'read_only_aux_compressor_peak_percent_and_mode', 'wing_dynamics_gain_reduction_meter_is_not_apply_value': True, 'wing_dynamics_gain_reduction_meter_path_source': 'Docs/manuals/wing/WING Remote Protocols v3.0.5.pdf', 'wing_dynamics_gain_reduction_aux_depth_proxy': True, 'wing_dynamics_gain_reduction_objective_requires_operator_confirmation': True, 'wing_dynamics_gain_reduction_objective_live_promotion_status': 'blocked_until_actual_gain_reduction_meter_readback_and_operator_review', 'wing_dynamics_gain_reduction_objective_operator_check': 'confirm WING aux PSE/LA cpeak activity and cmode after the bounded depth/threshold proposal; aux exposes no dyn/gr meter', 'wing_dynamics_gain_reduction_objective_direct_osc_write_keys_excluded': ['target_gain_reduction_db'], 'wing_aux_dynamics_on_mapping_required': True, 'wing_aux_dynamics_on_osc_path_suffix': 'dyn/on', 'wing_aux_dynamics_on_osc_value': 1, 'wing_aux_dynamics_on_value_type': 'integer_bool', 'wing_aux_dynamics_on_readback_must_match_osc_value': True, 'wing_aux_dynamics_threshold_mapping_required': True, 'wing_aux_dynamics_threshold_osc_path_suffix': 'dyn/thr', 'wing_aux_dynamics_threshold_osc_value': -20.3, 'wing_aux_dynamics_threshold_value_type': 'float', 'wing_aux_dynamics_threshold_readback_must_match_osc_value': True, 'wing_aux_dynamics_depth_mapping_required': True, 'wing_aux_dynamics_depth_osc_path_suffix': 'dyn/depth', 'wing_aux_dynamics_depth_osc_value': 2.9, 'wing_aux_dynamics_depth_value_type': 'float', 'wing_aux_dynamics_depth_readback_must_match_osc_value': True, 'wing_aux_dynamics_fast_mapping_required': True, 'wing_aux_dynamics_fast_osc_path_suffix': 'dyn/fast', 'wing_aux_dynamics_fast_osc_value': 0, 'wing_aux_dynamics_fast_value_type': 'integer_bool', 'wing_aux_dynamics_fast_readback_must_match_osc_value': True}
- `soundcheck_aux_playback_pse_la_dynamics_selected_dynamics_wing_mix_path_mismatch` (soundcheck_aux_playback_pse_la_dynamics): None
- `soundcheck_aux_playback_pse_la_dynamics_readback_contract_wing_mix_path_mismatch` (soundcheck_aux_playback_pse_la_dynamics): {'required': True, 'wing_osc_path_templates': ['/aux/{aux}/dyn/on', '/aux/{aux}/dyn/thr', '/aux/{aux}/dyn/depth', '/aux/{aux}/dyn/fast', '/aux/{aux}/dyn/peak', '/aux/{aux}/dyn/ingain', '/aux/{aux}/dyn/cpeak', '/aux/{aux}/dyn/cmode'], 'wing_rack_fw_3_0_5_native_osc_supported': True, 'dynamics_model_readback_required': False, 'dynamics_model_readback_must_equal': None, 'wing_compressor_model_readback_required': False, 'wing_compressor_model_readback_must_equal': None, 'wing_compressor_model_osc_path_suffix': None, 'wing_compressor_model_parameters_apply_only_when_model_matches': True, 'wing_aux_dynamics_pse_la_combo': True, 'wing_aux_dynamics_no_dyn_mdl_path': True, 'wing_aux_dynamics_no_ratio_attack_release_paths': True, 'wing_aux_dynamics_write_key_selection_policy': 'apply_only_selected_wing_fw_3_0_5_aux_pse_la_dynamics_value_keys', 'wing_aux_dynamics_apply_only_selected_write_keys': True, 'wing_aux_dynamics_selected_write_keys': ['wing_aux_dynamics_on_osc_value', 'wing_aux_dynamics_threshold_osc_value', 'wing_aux_dynamics_depth_osc_value', 'wing_aux_dynamics_fast_osc_value'], 'wing_aux_dynamics_forbidden_write_keys': ['wing_aux_dynamics_peak_osc_value'], 'wing_dynamics_required_model': None, 'wing_dynamics_current_model': None, 'wing_dynamics_current_enabled': None, 'wing_dynamics_current_operator_owned': None, 'wing_dynamics_model_readback_available': None, 'wing_dynamics_model_conflict': None, 'wing_dynamics_model_readback_required_for_active_slot': None, 'wing_dynamics_model_conflict_policy': None, 'wing_dynamics_model_conflict_blocker': None, 'wing_dynamics_model_readback_blocker': None, 'wing_dynamics_model_change_requires_operator_review': None, 'wing_dynamics_model_read_before_path': None, 'wing_dynamics_on_read_before_path': None, 'wing_dynamics_model_apply_only_when_current_model_matches': None, 'wing_dynamics_write_key_selection_policy': None, 'wing_dynamics_apply_only_selected_write_keys': None, 'wing_dynamics_selected_write_keys': None, 'wing_dynamics_forbidden_write_keys': None, 'read_before_paths': ['dyn/on', 'dyn/thr', 'dyn/depth', 'dyn/fast', 'dyn/peak', 'dyn/ingain', 'dyn/cpeak', 'dyn/cmode'], 'read_after_paths': ['dyn/on', 'dyn/thr', 'dyn/depth', 'dyn/fast', 'dyn/peak', 'dyn/ingain', 'dyn/cpeak', 'dyn/cmode'], 'makeup_gain_must_respect_headroom': True, 'soundcheck_dynamics_bounded_compression_proposal': True, 'soundcheck_dynamics_native_compressor_osc_only_required': True, 'soundcheck_dynamics_max_target_gain_reduction_db': 4.0, 'soundcheck_dynamics_max_makeup_gain_db': 1.0, 'soundcheck_dynamics_min_headroom_margin_db': 1.0, 'soundcheck_dynamics_required_true_peak_headroom_db': None, 'soundcheck_dynamics_required_summed_bus_true_peak_headroom_db': None, 'soundcheck_dynamics_target_gr_review_only': True, 'soundcheck_dynamics_target_gr_readback_required': True, 'soundcheck_dynamics_makeup_gain_headroom_required': True, 'soundcheck_dynamics_summed_bus_headroom_required': True, 'soundcheck_dynamics_compressor_model_readback_required': True, 'soundcheck_dynamics_gain_reduction_meter_readback_required': True, 'soundcheck_dynamics_feedback_recheck_required': True, 'soundcheck_dynamics_operator_confirmation_required': True, 'soundcheck_dynamics_readback_required': True, 'soundcheck_dynamics_rollback_ready': True, 'soundcheck_dynamics_live_promotion_status': 'blocked_until_real_wing_compressor_readback_gr_meter_headroom_feedback_recheck_and_operator_approval', 'soundcheck_dynamics_live_apply_blocked_without_wing_proof': True, 'soundcheck_master_processing_review_only_proposal': None, 'soundcheck_master_processing_native_osc_only_required': None, 'soundcheck_master_processing_target_surface': None, 'soundcheck_master_processing_types': None, 'soundcheck_master_processing_wing_apply_parameter_scope': None, 'soundcheck_master_processing_pa_feed_identity_required': None, 'soundcheck_master_processing_true_peak_headroom_recheck_required': None, 'soundcheck_master_processing_feedback_recheck_required': None, 'soundcheck_master_processing_bypass_level_match_required': None, 'soundcheck_master_processing_readback_required': None, 'soundcheck_master_processing_operator_confirmation_required': None, 'soundcheck_master_processing_rollback_ready': None, 'soundcheck_master_processing_no_routing_scene_snapshot_write_required': None, 'soundcheck_master_processing_bus_compression_max_target_gr_db': None, 'soundcheck_master_processing_limiter_max_target_gr_db': None, 'soundcheck_master_processing_live_promotion_status': None, 'soundcheck_master_processing_live_apply_blocked_without_wing_proof': None, 'live_dynamics_conservative_proposal': None, 'live_dynamics_stepwise_threshold_review_required': None, 'live_dynamics_max_threshold_step_db': None, 'live_dynamics_max_target_gain_reduction_db': None, 'live_dynamics_no_makeup_gain_required': None, 'live_dynamics_makeup_gain_db_limit': None, 'live_dynamics_gain_reduction_meter_readback_after_each_step_required': None, 'live_dynamics_previous_comp_model_readback_required': None, 'live_dynamics_operator_rehearsal_required': None, 'live_dynamics_live_promotion_status': None, 'live_dynamics_live_apply_blocked_without_wing_proof': None, 'review_only_parameter_keys': ['target_gain_reduction_db'], 'wing_dynamics_review_only_values': {'target_gain_reduction_db': 2.9}, 'wing_dynamics_review_only_parameter_reason_by_key': {'target_gain_reduction_db': 'target gain reduction is an operator/readback objective; WING Rack fw3.0.5 exposes compressor threshold/ratio/timing/knee/makeup/mix/envelope curve via OSC but no direct target gain-reduction write path'}, 'wing_dynamics_review_only_no_direct_osc_path': True, 'wing_dynamics_review_only_not_apply_ready': True, 'wing_dynamics_gain_reduction_objective_required': True, 'wing_dynamics_gain_reduction_objective_parameter_keys': ['target_gain_reduction_db'], 'wing_dynamics_gain_reduction_objective_values': {'target_gain_reduction_db': 2.9}, 'wing_dynamics_gain_reduction_target_db': 2.9, 'wing_dynamics_gain_reduction_max_db': None, 'wing_dynamics_gain_reduction_objective_tolerance_db': 1.0, 'wing_dynamics_gain_reduction_acceptance_min_db': 1.9, 'wing_dynamics_gain_reduction_acceptance_max_db': 3.9, 'wing_dynamics_gain_reduction_must_not_exceed_db': None, 'wing_dynamics_gain_reduction_objective_is_review_only': True, 'wing_dynamics_gain_reduction_objective_is_not_osc_write': True, 'wing_dynamics_gain_reduction_objective_no_direct_osc_write_path': True, 'wing_dynamics_gain_reduction_objective_apply_ready': False, 'wing_dynamics_gain_reduction_objective_requires_post_apply_meter_readback': True, 'wing_dynamics_gain_reduction_meter_readback_path_suffix': 'dyn/cpeak', 'wing_dynamics_gain_reduction_meter_readback_path_templates': ['/aux/{aux}/dyn/cpeak', '/aux/{aux}/dyn/cmode'], 'wing_dynamics_gain_reduction_meter_fallback_path_templates': [], 'wing_dynamics_gain_reduction_meter_value_type': 'read_only_aux_compressor_peak_percent_and_mode', 'wing_dynamics_gain_reduction_meter_is_not_apply_value': True, 'wing_dynamics_gain_reduction_meter_path_source': 'Docs/manuals/wing/WING Remote Protocols v3.0.5.pdf', 'wing_dynamics_gain_reduction_aux_depth_proxy': True, 'wing_dynamics_gain_reduction_objective_requires_operator_confirmation': True, 'wing_dynamics_gain_reduction_objective_live_promotion_status': 'blocked_until_actual_gain_reduction_meter_readback_and_operator_review', 'wing_dynamics_gain_reduction_objective_operator_check': 'confirm WING aux PSE/LA cpeak activity and cmode after the bounded depth/threshold proposal; aux exposes no dyn/gr meter', 'wing_dynamics_gain_reduction_objective_direct_osc_write_keys_excluded': ['target_gain_reduction_db'], 'wing_aux_dynamics_on_mapping_required': True, 'wing_aux_dynamics_on_osc_path_suffix': 'dyn/on', 'wing_aux_dynamics_on_osc_value': 1, 'wing_aux_dynamics_on_value_type': 'integer_bool', 'wing_aux_dynamics_on_readback_must_match_osc_value': True, 'wing_aux_dynamics_threshold_mapping_required': True, 'wing_aux_dynamics_threshold_osc_path_suffix': 'dyn/thr', 'wing_aux_dynamics_threshold_osc_value': -20.3, 'wing_aux_dynamics_threshold_value_type': 'float', 'wing_aux_dynamics_threshold_readback_must_match_osc_value': True, 'wing_aux_dynamics_depth_mapping_required': True, 'wing_aux_dynamics_depth_osc_path_suffix': 'dyn/depth', 'wing_aux_dynamics_depth_osc_value': 2.9, 'wing_aux_dynamics_depth_value_type': 'float', 'wing_aux_dynamics_depth_readback_must_match_osc_value': True, 'wing_aux_dynamics_fast_mapping_required': True, 'wing_aux_dynamics_fast_osc_path_suffix': 'dyn/fast', 'wing_aux_dynamics_fast_osc_value': 0, 'wing_aux_dynamics_fast_value_type': 'integer_bool', 'wing_aux_dynamics_fast_readback_must_match_osc_value': True}
- `soundcheck_aux_playback_pse_la_dynamics_readback_contract_wing_mix_native_path_missing` (soundcheck_aux_playback_pse_la_dynamics): []
- `soundcheck_aux_playback_pse_la_dynamics_readback_contract_wing_mix_contract_path_missing` (soundcheck_aux_playback_pse_la_dynamics): []
- `soundcheck_aux_playback_pse_la_dynamics_readback_contract_wing_mix_readback_missing` (soundcheck_aux_playback_pse_la_dynamics): {'required': True, 'wing_osc_path_templates': ['/aux/{aux}/dyn/on', '/aux/{aux}/dyn/thr', '/aux/{aux}/dyn/depth', '/aux/{aux}/dyn/fast', '/aux/{aux}/dyn/peak', '/aux/{aux}/dyn/ingain', '/aux/{aux}/dyn/cpeak', '/aux/{aux}/dyn/cmode'], 'wing_rack_fw_3_0_5_native_osc_supported': True, 'dynamics_model_readback_required': False, 'dynamics_model_readback_must_equal': None, 'wing_compressor_model_readback_required': False, 'wing_compressor_model_readback_must_equal': None, 'wing_compressor_model_osc_path_suffix': None, 'wing_compressor_model_parameters_apply_only_when_model_matches': True, 'wing_aux_dynamics_pse_la_combo': True, 'wing_aux_dynamics_no_dyn_mdl_path': True, 'wing_aux_dynamics_no_ratio_attack_release_paths': True, 'wing_aux_dynamics_write_key_selection_policy': 'apply_only_selected_wing_fw_3_0_5_aux_pse_la_dynamics_value_keys', 'wing_aux_dynamics_apply_only_selected_write_keys': True, 'wing_aux_dynamics_selected_write_keys': ['wing_aux_dynamics_on_osc_value', 'wing_aux_dynamics_threshold_osc_value', 'wing_aux_dynamics_depth_osc_value', 'wing_aux_dynamics_fast_osc_value'], 'wing_aux_dynamics_forbidden_write_keys': ['wing_aux_dynamics_peak_osc_value'], 'wing_dynamics_required_model': None, 'wing_dynamics_current_model': None, 'wing_dynamics_current_enabled': None, 'wing_dynamics_current_operator_owned': None, 'wing_dynamics_model_readback_available': None, 'wing_dynamics_model_conflict': None, 'wing_dynamics_model_readback_required_for_active_slot': None, 'wing_dynamics_model_conflict_policy': None, 'wing_dynamics_model_conflict_blocker': None, 'wing_dynamics_model_readback_blocker': None, 'wing_dynamics_model_change_requires_operator_review': None, 'wing_dynamics_model_read_before_path': None, 'wing_dynamics_on_read_before_path': None, 'wing_dynamics_model_apply_only_when_current_model_matches': None, 'wing_dynamics_write_key_selection_policy': None, 'wing_dynamics_apply_only_selected_write_keys': None, 'wing_dynamics_selected_write_keys': None, 'wing_dynamics_forbidden_write_keys': None, 'read_before_paths': ['dyn/on', 'dyn/thr', 'dyn/depth', 'dyn/fast', 'dyn/peak', 'dyn/ingain', 'dyn/cpeak', 'dyn/cmode'], 'read_after_paths': ['dyn/on', 'dyn/thr', 'dyn/depth', 'dyn/fast', 'dyn/peak', 'dyn/ingain', 'dyn/cpeak', 'dyn/cmode'], 'makeup_gain_must_respect_headroom': True, 'soundcheck_dynamics_bounded_compression_proposal': True, 'soundcheck_dynamics_native_compressor_osc_only_required': True, 'soundcheck_dynamics_max_target_gain_reduction_db': 4.0, 'soundcheck_dynamics_max_makeup_gain_db': 1.0, 'soundcheck_dynamics_min_headroom_margin_db': 1.0, 'soundcheck_dynamics_required_true_peak_headroom_db': None, 'soundcheck_dynamics_required_summed_bus_true_peak_headroom_db': None, 'soundcheck_dynamics_target_gr_review_only': True, 'soundcheck_dynamics_target_gr_readback_required': True, 'soundcheck_dynamics_makeup_gain_headroom_required': True, 'soundcheck_dynamics_summed_bus_headroom_required': True, 'soundcheck_dynamics_compressor_model_readback_required': True, 'soundcheck_dynamics_gain_reduction_meter_readback_required': True, 'soundcheck_dynamics_feedback_recheck_required': True, 'soundcheck_dynamics_operator_confirmation_required': True, 'soundcheck_dynamics_readback_required': True, 'soundcheck_dynamics_rollback_ready': True, 'soundcheck_dynamics_live_promotion_status': 'blocked_until_real_wing_compressor_readback_gr_meter_headroom_feedback_recheck_and_operator_approval', 'soundcheck_dynamics_live_apply_blocked_without_wing_proof': True, 'soundcheck_master_processing_review_only_proposal': None, 'soundcheck_master_processing_native_osc_only_required': None, 'soundcheck_master_processing_target_surface': None, 'soundcheck_master_processing_types': None, 'soundcheck_master_processing_wing_apply_parameter_scope': None, 'soundcheck_master_processing_pa_feed_identity_required': None, 'soundcheck_master_processing_true_peak_headroom_recheck_required': None, 'soundcheck_master_processing_feedback_recheck_required': None, 'soundcheck_master_processing_bypass_level_match_required': None, 'soundcheck_master_processing_readback_required': None, 'soundcheck_master_processing_operator_confirmation_required': None, 'soundcheck_master_processing_rollback_ready': None, 'soundcheck_master_processing_no_routing_scene_snapshot_write_required': None, 'soundcheck_master_processing_bus_compression_max_target_gr_db': None, 'soundcheck_master_processing_limiter_max_target_gr_db': None, 'soundcheck_master_processing_live_promotion_status': None, 'soundcheck_master_processing_live_apply_blocked_without_wing_proof': None, 'live_dynamics_conservative_proposal': None, 'live_dynamics_stepwise_threshold_review_required': None, 'live_dynamics_max_threshold_step_db': None, 'live_dynamics_max_target_gain_reduction_db': None, 'live_dynamics_no_makeup_gain_required': None, 'live_dynamics_makeup_gain_db_limit': None, 'live_dynamics_gain_reduction_meter_readback_after_each_step_required': None, 'live_dynamics_previous_comp_model_readback_required': None, 'live_dynamics_operator_rehearsal_required': None, 'live_dynamics_live_promotion_status': None, 'live_dynamics_live_apply_blocked_without_wing_proof': None, 'review_only_parameter_keys': ['target_gain_reduction_db'], 'wing_dynamics_review_only_values': {'target_gain_reduction_db': 2.9}, 'wing_dynamics_review_only_parameter_reason_by_key': {'target_gain_reduction_db': 'target gain reduction is an operator/readback objective; WING Rack fw3.0.5 exposes compressor threshold/ratio/timing/knee/makeup/mix/envelope curve via OSC but no direct target gain-reduction write path'}, 'wing_dynamics_review_only_no_direct_osc_path': True, 'wing_dynamics_review_only_not_apply_ready': True, 'wing_dynamics_gain_reduction_objective_required': True, 'wing_dynamics_gain_reduction_objective_parameter_keys': ['target_gain_reduction_db'], 'wing_dynamics_gain_reduction_objective_values': {'target_gain_reduction_db': 2.9}, 'wing_dynamics_gain_reduction_target_db': 2.9, 'wing_dynamics_gain_reduction_max_db': None, 'wing_dynamics_gain_reduction_objective_tolerance_db': 1.0, 'wing_dynamics_gain_reduction_acceptance_min_db': 1.9, 'wing_dynamics_gain_reduction_acceptance_max_db': 3.9, 'wing_dynamics_gain_reduction_must_not_exceed_db': None, 'wing_dynamics_gain_reduction_objective_is_review_only': True, 'wing_dynamics_gain_reduction_objective_is_not_osc_write': True, 'wing_dynamics_gain_reduction_objective_no_direct_osc_write_path': True, 'wing_dynamics_gain_reduction_objective_apply_ready': False, 'wing_dynamics_gain_reduction_objective_requires_post_apply_meter_readback': True, 'wing_dynamics_gain_reduction_meter_readback_path_suffix': 'dyn/cpeak', 'wing_dynamics_gain_reduction_meter_readback_path_templates': ['/aux/{aux}/dyn/cpeak', '/aux/{aux}/dyn/cmode'], 'wing_dynamics_gain_reduction_meter_fallback_path_templates': [], 'wing_dynamics_gain_reduction_meter_value_type': 'read_only_aux_compressor_peak_percent_and_mode', 'wing_dynamics_gain_reduction_meter_is_not_apply_value': True, 'wing_dynamics_gain_reduction_meter_path_source': 'Docs/manuals/wing/WING Remote Protocols v3.0.5.pdf', 'wing_dynamics_gain_reduction_aux_depth_proxy': True, 'wing_dynamics_gain_reduction_objective_requires_operator_confirmation': True, 'wing_dynamics_gain_reduction_objective_live_promotion_status': 'blocked_until_actual_gain_reduction_meter_readback_and_operator_review', 'wing_dynamics_gain_reduction_objective_operator_check': 'confirm WING aux PSE/LA cpeak activity and cmode after the bounded depth/threshold proposal; aux exposes no dyn/gr meter', 'wing_dynamics_gain_reduction_objective_direct_osc_write_keys_excluded': ['target_gain_reduction_db'], 'wing_aux_dynamics_on_mapping_required': True, 'wing_aux_dynamics_on_osc_path_suffix': 'dyn/on', 'wing_aux_dynamics_on_osc_value': 1, 'wing_aux_dynamics_on_value_type': 'integer_bool', 'wing_aux_dynamics_on_readback_must_match_osc_value': True, 'wing_aux_dynamics_threshold_mapping_required': True, 'wing_aux_dynamics_threshold_osc_path_suffix': 'dyn/thr', 'wing_aux_dynamics_threshold_osc_value': -20.3, 'wing_aux_dynamics_threshold_value_type': 'float', 'wing_aux_dynamics_threshold_readback_must_match_osc_value': True, 'wing_aux_dynamics_depth_mapping_required': True, 'wing_aux_dynamics_depth_osc_path_suffix': 'dyn/depth', 'wing_aux_dynamics_depth_osc_value': 2.9, 'wing_aux_dynamics_depth_value_type': 'float', 'wing_aux_dynamics_depth_readback_must_match_osc_value': True, 'wing_aux_dynamics_fast_mapping_required': True, 'wing_aux_dynamics_fast_osc_path_suffix': 'dyn/fast', 'wing_aux_dynamics_fast_osc_value': 0, 'wing_aux_dynamics_fast_value_type': 'integer_bool', 'wing_aux_dynamics_fast_readback_must_match_osc_value': True}
- `soundcheck_aux_playback_pse_la_dynamics_selected_dynamics_wing_mix_value_out_of_range` (soundcheck_aux_playback_pse_la_dynamics): {'value': None, 'minimum': 0.0, 'maximum': 100.0}
- `soundcheck_aux_playback_pse_la_dynamics_selected_dynamics_wing_ratio_mapping_missing` (soundcheck_aux_playback_pse_la_dynamics): ['wing_compressor_ratio_numeric_value', 'wing_compressor_ratio_osc_path_suffix', 'wing_compressor_ratio_osc_value', 'wing_compressor_ratio_source', 'wing_compressor_ratio_target_surface', 'wing_compressor_ratio_value_type']
- `soundcheck_aux_playback_pse_la_dynamics_readback_contract_wing_ratio_mapping_not_required` (soundcheck_aux_playback_pse_la_dynamics): {'required': True, 'wing_osc_path_templates': ['/aux/{aux}/dyn/on', '/aux/{aux}/dyn/thr', '/aux/{aux}/dyn/depth', '/aux/{aux}/dyn/fast', '/aux/{aux}/dyn/peak', '/aux/{aux}/dyn/ingain', '/aux/{aux}/dyn/cpeak', '/aux/{aux}/dyn/cmode'], 'wing_rack_fw_3_0_5_native_osc_supported': True, 'dynamics_model_readback_required': False, 'dynamics_model_readback_must_equal': None, 'wing_compressor_model_readback_required': False, 'wing_compressor_model_readback_must_equal': None, 'wing_compressor_model_osc_path_suffix': None, 'wing_compressor_model_parameters_apply_only_when_model_matches': True, 'wing_aux_dynamics_pse_la_combo': True, 'wing_aux_dynamics_no_dyn_mdl_path': True, 'wing_aux_dynamics_no_ratio_attack_release_paths': True, 'wing_aux_dynamics_write_key_selection_policy': 'apply_only_selected_wing_fw_3_0_5_aux_pse_la_dynamics_value_keys', 'wing_aux_dynamics_apply_only_selected_write_keys': True, 'wing_aux_dynamics_selected_write_keys': ['wing_aux_dynamics_on_osc_value', 'wing_aux_dynamics_threshold_osc_value', 'wing_aux_dynamics_depth_osc_value', 'wing_aux_dynamics_fast_osc_value'], 'wing_aux_dynamics_forbidden_write_keys': ['wing_aux_dynamics_peak_osc_value'], 'wing_dynamics_required_model': None, 'wing_dynamics_current_model': None, 'wing_dynamics_current_enabled': None, 'wing_dynamics_current_operator_owned': None, 'wing_dynamics_model_readback_available': None, 'wing_dynamics_model_conflict': None, 'wing_dynamics_model_readback_required_for_active_slot': None, 'wing_dynamics_model_conflict_policy': None, 'wing_dynamics_model_conflict_blocker': None, 'wing_dynamics_model_readback_blocker': None, 'wing_dynamics_model_change_requires_operator_review': None, 'wing_dynamics_model_read_before_path': None, 'wing_dynamics_on_read_before_path': None, 'wing_dynamics_model_apply_only_when_current_model_matches': None, 'wing_dynamics_write_key_selection_policy': None, 'wing_dynamics_apply_only_selected_write_keys': None, 'wing_dynamics_selected_write_keys': None, 'wing_dynamics_forbidden_write_keys': None, 'read_before_paths': ['dyn/on', 'dyn/thr', 'dyn/depth', 'dyn/fast', 'dyn/peak', 'dyn/ingain', 'dyn/cpeak', 'dyn/cmode'], 'read_after_paths': ['dyn/on', 'dyn/thr', 'dyn/depth', 'dyn/fast', 'dyn/peak', 'dyn/ingain', 'dyn/cpeak', 'dyn/cmode'], 'makeup_gain_must_respect_headroom': True, 'soundcheck_dynamics_bounded_compression_proposal': True, 'soundcheck_dynamics_native_compressor_osc_only_required': True, 'soundcheck_dynamics_max_target_gain_reduction_db': 4.0, 'soundcheck_dynamics_max_makeup_gain_db': 1.0, 'soundcheck_dynamics_min_headroom_margin_db': 1.0, 'soundcheck_dynamics_required_true_peak_headroom_db': None, 'soundcheck_dynamics_required_summed_bus_true_peak_headroom_db': None, 'soundcheck_dynamics_target_gr_review_only': True, 'soundcheck_dynamics_target_gr_readback_required': True, 'soundcheck_dynamics_makeup_gain_headroom_required': True, 'soundcheck_dynamics_summed_bus_headroom_required': True, 'soundcheck_dynamics_compressor_model_readback_required': True, 'soundcheck_dynamics_gain_reduction_meter_readback_required': True, 'soundcheck_dynamics_feedback_recheck_required': True, 'soundcheck_dynamics_operator_confirmation_required': True, 'soundcheck_dynamics_readback_required': True, 'soundcheck_dynamics_rollback_ready': True, 'soundcheck_dynamics_live_promotion_status': 'blocked_until_real_wing_compressor_readback_gr_meter_headroom_feedback_recheck_and_operator_approval', 'soundcheck_dynamics_live_apply_blocked_without_wing_proof': True, 'soundcheck_master_processing_review_only_proposal': None, 'soundcheck_master_processing_native_osc_only_required': None, 'soundcheck_master_processing_target_surface': None, 'soundcheck_master_processing_types': None, 'soundcheck_master_processing_wing_apply_parameter_scope': None, 'soundcheck_master_processing_pa_feed_identity_required': None, 'soundcheck_master_processing_true_peak_headroom_recheck_required': None, 'soundcheck_master_processing_feedback_recheck_required': None, 'soundcheck_master_processing_bypass_level_match_required': None, 'soundcheck_master_processing_readback_required': None, 'soundcheck_master_processing_operator_confirmation_required': None, 'soundcheck_master_processing_rollback_ready': None, 'soundcheck_master_processing_no_routing_scene_snapshot_write_required': None, 'soundcheck_master_processing_bus_compression_max_target_gr_db': None, 'soundcheck_master_processing_limiter_max_target_gr_db': None, 'soundcheck_master_processing_live_promotion_status': None, 'soundcheck_master_processing_live_apply_blocked_without_wing_proof': None, 'live_dynamics_conservative_proposal': None, 'live_dynamics_stepwise_threshold_review_required': None, 'live_dynamics_max_threshold_step_db': None, 'live_dynamics_max_target_gain_reduction_db': None, 'live_dynamics_no_makeup_gain_required': None, 'live_dynamics_makeup_gain_db_limit': None, 'live_dynamics_gain_reduction_meter_readback_after_each_step_required': None, 'live_dynamics_previous_comp_model_readback_required': None, 'live_dynamics_operator_rehearsal_required': None, 'live_dynamics_live_promotion_status': None, 'live_dynamics_live_apply_blocked_without_wing_proof': None, 'review_only_parameter_keys': ['target_gain_reduction_db'], 'wing_dynamics_review_only_values': {'target_gain_reduction_db': 2.9}, 'wing_dynamics_review_only_parameter_reason_by_key': {'target_gain_reduction_db': 'target gain reduction is an operator/readback objective; WING Rack fw3.0.5 exposes compressor threshold/ratio/timing/knee/makeup/mix/envelope curve via OSC but no direct target gain-reduction write path'}, 'wing_dynamics_review_only_no_direct_osc_path': True, 'wing_dynamics_review_only_not_apply_ready': True, 'wing_dynamics_gain_reduction_objective_required': True, 'wing_dynamics_gain_reduction_objective_parameter_keys': ['target_gain_reduction_db'], 'wing_dynamics_gain_reduction_objective_values': {'target_gain_reduction_db': 2.9}, 'wing_dynamics_gain_reduction_target_db': 2.9, 'wing_dynamics_gain_reduction_max_db': None, 'wing_dynamics_gain_reduction_objective_tolerance_db': 1.0, 'wing_dynamics_gain_reduction_acceptance_min_db': 1.9, 'wing_dynamics_gain_reduction_acceptance_max_db': 3.9, 'wing_dynamics_gain_reduction_must_not_exceed_db': None, 'wing_dynamics_gain_reduction_objective_is_review_only': True, 'wing_dynamics_gain_reduction_objective_is_not_osc_write': True, 'wing_dynamics_gain_reduction_objective_no_direct_osc_write_path': True, 'wing_dynamics_gain_reduction_objective_apply_ready': False, 'wing_dynamics_gain_reduction_objective_requires_post_apply_meter_readback': True, 'wing_dynamics_gain_reduction_meter_readback_path_suffix': 'dyn/cpeak', 'wing_dynamics_gain_reduction_meter_readback_path_templates': ['/aux/{aux}/dyn/cpeak', '/aux/{aux}/dyn/cmode'], 'wing_dynamics_gain_reduction_meter_fallback_path_templates': [], 'wing_dynamics_gain_reduction_meter_value_type': 'read_only_aux_compressor_peak_percent_and_mode', 'wing_dynamics_gain_reduction_meter_is_not_apply_value': True, 'wing_dynamics_gain_reduction_meter_path_source': 'Docs/manuals/wing/WING Remote Protocols v3.0.5.pdf', 'wing_dynamics_gain_reduction_aux_depth_proxy': True, 'wing_dynamics_gain_reduction_objective_requires_operator_confirmation': True, 'wing_dynamics_gain_reduction_objective_live_promotion_status': 'blocked_until_actual_gain_reduction_meter_readback_and_operator_review', 'wing_dynamics_gain_reduction_objective_operator_check': 'confirm WING aux PSE/LA cpeak activity and cmode after the bounded depth/threshold proposal; aux exposes no dyn/gr meter', 'wing_dynamics_gain_reduction_objective_direct_osc_write_keys_excluded': ['target_gain_reduction_db'], 'wing_aux_dynamics_on_mapping_required': True, 'wing_aux_dynamics_on_osc_path_suffix': 'dyn/on', 'wing_aux_dynamics_on_osc_value': 1, 'wing_aux_dynamics_on_value_type': 'integer_bool', 'wing_aux_dynamics_on_readback_must_match_osc_value': True, 'wing_aux_dynamics_threshold_mapping_required': True, 'wing_aux_dynamics_threshold_osc_path_suffix': 'dyn/thr', 'wing_aux_dynamics_threshold_osc_value': -20.3, 'wing_aux_dynamics_threshold_value_type': 'float', 'wing_aux_dynamics_threshold_readback_must_match_osc_value': True, 'wing_aux_dynamics_depth_mapping_required': True, 'wing_aux_dynamics_depth_osc_path_suffix': 'dyn/depth', 'wing_aux_dynamics_depth_osc_value': 2.9, 'wing_aux_dynamics_depth_value_type': 'float', 'wing_aux_dynamics_depth_readback_must_match_osc_value': True, 'wing_aux_dynamics_fast_mapping_required': True, 'wing_aux_dynamics_fast_osc_path_suffix': 'dyn/fast', 'wing_aux_dynamics_fast_osc_value': 0, 'wing_aux_dynamics_fast_value_type': 'integer_bool', 'wing_aux_dynamics_fast_readback_must_match_osc_value': True}
- `soundcheck_aux_playback_pse_la_dynamics_readback_contract_wing_ratio_path_mismatch` (soundcheck_aux_playback_pse_la_dynamics): {'required': True, 'wing_osc_path_templates': ['/aux/{aux}/dyn/on', '/aux/{aux}/dyn/thr', '/aux/{aux}/dyn/depth', '/aux/{aux}/dyn/fast', '/aux/{aux}/dyn/peak', '/aux/{aux}/dyn/ingain', '/aux/{aux}/dyn/cpeak', '/aux/{aux}/dyn/cmode'], 'wing_rack_fw_3_0_5_native_osc_supported': True, 'dynamics_model_readback_required': False, 'dynamics_model_readback_must_equal': None, 'wing_compressor_model_readback_required': False, 'wing_compressor_model_readback_must_equal': None, 'wing_compressor_model_osc_path_suffix': None, 'wing_compressor_model_parameters_apply_only_when_model_matches': True, 'wing_aux_dynamics_pse_la_combo': True, 'wing_aux_dynamics_no_dyn_mdl_path': True, 'wing_aux_dynamics_no_ratio_attack_release_paths': True, 'wing_aux_dynamics_write_key_selection_policy': 'apply_only_selected_wing_fw_3_0_5_aux_pse_la_dynamics_value_keys', 'wing_aux_dynamics_apply_only_selected_write_keys': True, 'wing_aux_dynamics_selected_write_keys': ['wing_aux_dynamics_on_osc_value', 'wing_aux_dynamics_threshold_osc_value', 'wing_aux_dynamics_depth_osc_value', 'wing_aux_dynamics_fast_osc_value'], 'wing_aux_dynamics_forbidden_write_keys': ['wing_aux_dynamics_peak_osc_value'], 'wing_dynamics_required_model': None, 'wing_dynamics_current_model': None, 'wing_dynamics_current_enabled': None, 'wing_dynamics_current_operator_owned': None, 'wing_dynamics_model_readback_available': None, 'wing_dynamics_model_conflict': None, 'wing_dynamics_model_readback_required_for_active_slot': None, 'wing_dynamics_model_conflict_policy': None, 'wing_dynamics_model_conflict_blocker': None, 'wing_dynamics_model_readback_blocker': None, 'wing_dynamics_model_change_requires_operator_review': None, 'wing_dynamics_model_read_before_path': None, 'wing_dynamics_on_read_before_path': None, 'wing_dynamics_model_apply_only_when_current_model_matches': None, 'wing_dynamics_write_key_selection_policy': None, 'wing_dynamics_apply_only_selected_write_keys': None, 'wing_dynamics_selected_write_keys': None, 'wing_dynamics_forbidden_write_keys': None, 'read_before_paths': ['dyn/on', 'dyn/thr', 'dyn/depth', 'dyn/fast', 'dyn/peak', 'dyn/ingain', 'dyn/cpeak', 'dyn/cmode'], 'read_after_paths': ['dyn/on', 'dyn/thr', 'dyn/depth', 'dyn/fast', 'dyn/peak', 'dyn/ingain', 'dyn/cpeak', 'dyn/cmode'], 'makeup_gain_must_respect_headroom': True, 'soundcheck_dynamics_bounded_compression_proposal': True, 'soundcheck_dynamics_native_compressor_osc_only_required': True, 'soundcheck_dynamics_max_target_gain_reduction_db': 4.0, 'soundcheck_dynamics_max_makeup_gain_db': 1.0, 'soundcheck_dynamics_min_headroom_margin_db': 1.0, 'soundcheck_dynamics_required_true_peak_headroom_db': None, 'soundcheck_dynamics_required_summed_bus_true_peak_headroom_db': None, 'soundcheck_dynamics_target_gr_review_only': True, 'soundcheck_dynamics_target_gr_readback_required': True, 'soundcheck_dynamics_makeup_gain_headroom_required': True, 'soundcheck_dynamics_summed_bus_headroom_required': True, 'soundcheck_dynamics_compressor_model_readback_required': True, 'soundcheck_dynamics_gain_reduction_meter_readback_required': True, 'soundcheck_dynamics_feedback_recheck_required': True, 'soundcheck_dynamics_operator_confirmation_required': True, 'soundcheck_dynamics_readback_required': True, 'soundcheck_dynamics_rollback_ready': True, 'soundcheck_dynamics_live_promotion_status': 'blocked_until_real_wing_compressor_readback_gr_meter_headroom_feedback_recheck_and_operator_approval', 'soundcheck_dynamics_live_apply_blocked_without_wing_proof': True, 'soundcheck_master_processing_review_only_proposal': None, 'soundcheck_master_processing_native_osc_only_required': None, 'soundcheck_master_processing_target_surface': None, 'soundcheck_master_processing_types': None, 'soundcheck_master_processing_wing_apply_parameter_scope': None, 'soundcheck_master_processing_pa_feed_identity_required': None, 'soundcheck_master_processing_true_peak_headroom_recheck_required': None, 'soundcheck_master_processing_feedback_recheck_required': None, 'soundcheck_master_processing_bypass_level_match_required': None, 'soundcheck_master_processing_readback_required': None, 'soundcheck_master_processing_operator_confirmation_required': None, 'soundcheck_master_processing_rollback_ready': None, 'soundcheck_master_processing_no_routing_scene_snapshot_write_required': None, 'soundcheck_master_processing_bus_compression_max_target_gr_db': None, 'soundcheck_master_processing_limiter_max_target_gr_db': None, 'soundcheck_master_processing_live_promotion_status': None, 'soundcheck_master_processing_live_apply_blocked_without_wing_proof': None, 'live_dynamics_conservative_proposal': None, 'live_dynamics_stepwise_threshold_review_required': None, 'live_dynamics_max_threshold_step_db': None, 'live_dynamics_max_target_gain_reduction_db': None, 'live_dynamics_no_makeup_gain_required': None, 'live_dynamics_makeup_gain_db_limit': None, 'live_dynamics_gain_reduction_meter_readback_after_each_step_required': None, 'live_dynamics_previous_comp_model_readback_required': None, 'live_dynamics_operator_rehearsal_required': None, 'live_dynamics_live_promotion_status': None, 'live_dynamics_live_apply_blocked_without_wing_proof': None, 'review_only_parameter_keys': ['target_gain_reduction_db'], 'wing_dynamics_review_only_values': {'target_gain_reduction_db': 2.9}, 'wing_dynamics_review_only_parameter_reason_by_key': {'target_gain_reduction_db': 'target gain reduction is an operator/readback objective; WING Rack fw3.0.5 exposes compressor threshold/ratio/timing/knee/makeup/mix/envelope curve via OSC but no direct target gain-reduction write path'}, 'wing_dynamics_review_only_no_direct_osc_path': True, 'wing_dynamics_review_only_not_apply_ready': True, 'wing_dynamics_gain_reduction_objective_required': True, 'wing_dynamics_gain_reduction_objective_parameter_keys': ['target_gain_reduction_db'], 'wing_dynamics_gain_reduction_objective_values': {'target_gain_reduction_db': 2.9}, 'wing_dynamics_gain_reduction_target_db': 2.9, 'wing_dynamics_gain_reduction_max_db': None, 'wing_dynamics_gain_reduction_objective_tolerance_db': 1.0, 'wing_dynamics_gain_reduction_acceptance_min_db': 1.9, 'wing_dynamics_gain_reduction_acceptance_max_db': 3.9, 'wing_dynamics_gain_reduction_must_not_exceed_db': None, 'wing_dynamics_gain_reduction_objective_is_review_only': True, 'wing_dynamics_gain_reduction_objective_is_not_osc_write': True, 'wing_dynamics_gain_reduction_objective_no_direct_osc_write_path': True, 'wing_dynamics_gain_reduction_objective_apply_ready': False, 'wing_dynamics_gain_reduction_objective_requires_post_apply_meter_readback': True, 'wing_dynamics_gain_reduction_meter_readback_path_suffix': 'dyn/cpeak', 'wing_dynamics_gain_reduction_meter_readback_path_templates': ['/aux/{aux}/dyn/cpeak', '/aux/{aux}/dyn/cmode'], 'wing_dynamics_gain_reduction_meter_fallback_path_templates': [], 'wing_dynamics_gain_reduction_meter_value_type': 'read_only_aux_compressor_peak_percent_and_mode', 'wing_dynamics_gain_reduction_meter_is_not_apply_value': True, 'wing_dynamics_gain_reduction_meter_path_source': 'Docs/manuals/wing/WING Remote Protocols v3.0.5.pdf', 'wing_dynamics_gain_reduction_aux_depth_proxy': True, 'wing_dynamics_gain_reduction_objective_requires_operator_confirmation': True, 'wing_dynamics_gain_reduction_objective_live_promotion_status': 'blocked_until_actual_gain_reduction_meter_readback_and_operator_review', 'wing_dynamics_gain_reduction_objective_operator_check': 'confirm WING aux PSE/LA cpeak activity and cmode after the bounded depth/threshold proposal; aux exposes no dyn/gr meter', 'wing_dynamics_gain_reduction_objective_direct_osc_write_keys_excluded': ['target_gain_reduction_db'], 'wing_aux_dynamics_on_mapping_required': True, 'wing_aux_dynamics_on_osc_path_suffix': 'dyn/on', 'wing_aux_dynamics_on_osc_value': 1, 'wing_aux_dynamics_on_value_type': 'integer_bool', 'wing_aux_dynamics_on_readback_must_match_osc_value': True, 'wing_aux_dynamics_threshold_mapping_required': True, 'wing_aux_dynamics_threshold_osc_path_suffix': 'dyn/thr', 'wing_aux_dynamics_threshold_osc_value': -20.3, 'wing_aux_dynamics_threshold_value_type': 'float', 'wing_aux_dynamics_threshold_readback_must_match_osc_value': True, 'wing_aux_dynamics_depth_mapping_required': True, 'wing_aux_dynamics_depth_osc_path_suffix': 'dyn/depth', 'wing_aux_dynamics_depth_osc_value': 2.9, 'wing_aux_dynamics_depth_value_type': 'float', 'wing_aux_dynamics_depth_readback_must_match_osc_value': True, 'wing_aux_dynamics_fast_mapping_required': True, 'wing_aux_dynamics_fast_osc_path_suffix': 'dyn/fast', 'wing_aux_dynamics_fast_osc_value': 0, 'wing_aux_dynamics_fast_value_type': 'integer_bool', 'wing_aux_dynamics_fast_readback_must_match_osc_value': True}
- `soundcheck_aux_playback_pse_la_dynamics_readback_contract_wing_ratio_readback_missing` (soundcheck_aux_playback_pse_la_dynamics): {'required': True, 'wing_osc_path_templates': ['/aux/{aux}/dyn/on', '/aux/{aux}/dyn/thr', '/aux/{aux}/dyn/depth', '/aux/{aux}/dyn/fast', '/aux/{aux}/dyn/peak', '/aux/{aux}/dyn/ingain', '/aux/{aux}/dyn/cpeak', '/aux/{aux}/dyn/cmode'], 'wing_rack_fw_3_0_5_native_osc_supported': True, 'dynamics_model_readback_required': False, 'dynamics_model_readback_must_equal': None, 'wing_compressor_model_readback_required': False, 'wing_compressor_model_readback_must_equal': None, 'wing_compressor_model_osc_path_suffix': None, 'wing_compressor_model_parameters_apply_only_when_model_matches': True, 'wing_aux_dynamics_pse_la_combo': True, 'wing_aux_dynamics_no_dyn_mdl_path': True, 'wing_aux_dynamics_no_ratio_attack_release_paths': True, 'wing_aux_dynamics_write_key_selection_policy': 'apply_only_selected_wing_fw_3_0_5_aux_pse_la_dynamics_value_keys', 'wing_aux_dynamics_apply_only_selected_write_keys': True, 'wing_aux_dynamics_selected_write_keys': ['wing_aux_dynamics_on_osc_value', 'wing_aux_dynamics_threshold_osc_value', 'wing_aux_dynamics_depth_osc_value', 'wing_aux_dynamics_fast_osc_value'], 'wing_aux_dynamics_forbidden_write_keys': ['wing_aux_dynamics_peak_osc_value'], 'wing_dynamics_required_model': None, 'wing_dynamics_current_model': None, 'wing_dynamics_current_enabled': None, 'wing_dynamics_current_operator_owned': None, 'wing_dynamics_model_readback_available': None, 'wing_dynamics_model_conflict': None, 'wing_dynamics_model_readback_required_for_active_slot': None, 'wing_dynamics_model_conflict_policy': None, 'wing_dynamics_model_conflict_blocker': None, 'wing_dynamics_model_readback_blocker': None, 'wing_dynamics_model_change_requires_operator_review': None, 'wing_dynamics_model_read_before_path': None, 'wing_dynamics_on_read_before_path': None, 'wing_dynamics_model_apply_only_when_current_model_matches': None, 'wing_dynamics_write_key_selection_policy': None, 'wing_dynamics_apply_only_selected_write_keys': None, 'wing_dynamics_selected_write_keys': None, 'wing_dynamics_forbidden_write_keys': None, 'read_before_paths': ['dyn/on', 'dyn/thr', 'dyn/depth', 'dyn/fast', 'dyn/peak', 'dyn/ingain', 'dyn/cpeak', 'dyn/cmode'], 'read_after_paths': ['dyn/on', 'dyn/thr', 'dyn/depth', 'dyn/fast', 'dyn/peak', 'dyn/ingain', 'dyn/cpeak', 'dyn/cmode'], 'makeup_gain_must_respect_headroom': True, 'soundcheck_dynamics_bounded_compression_proposal': True, 'soundcheck_dynamics_native_compressor_osc_only_required': True, 'soundcheck_dynamics_max_target_gain_reduction_db': 4.0, 'soundcheck_dynamics_max_makeup_gain_db': 1.0, 'soundcheck_dynamics_min_headroom_margin_db': 1.0, 'soundcheck_dynamics_required_true_peak_headroom_db': None, 'soundcheck_dynamics_required_summed_bus_true_peak_headroom_db': None, 'soundcheck_dynamics_target_gr_review_only': True, 'soundcheck_dynamics_target_gr_readback_required': True, 'soundcheck_dynamics_makeup_gain_headroom_required': True, 'soundcheck_dynamics_summed_bus_headroom_required': True, 'soundcheck_dynamics_compressor_model_readback_required': True, 'soundcheck_dynamics_gain_reduction_meter_readback_required': True, 'soundcheck_dynamics_feedback_recheck_required': True, 'soundcheck_dynamics_operator_confirmation_required': True, 'soundcheck_dynamics_readback_required': True, 'soundcheck_dynamics_rollback_ready': True, 'soundcheck_dynamics_live_promotion_status': 'blocked_until_real_wing_compressor_readback_gr_meter_headroom_feedback_recheck_and_operator_approval', 'soundcheck_dynamics_live_apply_blocked_without_wing_proof': True, 'soundcheck_master_processing_review_only_proposal': None, 'soundcheck_master_processing_native_osc_only_required': None, 'soundcheck_master_processing_target_surface': None, 'soundcheck_master_processing_types': None, 'soundcheck_master_processing_wing_apply_parameter_scope': None, 'soundcheck_master_processing_pa_feed_identity_required': None, 'soundcheck_master_processing_true_peak_headroom_recheck_required': None, 'soundcheck_master_processing_feedback_recheck_required': None, 'soundcheck_master_processing_bypass_level_match_required': None, 'soundcheck_master_processing_readback_required': None, 'soundcheck_master_processing_operator_confirmation_required': None, 'soundcheck_master_processing_rollback_ready': None, 'soundcheck_master_processing_no_routing_scene_snapshot_write_required': None, 'soundcheck_master_processing_bus_compression_max_target_gr_db': None, 'soundcheck_master_processing_limiter_max_target_gr_db': None, 'soundcheck_master_processing_live_promotion_status': None, 'soundcheck_master_processing_live_apply_blocked_without_wing_proof': None, 'live_dynamics_conservative_proposal': None, 'live_dynamics_stepwise_threshold_review_required': None, 'live_dynamics_max_threshold_step_db': None, 'live_dynamics_max_target_gain_reduction_db': None, 'live_dynamics_no_makeup_gain_required': None, 'live_dynamics_makeup_gain_db_limit': None, 'live_dynamics_gain_reduction_meter_readback_after_each_step_required': None, 'live_dynamics_previous_comp_model_readback_required': None, 'live_dynamics_operator_rehearsal_required': None, 'live_dynamics_live_promotion_status': None, 'live_dynamics_live_apply_blocked_without_wing_proof': None, 'review_only_parameter_keys': ['target_gain_reduction_db'], 'wing_dynamics_review_only_values': {'target_gain_reduction_db': 2.9}, 'wing_dynamics_review_only_parameter_reason_by_key': {'target_gain_reduction_db': 'target gain reduction is an operator/readback objective; WING Rack fw3.0.5 exposes compressor threshold/ratio/timing/knee/makeup/mix/envelope curve via OSC but no direct target gain-reduction write path'}, 'wing_dynamics_review_only_no_direct_osc_path': True, 'wing_dynamics_review_only_not_apply_ready': True, 'wing_dynamics_gain_reduction_objective_required': True, 'wing_dynamics_gain_reduction_objective_parameter_keys': ['target_gain_reduction_db'], 'wing_dynamics_gain_reduction_objective_values': {'target_gain_reduction_db': 2.9}, 'wing_dynamics_gain_reduction_target_db': 2.9, 'wing_dynamics_gain_reduction_max_db': None, 'wing_dynamics_gain_reduction_objective_tolerance_db': 1.0, 'wing_dynamics_gain_reduction_acceptance_min_db': 1.9, 'wing_dynamics_gain_reduction_acceptance_max_db': 3.9, 'wing_dynamics_gain_reduction_must_not_exceed_db': None, 'wing_dynamics_gain_reduction_objective_is_review_only': True, 'wing_dynamics_gain_reduction_objective_is_not_osc_write': True, 'wing_dynamics_gain_reduction_objective_no_direct_osc_write_path': True, 'wing_dynamics_gain_reduction_objective_apply_ready': False, 'wing_dynamics_gain_reduction_objective_requires_post_apply_meter_readback': True, 'wing_dynamics_gain_reduction_meter_readback_path_suffix': 'dyn/cpeak', 'wing_dynamics_gain_reduction_meter_readback_path_templates': ['/aux/{aux}/dyn/cpeak', '/aux/{aux}/dyn/cmode'], 'wing_dynamics_gain_reduction_meter_fallback_path_templates': [], 'wing_dynamics_gain_reduction_meter_value_type': 'read_only_aux_compressor_peak_percent_and_mode', 'wing_dynamics_gain_reduction_meter_is_not_apply_value': True, 'wing_dynamics_gain_reduction_meter_path_source': 'Docs/manuals/wing/WING Remote Protocols v3.0.5.pdf', 'wing_dynamics_gain_reduction_aux_depth_proxy': True, 'wing_dynamics_gain_reduction_objective_requires_operator_confirmation': True, 'wing_dynamics_gain_reduction_objective_live_promotion_status': 'blocked_until_actual_gain_reduction_meter_readback_and_operator_review', 'wing_dynamics_gain_reduction_objective_operator_check': 'confirm WING aux PSE/LA cpeak activity and cmode after the bounded depth/threshold proposal; aux exposes no dyn/gr meter', 'wing_dynamics_gain_reduction_objective_direct_osc_write_keys_excluded': ['target_gain_reduction_db'], 'wing_aux_dynamics_on_mapping_required': True, 'wing_aux_dynamics_on_osc_path_suffix': 'dyn/on', 'wing_aux_dynamics_on_osc_value': 1, 'wing_aux_dynamics_on_value_type': 'integer_bool', 'wing_aux_dynamics_on_readback_must_match_osc_value': True, 'wing_aux_dynamics_threshold_mapping_required': True, 'wing_aux_dynamics_threshold_osc_path_suffix': 'dyn/thr', 'wing_aux_dynamics_threshold_osc_value': -20.3, 'wing_aux_dynamics_threshold_value_type': 'float', 'wing_aux_dynamics_threshold_readback_must_match_osc_value': True, 'wing_aux_dynamics_depth_mapping_required': True, 'wing_aux_dynamics_depth_osc_path_suffix': 'dyn/depth', 'wing_aux_dynamics_depth_osc_value': 2.9, 'wing_aux_dynamics_depth_value_type': 'float', 'wing_aux_dynamics_depth_readback_must_match_osc_value': True, 'wing_aux_dynamics_fast_mapping_required': True, 'wing_aux_dynamics_fast_osc_path_suffix': 'dyn/fast', 'wing_aux_dynamics_fast_osc_value': 0, 'wing_aux_dynamics_fast_value_type': 'integer_bool', 'wing_aux_dynamics_fast_readback_must_match_osc_value': True}
- `soundcheck_aux_playback_pse_la_dynamics_selected_dynamics_wing_envelope_curve_mapping_missing` (soundcheck_aux_playback_pse_la_dynamics): ['wing_compressor_envelope_curve_mapping_policy', 'wing_compressor_envelope_curve_original_value', 'wing_compressor_envelope_curve_osc_path_suffix', 'wing_compressor_envelope_curve_osc_value', 'wing_compressor_envelope_curve_source', 'wing_compressor_envelope_curve_value_type']
- `soundcheck_aux_playback_pse_la_dynamics_readback_contract_wing_envelope_curve_mapping_not_required` (soundcheck_aux_playback_pse_la_dynamics): {'required': True, 'wing_osc_path_templates': ['/aux/{aux}/dyn/on', '/aux/{aux}/dyn/thr', '/aux/{aux}/dyn/depth', '/aux/{aux}/dyn/fast', '/aux/{aux}/dyn/peak', '/aux/{aux}/dyn/ingain', '/aux/{aux}/dyn/cpeak', '/aux/{aux}/dyn/cmode'], 'wing_rack_fw_3_0_5_native_osc_supported': True, 'dynamics_model_readback_required': False, 'dynamics_model_readback_must_equal': None, 'wing_compressor_model_readback_required': False, 'wing_compressor_model_readback_must_equal': None, 'wing_compressor_model_osc_path_suffix': None, 'wing_compressor_model_parameters_apply_only_when_model_matches': True, 'wing_aux_dynamics_pse_la_combo': True, 'wing_aux_dynamics_no_dyn_mdl_path': True, 'wing_aux_dynamics_no_ratio_attack_release_paths': True, 'wing_aux_dynamics_write_key_selection_policy': 'apply_only_selected_wing_fw_3_0_5_aux_pse_la_dynamics_value_keys', 'wing_aux_dynamics_apply_only_selected_write_keys': True, 'wing_aux_dynamics_selected_write_keys': ['wing_aux_dynamics_on_osc_value', 'wing_aux_dynamics_threshold_osc_value', 'wing_aux_dynamics_depth_osc_value', 'wing_aux_dynamics_fast_osc_value'], 'wing_aux_dynamics_forbidden_write_keys': ['wing_aux_dynamics_peak_osc_value'], 'wing_dynamics_required_model': None, 'wing_dynamics_current_model': None, 'wing_dynamics_current_enabled': None, 'wing_dynamics_current_operator_owned': None, 'wing_dynamics_model_readback_available': None, 'wing_dynamics_model_conflict': None, 'wing_dynamics_model_readback_required_for_active_slot': None, 'wing_dynamics_model_conflict_policy': None, 'wing_dynamics_model_conflict_blocker': None, 'wing_dynamics_model_readback_blocker': None, 'wing_dynamics_model_change_requires_operator_review': None, 'wing_dynamics_model_read_before_path': None, 'wing_dynamics_on_read_before_path': None, 'wing_dynamics_model_apply_only_when_current_model_matches': None, 'wing_dynamics_write_key_selection_policy': None, 'wing_dynamics_apply_only_selected_write_keys': None, 'wing_dynamics_selected_write_keys': None, 'wing_dynamics_forbidden_write_keys': None, 'read_before_paths': ['dyn/on', 'dyn/thr', 'dyn/depth', 'dyn/fast', 'dyn/peak', 'dyn/ingain', 'dyn/cpeak', 'dyn/cmode'], 'read_after_paths': ['dyn/on', 'dyn/thr', 'dyn/depth', 'dyn/fast', 'dyn/peak', 'dyn/ingain', 'dyn/cpeak', 'dyn/cmode'], 'makeup_gain_must_respect_headroom': True, 'soundcheck_dynamics_bounded_compression_proposal': True, 'soundcheck_dynamics_native_compressor_osc_only_required': True, 'soundcheck_dynamics_max_target_gain_reduction_db': 4.0, 'soundcheck_dynamics_max_makeup_gain_db': 1.0, 'soundcheck_dynamics_min_headroom_margin_db': 1.0, 'soundcheck_dynamics_required_true_peak_headroom_db': None, 'soundcheck_dynamics_required_summed_bus_true_peak_headroom_db': None, 'soundcheck_dynamics_target_gr_review_only': True, 'soundcheck_dynamics_target_gr_readback_required': True, 'soundcheck_dynamics_makeup_gain_headroom_required': True, 'soundcheck_dynamics_summed_bus_headroom_required': True, 'soundcheck_dynamics_compressor_model_readback_required': True, 'soundcheck_dynamics_gain_reduction_meter_readback_required': True, 'soundcheck_dynamics_feedback_recheck_required': True, 'soundcheck_dynamics_operator_confirmation_required': True, 'soundcheck_dynamics_readback_required': True, 'soundcheck_dynamics_rollback_ready': True, 'soundcheck_dynamics_live_promotion_status': 'blocked_until_real_wing_compressor_readback_gr_meter_headroom_feedback_recheck_and_operator_approval', 'soundcheck_dynamics_live_apply_blocked_without_wing_proof': True, 'soundcheck_master_processing_review_only_proposal': None, 'soundcheck_master_processing_native_osc_only_required': None, 'soundcheck_master_processing_target_surface': None, 'soundcheck_master_processing_types': None, 'soundcheck_master_processing_wing_apply_parameter_scope': None, 'soundcheck_master_processing_pa_feed_identity_required': None, 'soundcheck_master_processing_true_peak_headroom_recheck_required': None, 'soundcheck_master_processing_feedback_recheck_required': None, 'soundcheck_master_processing_bypass_level_match_required': None, 'soundcheck_master_processing_readback_required': None, 'soundcheck_master_processing_operator_confirmation_required': None, 'soundcheck_master_processing_rollback_ready': None, 'soundcheck_master_processing_no_routing_scene_snapshot_write_required': None, 'soundcheck_master_processing_bus_compression_max_target_gr_db': None, 'soundcheck_master_processing_limiter_max_target_gr_db': None, 'soundcheck_master_processing_live_promotion_status': None, 'soundcheck_master_processing_live_apply_blocked_without_wing_proof': None, 'live_dynamics_conservative_proposal': None, 'live_dynamics_stepwise_threshold_review_required': None, 'live_dynamics_max_threshold_step_db': None, 'live_dynamics_max_target_gain_reduction_db': None, 'live_dynamics_no_makeup_gain_required': None, 'live_dynamics_makeup_gain_db_limit': None, 'live_dynamics_gain_reduction_meter_readback_after_each_step_required': None, 'live_dynamics_previous_comp_model_readback_required': None, 'live_dynamics_operator_rehearsal_required': None, 'live_dynamics_live_promotion_status': None, 'live_dynamics_live_apply_blocked_without_wing_proof': None, 'review_only_parameter_keys': ['target_gain_reduction_db'], 'wing_dynamics_review_only_values': {'target_gain_reduction_db': 2.9}, 'wing_dynamics_review_only_parameter_reason_by_key': {'target_gain_reduction_db': 'target gain reduction is an operator/readback objective; WING Rack fw3.0.5 exposes compressor threshold/ratio/timing/knee/makeup/mix/envelope curve via OSC but no direct target gain-reduction write path'}, 'wing_dynamics_review_only_no_direct_osc_path': True, 'wing_dynamics_review_only_not_apply_ready': True, 'wing_dynamics_gain_reduction_objective_required': True, 'wing_dynamics_gain_reduction_objective_parameter_keys': ['target_gain_reduction_db'], 'wing_dynamics_gain_reduction_objective_values': {'target_gain_reduction_db': 2.9}, 'wing_dynamics_gain_reduction_target_db': 2.9, 'wing_dynamics_gain_reduction_max_db': None, 'wing_dynamics_gain_reduction_objective_tolerance_db': 1.0, 'wing_dynamics_gain_reduction_acceptance_min_db': 1.9, 'wing_dynamics_gain_reduction_acceptance_max_db': 3.9, 'wing_dynamics_gain_reduction_must_not_exceed_db': None, 'wing_dynamics_gain_reduction_objective_is_review_only': True, 'wing_dynamics_gain_reduction_objective_is_not_osc_write': True, 'wing_dynamics_gain_reduction_objective_no_direct_osc_write_path': True, 'wing_dynamics_gain_reduction_objective_apply_ready': False, 'wing_dynamics_gain_reduction_objective_requires_post_apply_meter_readback': True, 'wing_dynamics_gain_reduction_meter_readback_path_suffix': 'dyn/cpeak', 'wing_dynamics_gain_reduction_meter_readback_path_templates': ['/aux/{aux}/dyn/cpeak', '/aux/{aux}/dyn/cmode'], 'wing_dynamics_gain_reduction_meter_fallback_path_templates': [], 'wing_dynamics_gain_reduction_meter_value_type': 'read_only_aux_compressor_peak_percent_and_mode', 'wing_dynamics_gain_reduction_meter_is_not_apply_value': True, 'wing_dynamics_gain_reduction_meter_path_source': 'Docs/manuals/wing/WING Remote Protocols v3.0.5.pdf', 'wing_dynamics_gain_reduction_aux_depth_proxy': True, 'wing_dynamics_gain_reduction_objective_requires_operator_confirmation': True, 'wing_dynamics_gain_reduction_objective_live_promotion_status': 'blocked_until_actual_gain_reduction_meter_readback_and_operator_review', 'wing_dynamics_gain_reduction_objective_operator_check': 'confirm WING aux PSE/LA cpeak activity and cmode after the bounded depth/threshold proposal; aux exposes no dyn/gr meter', 'wing_dynamics_gain_reduction_objective_direct_osc_write_keys_excluded': ['target_gain_reduction_db'], 'wing_aux_dynamics_on_mapping_required': True, 'wing_aux_dynamics_on_osc_path_suffix': 'dyn/on', 'wing_aux_dynamics_on_osc_value': 1, 'wing_aux_dynamics_on_value_type': 'integer_bool', 'wing_aux_dynamics_on_readback_must_match_osc_value': True, 'wing_aux_dynamics_threshold_mapping_required': True, 'wing_aux_dynamics_threshold_osc_path_suffix': 'dyn/thr', 'wing_aux_dynamics_threshold_osc_value': -20.3, 'wing_aux_dynamics_threshold_value_type': 'float', 'wing_aux_dynamics_threshold_readback_must_match_osc_value': True, 'wing_aux_dynamics_depth_mapping_required': True, 'wing_aux_dynamics_depth_osc_path_suffix': 'dyn/depth', 'wing_aux_dynamics_depth_osc_value': 2.9, 'wing_aux_dynamics_depth_value_type': 'float', 'wing_aux_dynamics_depth_readback_must_match_osc_value': True, 'wing_aux_dynamics_fast_mapping_required': True, 'wing_aux_dynamics_fast_osc_path_suffix': 'dyn/fast', 'wing_aux_dynamics_fast_osc_value': 0, 'wing_aux_dynamics_fast_value_type': 'integer_bool', 'wing_aux_dynamics_fast_readback_must_match_osc_value': True}
- `soundcheck_aux_playback_pse_la_dynamics_readback_contract_wing_envelope_curve_path_mismatch` (soundcheck_aux_playback_pse_la_dynamics): {'required': True, 'wing_osc_path_templates': ['/aux/{aux}/dyn/on', '/aux/{aux}/dyn/thr', '/aux/{aux}/dyn/depth', '/aux/{aux}/dyn/fast', '/aux/{aux}/dyn/peak', '/aux/{aux}/dyn/ingain', '/aux/{aux}/dyn/cpeak', '/aux/{aux}/dyn/cmode'], 'wing_rack_fw_3_0_5_native_osc_supported': True, 'dynamics_model_readback_required': False, 'dynamics_model_readback_must_equal': None, 'wing_compressor_model_readback_required': False, 'wing_compressor_model_readback_must_equal': None, 'wing_compressor_model_osc_path_suffix': None, 'wing_compressor_model_parameters_apply_only_when_model_matches': True, 'wing_aux_dynamics_pse_la_combo': True, 'wing_aux_dynamics_no_dyn_mdl_path': True, 'wing_aux_dynamics_no_ratio_attack_release_paths': True, 'wing_aux_dynamics_write_key_selection_policy': 'apply_only_selected_wing_fw_3_0_5_aux_pse_la_dynamics_value_keys', 'wing_aux_dynamics_apply_only_selected_write_keys': True, 'wing_aux_dynamics_selected_write_keys': ['wing_aux_dynamics_on_osc_value', 'wing_aux_dynamics_threshold_osc_value', 'wing_aux_dynamics_depth_osc_value', 'wing_aux_dynamics_fast_osc_value'], 'wing_aux_dynamics_forbidden_write_keys': ['wing_aux_dynamics_peak_osc_value'], 'wing_dynamics_required_model': None, 'wing_dynamics_current_model': None, 'wing_dynamics_current_enabled': None, 'wing_dynamics_current_operator_owned': None, 'wing_dynamics_model_readback_available': None, 'wing_dynamics_model_conflict': None, 'wing_dynamics_model_readback_required_for_active_slot': None, 'wing_dynamics_model_conflict_policy': None, 'wing_dynamics_model_conflict_blocker': None, 'wing_dynamics_model_readback_blocker': None, 'wing_dynamics_model_change_requires_operator_review': None, 'wing_dynamics_model_read_before_path': None, 'wing_dynamics_on_read_before_path': None, 'wing_dynamics_model_apply_only_when_current_model_matches': None, 'wing_dynamics_write_key_selection_policy': None, 'wing_dynamics_apply_only_selected_write_keys': None, 'wing_dynamics_selected_write_keys': None, 'wing_dynamics_forbidden_write_keys': None, 'read_before_paths': ['dyn/on', 'dyn/thr', 'dyn/depth', 'dyn/fast', 'dyn/peak', 'dyn/ingain', 'dyn/cpeak', 'dyn/cmode'], 'read_after_paths': ['dyn/on', 'dyn/thr', 'dyn/depth', 'dyn/fast', 'dyn/peak', 'dyn/ingain', 'dyn/cpeak', 'dyn/cmode'], 'makeup_gain_must_respect_headroom': True, 'soundcheck_dynamics_bounded_compression_proposal': True, 'soundcheck_dynamics_native_compressor_osc_only_required': True, 'soundcheck_dynamics_max_target_gain_reduction_db': 4.0, 'soundcheck_dynamics_max_makeup_gain_db': 1.0, 'soundcheck_dynamics_min_headroom_margin_db': 1.0, 'soundcheck_dynamics_required_true_peak_headroom_db': None, 'soundcheck_dynamics_required_summed_bus_true_peak_headroom_db': None, 'soundcheck_dynamics_target_gr_review_only': True, 'soundcheck_dynamics_target_gr_readback_required': True, 'soundcheck_dynamics_makeup_gain_headroom_required': True, 'soundcheck_dynamics_summed_bus_headroom_required': True, 'soundcheck_dynamics_compressor_model_readback_required': True, 'soundcheck_dynamics_gain_reduction_meter_readback_required': True, 'soundcheck_dynamics_feedback_recheck_required': True, 'soundcheck_dynamics_operator_confirmation_required': True, 'soundcheck_dynamics_readback_required': True, 'soundcheck_dynamics_rollback_ready': True, 'soundcheck_dynamics_live_promotion_status': 'blocked_until_real_wing_compressor_readback_gr_meter_headroom_feedback_recheck_and_operator_approval', 'soundcheck_dynamics_live_apply_blocked_without_wing_proof': True, 'soundcheck_master_processing_review_only_proposal': None, 'soundcheck_master_processing_native_osc_only_required': None, 'soundcheck_master_processing_target_surface': None, 'soundcheck_master_processing_types': None, 'soundcheck_master_processing_wing_apply_parameter_scope': None, 'soundcheck_master_processing_pa_feed_identity_required': None, 'soundcheck_master_processing_true_peak_headroom_recheck_required': None, 'soundcheck_master_processing_feedback_recheck_required': None, 'soundcheck_master_processing_bypass_level_match_required': None, 'soundcheck_master_processing_readback_required': None, 'soundcheck_master_processing_operator_confirmation_required': None, 'soundcheck_master_processing_rollback_ready': None, 'soundcheck_master_processing_no_routing_scene_snapshot_write_required': None, 'soundcheck_master_processing_bus_compression_max_target_gr_db': None, 'soundcheck_master_processing_limiter_max_target_gr_db': None, 'soundcheck_master_processing_live_promotion_status': None, 'soundcheck_master_processing_live_apply_blocked_without_wing_proof': None, 'live_dynamics_conservative_proposal': None, 'live_dynamics_stepwise_threshold_review_required': None, 'live_dynamics_max_threshold_step_db': None, 'live_dynamics_max_target_gain_reduction_db': None, 'live_dynamics_no_makeup_gain_required': None, 'live_dynamics_makeup_gain_db_limit': None, 'live_dynamics_gain_reduction_meter_readback_after_each_step_required': None, 'live_dynamics_previous_comp_model_readback_required': None, 'live_dynamics_operator_rehearsal_required': None, 'live_dynamics_live_promotion_status': None, 'live_dynamics_live_apply_blocked_without_wing_proof': None, 'review_only_parameter_keys': ['target_gain_reduction_db'], 'wing_dynamics_review_only_values': {'target_gain_reduction_db': 2.9}, 'wing_dynamics_review_only_parameter_reason_by_key': {'target_gain_reduction_db': 'target gain reduction is an operator/readback objective; WING Rack fw3.0.5 exposes compressor threshold/ratio/timing/knee/makeup/mix/envelope curve via OSC but no direct target gain-reduction write path'}, 'wing_dynamics_review_only_no_direct_osc_path': True, 'wing_dynamics_review_only_not_apply_ready': True, 'wing_dynamics_gain_reduction_objective_required': True, 'wing_dynamics_gain_reduction_objective_parameter_keys': ['target_gain_reduction_db'], 'wing_dynamics_gain_reduction_objective_values': {'target_gain_reduction_db': 2.9}, 'wing_dynamics_gain_reduction_target_db': 2.9, 'wing_dynamics_gain_reduction_max_db': None, 'wing_dynamics_gain_reduction_objective_tolerance_db': 1.0, 'wing_dynamics_gain_reduction_acceptance_min_db': 1.9, 'wing_dynamics_gain_reduction_acceptance_max_db': 3.9, 'wing_dynamics_gain_reduction_must_not_exceed_db': None, 'wing_dynamics_gain_reduction_objective_is_review_only': True, 'wing_dynamics_gain_reduction_objective_is_not_osc_write': True, 'wing_dynamics_gain_reduction_objective_no_direct_osc_write_path': True, 'wing_dynamics_gain_reduction_objective_apply_ready': False, 'wing_dynamics_gain_reduction_objective_requires_post_apply_meter_readback': True, 'wing_dynamics_gain_reduction_meter_readback_path_suffix': 'dyn/cpeak', 'wing_dynamics_gain_reduction_meter_readback_path_templates': ['/aux/{aux}/dyn/cpeak', '/aux/{aux}/dyn/cmode'], 'wing_dynamics_gain_reduction_meter_fallback_path_templates': [], 'wing_dynamics_gain_reduction_meter_value_type': 'read_only_aux_compressor_peak_percent_and_mode', 'wing_dynamics_gain_reduction_meter_is_not_apply_value': True, 'wing_dynamics_gain_reduction_meter_path_source': 'Docs/manuals/wing/WING Remote Protocols v3.0.5.pdf', 'wing_dynamics_gain_reduction_aux_depth_proxy': True, 'wing_dynamics_gain_reduction_objective_requires_operator_confirmation': True, 'wing_dynamics_gain_reduction_objective_live_promotion_status': 'blocked_until_actual_gain_reduction_meter_readback_and_operator_review', 'wing_dynamics_gain_reduction_objective_operator_check': 'confirm WING aux PSE/LA cpeak activity and cmode after the bounded depth/threshold proposal; aux exposes no dyn/gr meter', 'wing_dynamics_gain_reduction_objective_direct_osc_write_keys_excluded': ['target_gain_reduction_db'], 'wing_aux_dynamics_on_mapping_required': True, 'wing_aux_dynamics_on_osc_path_suffix': 'dyn/on', 'wing_aux_dynamics_on_osc_value': 1, 'wing_aux_dynamics_on_value_type': 'integer_bool', 'wing_aux_dynamics_on_readback_must_match_osc_value': True, 'wing_aux_dynamics_threshold_mapping_required': True, 'wing_aux_dynamics_threshold_osc_path_suffix': 'dyn/thr', 'wing_aux_dynamics_threshold_osc_value': -20.3, 'wing_aux_dynamics_threshold_value_type': 'float', 'wing_aux_dynamics_threshold_readback_must_match_osc_value': True, 'wing_aux_dynamics_depth_mapping_required': True, 'wing_aux_dynamics_depth_osc_path_suffix': 'dyn/depth', 'wing_aux_dynamics_depth_osc_value': 2.9, 'wing_aux_dynamics_depth_value_type': 'float', 'wing_aux_dynamics_depth_readback_must_match_osc_value': True, 'wing_aux_dynamics_fast_mapping_required': True, 'wing_aux_dynamics_fast_osc_path_suffix': 'dyn/fast', 'wing_aux_dynamics_fast_osc_value': 0, 'wing_aux_dynamics_fast_value_type': 'integer_bool', 'wing_aux_dynamics_fast_readback_must_match_osc_value': True}
- `soundcheck_aux_playback_pse_la_dynamics_selected_dynamics_wing_envelope_curve_value_unsupported` (soundcheck_aux_playback_pse_la_dynamics): None
- `soundcheck_aux_playback_pse_la_dynamics_readback_contract_wing_envelope_curve_readback_missing` (soundcheck_aux_playback_pse_la_dynamics): {'required': True, 'wing_osc_path_templates': ['/aux/{aux}/dyn/on', '/aux/{aux}/dyn/thr', '/aux/{aux}/dyn/depth', '/aux/{aux}/dyn/fast', '/aux/{aux}/dyn/peak', '/aux/{aux}/dyn/ingain', '/aux/{aux}/dyn/cpeak', '/aux/{aux}/dyn/cmode'], 'wing_rack_fw_3_0_5_native_osc_supported': True, 'dynamics_model_readback_required': False, 'dynamics_model_readback_must_equal': None, 'wing_compressor_model_readback_required': False, 'wing_compressor_model_readback_must_equal': None, 'wing_compressor_model_osc_path_suffix': None, 'wing_compressor_model_parameters_apply_only_when_model_matches': True, 'wing_aux_dynamics_pse_la_combo': True, 'wing_aux_dynamics_no_dyn_mdl_path': True, 'wing_aux_dynamics_no_ratio_attack_release_paths': True, 'wing_aux_dynamics_write_key_selection_policy': 'apply_only_selected_wing_fw_3_0_5_aux_pse_la_dynamics_value_keys', 'wing_aux_dynamics_apply_only_selected_write_keys': True, 'wing_aux_dynamics_selected_write_keys': ['wing_aux_dynamics_on_osc_value', 'wing_aux_dynamics_threshold_osc_value', 'wing_aux_dynamics_depth_osc_value', 'wing_aux_dynamics_fast_osc_value'], 'wing_aux_dynamics_forbidden_write_keys': ['wing_aux_dynamics_peak_osc_value'], 'wing_dynamics_required_model': None, 'wing_dynamics_current_model': None, 'wing_dynamics_current_enabled': None, 'wing_dynamics_current_operator_owned': None, 'wing_dynamics_model_readback_available': None, 'wing_dynamics_model_conflict': None, 'wing_dynamics_model_readback_required_for_active_slot': None, 'wing_dynamics_model_conflict_policy': None, 'wing_dynamics_model_conflict_blocker': None, 'wing_dynamics_model_readback_blocker': None, 'wing_dynamics_model_change_requires_operator_review': None, 'wing_dynamics_model_read_before_path': None, 'wing_dynamics_on_read_before_path': None, 'wing_dynamics_model_apply_only_when_current_model_matches': None, 'wing_dynamics_write_key_selection_policy': None, 'wing_dynamics_apply_only_selected_write_keys': None, 'wing_dynamics_selected_write_keys': None, 'wing_dynamics_forbidden_write_keys': None, 'read_before_paths': ['dyn/on', 'dyn/thr', 'dyn/depth', 'dyn/fast', 'dyn/peak', 'dyn/ingain', 'dyn/cpeak', 'dyn/cmode'], 'read_after_paths': ['dyn/on', 'dyn/thr', 'dyn/depth', 'dyn/fast', 'dyn/peak', 'dyn/ingain', 'dyn/cpeak', 'dyn/cmode'], 'makeup_gain_must_respect_headroom': True, 'soundcheck_dynamics_bounded_compression_proposal': True, 'soundcheck_dynamics_native_compressor_osc_only_required': True, 'soundcheck_dynamics_max_target_gain_reduction_db': 4.0, 'soundcheck_dynamics_max_makeup_gain_db': 1.0, 'soundcheck_dynamics_min_headroom_margin_db': 1.0, 'soundcheck_dynamics_required_true_peak_headroom_db': None, 'soundcheck_dynamics_required_summed_bus_true_peak_headroom_db': None, 'soundcheck_dynamics_target_gr_review_only': True, 'soundcheck_dynamics_target_gr_readback_required': True, 'soundcheck_dynamics_makeup_gain_headroom_required': True, 'soundcheck_dynamics_summed_bus_headroom_required': True, 'soundcheck_dynamics_compressor_model_readback_required': True, 'soundcheck_dynamics_gain_reduction_meter_readback_required': True, 'soundcheck_dynamics_feedback_recheck_required': True, 'soundcheck_dynamics_operator_confirmation_required': True, 'soundcheck_dynamics_readback_required': True, 'soundcheck_dynamics_rollback_ready': True, 'soundcheck_dynamics_live_promotion_status': 'blocked_until_real_wing_compressor_readback_gr_meter_headroom_feedback_recheck_and_operator_approval', 'soundcheck_dynamics_live_apply_blocked_without_wing_proof': True, 'soundcheck_master_processing_review_only_proposal': None, 'soundcheck_master_processing_native_osc_only_required': None, 'soundcheck_master_processing_target_surface': None, 'soundcheck_master_processing_types': None, 'soundcheck_master_processing_wing_apply_parameter_scope': None, 'soundcheck_master_processing_pa_feed_identity_required': None, 'soundcheck_master_processing_true_peak_headroom_recheck_required': None, 'soundcheck_master_processing_feedback_recheck_required': None, 'soundcheck_master_processing_bypass_level_match_required': None, 'soundcheck_master_processing_readback_required': None, 'soundcheck_master_processing_operator_confirmation_required': None, 'soundcheck_master_processing_rollback_ready': None, 'soundcheck_master_processing_no_routing_scene_snapshot_write_required': None, 'soundcheck_master_processing_bus_compression_max_target_gr_db': None, 'soundcheck_master_processing_limiter_max_target_gr_db': None, 'soundcheck_master_processing_live_promotion_status': None, 'soundcheck_master_processing_live_apply_blocked_without_wing_proof': None, 'live_dynamics_conservative_proposal': None, 'live_dynamics_stepwise_threshold_review_required': None, 'live_dynamics_max_threshold_step_db': None, 'live_dynamics_max_target_gain_reduction_db': None, 'live_dynamics_no_makeup_gain_required': None, 'live_dynamics_makeup_gain_db_limit': None, 'live_dynamics_gain_reduction_meter_readback_after_each_step_required': None, 'live_dynamics_previous_comp_model_readback_required': None, 'live_dynamics_operator_rehearsal_required': None, 'live_dynamics_live_promotion_status': None, 'live_dynamics_live_apply_blocked_without_wing_proof': None, 'review_only_parameter_keys': ['target_gain_reduction_db'], 'wing_dynamics_review_only_values': {'target_gain_reduction_db': 2.9}, 'wing_dynamics_review_only_parameter_reason_by_key': {'target_gain_reduction_db': 'target gain reduction is an operator/readback objective; WING Rack fw3.0.5 exposes compressor threshold/ratio/timing/knee/makeup/mix/envelope curve via OSC but no direct target gain-reduction write path'}, 'wing_dynamics_review_only_no_direct_osc_path': True, 'wing_dynamics_review_only_not_apply_ready': True, 'wing_dynamics_gain_reduction_objective_required': True, 'wing_dynamics_gain_reduction_objective_parameter_keys': ['target_gain_reduction_db'], 'wing_dynamics_gain_reduction_objective_values': {'target_gain_reduction_db': 2.9}, 'wing_dynamics_gain_reduction_target_db': 2.9, 'wing_dynamics_gain_reduction_max_db': None, 'wing_dynamics_gain_reduction_objective_tolerance_db': 1.0, 'wing_dynamics_gain_reduction_acceptance_min_db': 1.9, 'wing_dynamics_gain_reduction_acceptance_max_db': 3.9, 'wing_dynamics_gain_reduction_must_not_exceed_db': None, 'wing_dynamics_gain_reduction_objective_is_review_only': True, 'wing_dynamics_gain_reduction_objective_is_not_osc_write': True, 'wing_dynamics_gain_reduction_objective_no_direct_osc_write_path': True, 'wing_dynamics_gain_reduction_objective_apply_ready': False, 'wing_dynamics_gain_reduction_objective_requires_post_apply_meter_readback': True, 'wing_dynamics_gain_reduction_meter_readback_path_suffix': 'dyn/cpeak', 'wing_dynamics_gain_reduction_meter_readback_path_templates': ['/aux/{aux}/dyn/cpeak', '/aux/{aux}/dyn/cmode'], 'wing_dynamics_gain_reduction_meter_fallback_path_templates': [], 'wing_dynamics_gain_reduction_meter_value_type': 'read_only_aux_compressor_peak_percent_and_mode', 'wing_dynamics_gain_reduction_meter_is_not_apply_value': True, 'wing_dynamics_gain_reduction_meter_path_source': 'Docs/manuals/wing/WING Remote Protocols v3.0.5.pdf', 'wing_dynamics_gain_reduction_aux_depth_proxy': True, 'wing_dynamics_gain_reduction_objective_requires_operator_confirmation': True, 'wing_dynamics_gain_reduction_objective_live_promotion_status': 'blocked_until_actual_gain_reduction_meter_readback_and_operator_review', 'wing_dynamics_gain_reduction_objective_operator_check': 'confirm WING aux PSE/LA cpeak activity and cmode after the bounded depth/threshold proposal; aux exposes no dyn/gr meter', 'wing_dynamics_gain_reduction_objective_direct_osc_write_keys_excluded': ['target_gain_reduction_db'], 'wing_aux_dynamics_on_mapping_required': True, 'wing_aux_dynamics_on_osc_path_suffix': 'dyn/on', 'wing_aux_dynamics_on_osc_value': 1, 'wing_aux_dynamics_on_value_type': 'integer_bool', 'wing_aux_dynamics_on_readback_must_match_osc_value': True, 'wing_aux_dynamics_threshold_mapping_required': True, 'wing_aux_dynamics_threshold_osc_path_suffix': 'dyn/thr', 'wing_aux_dynamics_threshold_osc_value': -20.3, 'wing_aux_dynamics_threshold_value_type': 'float', 'wing_aux_dynamics_threshold_readback_must_match_osc_value': True, 'wing_aux_dynamics_depth_mapping_required': True, 'wing_aux_dynamics_depth_osc_path_suffix': 'dyn/depth', 'wing_aux_dynamics_depth_osc_value': 2.9, 'wing_aux_dynamics_depth_value_type': 'float', 'wing_aux_dynamics_depth_readback_must_match_osc_value': True, 'wing_aux_dynamics_fast_mapping_required': True, 'wing_aux_dynamics_fast_osc_path_suffix': 'dyn/fast', 'wing_aux_dynamics_fast_osc_value': 0, 'wing_aux_dynamics_fast_value_type': 'integer_bool', 'wing_aux_dynamics_fast_readback_must_match_osc_value': True}
- `soundcheck_aux_playback_pse_la_dynamics_selected_dynamics_wing_auto_mapping_missing` (soundcheck_aux_playback_pse_la_dynamics): ['wing_compressor_auto_mapping_policy', 'wing_compressor_auto_max', 'wing_compressor_auto_min', 'wing_compressor_auto_original_value', 'wing_compressor_auto_osc_path_suffix', 'wing_compressor_auto_osc_value', 'wing_compressor_auto_source', 'wing_compressor_auto_value_type']
- `soundcheck_aux_playback_pse_la_dynamics_readback_contract_wing_auto_mapping_not_required` (soundcheck_aux_playback_pse_la_dynamics): {'required': True, 'wing_osc_path_templates': ['/aux/{aux}/dyn/on', '/aux/{aux}/dyn/thr', '/aux/{aux}/dyn/depth', '/aux/{aux}/dyn/fast', '/aux/{aux}/dyn/peak', '/aux/{aux}/dyn/ingain', '/aux/{aux}/dyn/cpeak', '/aux/{aux}/dyn/cmode'], 'wing_rack_fw_3_0_5_native_osc_supported': True, 'dynamics_model_readback_required': False, 'dynamics_model_readback_must_equal': None, 'wing_compressor_model_readback_required': False, 'wing_compressor_model_readback_must_equal': None, 'wing_compressor_model_osc_path_suffix': None, 'wing_compressor_model_parameters_apply_only_when_model_matches': True, 'wing_aux_dynamics_pse_la_combo': True, 'wing_aux_dynamics_no_dyn_mdl_path': True, 'wing_aux_dynamics_no_ratio_attack_release_paths': True, 'wing_aux_dynamics_write_key_selection_policy': 'apply_only_selected_wing_fw_3_0_5_aux_pse_la_dynamics_value_keys', 'wing_aux_dynamics_apply_only_selected_write_keys': True, 'wing_aux_dynamics_selected_write_keys': ['wing_aux_dynamics_on_osc_value', 'wing_aux_dynamics_threshold_osc_value', 'wing_aux_dynamics_depth_osc_value', 'wing_aux_dynamics_fast_osc_value'], 'wing_aux_dynamics_forbidden_write_keys': ['wing_aux_dynamics_peak_osc_value'], 'wing_dynamics_required_model': None, 'wing_dynamics_current_model': None, 'wing_dynamics_current_enabled': None, 'wing_dynamics_current_operator_owned': None, 'wing_dynamics_model_readback_available': None, 'wing_dynamics_model_conflict': None, 'wing_dynamics_model_readback_required_for_active_slot': None, 'wing_dynamics_model_conflict_policy': None, 'wing_dynamics_model_conflict_blocker': None, 'wing_dynamics_model_readback_blocker': None, 'wing_dynamics_model_change_requires_operator_review': None, 'wing_dynamics_model_read_before_path': None, 'wing_dynamics_on_read_before_path': None, 'wing_dynamics_model_apply_only_when_current_model_matches': None, 'wing_dynamics_write_key_selection_policy': None, 'wing_dynamics_apply_only_selected_write_keys': None, 'wing_dynamics_selected_write_keys': None, 'wing_dynamics_forbidden_write_keys': None, 'read_before_paths': ['dyn/on', 'dyn/thr', 'dyn/depth', 'dyn/fast', 'dyn/peak', 'dyn/ingain', 'dyn/cpeak', 'dyn/cmode'], 'read_after_paths': ['dyn/on', 'dyn/thr', 'dyn/depth', 'dyn/fast', 'dyn/peak', 'dyn/ingain', 'dyn/cpeak', 'dyn/cmode'], 'makeup_gain_must_respect_headroom': True, 'soundcheck_dynamics_bounded_compression_proposal': True, 'soundcheck_dynamics_native_compressor_osc_only_required': True, 'soundcheck_dynamics_max_target_gain_reduction_db': 4.0, 'soundcheck_dynamics_max_makeup_gain_db': 1.0, 'soundcheck_dynamics_min_headroom_margin_db': 1.0, 'soundcheck_dynamics_required_true_peak_headroom_db': None, 'soundcheck_dynamics_required_summed_bus_true_peak_headroom_db': None, 'soundcheck_dynamics_target_gr_review_only': True, 'soundcheck_dynamics_target_gr_readback_required': True, 'soundcheck_dynamics_makeup_gain_headroom_required': True, 'soundcheck_dynamics_summed_bus_headroom_required': True, 'soundcheck_dynamics_compressor_model_readback_required': True, 'soundcheck_dynamics_gain_reduction_meter_readback_required': True, 'soundcheck_dynamics_feedback_recheck_required': True, 'soundcheck_dynamics_operator_confirmation_required': True, 'soundcheck_dynamics_readback_required': True, 'soundcheck_dynamics_rollback_ready': True, 'soundcheck_dynamics_live_promotion_status': 'blocked_until_real_wing_compressor_readback_gr_meter_headroom_feedback_recheck_and_operator_approval', 'soundcheck_dynamics_live_apply_blocked_without_wing_proof': True, 'soundcheck_master_processing_review_only_proposal': None, 'soundcheck_master_processing_native_osc_only_required': None, 'soundcheck_master_processing_target_surface': None, 'soundcheck_master_processing_types': None, 'soundcheck_master_processing_wing_apply_parameter_scope': None, 'soundcheck_master_processing_pa_feed_identity_required': None, 'soundcheck_master_processing_true_peak_headroom_recheck_required': None, 'soundcheck_master_processing_feedback_recheck_required': None, 'soundcheck_master_processing_bypass_level_match_required': None, 'soundcheck_master_processing_readback_required': None, 'soundcheck_master_processing_operator_confirmation_required': None, 'soundcheck_master_processing_rollback_ready': None, 'soundcheck_master_processing_no_routing_scene_snapshot_write_required': None, 'soundcheck_master_processing_bus_compression_max_target_gr_db': None, 'soundcheck_master_processing_limiter_max_target_gr_db': None, 'soundcheck_master_processing_live_promotion_status': None, 'soundcheck_master_processing_live_apply_blocked_without_wing_proof': None, 'live_dynamics_conservative_proposal': None, 'live_dynamics_stepwise_threshold_review_required': None, 'live_dynamics_max_threshold_step_db': None, 'live_dynamics_max_target_gain_reduction_db': None, 'live_dynamics_no_makeup_gain_required': None, 'live_dynamics_makeup_gain_db_limit': None, 'live_dynamics_gain_reduction_meter_readback_after_each_step_required': None, 'live_dynamics_previous_comp_model_readback_required': None, 'live_dynamics_operator_rehearsal_required': None, 'live_dynamics_live_promotion_status': None, 'live_dynamics_live_apply_blocked_without_wing_proof': None, 'review_only_parameter_keys': ['target_gain_reduction_db'], 'wing_dynamics_review_only_values': {'target_gain_reduction_db': 2.9}, 'wing_dynamics_review_only_parameter_reason_by_key': {'target_gain_reduction_db': 'target gain reduction is an operator/readback objective; WING Rack fw3.0.5 exposes compressor threshold/ratio/timing/knee/makeup/mix/envelope curve via OSC but no direct target gain-reduction write path'}, 'wing_dynamics_review_only_no_direct_osc_path': True, 'wing_dynamics_review_only_not_apply_ready': True, 'wing_dynamics_gain_reduction_objective_required': True, 'wing_dynamics_gain_reduction_objective_parameter_keys': ['target_gain_reduction_db'], 'wing_dynamics_gain_reduction_objective_values': {'target_gain_reduction_db': 2.9}, 'wing_dynamics_gain_reduction_target_db': 2.9, 'wing_dynamics_gain_reduction_max_db': None, 'wing_dynamics_gain_reduction_objective_tolerance_db': 1.0, 'wing_dynamics_gain_reduction_acceptance_min_db': 1.9, 'wing_dynamics_gain_reduction_acceptance_max_db': 3.9, 'wing_dynamics_gain_reduction_must_not_exceed_db': None, 'wing_dynamics_gain_reduction_objective_is_review_only': True, 'wing_dynamics_gain_reduction_objective_is_not_osc_write': True, 'wing_dynamics_gain_reduction_objective_no_direct_osc_write_path': True, 'wing_dynamics_gain_reduction_objective_apply_ready': False, 'wing_dynamics_gain_reduction_objective_requires_post_apply_meter_readback': True, 'wing_dynamics_gain_reduction_meter_readback_path_suffix': 'dyn/cpeak', 'wing_dynamics_gain_reduction_meter_readback_path_templates': ['/aux/{aux}/dyn/cpeak', '/aux/{aux}/dyn/cmode'], 'wing_dynamics_gain_reduction_meter_fallback_path_templates': [], 'wing_dynamics_gain_reduction_meter_value_type': 'read_only_aux_compressor_peak_percent_and_mode', 'wing_dynamics_gain_reduction_meter_is_not_apply_value': True, 'wing_dynamics_gain_reduction_meter_path_source': 'Docs/manuals/wing/WING Remote Protocols v3.0.5.pdf', 'wing_dynamics_gain_reduction_aux_depth_proxy': True, 'wing_dynamics_gain_reduction_objective_requires_operator_confirmation': True, 'wing_dynamics_gain_reduction_objective_live_promotion_status': 'blocked_until_actual_gain_reduction_meter_readback_and_operator_review', 'wing_dynamics_gain_reduction_objective_operator_check': 'confirm WING aux PSE/LA cpeak activity and cmode after the bounded depth/threshold proposal; aux exposes no dyn/gr meter', 'wing_dynamics_gain_reduction_objective_direct_osc_write_keys_excluded': ['target_gain_reduction_db'], 'wing_aux_dynamics_on_mapping_required': True, 'wing_aux_dynamics_on_osc_path_suffix': 'dyn/on', 'wing_aux_dynamics_on_osc_value': 1, 'wing_aux_dynamics_on_value_type': 'integer_bool', 'wing_aux_dynamics_on_readback_must_match_osc_value': True, 'wing_aux_dynamics_threshold_mapping_required': True, 'wing_aux_dynamics_threshold_osc_path_suffix': 'dyn/thr', 'wing_aux_dynamics_threshold_osc_value': -20.3, 'wing_aux_dynamics_threshold_value_type': 'float', 'wing_aux_dynamics_threshold_readback_must_match_osc_value': True, 'wing_aux_dynamics_depth_mapping_required': True, 'wing_aux_dynamics_depth_osc_path_suffix': 'dyn/depth', 'wing_aux_dynamics_depth_osc_value': 2.9, 'wing_aux_dynamics_depth_value_type': 'float', 'wing_aux_dynamics_depth_readback_must_match_osc_value': True, 'wing_aux_dynamics_fast_mapping_required': True, 'wing_aux_dynamics_fast_osc_path_suffix': 'dyn/fast', 'wing_aux_dynamics_fast_osc_value': 0, 'wing_aux_dynamics_fast_value_type': 'integer_bool', 'wing_aux_dynamics_fast_readback_must_match_osc_value': True}
- `soundcheck_aux_playback_pse_la_dynamics_readback_contract_wing_auto_path_mismatch` (soundcheck_aux_playback_pse_la_dynamics): {'required': True, 'wing_osc_path_templates': ['/aux/{aux}/dyn/on', '/aux/{aux}/dyn/thr', '/aux/{aux}/dyn/depth', '/aux/{aux}/dyn/fast', '/aux/{aux}/dyn/peak', '/aux/{aux}/dyn/ingain', '/aux/{aux}/dyn/cpeak', '/aux/{aux}/dyn/cmode'], 'wing_rack_fw_3_0_5_native_osc_supported': True, 'dynamics_model_readback_required': False, 'dynamics_model_readback_must_equal': None, 'wing_compressor_model_readback_required': False, 'wing_compressor_model_readback_must_equal': None, 'wing_compressor_model_osc_path_suffix': None, 'wing_compressor_model_parameters_apply_only_when_model_matches': True, 'wing_aux_dynamics_pse_la_combo': True, 'wing_aux_dynamics_no_dyn_mdl_path': True, 'wing_aux_dynamics_no_ratio_attack_release_paths': True, 'wing_aux_dynamics_write_key_selection_policy': 'apply_only_selected_wing_fw_3_0_5_aux_pse_la_dynamics_value_keys', 'wing_aux_dynamics_apply_only_selected_write_keys': True, 'wing_aux_dynamics_selected_write_keys': ['wing_aux_dynamics_on_osc_value', 'wing_aux_dynamics_threshold_osc_value', 'wing_aux_dynamics_depth_osc_value', 'wing_aux_dynamics_fast_osc_value'], 'wing_aux_dynamics_forbidden_write_keys': ['wing_aux_dynamics_peak_osc_value'], 'wing_dynamics_required_model': None, 'wing_dynamics_current_model': None, 'wing_dynamics_current_enabled': None, 'wing_dynamics_current_operator_owned': None, 'wing_dynamics_model_readback_available': None, 'wing_dynamics_model_conflict': None, 'wing_dynamics_model_readback_required_for_active_slot': None, 'wing_dynamics_model_conflict_policy': None, 'wing_dynamics_model_conflict_blocker': None, 'wing_dynamics_model_readback_blocker': None, 'wing_dynamics_model_change_requires_operator_review': None, 'wing_dynamics_model_read_before_path': None, 'wing_dynamics_on_read_before_path': None, 'wing_dynamics_model_apply_only_when_current_model_matches': None, 'wing_dynamics_write_key_selection_policy': None, 'wing_dynamics_apply_only_selected_write_keys': None, 'wing_dynamics_selected_write_keys': None, 'wing_dynamics_forbidden_write_keys': None, 'read_before_paths': ['dyn/on', 'dyn/thr', 'dyn/depth', 'dyn/fast', 'dyn/peak', 'dyn/ingain', 'dyn/cpeak', 'dyn/cmode'], 'read_after_paths': ['dyn/on', 'dyn/thr', 'dyn/depth', 'dyn/fast', 'dyn/peak', 'dyn/ingain', 'dyn/cpeak', 'dyn/cmode'], 'makeup_gain_must_respect_headroom': True, 'soundcheck_dynamics_bounded_compression_proposal': True, 'soundcheck_dynamics_native_compressor_osc_only_required': True, 'soundcheck_dynamics_max_target_gain_reduction_db': 4.0, 'soundcheck_dynamics_max_makeup_gain_db': 1.0, 'soundcheck_dynamics_min_headroom_margin_db': 1.0, 'soundcheck_dynamics_required_true_peak_headroom_db': None, 'soundcheck_dynamics_required_summed_bus_true_peak_headroom_db': None, 'soundcheck_dynamics_target_gr_review_only': True, 'soundcheck_dynamics_target_gr_readback_required': True, 'soundcheck_dynamics_makeup_gain_headroom_required': True, 'soundcheck_dynamics_summed_bus_headroom_required': True, 'soundcheck_dynamics_compressor_model_readback_required': True, 'soundcheck_dynamics_gain_reduction_meter_readback_required': True, 'soundcheck_dynamics_feedback_recheck_required': True, 'soundcheck_dynamics_operator_confirmation_required': True, 'soundcheck_dynamics_readback_required': True, 'soundcheck_dynamics_rollback_ready': True, 'soundcheck_dynamics_live_promotion_status': 'blocked_until_real_wing_compressor_readback_gr_meter_headroom_feedback_recheck_and_operator_approval', 'soundcheck_dynamics_live_apply_blocked_without_wing_proof': True, 'soundcheck_master_processing_review_only_proposal': None, 'soundcheck_master_processing_native_osc_only_required': None, 'soundcheck_master_processing_target_surface': None, 'soundcheck_master_processing_types': None, 'soundcheck_master_processing_wing_apply_parameter_scope': None, 'soundcheck_master_processing_pa_feed_identity_required': None, 'soundcheck_master_processing_true_peak_headroom_recheck_required': None, 'soundcheck_master_processing_feedback_recheck_required': None, 'soundcheck_master_processing_bypass_level_match_required': None, 'soundcheck_master_processing_readback_required': None, 'soundcheck_master_processing_operator_confirmation_required': None, 'soundcheck_master_processing_rollback_ready': None, 'soundcheck_master_processing_no_routing_scene_snapshot_write_required': None, 'soundcheck_master_processing_bus_compression_max_target_gr_db': None, 'soundcheck_master_processing_limiter_max_target_gr_db': None, 'soundcheck_master_processing_live_promotion_status': None, 'soundcheck_master_processing_live_apply_blocked_without_wing_proof': None, 'live_dynamics_conservative_proposal': None, 'live_dynamics_stepwise_threshold_review_required': None, 'live_dynamics_max_threshold_step_db': None, 'live_dynamics_max_target_gain_reduction_db': None, 'live_dynamics_no_makeup_gain_required': None, 'live_dynamics_makeup_gain_db_limit': None, 'live_dynamics_gain_reduction_meter_readback_after_each_step_required': None, 'live_dynamics_previous_comp_model_readback_required': None, 'live_dynamics_operator_rehearsal_required': None, 'live_dynamics_live_promotion_status': None, 'live_dynamics_live_apply_blocked_without_wing_proof': None, 'review_only_parameter_keys': ['target_gain_reduction_db'], 'wing_dynamics_review_only_values': {'target_gain_reduction_db': 2.9}, 'wing_dynamics_review_only_parameter_reason_by_key': {'target_gain_reduction_db': 'target gain reduction is an operator/readback objective; WING Rack fw3.0.5 exposes compressor threshold/ratio/timing/knee/makeup/mix/envelope curve via OSC but no direct target gain-reduction write path'}, 'wing_dynamics_review_only_no_direct_osc_path': True, 'wing_dynamics_review_only_not_apply_ready': True, 'wing_dynamics_gain_reduction_objective_required': True, 'wing_dynamics_gain_reduction_objective_parameter_keys': ['target_gain_reduction_db'], 'wing_dynamics_gain_reduction_objective_values': {'target_gain_reduction_db': 2.9}, 'wing_dynamics_gain_reduction_target_db': 2.9, 'wing_dynamics_gain_reduction_max_db': None, 'wing_dynamics_gain_reduction_objective_tolerance_db': 1.0, 'wing_dynamics_gain_reduction_acceptance_min_db': 1.9, 'wing_dynamics_gain_reduction_acceptance_max_db': 3.9, 'wing_dynamics_gain_reduction_must_not_exceed_db': None, 'wing_dynamics_gain_reduction_objective_is_review_only': True, 'wing_dynamics_gain_reduction_objective_is_not_osc_write': True, 'wing_dynamics_gain_reduction_objective_no_direct_osc_write_path': True, 'wing_dynamics_gain_reduction_objective_apply_ready': False, 'wing_dynamics_gain_reduction_objective_requires_post_apply_meter_readback': True, 'wing_dynamics_gain_reduction_meter_readback_path_suffix': 'dyn/cpeak', 'wing_dynamics_gain_reduction_meter_readback_path_templates': ['/aux/{aux}/dyn/cpeak', '/aux/{aux}/dyn/cmode'], 'wing_dynamics_gain_reduction_meter_fallback_path_templates': [], 'wing_dynamics_gain_reduction_meter_value_type': 'read_only_aux_compressor_peak_percent_and_mode', 'wing_dynamics_gain_reduction_meter_is_not_apply_value': True, 'wing_dynamics_gain_reduction_meter_path_source': 'Docs/manuals/wing/WING Remote Protocols v3.0.5.pdf', 'wing_dynamics_gain_reduction_aux_depth_proxy': True, 'wing_dynamics_gain_reduction_objective_requires_operator_confirmation': True, 'wing_dynamics_gain_reduction_objective_live_promotion_status': 'blocked_until_actual_gain_reduction_meter_readback_and_operator_review', 'wing_dynamics_gain_reduction_objective_operator_check': 'confirm WING aux PSE/LA cpeak activity and cmode after the bounded depth/threshold proposal; aux exposes no dyn/gr meter', 'wing_dynamics_gain_reduction_objective_direct_osc_write_keys_excluded': ['target_gain_reduction_db'], 'wing_aux_dynamics_on_mapping_required': True, 'wing_aux_dynamics_on_osc_path_suffix': 'dyn/on', 'wing_aux_dynamics_on_osc_value': 1, 'wing_aux_dynamics_on_value_type': 'integer_bool', 'wing_aux_dynamics_on_readback_must_match_osc_value': True, 'wing_aux_dynamics_threshold_mapping_required': True, 'wing_aux_dynamics_threshold_osc_path_suffix': 'dyn/thr', 'wing_aux_dynamics_threshold_osc_value': -20.3, 'wing_aux_dynamics_threshold_value_type': 'float', 'wing_aux_dynamics_threshold_readback_must_match_osc_value': True, 'wing_aux_dynamics_depth_mapping_required': True, 'wing_aux_dynamics_depth_osc_path_suffix': 'dyn/depth', 'wing_aux_dynamics_depth_osc_value': 2.9, 'wing_aux_dynamics_depth_value_type': 'float', 'wing_aux_dynamics_depth_readback_must_match_osc_value': True, 'wing_aux_dynamics_fast_mapping_required': True, 'wing_aux_dynamics_fast_osc_path_suffix': 'dyn/fast', 'wing_aux_dynamics_fast_osc_value': 0, 'wing_aux_dynamics_fast_value_type': 'integer_bool', 'wing_aux_dynamics_fast_readback_must_match_osc_value': True}
- `soundcheck_aux_playback_pse_la_dynamics_selected_dynamics_wing_auto_value_unsupported` (soundcheck_aux_playback_pse_la_dynamics): None
- `soundcheck_aux_playback_pse_la_dynamics_selected_dynamics_wing_auto_value_type_mismatch` (soundcheck_aux_playback_pse_la_dynamics): None
- `soundcheck_aux_playback_pse_la_dynamics_readback_contract_wing_auto_readback_missing` (soundcheck_aux_playback_pse_la_dynamics): {'required': True, 'wing_osc_path_templates': ['/aux/{aux}/dyn/on', '/aux/{aux}/dyn/thr', '/aux/{aux}/dyn/depth', '/aux/{aux}/dyn/fast', '/aux/{aux}/dyn/peak', '/aux/{aux}/dyn/ingain', '/aux/{aux}/dyn/cpeak', '/aux/{aux}/dyn/cmode'], 'wing_rack_fw_3_0_5_native_osc_supported': True, 'dynamics_model_readback_required': False, 'dynamics_model_readback_must_equal': None, 'wing_compressor_model_readback_required': False, 'wing_compressor_model_readback_must_equal': None, 'wing_compressor_model_osc_path_suffix': None, 'wing_compressor_model_parameters_apply_only_when_model_matches': True, 'wing_aux_dynamics_pse_la_combo': True, 'wing_aux_dynamics_no_dyn_mdl_path': True, 'wing_aux_dynamics_no_ratio_attack_release_paths': True, 'wing_aux_dynamics_write_key_selection_policy': 'apply_only_selected_wing_fw_3_0_5_aux_pse_la_dynamics_value_keys', 'wing_aux_dynamics_apply_only_selected_write_keys': True, 'wing_aux_dynamics_selected_write_keys': ['wing_aux_dynamics_on_osc_value', 'wing_aux_dynamics_threshold_osc_value', 'wing_aux_dynamics_depth_osc_value', 'wing_aux_dynamics_fast_osc_value'], 'wing_aux_dynamics_forbidden_write_keys': ['wing_aux_dynamics_peak_osc_value'], 'wing_dynamics_required_model': None, 'wing_dynamics_current_model': None, 'wing_dynamics_current_enabled': None, 'wing_dynamics_current_operator_owned': None, 'wing_dynamics_model_readback_available': None, 'wing_dynamics_model_conflict': None, 'wing_dynamics_model_readback_required_for_active_slot': None, 'wing_dynamics_model_conflict_policy': None, 'wing_dynamics_model_conflict_blocker': None, 'wing_dynamics_model_readback_blocker': None, 'wing_dynamics_model_change_requires_operator_review': None, 'wing_dynamics_model_read_before_path': None, 'wing_dynamics_on_read_before_path': None, 'wing_dynamics_model_apply_only_when_current_model_matches': None, 'wing_dynamics_write_key_selection_policy': None, 'wing_dynamics_apply_only_selected_write_keys': None, 'wing_dynamics_selected_write_keys': None, 'wing_dynamics_forbidden_write_keys': None, 'read_before_paths': ['dyn/on', 'dyn/thr', 'dyn/depth', 'dyn/fast', 'dyn/peak', 'dyn/ingain', 'dyn/cpeak', 'dyn/cmode'], 'read_after_paths': ['dyn/on', 'dyn/thr', 'dyn/depth', 'dyn/fast', 'dyn/peak', 'dyn/ingain', 'dyn/cpeak', 'dyn/cmode'], 'makeup_gain_must_respect_headroom': True, 'soundcheck_dynamics_bounded_compression_proposal': True, 'soundcheck_dynamics_native_compressor_osc_only_required': True, 'soundcheck_dynamics_max_target_gain_reduction_db': 4.0, 'soundcheck_dynamics_max_makeup_gain_db': 1.0, 'soundcheck_dynamics_min_headroom_margin_db': 1.0, 'soundcheck_dynamics_required_true_peak_headroom_db': None, 'soundcheck_dynamics_required_summed_bus_true_peak_headroom_db': None, 'soundcheck_dynamics_target_gr_review_only': True, 'soundcheck_dynamics_target_gr_readback_required': True, 'soundcheck_dynamics_makeup_gain_headroom_required': True, 'soundcheck_dynamics_summed_bus_headroom_required': True, 'soundcheck_dynamics_compressor_model_readback_required': True, 'soundcheck_dynamics_gain_reduction_meter_readback_required': True, 'soundcheck_dynamics_feedback_recheck_required': True, 'soundcheck_dynamics_operator_confirmation_required': True, 'soundcheck_dynamics_readback_required': True, 'soundcheck_dynamics_rollback_ready': True, 'soundcheck_dynamics_live_promotion_status': 'blocked_until_real_wing_compressor_readback_gr_meter_headroom_feedback_recheck_and_operator_approval', 'soundcheck_dynamics_live_apply_blocked_without_wing_proof': True, 'soundcheck_master_processing_review_only_proposal': None, 'soundcheck_master_processing_native_osc_only_required': None, 'soundcheck_master_processing_target_surface': None, 'soundcheck_master_processing_types': None, 'soundcheck_master_processing_wing_apply_parameter_scope': None, 'soundcheck_master_processing_pa_feed_identity_required': None, 'soundcheck_master_processing_true_peak_headroom_recheck_required': None, 'soundcheck_master_processing_feedback_recheck_required': None, 'soundcheck_master_processing_bypass_level_match_required': None, 'soundcheck_master_processing_readback_required': None, 'soundcheck_master_processing_operator_confirmation_required': None, 'soundcheck_master_processing_rollback_ready': None, 'soundcheck_master_processing_no_routing_scene_snapshot_write_required': None, 'soundcheck_master_processing_bus_compression_max_target_gr_db': None, 'soundcheck_master_processing_limiter_max_target_gr_db': None, 'soundcheck_master_processing_live_promotion_status': None, 'soundcheck_master_processing_live_apply_blocked_without_wing_proof': None, 'live_dynamics_conservative_proposal': None, 'live_dynamics_stepwise_threshold_review_required': None, 'live_dynamics_max_threshold_step_db': None, 'live_dynamics_max_target_gain_reduction_db': None, 'live_dynamics_no_makeup_gain_required': None, 'live_dynamics_makeup_gain_db_limit': None, 'live_dynamics_gain_reduction_meter_readback_after_each_step_required': None, 'live_dynamics_previous_comp_model_readback_required': None, 'live_dynamics_operator_rehearsal_required': None, 'live_dynamics_live_promotion_status': None, 'live_dynamics_live_apply_blocked_without_wing_proof': None, 'review_only_parameter_keys': ['target_gain_reduction_db'], 'wing_dynamics_review_only_values': {'target_gain_reduction_db': 2.9}, 'wing_dynamics_review_only_parameter_reason_by_key': {'target_gain_reduction_db': 'target gain reduction is an operator/readback objective; WING Rack fw3.0.5 exposes compressor threshold/ratio/timing/knee/makeup/mix/envelope curve via OSC but no direct target gain-reduction write path'}, 'wing_dynamics_review_only_no_direct_osc_path': True, 'wing_dynamics_review_only_not_apply_ready': True, 'wing_dynamics_gain_reduction_objective_required': True, 'wing_dynamics_gain_reduction_objective_parameter_keys': ['target_gain_reduction_db'], 'wing_dynamics_gain_reduction_objective_values': {'target_gain_reduction_db': 2.9}, 'wing_dynamics_gain_reduction_target_db': 2.9, 'wing_dynamics_gain_reduction_max_db': None, 'wing_dynamics_gain_reduction_objective_tolerance_db': 1.0, 'wing_dynamics_gain_reduction_acceptance_min_db': 1.9, 'wing_dynamics_gain_reduction_acceptance_max_db': 3.9, 'wing_dynamics_gain_reduction_must_not_exceed_db': None, 'wing_dynamics_gain_reduction_objective_is_review_only': True, 'wing_dynamics_gain_reduction_objective_is_not_osc_write': True, 'wing_dynamics_gain_reduction_objective_no_direct_osc_write_path': True, 'wing_dynamics_gain_reduction_objective_apply_ready': False, 'wing_dynamics_gain_reduction_objective_requires_post_apply_meter_readback': True, 'wing_dynamics_gain_reduction_meter_readback_path_suffix': 'dyn/cpeak', 'wing_dynamics_gain_reduction_meter_readback_path_templates': ['/aux/{aux}/dyn/cpeak', '/aux/{aux}/dyn/cmode'], 'wing_dynamics_gain_reduction_meter_fallback_path_templates': [], 'wing_dynamics_gain_reduction_meter_value_type': 'read_only_aux_compressor_peak_percent_and_mode', 'wing_dynamics_gain_reduction_meter_is_not_apply_value': True, 'wing_dynamics_gain_reduction_meter_path_source': 'Docs/manuals/wing/WING Remote Protocols v3.0.5.pdf', 'wing_dynamics_gain_reduction_aux_depth_proxy': True, 'wing_dynamics_gain_reduction_objective_requires_operator_confirmation': True, 'wing_dynamics_gain_reduction_objective_live_promotion_status': 'blocked_until_actual_gain_reduction_meter_readback_and_operator_review', 'wing_dynamics_gain_reduction_objective_operator_check': 'confirm WING aux PSE/LA cpeak activity and cmode after the bounded depth/threshold proposal; aux exposes no dyn/gr meter', 'wing_dynamics_gain_reduction_objective_direct_osc_write_keys_excluded': ['target_gain_reduction_db'], 'wing_aux_dynamics_on_mapping_required': True, 'wing_aux_dynamics_on_osc_path_suffix': 'dyn/on', 'wing_aux_dynamics_on_osc_value': 1, 'wing_aux_dynamics_on_value_type': 'integer_bool', 'wing_aux_dynamics_on_readback_must_match_osc_value': True, 'wing_aux_dynamics_threshold_mapping_required': True, 'wing_aux_dynamics_threshold_osc_path_suffix': 'dyn/thr', 'wing_aux_dynamics_threshold_osc_value': -20.3, 'wing_aux_dynamics_threshold_value_type': 'float', 'wing_aux_dynamics_threshold_readback_must_match_osc_value': True, 'wing_aux_dynamics_depth_mapping_required': True, 'wing_aux_dynamics_depth_osc_path_suffix': 'dyn/depth', 'wing_aux_dynamics_depth_osc_value': 2.9, 'wing_aux_dynamics_depth_value_type': 'float', 'wing_aux_dynamics_depth_readback_must_match_osc_value': True, 'wing_aux_dynamics_fast_mapping_required': True, 'wing_aux_dynamics_fast_osc_path_suffix': 'dyn/fast', 'wing_aux_dynamics_fast_osc_value': 0, 'wing_aux_dynamics_fast_value_type': 'integer_bool', 'wing_aux_dynamics_fast_readback_must_match_osc_value': True}
- `soundcheck_aux_playback_pse_la_dynamics_selected_dynamics_wing_knee_mapping_missing` (soundcheck_aux_playback_pse_la_dynamics): ['wing_compressor_knee_osc_path_suffix', 'wing_compressor_knee_osc_value', 'wing_compressor_knee_quantization_error', 'wing_compressor_knee_quantized_from', 'wing_compressor_knee_source', 'wing_compressor_knee_value_type']
- `soundcheck_aux_playback_pse_la_dynamics_readback_contract_wing_knee_mapping_not_required` (soundcheck_aux_playback_pse_la_dynamics): {'required': True, 'wing_osc_path_templates': ['/aux/{aux}/dyn/on', '/aux/{aux}/dyn/thr', '/aux/{aux}/dyn/depth', '/aux/{aux}/dyn/fast', '/aux/{aux}/dyn/peak', '/aux/{aux}/dyn/ingain', '/aux/{aux}/dyn/cpeak', '/aux/{aux}/dyn/cmode'], 'wing_rack_fw_3_0_5_native_osc_supported': True, 'dynamics_model_readback_required': False, 'dynamics_model_readback_must_equal': None, 'wing_compressor_model_readback_required': False, 'wing_compressor_model_readback_must_equal': None, 'wing_compressor_model_osc_path_suffix': None, 'wing_compressor_model_parameters_apply_only_when_model_matches': True, 'wing_aux_dynamics_pse_la_combo': True, 'wing_aux_dynamics_no_dyn_mdl_path': True, 'wing_aux_dynamics_no_ratio_attack_release_paths': True, 'wing_aux_dynamics_write_key_selection_policy': 'apply_only_selected_wing_fw_3_0_5_aux_pse_la_dynamics_value_keys', 'wing_aux_dynamics_apply_only_selected_write_keys': True, 'wing_aux_dynamics_selected_write_keys': ['wing_aux_dynamics_on_osc_value', 'wing_aux_dynamics_threshold_osc_value', 'wing_aux_dynamics_depth_osc_value', 'wing_aux_dynamics_fast_osc_value'], 'wing_aux_dynamics_forbidden_write_keys': ['wing_aux_dynamics_peak_osc_value'], 'wing_dynamics_required_model': None, 'wing_dynamics_current_model': None, 'wing_dynamics_current_enabled': None, 'wing_dynamics_current_operator_owned': None, 'wing_dynamics_model_readback_available': None, 'wing_dynamics_model_conflict': None, 'wing_dynamics_model_readback_required_for_active_slot': None, 'wing_dynamics_model_conflict_policy': None, 'wing_dynamics_model_conflict_blocker': None, 'wing_dynamics_model_readback_blocker': None, 'wing_dynamics_model_change_requires_operator_review': None, 'wing_dynamics_model_read_before_path': None, 'wing_dynamics_on_read_before_path': None, 'wing_dynamics_model_apply_only_when_current_model_matches': None, 'wing_dynamics_write_key_selection_policy': None, 'wing_dynamics_apply_only_selected_write_keys': None, 'wing_dynamics_selected_write_keys': None, 'wing_dynamics_forbidden_write_keys': None, 'read_before_paths': ['dyn/on', 'dyn/thr', 'dyn/depth', 'dyn/fast', 'dyn/peak', 'dyn/ingain', 'dyn/cpeak', 'dyn/cmode'], 'read_after_paths': ['dyn/on', 'dyn/thr', 'dyn/depth', 'dyn/fast', 'dyn/peak', 'dyn/ingain', 'dyn/cpeak', 'dyn/cmode'], 'makeup_gain_must_respect_headroom': True, 'soundcheck_dynamics_bounded_compression_proposal': True, 'soundcheck_dynamics_native_compressor_osc_only_required': True, 'soundcheck_dynamics_max_target_gain_reduction_db': 4.0, 'soundcheck_dynamics_max_makeup_gain_db': 1.0, 'soundcheck_dynamics_min_headroom_margin_db': 1.0, 'soundcheck_dynamics_required_true_peak_headroom_db': None, 'soundcheck_dynamics_required_summed_bus_true_peak_headroom_db': None, 'soundcheck_dynamics_target_gr_review_only': True, 'soundcheck_dynamics_target_gr_readback_required': True, 'soundcheck_dynamics_makeup_gain_headroom_required': True, 'soundcheck_dynamics_summed_bus_headroom_required': True, 'soundcheck_dynamics_compressor_model_readback_required': True, 'soundcheck_dynamics_gain_reduction_meter_readback_required': True, 'soundcheck_dynamics_feedback_recheck_required': True, 'soundcheck_dynamics_operator_confirmation_required': True, 'soundcheck_dynamics_readback_required': True, 'soundcheck_dynamics_rollback_ready': True, 'soundcheck_dynamics_live_promotion_status': 'blocked_until_real_wing_compressor_readback_gr_meter_headroom_feedback_recheck_and_operator_approval', 'soundcheck_dynamics_live_apply_blocked_without_wing_proof': True, 'soundcheck_master_processing_review_only_proposal': None, 'soundcheck_master_processing_native_osc_only_required': None, 'soundcheck_master_processing_target_surface': None, 'soundcheck_master_processing_types': None, 'soundcheck_master_processing_wing_apply_parameter_scope': None, 'soundcheck_master_processing_pa_feed_identity_required': None, 'soundcheck_master_processing_true_peak_headroom_recheck_required': None, 'soundcheck_master_processing_feedback_recheck_required': None, 'soundcheck_master_processing_bypass_level_match_required': None, 'soundcheck_master_processing_readback_required': None, 'soundcheck_master_processing_operator_confirmation_required': None, 'soundcheck_master_processing_rollback_ready': None, 'soundcheck_master_processing_no_routing_scene_snapshot_write_required': None, 'soundcheck_master_processing_bus_compression_max_target_gr_db': None, 'soundcheck_master_processing_limiter_max_target_gr_db': None, 'soundcheck_master_processing_live_promotion_status': None, 'soundcheck_master_processing_live_apply_blocked_without_wing_proof': None, 'live_dynamics_conservative_proposal': None, 'live_dynamics_stepwise_threshold_review_required': None, 'live_dynamics_max_threshold_step_db': None, 'live_dynamics_max_target_gain_reduction_db': None, 'live_dynamics_no_makeup_gain_required': None, 'live_dynamics_makeup_gain_db_limit': None, 'live_dynamics_gain_reduction_meter_readback_after_each_step_required': None, 'live_dynamics_previous_comp_model_readback_required': None, 'live_dynamics_operator_rehearsal_required': None, 'live_dynamics_live_promotion_status': None, 'live_dynamics_live_apply_blocked_without_wing_proof': None, 'review_only_parameter_keys': ['target_gain_reduction_db'], 'wing_dynamics_review_only_values': {'target_gain_reduction_db': 2.9}, 'wing_dynamics_review_only_parameter_reason_by_key': {'target_gain_reduction_db': 'target gain reduction is an operator/readback objective; WING Rack fw3.0.5 exposes compressor threshold/ratio/timing/knee/makeup/mix/envelope curve via OSC but no direct target gain-reduction write path'}, 'wing_dynamics_review_only_no_direct_osc_path': True, 'wing_dynamics_review_only_not_apply_ready': True, 'wing_dynamics_gain_reduction_objective_required': True, 'wing_dynamics_gain_reduction_objective_parameter_keys': ['target_gain_reduction_db'], 'wing_dynamics_gain_reduction_objective_values': {'target_gain_reduction_db': 2.9}, 'wing_dynamics_gain_reduction_target_db': 2.9, 'wing_dynamics_gain_reduction_max_db': None, 'wing_dynamics_gain_reduction_objective_tolerance_db': 1.0, 'wing_dynamics_gain_reduction_acceptance_min_db': 1.9, 'wing_dynamics_gain_reduction_acceptance_max_db': 3.9, 'wing_dynamics_gain_reduction_must_not_exceed_db': None, 'wing_dynamics_gain_reduction_objective_is_review_only': True, 'wing_dynamics_gain_reduction_objective_is_not_osc_write': True, 'wing_dynamics_gain_reduction_objective_no_direct_osc_write_path': True, 'wing_dynamics_gain_reduction_objective_apply_ready': False, 'wing_dynamics_gain_reduction_objective_requires_post_apply_meter_readback': True, 'wing_dynamics_gain_reduction_meter_readback_path_suffix': 'dyn/cpeak', 'wing_dynamics_gain_reduction_meter_readback_path_templates': ['/aux/{aux}/dyn/cpeak', '/aux/{aux}/dyn/cmode'], 'wing_dynamics_gain_reduction_meter_fallback_path_templates': [], 'wing_dynamics_gain_reduction_meter_value_type': 'read_only_aux_compressor_peak_percent_and_mode', 'wing_dynamics_gain_reduction_meter_is_not_apply_value': True, 'wing_dynamics_gain_reduction_meter_path_source': 'Docs/manuals/wing/WING Remote Protocols v3.0.5.pdf', 'wing_dynamics_gain_reduction_aux_depth_proxy': True, 'wing_dynamics_gain_reduction_objective_requires_operator_confirmation': True, 'wing_dynamics_gain_reduction_objective_live_promotion_status': 'blocked_until_actual_gain_reduction_meter_readback_and_operator_review', 'wing_dynamics_gain_reduction_objective_operator_check': 'confirm WING aux PSE/LA cpeak activity and cmode after the bounded depth/threshold proposal; aux exposes no dyn/gr meter', 'wing_dynamics_gain_reduction_objective_direct_osc_write_keys_excluded': ['target_gain_reduction_db'], 'wing_aux_dynamics_on_mapping_required': True, 'wing_aux_dynamics_on_osc_path_suffix': 'dyn/on', 'wing_aux_dynamics_on_osc_value': 1, 'wing_aux_dynamics_on_value_type': 'integer_bool', 'wing_aux_dynamics_on_readback_must_match_osc_value': True, 'wing_aux_dynamics_threshold_mapping_required': True, 'wing_aux_dynamics_threshold_osc_path_suffix': 'dyn/thr', 'wing_aux_dynamics_threshold_osc_value': -20.3, 'wing_aux_dynamics_threshold_value_type': 'float', 'wing_aux_dynamics_threshold_readback_must_match_osc_value': True, 'wing_aux_dynamics_depth_mapping_required': True, 'wing_aux_dynamics_depth_osc_path_suffix': 'dyn/depth', 'wing_aux_dynamics_depth_osc_value': 2.9, 'wing_aux_dynamics_depth_value_type': 'float', 'wing_aux_dynamics_depth_readback_must_match_osc_value': True, 'wing_aux_dynamics_fast_mapping_required': True, 'wing_aux_dynamics_fast_osc_path_suffix': 'dyn/fast', 'wing_aux_dynamics_fast_osc_value': 0, 'wing_aux_dynamics_fast_value_type': 'integer_bool', 'wing_aux_dynamics_fast_readback_must_match_osc_value': True}
- `soundcheck_aux_playback_pse_la_dynamics_readback_contract_wing_knee_path_mismatch` (soundcheck_aux_playback_pse_la_dynamics): {'required': True, 'wing_osc_path_templates': ['/aux/{aux}/dyn/on', '/aux/{aux}/dyn/thr', '/aux/{aux}/dyn/depth', '/aux/{aux}/dyn/fast', '/aux/{aux}/dyn/peak', '/aux/{aux}/dyn/ingain', '/aux/{aux}/dyn/cpeak', '/aux/{aux}/dyn/cmode'], 'wing_rack_fw_3_0_5_native_osc_supported': True, 'dynamics_model_readback_required': False, 'dynamics_model_readback_must_equal': None, 'wing_compressor_model_readback_required': False, 'wing_compressor_model_readback_must_equal': None, 'wing_compressor_model_osc_path_suffix': None, 'wing_compressor_model_parameters_apply_only_when_model_matches': True, 'wing_aux_dynamics_pse_la_combo': True, 'wing_aux_dynamics_no_dyn_mdl_path': True, 'wing_aux_dynamics_no_ratio_attack_release_paths': True, 'wing_aux_dynamics_write_key_selection_policy': 'apply_only_selected_wing_fw_3_0_5_aux_pse_la_dynamics_value_keys', 'wing_aux_dynamics_apply_only_selected_write_keys': True, 'wing_aux_dynamics_selected_write_keys': ['wing_aux_dynamics_on_osc_value', 'wing_aux_dynamics_threshold_osc_value', 'wing_aux_dynamics_depth_osc_value', 'wing_aux_dynamics_fast_osc_value'], 'wing_aux_dynamics_forbidden_write_keys': ['wing_aux_dynamics_peak_osc_value'], 'wing_dynamics_required_model': None, 'wing_dynamics_current_model': None, 'wing_dynamics_current_enabled': None, 'wing_dynamics_current_operator_owned': None, 'wing_dynamics_model_readback_available': None, 'wing_dynamics_model_conflict': None, 'wing_dynamics_model_readback_required_for_active_slot': None, 'wing_dynamics_model_conflict_policy': None, 'wing_dynamics_model_conflict_blocker': None, 'wing_dynamics_model_readback_blocker': None, 'wing_dynamics_model_change_requires_operator_review': None, 'wing_dynamics_model_read_before_path': None, 'wing_dynamics_on_read_before_path': None, 'wing_dynamics_model_apply_only_when_current_model_matches': None, 'wing_dynamics_write_key_selection_policy': None, 'wing_dynamics_apply_only_selected_write_keys': None, 'wing_dynamics_selected_write_keys': None, 'wing_dynamics_forbidden_write_keys': None, 'read_before_paths': ['dyn/on', 'dyn/thr', 'dyn/depth', 'dyn/fast', 'dyn/peak', 'dyn/ingain', 'dyn/cpeak', 'dyn/cmode'], 'read_after_paths': ['dyn/on', 'dyn/thr', 'dyn/depth', 'dyn/fast', 'dyn/peak', 'dyn/ingain', 'dyn/cpeak', 'dyn/cmode'], 'makeup_gain_must_respect_headroom': True, 'soundcheck_dynamics_bounded_compression_proposal': True, 'soundcheck_dynamics_native_compressor_osc_only_required': True, 'soundcheck_dynamics_max_target_gain_reduction_db': 4.0, 'soundcheck_dynamics_max_makeup_gain_db': 1.0, 'soundcheck_dynamics_min_headroom_margin_db': 1.0, 'soundcheck_dynamics_required_true_peak_headroom_db': None, 'soundcheck_dynamics_required_summed_bus_true_peak_headroom_db': None, 'soundcheck_dynamics_target_gr_review_only': True, 'soundcheck_dynamics_target_gr_readback_required': True, 'soundcheck_dynamics_makeup_gain_headroom_required': True, 'soundcheck_dynamics_summed_bus_headroom_required': True, 'soundcheck_dynamics_compressor_model_readback_required': True, 'soundcheck_dynamics_gain_reduction_meter_readback_required': True, 'soundcheck_dynamics_feedback_recheck_required': True, 'soundcheck_dynamics_operator_confirmation_required': True, 'soundcheck_dynamics_readback_required': True, 'soundcheck_dynamics_rollback_ready': True, 'soundcheck_dynamics_live_promotion_status': 'blocked_until_real_wing_compressor_readback_gr_meter_headroom_feedback_recheck_and_operator_approval', 'soundcheck_dynamics_live_apply_blocked_without_wing_proof': True, 'soundcheck_master_processing_review_only_proposal': None, 'soundcheck_master_processing_native_osc_only_required': None, 'soundcheck_master_processing_target_surface': None, 'soundcheck_master_processing_types': None, 'soundcheck_master_processing_wing_apply_parameter_scope': None, 'soundcheck_master_processing_pa_feed_identity_required': None, 'soundcheck_master_processing_true_peak_headroom_recheck_required': None, 'soundcheck_master_processing_feedback_recheck_required': None, 'soundcheck_master_processing_bypass_level_match_required': None, 'soundcheck_master_processing_readback_required': None, 'soundcheck_master_processing_operator_confirmation_required': None, 'soundcheck_master_processing_rollback_ready': None, 'soundcheck_master_processing_no_routing_scene_snapshot_write_required': None, 'soundcheck_master_processing_bus_compression_max_target_gr_db': None, 'soundcheck_master_processing_limiter_max_target_gr_db': None, 'soundcheck_master_processing_live_promotion_status': None, 'soundcheck_master_processing_live_apply_blocked_without_wing_proof': None, 'live_dynamics_conservative_proposal': None, 'live_dynamics_stepwise_threshold_review_required': None, 'live_dynamics_max_threshold_step_db': None, 'live_dynamics_max_target_gain_reduction_db': None, 'live_dynamics_no_makeup_gain_required': None, 'live_dynamics_makeup_gain_db_limit': None, 'live_dynamics_gain_reduction_meter_readback_after_each_step_required': None, 'live_dynamics_previous_comp_model_readback_required': None, 'live_dynamics_operator_rehearsal_required': None, 'live_dynamics_live_promotion_status': None, 'live_dynamics_live_apply_blocked_without_wing_proof': None, 'review_only_parameter_keys': ['target_gain_reduction_db'], 'wing_dynamics_review_only_values': {'target_gain_reduction_db': 2.9}, 'wing_dynamics_review_only_parameter_reason_by_key': {'target_gain_reduction_db': 'target gain reduction is an operator/readback objective; WING Rack fw3.0.5 exposes compressor threshold/ratio/timing/knee/makeup/mix/envelope curve via OSC but no direct target gain-reduction write path'}, 'wing_dynamics_review_only_no_direct_osc_path': True, 'wing_dynamics_review_only_not_apply_ready': True, 'wing_dynamics_gain_reduction_objective_required': True, 'wing_dynamics_gain_reduction_objective_parameter_keys': ['target_gain_reduction_db'], 'wing_dynamics_gain_reduction_objective_values': {'target_gain_reduction_db': 2.9}, 'wing_dynamics_gain_reduction_target_db': 2.9, 'wing_dynamics_gain_reduction_max_db': None, 'wing_dynamics_gain_reduction_objective_tolerance_db': 1.0, 'wing_dynamics_gain_reduction_acceptance_min_db': 1.9, 'wing_dynamics_gain_reduction_acceptance_max_db': 3.9, 'wing_dynamics_gain_reduction_must_not_exceed_db': None, 'wing_dynamics_gain_reduction_objective_is_review_only': True, 'wing_dynamics_gain_reduction_objective_is_not_osc_write': True, 'wing_dynamics_gain_reduction_objective_no_direct_osc_write_path': True, 'wing_dynamics_gain_reduction_objective_apply_ready': False, 'wing_dynamics_gain_reduction_objective_requires_post_apply_meter_readback': True, 'wing_dynamics_gain_reduction_meter_readback_path_suffix': 'dyn/cpeak', 'wing_dynamics_gain_reduction_meter_readback_path_templates': ['/aux/{aux}/dyn/cpeak', '/aux/{aux}/dyn/cmode'], 'wing_dynamics_gain_reduction_meter_fallback_path_templates': [], 'wing_dynamics_gain_reduction_meter_value_type': 'read_only_aux_compressor_peak_percent_and_mode', 'wing_dynamics_gain_reduction_meter_is_not_apply_value': True, 'wing_dynamics_gain_reduction_meter_path_source': 'Docs/manuals/wing/WING Remote Protocols v3.0.5.pdf', 'wing_dynamics_gain_reduction_aux_depth_proxy': True, 'wing_dynamics_gain_reduction_objective_requires_operator_confirmation': True, 'wing_dynamics_gain_reduction_objective_live_promotion_status': 'blocked_until_actual_gain_reduction_meter_readback_and_operator_review', 'wing_dynamics_gain_reduction_objective_operator_check': 'confirm WING aux PSE/LA cpeak activity and cmode after the bounded depth/threshold proposal; aux exposes no dyn/gr meter', 'wing_dynamics_gain_reduction_objective_direct_osc_write_keys_excluded': ['target_gain_reduction_db'], 'wing_aux_dynamics_on_mapping_required': True, 'wing_aux_dynamics_on_osc_path_suffix': 'dyn/on', 'wing_aux_dynamics_on_osc_value': 1, 'wing_aux_dynamics_on_value_type': 'integer_bool', 'wing_aux_dynamics_on_readback_must_match_osc_value': True, 'wing_aux_dynamics_threshold_mapping_required': True, 'wing_aux_dynamics_threshold_osc_path_suffix': 'dyn/thr', 'wing_aux_dynamics_threshold_osc_value': -20.3, 'wing_aux_dynamics_threshold_value_type': 'float', 'wing_aux_dynamics_threshold_readback_must_match_osc_value': True, 'wing_aux_dynamics_depth_mapping_required': True, 'wing_aux_dynamics_depth_osc_path_suffix': 'dyn/depth', 'wing_aux_dynamics_depth_osc_value': 2.9, 'wing_aux_dynamics_depth_value_type': 'float', 'wing_aux_dynamics_depth_readback_must_match_osc_value': True, 'wing_aux_dynamics_fast_mapping_required': True, 'wing_aux_dynamics_fast_osc_path_suffix': 'dyn/fast', 'wing_aux_dynamics_fast_osc_value': 0, 'wing_aux_dynamics_fast_value_type': 'integer_bool', 'wing_aux_dynamics_fast_readback_must_match_osc_value': True}
- `soundcheck_aux_playback_pse_la_dynamics_readback_contract_wing_knee_readback_missing` (soundcheck_aux_playback_pse_la_dynamics): {'required': True, 'wing_osc_path_templates': ['/aux/{aux}/dyn/on', '/aux/{aux}/dyn/thr', '/aux/{aux}/dyn/depth', '/aux/{aux}/dyn/fast', '/aux/{aux}/dyn/peak', '/aux/{aux}/dyn/ingain', '/aux/{aux}/dyn/cpeak', '/aux/{aux}/dyn/cmode'], 'wing_rack_fw_3_0_5_native_osc_supported': True, 'dynamics_model_readback_required': False, 'dynamics_model_readback_must_equal': None, 'wing_compressor_model_readback_required': False, 'wing_compressor_model_readback_must_equal': None, 'wing_compressor_model_osc_path_suffix': None, 'wing_compressor_model_parameters_apply_only_when_model_matches': True, 'wing_aux_dynamics_pse_la_combo': True, 'wing_aux_dynamics_no_dyn_mdl_path': True, 'wing_aux_dynamics_no_ratio_attack_release_paths': True, 'wing_aux_dynamics_write_key_selection_policy': 'apply_only_selected_wing_fw_3_0_5_aux_pse_la_dynamics_value_keys', 'wing_aux_dynamics_apply_only_selected_write_keys': True, 'wing_aux_dynamics_selected_write_keys': ['wing_aux_dynamics_on_osc_value', 'wing_aux_dynamics_threshold_osc_value', 'wing_aux_dynamics_depth_osc_value', 'wing_aux_dynamics_fast_osc_value'], 'wing_aux_dynamics_forbidden_write_keys': ['wing_aux_dynamics_peak_osc_value'], 'wing_dynamics_required_model': None, 'wing_dynamics_current_model': None, 'wing_dynamics_current_enabled': None, 'wing_dynamics_current_operator_owned': None, 'wing_dynamics_model_readback_available': None, 'wing_dynamics_model_conflict': None, 'wing_dynamics_model_readback_required_for_active_slot': None, 'wing_dynamics_model_conflict_policy': None, 'wing_dynamics_model_conflict_blocker': None, 'wing_dynamics_model_readback_blocker': None, 'wing_dynamics_model_change_requires_operator_review': None, 'wing_dynamics_model_read_before_path': None, 'wing_dynamics_on_read_before_path': None, 'wing_dynamics_model_apply_only_when_current_model_matches': None, 'wing_dynamics_write_key_selection_policy': None, 'wing_dynamics_apply_only_selected_write_keys': None, 'wing_dynamics_selected_write_keys': None, 'wing_dynamics_forbidden_write_keys': None, 'read_before_paths': ['dyn/on', 'dyn/thr', 'dyn/depth', 'dyn/fast', 'dyn/peak', 'dyn/ingain', 'dyn/cpeak', 'dyn/cmode'], 'read_after_paths': ['dyn/on', 'dyn/thr', 'dyn/depth', 'dyn/fast', 'dyn/peak', 'dyn/ingain', 'dyn/cpeak', 'dyn/cmode'], 'makeup_gain_must_respect_headroom': True, 'soundcheck_dynamics_bounded_compression_proposal': True, 'soundcheck_dynamics_native_compressor_osc_only_required': True, 'soundcheck_dynamics_max_target_gain_reduction_db': 4.0, 'soundcheck_dynamics_max_makeup_gain_db': 1.0, 'soundcheck_dynamics_min_headroom_margin_db': 1.0, 'soundcheck_dynamics_required_true_peak_headroom_db': None, 'soundcheck_dynamics_required_summed_bus_true_peak_headroom_db': None, 'soundcheck_dynamics_target_gr_review_only': True, 'soundcheck_dynamics_target_gr_readback_required': True, 'soundcheck_dynamics_makeup_gain_headroom_required': True, 'soundcheck_dynamics_summed_bus_headroom_required': True, 'soundcheck_dynamics_compressor_model_readback_required': True, 'soundcheck_dynamics_gain_reduction_meter_readback_required': True, 'soundcheck_dynamics_feedback_recheck_required': True, 'soundcheck_dynamics_operator_confirmation_required': True, 'soundcheck_dynamics_readback_required': True, 'soundcheck_dynamics_rollback_ready': True, 'soundcheck_dynamics_live_promotion_status': 'blocked_until_real_wing_compressor_readback_gr_meter_headroom_feedback_recheck_and_operator_approval', 'soundcheck_dynamics_live_apply_blocked_without_wing_proof': True, 'soundcheck_master_processing_review_only_proposal': None, 'soundcheck_master_processing_native_osc_only_required': None, 'soundcheck_master_processing_target_surface': None, 'soundcheck_master_processing_types': None, 'soundcheck_master_processing_wing_apply_parameter_scope': None, 'soundcheck_master_processing_pa_feed_identity_required': None, 'soundcheck_master_processing_true_peak_headroom_recheck_required': None, 'soundcheck_master_processing_feedback_recheck_required': None, 'soundcheck_master_processing_bypass_level_match_required': None, 'soundcheck_master_processing_readback_required': None, 'soundcheck_master_processing_operator_confirmation_required': None, 'soundcheck_master_processing_rollback_ready': None, 'soundcheck_master_processing_no_routing_scene_snapshot_write_required': None, 'soundcheck_master_processing_bus_compression_max_target_gr_db': None, 'soundcheck_master_processing_limiter_max_target_gr_db': None, 'soundcheck_master_processing_live_promotion_status': None, 'soundcheck_master_processing_live_apply_blocked_without_wing_proof': None, 'live_dynamics_conservative_proposal': None, 'live_dynamics_stepwise_threshold_review_required': None, 'live_dynamics_max_threshold_step_db': None, 'live_dynamics_max_target_gain_reduction_db': None, 'live_dynamics_no_makeup_gain_required': None, 'live_dynamics_makeup_gain_db_limit': None, 'live_dynamics_gain_reduction_meter_readback_after_each_step_required': None, 'live_dynamics_previous_comp_model_readback_required': None, 'live_dynamics_operator_rehearsal_required': None, 'live_dynamics_live_promotion_status': None, 'live_dynamics_live_apply_blocked_without_wing_proof': None, 'review_only_parameter_keys': ['target_gain_reduction_db'], 'wing_dynamics_review_only_values': {'target_gain_reduction_db': 2.9}, 'wing_dynamics_review_only_parameter_reason_by_key': {'target_gain_reduction_db': 'target gain reduction is an operator/readback objective; WING Rack fw3.0.5 exposes compressor threshold/ratio/timing/knee/makeup/mix/envelope curve via OSC but no direct target gain-reduction write path'}, 'wing_dynamics_review_only_no_direct_osc_path': True, 'wing_dynamics_review_only_not_apply_ready': True, 'wing_dynamics_gain_reduction_objective_required': True, 'wing_dynamics_gain_reduction_objective_parameter_keys': ['target_gain_reduction_db'], 'wing_dynamics_gain_reduction_objective_values': {'target_gain_reduction_db': 2.9}, 'wing_dynamics_gain_reduction_target_db': 2.9, 'wing_dynamics_gain_reduction_max_db': None, 'wing_dynamics_gain_reduction_objective_tolerance_db': 1.0, 'wing_dynamics_gain_reduction_acceptance_min_db': 1.9, 'wing_dynamics_gain_reduction_acceptance_max_db': 3.9, 'wing_dynamics_gain_reduction_must_not_exceed_db': None, 'wing_dynamics_gain_reduction_objective_is_review_only': True, 'wing_dynamics_gain_reduction_objective_is_not_osc_write': True, 'wing_dynamics_gain_reduction_objective_no_direct_osc_write_path': True, 'wing_dynamics_gain_reduction_objective_apply_ready': False, 'wing_dynamics_gain_reduction_objective_requires_post_apply_meter_readback': True, 'wing_dynamics_gain_reduction_meter_readback_path_suffix': 'dyn/cpeak', 'wing_dynamics_gain_reduction_meter_readback_path_templates': ['/aux/{aux}/dyn/cpeak', '/aux/{aux}/dyn/cmode'], 'wing_dynamics_gain_reduction_meter_fallback_path_templates': [], 'wing_dynamics_gain_reduction_meter_value_type': 'read_only_aux_compressor_peak_percent_and_mode', 'wing_dynamics_gain_reduction_meter_is_not_apply_value': True, 'wing_dynamics_gain_reduction_meter_path_source': 'Docs/manuals/wing/WING Remote Protocols v3.0.5.pdf', 'wing_dynamics_gain_reduction_aux_depth_proxy': True, 'wing_dynamics_gain_reduction_objective_requires_operator_confirmation': True, 'wing_dynamics_gain_reduction_objective_live_promotion_status': 'blocked_until_actual_gain_reduction_meter_readback_and_operator_review', 'wing_dynamics_gain_reduction_objective_operator_check': 'confirm WING aux PSE/LA cpeak activity and cmode after the bounded depth/threshold proposal; aux exposes no dyn/gr meter', 'wing_dynamics_gain_reduction_objective_direct_osc_write_keys_excluded': ['target_gain_reduction_db'], 'wing_aux_dynamics_on_mapping_required': True, 'wing_aux_dynamics_on_osc_path_suffix': 'dyn/on', 'wing_aux_dynamics_on_osc_value': 1, 'wing_aux_dynamics_on_value_type': 'integer_bool', 'wing_aux_dynamics_on_readback_must_match_osc_value': True, 'wing_aux_dynamics_threshold_mapping_required': True, 'wing_aux_dynamics_threshold_osc_path_suffix': 'dyn/thr', 'wing_aux_dynamics_threshold_osc_value': -20.3, 'wing_aux_dynamics_threshold_value_type': 'float', 'wing_aux_dynamics_threshold_readback_must_match_osc_value': True, 'wing_aux_dynamics_depth_mapping_required': True, 'wing_aux_dynamics_depth_osc_path_suffix': 'dyn/depth', 'wing_aux_dynamics_depth_osc_value': 2.9, 'wing_aux_dynamics_depth_value_type': 'float', 'wing_aux_dynamics_depth_readback_must_match_osc_value': True, 'wing_aux_dynamics_fast_mapping_required': True, 'wing_aux_dynamics_fast_osc_path_suffix': 'dyn/fast', 'wing_aux_dynamics_fast_osc_value': 0, 'wing_aux_dynamics_fast_value_type': 'integer_bool', 'wing_aux_dynamics_fast_readback_must_match_osc_value': True}
- `soundcheck_aux_playback_pse_la_dynamics_missing_selected_parameter` (soundcheck_aux_playback_pse_la_dynamics): compressor_enabled
- `smart_dsp_replay_pass_rate_below_threshold` (summary): 0.993 < 1.0
