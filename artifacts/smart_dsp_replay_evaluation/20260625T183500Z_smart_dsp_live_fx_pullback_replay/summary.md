# Smart DSP Replay Evaluation

- schema: `automixer.smart_dsp_replay_evaluation.v1`
- status: `passed`
- method_id: `smart_dsp_strategy_engine`
- scenario_count: `62`
- passed_scenario_count: `62`
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
- candidate_count: `10`
- selected_total_score: `0.901`
- global_blockers: `none`

### soundcheck_vocal_noise_floor_expander

- status: `passed`
- expected_selected_strategy_id: `vocal_phrase_noise_downward_expander`
- actual_selected_strategy_id: `vocal_phrase_noise_downward_expander`
- actual_method_family: `expander`
- candidate_count: `7`
- selected_total_score: `0.878`
- global_blockers: `none`

### soundcheck_line_noise_hum_notch

- status: `passed`
- expected_selected_strategy_id: `mains_hum_notch_stack`
- actual_selected_strategy_id: `mains_hum_notch_stack`
- actual_method_family: `eq`
- candidate_count: `6`
- selected_total_score: `0.893`
- global_blockers: `none`

### soundcheck_dynamic_muddy_vocal

- status: `passed`
- expected_selected_strategy_id: `vocal_dynamic_low_mid_de_mud`
- actual_selected_strategy_id: `vocal_dynamic_low_mid_de_mud`
- actual_method_family: `dynamic_eq`
- candidate_count: `7`
- selected_total_score: `0.859`
- global_blockers: `none`

### soundcheck_dull_vocal_presence_lift

- status: `passed`
- expected_selected_strategy_id: `soundcheck_presence_air_lift`
- actual_selected_strategy_id: `soundcheck_presence_air_lift`
- actual_method_family: `eq`
- candidate_count: `3`
- selected_total_score: `0.801`
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

### soundcheck_harsh_sibilant_vocal_split_chain

- status: `passed`
- expected_selected_strategy_id: `vocal_de_esser_dynamic_presence_combo`
- actual_selected_strategy_id: `vocal_de_esser_dynamic_presence_combo`
- actual_method_family: `dynamic_eq`
- candidate_count: `4`
- selected_total_score: `0.886`
- global_blockers: `none`

### live_sibilant_vocal

- status: `passed`
- expected_selected_strategy_id: `vocal_de_esser`
- actual_selected_strategy_id: `vocal_de_esser`
- actual_method_family: `de_esser`
- candidate_count: `3`
- selected_total_score: `0.878`
- global_blockers: `none`

### live_feedback_risk_beats_sibilance_correction

- status: `passed`
- expected_selected_strategy_id: `feedback_gain_pullback`
- actual_selected_strategy_id: `feedback_gain_pullback`
- actual_method_family: `gain_fader`
- candidate_count: `6`
- selected_total_score: `0.838`
- global_blockers: `none`

### soundcheck_harsh_dynamic_vocal_envelope

- status: `passed`
- expected_selected_strategy_id: `vocal_compressor_envelope_smoothing`
- actual_selected_strategy_id: `vocal_compressor_envelope_smoothing`
- actual_method_family: `dynamics`
- candidate_count: `7`
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
- expected_selected_strategy_id: `drum_cymbal_dynamic_de_edge`
- actual_selected_strategy_id: `drum_cymbal_dynamic_de_edge`
- actual_method_family: `dynamic_eq`
- candidate_count: `3`
- selected_total_score: `0.862`
- global_blockers: `none`

### live_harsh_overhead_cymbal_static_air_cut

- status: `passed`
- expected_selected_strategy_id: `drum_cymbal_static_air_cut`
- actual_selected_strategy_id: `drum_cymbal_static_air_cut`
- actual_method_family: `eq`
- candidate_count: `3`
- selected_total_score: `0.866`
- global_blockers: `none`

### inferred_vocal_identity_requires_confirmation

- status: `passed`
- expected_selected_strategy_id: `vocal_de_esser`
- actual_selected_strategy_id: `vocal_de_esser`
- actual_method_family: `de_esser`
- candidate_count: `3`
- selected_total_score: `0.874`
- global_blockers: `source_identity_not_confirmed, source_identity_confidence_low`

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
- expected_selected_strategy_id: `gentle_channel_compression`
- actual_selected_strategy_id: `gentle_channel_compression`
- actual_method_family: `dynamics`
- candidate_count: `3`
- selected_total_score: `0.813`
- global_blockers: `none`

### soundcheck_bass_sustain_leveling_compression

- status: `passed`
- expected_selected_strategy_id: `bass_sustain_leveling_compression`
- actual_selected_strategy_id: `bass_sustain_leveling_compression`
- actual_method_family: `dynamics`
- candidate_count: `4`
- selected_total_score: `0.878`
- global_blockers: `none`

### live_guitar_pick_sustain_compression

- status: `passed`
- expected_selected_strategy_id: `guitar_pick_sustain_compression`
- actual_selected_strategy_id: `guitar_pick_sustain_compression`
- actual_method_family: `dynamics`
- candidate_count: `4`
- selected_total_score: `0.872`
- global_blockers: `none`

### soundcheck_quiet_vocal_level_plane_gain_up

- status: `passed`
- expected_selected_strategy_id: `soundcheck_bounded_gain_up`
- actual_selected_strategy_id: `soundcheck_bounded_gain_up`
- actual_method_family: `gain_fader`
- candidate_count: `3`
- selected_total_score: `0.832`
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

### live_quiet_vocal_bus_group_gain_hold

- status: `passed`
- expected_selected_strategy_id: `hold_group_bus_for_member_evidence`
- actual_selected_strategy_id: `hold_group_bus_for_member_evidence`
- actual_method_family: `analysis_only`
- candidate_count: `3`
- selected_total_score: `0.813`
- global_blockers: `none`

### kick_bass_masking_low_headroom

- status: `passed`
- expected_selected_strategy_id: `kick_bass_pocket_eq_split`
- actual_selected_strategy_id: `kick_bass_pocket_eq_split`
- actual_method_family: `eq`
- candidate_count: `7`
- selected_total_score: `0.858`
- global_blockers: `true_peak_headroom_low`

### soundcheck_vocal_intelligibility_dynamic_unmasking

- status: `passed`
- expected_selected_strategy_id: `vocal_intelligibility_dynamic_unmasking`
- actual_selected_strategy_id: `vocal_intelligibility_dynamic_unmasking`
- actual_method_family: `dynamic_eq`
- candidate_count: `3`
- selected_total_score: `0.893`
- global_blockers: `none`

### live_vocal_intelligibility_static_pocket

- status: `passed`
- expected_selected_strategy_id: `vocal_intelligibility_static_pocket_eq`
- actual_selected_strategy_id: `vocal_intelligibility_static_pocket_eq`
- actual_method_family: `eq`
- candidate_count: `3`
- selected_total_score: `0.894`
- global_blockers: `none`

### soundcheck_kick_bass_dynamic_pocket

- status: `passed`
- expected_selected_strategy_id: `kick_bass_dynamic_pocket_ducking`
- actual_selected_strategy_id: `kick_bass_dynamic_pocket_ducking`
- actual_method_family: `dynamic_eq`
- candidate_count: `3`
- selected_total_score: `0.87`
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
- expected_selected_strategy_id: `manual_transient_hold_for_operator`
- actual_selected_strategy_id: `manual_transient_hold_for_operator`
- actual_method_family: `analysis_only`
- candidate_count: `3`
- selected_total_score: `0.826`
- global_blockers: `true_peak_headroom_low`

### noisy_tom_tail_risk

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

### extreme_noisy_tom_manual_mute_review

- status: `passed`
- expected_selected_strategy_id: `manual_tom_mute_between_hits`
- actual_selected_strategy_id: `manual_tom_mute_between_hits`
- actual_method_family: `mute`
- candidate_count: `5`
- selected_total_score: `0.846`
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
- expected_selected_strategy_id: `shorten_fx_decay_filter`
- actual_selected_strategy_id: `shorten_fx_decay_filter`
- actual_method_family: `fx`
- candidate_count: `5`
- selected_total_score: `0.841`
- global_blockers: `none`

### soundcheck_delay_smear_fx_ducking_cleanup

- status: `passed`
- expected_selected_strategy_id: `duck_delay_return_cleanup`
- actual_selected_strategy_id: `duck_delay_return_cleanup`
- actual_method_family: `fx`
- candidate_count: `5`
- selected_total_score: `0.856`
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
- expected_selected_strategy_id: `center_hold_for_phase_safety`
- actual_selected_strategy_id: `center_hold_for_phase_safety`
- actual_method_family: `pan`
- candidate_count: `2`
- selected_total_score: `0.822`
- global_blockers: `none`

### soundcheck_phase_safe_pan_delta

- status: `passed`
- expected_selected_strategy_id: `phase_safe_pan_target`
- actual_selected_strategy_id: `phase_safe_pan_target`
- actual_method_family: `pan`
- candidate_count: `2`
- selected_total_score: `0.799`
- global_blockers: `none`

### live_pan_change_holds_for_operator_review

- status: `passed`
- expected_selected_strategy_id: `center_hold_for_phase_safety`
- actual_selected_strategy_id: `center_hold_for_phase_safety`
- actual_method_family: `pan`
- candidate_count: `2`
- selected_total_score: `0.851`
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
- expected_selected_strategy_id: `mono_compatibility_width_reduction`
- actual_selected_strategy_id: `mono_compatibility_width_reduction`
- actual_method_family: `pan`
- candidate_count: `6`
- selected_total_score: `0.831`
- global_blockers: `none`

### live_phase_mono_risk_beats_harshness_eq

- status: `passed`
- expected_selected_strategy_id: `hold_phase_mono_processing_for_operator`
- actual_selected_strategy_id: `hold_phase_mono_processing_for_operator`
- actual_method_family: `analysis_only`
- candidate_count: `6`
- selected_total_score: `0.818`
- global_blockers: `none`

### master_true_peak_overload

- status: `passed`
- expected_selected_strategy_id: `master_trim_for_true_peak_headroom`
- actual_selected_strategy_id: `master_trim_for_true_peak_headroom`
- actual_method_family: `master_bus`
- candidate_count: `3`
- selected_total_score: `0.878`
- global_blockers: `none`

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
- actual_method_family: `master_bus`
- candidate_count: `1`
- selected_total_score: `0.831`
- global_blockers: `none`

### master_presence_broad_eq_cleanup

- status: `passed`
- expected_selected_strategy_id: `master_presence_broad_eq_cleanup`
- actual_selected_strategy_id: `master_presence_broad_eq_cleanup`
- actual_method_family: `master_bus`
- candidate_count: `1`
- selected_total_score: `0.838`
- global_blockers: `none`

### master_dynamic_range_bus_compression

- status: `passed`
- expected_selected_strategy_id: `gentle_bus_compression`
- actual_selected_strategy_id: `gentle_bus_compression`
- actual_method_family: `master_bus`
- candidate_count: `1`
- selected_total_score: `0.836`
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
