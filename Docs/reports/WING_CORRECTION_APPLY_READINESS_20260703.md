# WING Correction Apply Readiness

Generated: `2026-07-03T06:45:12Z`

## Current Boundary

- Current supervised policy write kinds: `fader`, `gain`
- Direct live widening allowed: `false`
- Release-ready live families require complete apply, physical readback, rollback, and emergency-stop/block evidence.
- New correction families require separate ADR, bounded proposal contract, physical readback, rollback, emergency-stop/block proof, and explicit operator approval.

## Summary

- Family count: `11`
- Policy-allowed supervised families: `2`
- Release-ready live families: `0`
- Blocked families: `11`
- Next blocked family: `fader_balance`

## Families

| Family | Stage | Policy supervised | Release live ready | Write kinds | Blockers | Next action |
| --- | --- | --- | --- | --- | --- | --- |
| Channel fader and balance | `supervised_candidate` | `true` | `false` | `fader` | `physical_rollback_proof_required`, `emergency_stop_or_blocked_apply_proof_required`, `release_readiness_not_closed` | Rerun the narrow fader proof until apply, physical readback, rollback, and emergency-stop/block evidence all pass; do not reuse this proof for other families. |
| Input trim and gain staging | `supervised_candidate` | `true` | `false` | `gain` | `physical_gain_proof_required`, `physical_rollback_proof_required`, `emergency_stop_or_blocked_apply_proof_required`, `release_readiness_not_closed` | Run a separate negative or headroom-safe gain proof with physical readback, rollback, and emergency-stop/block evidence before claiming gain readiness beyond policy allowance. |
| Pan and stereo placement | `proposal_only` | `false` | `false` | `pan` | `not_in_wing_deployment_live_write_kinds`, `separate_hardware_proof_required`, `mono_phase_safety_required` | Promote first among non-fader/gain families only after a tiny operator-approved pan proof with physical readback, rollback, and mono/phase safety evidence. |
| Mute and unmute management | `proposal_only` | `false` | `false` | `mute` | `not_in_wing_deployment_live_write_kinds`, `explicit_mute_operator_ux_required`, `separate_hardware_proof_required` | Keep mute changes review-only until there is a mute-specific UX, intent confirmation, physical readback, rollback, and emergency-state proof. |
| EQ, HPF, and filtering | `proposal_only` | `false` | `false` | `eq`, `hpf` | `not_in_wing_deployment_live_write_kinds`, `eq_band_identity_required`, `feedback_and_headroom_safety_required`, `separate_hardware_proof_required` | Do not apply EQ live until one bounded cut-only band proof verifies address mapping, readback, rollback, protected-band policy, and feedback safety. |
| Compressor, dynamics, and de-esser | `proposal_only` | `false` | `false` | `compressor`, `deesser` | `not_in_wing_deployment_live_write_kinds`, `makeup_gain_headroom_guard_required`, `gain_reduction_evidence_required`, `separate_hardware_proof_required` | Keep compressor/de-esser review-only until a threshold-only or makeup-neutral proof verifies dynamics readback, GR/headroom safety, and rollback. |
| Gate and expander | `proposal_only` | `false` | `false` | `gate`, `expander` | `not_in_wing_deployment_live_write_kinds`, `source_identity_trust_required`, `chatter_safety_required`, `separate_hardware_proof_required` | Use offline/live telemetry to prove false-open/false-close behavior; only then run a bounded threshold proof with rollback. |
| Phase, polarity, and delay | `proposal_only` | `false` | `false` | `phase`, `polarity`, `delay` | `not_in_wing_deployment_live_write_kinds`, `stage_position_or_measurement_context_required`, `mono_sum_safety_required`, `separate_hardware_proof_required` | Keep phase/delay proposals review-only until measurement context, mono-sum safety, physical readback, and rollback are proven. |
| FX, inserts, sends, and returns | `proposal_only` | `false` | `false` | `fx`, `insert`, `send` | `not_in_wing_deployment_live_write_kinds`, `fx_parameter_schema_required`, `monitor_feedback_safety_required`, `separate_hardware_proof_required` | Keep FX/send changes proposal-only until send/return readback, monitor safety, FX parameter schema, feedback checks, and rollback are proven. |
| Bus, group, DCA, matrix, and master output | `proposal_only` | `false` | `false` | `bus_fader`, `dca_fader`, `matrix_fader`, `master_fader` | `not_in_wing_deployment_live_write_kinds`, `pa_feed_safety_required`, `assignment_freshness_required`, `separate_hardware_proof_required` | Do not promote broad output surfaces until channel-level families are proven; require PA-feed exclusion, true-peak safety, readback, rollback, and operator UX. |
| Routing, scenes, snapshots, and presets | `proposal_only` | `false` | `false` | `routing`, `scene`, `snapshot`, `preset` | `not_in_wing_deployment_live_write_kinds`, `broad_console_state_risk`, `scoped_diff_required`, `separate_hardware_proof_required` | Keep routing/scene/preset behavior manual-review only until there is a scoped diff, full readback, rollback/restore plan, and explicit operator runbook. |

## Safety Rule

Do not enable all WING correction methods by adding them to one global live-write list. Promote exactly one family at a time after its own hardware proof.
