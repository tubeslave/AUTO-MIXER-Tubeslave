# TUB-309: Shadow-Mode Comparison Report Schema

Date: 2026-05-16

## Scope

This note defines the dry-run-only report contract written by
`ai_mixing_pipeline/reports/shadow_report_writer.py`.

Safety boundary:
- These reports are replay artifacts only.
- They must not write to a mixer.
- They must preserve evidence proving `dry_run_only=true` and
  `live_mixer_writes=false`.
- Safety evidence is fail-closed: manifest flags and every proposal row are
  checked together before a report can be considered promotion-ready.

## Output files

`write_shadow_mode_reports(...)` writes:

1. `shadow_mode_comparison.json`
2. `rollback_simulation_report.json`
3. `promotion_readiness_report.json`
4. `confidence_trajectory.json`
5. `shadow_mode_summary.json`
6. `shadow_mode_summary.md`

All JSON files currently use `schema_version=replay_shadow_report/v1`.

## Comparison report

`shadow_mode_comparison.json` is the main operator-vs-AI comparison payload.

Top-level fields:
- `manifest_id`
- `trim_snapshot_id`
- `graph_checkpoint_id`
- `selected_proposal_id`
- `selected_family`
- `operator_count`
- `ai_count`
- `operator_rows`
- `ai_rows`
- `ai_would_do_this`
- `operator_action_capture`
- `disagreements`
- `disagreement_summary`
- `evidence`

The comparison `evidence` block includes:
- `manifest_dry_run_only`
- `manifest_live_mixer_writes`
- `proposal_live_mixer_write_count`
- `proposal_non_dry_run_count`
- aggregated `dry_run_only`
- aggregated `live_mixer_writes`

The aggregated fields are conservative. If any proposal row contains
`live_mixer_writes=true` or a transport policy other than `dry_run_only`, the
report is marked not dry-run-only and not promotion-ready.

Each row in `operator_rows`, `ai_rows`, `ai_would_do_this`, and
`operator_action_capture` preserves dry-run evidence already present in replay
artifacts:
- `proposal_id`
- `source_proposal_id`
- `family`
- `source_system`
- `action_type`
- `target`
- `current_state`
- `requested_state`
- `source_payload`
- `confidence`
- `rank`
- `selected`
- `replay_signature`
- `replay_correlation_id`
- `auto_apply_blocked`
- `transport_policy`
- `live_mixer_writes`
- `safety`

Notes:
- `ai_would_do_this` points at the highest-ranked non-operator proposal.
- `operator_action_capture` points at the selected operator row when one exists,
  otherwise the highest-ranked operator row.
- `source_payload` is the preferred place to preserve raw operator guidance such
  as `rec_id`, `type`, `message`, and other LiveApply-adjacent fields.

## Disagreement taxonomy

`disagreements` only records cases where the replay-selected outcome diverges in
a way that matters for supervised shadow review.

Currently emitted codes:
- `operator_selected`
  - taxonomy: `operator_override`
  - meaning: an operator-side proposal outranked the top non-operator proposal
    in the dry-run replay.
- `operator_only`
  - taxonomy: `operator_only`
  - meaning: the replay selected an operator-side proposal and no AI-side
    proposal was available for comparison.

`disagreement_summary` aggregates the disagreement list into bounded metrics:
- `comparison_count`
- `disagreement_count`
- `disagreement_rate`
- `classes.by_code`
- `classes.by_taxonomy`
- candidate-presence booleans for operator and AI rows

## Promotion-readiness evidence

`promotion_readiness_report.json` carries:
- readiness boolean
- readiness score
- `dry_run_only`
- `live_mixer_writes`
- validation severity counts
- disagreement count
- blocked rollback reason count
- `aggregation`
- `promotion_readiness_gates`
- `gate_summary`
- evidence bundle

`aggregation` includes:
- `disagreement_rate`
- `disagreement_classes`
- `confidence_trend`
- `rollback_frequency`
- `audit_coverage`

`promotion_readiness_gates` lists explicit blocking vs advisory checks. Current
blocking gates cover:
- `dry_run_only`
- `no_live_mixer_writes`
- `validation_errors_clear`
- `audit_coverage_complete`
- `readiness_score_threshold`

Current advisory gates cover:
- `disagreement_rate_clear`
- `rollback_review_clear`

The evidence bundle includes:
- `trim_snapshot_id`
- `graph_checkpoint_id`
- `timeline_id`
- `timeline_signature`
- `manifest_signature`
- `selected_proposal_id`
- `disagreement_codes`
- `rollback_blocked_reasons`
- `validation_findings`
- `replay_correlation_ids`

## Rollback report

`rollback_simulation_report.json` preserves:
- `trim_snapshot_id`
- `graph_checkpoint_id`
- `selected_proposal_id`
- `proposed_action_timeline`
- `possible_rollbacks`
- `blocked_reasons`
- `replay_correlation_ids`

This report is still replay-only. `blocked_reasons` should be reviewed before
any future supervised-live promotion work.

## Confidence trajectory

`confidence_trajectory.json` stores rank-ordered trajectory rows plus:
- `manifest_id`
- `selected_proposal_id`
- `selected_family`
- `top_operator_proposal_id`
- `top_ai_proposal_id`
- `trend_summary`

## Fixtures

Deterministic offline fixture cases live in
`tests/shadow_mode_report_fixtures.py`.

Current scenarios:
1. `operator_selected_shadow_override`
2. `ai_selected_ready_shadow_run`
3. `speech_environment_ai_only_shadow_run`
4. `music_environment_ai_only_shadow_run`

Each fixture pins:
- replay manifest
- replay executor result
- validation report
- fixed `generated_at`
- expected comparison, rollback, and readiness evidence
