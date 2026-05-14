from __future__ import annotations

from ai_mixing_pipeline.decision_layer.replay_stress_scale import (
    REPLAY_STRESS_SCALE_SCHEMA_VERSION,
    ReplayStressScenarioSpec,
    run_replay_stress_scenario,
    run_replay_stress_suite,
)


def test_replay_stress_scenario_stays_replay_only_and_deterministic() -> None:
    result = run_replay_stress_scenario(
        ReplayStressScenarioSpec(
            name="unit_replay_stress",
            analyzer_events=18,
            critic_passes=4,
            decision_burst_size=18,
            executor_batches=3,
            checkpoint_copies=2,
            rewind_iterations=3,
            branch_variants=2,
            scenario_repetitions=3,
        )
    )

    assert result.schema_version == REPLAY_STRESS_SCALE_SCHEMA_VERSION
    assert result.governance["dry_run_only"] is True
    assert result.governance["live_mixer_writes"] is False
    assert result.governance["executor_dry_run_only"] is True
    assert result.governance["transport_mutation_allowed"] is False
    assert result.determinism["ranking_stable"] is True
    assert result.determinism["executor_trace_stable"] is True
    assert result.determinism["concurrent_trace_stable"] is True
    assert result.determinism["policy_validation_valid"] is True
    assert result.inventory["proposal_count"] == 18
    assert result.inventory["rewind_context_count"] == 3
    assert result.inventory["concurrent_repetition_count"] == 3
    assert result.saturation["max_critic_inventory_per_proposal"] == 4
    assert result.latencies_ms["total"] >= 0.0


def test_replay_stress_suite_aggregates_multiple_scenarios() -> None:
    suite = run_replay_stress_suite(
        [
            ReplayStressScenarioSpec(name="suite_a", analyzer_events=10, decision_burst_size=10),
            ReplayStressScenarioSpec(name="suite_b", analyzer_events=14, decision_burst_size=14, critic_passes=5),
        ]
    )

    assert suite.schema_version == REPLAY_STRESS_SCALE_SCHEMA_VERSION
    assert suite.governance["all_dry_run_only"] is True
    assert suite.governance["all_live_mixer_writes_disabled"] is True
    assert suite.aggregate["scenario_count"] == 2
    assert suite.aggregate["total_proposals"] == 24
    assert suite.aggregate["all_ranking_stable"] is True
    assert suite.aggregate["all_executor_traces_stable"] is True
    assert suite.aggregate["all_policy_validation_valid"] is True


def test_replay_stress_scale_metrics_grow_with_larger_workload() -> None:
    small = run_replay_stress_scenario(
        ReplayStressScenarioSpec(name="scale_small", analyzer_events=8, decision_burst_size=8, critic_passes=2)
    )
    large = run_replay_stress_scenario(
        ReplayStressScenarioSpec(name="scale_large", analyzer_events=20, decision_burst_size=20, critic_passes=5)
    )

    assert large.inventory["proposal_count"] > small.inventory["proposal_count"]
    assert large.inventory["timeline_event_count"] > small.inventory["timeline_event_count"]
    assert large.saturation["timeline_bytes"] > small.saturation["timeline_bytes"]
    assert large.saturation["manifest_bytes"] > small.saturation["manifest_bytes"]
    assert large.saturation["max_critic_inventory_per_proposal"] > small.saturation["max_critic_inventory_per_proposal"]


def test_replay_stress_scenario_writes_shadow_reports_when_enabled(tmp_path) -> None:
    result = run_replay_stress_scenario(
        ReplayStressScenarioSpec(
            name="reporting_stress",
            analyzer_events=8,
            critic_passes=2,
            decision_burst_size=8,
            executor_batches=1,
            checkpoint_copies=1,
            rewind_iterations=1,
            branch_variants=1,
            scenario_repetitions=1,
            report_dir=str(tmp_path),
        )
    )

    shadow_reports = result.artifacts["shadow_reports"]
    assert set(shadow_reports.keys()) == {
        "shadow_mode_comparison",
        "confidence_trajectory",
        "rollback_simulation",
        "promotion_readiness",
        "shadow_mode_summary",
    }
    for report_path in shadow_reports.values():
        path = tmp_path / report_path
        assert path.exists()
