from pathlib import Path

from backend.deesser_vocal_sibilance import (
    DeesserAnalysisInput,
    analyze_deesser_need,
    build_deesser_proposals,
    estimate_sibilance_ratio,
)
from backend.processing_proposal_adapters import build_proposal_only_method_contracts
from backend.processing_registry import get_processing_method
from backend.project_goal_capability_matrix import get_project_goal_capability


def test_deesser_module_is_transport_free():
    source = Path("backend/deesser_vocal_sibilance.py").read_text()

    forbidden = [
        "WingClient",
        "EnhancedOSCClient",
        ".send(",
        "set_channel_deesser",
        "set_deesser",
        "set_eq(",
        "set_compressor(",
    ]
    for token in forbidden:
        assert token not in source


def test_vocal_sibilance_creates_proposal_only_recommendation():
    recommendation = analyze_deesser_need(
        DeesserAnalysisInput(
            channel_id=7,
            channel_label="Lead Vox",
            source_role="lead_vocal",
            band_energy_db={
                "low_mid": -38.0,
                "mid": -30.0,
                "presence": -24.0,
                "sibilance": -16.0,
                "air": -18.0,
            },
            sibilance_excess_db=4.5,
            peak_dbfs=-7.0,
            contribution_share=0.72,
            persistence_cycles=3,
            true_peak_headroom_db=5.0,
        )
    )

    assert recommendation.status == "proposal_only"
    assert recommendation.action_type == "deesser_proposal"
    assert 5500.0 <= recommendation.frequency_hz <= 10500.0
    assert -42.0 <= recommendation.threshold_db <= -12.0
    assert 1.5 <= recommendation.ratio <= 6.0
    assert recommendation.confidence >= 0.45
    assert recommendation.blockers == ()
    assert recommendation.evidence["transport_free"] is True

    proposal = recommendation.to_operator_proposal()
    assert proposal["method_id"] == "deesser_vocal_sibilance"
    assert proposal["can_import_to_queue"] is False
    assert proposal["can_apply_live"] is False
    assert "requested_change" not in proposal
    assert "unsupported_value_type:deesser" in proposal["blockers"]
    assert proposal["evidence"]["live_write"] is False


def test_non_vocal_sibilance_is_not_deesser_proposal():
    recommendation = analyze_deesser_need(
        DeesserAnalysisInput(
            channel_id=3,
            channel_label="OH L",
            source_role="overhead",
            band_energy_db={"sibilance": -12.0, "presence": -24.0},
            sibilance_excess_db=7.0,
            contribution_share=0.8,
            persistence_cycles=3,
        )
    )

    assert recommendation.status == "analysis_only"
    assert "source_not_vocal" in recommendation.blockers


def test_low_sibilance_vocal_is_analysis_only():
    recommendation = analyze_deesser_need(
        DeesserAnalysisInput(
            channel_id=8,
            source_role="backing_vocal",
            band_energy_db={"sibilance": -42.0, "presence": -25.0, "mid": -24.0},
            sibilance_excess_db=0.5,
            persistence_cycles=1,
        )
    )

    assert recommendation.status == "analysis_only"
    assert "sibilance_below_threshold" in recommendation.blockers


def test_deesser_ratio_estimator_uses_sibilance_band_share():
    ratio = estimate_sibilance_ratio(
        {"mid": -30.0, "presence": -28.0, "sibilance": -16.0, "air": -28.0}
    )

    assert ratio > 0.5


def test_build_deesser_proposals_filters_to_proposal_only_items():
    proposals = build_deesser_proposals(
        [
            DeesserAnalysisInput(
                channel_id=7,
                source_role="lead_vocal",
                band_energy_db={"sibilance": -15.0, "presence": -25.0},
                sibilance_excess_db=4.0,
                contribution_share=0.7,
                persistence_cycles=3,
            ),
            DeesserAnalysisInput(
                channel_id=3,
                source_role="overhead",
                band_energy_db={"sibilance": -15.0, "presence": -25.0},
                sibilance_excess_db=4.0,
                contribution_share=0.7,
                persistence_cycles=3,
            ),
        ]
    )

    assert len(proposals) == 1
    assert proposals[0]["channel"] == 7


def test_deesser_is_registered_as_proposal_only_goal_capability():
    method = get_processing_method("deesser_vocal_sibilance")
    capability = get_project_goal_capability("deessing")
    contracts = {item["method_id"]: item for item in build_proposal_only_method_contracts()}

    assert method is not None
    assert method.status == "proposal_only"
    assert method.can_write_live is False
    assert capability is not None
    assert capability.status == "proposal_only"
    assert capability.method_ids == ("deesser_vocal_sibilance",)
    assert contracts["deesser_vocal_sibilance"]["proposal_contract"]["value_type"] == "deesser"
