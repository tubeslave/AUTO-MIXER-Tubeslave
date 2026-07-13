from __future__ import annotations

import pytest

from autofoh_safety import (
    BusCompressorAdjust,
    BusEQMove,
    BusFaderMove,
    ChannelEQMove,
    ChannelFaderMove,
    ChannelGainMove,
    ChannelMuteMove,
    CompressorAdjust,
    DCAFaderMove,
    DelayAdjust,
    GateAdjust,
    HighPassAdjust,
    MasterFaderMove,
    NoOp,
    PanAdjust,
    PolarityAdjust,
    SendLevelAdjust,
)
from soundcheck_transaction_plan import (
    build_soundcheck_transaction_plans,
    operation_from_soundcheck_action,
    soundcheck_manifest_content_sha256,
)


def test_every_soundcheck_module_maps_to_absolute_semantic_operation():
    actions = [
        ChannelGainMove("gain", 3, -4.0),
        HighPassAdjust("hpf", 3, 90.0, True),
        ChannelEQMove("eq", 3, 2, 350.0, -2.5, 1.2),
        GateAdjust("gate", 3, -42.0, 18.0, 5.0, 40.0, 120.0, True),
        CompressorAdjust("comp", 3, -18.0, 3.0, 20.0, 100.0, 0.0, True),
        PolarityAdjust("phase", 3, True),
        DelayAdjust("delay", 3, 1.2, True),
        PanAdjust("pan", 3, -0.25),
        ChannelFaderMove("fader", 3, -12.0),
        SendLevelAdjust("fx", 3, 2, -18.0),
        BusEQMove("bus eq", 1, 1, 250.0, -1.5, 1.0),
        BusCompressorAdjust("bus comp", 1, -12.0, 2.0, 30.0, 120.0),
        BusFaderMove("bus fader", 1, -6.0),
        DCAFaderMove("dca", 1, -3.0),
        MasterFaderMove("master", 1, -8.0),
        ChannelMuteMove("mute", 3, True),
    ]

    operations = [operation_from_soundcheck_action(action) for action in actions]

    assert {operation.family_id for operation in operations} == {
        "gain_trim",
        "eq_hpf",
        "gate_expander",
        "dynamics_compressor",
        "pan_phase_delay",
        "fader",
        "fx_sends_inserts",
        "bus_group_dca_matrix",
        "master_output",
        "mute",
    }
    assert all(operation.metadata["absolute_target"] for operation in operations)
    assert all(operation.metadata["live_mixer_writes"] is False for operation in operations)
    assert operations[2].target_value == {
        "band": 2,
        "freq_hz": 350.0,
        "gain_db": -2.5,
        "q": 1.2,
    }


def test_dependency_order_is_deterministic_and_high_risk_is_separate():
    actions = [
        ChannelMuteMove("mute", 2, True),
        ChannelFaderMove("fader", 2, -10.0),
        CompressorAdjust("comp", 2, -18.0, 3.0, 10.0, 80.0),
        GateAdjust("gate", 2, -40.0, 20.0, 5.0, 30.0, 100.0),
        ChannelEQMove("eq", 2, 1, 200.0, -2.0, 1.0),
        ChannelGainMove("gain", 2, -3.0),
        MasterFaderMove("master", 1, -8.0),
        SendLevelAdjust("fx", 2, 1, -16.0),
    ]

    first = build_soundcheck_transaction_plans(actions)
    second = build_soundcheck_transaction_plans(reversed(actions))

    assert len(first.normal_plans) == 5
    assert all(len(plan.operations) == 1 for plan in first.plans)
    assert [plan.operations[0].family_id for plan in first.normal_plans] == [
        "gain_trim",
        "fader",
        "gate_expander",
        "eq_hpf",
        "dynamics_compressor",
    ]
    assert [
        plan.operations[0].family_id for plan in first.separate_approval_plans
    ] == ["fx_sends_inserts", "master_output", "mute"]
    assert [plan.transaction_id for plan in first.plans] == [
        plan.transaction_id for plan in second.plans
    ]
    assert [plan.operations[0].metadata["stage"] for plan in first.plans] == [
        "foundation_level",
        "foundation_level",
        "source_control",
        "tonal_dynamics",
        "tonal_dynamics",
        "spatial_fx",
        "destructive_high_risk",
        "destructive_high_risk",
    ]
    payload = first.to_dict()
    assert [
        plan["operations"][0]["family_id"]
        for plan in payload["plans"]
    ] == [
        "gain_trim",
        "fader",
        "gate_expander",
        "eq_hpf",
        "dynamics_compressor",
        "fx_sends_inserts",
        "master_output",
        "mute",
    ]
    assert payload["rollback_boundary"] == "semantic_child_transaction"
    assert payload["stage_atomicity"] is False
    assert payload["plan_count"] == payload["operation_count"] == 8
    assert payload["schema_version"] == "soundcheck_batch_manifest/v2"
    assert payload["manifest_content_sha256"] == (
        soundcheck_manifest_content_sha256(payload)
    )
    assert {
        plan["manifest_content_sha256"] for plan in payload["plans"]
    } == {payload["manifest_content_sha256"]}


def test_serialized_same_stage_order_is_semantic_not_transaction_hash():
    payload = build_soundcheck_transaction_plans(
        [
            CompressorAdjust(
                "compress after tone",
                3,
                -18.0,
                3.0,
                20.0,
                100.0,
                0.0,
                True,
            ),
            ChannelEQMove("tone first", 3, 1, 250.0, -2.0, 1.0),
        ],
        run_id="same-stage-semantic-order",
    ).to_dict()

    assert [
        plan["operations"][0]["family_id"]
        for plan in payload["plans"]
    ] == ["eq_hpf", "dynamics_compressor"]


def test_serialized_bundle_is_explicitly_proposal_only_and_exposes_lifecycle():
    bundle = build_soundcheck_transaction_plans(
        [ChannelEQMove("remove mud", 4, 2, 300.0, -2.0, 1.1)]
    )

    payload = bundle.to_dict()

    assert payload["execution_allowed"] is False
    assert payload["live_mixer_writes"] is False
    assert payload["transport_policy"] == "proposal_only"
    assert payload["value_semantics"] == "absolute_target"
    contract = payload["plans"][0]["operations"][0]["family_contract"]
    assert contract["lifecycle"] == "hardware_proven"
    assert contract["normal_supervised_apply_allowed"] is False
    assert contract["live_allowed_write_kinds"] == []
    assert contract["blockers"] == [
        "owner_readiness_promotion_required",
        "eq_live_apply_is_cut_only",
    ]


def test_run_identity_is_unique_while_content_fingerprint_stays_stable():
    actions = [ChannelFaderMove("same correction", 2, -12.0)]

    first = build_soundcheck_transaction_plans(actions, run_id="physical-run-a")
    second = build_soundcheck_transaction_plans(actions, run_id="physical-run-b")

    assert first.plan_fingerprint == second.plan_fingerprint
    assert first.run_id != second.run_id
    assert first.plans[0].transaction_id != second.plans[0].transaction_id


@pytest.mark.parametrize("revision", [True, 0, -1, 1.5, "1"])
def test_plan_revision_is_a_strict_positive_integer(revision):
    with pytest.raises(ValueError, match="plan_revision_must_be_positive_integer"):
        build_soundcheck_transaction_plans(
            [ChannelFaderMove("revision", 2, -12.0)],
            run_id="revision-test",
            plan_revision=revision,
        )


def test_duplicate_logical_correction_is_canonically_collapsed():
    action = ChannelFaderMove("duplicate", 2, -12.0)

    bundle = build_soundcheck_transaction_plans(
        [action, action],
        run_id="duplicate-test",
    )

    assert len(bundle.plans) == 1
    assert bundle.unsupported_actions == ()

    rejected = ChannelEQMove("duplicate boost", 2, 1, 200.0, 2.0, 1.0)
    rejected_bundle = build_soundcheck_transaction_plans(
        [rejected, rejected],
        run_id="duplicate-rejected-test",
    )
    assert len(rejected_bundle.unsupported_actions) == 1


def test_independent_reasons_for_same_absolute_target_merge_with_provenance():
    bundle = build_soundcheck_transaction_plans(
        [
            ChannelFaderMove("level plane", 2, -12.0),
            ChannelFaderMove("masking balance", 2, -12.0),
        ],
        run_id="agreeing-targets",
    )

    assert len(bundle.plans) == 1
    assert bundle.unsupported_actions == ()
    operation = bundle.plans[0].operations[0]
    assert operation.target_value == -12.0
    assert operation.metadata["merged_recommendation_count"] == 2
    assert operation.metadata["merged_reasons"] == [
        "level plane",
        "masking balance",
    ]
    assert len(operation.metadata["merged_logical_correction_ids"]) == 2


def test_conflicting_targets_are_local_residuals_and_keep_unrelated_plan():
    bundle = build_soundcheck_transaction_plans(
        [
            ChannelFaderMove("unrelated safe", 1, -10.0),
            ChannelFaderMove("first target", 2, -12.0),
            ChannelFaderMove("conflicting target", 2, -6.0),
        ],
        run_id="conflict-test",
    )

    assert [plan.operations[0].channel for plan in bundle.plans] == [1]
    assert len(bundle.unsupported_actions) == 2
    assert {item["reason"] for item in bundle.unsupported_actions} == {
        "conflicting_semantic_target"
    }


def test_noop_is_preserved_as_non_actionable_evidence():
    bundle = build_soundcheck_transaction_plans(
        [NoOp("signal unavailable", "wait for source")]
    )

    assert bundle.plans == ()
    residual = bundle.unsupported_actions[0]
    assert residual["action_type"] == "NoOp"
    assert residual["reason"] == "signal unavailable"
    assert residual["message"] == "wait for source"
    assert residual["status"] == "not_actionable"
    assert residual["disposition"] == "operator_review"
    assert residual["required_for"] == "safe_core"
    assert residual["musically_required"] is True


def test_eq_boost_remains_review_evidence_and_is_excluded_from_live_plan():
    bundle = build_soundcheck_transaction_plans(
        [ChannelEQMove("requested tone boost", 1, 1, 120.0, 4.0, 1.0)]
    )

    assert bundle.plans == ()
    residual = bundle.unsupported_actions[0]
    assert residual["action_type"] == "ChannelEQMove"
    assert residual["reason"] == "live_channel_eq_boost_is_not_allowed"
    assert residual["status"] == "unsupported"
    assert residual["disposition"] == "operator_review"
    assert residual["required_for"] == "show_mix"


def test_inactive_negative_infinity_dca_is_not_emitted_as_fake_live_move():
    bundle = build_soundcheck_transaction_plans(
        [DCAFaderMove("inactive DCA", 1, -138.0)]
    )

    assert bundle.plans == ()
    residual = bundle.unsupported_actions[0]
    assert residual["action_type"] == "DCAFaderMove"
    assert residual["reason"] == "inactive_dca_target_is_negative_infinity"
    assert residual["status"] == "unsupported"
    assert residual["disposition"] == "operator_review"
