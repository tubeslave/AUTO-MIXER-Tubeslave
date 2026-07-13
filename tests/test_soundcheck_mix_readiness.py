from __future__ import annotations

import asyncio
import copy
from contextlib import asynccontextmanager, contextmanager
import json
from pathlib import Path
import time

import pytest
import server as server_module

from autofoh_safety import (
    BusCompressorMakeupAdjust,
    BusEQMove,
    ChannelEQMove,
    ChannelFaderMove,
    ChannelGainMove,
    CompressorAdjust,
    GateAdjust,
    HighPassAdjust,
    PanAdjust,
)
from operator_recommendation_bridge import build_soundcheck_operator_proposals
from operator_proposal_queue import OperatorProposalQueue
from operator_proposal_queue import _apply_blockers_for_transaction_plan
from server import AutoMixerServer
from soundcheck_mix_readiness import (
    aggregate_soundcheck_readiness,
    assess_soundcheck_manifest,
    build_soundcheck_replan_draft,
    build_proposal_only_readiness,
    coupled_safe_alternative_plan_errors,
    plan_preflight_assessment,
    soundcheck_manifest_logical_id_errors,
    validate_replacement,
)
from soundcheck_transaction_plan import (
    STAGE_ORDER,
    bind_soundcheck_manifest_content_sha256,
    build_soundcheck_transaction_plans,
    operation_from_soundcheck_action,
)
from supervised_console_parameter_registry import (
    FAMILY_CONTRACTS,
    supervised_console_family_registry,
)
from supervised_console_transaction import AudioRecheckResult, ConsoleTransactionOperation
from wing_supervised_console_adapters import (
    WingSupervisedFamilyAdapter,
    audible_state_reads_for_operation,
    parameter_writes_for_operation,
)


def _operation_from_raw(raw):
    return ConsoleTransactionOperation(
        operation_id=raw["operation_id"],
        family_id=raw["family_id"],
        write_kind=raw["write_kind"],
        target_value=raw["target_value"],
        channel=raw.get("channel"),
        parameter=raw.get("parameter", ""),
        metadata=dict(raw.get("metadata") or {}),
    )


def _m006_shape():
    gain = [ChannelGainMove("M006 trim", 1, -9.0)]
    hpf = [
        HighPassAdjust(f"M006 hpf {channel}", channel, 70.0 + channel, True)
        for channel in range(1, 11)
    ]
    eq = [
        ChannelEQMove(
            f"M006 eq {channel}",
            channel,
            ((channel - 1) % 4) + 1,
            180.0 + channel * 20.0,
            -2.0,
            1.0,
        )
        for channel in range(1, 8)
    ]
    gate = [
        GateAdjust(
            f"M006 gate {channel}",
            channel,
            -40.0,
            18.0,
            5.0,
            40.0,
            120.0,
            True,
        )
        for channel in range(1, 6)
    ]
    compressor = [
        CompressorAdjust(
            f"M006 compressor {channel}",
            channel,
            -18.0,
            2.0,
            20.0,
            100.0,
            0.0,
            True,
        )
        for channel in range(1, 3)
    ]
    pan = [
        PanAdjust(
            f"M006 pan {channel}",
            channel,
            60.0 if channel <= 6 else 20.0,
        )
        for channel in range(1, 15)
    ]
    fader = [
        ChannelFaderMove(
            f"M006 fader {channel}",
            channel,
            -6.0 if channel <= 4 else -12.0,
        )
        for channel in range(1, 13)
    ]
    actions = gain + hpf + eq + gate + compressor + pan + fader
    groups = {
        "gain": gain,
        "hpf": hpf,
        "eq": eq,
        "gate": gate,
        "compressor": compressor,
        "pan": pan,
        "fader": fader,
    }
    manifest = build_soundcheck_transaction_plans(
        actions,
        run_id="m006-regression",
    ).to_dict()
    operations = {
        raw["operation_id"]: _operation_from_raw(raw)
        for plan in manifest["plans"]
        for raw in plan["operations"]
    }
    snapshots = {}
    for operation_id, operation in operations.items():
        writes = parameter_writes_for_operation(operation)
        snapshots[operation_id] = {
            "physical_verified": True,
            "values": {
                **{write.address: write.value for write in writes},
                **{
                    read.address: read.value
                    for read in audible_state_reads_for_operation(operation)
                },
            },
        }

    def operation_id(action):
        return operation_from_soundcheck_action(action).operation_id

    gain_id = operation_id(gain[0])
    snapshots[gain_id]["values"]["/ch/1/in/set/trim"] = 0.0

    blocked_eq_id = operation_id(eq[0])
    blocked_eq = operations[blocked_eq_id]
    blocked_eq_gain = next(
        write.address
        for write in parameter_writes_for_operation(blocked_eq)
        if write.address.endswith("g")
    )
    snapshots[blocked_eq_id]["values"][blocked_eq_gain] = -4.0
    for action in eq:
        snapshots[operation_id(action)]["audible_state_verified"] = True
    for action in eq[3:]:
        item_id = operation_id(action)
        operation = operations[item_id]
        gain_address = next(
            write.address
            for write in parameter_writes_for_operation(operation)
            if write.address.endswith("g")
        )
        snapshots[item_id]["values"][gain_address] = -1.0

    blocked_gate_id = operation_id(gate[0])
    snapshots[blocked_gate_id]["values"]["/ch/1/gate/thr"] = -35.0
    for action in gate[1:]:
        item_id = operation_id(action)
        channel = action.channel_id
        snapshots[item_id]["values"][f"/ch/{channel}/gate/range"] = 17.0

    for action in compressor:
        item_id = operation_id(action)
        channel = action.channel_id
        snapshots[item_id]["values"][f"/ch/{channel}/dyn/gain"] = -2.0

    for action in pan[:6]:
        item_id = operation_id(action)
        snapshots[item_id]["values"][f"/ch/{action.channel_id}/pan"] = 0.0
    for action in pan[6:]:
        item_id = operation_id(action)
        snapshots[item_id]["values"][f"/ch/{action.channel_id}/pan"] = 15.0

    for action in fader[:4]:
        item_id = operation_id(action)
        snapshots[item_id]["values"][f"/ch/{action.channel_id}/fdr"] = -12.0
    for action in fader[4:]:
        item_id = operation_id(action)
        snapshots[item_id]["values"][f"/ch/{action.channel_id}/fdr"] = -10.0

    adapters = {
        family_id: WingSupervisedFamilyAdapter(family_id, object())
        for family_id in {operation.family_id for operation in operations.values()}
    }
    return manifest, groups, operations, snapshots, adapters


def _ready_evidence():
    now = time.time()
    return {
        "source_coverage": {
            "verified": True,
            "captured_at_epoch": now,
            "max_age_sec": 300.0,
            "active_sources": 12,
            "stage_coverage": {
                "foundation_level": True,
                "source_control": True,
                "phase_alignment_review": True,
                "tonal_dynamics": True,
                "group_balance": True,
                "spatial_fx": True,
            },
        },
        "main_47_48_evidence": {
            "route_verified": True,
            "fresh": True,
            "signal_present": True,
            "headroom_safe": True,
            "captured_at_epoch": now,
            "max_age_sec": 5.0,
        },
    }


def _source_fader_intents(run_id, captured_at_epoch, targets, baselines):
    return {
        str(channel): {
            "mode": (
                "absolute_target"
                if channel in targets
                else "hold_current"
            ),
            "analysis_baseline_db": baselines[channel],
            "absolute_target_db": targets.get(channel),
            "analysis_run_id": run_id,
            "analysis_captured_at_epoch": captured_at_epoch,
            "source_name": f"Source {channel}",
            "source_role": "source",
            "signal_verified": True,
            "foundation_analyzed": True,
            "main_send_on": True,
            "main_send_level_db": 0.0,
            "main_send_pre": False,
            "input_routing": {
                "main_group": "LOCAL",
                "main_channel": channel,
                "alt_group": "OFF",
                "alt_channel": 0,
            },
            "channel_main_path_locally_enabled": True,
            "assigned_dca_ids": [],
            "assigned_dca_states": {},
            "downstream_dca_path_verified": True,
            "analyzed_source_path_enabled": True,
        }
        for channel in sorted(set(targets) | set(baselines))
    }


def _source_audible_paths(baselines, captured_at_epoch, *, max_age_sec=30.0):
    return {
        str(channel): {
            "channel": channel,
            "current_fader_db": baseline,
            "current_muted": False,
            "current_main_send_on": True,
            "current_main_send_level_db": 0.0,
            "physical_verified": True,
            "two_pass_consistent": True,
            "matches_analysis_baseline": True,
            "matches_analysis_main_send": True,
            "channel_main_path_locally_enabled": True,
            "downstream_dca_path_verified": True,
            "captured_at_epoch": captured_at_epoch,
            "max_age_sec": max_age_sec,
            "reason": "selected_source_path_verified",
        }
        for channel, baseline in baselines.items()
    }


def _committed_outcome():
    now = time.time()
    return {
        "status": "applied",
        "physical_verified": True,
        "audio_recheck_passed": True,
        "audible_state_verified": True,
        "wal_terminal": "committed",
        "final_reconciliation_verified": True,
        "reconciliation_captured_at_epoch": now,
        "reconciliation_max_age_sec": 30.0,
    }


def test_exact_m006_shape_keeps_36_local_safe_and_15_visible_residuals():
    manifest, _groups, _operations, snapshots, adapters = _m006_shape()

    assessment = assess_soundcheck_manifest(
        manifest,
        snapshots_by_operation=snapshots,
        adapters=adapters,
    )

    assert manifest["plan_count"] == manifest["operation_count"] == 51
    assert manifest["logical_correction_count"] == 51
    assert assessment["logical_correction_count"] == 51
    assert assessment["locally_admissible_count"] == 36
    assert assessment["physical_change_candidate_count"] == 24
    assert assessment["already_satisfied_count"] == 12
    assert assessment["local_residual_count"] == 15
    assert assessment["local_reason_counts"] == {
        "compressor_makeup_increase_not_allowed": 2,
        "dynamics_threshold_delta_exceeds_3db": 1,
        "fader_boost_not_allowed": 4,
        "gain_down_delta_exceeds_6db": 1,
        "pan_delta_exceeds_10_percent": 6,
        "relative_eq_boost_not_allowed": 1,
    }
    assert len(assessment["executable_child_transaction_ids"]) == 24
    assert assessment["live_authority_changed"] is False


def test_archived_m006_shape_with_counterfactual_verified_holds_replans_fader_vector():
    fixture = json.loads(
        (
            Path(__file__).parent
            / "fixtures"
            / "m006_exact_transaction_20260711.json"
        ).read_text(encoding="utf-8")
    )
    assert fixture["source"] == {
        "event_line_sha256": "0ce7efb80f19e38f0a5e926467f77845b05ebee9820c78e7ec362b2443128d51",
        "file_sha256": "aaaba5d3b0dd7e9fb9f54b12affdf54ff444a16bba31c1d92297afda39941fc4",
        "line_number_1_based": 85419,
        "relative_path": "artifacts/wing_telemetry/manual_write_gate/manual_write_events.jsonl",
        "timestamp": "2026-07-11T14:37:51Z",
        "transaction_id": "soundcheck:normal:13c43cc98a3c487c",
    }
    plans = []
    snapshots = {}
    adapters = {}
    for index, row in enumerate(fixture["operations"]):
        operation = {
            key: row[key]
            for key in (
                "operation_id",
                "family_id",
                "write_kind",
                "target_value",
                "channel",
                "parameter",
                "metadata",
            )
        }
        plans.append(
            {
                "transaction_id": f"m006-fixture:{index}",
                "operations": [operation],
                "logical_correction_ids": [row["operation_id"]],
                "stage": row["metadata"]["stage"],
                "stage_order": STAGE_ORDER[row["metadata"]["stage"]],
                "required_for": row["metadata"]["required_for"],
            }
        )
        snapshots[row["operation_id"]] = {
            "physical_verified": True,
            "values": row["snapshot_values"],
        }
        adapters.setdefault(
            row["family_id"],
            WingSupervisedFamilyAdapter(row["family_id"], object()),
        )
    manifest = {
        "run_id": "m006-exact-audit-fixture",
        "plan_revision": 1,
        "plan_fingerprint": fixture["source"]["event_line_sha256"],
        "plans": plans,
        "unsupported_actions": [],
    }

    assessment = assess_soundcheck_manifest(
        manifest,
        snapshots_by_operation=snapshots,
        adapters=adapters,
    )

    assert fixture["operation_count"] == len(plans) == 51
    assert assessment["locally_admissible_count"] == 36
    assert assessment["physical_change_candidate_count"] == 24
    assert assessment["already_satisfied_count"] == 12
    assert assessment["local_residual_count"] == 15
    assert assessment["disposition_counts"] == {
        "already_satisfied": 12,
        "apply": 24,
        "operator_review": 7,
        "split_steps": 8,
    }
    now = time.time()
    fader_targets = {
        row["channel"]: float(row["target_value"])
        for row in fixture["operations"]
        if row["family_id"] == "fader"
    }
    fader_baselines = {
        row["channel"]: float(
            row["snapshot_values"][f"/ch/{row['channel']}/fdr"]
        )
        for row in fixture["operations"]
        if row["family_id"] == "fader"
    }
    # The archived audit contains no physical fader rows for selected channels
    # 4 and 5. These two values are deliberately counterfactual, explicit
    # fresh hold evidence used to regression-test the M006-shaped vector; they
    # are not claimed as historical M006 readback.
    fader_baselines.update({4: -8.0, 5: -8.0})
    selected_channels = [*range(1, 14), 16]
    assessment["source_coverage"] = {
        "verified": True,
        "captured_at_epoch": now,
        "max_age_sec": 300.0,
        "selected_channel_ids": selected_channels,
        "stage_coverage": {"foundation_level": True},
        "per_source_stage_coverage": {
            str(channel): {"foundation_level": "analyzed"}
            for channel in selected_channels
        },
        "per_source_fader_intent": _source_fader_intents(
            manifest["run_id"],
            now,
            fader_targets,
            fader_baselines,
        ),
    }
    assessment["main_47_48_evidence"] = {
        "route_verified": True,
        "fresh": True,
        "signal_present": True,
        "headroom_safe": True,
        "captured_at_epoch": now,
        "max_age_sec": 5.0,
    }
    assessment["selected_source_fader_holds"] = {
        str(channel): {
            "channel": channel,
            "current_fader_db": fader_baselines[channel],
            "analysis_baseline_db": fader_baselines[channel],
            "physical_verified": True,
            "two_pass_consistent": True,
            "matches_analysis_baseline": True,
            "captured_at_epoch": now,
            "max_age_sec": 30.0,
        }
        for channel in (4, 5)
    }
    assessment["selected_source_audible_paths"] = _source_audible_paths(
        fader_baselines,
        now,
    )

    draft = build_soundcheck_replan_draft(
        manifest,
        assessment,
        snapshots_by_operation=snapshots,
        adapters=adapters,
    )

    assert draft["draft_available"] is True
    assert draft["split_operation_count"] == 8
    assert draft["safe_alternative_operation_count"] == 14, (
        draft["operator_requests"],
        draft["safe_alternatives"],
        draft["safe_alternative_blockers"],
    )
    assert draft["safe_alternatives"][0][
        "hold_current_channel_ids"
    ] == [4, 5]
    fader_requests = [
        request
        for request in draft["operator_requests"]
        if request["reason"] == "fader_boost_not_allowed"
    ]
    assert len(fader_requests) == 4
    assert all(
        request["safe_automatic_replacement_available"] is True
        for request in fader_requests
    )
    unresolved_requests = [
        request
        for request in draft["operator_requests"]
        if request["reason"]
        in {
            "relative_eq_boost_not_allowed",
            "compressor_makeup_increase_not_allowed",
        }
    ]
    assert len(unresolved_requests) == 3
    assert all(
        request["safe_automatic_replacement_available"] is False
        for request in unresolved_requests
    )
    coupled = [
        plan
        for plan in draft["revised_manifest"]["plans"]
        if plan.get("coupled_safe_alternative_group") is True
    ]
    assert len(coupled) == 1
    assert len(coupled[0]["operations"]) == 14


def test_global_snapshot_failure_stops_every_child_before_execution():
    manifest, _groups, operations, snapshots, adapters = _m006_shape()
    missing_id = next(iter(operations))
    snapshots.pop(missing_id)

    assessment = assess_soundcheck_manifest(
        manifest,
        snapshots_by_operation=snapshots,
        adapters=adapters,
    )

    assert assessment["global_blockers"] == [
        f"physical_snapshot_unverified:{missing_id}"
    ]
    assert assessment["executable_child_transaction_ids"] == []
    assert assessment["disposition_counts"] == {"blocked_global": 51}


def test_m006_partial_success_cannot_claim_ready_mix():
    manifest, _groups, _operations, snapshots, adapters = _m006_shape()
    assessment = assess_soundcheck_manifest(
        manifest,
        snapshots_by_operation=snapshots,
        adapters=adapters,
    )
    outcomes = {}
    for item in assessment["assessments"]:
        if item["disposition"] == "apply":
            outcomes[item["operation_id"]] = _committed_outcome()
        elif item["disposition"] == "already_satisfied":
            reconciled_at = time.time()
            outcomes[item["operation_id"]] = {
                "disposition": "already_satisfied",
                "physical_verified": True,
                "audible_state_verified": item["audible_state_verified"],
                "evidence_fresh": True,
                "final_reconciliation_verified": True,
                "reconciliation_captured_at_epoch": reconciled_at,
                "reconciliation_max_age_sec": 30.0,
            }
        else:
            outcomes[item["operation_id"]] = {
                "disposition": (
                    "operator_review"
                    if item["disposition"] == "operator_review"
                    else "pending"
                ),
                "reason": item["reason"],
            }

    readiness = aggregate_soundcheck_readiness(
        manifest,
        outcomes=outcomes,
        proposal_only=False,
        **_ready_evidence(),
    )

    assert readiness["mix_readiness"] == "soundcheck_incomplete"
    assert readiness["logical_counts"]["applied"] == 24
    assert readiness["logical_counts"]["already_satisfied"] == 12
    assert len(readiness["unresolved_required_operation_ids"]) == 15


def test_m006_split_residuals_produce_a_bounded_next_revision_and_operator_requests():
    manifest, _groups, _operations, snapshots, adapters = _m006_shape()
    assessment = assess_soundcheck_manifest(
        manifest,
        snapshots_by_operation=snapshots,
        adapters=adapters,
    )

    draft = build_soundcheck_replan_draft(
        manifest,
        assessment,
        snapshots_by_operation=snapshots,
        adapters=adapters,
    )

    assert draft["draft_available"] is True
    assert draft["from_plan_revision"] == 1
    assert draft["to_plan_revision"] == 2
    assert draft["split_operation_count"] == 8
    assert draft["deferred_target_count"] == 8
    assert len(draft["operator_requests"]) == 7
    assert all(
        request["safe_automatic_replacement_available"] is False
        for request in draft["operator_requests"]
    )
    revised = draft["revised_manifest"]
    assert revised["plan_revision"] == 2
    assert revised["plan_fingerprint"] != manifest["plan_fingerprint"]
    assert revised["live_mixer_writes"] is False
    assert revised["execution_allowed"] is False
    assert len(revised["readiness_residuals"]) == 8

    revised_snapshots = {}
    for plan in revised["plans"]:
        raw = plan["operations"][0]
        original_id = raw["metadata"].get("replacement_of", raw["operation_id"])
        revised_snapshots[raw["operation_id"]] = snapshots[original_id]
    revised_assessment = assess_soundcheck_manifest(
        revised,
        snapshots_by_operation=revised_snapshots,
        adapters=adapters,
    )
    assert revised_assessment["locally_admissible_count"] == 44
    assert revised_assessment["local_residual_count"] == 7


def test_split_replan_continues_from_verified_step_without_exceeding_delta_budget():
    manifest = build_soundcheck_transaction_plans(
        [PanAdjust("wide image", 1, 40.0)],
        run_id="split-chain",
    ).to_dict()
    first_source_epoch = time.time() - 2.0
    manifest["source_coverage"] = {
        "verified": True,
        "captured_at_epoch": first_source_epoch,
        "max_age_sec": 300.0,
    }
    raw = manifest["plans"][0]["operations"][0]
    adapter = WingSupervisedFamilyAdapter("pan_phase_delay", object())
    first_snapshots = {
        raw["operation_id"]: {
            "physical_verified": True,
            "values": {"/ch/1/pan": 0.0},
        }
    }
    first_assessment = assess_soundcheck_manifest(
        manifest,
        snapshots_by_operation=first_snapshots,
        adapters={"pan_phase_delay": adapter},
    )
    first_assessment["source_coverage"] = dict(manifest["source_coverage"])
    first_draft = build_soundcheck_replan_draft(
        manifest,
        first_assessment,
        snapshots_by_operation=first_snapshots,
        adapters={"pan_phase_delay": adapter},
    )
    first_step = first_draft["revised_manifest"]["plans"][0]["operations"][0]
    assert first_step["target_value"] == 10.0

    second_manifest = first_draft["revised_manifest"]
    second_manifest["source_coverage"] = {
        **second_manifest["source_coverage"],
        "captured_at_epoch": first_source_epoch + 1.0,
    }
    second_snapshots = {
        first_step["operation_id"]: {
            "physical_verified": True,
            "values": {"/ch/1/pan": 10.0},
        }
    }
    second_assessment = assess_soundcheck_manifest(
        second_manifest,
        snapshots_by_operation=second_snapshots,
        adapters={"pan_phase_delay": adapter},
    )
    second_assessment["source_coverage"] = dict(
        second_manifest["source_coverage"]
    )
    second_draft = build_soundcheck_replan_draft(
        second_manifest,
        second_assessment,
        snapshots_by_operation=second_snapshots,
        adapters={"pan_phase_delay": adapter},
    )
    second_step = second_draft["revised_manifest"]["plans"][0]["operations"][0]

    assert second_draft["draft_available"] is True
    assert second_step["target_value"] == 20.0
    assert second_step["metadata"]["final_target_value"] == 40.0
    assert second_step["metadata"]["split_step_index"] == 2


def test_split_replan_requires_fresh_source_analysis_between_applied_steps():
    manifest = build_soundcheck_transaction_plans(
        [PanAdjust("wide image", 1, 40.0)],
        run_id="split-chain-needs-source-reanalysis",
    ).to_dict()
    source_epoch = time.time()
    manifest["source_coverage"] = {
        "verified": True,
        "captured_at_epoch": source_epoch,
        "max_age_sec": 300.0,
    }
    raw = manifest["plans"][0]["operations"][0]
    adapter = WingSupervisedFamilyAdapter("pan_phase_delay", object())
    first_snapshots = {
        raw["operation_id"]: {
            "physical_verified": True,
            "values": {"/ch/1/pan": 0.0},
        }
    }
    first_assessment = assess_soundcheck_manifest(
        manifest,
        snapshots_by_operation=first_snapshots,
        adapters={"pan_phase_delay": adapter},
    )
    first_assessment["source_coverage"] = dict(manifest["source_coverage"])
    first_draft = build_soundcheck_replan_draft(
        manifest,
        first_assessment,
        snapshots_by_operation=first_snapshots,
        adapters={"pan_phase_delay": adapter},
    )
    second_manifest = first_draft["revised_manifest"]
    first_step = second_manifest["plans"][0]["operations"][0]
    second_snapshots = {
        first_step["operation_id"]: {
            "physical_verified": True,
            "values": {"/ch/1/pan": 10.0},
        }
    }
    second_assessment = assess_soundcheck_manifest(
        second_manifest,
        snapshots_by_operation=second_snapshots,
        adapters={"pan_phase_delay": adapter},
    )
    second_assessment["source_coverage"] = dict(
        second_manifest["source_coverage"]
    )

    blocked = build_soundcheck_replan_draft(
        second_manifest,
        second_assessment,
        snapshots_by_operation=second_snapshots,
        adapters={"pan_phase_delay": adapter},
    )

    assert blocked["draft_available"] is False
    assert blocked["operator_requests"][0]["reason"] == (
        "fresh_source_reanalysis_required_between_split_steps"
    )
    assert (
        second_manifest["readiness_residuals"][0]["channel"]
        == 1
    )


def test_replan_preserves_unmatched_split_residual_when_other_step_advances():
    manifest = build_soundcheck_transaction_plans(
        [
            PanAdjust("wide one", 1, 40.0),
            PanAdjust("wide two", 2, 40.0),
        ],
        run_id="split-residual-preservation",
    ).to_dict()
    first_epoch = time.time() - 5.0
    manifest["source_coverage"] = {
        "verified": True,
        "captured_at_epoch": first_epoch,
        "max_age_sec": 300.0,
    }
    adapter = WingSupervisedFamilyAdapter("pan_phase_delay", object())
    first_snapshots = {
        plan["operations"][0]["operation_id"]: {
            "physical_verified": True,
            "values": {
                f"/ch/{plan['operations'][0]['channel']}/pan": 0.0,
            },
        }
        for plan in manifest["plans"]
    }
    first_assessment = assess_soundcheck_manifest(
        manifest,
        snapshots_by_operation=first_snapshots,
        adapters={"pan_phase_delay": adapter},
    )
    first_assessment["source_coverage"] = dict(manifest["source_coverage"])
    first_draft = build_soundcheck_replan_draft(
        manifest,
        first_assessment,
        snapshots_by_operation=first_snapshots,
        adapters={"pan_phase_delay": adapter},
    )
    second_manifest = first_draft["revised_manifest"]
    second_manifest["source_coverage"] = {
        **second_manifest["source_coverage"],
        "captured_at_epoch": first_epoch + 1.0,
    }
    operations = {
        plan["operations"][0]["channel"]: plan["operations"][0]
        for plan in second_manifest["plans"]
    }
    second_snapshots = {
        operation["operation_id"]: {
            "physical_verified": True,
            "values": {f"/ch/{channel}/pan": 10.0},
        }
        for channel, operation in operations.items()
    }
    second_assessment = assess_soundcheck_manifest(
        second_manifest,
        snapshots_by_operation=second_snapshots,
        adapters={"pan_phase_delay": adapter},
    )
    second_assessment["source_coverage"] = dict(
        second_manifest["source_coverage"]
    )
    blocked_operation_id = operations[1]["operation_id"]
    for row in second_assessment["assessments"]:
        if row["operation_id"] == blocked_operation_id:
            row["disposition"] = "operator_review"
            row["reason"] = "operator_reanalysis_required"

    second_draft = build_soundcheck_replan_draft(
        second_manifest,
        second_assessment,
        snapshots_by_operation=second_snapshots,
        adapters={"pan_phase_delay": adapter},
    )

    assert second_draft["draft_available"] is True
    residuals = second_draft["revised_manifest"]["readiness_residuals"]
    by_channel = {row["channel"]: row for row in residuals}
    assert set(by_channel) == {1, 2}
    assert by_channel[1]["current_step_operation_id"] == blocked_operation_id
    assert by_channel[2]["current_step_operation_id"] != (
        operations[2]["operation_id"]
    )


def test_multi_constraint_compressor_split_advances_all_fields_and_converges():
    action = CompressorAdjust(
        "deep compression",
        1,
        -20.0,
        10.0,
        20.0,
        100.0,
        -6.0,
        True,
    )
    manifest = build_soundcheck_transaction_plans(
        [action],
        run_id="compressor-multi-split",
    ).to_dict()
    source_epoch = time.time() - 20.0
    manifest["source_coverage"] = {
        "verified": True,
        "captured_at_epoch": source_epoch,
        "max_age_sec": 300.0,
    }
    adapter = WingSupervisedFamilyAdapter("dynamics_compressor", object())
    current = {
        "/ch/1/dyn/thr": -10.0,
        "/ch/1/dyn/ratio": 1.0,
        "/ch/1/dyn/att": 20.0,
        "/ch/1/dyn/rel": 100.0,
        "/ch/1/dyn/gain": 0.0,
        "/ch/1/dyn/on": 1,
    }
    seen = set()
    final_target = manifest["plans"][0]["operations"][0]["target_value"]

    for _ in range(8):
        raw = manifest["plans"][0]["operations"][0]
        snapshots = {
            raw["operation_id"]: {
                "physical_verified": True,
                "audible_state_verified": True,
                "values": dict(current),
            }
        }
        assessment = assess_soundcheck_manifest(
            manifest,
            snapshots_by_operation=snapshots,
            adapters={"dynamics_compressor": adapter},
        )
        assessment["source_coverage"] = dict(manifest["source_coverage"])
        draft = build_soundcheck_replan_draft(
            manifest,
            assessment,
            snapshots_by_operation=snapshots,
            adapters={"dynamics_compressor": adapter},
        )
        assert draft["draft_available"] is True
        next_raw = draft["revised_manifest"]["plans"][0]["operations"][0]
        next_target = next_raw["target_value"]
        signature = (
            next_target["threshold_db"],
            next_target["ratio"],
            next_target["makeup_db"],
        )
        assert signature not in seen
        seen.add(signature)
        assert abs(next_target["threshold_db"] - current["/ch/1/dyn/thr"]) <= 3.0
        assert abs(next_target["ratio"] - current["/ch/1/dyn/ratio"]) <= 2.0
        assert abs(next_target["makeup_db"] - current["/ch/1/dyn/gain"]) <= 3.0
        current.update(
            {
                "/ch/1/dyn/thr": next_target["threshold_db"],
                "/ch/1/dyn/ratio": next_target["ratio"],
                "/ch/1/dyn/gain": next_target["makeup_db"],
            }
        )
        manifest = draft["revised_manifest"]
        source_epoch += 1.0
        manifest["source_coverage"] = {
            **manifest["source_coverage"],
            "captured_at_epoch": source_epoch,
        }
        if next_target == final_target:
            break
    else:  # pragma: no cover - explicit anti-loop assertion
        pytest.fail("compressor split did not converge within eight revisions")

    assert len(seen) <= 5


def test_trusted_fader_reanalysis_replaces_boost_with_global_safe_downward_translation():
    manifest = build_soundcheck_transaction_plans(
        [
            ChannelFaderMove("bring vocal forward", 1, -6.0),
            ChannelFaderMove("balance guitar", 2, -12.0),
        ],
        run_id="safe-fader-translation",
    ).to_dict()
    operations = {
        raw["operation_id"]: _operation_from_raw(raw)
        for plan in manifest["plans"]
        for raw in plan["operations"]
    }
    snapshots = {}
    for operation_id, operation in operations.items():
        current = -12.0 if operation.channel == 1 else -10.0
        snapshots[operation_id] = {
            "physical_verified": True,
            "values": {f"/ch/{operation.channel}/fdr": current},
        }
    adapter = WingSupervisedFamilyAdapter("fader", object())
    assessment = assess_soundcheck_manifest(
        manifest,
        snapshots_by_operation=snapshots,
        adapters={"fader": adapter},
    )
    now = time.time()
    assessment["source_coverage"] = {
        "verified": True,
        "captured_at_epoch": now,
        "max_age_sec": 300.0,
        "selected_channel_ids": [1, 2],
        "stage_coverage": {"foundation_level": True},
        "per_source_stage_coverage": {
            "1": {"foundation_level": "analyzed"},
            "2": {"foundation_level": "analyzed"},
        },
        "per_source_fader_intent": _source_fader_intents(
            manifest["run_id"],
            now,
            {1: -6.0, 2: -12.0},
            {1: -12.0, 2: -10.0},
        ),
    }
    assessment["main_47_48_evidence"] = {
        "route_verified": True,
        "fresh": True,
        "signal_present": True,
        "headroom_safe": True,
        "captured_at_epoch": now,
        "max_age_sec": 5.0,
    }
    assessment["selected_source_audible_paths"] = _source_audible_paths(
        {1: -12.0, 2: -10.0},
        now,
    )

    draft = build_soundcheck_replan_draft(
        manifest,
        assessment,
        snapshots_by_operation=snapshots,
        adapters={"fader": adapter},
    )

    assert draft["draft_available"] is True
    assert draft["safe_alternative_operation_count"] == 2
    assert draft["safe_alternatives"][0]["strategy"] == (
        "global_fader_translation_v1"
    )
    assert draft["safe_alternatives"][0]["final_target_vector_preserved"] is True
    assert draft["safe_alternatives"][0]["relative_balance_preserved"] is False
    assert draft["safe_alternatives"][0]["current_step_transitional"] is True
    coupled = [
        plan
        for plan in draft["revised_manifest"]["plans"]
        if plan.get("coupled_safe_alternative_group") is True
    ]
    assert len(coupled) == 1
    assert coupled[0]["atomic_group_commit_required"] is True
    assert len(coupled[0]["operations"]) == 2
    assert any(
        request["reason"] == "fader_boost_not_allowed"
        and request["safe_automatic_replacement_available"] is True
        for request in draft["operator_requests"]
    )
    targets = {
        raw["channel"]: raw["target_value"]
        for plan in draft["revised_manifest"]["plans"]
        for raw in plan["operations"]
    }
    assert targets == {1: -12.0, 2: -16.0}
    assert all(
        raw["metadata"]["constructive_lane_used"] is False
        for plan in draft["revised_manifest"]["plans"]
        for raw in plan["operations"]
    )


def test_global_translation_uses_fresh_hold_current_for_selected_source_without_target():
    manifest = build_soundcheck_transaction_plans(
        [
            ChannelFaderMove("bring vocal forward", 1, -6.0),
            ChannelFaderMove("balance guitar", 2, -12.0),
        ],
        run_id="safe-fader-translation-with-hold",
    ).to_dict()
    operations = {
        raw["operation_id"]: _operation_from_raw(raw)
        for plan in manifest["plans"]
        for raw in plan["operations"]
    }
    snapshots = {}
    for operation_id, operation in operations.items():
        current = -12.0 if operation.channel == 1 else -10.0
        snapshots[operation_id] = {
            "physical_verified": True,
            "values": {f"/ch/{operation.channel}/fdr": current},
        }
    adapter = WingSupervisedFamilyAdapter("fader", object())
    assessment = assess_soundcheck_manifest(
        manifest,
        snapshots_by_operation=snapshots,
        adapters={"fader": adapter},
    )
    now = time.time()
    assessment["source_coverage"] = {
        "verified": True,
        "captured_at_epoch": now,
        "max_age_sec": 300.0,
        "selected_channel_ids": [1, 2, 3],
        "stage_coverage": {"foundation_level": True},
        "per_source_stage_coverage": {
            str(channel): {"foundation_level": "analyzed"}
            for channel in (1, 2, 3)
        },
        "per_source_fader_intent": _source_fader_intents(
            manifest["run_id"],
            now,
            {1: -6.0, 2: -12.0},
            {1: -12.0, 2: -10.0, 3: -8.0},
        ),
    }
    assessment["main_47_48_evidence"] = {
        "route_verified": True,
        "fresh": True,
        "signal_present": True,
        "headroom_safe": True,
        "captured_at_epoch": now,
        "max_age_sec": 5.0,
    }
    assessment["selected_source_fader_holds"] = {
        "3": {
            "channel": 3,
            "current_fader_db": -8.0,
            "physical_verified": True,
            "two_pass_consistent": True,
            "matches_analysis_baseline": True,
            "analysis_baseline_db": -8.0,
            "captured_at_epoch": now,
            "max_age_sec": 30.0,
        }
    }
    assessment["selected_source_audible_paths"] = _source_audible_paths(
        {1: -12.0, 2: -10.0, 3: -8.0},
        now,
    )

    draft = build_soundcheck_replan_draft(
        manifest,
        assessment,
        snapshots_by_operation=snapshots,
        adapters={"fader": adapter},
    )

    assert draft["draft_available"] is True
    assert draft["safe_alternative_operation_count"] == 3
    assert draft["safe_alternatives"][0][
        "hold_current_channel_ids"
    ] == [3]
    targets = {
        raw["channel"]: raw["target_value"]
        for plan in draft["revised_manifest"]["plans"]
        for raw in plan["operations"]
    }
    assert targets == {1: -12.0, 2: -16.0, 3: -14.0}
    assert any(
        residual["channel"] == 2
        and residual["final_target_value"] == -18.0
        for residual in draft["revised_manifest"]["readiness_residuals"]
    )
    hold = next(
        raw
        for plan in draft["revised_manifest"]["plans"]
        for raw in plan["operations"]
        if raw["channel"] == 3
    )
    assert hold["metadata"]["supporting_hold_current_operation"] is True
    assert "replacement_of" not in hold["metadata"]
    assert hold["metadata"]["member_role"] == "hold_current_support"
    assert hold["metadata"]["constructive_lane_used"] is False


def test_global_translation_rejects_stale_hold_current_readback():
    manifest = build_soundcheck_transaction_plans(
        [ChannelFaderMove("bring vocal forward", 1, -6.0)],
        run_id="stale-hold-fails-closed",
    ).to_dict()
    raw = manifest["plans"][0]["operations"][0]
    snapshots = {
        raw["operation_id"]: {
            "physical_verified": True,
            "values": {"/ch/1/fdr": -12.0},
        }
    }
    adapter = WingSupervisedFamilyAdapter("fader", object())
    assessment = assess_soundcheck_manifest(
        manifest,
        snapshots_by_operation=snapshots,
        adapters={"fader": adapter},
    )
    now = time.time()
    assessment["source_coverage"] = {
        "verified": True,
        "captured_at_epoch": now,
        "max_age_sec": 300.0,
        "selected_channel_ids": [1, 2],
        "stage_coverage": {"foundation_level": True},
        "per_source_stage_coverage": {
            "1": {"foundation_level": "analyzed"},
            "2": {"foundation_level": "analyzed"},
        },
        "per_source_fader_intent": _source_fader_intents(
            manifest["run_id"],
            now,
            {1: -6.0},
            {1: -12.0, 2: -10.0},
        ),
    }
    assessment["main_47_48_evidence"] = {
        "route_verified": True,
        "fresh": True,
        "signal_present": True,
        "headroom_safe": True,
        "captured_at_epoch": now,
        "max_age_sec": 5.0,
    }
    assessment["selected_source_fader_holds"] = {
        "2": {
            "channel": 2,
            "current_fader_db": -10.0,
            "physical_verified": True,
            "two_pass_consistent": True,
            "matches_analysis_baseline": True,
            "analysis_baseline_db": -10.0,
            "captured_at_epoch": now - 60.0,
            "max_age_sec": 30.0,
        }
    }
    assessment["selected_source_audible_paths"] = _source_audible_paths(
        {1: -12.0, 2: -10.0},
        now,
    )

    draft = build_soundcheck_replan_draft(
        manifest,
        assessment,
        snapshots_by_operation=snapshots,
        adapters={"fader": adapter},
    )

    assert draft["draft_available"] is False
    assert draft["operator_requests"][0][
        "safe_automatic_replacement_available"
    ] is False


@pytest.mark.asyncio
async def test_parent_preflight_collects_two_pass_hold_current_fader_readback():
    manifest = build_soundcheck_transaction_plans(
        [ChannelFaderMove("bring vocal forward", 1, -6.0)],
        run_id="preflight-hold-current",
    ).to_dict()
    evidence = _ready_evidence()["source_coverage"]
    manifest["source_coverage"] = {
        **evidence,
        "selected_channel_ids": [1, 2],
        "signalled_channel_ids": [1, 2],
        "per_source_stage_coverage": {
            str(channel): {
                stage: "analyzed"
                for stage in (
                    "foundation_level",
                    "source_control",
                    "phase_alignment_review",
                    "tonal_dynamics",
                    "group_balance",
                    "spatial_fx",
                )
            }
            for channel in (1, 2)
        },
        "per_source_fader_intent": _source_fader_intents(
            manifest["run_id"],
            evidence["captured_at_epoch"],
            {1: -6.0},
            {1: -12.0, 2: -10.0},
        ),
    }
    server = _server_for_readiness_io(
        manifest,
        {"/ch/1/fdr": -12.0, "/ch/2/fdr": -10.0},
    )

    assessment = await server.preflight_soundcheck_manifest(manifest)

    hold = assessment["selected_source_fader_holds"]["2"]
    assert hold["physical_verified"] is True
    assert hold["two_pass_consistent"] is True
    assert hold["current_fader_db"] == -10.0
    assert server.mixer_client.queries.count("/ch/2/fdr") == 2


@pytest.mark.parametrize(
    ("drift_address", "drift_value", "expected_reason", "uses_dca"),
    [
        ("/ch/1/mute", 1, "selected_source_muted_after_analysis", False),
        (
            "/ch/1/main/1/on",
            0,
            "selected_source_main_send_off_after_analysis",
            False,
        ),
        (
            "/ch/1/main/1/pre",
            1,
            "selected_source_main_send_pre_fader",
            False,
        ),
        (
            "/ch/1/in/conn/grp",
            "USB",
            "selected_source_input_routing_changed_after_analysis",
            False,
        ),
        (
            "/dca/1/mute",
            1,
            "selected_source_dca_path_changed_or_inactive",
            True,
        ),
    ],
)
@pytest.mark.asyncio
async def test_all_hold_no_operation_plan_rechecks_complete_source_path(
    drift_address,
    drift_value,
    expected_reason,
    uses_dca,
):
    manifest = build_soundcheck_transaction_plans(
        [],
        run_id=f"all-hold-path-{expected_reason}",
    ).to_dict()
    evidence = _ready_evidence()["source_coverage"]
    fader_intents = _source_fader_intents(
        manifest["run_id"],
        evidence["captured_at_epoch"],
        {},
        {1: -12.0},
    )
    values = {"/ch/1/fdr": -12.0}
    if uses_dca:
        fader_intents["1"].update(
            {
                "assigned_dca_ids": [1],
                "assigned_dca_states": {
                    "1": {
                        "muted": False,
                        "locally_enabled": True,
                        "fader_db": -3.0,
                    }
                },
            }
        )
        values.update(
            {
                "/ch/1/tags": "#D1",
                "/dca/1/mute": 0,
                "/dca/1/fdr": -3.0,
            }
        )
    manifest["source_coverage"] = {
        **evidence,
        "selected_channel_ids": [1],
        "signalled_channel_ids": [1],
        "per_source_stage_coverage": {
            "1": {
                stage: "analyzed"
                for stage in (
                    "foundation_level",
                    "source_control",
                    "phase_alignment_review",
                    "tonal_dynamics",
                    "group_balance",
                    "spatial_fx",
                )
            }
        },
        "per_source_fader_intent": fader_intents,
    }
    server = _server_for_readiness_io(manifest, values)

    preflight = await server.preflight_soundcheck_manifest(manifest)
    ready = await server.reconcile_soundcheck_readiness(manifest["run_id"])
    server.mixer_client.values[drift_address] = drift_value
    drifted = await server.reconcile_soundcheck_readiness(manifest["run_id"])

    assert preflight["selected_source_audible_paths"]["1"]["reason"] == (
        "selected_source_path_verified"
    )
    assert ready["mix_readiness"] == "show_mix_ready"
    assert drifted["mix_readiness"] == "soundcheck_incomplete"
    assert drifted["source_coverage"]["selected_source_path_evidence"]["1"][
        "reason"
    ] == expected_reason
    assert server.mixer_client.sent == []


@pytest.mark.asyncio
async def test_coupled_fader_translation_refreshes_and_validates_as_one_child():
    manifest = build_soundcheck_transaction_plans(
        [
            ChannelFaderMove("bring vocal forward", 1, -6.0),
            ChannelFaderMove("balance guitar", 2, -12.0),
        ],
        run_id="coupled-fader-refresh",
    ).to_dict()
    now = time.time()
    source_coverage = {
        "verified": True,
        "captured_at_epoch": now,
        "max_age_sec": 300.0,
        "selected_channel_ids": [1, 2],
        "stage_coverage": {"foundation_level": True},
        "per_source_stage_coverage": {
            "1": {"foundation_level": "analyzed"},
            "2": {"foundation_level": "analyzed"},
        },
        "per_source_fader_intent": _source_fader_intents(
            manifest["run_id"],
            now,
            {1: -6.0, 2: -12.0},
            {1: -12.0, 2: -10.0},
        ),
    }
    manifest["source_coverage"] = source_coverage
    operations = {
        raw["operation_id"]: _operation_from_raw(raw)
        for plan in manifest["plans"]
        for raw in plan["operations"]
    }
    snapshots = {
        operation_id: {
            "physical_verified": True,
            "values": {
                f"/ch/{operation.channel}/fdr": (
                    -12.0 if operation.channel == 1 else -10.0
                )
            },
        }
        for operation_id, operation in operations.items()
    }
    adapter = WingSupervisedFamilyAdapter("fader", object())
    assessment = assess_soundcheck_manifest(
        manifest,
        snapshots_by_operation=snapshots,
        adapters={"fader": adapter},
    )
    assessment["source_coverage"] = source_coverage
    assessment["main_47_48_evidence"] = {
        "route_verified": True,
        "fresh": True,
        "signal_present": True,
        "headroom_safe": True,
        "captured_at_epoch": now,
        "max_age_sec": 5.0,
    }
    assessment["selected_source_audible_paths"] = _source_audible_paths(
        {1: -12.0, 2: -10.0},
        now,
    )
    draft = build_soundcheck_replan_draft(
        manifest,
        assessment,
        snapshots_by_operation=snapshots,
        adapters={"fader": adapter},
    )
    revised = draft["revised_manifest"]
    revised["source_coverage"] = source_coverage
    coupled = next(
        plan
        for plan in revised["plans"]
        if plan.get("coupled_safe_alternative_group") is True
    )
    server = _server_for_readiness_io(
        revised,
        {"/ch/1/fdr": -12.0, "/ch/2/fdr": -10.0},
    )
    server._active_soundcheck_run_id = revised["run_id"]

    refreshed = await server.refresh_soundcheck_child_preflight(coupled)
    proposal = type(
        "Proposal",
        (),
        {"raw": {"supervised_transaction_plan": coupled}},
    )()
    validation = server.validate_soundcheck_child_for_apply(proposal)

    assert refreshed["refreshed"] is True
    assert len(refreshed["assessments"]) == 2
    assert {
        row["disposition"]
        for row in refreshed["assessments"]
    } == {"apply", "already_satisfied"}
    assert validation["allowed"] is True, validation


def _coupled_fader_refresh_case(run_id, *, targets=None):
    targets = dict(targets or {1: -6.0, 2: -12.0})
    manifest = build_soundcheck_transaction_plans(
        [
            ChannelFaderMove("bring vocal forward", 1, targets[1]),
            ChannelFaderMove("balance guitar", 2, targets[2]),
        ],
        run_id=run_id,
    ).to_dict()
    now = time.time()
    source_coverage = {
        "verified": True,
        "captured_at_epoch": now,
        "max_age_sec": 300.0,
        "selected_channel_ids": [1, 2],
        "stage_coverage": {"foundation_level": True},
        "per_source_stage_coverage": {
            "1": {"foundation_level": "analyzed"},
            "2": {"foundation_level": "analyzed"},
        },
        "per_source_fader_intent": _source_fader_intents(
            manifest["run_id"],
            now,
            targets,
            {1: -12.0, 2: -10.0},
        ),
    }
    manifest["source_coverage"] = source_coverage
    operations = {
        raw["operation_id"]: _operation_from_raw(raw)
        for plan in manifest["plans"]
        for raw in plan["operations"]
    }
    snapshots = {
        operation_id: {
            "physical_verified": True,
            "values": {
                f"/ch/{operation.channel}/fdr": (
                    -12.0 if operation.channel == 1 else -10.0
                )
            },
        }
        for operation_id, operation in operations.items()
    }
    adapter = WingSupervisedFamilyAdapter("fader", object())
    assessment = assess_soundcheck_manifest(
        manifest,
        snapshots_by_operation=snapshots,
        adapters={"fader": adapter},
    )
    assessment["source_coverage"] = source_coverage
    assessment["selected_source_audible_paths"] = _source_audible_paths(
        {1: -12.0, 2: -10.0},
        now,
    )
    assessment["main_47_48_evidence"] = {
        "route_verified": True,
        "fresh": True,
        "signal_present": True,
        "headroom_safe": True,
        "captured_at_epoch": now,
        "max_age_sec": 5.0,
    }
    draft = build_soundcheck_replan_draft(
        manifest,
        assessment,
        snapshots_by_operation=snapshots,
        adapters={"fader": adapter},
    )
    revised = draft["revised_manifest"]
    revised["source_coverage"] = source_coverage
    coupled = next(
        plan
        for plan in revised["plans"]
        if plan.get("coupled_safe_alternative_group") is True
    )
    server = _server_for_readiness_io(
        revised,
        {"/ch/1/fdr": -12.0, "/ch/2/fdr": -10.0},
    )
    server._active_soundcheck_run_id = revised["run_id"]
    return server, coupled


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("address", "value"),
    [
        ("/ch/2/mute", 1),
        ("/ch/2/main/1/on", 0),
        ("/ch/2/main/1/pre", 1),
        ("/ch/2/in/conn/in", 99),
        ("/ch/2/tags", "#D2"),
        ("/ch/1/fdr", -11.0),
    ],
)
async def test_coupled_group_source_path_drift_is_local_and_blocks_before_send(
    address,
    value,
):
    server, coupled = _coupled_fader_refresh_case(
        "coupled-path-drift-" + address.replace("/", "-")
    )
    server.mixer_client.values[address] = value

    refreshed = await server.refresh_soundcheck_child_preflight(coupled)

    assert refreshed["refreshed"] is True
    assert refreshed["child_local_blockers_by_transaction"][
        coupled["transaction_id"]
    ]
    assert server._soundcheck_global_blockers[coupled["run_id"]] == []
    assert server.mixer_client.sent == []


@pytest.mark.asyncio
async def test_coupled_group_missing_mute_readback_is_local_and_fail_closed():
    server, coupled = _coupled_fader_refresh_case("coupled-missing-mute")
    server.mixer_client.values.pop("/ch/2/mute")

    refreshed = await server.refresh_soundcheck_child_preflight(coupled)

    local = refreshed["child_local_blockers_by_transaction"][
        coupled["transaction_id"]
    ]
    assert any("readback_failed" in reason for reason in local)
    assert server._soundcheck_global_blockers[coupled["run_id"]] == []
    assert server.mixer_client.sent == []


def test_coupled_group_contract_binds_original_targets_to_source_intent():
    server, coupled = _coupled_fader_refresh_case("intent-binding")
    tampered = copy.deepcopy(coupled)
    member = next(
        operation
        for operation in tampered["operations"]
        if operation["channel"] == 2
    )
    member["metadata"]["pre_translation_final_target_value"] -= 1.0
    member["metadata"]["final_target_value"] -= 1.0

    errors = coupled_safe_alternative_plan_errors(
        tampered,
        source_coverage=server._soundcheck_manifests["intent-binding"][
            "source_coverage"
        ],
    )

    assert "coupled_group_source_intent_binding_mismatch" in errors


def test_coupled_group_requires_exact_ordered_logical_operation_membership():
    server, coupled = _coupled_fader_refresh_case("logical-membership")
    manifest = copy.deepcopy(server._soundcheck_manifests["logical-membership"])
    group = next(
        plan
        for plan in manifest["plans"]
        if plan["transaction_id"] == coupled["transaction_id"]
    )
    group["logical_correction_ids"] = group["logical_correction_ids"][:-1]

    errors = soundcheck_manifest_logical_id_errors(manifest)

    assert any(
        error.startswith("plan_logical_ids_must_equal_ordered_operation_ids")
        for error in errors
    )
    assert "coupled_group_logical_ids_mismatch" in errors


def test_mixed_group_preflight_is_applyable_without_expanding_live_authority():
    _server, coupled = _coupled_fader_refresh_case("aggregate-group-preflight")
    operation_ids = coupled["logical_correction_ids"]
    now = time.time()
    by_operation = {
        operation_ids[0]: {
            "operation_id": operation_ids[0],
            "disposition": "already_satisfied",
            "reason": "physical_target_already_satisfied",
            "physical_verified": True,
            "audible_state_verified": True,
            "captured_at_epoch": now,
            "max_age_sec": 30.0,
        },
        operation_ids[1]: {
            "operation_id": operation_ids[1],
            "disposition": "apply",
            "reason": "transition_within_live_bounds",
            "physical_verified": True,
            "audible_state_verified": True,
            "captured_at_epoch": now,
            "max_age_sec": 30.0,
        },
    }
    registry_before = copy.deepcopy(supervised_console_family_registry())
    aggregate = plan_preflight_assessment(coupled, by_operation)
    blockers = _apply_blockers_for_transaction_plan(
        {**coupled, "preflight_assessment": aggregate}
    )
    registry_after = supervised_console_family_registry()

    assert aggregate["disposition"] == "apply"
    assert aggregate["reason"] == "coupled_group_preflight_applyable"
    assert blockers == ["write_kind_not_live_allowed:fader:fader"]
    assert registry_after == registry_before
    assert registry_after["fader"]["live_allowed_write_kinds"] == []


def test_safe_core_and_show_mix_are_separate_readiness_levels():
    actions = [
        ChannelFaderMove("dry balance", 1, -12.0),
        ChannelEQMove("show tone", 1, 1, 250.0, -2.0, 1.0),
    ]
    manifest = build_soundcheck_transaction_plans(
        actions,
        run_id="readiness-levels",
    ).to_dict()
    by_kind = {
        raw["write_kind"]: raw["operation_id"]
        for plan in manifest["plans"]
        for raw in plan["operations"]
    }
    outcomes = {
        by_kind["fader"]: _committed_outcome(),
        by_kind["eq"]: {
            "disposition": "operator_review",
            "reason": "relative_eq_boost_not_allowed",
        },
    }

    safe_core = aggregate_soundcheck_readiness(
        manifest,
        outcomes=outcomes,
        proposal_only=False,
        **_ready_evidence(),
    )
    assert safe_core["mix_readiness"] == "safe_core_ready"

    outcomes[by_kind["eq"]] = _committed_outcome()
    show_mix = aggregate_soundcheck_readiness(
        manifest,
        outcomes=outcomes,
        proposal_only=False,
        **_ready_evidence(),
    )
    assert show_mix["mix_readiness"] == "show_mix_ready"


def test_proposal_only_and_missing_evidence_are_always_incomplete():
    manifest = build_soundcheck_transaction_plans(
        [ChannelFaderMove("balance", 1, -12.0)],
        run_id="proposal-only",
    ).to_dict()
    operation_id = manifest["plans"][0]["operations"][0]["operation_id"]

    proposal = build_proposal_only_readiness(manifest)
    no_evidence = aggregate_soundcheck_readiness(
        manifest,
        outcomes={operation_id: _committed_outcome()},
        proposal_only=False,
    )

    assert proposal["mix_readiness"] == "soundcheck_incomplete"
    assert proposal["execution_status"] == "awaiting_operator_review"
    assert no_evidence["mix_readiness"] == "soundcheck_incomplete"
    assert no_evidence["readiness_evidence_complete"] is False


def test_eq_cannot_resolve_show_mix_without_physical_audible_enable_evidence():
    manifest = build_soundcheck_transaction_plans(
        [ChannelEQMove("tone", 1, 1, 250.0, -2.0, 1.0)],
        run_id="eq-audible-proof",
    ).to_dict()
    operation_id = manifest["plans"][0]["operations"][0]["operation_id"]
    outcome = _committed_outcome()
    outcome["audible_state_verified"] = False

    readiness = aggregate_soundcheck_readiness(
        manifest,
        outcomes={operation_id: outcome},
        proposal_only=False,
        **_ready_evidence(),
    )

    assert readiness["mix_readiness"] == "safe_core_ready"
    assert readiness["logical_counts"]["applied"] == 0
    assert readiness["unresolved_required_operation_ids"] == [operation_id]


def test_bus_tonal_changes_require_physical_enable_readback_before_show_ready():
    for action in (
        BusEQMove("bus tone", 1, 2, 250.0, -2.0, 1.0),
        BusCompressorMakeupAdjust("bus makeup", 1, -2.0),
    ):
        manifest = build_soundcheck_transaction_plans(
            [action],
            run_id=f"bus-audible-{action.action_type}",
        ).to_dict()
        operation = manifest["plans"][0]["operations"][0]
        assert operation["metadata"]["audible_state_requirements"]
        outcome = _committed_outcome()
        outcome["audible_state_verified"] = False

        readiness = aggregate_soundcheck_readiness(
            manifest,
            outcomes={operation["operation_id"]: outcome},
            proposal_only=False,
            **_ready_evidence(),
        )

        assert readiness["mix_readiness"] == "safe_core_ready"
        assert readiness["logical_counts"]["applied"] == 0


def test_duplicate_logical_identity_is_a_global_manifest_contract_failure():
    manifest = build_soundcheck_transaction_plans(
        [ChannelFaderMove("balance", 1, -12.0)],
        run_id="duplicate-manifest-id",
    ).to_dict()
    operation_id = manifest["plans"][0]["operations"][0]["operation_id"]
    manifest["unsupported_actions"] = [
        {
            "logical_correction_id": operation_id,
            "reason": "malformed_duplicate",
            "disposition": "operator_review",
        }
    ]

    readiness = aggregate_soundcheck_readiness(
        manifest,
        outcomes={operation_id: _committed_outcome()},
        proposal_only=False,
        **_ready_evidence(),
    )

    assert readiness["mix_readiness"] == "soundcheck_incomplete"
    assert readiness["safety_state"] == "blocked_global"
    assert readiness["logical_counts"]["total"] == 1
    assert readiness["global_blockers"] == [
        f"duplicate_logical_correction_id:{operation_id}"
    ]


def test_matching_eq_target_with_processor_off_is_operator_review_not_already_satisfied():
    manifest = build_soundcheck_transaction_plans(
        [ChannelEQMove("tone", 1, 1, 250.0, -2.0, 1.0)],
        run_id="eq-off-target-match",
    ).to_dict()
    raw = manifest["plans"][0]["operations"][0]
    operation = _operation_from_raw(raw)
    adapter = WingSupervisedFamilyAdapter("eq_hpf", object())
    values = {
        write.address: write.value
        for write in parameter_writes_for_operation(operation)
    }
    values["/ch/1/eq/on"] = 0

    assessment = assess_soundcheck_manifest(
        manifest,
        snapshots_by_operation={
            operation.operation_id: {
                "physical_verified": True,
                "values": values,
            }
        },
        adapters={"eq_hpf": adapter},
    )

    item = assessment["assessments"][0]
    assert item["disposition"] == "operator_review"
    assert item["reason"] == (
        "processing_enable_state_change_requires_separate_approval"
    )
    assert item["audible_state_verified"] is False


def test_main_and_reconciliation_evidence_expire_on_status_reaggregation():
    manifest = build_soundcheck_transaction_plans(
        [ChannelFaderMove("balance", 1, -12.0)],
        run_id="readiness-expiry",
    ).to_dict()
    operation_id = manifest["plans"][0]["operations"][0]["operation_id"]
    outcome = _committed_outcome()
    outcome["reconciliation_captured_at_epoch"] = time.time() - 31.0
    evidence = _ready_evidence()
    evidence["main_47_48_evidence"]["captured_at_epoch"] = time.time() - 6.0

    readiness = aggregate_soundcheck_readiness(
        manifest,
        outcomes={operation_id: outcome},
        proposal_only=False,
        **evidence,
    )

    assert readiness["mix_readiness"] == "soundcheck_incomplete"
    assert readiness["main_47_48_evidence_fresh"] is False
    assert readiness["logical_counts"]["pending"] == 1
    assert readiness["unresolved_reasons"]["soundcheck:evidence:fresh"] == (
        "main_47_48_evidence_stale"
    )


def test_resolved_fader_without_explicit_show_stage_coverage_is_not_show_ready():
    manifest = build_soundcheck_transaction_plans(
        [ChannelFaderMove("dry balance", 1, -12.0)],
        run_id="missing-show-coverage",
    ).to_dict()
    operation_id = manifest["plans"][0]["operations"][0]["operation_id"]

    readiness = aggregate_soundcheck_readiness(
        manifest,
        outcomes={operation_id: _committed_outcome()},
        proposal_only=False,
        source_coverage={
            "verified": True,
            "captured_at_epoch": time.time(),
            "max_age_sec": 300.0,
            "stage_coverage": {
                "source_control": True,
                "phase_alignment_review": True,
            },
        },
        main_47_48_evidence=_ready_evidence()["main_47_48_evidence"],
    )

    assert readiness["mix_readiness"] == "safe_core_ready"
    assert readiness["show_mix_stage_coverage_complete"] is False
    assert readiness["execution_status"] == "child_execution_complete_mix_incomplete"


def test_selected_sources_require_detailed_per_source_stage_coverage():
    manifest = build_soundcheck_transaction_plans(
        [ChannelFaderMove("balance", 1, -12.0)],
        run_id="missing-per-source-coverage",
    ).to_dict()
    operation_id = manifest["plans"][0]["operations"][0]["operation_id"]
    aggregate_only = {
        **_ready_evidence()["source_coverage"],
        "selected_channel_ids": [1, 2],
    }

    readiness = aggregate_soundcheck_readiness(
        manifest,
        outcomes={operation_id: _committed_outcome()},
        proposal_only=False,
        source_coverage=aggregate_only,
        main_47_48_evidence=_ready_evidence()["main_47_48_evidence"],
    )

    assert readiness["mix_readiness"] == "soundcheck_incomplete"
    assert readiness["safe_core_stage_coverage_complete"] is False
    assert readiness["show_mix_stage_coverage_complete"] is False


def test_global_blocker_and_rolled_back_applied_ids_never_resolve_readiness():
    manifest = build_soundcheck_transaction_plans(
        [ChannelFaderMove("balance", 1, -12.0)],
        run_id="rollback-truth",
    ).to_dict()
    operation_id = manifest["plans"][0]["operations"][0]["operation_id"]
    rolled_back = {
        "status": "rolled_back",
        "applied_operation_ids": [operation_id],
        "physical_verified": True,
        "audio_recheck_passed": True,
        "wal_terminal": "rolled_back",
    }

    readiness = aggregate_soundcheck_readiness(
        manifest,
        outcomes={operation_id: rolled_back},
        proposal_only=False,
        global_blockers=["writer_conflict"],
        **_ready_evidence(),
    )

    assert readiness["mix_readiness"] == "soundcheck_incomplete"
    assert readiness["safety_state"] == "blocked_global"
    assert readiness["logical_counts"]["applied"] == 0
    assert readiness["physical_step_counts"]["rolled_back"] == 1


def test_replacement_requires_new_evidence_revision_and_no_risk_escalation():
    original = {
        "operation_id": "fader-original",
        "family_id": "fader",
        "write_kind": "fader",
    }
    replacement = {
        "operation_id": "fader-replacement",
        "family_id": "fader",
        "write_kind": "fader",
        "target_value": -13.0,
        "metadata": {
            "replacement_of": "fader-original",
            "substitution_hop": 1,
            "trusted_reanalysis_id": "analysis-2",
            "analysis_evidence_digest": "abc123",
            "absolute_target": True,
            "plan_revision": 2,
        },
    }

    accepted = validate_replacement(
        original,
        replacement,
        effective_live_write_kinds={"fader"},
        trusted_reanalysis_evidence={"analysis-2": "abc123"},
        current_plan_revision=1,
    )
    assert accepted["accepted"] is True

    replacement["family_id"] = "fx_sends_inserts"
    replacement["write_kind"] = "send"
    rejected = validate_replacement(
        original,
        replacement,
        effective_live_write_kinds={"fader", "send"},
        trusted_reanalysis_evidence={"analysis-2": "abc123"},
        current_plan_revision=1,
    )
    assert rejected["accepted"] is False
    assert "replacement_risk_escalation_not_allowed" in rejected["reasons"]

    replacement["family_id"] = "fader"
    replacement["write_kind"] = "gain"
    mismatched = validate_replacement(
        original,
        replacement,
        effective_live_write_kinds={"gain"},
        trusted_reanalysis_evidence={"analysis-2": "abc123"},
        current_plan_revision=1,
    )
    assert mismatched["accepted"] is False
    assert "replacement_write_kind_family_mismatch" in mismatched["reasons"]


def test_bridge_has_one_apply_surface_for_planned_fader_correction():
    action = ChannelFaderMove("one correction", 2, -8.5)
    manifest = build_soundcheck_transaction_plans(
        [action],
        run_id="single-surface",
    ).to_dict()
    bundle = {
        "proposals": [
            {
                "proposal_id": "generic-fader",
                "action_type": "set_fader_db",
                "target": {"mixer_channel": 2},
                "current_state": {"fader_db": -10.0},
                "requested_state": {"fader_db": -8.5},
            }
        ],
        "supervised_transaction_plans": manifest,
    }

    proposals = build_soundcheck_operator_proposals(bundle)

    assert len(proposals) == 1
    assert proposals[0]["kind"] == "supervised_soundcheck_transaction"
    assert proposals[0]["logical_correction_ids"] == [
        operation_from_soundcheck_action(action).operation_id
    ]


def test_manifest_does_not_expand_standing_live_authority():
    before = {
        contract.family_id: contract.live_allowed_write_kinds
        for contract in FAMILY_CONTRACTS
    }
    build_soundcheck_transaction_plans(
        [ChannelFaderMove("decompose only", 1, -12.0)],
        run_id="no-authority-change",
    ).to_dict()
    after = {
        contract.family_id: contract.live_allowed_write_kinds
        for contract in FAMILY_CONTRACTS
    }

    assert before == after
    assert all(not kinds for kinds in after.values())


def test_first_trusted_import_rejects_nested_target_digest_tamper():
    manifest = build_soundcheck_transaction_plans(
        [ChannelFaderMove("content bound", 1, -12.0)],
        run_id="content-digest-tamper",
    ).to_dict()
    manifest["plans"][0]["operations"][0]["target_value"] = -3.0
    server = object.__new__(AutoMixerServer)
    server._soundcheck_manifests = {}
    server._soundcheck_parent_journal_error = None
    server.last_soundcheck_readiness = {}
    server.get_operator_mode_status = lambda: {}

    result = server.import_soundcheck_recommendations_to_operator_queue(
        {"supervised_transaction_plans": manifest},
        trusted_engine_bundle=True,
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "soundcheck_manifest_content_digest_mismatch"


@pytest.mark.parametrize(
    "schema_version",
    [None, "soundcheck_batch_manifest/v1", "soundcheck_batch_manifest/v3"],
)
def test_first_trusted_import_rejects_non_v2_manifest_schema(schema_version):
    manifest = build_soundcheck_transaction_plans(
        [ChannelFaderMove("schema bound", 1, -12.0)],
        run_id=f"schema-reject-{schema_version}",
    ).to_dict()
    if schema_version is None:
        manifest.pop("schema_version", None)
    else:
        manifest["schema_version"] = schema_version
    server = object.__new__(AutoMixerServer)
    server._soundcheck_manifests = {}
    server._soundcheck_parent_journal_error = None
    server.last_soundcheck_readiness = {}
    server.get_operator_mode_status = lambda: {}

    result = server.import_soundcheck_recommendations_to_operator_queue(
        {"supervised_transaction_plans": manifest},
        trusted_engine_bundle=True,
    )

    assert result["status"] == "blocked"
    assert result["reason"] == "unsupported_soundcheck_manifest_schema"


def test_manifest_journal_failure_has_no_publication_and_retry_can_succeed():
    manifest = build_soundcheck_transaction_plans(
        [ChannelFaderMove("durable registration", 1, -12.0)],
        run_id="durable-registration-retry",
    ).to_dict()
    server = object.__new__(AutoMixerServer)
    server.operator_proposal_queue = OperatorProposalQueue()
    server.mixer_client = None
    server._soundcheck_manifests = {}
    server._soundcheck_outcomes = {}
    server._soundcheck_global_blockers = {}
    server._soundcheck_preflight_assessments = {}
    server._soundcheck_replan_drafts = {}
    server._soundcheck_console_change_subscriptions = {}
    server._active_soundcheck_run_id = ""
    server._soundcheck_parent_journal_error = None
    server.last_soundcheck_readiness = {}
    server.get_operator_mode_status = lambda: {
        "mode": "monitor",
        "capabilities": {
            "can_create_proposals": True,
            "can_apply_to_console": False,
            "allowed_live_write_kinds": [],
        },
    }
    server._append_soundcheck_parent_journal = lambda *_args, **_kwargs: None

    blocked = server.import_soundcheck_recommendations_to_operator_queue(
        {"proposals": [], "supervised_transaction_plans": manifest},
        trusted_engine_bundle=True,
    )

    assert blocked["reason"] == "soundcheck_parent_journal_unavailable"
    assert manifest["run_id"] not in server._soundcheck_manifests
    assert server.operator_proposal_queue.summary()["pending_count"] == 0
    assert server._active_soundcheck_run_id == ""

    server._append_soundcheck_parent_journal = (
        lambda *_args, **_kwargs: "/tmp/soundcheck-parent-journal"
    )
    retried = server.import_soundcheck_recommendations_to_operator_queue(
        {"proposals": [], "supervised_transaction_plans": manifest},
        trusted_engine_bundle=True,
    )

    assert retried.get("reason") != "soundcheck_parent_journal_unavailable"
    assert manifest["run_id"] in server._soundcheck_manifests
    assert server._active_soundcheck_run_id == manifest["run_id"]


def test_revision_import_rejects_applying_child_before_journal_append():
    current = build_soundcheck_transaction_plans(
        [ChannelFaderMove("current", 1, -12.0)],
        run_id="applying-before-journal",
    ).to_dict()
    revised = copy.deepcopy(current)
    revised["plan_revision"] = 2
    revised["batch_id"] = f"{current['run_id']}:r2"
    revised["source_coverage"] = {"captured_at_epoch": time.time()}
    for plan in revised["plans"]:
        plan["plan_revision"] = 2
    revised["normal_plans"] = list(revised["plans"])
    revised["separate_approval_plans"] = []
    bind_soundcheck_manifest_content_sha256(revised)

    class ApplyingQueue:
        def pending_actions(self):
            return [
                {
                    "id": "claimed-child",
                    "status": "applying",
                    "raw": {"soundcheck_run_id": current["run_id"]},
                }
            ]

    server = object.__new__(AutoMixerServer)
    server.operator_proposal_queue = ApplyingQueue()
    server._soundcheck_manifests = {current["run_id"]: current}
    server._active_soundcheck_run_id = current["run_id"]
    server._soundcheck_parent_journal_error = None
    server.last_soundcheck_readiness = {"run_id": current["run_id"]}
    server.get_operator_mode_status = lambda: {}
    journal_calls = []
    server._append_soundcheck_parent_journal = (
        lambda *_args, **_kwargs: journal_calls.append(True) or "/tmp/journal"
    )

    result = server.import_soundcheck_recommendations_to_operator_queue(
        {"proposals": [], "supervised_transaction_plans": revised},
        trusted_engine_bundle=True,
    )

    assert result["reason"] == "soundcheck_previous_revision_apply_in_progress"
    assert journal_calls == []
    assert server._soundcheck_manifests[current["run_id"]][
        "plan_revision"
    ] == 1


def test_initial_accepted_bundle_acknowledgement_makes_reimport_idempotent():
    from auto_soundcheck_engine import AutoSoundcheckEngine

    engine = AutoSoundcheckEngine(
        num_channels=1,
        selected_channels=[1],
        auto_discover=False,
        observe_only=True,
    )
    engine._soundcheck_transaction_actions = [
        ChannelFaderMove("accepted initial target", 1, -12.0)
    ]
    raw_bundle = engine.get_soundcheck_recommendation_bundle()
    accepted_bundle = copy.deepcopy(raw_bundle)
    accepted_manifest = accepted_bundle["supervised_transaction_plans"]
    accepted_manifest["plans"] = [
        {
            **dict(plan),
            "preflight_assessment": {"disposition": "operator_review"},
        }
        for plan in accepted_manifest.get("plans") or []
    ]

    server = object.__new__(AutoMixerServer)
    server.operator_proposal_queue = OperatorProposalQueue()
    server.mixer_client = None
    server.auto_soundcheck_engine = engine
    server._soundcheck_manifests = {}
    server._soundcheck_outcomes = {}
    server._soundcheck_global_blockers = {}
    server._soundcheck_preflight_assessments = {}
    server._soundcheck_replan_drafts = {}
    server._soundcheck_console_change_subscriptions = {}
    server._active_soundcheck_run_id = ""
    server._soundcheck_parent_journal_error = None
    server.last_soundcheck_readiness = {}
    server.get_operator_mode_status = lambda: {
        "mode": "monitor",
        "capabilities": {
            "can_create_proposals": True,
            "can_apply_to_console": False,
            "allowed_live_write_kinds": [],
        },
    }
    server._append_soundcheck_parent_journal = (
        lambda *_args, **_kwargs: "/tmp/soundcheck-parent-journal"
    )

    first = server.import_soundcheck_recommendations_to_operator_queue(
        accepted_bundle,
        trusted_engine_bundle=True,
    )
    second = server.import_soundcheck_recommendations_to_operator_queue()

    assert first["engine_bundle_acknowledged"] is True
    assert second.get("reason") != "soundcheck_manifest_same_identity_content_mismatch"
    assert engine.get_soundcheck_recommendation_bundle()[
        "supervised_transaction_plans"
    ] == accepted_manifest


def test_fresh_evidence_revision_can_keep_same_musical_fingerprint():
    current = build_soundcheck_transaction_plans(
        [ChannelFaderMove("same target", 1, -12.0)],
        run_id="evidence-only-revision",
    ).to_dict()
    revised = copy.deepcopy(current)
    revised["plan_revision"] = 2
    revised["batch_id"] = f"{current['run_id']}:r2"
    revised["source_coverage"] = {
        "verified": True,
        "captured_at_epoch": time.time(),
        "max_age_sec": 30.0,
    }
    for plan in revised["plans"]:
        plan["plan_revision"] = 2
    revised["normal_plans"] = list(revised["plans"])
    revised["separate_approval_plans"] = []
    bind_soundcheck_manifest_content_sha256(revised)
    assert revised["plan_fingerprint"] == current["plan_fingerprint"]
    assert revised["manifest_content_sha256"] != current["manifest_content_sha256"]

    server = object.__new__(AutoMixerServer)
    server.operator_proposal_queue = OperatorProposalQueue()
    server.mixer_client = None
    server._soundcheck_manifests = {current["run_id"]: current}
    server._soundcheck_outcomes = {current["run_id"]: {}}
    server._soundcheck_global_blockers = {current["run_id"]: []}
    server._soundcheck_preflight_assessments = {}
    server._soundcheck_replan_drafts = {}
    server._soundcheck_console_change_subscriptions = {}
    server._active_soundcheck_run_id = current["run_id"]
    server._soundcheck_parent_journal_error = None
    server.last_soundcheck_readiness = {"run_id": current["run_id"]}
    server.get_operator_mode_status = lambda: {
        "mode": "monitor",
        "capabilities": {
            "can_create_proposals": True,
            "can_apply_to_console": False,
            "allowed_live_write_kinds": [],
        },
    }
    server._append_soundcheck_parent_journal = (
        lambda *_args, **_kwargs: "/tmp/soundcheck-parent-journal"
    )

    imported = server.import_soundcheck_recommendations_to_operator_queue(
        {"proposals": [], "supervised_transaction_plans": revised},
        trusted_engine_bundle=True,
    )

    assert imported.get("reason") != (
        "soundcheck_manifest_revision_without_content_change"
    )
    assert server._soundcheck_manifests[current["run_id"]][
        "plan_revision"
    ] == 2


@pytest.mark.asyncio
async def test_stage_reanalysis_promotes_only_next_same_run_revision():
    current = build_soundcheck_transaction_plans(
        [ChannelFaderMove("first analysis", 1, -12.0)],
        run_id="stage-reanalysis-contract",
    ).to_dict()
    revised = copy.deepcopy(current)
    revised["plan_revision"] = 2
    revised["batch_id"] = f"{current['run_id']}:r2"
    revised["source_coverage"] = {
        "verified": True,
        "captured_at_epoch": time.time(),
        "max_age_sec": 30.0,
    }
    for plan in revised["plans"]:
        plan["plan_revision"] = 2
    revised["normal_plans"] = list(revised["plans"])
    revised["separate_approval_plans"] = []
    bind_soundcheck_manifest_content_sha256(revised)

    class FakeEngine:
        revision = 1
        prepared = 0
        discarded = []

        def prepare_soundcheck_revision_candidate(
            self,
            run_id,
            *,
            expected_plan_revision,
            expected_manifest_content_sha256,
        ):
            assert run_id == current["run_id"]
            assert expected_plan_revision == self.revision
            assert expected_manifest_content_sha256 == current[
                "manifest_content_sha256"
            ]
            self.prepared += 1
            return {
                "candidate_id": "candidate-r2",
                "run_id": run_id,
                "base_plan_revision": self.revision,
                "target_plan_revision": self.revision + 1,
                "base_manifest_content_sha256": expected_manifest_content_sha256,
                "bundle": {
                    "proposals": [],
                    "supervised_transaction_plans": revised,
                },
            }

        def commit_soundcheck_revision_candidate(
            self,
            candidate_id,
            *,
            accepted_manifest_content_sha256,
            accepted_bundle,
        ):
            assert candidate_id == "candidate-r2"
            assert accepted_manifest_content_sha256 == revised[
                "manifest_content_sha256"
            ]
            assert accepted_bundle["supervised_transaction_plans"][
                "manifest_content_sha256"
            ] == revised["manifest_content_sha256"]
            self.revision = 2

        def discard_soundcheck_revision_candidate(self, candidate_id):
            self.discarded.append(candidate_id)
            return True

        def discard_soundcheck_revision_candidates_for_run(self, run_id):
            assert run_id == current["run_id"]
            return 0

        def validate_soundcheck_revision_candidate(self, candidate_id):
            assert candidate_id == "candidate-r2"
            return True

        @contextmanager
        def soundcheck_revision_publication_scope(self, candidate_id):
            self.validate_soundcheck_revision_candidate(candidate_id)
            yield

    server = object.__new__(AutoMixerServer)
    server.operator_proposal_queue = OperatorProposalQueue()
    server._soundcheck_manifests = {current["run_id"]: current}
    server._active_soundcheck_run_id = current["run_id"]
    server.last_soundcheck_readiness = {"run_id": current["run_id"]}
    server.auto_soundcheck_engine = FakeEngine()
    server._soundcheck_reanalysis_lock = asyncio.Lock()
    server._soundcheck_preflight_assessments = {}
    server._soundcheck_preflight_timestamps = {}
    server._soundcheck_replan_drafts = {}
    server._soundcheck_global_blockers = {}
    imported = []

    writer_lock = asyncio.Lock()

    @asynccontextmanager
    async def writer_scope():
        async with writer_lock:
            yield True

    server._console_write_scope = writer_scope

    async def preflight(manifest):
        return {
            "run_id": manifest["run_id"],
            "plan_revision": manifest["plan_revision"],
            "plan_fingerprint": manifest["plan_fingerprint"],
            "assessments": [],
            "global_blockers": [],
        }

    server.preflight_soundcheck_manifest = preflight
    def import_bundle(bundle, **_kwargs):
        imported.append(bundle)
        server._soundcheck_manifests[current["run_id"]] = copy.deepcopy(
            bundle["supervised_transaction_plans"]
        )
        return {"status": "ok", "success": True}

    server.import_soundcheck_recommendations_to_operator_queue = import_bundle

    await writer_lock.acquire()
    reanalysis_task = asyncio.create_task(
        server.reanalyze_soundcheck_stage(
            run_id=current["run_id"],
            approved=True,
            approval_id="stage-reanalysis-approval",
        )
    )
    await asyncio.sleep(0.01)
    assert server.auto_soundcheck_engine.prepared == 0
    writer_lock.release()
    result = await reanalysis_task

    assert result["status"] == "ready_for_review"
    assert result["from_plan_revision"] == 1
    assert result["to_plan_revision"] == 2
    assert result["live_mixer_writes"] is False
    assert result["live_authority_changed"] is False
    assert server.auto_soundcheck_engine.revision == 2
    assert server.auto_soundcheck_engine.prepared == 1
    assert server.auto_soundcheck_engine.discarded == []
    assert imported[0]["supervised_transaction_plans"]["run_id"] == current[
        "run_id"
    ]


@pytest.mark.asyncio
async def test_stage_reanalysis_failed_import_discards_candidate_and_retry_is_n_plus_one():
    current = build_soundcheck_transaction_plans(
        [ChannelFaderMove("first analysis", 1, -12.0)],
        run_id="stage-reanalysis-import-retry",
    ).to_dict()

    def next_revision(base):
        revised = copy.deepcopy(base)
        revised["plan_revision"] = 2
        revised["batch_id"] = f"{base['run_id']}:r2"
        revised["source_coverage"] = {
            "verified": True,
            "captured_at_epoch": time.time(),
            "max_age_sec": 30.0,
        }
        for plan in revised["plans"]:
            plan["plan_revision"] = 2
        revised["normal_plans"] = list(revised["plans"])
        revised["separate_approval_plans"] = []
        return bind_soundcheck_manifest_content_sha256(revised)

    revised = next_revision(current)

    class FakeEngine:
        revision = 1
        sequence = 0
        discarded = []

        def prepare_soundcheck_revision_candidate(
            self,
            run_id,
            *,
            expected_plan_revision,
            expected_manifest_content_sha256,
        ):
            assert run_id == current["run_id"]
            assert expected_plan_revision == self.revision == 1
            self.sequence += 1
            return {
                "candidate_id": f"candidate-{self.sequence}",
                "run_id": run_id,
                "base_plan_revision": 1,
                "target_plan_revision": 2,
                "base_manifest_content_sha256": expected_manifest_content_sha256,
                "bundle": {
                    "proposals": [],
                    "supervised_transaction_plans": revised,
                },
            }

        def commit_soundcheck_revision_candidate(
            self,
            candidate_id,
            *,
            accepted_manifest_content_sha256,
            accepted_bundle,
        ):
            assert candidate_id == "candidate-2"
            assert accepted_manifest_content_sha256 == revised[
                "manifest_content_sha256"
            ]
            assert accepted_bundle["supervised_transaction_plans"][
                "manifest_content_sha256"
            ] == revised["manifest_content_sha256"]
            self.revision = 2

        def discard_soundcheck_revision_candidate(self, candidate_id):
            self.discarded.append(candidate_id)
            return True

        def discard_soundcheck_revision_candidates_for_run(self, run_id):
            assert run_id == current["run_id"]
            return 0

        def validate_soundcheck_revision_candidate(self, candidate_id):
            assert candidate_id in {"candidate-1", "candidate-2"}
            return True

        @contextmanager
        def soundcheck_revision_publication_scope(self, candidate_id):
            self.validate_soundcheck_revision_candidate(candidate_id)
            yield

    server = object.__new__(AutoMixerServer)
    server.operator_proposal_queue = OperatorProposalQueue()
    server._soundcheck_manifests = {current["run_id"]: current}
    server._active_soundcheck_run_id = current["run_id"]
    server.last_soundcheck_readiness = {"run_id": current["run_id"]}
    server.auto_soundcheck_engine = FakeEngine()
    server._soundcheck_reanalysis_lock = asyncio.Lock()
    server._soundcheck_preflight_assessments = {}
    server._soundcheck_preflight_timestamps = {}
    server._soundcheck_replan_drafts = {}
    server._soundcheck_global_blockers = {}

    @asynccontextmanager
    async def writer_scope():
        yield True

    server._console_write_scope = writer_scope

    async def preflight(manifest):
        return {
            "run_id": manifest["run_id"],
            "plan_revision": manifest["plan_revision"],
            "plan_fingerprint": manifest["plan_fingerprint"],
            "assessments": [],
            "global_blockers": [],
        }

    server.preflight_soundcheck_manifest = preflight
    import_attempt = 0

    def import_bundle(bundle, **_kwargs):
        nonlocal import_attempt
        import_attempt += 1
        if import_attempt == 1:
            return {
                "status": "blocked",
                "success": False,
                "reason": "soundcheck_parent_journal_unavailable",
            }
        server._soundcheck_manifests[current["run_id"]] = copy.deepcopy(
            bundle["supervised_transaction_plans"]
        )
        return {"status": "ok", "success": True}

    server.import_soundcheck_recommendations_to_operator_queue = import_bundle

    first = await server.reanalyze_soundcheck_stage(
        run_id=current["run_id"],
        approved=True,
        approval_id="first-attempt",
    )
    second = await server.reanalyze_soundcheck_stage(
        run_id=current["run_id"],
        approved=True,
        approval_id="retry-attempt",
    )

    assert first["reason"] == "soundcheck_parent_journal_unavailable"
    assert first["fresh_analysis_required"] is True
    assert server.auto_soundcheck_engine.discarded == ["candidate-1"]
    assert second["status"] == "ready_for_review"
    assert second["from_plan_revision"] == 1
    assert second["to_plan_revision"] == 2
    assert server.auto_soundcheck_engine.revision == 2


def test_server_aggregates_child_results_but_never_infers_ready_from_one_apply():
    actions = [
        ChannelFaderMove("dry balance", 1, -12.0),
        ChannelEQMove("tone", 1, 1, 250.0, -2.0, 1.0),
    ]
    manifest = build_soundcheck_transaction_plans(
        actions,
        run_id="server-aggregate",
    ).to_dict()
    server = object.__new__(AutoMixerServer)
    server._soundcheck_manifests = {"server-aggregate": manifest}
    server._soundcheck_outcomes = {"server-aggregate": {}}
    server._soundcheck_global_blockers = {"server-aggregate": []}
    server._soundcheck_preflight_assessments = {
        "server-aggregate": {
            **_ready_evidence(),
            "assessments": [],
        }
    }
    server.last_soundcheck_readiness = build_proposal_only_readiness(manifest)
    fader_plan = next(
        plan
        for plan in manifest["plans"]
        if plan["operations"][0]["write_kind"] == "fader"
    )
    proposal = type(
        "Proposal",
        (),
        {
            "raw": {
                "supervised_transaction_plan": fader_plan,
                "soundcheck_run_id": "server-aggregate",
                "logical_correction_ids": fader_plan["logical_correction_ids"],
            }
        },
    )()

    readiness = server.record_soundcheck_child_result(
        proposal,
        {
            **_committed_outcome(),
            "reason": "transaction_applied_and_verified",
        },
    )

    assert readiness["logical_counts"]["applied"] == 0
    assert readiness["logical_counts"]["pending"] == 2
    assert readiness["mix_readiness"] == "soundcheck_incomplete"


class _ReadOnlyWing:
    is_connected = True

    def __init__(self, values):
        self.values = dict(values)
        self._observed_values = dict(values)
        self._generation = 0
        self._address_generations = {address: 0 for address in values}
        self.callbacks = {}
        self.queries = []
        self.sent = []

    def query_parameter(self, address, timeout):
        self.queries.append(address)
        return {
            "status": "received",
            "address": address,
            "value": self.values.get(address),
            "confirmed_round_trips": 2,
        }

    def send(self, address, value):
        self.sent.append((address, value))
        raise AssertionError("read-only preflight/reconciliation must never send")

    def get_observed_state_revision(self, addresses):
        for address in addresses:
            value = self.values.get(address)
            if self._observed_values.get(address) != value:
                self._generation += 1
                self._address_generations[address] = self._generation
                self._observed_values[address] = value
        return {
            "generation": self._generation,
            "addresses": {
                address: {
                    "generation": self._address_generations.get(address, 0),
                    "value": self.values.get(address),
                    "observed_at_epoch": time.time(),
                }
                for address in addresses
            },
            "captured_at_epoch": time.time(),
        }

    def subscribe(self, pattern, callback):
        self.callbacks.setdefault(pattern, []).append(callback)

    def external_update(self, address, value):
        self.values[address] = value
        self.get_observed_state_revision([address])
        for callback in list(self.callbacks.get("*", [])):
            callback(address, value)


def _current_test_console_fence(mixer, addresses=None):
    checkpoint = mixer.get_observed_state_revision(
        list(addresses or mixer.values)
    )
    rows = checkpoint["addresses"]
    return {
        "verified": True,
        "addresses": list(rows),
        "address_generations": {
            address: row["generation"] for address, row in rows.items()
        },
        "observed_values": {
            address: row["value"] for address, row in rows.items()
        },
        "global_generation_required": False,
        "global_generation": checkpoint["generation"],
    }


def _server_for_readiness_io(manifest, values):
    server = object.__new__(AutoMixerServer)
    run_id = manifest["run_id"]
    server._soundcheck_manifests = {run_id: manifest}
    server._soundcheck_outcomes = {run_id: {}}
    server._soundcheck_global_blockers = {run_id: []}
    server._soundcheck_preflight_assessments = {
        run_id: {
            "run_id": run_id,
            "plan_revision": manifest["plan_revision"],
            "plan_fingerprint": manifest["plan_fingerprint"],
            "assessments": [],
            "source_coverage": manifest.get("source_coverage", {}),
        }
    }
    server._soundcheck_preflight_timestamps = {}
    server._manual_write_emergency_stopped = False
    expanded_values = dict(values)
    for address in list(expanded_values):
        if not (address.startswith("/ch/") and address.endswith("/fdr")):
            continue
        channel = address.split("/")[2]
        expanded_values.setdefault(f"/ch/{channel}/mute", 0)
        expanded_values.setdefault(f"/ch/{channel}/main/1/on", 1)
        expanded_values.setdefault(f"/ch/{channel}/main/1/lvl", 0.0)
        expanded_values.setdefault(f"/ch/{channel}/main/1/pre", 0)
        expanded_values.setdefault(f"/ch/{channel}/in/conn/grp", "LOCAL")
        expanded_values.setdefault(f"/ch/{channel}/in/conn/in", int(channel))
        expanded_values.setdefault(f"/ch/{channel}/in/conn/altgrp", "OFF")
        expanded_values.setdefault(f"/ch/{channel}/in/conn/altin", 0)
        expanded_values.setdefault(f"/ch/{channel}/tags", "")
    server.mixer_client = _ReadOnlyWing(expanded_values)
    server._soundcheck_preflight_assessments[run_id][
        "console_change_fence"
    ] = _current_test_console_fence(server.mixer_client)
    server.last_soundcheck_readiness = {}

    async def audio_recheck(*_args):
        return AudioRecheckResult(
            "passed",
            "master_true_peak_safe",
            {
                "route_verified": True,
                "master_channels": [47, 48],
                "master_sample_peaks_dbfs": [-12.0, -11.0],
                "master_true_peak_dbtp": -8.0,
                "age_sec": 0.0,
                "max_age_sec": 5.0,
                "ceiling_dbtp": -1.0,
                "measurement_source": "test_main_47_48",
                "route_proof_id": "test-route-proof",
            },
        )

    server._supervised_transaction_audio_recheck = audio_recheck
    return server


@pytest.mark.asyncio
async def test_stale_batch_preflight_is_refreshed_per_child_without_transport_write():
    manifest = build_soundcheck_transaction_plans(
        [ChannelFaderMove("balance", 1, -12.0)],
        run_id="per-child-refresh",
    ).to_dict()
    manifest["source_coverage"] = _ready_evidence()["source_coverage"]
    plan = manifest["plans"][0]
    operation_id = plan["operations"][0]["operation_id"]
    server = _server_for_readiness_io(manifest, {"/ch/1/fdr": -10.0})
    server._soundcheck_preflight_assessments[manifest["run_id"]]["assessments"] = [
        {
            "operation_id": operation_id,
            "disposition": "apply",
            "captured_at_epoch": time.time() - 60.0,
            "max_age_sec": 30.0,
        }
    ]
    proposal = type(
        "Proposal",
        (),
        {"raw": {"supervised_transaction_plan": plan}},
    )()

    stale = server.validate_soundcheck_child_for_apply(proposal)
    refreshed = await server.refresh_soundcheck_child_preflight(plan)
    current = server.validate_soundcheck_child_for_apply(proposal)

    assert stale["reason"] == "soundcheck_physical_preflight_stale"
    assert refreshed["refreshed"] is True
    assert current["allowed"] is True
    assert server.mixer_client.queries
    assert server.mixer_client.sent == []


@pytest.mark.asyncio
async def test_child_refresh_keeps_snapshot_time_and_fails_if_audio_finishes_after_ttl(
    monkeypatch,
):
    manifest = build_soundcheck_transaction_plans(
        [ChannelFaderMove("balance", 1, -12.0)],
        run_id="slow-child-refresh",
    ).to_dict()
    manifest["source_coverage"] = _ready_evidence()["source_coverage"]
    plan = manifest["plans"][0]
    clock = {"now": 1000.0}
    monkeypatch.setattr(server_module.time, "time", lambda: clock["now"])
    server = _server_for_readiness_io(manifest, {"/ch/1/fdr": -10.0})

    async def slow_audio(*_args):
        clock["now"] = 1040.0
        return AudioRecheckResult(
            "passed",
            "master_true_peak_safe",
            {
                "route_verified": True,
                "master_channels": [47, 48],
                "master_sample_peaks_dbfs": [-12.0, -11.0],
                "master_true_peak_dbtp": -8.0,
                "age_sec": 0.0,
                "max_age_sec": 5.0,
                "ceiling_dbtp": -1.0,
            },
        )

    server._supervised_transaction_audio_recheck = slow_audio
    refreshed = await server.refresh_soundcheck_child_preflight(plan)

    item = next(
        row
        for row in refreshed["assessments"]
        if row["operation_id"] == plan["operations"][0]["operation_id"]
    )
    assert item["captured_at_epoch"] == 1000.0
    assert "soundcheck_preflight_collection_stale" in refreshed["global_blockers"]
    assert server.mixer_client.sent == []


@pytest.mark.asyncio
async def test_child_refresh_rejects_physical_drift_between_consistency_passes():
    manifest = build_soundcheck_transaction_plans(
        [ChannelFaderMove("balance", 1, -12.0)],
        run_id="child-consistency-drift",
    ).to_dict()
    manifest["source_coverage"] = _ready_evidence()["source_coverage"]
    plan = manifest["plans"][0]
    server = _server_for_readiness_io(manifest, {"/ch/1/fdr": -10.0})
    original_query = server.mixer_client.query_parameter
    drifted = {"done": False}

    def drift_after_first_read(address, timeout):
        result = original_query(address, timeout)
        if not drifted["done"]:
            drifted["done"] = True
            server.mixer_client.values[address] = -9.0
        return result

    server.mixer_client.query_parameter = drift_after_first_read

    refreshed = await server.refresh_soundcheck_child_preflight(plan)

    assert any(
        blocker.startswith("soundcheck_preflight_consistency_mismatch:")
        for blocker in refreshed["global_blockers"]
    )
    assert server.mixer_client.sent == []


def test_later_stage_gate_rejects_stale_foundation_checkpoint():
    manifest = build_soundcheck_transaction_plans(
        [
            ChannelFaderMove("foundation", 1, -12.0),
            ChannelEQMove("tone", 1, 1, 250.0, -2.0, 1.0),
        ],
        run_id="stage-checkpoint-stale",
    ).to_dict()
    foundation = next(
        plan for plan in manifest["plans"] if plan["stage"] == "foundation_level"
    )
    tonal = next(
        plan for plan in manifest["plans"] if plan["stage"] == "tonal_dynamics"
    )
    foundation_id = foundation["operations"][0]["operation_id"]
    tonal_id = tonal["operations"][0]["operation_id"]
    now = time.time()
    server = object.__new__(AutoMixerServer)
    server._soundcheck_manifests = {manifest["run_id"]: manifest}
    server._soundcheck_global_blockers = {manifest["run_id"]: []}
    server._soundcheck_outcomes = {
        manifest["run_id"]: {
            foundation_id: {
                "disposition": "already_satisfied",
                "physical_verified": True,
                "audible_state_verified": True,
                "evidence_captured_at_epoch": now - 60.0,
                "evidence_max_age_sec": 30.0,
                "final_reconciliation_verified": True,
                "reconciliation_captured_at_epoch": now - 60.0,
                "reconciliation_max_age_sec": 30.0,
            }
        }
    }
    server._soundcheck_preflight_assessments = {
        manifest["run_id"]: {
            "assessments": [
                {
                    "operation_id": foundation_id,
                    "disposition": "already_satisfied",
                    "physical_verified": True,
                    "audible_state_verified": True,
                    "captured_at_epoch": now - 60.0,
                    "max_age_sec": 30.0,
                },
                {
                    "operation_id": tonal_id,
                    "disposition": "apply",
                    "physical_verified": True,
                    "audible_state_verified": True,
                    "captured_at_epoch": now,
                    "max_age_sec": 30.0,
                },
            ]
        }
    }
    server.mixer_client = _ReadOnlyWing({"/ch/1/fdr": -12.0})
    server._soundcheck_preflight_assessments[manifest["run_id"]][
        "console_change_fence"
    ] = _current_test_console_fence(server.mixer_client)
    proposal = type(
        "Proposal",
        (),
        {"raw": {"supervised_transaction_plan": tonal}},
    )()

    result = server.validate_soundcheck_child_for_apply(proposal)

    assert result["allowed"] is False
    assert result["reason"] == "soundcheck_prior_stage_unresolved"
    assert result["unresolved_prior_operation_ids"] == [foundation_id]


def test_local_foundation_operator_residual_does_not_block_independent_tonal_child():
    manifest = build_soundcheck_transaction_plans(
        [
            ChannelFaderMove("blocked vocal lift", 1, -6.0),
            ChannelEQMove("independent cut", 2, 1, 250.0, -2.0, 1.0),
        ],
        run_id="local-residual-does-not-stop-stage",
    ).to_dict()
    foundation = next(
        plan for plan in manifest["plans"] if plan["stage"] == "foundation_level"
    )
    tonal = next(
        plan for plan in manifest["plans"] if plan["stage"] == "tonal_dynamics"
    )
    foundation_id = foundation["operations"][0]["operation_id"]
    tonal_id = tonal["operations"][0]["operation_id"]
    now = time.time()
    server = object.__new__(AutoMixerServer)
    server._soundcheck_manifests = {manifest["run_id"]: manifest}
    server._soundcheck_global_blockers = {manifest["run_id"]: []}
    server._soundcheck_outcomes = {
        manifest["run_id"]: {
            foundation_id: {
                "disposition": "operator_review",
                "reason": "fader_boost_not_allowed",
                "physical_verified": True,
            }
        }
    }
    server._soundcheck_preflight_assessments = {
        manifest["run_id"]: {
            "assessments": [
                {
                    "operation_id": foundation_id,
                    "disposition": "operator_review",
                    "reason": "fader_boost_not_allowed",
                    "physical_verified": True,
                    "audible_state_verified": True,
                    "captured_at_epoch": now,
                    "max_age_sec": 30.0,
                },
                {
                    "operation_id": tonal_id,
                    "disposition": "apply",
                    "physical_verified": True,
                    "audible_state_verified": True,
                    "captured_at_epoch": now,
                    "max_age_sec": 30.0,
                },
            ]
        }
    }
    server.mixer_client = _ReadOnlyWing(
        {"/ch/1/fdr": -12.0, "/ch/2/eq/1g": -2.0}
    )
    server._soundcheck_preflight_assessments[manifest["run_id"]][
        "console_change_fence"
    ] = _current_test_console_fence(server.mixer_client)
    proposal = type(
        "Proposal",
        (),
        {"raw": {"supervised_transaction_plan": tonal}},
    )()

    result = server.validate_soundcheck_child_for_apply(proposal)

    assert result["allowed"] is True


def test_unsupported_eq_boost_blocks_same_source_later_stage_until_reanalysis():
    manifest = build_soundcheck_transaction_plans(
        [
            ChannelEQMove("unsafe boost", 1, 1, 250.0, 2.0, 1.0),
            PanAdjust("later pan", 1, 5.0),
        ],
        run_id="unsupported-same-source-dependency",
    ).to_dict()
    assert manifest["unsupported_actions"][0]["channel"] == 1
    spatial = manifest["plans"][0]
    operation_id = spatial["operations"][0]["operation_id"]
    now = time.time()
    server = object.__new__(AutoMixerServer)
    server._soundcheck_manifests = {manifest["run_id"]: manifest}
    server._soundcheck_global_blockers = {manifest["run_id"]: []}
    server._soundcheck_outcomes = {manifest["run_id"]: {}}
    server.mixer_client = _ReadOnlyWing({"/ch/1/pan": 0.0})
    server._soundcheck_preflight_assessments = {
        manifest["run_id"]: {
            "console_change_fence": _current_test_console_fence(
                server.mixer_client
            ),
            "assessments": [
                {
                    "operation_id": operation_id,
                    "disposition": "apply",
                    "physical_verified": True,
                    "audible_state_verified": True,
                    "captured_at_epoch": now,
                    "max_age_sec": 30.0,
                }
            ],
        }
    }
    proposal = type(
        "Proposal",
        (),
        {"raw": {"supervised_transaction_plan": spatial}},
    )()

    result = server.validate_soundcheck_child_for_apply(proposal)

    assert result["allowed"] is False
    assert result["reason"] == "soundcheck_dependency_reanalysis_required"
    assert result["dependency_operation_ids"] == [
        manifest["unsupported_actions"][0]["logical_correction_id"]
    ]


def test_unsupported_eq_boost_blocks_same_source_compressor_in_same_stage():
    manifest = build_soundcheck_transaction_plans(
        [
            ChannelEQMove("unsafe boost", 1, 1, 250.0, 2.0, 1.0),
            CompressorAdjust(
                "same-stage compressor",
                1,
                -18.0,
                2.0,
                20.0,
                100.0,
                0.0,
                True,
            ),
        ],
        run_id="unsupported-same-stage-dependency",
    ).to_dict()
    compressor_plan = manifest["plans"][0]
    operation_id = compressor_plan["operations"][0]["operation_id"]
    now = time.time()
    server = object.__new__(AutoMixerServer)
    server._soundcheck_manifests = {manifest["run_id"]: manifest}
    server._soundcheck_global_blockers = {manifest["run_id"]: []}
    server._soundcheck_outcomes = {manifest["run_id"]: {}}
    server.mixer_client = _ReadOnlyWing({"/ch/1/dyn/thr": -24.0})
    server._soundcheck_preflight_assessments = {
        manifest["run_id"]: {
            "console_change_fence": _current_test_console_fence(
                server.mixer_client
            ),
            "assessments": [
                {
                    "operation_id": operation_id,
                    "disposition": "apply",
                    "physical_verified": True,
                    "audible_state_verified": True,
                    "captured_at_epoch": now,
                    "max_age_sec": 30.0,
                }
            ],
        }
    }
    proposal = type(
        "Proposal",
        (),
        {"raw": {"supervised_transaction_plan": compressor_plan}},
    )()

    result = server.validate_soundcheck_child_for_apply(proposal)

    assert result["allowed"] is False
    assert result["reason"] == "soundcheck_dependency_reanalysis_required"
    assert result["dependency_operation_ids"] == [
        manifest["unsupported_actions"][0]["logical_correction_id"]
    ]


def test_exact_m006_split_trim_blocks_same_stage_fader_until_reanalysis():
    manifest = build_soundcheck_transaction_plans(
        [
            ChannelGainMove("M006 trim", 1, -2.1728959600644924),
            ChannelFaderMove("M006 fader", 1, -3.8945411541898274),
        ],
        run_id="m006-trim-fader-dependency",
    ).to_dict()
    trim_plan, fader_plan = manifest["plans"]
    trim_id = trim_plan["operations"][0]["operation_id"]
    fader_id = fader_plan["operations"][0]["operation_id"]
    now = time.time()
    server = object.__new__(AutoMixerServer)
    server._soundcheck_manifests = {manifest["run_id"]: manifest}
    server._active_soundcheck_run_id = manifest["run_id"]
    server._soundcheck_global_blockers = {manifest["run_id"]: []}
    server._soundcheck_outcomes = {manifest["run_id"]: {}}
    server.mixer_client = _ReadOnlyWing(
        {
            "/ch/1/in/set/trim": 4.600000381469727,
            "/ch/1/fdr": -3.0999984741210938,
        }
    )
    server._soundcheck_preflight_assessments = {
        manifest["run_id"]: {
            "console_change_fence": _current_test_console_fence(
                server.mixer_client
            ),
            "assessments": [
                {
                    "operation_id": trim_id,
                    "disposition": "split_steps",
                    "physical_verified": True,
                    "audible_state_verified": True,
                    "captured_at_epoch": now,
                    "max_age_sec": 30.0,
                },
                {
                    "operation_id": fader_id,
                    "disposition": "apply",
                    "physical_verified": True,
                    "audible_state_verified": True,
                    "captured_at_epoch": now,
                    "max_age_sec": 30.0,
                },
            ],
        }
    }
    proposal = type(
        "Proposal",
        (),
        {"raw": {"supervised_transaction_plan": fader_plan}},
    )()

    result = server.validate_soundcheck_child_for_apply(proposal)

    assert result["allowed"] is False
    assert result["reason"] == "soundcheck_dependency_reanalysis_required"
    assert result["dependency_operation_ids"] == [trim_id]


def test_independent_same_stage_channel_does_not_wait_on_unrelated_sibling():
    manifest = build_soundcheck_transaction_plans(
        [
            ChannelFaderMove("channel one", 1, -12.0),
            ChannelFaderMove("channel two", 2, -14.0),
        ],
        run_id="independent-same-stage",
    ).to_dict()
    first_plan, second_plan = manifest["plans"]
    now = time.time()
    server = object.__new__(AutoMixerServer)
    server._soundcheck_manifests = {manifest["run_id"]: manifest}
    server._active_soundcheck_run_id = manifest["run_id"]
    server._soundcheck_global_blockers = {manifest["run_id"]: []}
    server._soundcheck_outcomes = {manifest["run_id"]: {}}
    server.mixer_client = _ReadOnlyWing(
        {"/ch/1/fdr": -10.0, "/ch/2/fdr": -13.0}
    )
    server._soundcheck_preflight_assessments = {
        manifest["run_id"]: {
            "console_change_fence": _current_test_console_fence(
                server.mixer_client
            ),
            "assessments": [
                {
                    "operation_id": plan["operations"][0]["operation_id"],
                    "disposition": "apply",
                    "physical_verified": True,
                    "audible_state_verified": True,
                    "captured_at_epoch": now,
                    "max_age_sec": 30.0,
                }
                for plan in (first_plan, second_plan)
            ],
        }
    }
    proposal = type(
        "Proposal",
        (),
        {"raw": {"supervised_transaction_plan": second_plan}},
    )()

    result = server.validate_soundcheck_child_for_apply(proposal)

    assert result["allowed"] is True


@pytest.mark.asyncio
async def test_same_source_prior_same_stage_checkpoint_is_refreshed():
    manifest = build_soundcheck_transaction_plans(
        [
            ChannelGainMove("M006 trim", 1, -2.1728959600644924),
            ChannelFaderMove("M006 fader", 1, -3.8945411541898274),
        ],
        run_id="m006-same-stage-refresh",
    ).to_dict()
    manifest["source_coverage"] = _ready_evidence()["source_coverage"]
    trim_plan, fader_plan = manifest["plans"]
    trim_id = trim_plan["operations"][0]["operation_id"]
    server = _server_for_readiness_io(
        manifest,
        {
            "/ch/1/in/set/trim": -2.1728959600644924,
            "/ch/1/fdr": -3.0999984741210938,
        },
    )
    server._active_soundcheck_run_id = manifest["run_id"]
    server._soundcheck_preflight_assessments[manifest["run_id"]][
        "assessments"
    ] = [
        {
            "operation_id": trim_id,
            "disposition": "apply",
            "physical_verified": True,
            "audible_state_verified": True,
            "captured_at_epoch": time.time(),
            "max_age_sec": 30.0,
        }
    ]

    refreshed = await server.refresh_soundcheck_child_preflight(fader_plan)

    assert refreshed["refreshed"] is True
    assert "/ch/1/in/set/trim" in server.mixer_client.queries
    assert "/ch/1/fdr" in server.mixer_client.queries


@pytest.mark.asyncio
async def test_later_tonal_child_rechecks_complete_prior_fader_source_path():
    manifest = build_soundcheck_transaction_plans(
        [
            ChannelFaderMove("foundation balance", 2, -12.0),
            ChannelEQMove("later tonal correction", 2, 1, 300.0, -2.0, 1.0),
        ],
        run_id="dependent-source-path-refresh",
    ).to_dict()
    now = time.time()
    evidence = _ready_evidence()["source_coverage"]
    manifest["source_coverage"] = {
        **evidence,
        "captured_at_epoch": now,
        "selected_channel_ids": [2],
        "signalled_channel_ids": [2],
        "per_source_stage_coverage": {
            "2": {
                stage: "analyzed"
                for stage in (
                    "foundation_level",
                    "source_control",
                    "phase_alignment_review",
                    "tonal_dynamics",
                    "group_balance",
                    "spatial_fx",
                )
            }
        },
        "per_source_fader_intent": _source_fader_intents(
            manifest["run_id"],
            now,
            {2: -12.0},
            {2: -12.0},
        ),
    }
    fader_plan = next(
        plan
        for plan in manifest["plans"]
        if plan["operations"][0]["family_id"] == "fader"
    )
    tonal_plan = next(
        plan
        for plan in manifest["plans"]
        if plan["operations"][0]["write_kind"] == "eq"
    )
    fader_id = fader_plan["operations"][0]["operation_id"]
    tonal_operation = _operation_from_raw(tonal_plan["operations"][0])
    values = {"/ch/2/fdr": -12.0}
    for write in parameter_writes_for_operation(tonal_operation):
        values[write.address] = (
            0.0 if write.address.endswith("g") else write.value
        )
    for read in audible_state_reads_for_operation(tonal_operation):
        values[read.address] = read.value
    server = _server_for_readiness_io(manifest, values)
    server._active_soundcheck_run_id = manifest["run_id"]
    server._soundcheck_preflight_assessments[manifest["run_id"]][
        "assessments"
    ] = [
        {
            "operation_id": fader_id,
            "disposition": "already_satisfied",
            "physical_verified": True,
            "audible_state_verified": True,
            "captured_at_epoch": now,
            "max_age_sec": 30.0,
        }
    ]
    server._soundcheck_outcomes[manifest["run_id"]][fader_id] = {
        "status": "preflight_classified",
        "disposition": "already_satisfied",
        "physical_verified": True,
        "audible_state_verified": True,
        "final_reconciliation_verified": True,
        "reconciliation_captured_at_epoch": now,
        "reconciliation_max_age_sec": 30.0,
        "wal_terminal": "not_applicable",
    }
    # Simulate a physical change with a lost callback/state-generation update.
    server.mixer_client.values["/ch/2/mute"] = 1

    refreshed = await server.refresh_soundcheck_child_preflight(tonal_plan)
    proposal = type(
        "Proposal",
        (),
        {"raw": {"supervised_transaction_plan": tonal_plan}},
    )()
    gate = server.validate_soundcheck_child_for_apply(proposal)

    blockers = refreshed["child_local_blockers_by_transaction"][
        tonal_plan["transaction_id"]
    ]
    assert any(
        blocker.startswith(
            "selected_source_dependency_path_unverified:2:"
            "selected_source_muted_after_analysis"
        )
        for blocker in blockers
    )
    assert gate["allowed"] is False
    assert gate["reason"] == "soundcheck_child_local_blocker_active"
    assert server.mixer_client.sent == []


@pytest.mark.asyncio
async def test_all_hold_tonal_plan_rechecks_path_before_apply_and_final_ready():
    manifest = build_soundcheck_transaction_plans(
        [ChannelEQMove("tonal correction", 1, 1, 300.0, -2.0, 1.0)],
        run_id="all-hold-tonal-source-path",
    ).to_dict()
    now = time.time()
    evidence = _ready_evidence()["source_coverage"]
    manifest["source_coverage"] = {
        **evidence,
        "captured_at_epoch": now,
        "selected_channel_ids": [1],
        "signalled_channel_ids": [1],
        "per_source_stage_coverage": {
            "1": {
                stage: "analyzed"
                for stage in (
                    "foundation_level",
                    "source_control",
                    "phase_alignment_review",
                    "tonal_dynamics",
                    "group_balance",
                    "spatial_fx",
                )
            }
        },
        "per_source_fader_intent": _source_fader_intents(
            manifest["run_id"],
            now,
            {},
            {1: -12.0},
        ),
    }
    plan = manifest["plans"][0]
    operation = _operation_from_raw(plan["operations"][0])
    values = {"/ch/1/fdr": -12.0}
    writes = parameter_writes_for_operation(operation)
    for write in writes:
        values[write.address] = (
            0.0 if write.address.endswith("g") else write.value
        )
    for read in audible_state_reads_for_operation(operation):
        values[read.address] = read.value
    server = _server_for_readiness_io(manifest, values)
    server._active_soundcheck_run_id = manifest["run_id"]

    server.mixer_client.values["/ch/1/mute"] = 1
    blocked = await server.refresh_soundcheck_child_preflight(plan)

    assert any(
        blocker.startswith(
            "selected_source_dependency_path_unverified:1:"
            "selected_source_muted_after_analysis"
        )
        for blocker in blocked["child_local_blockers_by_transaction"][
            plan["transaction_id"]
        ]
    )

    server.mixer_client.values["/ch/1/mute"] = 0
    for write in writes:
        server.mixer_client.values[write.address] = write.value
    server._soundcheck_outcomes[manifest["run_id"]][operation.operation_id] = {
        "status": "applied",
        "physical_verified": True,
        "audio_recheck_passed": True,
        "audible_state_verified": True,
        "wal_terminal": "committed",
        "final_reconciliation_verified": False,
    }
    ready = await server.reconcile_soundcheck_readiness(manifest["run_id"])
    server.mixer_client.values["/ch/1/mute"] = 1
    drifted = await server.reconcile_soundcheck_readiness(manifest["run_id"])

    assert ready["mix_readiness"] == "show_mix_ready"
    assert drifted["mix_readiness"] == "soundcheck_incomplete"
    assert drifted["source_coverage"]["selected_source_path_evidence"]["1"][
        "reason"
    ] == "selected_source_muted_after_analysis"
    assert server.mixer_client.sent == []


@pytest.mark.asyncio
async def test_final_reconciliation_promotes_only_current_physical_targets_and_detects_drift():
    manifest = build_soundcheck_transaction_plans(
        [ChannelFaderMove("balance", 1, -12.0)],
        run_id="final-reconciliation",
    ).to_dict()
    manifest["source_coverage"] = _ready_evidence()["source_coverage"]
    plan = manifest["plans"][0]
    operation_id = plan["operations"][0]["operation_id"]
    server = _server_for_readiness_io(manifest, {"/ch/1/fdr": -12.0})
    server._soundcheck_outcomes[manifest["run_id"]][operation_id] = {
        "status": "applied",
        "reason": "transaction_applied_and_verified",
        "physical_verified": True,
        "audio_recheck_passed": True,
        "audible_state_verified": True,
        "wal_terminal": "committed",
        "final_reconciliation_verified": False,
    }

    ready = await server.reconcile_soundcheck_readiness(manifest["run_id"])
    server.mixer_client.values["/ch/1/fdr"] = -8.0
    drifted = await server.reconcile_soundcheck_readiness(manifest["run_id"])

    assert ready["mix_readiness"] == "show_mix_ready"
    assert drifted["mix_readiness"] == "soundcheck_incomplete"
    assert server._soundcheck_outcomes[manifest["run_id"]][operation_id][
        "final_reconciliation_verified"
    ] is False
    assert server.mixer_client.sent == []


@pytest.mark.asyncio
async def test_final_reconciliation_revokes_safe_core_when_source_path_is_muted():
    server, _coupled = _coupled_fader_refresh_case(
        "final-source-path-loss",
        targets={1: -8.0, 2: -12.0},
    )
    manifest = server._soundcheck_manifests["final-source-path-loss"]
    source_coverage = manifest["source_coverage"]
    source_coverage["stage_coverage"].update(
        {
            "source_control": True,
            "phase_alignment_review": True,
        }
    )
    for stage_coverage in source_coverage["per_source_stage_coverage"].values():
        stage_coverage.update(
            {
                "source_control": "analyzed",
                "phase_alignment_review": "analyzed",
            }
        )
    for plan in manifest["plans"]:
        for operation in plan["operations"]:
            operation_id = operation["operation_id"]
            channel = operation["channel"]
            server.mixer_client.values[f"/ch/{channel}/fdr"] = operation[
                "target_value"
            ]
            server._soundcheck_outcomes[manifest["run_id"]][operation_id] = {
                "status": "applied",
                "reason": "transaction_applied_and_verified",
                "physical_verified": True,
                "audio_recheck_passed": True,
                "audible_state_verified": True,
                "wal_terminal": "committed",
                "final_reconciliation_verified": False,
            }

    ready = await server.reconcile_soundcheck_readiness(manifest["run_id"])
    server.mixer_client.values["/ch/2/mute"] = 1
    muted = await server.reconcile_soundcheck_readiness(manifest["run_id"])

    assert ready["mix_readiness"] == "safe_core_ready"
    assert ready["source_coverage"]["selected_source_path_evidence"]["2"][
        "reason"
    ] == (
        "selected_source_path_verified"
    )
    assert muted["mix_readiness"] == "soundcheck_incomplete"
    assert muted["source_coverage"]["selected_source_path_evidence"]["2"][
        "reason"
    ] == (
        "selected_source_muted_after_analysis"
    )
    assert muted["baseline_verified"] is False
    channel_two_fader_ids = [
        operation["operation_id"]
        for plan in manifest["plans"]
        for operation in plan["operations"]
        if operation["family_id"] == "fader" and operation["channel"] == 2
    ]
    assert channel_two_fader_ids
    for operation_id in channel_two_fader_ids:
        outcome = server._soundcheck_outcomes[manifest["run_id"]][operation_id]
        assert outcome["final_reconciliation_verified"] is False
        assert outcome["final_reconciliation_reason"] == (
            "selected_source_muted_after_analysis"
        )
    tonal_plan = copy.deepcopy(
        build_soundcheck_transaction_plans(
            [ChannelEQMove("later tonal correction", 2, 1, 300.0, -2.0, 1.0)],
            run_id=manifest["run_id"],
            plan_revision=manifest["plan_revision"],
        ).to_dict()["plans"][0]
    )
    tonal_plan["plan_fingerprint"] = manifest["plan_fingerprint"]
    tonal_plan["manifest_content_sha256"] = manifest[
        "manifest_content_sha256"
    ]
    manifest["plans"].append(tonal_plan)
    tonal_operation_id = tonal_plan["operations"][0]["operation_id"]
    server._soundcheck_preflight_assessments[manifest["run_id"]][
        "assessments"
    ].append(
        {
            "operation_id": tonal_operation_id,
            "disposition": "apply",
            "reason": "transition_preflight_passed",
            "physical_verified": True,
            "audible_state_verified": True,
            "captured_at_epoch": time.time(),
            "max_age_sec": 30.0,
        }
    )
    server._soundcheck_preflight_assessments[manifest["run_id"]][
        "source_balance_deadline_by_transaction"
    ] = {tonal_plan["transaction_id"]: time.time() + 30.0}
    server._soundcheck_console_fence_still_current = (
        lambda *_args, **_kwargs: True
    )
    proposal = type(
        "Proposal",
        (),
        {"raw": {"supervised_transaction_plan": tonal_plan}},
    )()

    next_stage = server.validate_soundcheck_child_for_apply(proposal)

    assert next_stage["allowed"] is False
    assert next_stage["reason"] == "soundcheck_prior_stage_unresolved"
    assert set(next_stage["unresolved_prior_operation_ids"]).issuperset(
        channel_two_fader_ids
    )
    assert server.mixer_client.sent == []


@pytest.mark.asyncio
async def test_final_reconciliation_rejects_snapshot_sweep_older_than_ttl(monkeypatch):
    manifest = build_soundcheck_transaction_plans(
        [
            ChannelFaderMove("one", 1, -12.0),
            ChannelFaderMove("two", 2, -12.0),
        ],
        run_id="stale-final-sweep",
    ).to_dict()
    manifest["source_coverage"] = _ready_evidence()["source_coverage"]
    clock = {"now": 1000.0, "queries": 0}
    monkeypatch.setattr(server_module.time, "time", lambda: clock["now"])
    server = _server_for_readiness_io(
        manifest,
        {"/ch/1/fdr": -12.0, "/ch/2/fdr": -12.0},
    )
    original_query = server.mixer_client.query_parameter

    def delayed_query(address, timeout):
        result = original_query(address, timeout)
        clock["queries"] += 1
        if clock["queries"] == 1:
            clock["now"] = 1000.0
        elif clock["queries"] == 2:
            clock["now"] = 1031.0
        return result

    server.mixer_client.query_parameter = delayed_query
    for plan in manifest["plans"]:
        operation_id = plan["operations"][0]["operation_id"]
        server._soundcheck_outcomes[manifest["run_id"]][operation_id] = {
            "status": "applied",
            "physical_verified": True,
            "audio_recheck_passed": True,
            "audible_state_verified": True,
            "wal_terminal": "committed",
            "final_reconciliation_verified": False,
        }

    readiness = await server.reconcile_soundcheck_readiness(manifest["run_id"])

    assert readiness["mix_readiness"] == "soundcheck_incomplete"
    assert "soundcheck_final_reconciliation_collection_stale" in (
        readiness["global_blockers"]
    )
    assert server.mixer_client.sent == []


@pytest.mark.asyncio
async def test_final_reconciliation_rejects_mid_sweep_operator_drift():
    manifest = build_soundcheck_transaction_plans(
        [
            ChannelFaderMove("one", 1, -12.0),
            ChannelFaderMove("two", 2, -12.0),
        ],
        run_id="mid-sweep-drift",
    ).to_dict()
    manifest["source_coverage"] = _ready_evidence()["source_coverage"]
    server = _server_for_readiness_io(
        manifest,
        {"/ch/1/fdr": -12.0, "/ch/2/fdr": -12.0},
    )
    original_query = server.mixer_client.query_parameter
    drift_address = "/ch/1/fdr"
    drifted = {"done": False}

    def drift_after_first_confirmed_read(address, timeout):
        result = original_query(address, timeout)
        if address == drift_address and not drifted["done"]:
            drifted["done"] = True
            server.mixer_client.values[address] = -8.0
        return result

    server.mixer_client.query_parameter = drift_after_first_confirmed_read
    for plan in manifest["plans"]:
        operation_id = plan["operations"][0]["operation_id"]
        server._soundcheck_outcomes[manifest["run_id"]][operation_id] = {
            "status": "applied",
            "physical_verified": True,
            "audio_recheck_passed": True,
            "audible_state_verified": True,
            "wal_terminal": "committed",
            "final_reconciliation_verified": False,
        }

    readiness = await server.reconcile_soundcheck_readiness(manifest["run_id"])

    assert readiness["mix_readiness"] == "soundcheck_incomplete"
    assert any(
        blocker.startswith("soundcheck_final_reconciliation_consistency_mismatch:")
        for blocker in readiness["global_blockers"]
    )
    assert server.mixer_client.sent == []


@pytest.mark.asyncio
async def test_final_reconciliation_fence_rejects_trailing_second_pass_drift():
    manifest = build_soundcheck_transaction_plans(
        [
            ChannelFaderMove("one", 1, -12.0),
            ChannelFaderMove("two", 2, -12.0),
        ],
        run_id="trailing-second-pass-drift",
    ).to_dict()
    manifest["source_coverage"] = _ready_evidence()["source_coverage"]
    server = _server_for_readiness_io(
        manifest,
        {"/ch/1/fdr": -12.0, "/ch/2/fdr": -12.0},
    )
    original_query = server.mixer_client.query_parameter
    query_count = 0

    def drift_after_first_read_of_second_pass(address, timeout):
        nonlocal query_count
        result = original_query(address, timeout)
        query_count += 1
        if query_count == 3:
            server.mixer_client.values["/ch/1/fdr"] = -8.0
        return result

    server.mixer_client.query_parameter = drift_after_first_read_of_second_pass
    for plan in manifest["plans"]:
        operation_id = plan["operations"][0]["operation_id"]
        server._soundcheck_outcomes[manifest["run_id"]][operation_id] = {
            "status": "applied",
            "physical_verified": True,
            "audio_recheck_passed": True,
            "audible_state_verified": True,
            "wal_terminal": "committed",
            "final_reconciliation_verified": False,
        }

    readiness = await server.reconcile_soundcheck_readiness(manifest["run_id"])

    assert readiness["mix_readiness"] == "soundcheck_incomplete"
    assert any(
        blocker.startswith(
            "soundcheck_final_reconciliation_change_fence_failed:"
        )
        for blocker in readiness["global_blockers"]
    )


@pytest.mark.asyncio
async def test_zero_operation_readiness_uses_global_generation_and_push_invalidation():
    manifest = build_soundcheck_transaction_plans(
        [],
        run_id="zero-operation-global-fence",
    ).to_dict()
    manifest["source_coverage"] = _ready_evidence()["source_coverage"]
    server = _server_for_readiness_io(manifest, {})

    ready = await server.reconcile_soundcheck_readiness(manifest["run_id"])
    server.mixer_client.external_update("/ch/16/eq/1g", -3.0)

    assert ready["mix_readiness"] == "show_mix_ready"
    assert server.last_soundcheck_readiness["mix_readiness"] == (
        "soundcheck_incomplete"
    )
    assert "console_state_changed_after_reconciliation" in (
        server.last_soundcheck_readiness["global_blockers"]
    )


@pytest.mark.asyncio
async def test_emergency_and_disconnect_immediately_downgrade_ready_run():
    for blocker in ("emergency_stop_active", "mixer_not_connected"):
        manifest = build_soundcheck_transaction_plans(
            [],
            run_id=f"runtime-global-{blocker}",
        ).to_dict()
        manifest["source_coverage"] = _ready_evidence()["source_coverage"]
        server = _server_for_readiness_io(manifest, {})
        ready = await server.reconcile_soundcheck_readiness(manifest["run_id"])
        assert ready["mix_readiness"] == "show_mix_ready"

        if blocker == "emergency_stop_active":
            server._manual_write_emergency_stopped = True
        else:
            server.mixer_client.is_connected = False
            fence = server._soundcheck_preflight_assessments[
                manifest["run_id"]
            ]["console_change_fence"]
            assert server._soundcheck_console_fence_still_current(fence) is False

        downgraded = server.refresh_soundcheck_readiness_truth()

        assert downgraded["mix_readiness"] == "soundcheck_incomplete"
        assert blocker in downgraded["global_blockers"]


@pytest.mark.asyncio
async def test_capture_stall_gets_specific_readiness_blocker():
    manifest = build_soundcheck_transaction_plans(
        [],
        run_id="capture-stall-readiness",
    ).to_dict()
    manifest["source_coverage"] = _ready_evidence()["source_coverage"]
    server = _server_for_readiness_io(manifest, {})
    await server.reconcile_soundcheck_readiness(manifest["run_id"])
    fence = server._soundcheck_preflight_assessments[
        manifest["run_id"]
    ]["console_change_fence"]
    fence["capture_identity"] = {
        "device_name": "Unit 48ch USB",
        "device_index": 7,
        "num_channels": 48,
        "sample_rate": 48000,
        "source_type": "sounddevice",
    }
    server.audio_capture = type(
        "StalledCapture",
        (),
        {
            "running": True,
            "get_status": lambda self: {
                "running": True,
                "real_input": True,
                "callback_age_sec": 2.0,
                "device_name": "Unit 48ch USB",
                "device_index": 7,
                "num_channels": 48,
                "sample_rate": 48000,
                "source_type": "sounddevice",
            },
        },
    )()

    downgraded = server.refresh_soundcheck_readiness_truth()

    assert downgraded["mix_readiness"] == "soundcheck_incomplete"
    assert "master_capture_changed_or_stale_after_reconciliation" in (
        downgraded["global_blockers"]
    )


def test_expected_current_child_mutation_does_not_self_invalidate_dependency_fence():
    manifest = build_soundcheck_transaction_plans(
        [ChannelFaderMove("balance", 1, -12.0)],
        run_id="expected-self-mutation",
    ).to_dict()
    server = _server_for_readiness_io(manifest, {"/ch/1/fdr": -10.0})
    fence = _current_test_console_fence(server.mixer_client)
    server._soundcheck_expected_mutation_addresses = {
        manifest["run_id"]: {
            "/ch/1/fdr": {"target_value": -12.0, "tolerance": 0.05}
        }
    }
    server._arm_soundcheck_console_change_invalidator(
        manifest["run_id"],
        fence,
    )

    server.mixer_client.external_update("/ch/1/fdr", -12.0)
    assert server._soundcheck_global_blockers[manifest["run_id"]] == []

    server._soundcheck_expected_mutation_addresses.clear()
    server.mixer_client.external_update("/ch/1/fdr", -9.0)
    assert "console_state_changed_after_reconciliation" in (
        server._soundcheck_global_blockers[manifest["run_id"]]
    )


def test_arming_new_run_retires_old_console_invalidator():
    old_manifest = build_soundcheck_transaction_plans(
        [],
        run_id="old-fence-run",
    ).to_dict()
    new_manifest = build_soundcheck_transaction_plans(
        [],
        run_id="new-fence-run",
    ).to_dict()
    server = _server_for_readiness_io(old_manifest, {"/ch/1/fdr": -10.0})
    server._soundcheck_manifests[new_manifest["run_id"]] = new_manifest
    server._soundcheck_outcomes[new_manifest["run_id"]] = {}
    server._soundcheck_global_blockers[new_manifest["run_id"]] = []
    server._soundcheck_preflight_assessments[new_manifest["run_id"]] = {}
    fence = _current_test_console_fence(server.mixer_client)

    server._active_soundcheck_run_id = old_manifest["run_id"]
    server.last_soundcheck_readiness = {"run_id": old_manifest["run_id"]}
    assert server._arm_soundcheck_console_change_invalidator(
        old_manifest["run_id"],
        fence,
    )
    server._active_soundcheck_run_id = new_manifest["run_id"]
    server.last_soundcheck_readiness = {"run_id": new_manifest["run_id"]}
    assert server._arm_soundcheck_console_change_invalidator(
        new_manifest["run_id"],
        fence,
    )

    assert len(server.mixer_client.callbacks["*"]) == 1
    server.mixer_client.external_update("/ch/1/fdr", -11.0)
    assert server._soundcheck_global_blockers[old_manifest["run_id"]] == []
    assert "console_state_changed_after_reconciliation" in (
        server._soundcheck_global_blockers[new_manifest["run_id"]]
    )


@pytest.mark.asyncio
async def test_superseded_run_finalize_and_delayed_replan_cannot_replace_active_truth():
    old_manifest = build_soundcheck_transaction_plans(
        [],
        run_id="old-active-run",
    ).to_dict()
    new_manifest = build_soundcheck_transaction_plans(
        [],
        run_id="new-active-run",
    ).to_dict()
    server = _server_for_readiness_io(old_manifest, {})
    server._soundcheck_manifests[new_manifest["run_id"]] = new_manifest
    server._soundcheck_outcomes[new_manifest["run_id"]] = {}
    server._soundcheck_global_blockers[new_manifest["run_id"]] = []
    server._soundcheck_preflight_assessments[new_manifest["run_id"]] = {}
    server._active_soundcheck_run_id = new_manifest["run_id"]
    server.last_soundcheck_readiness = {
        "run_id": new_manifest["run_id"],
        "mix_readiness": "soundcheck_incomplete",
    }
    server._soundcheck_replan_drafts = {
        old_manifest["run_id"]: {
            "draft_id": "stale-old-draft",
            "from_plan_revision": 1,
            "to_plan_revision": 2,
            "revised_manifest": {
                **old_manifest,
                "plan_revision": 2,
                "plan_fingerprint": "stale-old-revision",
            },
        }
    }

    finalized = await server.finalize_soundcheck_readiness_if_no_apply(
        old_manifest["run_id"]
    )
    accepted = await server.accept_soundcheck_replan_draft(
        run_id=old_manifest["run_id"],
        draft_id="stale-old-draft",
        approved=True,
        approval_id="late-operator-click",
    )

    assert finalized["mix_readiness"] == "soundcheck_incomplete"
    assert "soundcheck_run_superseded" in finalized["global_blockers"]
    assert accepted["reason"] == "soundcheck_run_superseded"
    assert server._active_soundcheck_run_id == new_manifest["run_id"]
    assert server.last_soundcheck_readiness["run_id"] == new_manifest["run_id"]


def test_known_inactive_manifest_cannot_be_reimported_or_revised():
    old_manifest = build_soundcheck_transaction_plans(
        [],
        run_id="known-old-run",
    ).to_dict()
    new_manifest = build_soundcheck_transaction_plans(
        [],
        run_id="known-new-run",
    ).to_dict()
    server = object.__new__(AutoMixerServer)
    server._soundcheck_manifests = {
        old_manifest["run_id"]: old_manifest,
        new_manifest["run_id"]: new_manifest,
    }
    server._active_soundcheck_run_id = new_manifest["run_id"]
    server._soundcheck_parent_journal_error = None
    server.last_soundcheck_readiness = {"run_id": new_manifest["run_id"]}
    server.get_operator_mode_status = lambda: {}

    same = server.import_soundcheck_recommendations_to_operator_queue(
        {"supervised_transaction_plans": old_manifest},
        trusted_engine_bundle=True,
    )
    revised_old = {
        **old_manifest,
        "plan_revision": 2,
        "plan_fingerprint": "known-old-revision-2",
    }
    bind_soundcheck_manifest_content_sha256(revised_old)
    revised = server.import_soundcheck_recommendations_to_operator_queue(
        {"supervised_transaction_plans": revised_old},
        trusted_engine_bundle=True,
    )

    assert same["reason"] == "soundcheck_run_superseded"
    assert revised["reason"] == "soundcheck_run_superseded"
    assert server._active_soundcheck_run_id == new_manifest["run_id"]


@pytest.mark.asyncio
async def test_reconcile_rechecks_active_run_after_await_before_publishing():
    old_manifest = build_soundcheck_transaction_plans(
        [],
        run_id="reconcile-old-run",
    ).to_dict()
    old_manifest["source_coverage"] = _ready_evidence()["source_coverage"]
    new_manifest = build_soundcheck_transaction_plans(
        [],
        run_id="reconcile-new-run",
    ).to_dict()
    server = _server_for_readiness_io(old_manifest, {})
    server._soundcheck_manifests[new_manifest["run_id"]] = new_manifest
    server._soundcheck_outcomes[new_manifest["run_id"]] = {}
    server._soundcheck_global_blockers[new_manifest["run_id"]] = []
    server._soundcheck_preflight_assessments[new_manifest["run_id"]] = {}
    server._active_soundcheck_run_id = old_manifest["run_id"]
    started = asyncio.Event()
    release = asyncio.Event()

    async def paused_audio_recheck(*_args):
        started.set()
        await release.wait()
        return AudioRecheckResult(
            "passed",
            "master_true_peak_safe",
            {
                "route_verified": True,
                "fresh": True,
                "signal_present": True,
                "headroom_safe": True,
            },
        )

    server._supervised_transaction_audio_recheck = paused_audio_recheck
    task = asyncio.create_task(
        server.reconcile_soundcheck_readiness(old_manifest["run_id"])
    )
    await started.wait()
    server._active_soundcheck_run_id = new_manifest["run_id"]
    server.last_soundcheck_readiness = {
        "run_id": new_manifest["run_id"],
        "mix_readiness": "soundcheck_incomplete",
    }
    release.set()

    result = await task

    assert result["mix_readiness"] == "soundcheck_incomplete"
    assert "soundcheck_run_superseded" in result["global_blockers"]
    assert server.last_soundcheck_readiness["run_id"] == new_manifest["run_id"]


@pytest.mark.asyncio
async def test_accept_replan_rechecks_active_run_after_preflight_await():
    old_manifest = build_soundcheck_transaction_plans(
        [],
        run_id="accept-race-old",
    ).to_dict()
    revised = {
        **old_manifest,
        "plan_revision": 2,
        "plan_fingerprint": "accept-race-old-revision-2",
    }
    new_manifest = build_soundcheck_transaction_plans(
        [],
        run_id="accept-race-new",
    ).to_dict()
    server = object.__new__(AutoMixerServer)
    server._soundcheck_manifests = {
        old_manifest["run_id"]: old_manifest,
        new_manifest["run_id"]: new_manifest,
    }
    server._active_soundcheck_run_id = old_manifest["run_id"]
    server.last_soundcheck_readiness = {"run_id": old_manifest["run_id"]}
    server._soundcheck_replan_drafts = {
        old_manifest["run_id"]: {
            "draft_id": "accept-race-draft",
            "from_plan_revision": 1,
            "to_plan_revision": 2,
            "revised_manifest": revised,
        }
    }
    imported = []

    async def preflight(_manifest):
        server._active_soundcheck_run_id = new_manifest["run_id"]
        server.last_soundcheck_readiness = {"run_id": new_manifest["run_id"]}
        return {"assessments": [], "global_blockers": []}

    server.preflight_soundcheck_manifest = preflight
    server.import_soundcheck_recommendations_to_operator_queue = (
        lambda *_args, **_kwargs: imported.append(True) or {}
    )

    result = await server.accept_soundcheck_replan_draft(
        run_id=old_manifest["run_id"],
        draft_id="accept-race-draft",
        approved=True,
        approval_id="race-approval",
    )

    assert result["reason"] == "soundcheck_run_superseded"
    assert imported == []
    assert server._active_soundcheck_run_id == new_manifest["run_id"]


@pytest.mark.asyncio
async def test_all_already_satisfied_plan_has_read_only_finalize_path():
    manifest = build_soundcheck_transaction_plans(
        [ChannelFaderMove("already balanced", 1, -12.0)],
        run_id="all-satisfied-finalize",
    ).to_dict()
    manifest["source_coverage"] = _ready_evidence()["source_coverage"]
    plan = manifest["plans"][0]
    server = _server_for_readiness_io(manifest, {"/ch/1/fdr": -12.0})

    refreshed = await server.refresh_soundcheck_child_preflight(plan)
    readiness = await server.finalize_soundcheck_readiness_if_no_apply(
        manifest["run_id"]
    )

    assert refreshed["assessments"][0]["disposition"] == "already_satisfied"
    assert readiness["mix_readiness"] == "show_mix_ready"
    assert server.mixer_client.sent == []


def test_parent_manifest_and_child_outcome_restore_durably_but_readiness_reopens(
    tmp_path,
):
    manifest = build_soundcheck_transaction_plans(
        [ChannelFaderMove("balance", 1, -12.0)],
        run_id="durable-parent",
    ).to_dict()
    operation_id = manifest["plans"][0]["operations"][0]["operation_id"]
    journal_path = tmp_path / "soundcheck_parent_journal.jsonl"
    writer = object.__new__(AutoMixerServer)
    writer._console_runtime_dir = tmp_path
    writer._soundcheck_parent_journal_path = journal_path
    assert writer._append_soundcheck_parent_journal(
        "manifest_registered",
        manifest["run_id"],
        plan_revision=1,
        plan_fingerprint=manifest["plan_fingerprint"],
        manifest=manifest,
    )
    assert writer._append_soundcheck_parent_journal(
        "child_outcome",
        manifest["run_id"],
        plan_revision=1,
        plan_fingerprint=manifest["plan_fingerprint"],
        manifest_content_sha256=manifest["manifest_content_sha256"],
        transaction_id=manifest["plans"][0]["transaction_id"],
        operation_id=operation_id,
        outcome=_committed_outcome(),
    )
    # A byte-for-byte duplicate registration is idempotent and must not erase
    # an already journaled child outcome during recovery.
    assert writer._append_soundcheck_parent_journal(
        "manifest_registered",
        manifest["run_id"],
        plan_revision=1,
        plan_fingerprint=manifest["plan_fingerprint"],
        manifest=manifest,
    )
    # Child rows are bound to the exact content digest, not only run/revision.
    assert writer._append_soundcheck_parent_journal(
        "child_outcome",
        manifest["run_id"],
        plan_revision=1,
        plan_fingerprint=manifest["plan_fingerprint"],
        manifest_content_sha256="wrong-content-digest",
        transaction_id=manifest["plans"][0]["transaction_id"],
        operation_id=operation_id,
        outcome={"status": "rolled_back"},
    )
    assert writer._append_soundcheck_parent_journal(
        "child_outcome",
        manifest["run_id"],
        plan_revision=1,
        plan_fingerprint=manifest["plan_fingerprint"],
        manifest_content_sha256=manifest["manifest_content_sha256"],
        transaction_id="foreign-transaction",
        operation_id=operation_id,
        outcome={"status": "manual_review_required"},
    )

    restored = object.__new__(AutoMixerServer)
    restored._soundcheck_parent_journal_path = journal_path
    restored._soundcheck_parent_journal_error = None
    restored._soundcheck_manifests = {}
    restored._soundcheck_outcomes = {}
    restored._soundcheck_global_blockers = {}
    restored._soundcheck_preflight_assessments = {}
    restored.last_soundcheck_readiness = {}
    restored._recover_soundcheck_parent_journal_state()

    assert restored._soundcheck_manifests[manifest["run_id"]][
        "plan_fingerprint"
    ] == manifest["plan_fingerprint"]
    outcome = restored._soundcheck_outcomes[manifest["run_id"]][operation_id]
    assert outcome["status"] == "applied"
    assert outcome["restored_from_parent_journal"] is True
    assert outcome["final_reconciliation_verified"] is False
    assert restored._soundcheck_global_blockers[manifest["run_id"]] == [
        "soundcheck_restart_requires_fresh_preflight_and_reimport"
    ]
    assert restored.last_soundcheck_readiness["run_id"] == manifest["run_id"]
    assert restored.last_soundcheck_readiness["mix_readiness"] == (
        "soundcheck_incomplete"
    )
    assert restored.refresh_soundcheck_readiness_truth()["global_blockers"] == [
        "mixer_not_connected",
        "soundcheck_restart_requires_fresh_preflight_and_reimport",
    ]


def test_parent_journal_rejects_same_revision_content_collision(tmp_path):
    first = build_soundcheck_transaction_plans(
        [ChannelFaderMove("balance", 1, -12.0)],
        run_id="durable-parent-collision",
    ).to_dict()
    collision = copy.deepcopy(first)
    collision["source_coverage"] = {"unexpected": "different-content"}
    bind_soundcheck_manifest_content_sha256(collision)
    journal_path = tmp_path / "soundcheck_parent_journal.jsonl"
    writer = object.__new__(AutoMixerServer)
    writer._console_runtime_dir = tmp_path
    writer._soundcheck_parent_journal_path = journal_path
    for manifest in (first, collision):
        assert writer._append_soundcheck_parent_journal(
            "manifest_registered",
            manifest["run_id"],
            plan_revision=manifest["plan_revision"],
            plan_fingerprint=manifest["plan_fingerprint"],
            manifest=manifest,
        )

    restored = object.__new__(AutoMixerServer)
    restored._soundcheck_parent_journal_path = journal_path
    restored._soundcheck_parent_journal_error = None
    restored._soundcheck_manifests = {}
    restored._soundcheck_outcomes = {}
    restored._soundcheck_global_blockers = {}
    restored._soundcheck_preflight_assessments = {}
    restored.last_soundcheck_readiness = {}

    restored._recover_soundcheck_parent_journal_state()

    assert "soundcheck_parent_manifest_revision_collision" in str(
        restored._soundcheck_parent_journal_error
    )


@pytest.mark.asyncio
async def test_real_replan_accept_contract_keeps_soundcheck_result_event_type():
    server = object.__new__(AutoMixerServer)
    current = {
        "run_id": "accept-replan",
        "plan_revision": 1,
        "plan_fingerprint": "old",
        "plans": [],
    }
    revised = {
        **current,
        "plan_revision": 2,
        "plan_fingerprint": "new",
    }
    server._soundcheck_manifests = {"accept-replan": current}
    server._soundcheck_replan_drafts = {
        "accept-replan": {
            "draft_id": "draft-2",
            "from_plan_revision": 1,
            "to_plan_revision": 2,
            "revised_manifest": revised,
        }
    }

    async def preflight(_manifest):
        return {"assessments": [], "global_blockers": []}

    server.preflight_soundcheck_manifest = preflight
    server.import_soundcheck_recommendations_to_operator_queue = (
        lambda *_args, **_kwargs: {
            "type": "operator_proposals_imported",
            "status": "ok",
            "success": True,
        }
    )

    result = await server.accept_soundcheck_replan_draft(
        run_id="accept-replan",
        draft_id="draft-2",
        approved=True,
        approval_id="operator-replan",
    )

    assert result["type"] == "soundcheck_replan_result"
    assert result["status"] == "ok"
    assert result["to_plan_revision"] == 2
