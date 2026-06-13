from pathlib import Path

from backend.wing_readonly_preflight_contract import (
    WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND,
    WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS,
    WING_READONLY_PREFLIGHT_EVENTS_ARTIFACT_NAME,
    WING_READONLY_PREFLIGHT_PROOF_SUMMARY_ARTIFACT,
)
from backend.supervised_wing_proof_plan import build_supervised_wing_proof_plan

PRESTART_BACKEND_PORT_CHECK_COMMAND = list(
    WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND
)
PRESTART_BACKEND_PORT_CHECK_FAILURE_DETAIL_FIELDS = list(
    WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS
)
GAIN_REQUIRED_EVIDENCE_COMPONENTS = [
    "physical_pre_read",
    "bounded_gain_down_apply",
    "apply_readback",
    "rollback_readback",
    "emergency_stop_block",
    "unsafe_gain_up_followup_blocked",
    "gain_up_operator_authored_opt_in",
]


def assert_prestart_check(summary: dict) -> None:
    assert summary["readonly_preflight_prestart_check_required"] is True
    assert (
        summary["readonly_preflight_prestart_check_must_pass_before_starting_backend"]
        is True
    )
    assert (
        summary["readonly_preflight_prestart_check_command"]
        == PRESTART_BACKEND_PORT_CHECK_COMMAND
    )
    assert (
        summary["readonly_preflight_prestart_check_failure_detail_fields"]
        == PRESTART_BACKEND_PORT_CHECK_FAILURE_DETAIL_FIELDS
    )


def evidence_snapshot(*rows):
    return {
        "type": "supervised_promotion_evidence",
        "safety": {
            "live_widening_allowed": False,
            "automatic_safety_update_allowed": False,
        },
        "rows": list(rows),
    }


def evidence_row(value_type: str, *, review_candidate: bool = False, blockers=None):
    return {
        "id": f"{value_type}_supervised_evidence",
        "value_type": value_type,
        "safety_row_review_candidate": review_candidate,
        "blockers": list(blockers or []),
    }


def accepted_review_decision_ledger() -> dict:
    return {
        "type": "wing_readonly_preflight_review_decision_ledger",
        "status": "operator_review_decision_records_available",
        "safety": {
            "read_only": True,
            "live_write": False,
            "execution_allowed_by_decision": False,
            "readiness_matrix_update_allowed_by_decision": False,
            "readiness_closure_allowed_by_decision": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
        },
        "summary": {
            "ledger_status": "operator_review_decision_records_available",
            "accepted_record_count": 1,
            "rejected_record_count": 0,
            "accepted_source_ids": [
                "artifacts/wing_readonly_preflight/run/summary.json"
            ],
            "candidate_reviews": [
                {
                    "source_id": "artifacts/wing_readonly_preflight/run/summary.json",
                    "status": "valid_readonly_preflight_artifact",
                    "backend_port_prestart_check_required": True,
                    "backend_port_prestart_check_must_pass_before_starting_backend": True,
                    "backend_port_prestart_check_command": list(
                        PRESTART_BACKEND_PORT_CHECK_COMMAND
                    ),
                    "backend_port_prestart_check_failure_detail_fields": list(
                        PRESTART_BACKEND_PORT_CHECK_FAILURE_DETAIL_FIELDS
                    ),
                    "backend_port_prestart_summary_required": True,
                    "backend_port_prestart_summary_path": (
                        "artifacts/no_wing_release_check/run/summary.json"
                    ),
                    "backend_port_prestart_summary_sha256": "3" * 64,
                    "backend_port_prestart_summary_type": "no_wing_release_check",
                    "backend_port_prestart_summary_status": "passed",
                    "backend_port_prestart_summary_connection_policy": (
                        "no_hardware_no_network_no_live_writes"
                    ),
                    "backend_port_prestart_summary_live_write": False,
                    "backend_port_prestart_check_status": "passed",
                }
            ],
            "rejected_source_ids": [],
            "invalid_record_count": 0,
            "unreviewed_candidate_count": 0,
            "manual_readiness_update_candidate_available": True,
            "readiness_matrix_update_allowed_by_decision": False,
            "readiness_closure_allowed_by_decision": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "blocked_by": [
                "readiness_matrix_update_still_requires_separate_manual_action"
            ],
        },
    }


def preflight_ready_snapshot() -> dict:
    return {
        "type": "autonomy_promotion_readiness",
        "summary": {
            "wing_readonly_preflight_evidence": {
                "readiness_update_candidate_available": True,
                "valid_candidate_count": 1,
                "invalid_candidate_count": 0,
                "readiness_closure_allowed_by_evidence": False,
                "live_authority_granted": False,
                "full_autonomous_ready": False,
                "evidence_status": "readonly_preflight_candidate_available",
                "blocked_by": [
                    "operator_must_review_readonly_preflight_candidate_before_readiness_update"
                ],
                "prestart_backend_port_check_required": True,
                "prestart_backend_port_check_must_pass_before_starting_backend": True,
                "prestart_backend_port_check_command": list(
                    PRESTART_BACKEND_PORT_CHECK_COMMAND
                ),
                "prestart_backend_port_check_failure_detail_fields": list(
                    PRESTART_BACKEND_PORT_CHECK_FAILURE_DETAIL_FIELDS
                ),
                "next_required_actions": [
                    "Manually review the read-only preflight candidate before updating readiness rows."
                ],
            },
            "wing_readonly_preflight_review_decision_ledger": accepted_review_decision_ledger(),
        },
    }


def preflight_candidate_without_review_decision_snapshot() -> dict:
    readiness = preflight_ready_snapshot()
    readiness["summary"]["wing_readonly_preflight_review_decision_ledger"] = {
        "type": "wing_readonly_preflight_review_decision_ledger",
        "status": "operator_review_required",
        "summary": {
            "ledger_status": "operator_review_required",
            "accepted_record_count": 0,
            "rejected_record_count": 0,
            "invalid_record_count": 0,
            "unreviewed_candidate_count": 1,
            "manual_readiness_update_candidate_available": False,
            "readiness_matrix_update_allowed_by_decision": False,
            "readiness_closure_allowed_by_decision": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "blocked_by": [
                "operator_must_record_readonly_preflight_review_decision"
            ],
        },
    }
    return readiness


def preflight_candidate_with_conflicting_review_decisions_snapshot() -> dict:
    readiness = preflight_ready_snapshot()
    readiness["summary"]["wing_readonly_preflight_review_decision_ledger"] = {
        "type": "wing_readonly_preflight_review_decision_ledger",
        "status": "operator_review_decision_replacement_required",
        "summary": {
            "ledger_status": "operator_review_decision_replacement_required",
            "accepted_record_count": 0,
            "rejected_record_count": 0,
            "invalid_record_count": 2,
            "invalid_record_ids": ["review-accept", "review-reject"],
            "duplicate_source_ids": [
                "artifacts/wing_readonly_preflight/run/summary.json"
            ],
            "conflicting_source_ids": [
                "artifacts/wing_readonly_preflight/run/summary.json"
            ],
            "unreviewed_candidate_count": 0,
            "manual_readiness_update_candidate_available": False,
            "readiness_matrix_update_allowed_by_decision": False,
            "readiness_closure_allowed_by_decision": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "blocked_by": [
                "operator_must_replace_invalid_readonly_preflight_review_decisions"
            ],
        },
    }
    return readiness


def test_supervised_wing_proof_plan_module_is_transport_free():
    source = Path("backend/supervised_wing_proof_plan.py").read_text(encoding="utf-8")

    forbidden = [
        "WingClient",
        "EnhancedOSCClient",
        "websockets",
        "socket",
        ".send(",
        "mixer_client",
        "set_fader",
        "set_gain",
        "apply_operator_proposal",
        "requested_change",
        "requested_changes",
    ]
    for token in forbidden:
        assert token not in source


def test_plan_builds_gain_and_fader_physical_proof_steps_without_execution_authority():
    plan = build_supervised_wing_proof_plan(
        evidence_snapshot(
            evidence_row("gain", blockers=["physical_candidate_missing"]),
            evidence_row("fader", blockers=["physical_candidate_missing", "blocked_artifacts_for_value_type"]),
        ),
        autonomy_readiness_snapshot=preflight_ready_snapshot(),
        backend_url="ws://proof-backend",
        artifact_root="artifacts/proof",
        channels_by_value_type={"gain": (3,), "fader": (2, 16)},
    )

    assert plan["type"] == "supervised_wing_proof_plan"
    assert plan["status"] == "operator_proof_required"
    assert plan["safety"] == {
        "read_only": True,
        "live_write": False,
        "execution_allowed_by_plan": False,
        "live_widening_allowed": False,
        "automatic_safety_update_allowed": False,
        "full_autonomous_ready": False,
        "operator_required": True,
    }
    assert plan["summary"]["step_count"] == 3
    assert plan["summary"]["planned_value_types"] == ["fader", "gain"]
    assert plan["summary"]["planned_channels"] == [3, 2, 16]
    assert plan["summary"]["readonly_preflight_required"] is True
    assert plan["summary"]["readonly_preflight_candidate_available"] is True
    assert plan["summary"]["readonly_preflight_evidence_status"] == "readonly_preflight_candidate_available"
    assert_prestart_check(plan["summary"])
    assert plan["summary"]["readonly_preflight_step_count"] == 0
    assert plan["summary"]["readonly_preflight_review_decision_required"] is True
    assert plan["summary"]["readonly_preflight_review_decision_accepted"] is True
    assert (
        plan["summary"]["readonly_preflight_review_decision_status"]
        == "operator_review_decision_records_available"
    )
    assert plan["summary"]["readonly_preflight_review_decision_accepted_record_count"] == 1
    assert plan["summary"]["readonly_preflight_review_decision_accepted_source_ids"] == [
        "artifacts/wing_readonly_preflight/run/summary.json"
    ]
    assert (
        plan["summary"][
            "readonly_preflight_review_decision_candidate_prestart_contract_valid"
        ]
        is True
    )
    assert plan["summary"][
        "readonly_preflight_review_decision_candidate_prestart_source_ids"
    ] == ["artifacts/wing_readonly_preflight/run/summary.json"]
    assert plan["summary"]["readonly_preflight_review_decision_rejected_source_ids"] == []
    assert plan["readonly_preflight_steps"] == []

    gain_step = plan["steps"][0]
    assert gain_step["id"] == "gain_physical_proof_ch3"
    assert gain_step["live_execution_required"] is True
    assert gain_step["execution_allowed_by_plan"] is False
    assert gain_step["operator_approval_required"] is True
    assert gain_step["delta_db"] == -0.25
    assert gain_step["command"][0] == "python3"
    assert "--execute" in gain_step["command"]
    assert "--operator-confirm-channel" in gain_step["command"]
    assert gain_step["command"][gain_step["command"].index("--operator-confirm-channel") + 1] == "3"
    assert "--operator-confirm-value-type" in gain_step["command"]
    assert (
        gain_step["command"][gain_step["command"].index("--operator-confirm-value-type") + 1]
        == "gain"
    )
    assert "--operator-confirm-delta-db" in gain_step["command"]
    assert (
        gain_step["command"][gain_step["command"].index("--operator-confirm-delta-db") + 1]
        == "-0.25"
    )
    assert "--prove-blocked-apply" in gain_step["command"]
    assert "--prove-emergency-stop" in gain_step["command"]
    assert gain_step["required_evidence_components"] == GAIN_REQUIRED_EVIDENCE_COMPONENTS
    assert "ws://proof-backend" in gain_step["command"]
    assert (
        "A clean read-only WING preflight candidate exists and has been manually reviewed."
        in gain_step["prerequisites"]
    )
    assert "Use gain-down proof only; gain-up remains out of scope for this plan." in gain_step["prerequisites"]

    fader_step = plan["steps"][1]
    assert "Review blocked prior proof artifacts for this value type before trusting a new result." in fader_step["prerequisites"]
    assert "blocked_artifacts_for_value_type" in fader_step["evidence_blockers_to_review"]


def test_plan_blocks_physical_proof_steps_until_review_decision_is_accepted():
    plan = build_supervised_wing_proof_plan(
        evidence_snapshot(
            evidence_row("gain", blockers=["physical_candidate_missing"]),
            evidence_row("fader", blockers=["physical_candidate_missing"]),
        ),
        autonomy_readiness_snapshot=preflight_candidate_without_review_decision_snapshot(),
    )

    assert plan["status"] == "manual_review_ready"
    assert plan["summary"]["step_count"] == 0
    assert plan["summary"]["readonly_preflight_candidate_available"] is True
    assert_prestart_check(plan["summary"])
    assert plan["summary"]["readonly_preflight_step_count"] == 0
    assert plan["summary"]["readonly_preflight_review_decision_required"] is True
    assert plan["summary"]["readonly_preflight_review_decision_accepted"] is False
    assert (
        plan["summary"]["readonly_preflight_review_decision_status"]
        == "operator_review_required"
    )
    assert (
        "accepted_readonly_preflight_review_decision_required_before_supervised_proof"
        in plan["summary"]["global_blockers"]
    )
    assert (
        "operator_must_record_readonly_preflight_review_decision"
        in plan["summary"]["global_blockers"]
    )
    assert plan["readonly_preflight_steps"] == []
    assert plan["steps"] == []


def test_plan_blocks_physical_proof_steps_when_accepted_candidate_lacks_prestart_contract():
    readiness = preflight_ready_snapshot()
    ledger = accepted_review_decision_ledger()
    ledger["summary"]["candidate_reviews"] = [
        {
            "source_id": "artifacts/wing_readonly_preflight/run/summary.json",
            "status": "valid_readonly_preflight_artifact",
            "backend_port_prestart_check_required": False,
            "backend_port_prestart_check_must_pass_before_starting_backend": True,
            "backend_port_prestart_check_command": list(
                PRESTART_BACKEND_PORT_CHECK_COMMAND
            ),
            "backend_port_prestart_check_failure_detail_fields": list(
                PRESTART_BACKEND_PORT_CHECK_FAILURE_DETAIL_FIELDS
            ),
            "backend_port_prestart_summary_required": True,
            "backend_port_prestart_summary_path": (
                "artifacts/no_wing_release_check/run/summary.json"
            ),
            "backend_port_prestart_summary_sha256": "3" * 64,
            "backend_port_prestart_summary_type": "no_wing_release_check",
            "backend_port_prestart_summary_status": "passed",
            "backend_port_prestart_summary_connection_policy": (
                "no_hardware_no_network_no_live_writes"
            ),
            "backend_port_prestart_summary_live_write": False,
            "backend_port_prestart_check_status": "passed",
        }
    ]
    readiness["summary"]["wing_readonly_preflight_review_decision_ledger"] = ledger

    plan = build_supervised_wing_proof_plan(
        evidence_snapshot(evidence_row("gain", blockers=["physical_candidate_missing"])),
        autonomy_readiness_snapshot=readiness,
    )

    assert plan["status"] == "manual_review_ready"
    assert plan["summary"]["step_count"] == 0
    assert plan["summary"]["readonly_preflight_review_decision_accepted"] is False
    assert (
        plan["summary"][
            "readonly_preflight_review_decision_candidate_prestart_contract_valid"
        ]
        is False
    )
    assert (
        plan["summary"]["readonly_preflight_review_decision_candidate_prestart_source_ids"]
        == []
    )
    assert (
        "accepted_readonly_preflight_candidate_prestart_contract_required_before_supervised_proof"
        in plan["summary"]["global_blockers"]
    )


def test_plan_blocks_physical_proof_steps_when_accepted_candidate_lacks_prestart_summary_binding():
    readiness = preflight_ready_snapshot()
    ledger = accepted_review_decision_ledger()
    ledger["summary"]["candidate_reviews"][0]["backend_port_prestart_summary_path"] = ""
    readiness["summary"]["wing_readonly_preflight_review_decision_ledger"] = ledger

    plan = build_supervised_wing_proof_plan(
        evidence_snapshot(evidence_row("gain", blockers=["physical_candidate_missing"])),
        autonomy_readiness_snapshot=readiness,
    )

    assert plan["status"] == "manual_review_ready"
    assert plan["summary"]["step_count"] == 0
    assert (
        plan["summary"][
            "readonly_preflight_review_decision_candidate_prestart_contract_valid"
        ]
        is False
    )
    assert plan["summary"][
        "readonly_preflight_review_decision_candidate_prestart_source_ids"
    ] == []
    assert (
        "accepted_readonly_preflight_candidate_prestart_contract_required_before_supervised_proof"
        in plan["summary"]["global_blockers"]
    )
    assert plan["steps"] == []


def test_plan_blocks_physical_proof_steps_for_conflicting_review_decisions():
    plan = build_supervised_wing_proof_plan(
        evidence_snapshot(
            evidence_row("gain", blockers=["physical_candidate_missing"]),
            evidence_row("fader", blockers=["physical_candidate_missing"]),
        ),
        autonomy_readiness_snapshot=preflight_candidate_with_conflicting_review_decisions_snapshot(),
    )

    assert plan["status"] == "manual_review_ready"
    assert plan["summary"]["step_count"] == 0
    assert plan["summary"]["readonly_preflight_candidate_available"] is True
    assert_prestart_check(plan["summary"])
    assert plan["summary"]["readonly_preflight_review_decision_accepted"] is False
    assert (
        plan["summary"]["readonly_preflight_review_decision_status"]
        == "operator_review_decision_replacement_required"
    )
    assert plan["summary"]["readonly_preflight_review_decision_invalid_record_count"] == 2
    assert plan["summary"]["readonly_preflight_review_decision_invalid_record_ids"] == [
        "review-accept",
        "review-reject",
    ]
    assert plan["summary"]["readonly_preflight_review_decision_duplicate_source_ids"] == [
        "artifacts/wing_readonly_preflight/run/summary.json"
    ]
    assert plan["summary"]["readonly_preflight_review_decision_conflicting_source_ids"] == [
        "artifacts/wing_readonly_preflight/run/summary.json"
    ]
    assert (
        "accepted_readonly_preflight_review_decision_required_before_supervised_proof"
        in plan["summary"]["global_blockers"]
    )
    assert (
        "operator_must_replace_invalid_readonly_preflight_review_decisions"
        in plan["summary"]["global_blockers"]
    )
    assert plan["readonly_preflight_steps"] == []
    assert plan["steps"] == []


def test_plan_skips_value_type_that_already_has_manual_review_candidate():
    plan = build_supervised_wing_proof_plan(
        evidence_snapshot(
            evidence_row("gain", review_candidate=True),
            evidence_row("fader", blockers=["physical_candidate_missing"]),
        ),
        autonomy_readiness_snapshot=preflight_ready_snapshot(),
        channels_by_value_type={"fader": (4,)},
    )

    assert plan["status"] == "operator_proof_required"
    assert [step["value_type"] for step in plan["steps"]] == ["fader"]
    assert plan["steps"][0]["channel"] == 4


def test_plan_is_manual_review_ready_when_all_value_types_have_candidates():
    plan = build_supervised_wing_proof_plan(
        evidence_snapshot(
            evidence_row("gain", review_candidate=True),
            evidence_row("fader", review_candidate=True),
        ),
        autonomy_readiness_snapshot=preflight_ready_snapshot(),
    )

    assert plan["status"] == "manual_review_ready"
    assert plan["summary"]["step_count"] == 0
    assert plan["steps"] == []
    assert plan["safety"]["execution_allowed_by_plan"] is False


def test_plan_blocks_physical_proof_steps_until_readonly_preflight_candidate_exists():
    plan = build_supervised_wing_proof_plan(
        evidence_snapshot(
            evidence_row("gain", blockers=["physical_candidate_missing"]),
            evidence_row("fader", blockers=["physical_candidate_missing"]),
        ),
        autonomy_readiness_snapshot={
            "type": "autonomy_promotion_readiness",
            "summary": {
                "wing_readonly_preflight_evidence": {
                    "readiness_update_candidate_available": False,
                    "valid_candidate_count": 0,
                    "invalid_candidate_count": 0,
                    "evidence_status": "operator_preflight_required",
                    "blocked_by": ["operator_must_run_wing_readonly_preflight_runner"],
                    "prestart_backend_port_check_required": True,
                    "prestart_backend_port_check_must_pass_before_starting_backend": True,
                    "prestart_backend_port_check_command": list(
                        PRESTART_BACKEND_PORT_CHECK_COMMAND
                    ),
                    "prestart_backend_port_check_failure_detail_fields": list(
                        PRESTART_BACKEND_PORT_CHECK_FAILURE_DETAIL_FIELDS
                    ),
                    "next_required_actions": [
                        "Before starting the intended backend/WING session, run the no-WING quick gate and resolve only confirmed unexpected listeners.",
                        "Run the read-only WING preflight runner and capture physical readback artifacts."
                    ],
                },
            },
        },
    )

    assert plan["status"] == "blocked_by_readonly_preflight"
    assert plan["summary"]["step_count"] == 0
    assert plan["summary"]["planned_value_types"] == []
    assert plan["summary"]["readonly_preflight_required"] is True
    assert plan["summary"]["readonly_preflight_candidate_available"] is False
    assert plan["summary"]["readonly_preflight_evidence_status"] == "operator_preflight_required"
    assert_prestart_check(plan["summary"])
    assert plan["summary"]["readonly_preflight_step_count"] == 1
    assert "readonly_preflight_candidate_required_before_supervised_proof" in plan["summary"]["global_blockers"]
    assert "operator_must_run_wing_readonly_preflight_runner" in plan["summary"]["global_blockers"]
    preflight_step = plan["readonly_preflight_steps"][0]
    assert preflight_step["id"] == "wing_readonly_preflight_required"
    assert preflight_step["read_only"] is True
    assert preflight_step["live_execution_required"] is False
    assert preflight_step["execution_allowed_by_plan"] is False
    assert preflight_step["operator_approval_required"] is True
    assert preflight_step["command"][:2] == [
        "python3",
        "tools/wing_readonly_preflight_runner.py",
    ]
    assert "--require-physical-readback" in preflight_step["command"]
    assert "--execute" not in preflight_step["command"]
    assert "tools/wing_operator_proposal_validation_runner.py" not in preflight_step["command"]
    assert (
        WING_READONLY_PREFLIGHT_PROOF_SUMMARY_ARTIFACT
        in preflight_step["expected_artifacts"]
    )
    assert (
        f"{WING_READONLY_PREFLIGHT_EVENTS_ARTIFACT_NAME} with "
        "get_operator_mode/get_state/query_mixer_parameter only"
    ) in preflight_step["expected_artifacts"]
    assert plan["steps"] == []


def test_plan_blocks_physical_proof_steps_for_mixed_preflight_artifacts():
    plan = build_supervised_wing_proof_plan(
        evidence_snapshot(
            evidence_row("gain", blockers=["physical_candidate_missing"]),
            evidence_row("fader", blockers=["physical_candidate_missing"]),
        ),
        autonomy_readiness_snapshot={
            "type": "autonomy_promotion_readiness",
            "summary": {
                "wing_readonly_preflight_evidence": {
                    "readiness_update_candidate_available": False,
                    "valid_candidate_count": 1,
                    "invalid_candidate_count": 1,
                    "evidence_status": "readonly_preflight_artifact_replacement_required",
                    "blocked_by": [
                        "operator_must_replace_invalid_readonly_preflight_artifact"
                    ],
                    "prestart_backend_port_check_required": True,
                    "prestart_backend_port_check_must_pass_before_starting_backend": True,
                    "prestart_backend_port_check_command": list(
                        PRESTART_BACKEND_PORT_CHECK_COMMAND
                    ),
                    "prestart_backend_port_check_failure_detail_fields": list(
                        PRESTART_BACKEND_PORT_CHECK_FAILURE_DETAIL_FIELDS
                    ),
                    "next_required_actions": [
                        "Do not update readiness rows from missing or invalid preflight evidence."
                    ],
                },
            },
        },
    )

    assert plan["status"] == "blocked_by_readonly_preflight"
    assert plan["summary"]["step_count"] == 0
    assert plan["summary"]["readonly_preflight_candidate_available"] is False
    assert (
        plan["summary"]["readonly_preflight_evidence_status"]
        == "readonly_preflight_artifact_replacement_required"
    )
    assert_prestart_check(plan["summary"])
    assert plan["summary"]["readonly_preflight_blocked_by"] == [
        "operator_must_replace_invalid_readonly_preflight_artifact"
    ]
    assert plan["summary"]["readonly_preflight_step_count"] == 1
    assert "readonly_preflight_candidate_required_before_supervised_proof" in plan["summary"]["global_blockers"]
    assert "operator_must_replace_invalid_readonly_preflight_artifact" in plan["summary"]["global_blockers"]
    assert plan["steps"] == []

    preflight_step = plan["readonly_preflight_steps"][0]
    assert preflight_step["read_only"] is True
    assert preflight_step["execution_allowed_by_plan"] is False
    assert "operator_must_replace_invalid_readonly_preflight_artifact" in preflight_step["abort_conditions"]
    assert "--require-physical-readback" in preflight_step["command"]
    assert "--execute" not in preflight_step["command"]


def test_invalid_artifacts_block_plan_generation():
    plan = build_supervised_wing_proof_plan(
        evidence_snapshot(
            evidence_row("gain", blockers=["invalid_proof_summaries_present"]),
            evidence_row("fader", blockers=["physical_candidate_missing"]),
        )
    )

    assert plan["status"] == "blocked_by_invalid_artifacts"
    assert plan["steps"] == []
    assert plan["summary"]["global_blockers"] == ["invalid_proof_summaries_present"]
