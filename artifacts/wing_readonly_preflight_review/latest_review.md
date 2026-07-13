{
  "artifact_count": 1,
  "artifact_operator_next_steps": [],
  "artifact_reviews": [
    {
      "artifact_manifest_verifier_must_pass_before_websocket": true,
      "artifact_manifest_verifier_passed_before_websocket": true,
      "artifact_manifest_verifier_websocket_opened_after_pass": true,
      "artifact_root": "artifacts/wing_readonly_preflight",
      "backend_port_prestart_check_command": [
        "PYTHONPATH=backend",
        "python3",
        "tools/no_wing_release_check.py",
        "--quick"
      ],
      "backend_port_prestart_check_command_text": "PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick",
      "backend_port_prestart_check_failure_detail_fields": [
        "listener_processes",
        "listener_process_error",
        "listener_confirmation_checklist",
        "preflight_blocked_until_listener_confirmed",
        "intended_backend_confirmation_required",
        "safe_to_stop_listener_by_gate",
        "listener_confirmation_template_status",
        "listener_confirmation_template_path",
        "listener_confirmation_template_acceptance_status",
        "listener_confirmation_template_safe_to_stop_listener_by_gate",
        "listener_confirmation_template_live_authority_granted",
        "listener_confirmation_template_accepted_by_template",
        "listener_confirmation_template_operator_placeholder_fields",
        "listener_confirmation_template_execution_allowed_by_template",
        "listener_confirmation_template_readiness_closure_allowed_by_template",
        "listener_confirmation_template_required_edits",
        "backend_port_prestart_gate_execution_allowed",
        "backend_port_prestart_gate_readiness_closure_allowed",
        "backend_port_prestart_gate_live_authority_granted",
        "fail_closed_reason",
        "operator_next_step"
      ],
      "backend_port_prestart_check_must_pass_before_starting_backend": true,
      "backend_port_prestart_check_required": true,
      "backend_port_prestart_check_status": "passed",
      "backend_port_prestart_gate_execution_allowed": false,
      "backend_port_prestart_gate_execution_allowed_matches_plan": true,
      "backend_port_prestart_gate_live_authority_granted": false,
      "backend_port_prestart_gate_live_authority_granted_matches_plan": true,
      "backend_port_prestart_gate_readiness_closure_allowed": false,
      "backend_port_prestart_gate_readiness_closure_allowed_matches_plan": true,
      "backend_port_prestart_summary_connection_policy": "no_hardware_no_network_no_live_writes",
      "backend_port_prestart_summary_contacts_backend": false,
      "backend_port_prestart_summary_contacts_wing": false,
      "backend_port_prestart_summary_executes_tests": false,
      "backend_port_prestart_summary_live_write": false,
      "backend_port_prestart_summary_mutates_console_state": false,
      "backend_port_prestart_summary_path": "artifacts/no_wing_release_check/20260712T082407Z_no_wing_release_check/backend_port_prestart/summary.json",
      "backend_port_prestart_summary_queries_console": false,
      "backend_port_prestart_summary_read_only": true,
      "backend_port_prestart_summary_required": true,
      "backend_port_prestart_summary_sha256": "50e0c8017392eb1ad5667b194a502355c8edb4768d490055ab49b95c86f7df82",
      "backend_port_prestart_summary_status": "passed",
      "backend_port_prestart_summary_type": "no_wing_release_check",
      "captured_at": "2026-07-12T08:29:03.069Z",
      "channels": [
        3
      ],
      "contacts_backend": true,
      "event_log_run_id_matches_summary": true,
      "events_path": "artifacts/wing_readonly_preflight/20260712T082903Z_wing_readonly_preflight/events.jsonl",
      "events_path_matches_expected_artifact": true,
      "events_sha256": "f56502b3b8f5e90d8b3f00b2ef0ee62595c3d5d98c2cf75a623416e50365ca61",
      "expected_artifacts": [
        "artifacts/wing_readonly_preflight/<run_id>/summary.json",
        "artifacts/wing_readonly_preflight/<run_id>/events.jsonl"
      ],
      "expected_events_artifact": "artifacts/wing_readonly_preflight/20260712T082903Z_wing_readonly_preflight/events.jsonl",
      "expected_summary_artifact": "artifacts/wing_readonly_preflight/20260712T082903Z_wing_readonly_preflight/summary.json",
      "forbidden_request_types_sent": [],
      "live_authority_granted": false,
      "live_authority_granted_matches_plan": true,
      "operator_confirmed_channels": [
        3
      ],
      "operator_confirmed_session_ready": true,
      "operator_next_step": null,
      "operator_session_confirmation_required": true,
      "physical_readback_count": 3,
      "plan_backend_port_prestart_gate_execution_allowed": false,
      "plan_backend_port_prestart_gate_live_authority_granted": false,
      "plan_backend_port_prestart_gate_readiness_closure_allowed": false,
      "plan_live_authority_granted": false,
      "plan_safe_to_stop_listener_by_gate": false,
      "proposal_request_sent": false,
      "queries_console": true,
      "readback_count": 3,
      "run_id": "20260712T082903Z_wing_readonly_preflight",
      "safe_to_stop_listener_by_gate": false,
      "safe_to_stop_listener_by_gate_matches_plan": true,
      "sent_request_types": [
        "get_operator_mode",
        "get_state",
        "query_mixer_parameter",
        "query_mixer_parameter",
        "query_mixer_parameter"
      ],
      "source_id": "artifacts/wing_readonly_preflight/20260712T082903Z_wing_readonly_preflight/summary.json",
      "status": "valid_readonly_preflight_artifact",
      "summary_path": "artifacts/wing_readonly_preflight/20260712T082903Z_wing_readonly_preflight/summary.json",
      "summary_path_matches_expected_artifact": true,
      "summary_sha256": "e414be53f0fec16be9bd37fc8b065210f6d5770f53bb6700a3380eb5a73bf47b",
      "value_types": [
        "fader",
        "gain",
        "pan"
      ],
      "violations": [],
      "write_request_sent": false
    }
  ],
  "artifact_sources": [
    "artifacts/wing_readonly_preflight/20260712T082903Z_wing_readonly_preflight/summary.json"
  ],
  "blocked_by": [
    "operator_must_review_readonly_preflight_candidate_before_readiness_update"
  ],
  "evidence": {
    "artifacts": [
      {
        "accepted_as_readonly_preflight_candidate": true,
        "artifact_manifest_verifier_must_pass_before_websocket": true,
        "artifact_manifest_verifier_passed_before_websocket": true,
        "artifact_manifest_verifier_websocket_opened_after_pass": true,
        "artifact_root": "artifacts/wing_readonly_preflight",
        "backend_port_prestart_check_command": [
          "PYTHONPATH=backend",
          "python3",
          "tools/no_wing_release_check.py",
          "--quick"
        ],
        "backend_port_prestart_check_command_text": "PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick",
        "backend_port_prestart_check_failure_detail_fields": [
          "listener_processes",
          "listener_process_error",
          "listener_confirmation_checklist",
          "preflight_blocked_until_listener_confirmed",
          "intended_backend_confirmation_required",
          "safe_to_stop_listener_by_gate",
          "listener_confirmation_template_status",
          "listener_confirmation_template_path",
          "listener_confirmation_template_acceptance_status",
          "listener_confirmation_template_safe_to_stop_listener_by_gate",
          "listener_confirmation_template_live_authority_granted",
          "listener_confirmation_template_accepted_by_template",
          "listener_confirmation_template_operator_placeholder_fields",
          "listener_confirmation_template_execution_allowed_by_template",
          "listener_confirmation_template_readiness_closure_allowed_by_template",
          "listener_confirmation_template_required_edits",
          "backend_port_prestart_gate_execution_allowed",
          "backend_port_prestart_gate_readiness_closure_allowed",
          "backend_port_prestart_gate_live_authority_granted",
          "fail_closed_reason",
          "operator_next_step"
        ],
        "backend_port_prestart_check_must_pass_before_starting_backend": true,
        "backend_port_prestart_check_required": true,
        "backend_port_prestart_check_status": "passed",
        "backend_port_prestart_gate_execution_allowed": false,
        "backend_port_prestart_gate_execution_allowed_matches_plan": true,
        "backend_port_prestart_gate_live_authority_granted": false,
        "backend_port_prestart_gate_live_authority_granted_matches_plan": true,
        "backend_port_prestart_gate_readiness_closure_allowed": false,
        "backend_port_prestart_gate_readiness_closure_allowed_matches_plan": true,
        "backend_port_prestart_summary_connection_policy": "no_hardware_no_network_no_live_writes",
        "backend_port_prestart_summary_contacts_backend": false,
        "backend_port_prestart_summary_contacts_wing": false,
        "backend_port_prestart_summary_executes_tests": false,
        "backend_port_prestart_summary_live_write": false,
        "backend_port_prestart_summary_mutates_console_state": false,
        "backend_port_prestart_summary_path": "artifacts/no_wing_release_check/20260712T082407Z_no_wing_release_check/backend_port_prestart/summary.json",
        "backend_port_prestart_summary_queries_console": false,
        "backend_port_prestart_summary_read_only": true,
        "backend_port_prestart_summary_required": true,
        "backend_port_prestart_summary_sha256": "50e0c8017392eb1ad5667b194a502355c8edb4768d490055ab49b95c86f7df82",
        "backend_port_prestart_summary_status": "passed",
        "backend_port_prestart_summary_type": "no_wing_release_check",
        "captured_at": "2026-07-12T08:29:03.069Z",
        "channels": [
          3
        ],
        "connection_mode": "wing",
        "contacts_backend": true,
        "event_log_run_id_matches_summary": true,
        "events_path": "artifacts/wing_readonly_preflight/20260712T082903Z_wing_readonly_preflight/events.jsonl",
        "events_path_matches_expected_artifact": true,
        "events_sha256": "f56502b3b8f5e90d8b3f00b2ef0ee62595c3d5d98c2cf75a623416e50365ca61",
        "expected_artifacts": [
          "artifacts/wing_readonly_preflight/<run_id>/summary.json",
          "artifacts/wing_readonly_preflight/<run_id>/events.jsonl"
        ],
        "expected_events_artifact": "artifacts/wing_readonly_preflight/20260712T082903Z_wing_readonly_preflight/events.jsonl",
        "expected_summary_artifact": "artifacts/wing_readonly_preflight/20260712T082903Z_wing_readonly_preflight/summary.json",
        "forbidden_request_types_sent": [],
        "live_authority_granted": false,
        "live_authority_granted_matches_plan": true,
        "operator_confirmed_channels": [
          3
        ],
        "operator_confirmed_session_ready": true,
        "operator_next_step": null,
        "operator_session_confirmation_required": true,
        "physical_readback_count": 3,
        "plan_backend_port_prestart_gate_execution_allowed": false,
        "plan_backend_port_prestart_gate_live_authority_granted": false,
        "plan_backend_port_prestart_gate_readiness_closure_allowed": false,
        "plan_live_authority_granted": false,
        "plan_request_sequence": [
          {
            "type": "get_operator_mode"
          },
          {
            "type": "get_state"
          },
          {
            "address": "/ch/3/fdr",
            "channel": 3,
            "request_id": "wing-readonly-preflight:ch3:fader",
            "type": "query_mixer_parameter",
            "value_type": "fader"
          },
          {
            "address": "/ch/3/in/set/trim",
            "channel": 3,
            "request_id": "wing-readonly-preflight:ch3:gain",
            "type": "query_mixer_parameter",
            "value_type": "gain"
          },
          {
            "address": "/ch/3/pan",
            "channel": 3,
            "request_id": "wing-readonly-preflight:ch3:pan",
            "type": "query_mixer_parameter",
            "value_type": "pan"
          }
        ],
        "plan_safe_to_stop_listener_by_gate": false,
        "proposal_request_sent": false,
        "queries_console": true,
        "readback_count": 3,
        "request_sequence": [
          {
            "type": "get_operator_mode"
          },
          {
            "type": "get_state"
          },
          {
            "address": "/ch/3/fdr",
            "channel": 3,
            "request_id": "wing-readonly-preflight:ch3:fader",
            "type": "query_mixer_parameter",
            "value_type": "fader"
          },
          {
            "address": "/ch/3/in/set/trim",
            "channel": 3,
            "request_id": "wing-readonly-preflight:ch3:gain",
            "type": "query_mixer_parameter",
            "value_type": "gain"
          },
          {
            "address": "/ch/3/pan",
            "channel": 3,
            "request_id": "wing-readonly-preflight:ch3:pan",
            "type": "query_mixer_parameter",
            "value_type": "pan"
          }
        ],
        "request_sequence_matches_plan": true,
        "run_id": "20260712T082903Z_wing_readonly_preflight",
        "safe_to_stop_listener_by_gate": false,
        "safe_to_stop_listener_by_gate_matches_plan": true,
        "sent_request_types": [
          "get_operator_mode",
          "get_state",
          "query_mixer_parameter",
          "query_mixer_parameter",
          "query_mixer_parameter"
        ],
        "source_id": "artifacts/wing_readonly_preflight/20260712T082903Z_wing_readonly_preflight/summary.json",
        "status": "valid_readonly_preflight_artifact",
        "summary_path": "artifacts/wing_readonly_preflight/20260712T082903Z_wing_readonly_preflight/summary.json",
        "summary_path_matches_expected_artifact": true,
        "summary_sha256": "e414be53f0fec16be9bd37fc8b065210f6d5770f53bb6700a3380eb5a73bf47b",
        "value_types": [
          "fader",
          "gain",
          "pan"
        ],
        "violations": [],
        "write_request_sent": false
      }
    ],
    "generated_at": "2026-07-12T08:31:51Z",
    "method_id": "wing_readonly_preflight_evidence",
    "mvp_blocker_matrix": [
      {
        "automatic_live_authority_allowed": false,
        "closed": false,
        "id": "wing_readonly_connection_proof",
        "manual_review_required": true,
        "status": "candidate_available"
      },
      {
        "automatic_live_authority_allowed": false,
        "closed": false,
        "id": "physical_console_readback_proof",
        "manual_review_required": true,
        "status": "candidate_available"
      },
      {
        "automatic_live_authority_allowed": false,
        "closed": false,
        "id": "operator_visible_evidence_report",
        "manual_review_required": true,
        "status": "candidate_available"
      },
      {
        "automatic_live_authority_allowed": false,
        "closed": false,
        "id": "readiness_matrix_update_based_on_real_proof",
        "manual_review_required": true,
        "status": "candidate_available"
      }
    ],
    "safety": {
      "executes_tests": false,
      "execution_allowed_by_evidence": false,
      "full_autonomous_ready": false,
      "live_authority_granted": false,
      "live_widening_allowed": false,
      "live_write": false,
      "mutates_console_state": false,
      "operator_required": true,
      "queries_console": false,
      "read_only": true,
      "readiness_closure_allowed_by_evidence": false
    },
    "schema_id": "wing_readonly_preflight.v1",
    "status": "readonly_preflight_candidate_available",
    "summary": {
      "accepted_as_readonly_preflight_candidate": true,
      "artifact_count": 1,
      "artifact_search_roots": [
        "artifacts/wing_readonly_preflight"
      ],
      "blocked_by": [
        "operator_must_review_readonly_preflight_candidate_before_readiness_update"
      ],
      "full_autonomous_ready": false,
      "invalid_candidate_count": 0,
      "latest_valid_candidate_captured_at": "2026-07-12T08:29:03.069Z",
      "latest_valid_candidate_plan_request_sequence": [
        {
          "type": "get_operator_mode"
        },
        {
          "type": "get_state"
        },
        {
          "address": "/ch/3/fdr",
          "channel": 3,
          "request_id": "wing-readonly-preflight:ch3:fader",
          "type": "query_mixer_parameter",
          "value_type": "fader"
        },
        {
          "address": "/ch/3/in/set/trim",
          "channel": 3,
          "request_id": "wing-readonly-preflight:ch3:gain",
          "type": "query_mixer_parameter",
          "value_type": "gain"
        },
        {
          "address": "/ch/3/pan",
          "channel": 3,
          "request_id": "wing-readonly-preflight:ch3:pan",
          "type": "query_mixer_parameter",
          "value_type": "pan"
        }
      ],
      "latest_valid_candidate_request_sequence": [
        {
          "type": "get_operator_mode"
        },
        {
          "type": "get_state"
        },
        {
          "address": "/ch/3/fdr",
          "channel": 3,
          "request_id": "wing-readonly-preflight:ch3:fader",
          "type": "query_mixer_parameter",
          "value_type": "fader"
        },
        {
          "address": "/ch/3/in/set/trim",
          "channel": 3,
          "request_id": "wing-readonly-preflight:ch3:gain",
          "type": "query_mixer_parameter",
          "value_type": "gain"
        },
        {
          "address": "/ch/3/pan",
          "channel": 3,
          "request_id": "wing-readonly-preflight:ch3:pan",
          "type": "query_mixer_parameter",
          "value_type": "pan"
        }
      ],
      "latest_valid_candidate_request_sequence_matches_plan": true,
      "latest_valid_candidate_run_id": "20260712T082903Z_wing_readonly_preflight",
      "latest_valid_candidate_source_id": "artifacts/wing_readonly_preflight/20260712T082903Z_wing_readonly_preflight/summary.json",
      "live_authority_granted": false,
      "mvp_blocker_candidate_count": 4,
      "mvp_blockers_closed_by_evidence": [],
      "next_global_step": "Operator reviews the valid read-only preflight candidate before readiness matrix rows may be updated.",
      "operator_review_required": true,
      "readiness_closure_allowed_by_evidence": false,
      "readiness_matrix_update_allowed": false,
      "readiness_matrix_update_candidate_available": true,
      "valid_candidate_count": 1,
      "violations": []
    },
    "type": "wing_readonly_preflight_evidence"
  },
  "evidence_status": "readonly_preflight_candidate_available",
  "execution_allowed_by_review": false,
  "full_autonomous_ready": false,
  "generated_at": "2026-07-12T08:31:51Z",
  "invalid_candidate_count": 0,
  "latest_valid_candidate_run_id": "20260712T082903Z_wing_readonly_preflight",
  "live_authority_granted": false,
  "live_write": false,
  "manual_review_checklist": [
    {
      "id": "confirm_artifact_paths",
      "label": "Confirm summary_path and events_path point to the intended real WING run."
    },
    {
      "id": "confirm_operator_channel_scope",
      "label": "Confirm operator_confirmed_channels exactly match the queried channels."
    },
    {
      "id": "confirm_operator_session_ready",
      "label": "Confirm operator_confirmed_session_ready is true for the intended backend/WING session."
    },
    {
      "id": "confirm_physical_readbacks",
      "label": "Confirm every readback is verified_from_physical_readback for the declared value types."
    },
    {
      "id": "confirm_event_log_readonly_sequence",
      "label": "Confirm events.jsonl contains only get_operator_mode, get_state, and query_mixer_parameter sends."
    },
    {
      "id": "confirm_event_log_run_id_binding",
      "label": "Confirm every events.jsonl row has the same run_id as summary.json."
    },
    {
      "id": "confirm_no_live_authority",
      "label": "Confirm the review does not grant live authority or close readiness rows."
    }
  ],
  "manual_review_decision_required": true,
  "manual_review_required_before_readiness_update": true,
  "next_required_action": "Operator manually reviews summary.json and events.jsonl before readiness update.",
  "queries_console": false,
  "read_only": true,
  "readiness_update_candidate_available": true,
  "request_id_correlation_required": true,
  "request_id_policy": "required_for_every_query_mixer_parameter_response",
  "review_decision_template_available": true,
  "review_decision_template_required_edits": [
    "set decision to accepted or rejected after manual review",
    "set reviewed_by to the operator name",
    "set reviewed_at to a UTC timestamp at or after artifact_captured_at",
    "remove template_status or replace the generated pending marker before ledger acceptance"
  ],
  "review_decision_template_required_fields": [
    "review_id",
    "artifact_source_id",
    "run_id",
    "artifact_captured_at",
    "artifact_root",
    "summary_path",
    "summary_sha256",
    "events_path",
    "events_sha256",
    "channels",
    "value_types",
    "operator_confirmed_channels",
    "operator_confirmed_session_ready",
    "readback_count",
    "physical_readback_count",
    "request_id_policy",
    "request_id_correlation_required",
    "event_log_run_id_matches_summary",
    "decision",
    "reviewed_by",
    "reviewed_at",
    "checklist_ids",
    "backend_port_prestart_check_required",
    "backend_port_prestart_check_must_pass_before_starting_backend",
    "backend_port_prestart_check_command",
    "backend_port_prestart_check_command_text",
    "backend_port_prestart_check_failure_detail_fields",
    "backend_port_prestart_summary_required",
    "backend_port_prestart_summary_path",
    "backend_port_prestart_summary_sha256",
    "backend_port_prestart_summary_type",
    "backend_port_prestart_summary_status",
    "backend_port_prestart_summary_connection_policy",
    "backend_port_prestart_summary_read_only",
    "backend_port_prestart_summary_live_write",
    "backend_port_prestart_summary_contacts_backend",
    "backend_port_prestart_summary_contacts_wing",
    "backend_port_prestart_summary_mutates_console_state",
    "backend_port_prestart_summary_queries_console",
    "backend_port_prestart_summary_executes_tests",
    "backend_port_prestart_check_status",
    "artifact_manifest_verifier_must_pass_before_websocket",
    "artifact_manifest_verifier_passed_before_websocket",
    "artifact_manifest_verifier_websocket_opened_after_pass",
    "safe_to_stop_listener_by_gate",
    "live_authority_granted",
    "backend_port_prestart_gate_execution_allowed",
    "backend_port_prestart_gate_readiness_closure_allowed",
    "backend_port_prestart_gate_live_authority_granted",
    "plan_safe_to_stop_listener_by_gate",
    "plan_live_authority_granted",
    "plan_backend_port_prestart_gate_execution_allowed",
    "plan_backend_port_prestart_gate_readiness_closure_allowed",
    "plan_backend_port_prestart_gate_live_authority_granted",
    "safe_to_stop_listener_by_gate_matches_plan",
    "live_authority_granted_matches_plan",
    "backend_port_prestart_gate_execution_allowed_matches_plan",
    "backend_port_prestart_gate_readiness_closure_allowed_matches_plan",
    "backend_port_prestart_gate_live_authority_granted_matches_plan"
  ],
  "review_decision_template_status": "operator_must_replace_placeholders_before_ledger_acceptance",
  "source_authority_denied": true,
  "source_authority_flags": {
    "full_autonomous_ready": false,
    "live_authority_granted": false,
    "readiness_closure_allowed_by_evidence": false
  },
  "status": "candidate_ready_for_manual_review",
  "type": "wing_readonly_preflight_artifact_review",
  "valid_candidate_count": 1,
  "violations": []
}