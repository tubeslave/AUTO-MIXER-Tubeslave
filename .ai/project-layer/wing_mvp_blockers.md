# WING MVP Blockers

Purpose: this is the live task table for the supervised WING Rack MVP. Codex
must choose the next task from the first row that is not `closed`.

Status values:

- `open`: no accepted real proof yet.
- `tooling_ready_waits_real_wing`: code/report tooling exists, but real WING
  artifact is still required.
- `blocked_by_readonly_preflight`: cannot start until a clean read-only
  preflight candidate is reviewed.
- `blocked_by_real_artifact_review`: cannot close until a real artifact is
  manually reviewed.
- `closed`: accepted real proof exists.

| Blocker | Status | Evidence | Next action |
| --- | --- | --- | --- |
| WING read-only connection proof | tooling_ready_waits_real_wing | `tools/wing_readonly_preflight_runner.py`; `tools/review_wing_readonly_preflight_artifacts.py`; `tools/review_wing_readonly_preflight_decisions.py`; `tools/build_wing_readonly_preflight_handoff_packet.py`; `backend/wing_readonly_preflight_review_decision.py`; `backend/supervised_wing_proof_plan.py`; `backend/wing_readonly_preflight_contract.py`; `tools/no_wing_release_check.py`; `tools/verify_no_wing_artifact_manifest.py`; `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_RUNNER_20260606.md`; `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_EVIDENCE_20260606.md`; `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_EVENT_LOG_CROSSCHECK_20260606.md`; `Docs/reports/AUTOMIXER_WING_MVP_READONLY_EVENT_LOG_RUN_ID_BINDING_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_READONLY_EVENT_LOG_RUN_ID_FAIL_CLOSED_CONTRACT_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_REVIEW_CHECKLIST_RUN_ID_BINDING_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_REVIEW_DECISION_RUN_ID_VERDICT_FIELD_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_REVIEW_DECISION_REQUEST_ID_CORRELATION_FIELD_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_REVIEW_DECISION_SCOPE_FIELDS_20260614.md`; `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_REVIEW_RELEASE_GATE_20260607.md`; `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_OPERATOR_CONFIRMATION_20260607.md`; `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_SESSION_CONFIRMATION_GATE_20260607.md`; `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_SESSION_READY_REVIEW_CHECKLIST_20260607.md`; `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_DECISION_ARTIFACT_HASH_BINDING_20260607.md`; `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_MANUAL_REVIEW_CHECKLIST_20260607.md`; `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_REVIEW_DECISION_LEDGER_20260607.md`; `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_20260607.md`; `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_HANDOFF_PACKET_20260607.md`; `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_DECISION_GATED_PROOF_PLAN_20260607.md`; `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_HANDOFF_FORBIDDEN_REQUEST_COVERAGE_20260607.md`; `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_TRANSPORT_TIMEOUT_ARTIFACT_20260607.md`; `Docs/reports/AUTOMIXER_WING_MVP_READONLY_ARTIFACT_CONTRACT_CONSTANTS_20260613.md`; `Docs/reports/AUTOMIXER_WING_MVP_DECISION_REVIEW_PRESTART_READY_GATE_20260613.md`; `Docs/reports/AUTOMIXER_WING_MVP_DECISION_REVIEW_PRESTART_SUMMARY_FIELDS_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_DECISION_REVIEW_PRESTART_BOOL_RENDERING_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_DIRECTION_GUARD_SKIP_SELF_REPORTS_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_PRESTART_SUMMARY_SHA_BINDING_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_PRESTART_SUMMARY_SHA_AVAILABILITY_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_GATE_MATCH_VERDICTS_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_PRESTART_SUMMARY_CONTRACT_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_CURRENT_FIRST_PRESTART_SUMMARY_MATCH_FLAGS_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_OPERATOR_SEQUENCE_CARDINALITY_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_ANCHOR_DIGEST_REVIEW_STEP_INDEX_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_DECLARED_ANCHOR_DIGEST_REVIEW_INDEX_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_DECLARED_OPERATOR_SEQUENCE_COUNT_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_OPERATOR_SEQUENCE_SHA256_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_OPERATOR_SEQUENCE_CONTRACT_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_ACCEPTED_CANDIDATE_PRESTART_DIAGNOSTICS_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_CURRENT_FIRST_COMMAND_DIRECTOR_SOURCE_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_PRESTART_SUMMARY_TYPE_CURRENT_COMMAND_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_PRESTART_SUMMARY_TYPE_BLOCKER_ROW_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_PRESTART_SUMMARY_POLICY_RAILS_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_PRESTART_SUMMARY_POLICY_BLOCKER_ROW_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_READINESS_PRESTART_SUMMARY_POLICY_BRIDGE_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_READINESS_PRESTART_SUMMARY_CONTRACT_VERDICT_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_BACKEND_PRESTART_LISTENER_PROBE_ERROR_FIELD_20260613.md`; `Docs/reports/AUTOMIXER_WING_MVP_BACKEND_PORT_LOCAL_SOCKET_TEST_HELPER_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_NOWING_QUICK_BUSY_PORT_FAIL_CLOSED_20260613.md`; `Docs/reports/AUTOMIXER_WING_MVP_BACKEND_LISTENER_CONFIRMATION_ARTIFACT_20260613.md`; `Docs/reports/AUTOMIXER_WING_MVP_BACKEND_LISTENER_CONFIRMATION_TEMPLATE_20260613.md`; `Docs/reports/AUTOMIXER_WING_MVP_BACKEND_LISTENER_TEMPLATE_EDIT_GATE_20260613.md`; `Docs/reports/AUTOMIXER_WING_MVP_BACKEND_LISTENER_TEMPLATE_PENDING_SUMMARY_FIELDS_20260613.md`; `Docs/reports/AUTOMIXER_WING_MVP_BACKEND_LISTENER_TEMPLATE_AUTHORITY_SUMMARY_FIELDS_20260613.md`; `Docs/reports/AUTOMIXER_WING_MVP_BACKEND_PRESTART_TEMPLATE_FAILURE_DETAIL_FIELDS_20260613.md`; `Docs/reports/AUTOMIXER_WING_MVP_BACKEND_PRESTART_TEMPLATE_REVIEW_FIELDS_20260613.md`; `Docs/reports/AUTOMIXER_WING_MVP_BUSY_PORT_PRESTART_DETAIL_FIELDS_SUMMARY_20260613.md`; `Docs/reports/AUTOMIXER_WING_MVP_BUSY_PORT_PRESTART_COMMAND_SUMMARY_20260613.md`; `Docs/reports/AUTOMIXER_WING_MVP_BUSY_PORT_PRESTART_REQUIRED_FLAGS_SUMMARY_20260613.md`; `Docs/reports/AUTOMIXER_WING_MVP_BUSY_PORT_PRESTART_GATE_AUTHORITY_SUMMARY_20260613.md`; `Docs/reports/AUTOMIXER_WING_MVP_READONLY_PRESTART_SUMMARY_BINDING_20260613.md`; `Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_PRESTART_SUMMARY_PLACEHOLDER_20260613.md`; `Docs/reports/AUTOMIXER_WING_MVP_PRESTART_SUMMARY_CONTACT_BOUNDARY_20260613.md`; `Docs/reports/AUTOMIXER_WING_MVP_PRESTART_FAIL_CLOSED_CONDITION_SYNC_20260613.md`; `Docs/reports/AUTOMIXER_WING_MVP_FAIL_CLOSED_CONTRACT_SOURCE_20260613.md`; `Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_GATE_CONTRACT_SOURCE_20260613.md`; `Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_EXPECTED_ARTIFACTS_CONTRACT_20260613.md`; `Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_EXPECTED_ARTIFACTS_NEXT_ACTION_20260613.md`; `Docs/reports/AUTOMIXER_WING_MVP_REAL_ARTIFACT_EXPECTED_ARTIFACTS_BINDING_20260613.md`; `Docs/reports/AUTOMIXER_WING_MVP_DECISION_EXPECTED_ARTIFACTS_BINDING_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_SUPERVISED_GATE_EXPECTED_ARTIFACT_MARKER_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_BACKEND_CONNECTION_CONFIRMATION_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_COMMAND_ARTIFACT_BINDING_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_PRESTART_SUMMARY_COMMAND_BINDING_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_BACKEND_PRESTART_COMMAND_SUMMARY_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_BACKEND_PRESTART_COMMAND_TEXT_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_PRESTART_BACKEND_NOAUTHORITY_DETAIL_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_LISTENER_TEMPLATE_ACCEPTANCE_STATUS_FIELD_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_LISTENER_TEMPLATE_PATH_FAILURE_FIELD_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_DENIAL_FLAG_SUMMARY_TRUTH_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_SAFETY_FLAG_SUMMARY_TRUTH_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_PROOF_PLAN_PRESTART_VIOLATIONS_MAP_SUMMARY_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_PROOF_PLAN_CANONICAL_PRESTART_HELPER_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_PROOF_PLAN_PREFLIGHT_COMMAND_BINDINGS_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_PROOF_PLAN_OPERATOR_CONFIRMATION_BINDINGS_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_PROOF_PLAN_CANONICAL_PRESTART_DIAGNOSTICS_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_PROOF_PLAN_SUMMARY_FIELDS_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_AUTOLOOP_PROOF_PLAN_SUMMARY_FIELDS_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_CURRENT_FIRST_COMMAND_PROOF_PLAN_SUMMARY_FIELDS_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_CURRENT_FIRST_COMMAND_SELECTED_ACTION_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_DIRECTOR_ACTION_COVERAGE_DIGEST_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_DIRECTOR_ACTION_DIGEST_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_CURRENT_COMMAND_DIRECTOR_ACTION_DIGEST_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_CURRENT_COMMAND_DIRECTOR_DIGEST_RENDERED_FIELDS_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_DECLARED_DIRECTOR_SELF_CHECK_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_SUPERVISED_SUMMARY_FIELD_CONTRACT_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_CURRENT_EVIDENCE_LATEST_MATCH_REPORTS_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_CURRENT_EVIDENCE_INDEX_SYNC_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_NOWING_FRONTEND_ARTIFACT_PRESERVATION_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_NOWING_FRONTEND_BIND_ENVIRONMENT_BLOCKED_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_RUNNER_ENVIRONMENT_BLOCKED_PRESTART_SUMMARY_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_PRESTART_SUMMARY_ACTUAL_METADATA_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_REQUIRED_FALSE_DETAIL_CONTRACT_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_DENIAL_MATCH_FLAGS_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_DENIAL_MATCH_FLAG_RELEASE_GATE_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_CURRENT_COMMAND_REQUIRED_FALSE_DETAIL_MATCH_FLAGS_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_CURRENT_COMMAND_DECLARED_DENIAL_MATCH_FLAGS_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_RUNNER_PRESTART_DETAIL_LIST_CONTRACT_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_EVIDENCE_PRESTART_DETAIL_CONTRACT_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_CANDIDATE_PRESTART_REQUIRED_FALSE_TRACE_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_PRESTART_REQUIRED_FALSE_PLAN_MATCH_TRACE_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_PRESTART_PLAN_FIELDS_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_NOWING_HANDOFF_PLAN_FIELD_MISSING_GATE_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_ACCEPTED_CANDIDATE_PRESTART_SHA_FORMAT_GATE_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_NOWING_PLAN_PRESTART_SHA_FORMAT_GATE_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_READONLY_RUNTIME_ARTIFACT_ROOT_GATE_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_EVIDENCE_SHA_FORMAT_GATE_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_REVIEW_DECISION_ARTIFACT_SHA_FORMAT_GATE_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_REVIEW_DECISION_REVIEW_ID_REQUIRED_GATE_20260615.md`; `Docs/reports/AUTOMIXER_WING_MVP_REVIEW_DECISION_TEMPLATE_STATUS_UNEDITED_GATE_20260615.md`; `Docs/reports/AUTOMIXER_WING_MVP_REVIEW_DECISION_TEMPLATE_REQUIRED_EDITS_SURFACE_20260615.md`; `Docs/reports/AUTOMIXER_WING_MVP_REVIEW_DECISION_ARTIFACT_ROOT_GATE_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_DECISION_REVIEW_ARTIFACT_ROOT_REPORT_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_DECISION_REVIEW_ARTIFACT_ROOT_CONTRACT_SURFACE_20260615.md`; `Docs/reports/AUTOMIXER_WING_MVP_DECISION_REVIEW_INVALID_RECORD_DIAGNOSTICS_20260615.md`; `Docs/reports/AUTOMIXER_WING_MVP_NOWING_REVIEW_DECISION_INVALID_RECORD_DIAGNOSTICS_20260615.md`; `Docs/reports/AUTOMIXER_WING_MVP_PROOF_PLAN_REVIEW_DECISION_INVALID_RECORD_DIAGNOSTICS_20260615.md`; `Docs/reports/AUTOMIXER_WING_MVP_CURRENT_COMMAND_INVALID_RECORD_MARKER_20260615.md`; `Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_INVALID_RECORD_MARKER_20260615.md`; `Docs/reports/AUTOMIXER_WING_MVP_SUPERVISED_PRESTART_ARTIFACT_ROOT_MARKER_GATE_20260615.md`; `Docs/reports/AUTOMIXER_WING_MVP_RUNBOOK_ACCEPTED_DECISION_ARTIFACT_ROOT_SHAPE_20260615.md`; `Docs/reports/AUTOMIXER_WING_MVP_PROOF_PLAN_ARTIFACT_ROOT_TRACE_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_PROOF_PLAN_ARTIFACT_ROOT_SUMMARY_FIELDS_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_PROOF_PLAN_REQUIRED_SUMMARY_FIELD_GATE_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_NOWING_PROOF_PLAN_REQUIRED_FIELD_SUMMARY_EMBED_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_NOWING_PROOF_PLAN_REQUIRED_FIELD_MARKDOWN_SUMMARY_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_CURRENT_COMMAND_MARKDOWN_SUMMARY_SURFACE_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_NOWING_OPERATOR_MARKDOWN_SUMMARY_SELF_CHECK_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_NOWING_OPERATOR_MARKDOWN_SUMMARY_SHA_BINDING_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_NOWING_OPERATOR_MARKDOWN_SUMMARY_SHA_MATCH_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_NOWING_ARTIFACT_MANIFEST_SIDECAR_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_NOWING_ARTIFACT_MANIFEST_REQUIRED_SET_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_NOWING_ARTIFACT_MANIFEST_SHA_AVAILABILITY_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_NOWING_ARTIFACT_MANIFEST_SUMMARY_SURFACE_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_NOWING_ARTIFACT_MANIFEST_SUMMARY_SURFACE_SELF_CHECK_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_NOWING_ARTIFACT_MANIFEST_CONTRACT_DIGEST_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_NOWING_ARTIFACT_MANIFEST_CROSS_SURFACE_DIGEST_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_NOWING_MANIFEST_CROSS_SURFACE_SUMMARY_20260615.md`; `Docs/reports/AUTOMIXER_WING_MVP_NOWING_MANIFEST_SUMMARY_JSON_SELF_CHECK_20260615.md`; `Docs/reports/AUTOMIXER_WING_MVP_NOWING_MANIFEST_SUMMARY_MARKDOWN_SELF_CHECK_RENDER_20260615.md`; `Docs/reports/AUTOMIXER_WING_MVP_NOWING_MANIFEST_CROSS_SURFACE_SUMMARY_SELF_CHECK_20260615.md`; `Docs/reports/AUTOMIXER_WING_MVP_NOWING_MANIFEST_CROSS_SURFACE_SUBCHECK_RENDER_20260615.md`; `Docs/reports/AUTOMIXER_WING_MVP_NOWING_MANIFEST_CROSS_SURFACE_SAFETY_DENIAL_20260615.md`; `Docs/reports/AUTOMIXER_WING_MVP_NOWING_ARTIFACT_MANIFEST_SELF_CONSISTENCY_20260615.md`; `Docs/reports/AUTOMIXER_WING_MVP_NOWING_ARTIFACT_MANIFEST_SELF_CONSISTENCY_SUMMARY_SURFACE_20260615.md`; `Docs/reports/AUTOMIXER_WING_MVP_NOWING_ARTIFACT_MANIFEST_SELF_CONSISTENCY_SUMMARY_SURFACE_VALUE_GATE_20260615.md`; `Docs/reports/AUTOMIXER_WING_MVP_NOWING_ARTIFACT_MANIFEST_SELF_CONSISTENCY_SUMMARY_SURFACE_SIDECAR_BINDING_20260615.md`; `Docs/reports/AUTOMIXER_WING_MVP_NOWING_ARTIFACT_MANIFEST_VERIFIER_20260615.md`; `Docs/reports/AUTOMIXER_WING_MVP_READONLY_RUNBOOK_NOWING_MANIFEST_SIDECAR_CHECKLIST_20260615.md`; `Docs/reports/AUTOMIXER_WING_MVP_CURRENT_COMMAND_MANIFEST_SIDECAR_CHECKLIST_20260615.md`; `Docs/reports/AUTOMIXER_WING_MVP_PROJECT_DIRECTOR_SELECTED_ACTION_SUMMARY_JSON_SURFACE_20260615.md`; `Docs/reports/AUTOMIXER_WING_MVP_BACKEND_PRESTART_GATE_SUMMARY_JSON_SURFACE_20260615.md`; `Docs/reports/AUTOMIXER_WING_MVP_BACKEND_PRESTART_GATE_OBJECT_SUMMARY_JSON_SURFACE_20260615.md`; `Docs/reports/AUTOMIXER_WING_MVP_BACKEND_PRESTART_GATE_AUTHORITY_DENIAL_SUMMARY_JSON_SURFACE_20260615.md`; `Docs/reports/AUTOMIXER_WING_MVP_BACKEND_PRESTART_FAILURE_DETAIL_SUMMARY_JSON_SURFACE_20260615.md`; `Docs/reports/AUTOMIXER_WING_MVP_PACKET_SAFETY_BOUNDARY_SUMMARY_JSON_SURFACE_20260615.md`; `Docs/reports/AUTOMIXER_WING_MVP_BLOCKER_ANCHOR_SUMMARY_JSON_SURFACE_20260615.md`; `Docs/reports/AUTOMIXER_WING_MVP_EXPECTED_ARTIFACTS_SUMMARY_JSON_SURFACE_20260615.md`; `Docs/reports/AUTOMIXER_WING_MVP_ACCEPTED_CANDIDATE_EXPECTED_ARTIFACT_TRACE_SUMMARY_JSON_SURFACE_20260615.md`; `Docs/reports/AUTOMIXER_WING_MVP_ACCEPTED_CANDIDATE_EXPECTED_ARTIFACT_TRACE_REQUIRED_FIELDS_20260615.md`; `Docs/reports/AUTOMIXER_WING_MVP_ACCEPTED_CANDIDATE_EXPECTED_ARTIFACT_TRACE_VALUE_GATE_20260615.md`; `Docs/reports/AUTOMIXER_WING_MVP_ACCEPTED_CANDIDATE_EXPECTED_ARTIFACT_TRACE_VALUE_GATE_SUMMARY_SURFACE_20260615.md`; `Docs/reports/AUTOMIXER_WING_MVP_ACCEPTED_CANDIDATE_EXPECTED_ARTIFACT_TRACE_VALUE_GATE_SUMMARY_SURFACE_SELF_CHECK_20260615.md`; `Docs/reports/AUTOMIXER_WING_MVP_PROJECT_DIRECTOR_VALUE_GATE_SELF_CHECK_CURRENT_COMMAND_SYNC_20260615.md`; `Docs/reports/AUTOMIXER_WING_MVP_BACKEND_PRESTART_SUMMARY_JSON_SURFACE_20260615.md`; `Docs/reports/AUTOMIXER_WING_MVP_COMMAND_BINDING_SUMMARY_JSON_SURFACE_20260615.md`; `Docs/reports/AUTOMIXER_WING_MVP_OPERATOR_SEQUENCE_SUMMARY_JSON_SURFACE_20260615.md`; `Docs/reports/AUTOMIXER_WING_MVP_OPERATOR_SEQUENCE_STEP_GATE_SUMMARY_JSON_SURFACE_20260615.md`; `Docs/reports/AUTOMIXER_WING_MVP_SUPERVISED_PROOF_PLAN_FIELDS_SUMMARY_JSON_SURFACE_20260615.md`; `Docs/reports/AUTOMIXER_WING_MVP_NEXT_PROOF_PLAN_CANDIDATE_CONTRACTS_SUMMARY_JSON_SURFACE_20260615.md`; `Docs/reports/AUTOMIXER_WING_MVP_NEXT_PROOF_PLAN_REQUIRED_FIELDS_SUMMARY_JSON_SURFACE_20260615.md`; `Docs/reports/AUTOMIXER_WING_MVP_PROOF_PLAN_PRESTART_VIOLATIONS_STEP_GATE_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_EVIDENCE_BOOLEAN_NUMERIC_SCOPE_GATE_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_EVIDENCE_STRICT_INTEGER_SCOPE_GATE_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_REVIEW_TEMPLATE_COUNT_TYPE_GATE_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_REVIEW_TEMPLATE_STRICT_SCOPE_TYPE_GATE_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_REVIEW_TEMPLATE_VALUE_TYPE_GATE_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_REVIEW_TEMPLATE_ARTIFACT_SHA_GATE_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_REVIEW_TEMPLATE_EXPECTED_ARTIFACT_PATH_GATE_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_ACCEPTED_CANDIDATE_EXPECTED_ARTIFACT_PATH_CONTRACT_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_ACCEPTED_CANDIDATE_ARTIFACT_ROOT_PATH_SCOPE_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_ACCEPTED_CANDIDATE_ARTIFACT_ROOT_PARENT_TRAVERSAL_GATE_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_ACCEPTED_CANDIDATE_ARTIFACT_ROOT_CANONICAL_ROOT_GATE_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_REVIEW_DECISION_LEDGER_ARTIFACT_ROOT_CONTRACT_GATE_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_NOWING_REVIEW_DECISION_LEDGER_ARTIFACT_ROOT_SUMMARY_SURFACE_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_REVIEW_DECISION_STRICT_SCOPE_TYPE_GATE_20260614.md`; `Docs/reports/AUTOMIXER_WING_MVP_PROGRESS_DIRECTOR_EVIDENCE_SUMMARY_20260614.md` | Run `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick` before starting the intended backend/WING session, then run `Docs/runbooks/WING_READONLY_PREFLIGHT.md`. Before any supervised proof, the no-WING handoff summary must show `operator_sequence_count: 12`, `expected_operator_sequence_count: 12`, `operator_sequence_count_matches_contract: true`, `operator_sequence_sha256`, `actual_operator_sequence_sha256`, `operator_sequence_sha256_matches_actual: true`, `expected_operator_sequence_sha256`, `actual_expected_operator_sequence_sha256`, `operator_sequence_matches_contract: true`, `operator_sequence_sha256_matches_contract: true`, `declared_expected_operator_sequence_sha256_matches_contract: true`, `declared_operator_sequence_matches_contract_flag: true`, `declared_operator_sequence_matches_contract_flag_matches_actual: true`, `declared_operator_sequence_sha256_matches_contract_flag: true`, `declared_operator_sequence_sha256_matches_contract_flag_matches_actual: true`, `declared_operator_sequence_count: 12`, `declared_expected_operator_sequence_count: 12`, `declared_operator_sequence_count_contract_flag: true`, `declared_operator_sequence_count_matches_actual: true`, `declared_operator_sequence_count_matches_contract: true`, `declared_expected_operator_sequence_count_matches_contract: true`, `declared_operator_sequence_count_contract_flag_matches_declared_count: true`, `backend_port_prestart_gate` exactly matching `expected_backend_port_prestart_gate`, with `backend_port_prestart_gate.grants_live_authority: false`, `backend_port_prestart_gate.must_pass_before_backend_wing_session: true`, `backend_port_prestart_gate.operator_sequence_step_index: 0`, `backend_port_prestart_gate.operator_sequence_step_must_be_first: true`, `backend_port_prestart_gate.runs_command: prestart_backend_port_check`, `backend_port_prestart_gate.required_markers` exactly listing `Before starting the intended backend/WING session`, `prestart_backend_port_check`, and `unexpected listeners`, the failed busy-port no-WING summary and no-WING handoff summary must both show `backend_port_prestart_check_command` exactly matching `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick`, `backend_port_prestart_check_required: true`, `backend_port_prestart_check_must_pass_before_starting_backend: true`, `backend_port_prestart_gate_execution_allowed: false`, `backend_port_prestart_gate_readiness_closure_allowed: false`, `backend_port_prestart_gate_live_authority_granted: false`, and `backend_port_prestart_check_failure_detail_fields` exactly listing `listener_processes`, `listener_process_error`, `listener_confirmation_checklist`, `preflight_blocked_until_listener_confirmed`, `intended_backend_confirmation_required`, `safe_to_stop_listener_by_gate`, `listener_confirmation_template_status`, `listener_confirmation_template_path`, `listener_confirmation_template_acceptance_status`, `listener_confirmation_template_safe_to_stop_listener_by_gate`, `listener_confirmation_template_live_authority_granted`, `listener_confirmation_template_accepted_by_template`, `listener_confirmation_template_operator_placeholder_fields`, `listener_confirmation_template_execution_allowed_by_template`, `listener_confirmation_template_readiness_closure_allowed_by_template`, `listener_confirmation_template_required_edits`, `backend_port_prestart_gate_execution_allowed`, `backend_port_prestart_gate_readiness_closure_allowed`, `backend_port_prestart_gate_live_authority_granted`, `fail_closed_reason`, and `operator_next_step`; if a backend listener is already present, the failed no-WING summary must expose `listener_confirmation_template_path`, `listener_confirmation_template_status: pending_operator_edit`, `listener_confirmation_template_safe_to_stop_listener_by_gate: false`, and `listener_confirmation_template_live_authority_granted: false`, `listener_confirmation_template_accepted_by_template: false`, `listener_confirmation_template_operator_placeholder_fields` listing `operator_reviewed_at` and `operator_name`, `listener_confirmation_template_execution_allowed_by_template: false`, `listener_confirmation_template_readiness_closure_allowed_by_template: false`, `listener_confirmation_template_required_edits` including `set template_status to operator_reviewed_and_edited if using the generated template` and `set accepted_by_template to true if using the generated template`, and any template-derived accepted confirmation must replace `REPLACE_WITH_*` operator placeholders, set `template_status: operator_reviewed_and_edited`, and set `accepted_by_template: true`, `manual_review_required_before_readiness_update: true`, `readiness_matrix_update_allowed_by_packet: false`, `readiness_closure_allowed_by_packet: false`, `read_only: true`, `live_write: false`, `mutates_console_state: false`, `queries_console: false`, `executes_tests: false`, `operator_required: true`, `expected_plan_required_for_real_run: true`, `physical_preflight_requires_existing_backend_connection: true`, `backend_connection_confirmation_step_index: 3`, `operator_sequence_has_backend_connection_confirmation: true`, `operator_sequence_backend_connection_confirmation_step_index: 3`, `operator_sequence_backend_connection_confirmation_before_capture: true`, `operator_sequence_backend_connection_confirmation_step` including `existing backend connection`, `intended WING Rack`, and `read-only inspection`, `operator_sequence_validate_readonly_preflight_inputs_step_index: 5`, `expected_operator_sequence_validate_readonly_preflight_inputs_step_index: 5`, `operator_sequence_validate_readonly_preflight_inputs_step_index_matches_contract: true`, `operator_sequence_validate_readonly_preflight_inputs_step: Run validate_readonly_preflight_inputs before capture; it must validate real-run inputs and expected-plan bindings without opening transport.`, `expected_operator_sequence_validate_readonly_preflight_inputs_step: Run validate_readonly_preflight_inputs before capture; it must validate real-run inputs and expected-plan bindings without opening transport.`, `operator_sequence_validate_readonly_preflight_inputs_step_matches_contract: true`, `operator_sequence_capture_readonly_preflight_step_index: 6`, `backend_port_prestart_check_required: true`, `backend_port_prestart_check_must_pass_before_starting_backend: true`, `backend_port_prestart_gate_execution_allowed: false`, `backend_port_prestart_gate_readiness_closure_allowed: false`, `backend_port_prestart_gate_live_authority_granted: false`, `backend_port_prestart_summary_required: true`, `backend_port_prestart_summary_type: no_wing_release_check`, `backend_port_prestart_summary_connection_policy: no_hardware_no_network_no_live_writes`, `backend_port_prestart_summary_path_placeholder` pointing to the passed no-WING prestart `summary.json`, `backend_port_prestart_summary_sha256_matches_plan: true`, `backend_port_prestart_summary_sha256_available: true`, `backend_port_prestart_summary_sha256_available_matches_digest: true`, `preview_command_prestart_summary` matching that path, `preview_command_prestart_summary_matches_plan: true`, `validate_inputs_command_prestart_summary` matching that path, `validate_inputs_command_prestart_summary_matches_plan: true`, `capture_command_prestart_summary` matching that path, `capture_command_prestart_summary_matches_plan: true`, `execution_allowed_by_packet: false`, `live_authority_granted: false`, `full_autonomous_ready: false`, `next_manual_artifact_required: real wing_readonly_preflight.v1 summary.json`, `expected_artifacts` exactly matching `expected_artifact_globs` and `plan_expected_artifacts`, `expected_artifacts_match_plan: true`, `summary_glob_matches_plan: true`, `event_log_glob_matches_plan: true`, `preview_command_artifact_root_matches_packet: true`, `preview_command_plan_output_matches_packet: true`, `validate_inputs_command_uses_validate_run_inputs_only: true`, `validate_inputs_command_uses_plan_only: false`, `validate_inputs_command_artifact_root_matches_packet: true`, `validate_inputs_command_expected_plan_matches_packet: true`, `capture_command_artifact_root_matches_packet: true`, `capture_command_expected_plan_matches_packet: true`, `review_command_artifact_root_matches_packet: true`, `decision_review_command_artifact_root_matches_packet: true`, `decision_review_command_decision_root_matches_packet: true`, `backend_port_prestart_check_operator_sequence_step_index: 0`, `backend_port_prestart_check_operator_sequence_step` including `prestart_backend_port_check` and `unexpected listeners`, `backend_port_prestart_check_must_be_first_operator_step: true`, `backend_port_prestart_check_operator_sequence_step_is_first: true`, `mvp_blocker_anchor` matching the current first open blocker row, `mvp_blocker_anchor_matches_current_first_open: true`, `mvp_blocker_anchor_sha256_matches_current_first_open: true`, `mvp_blocker_anchor.packet_does_not_close_blocker: true`, `operator_sequence_has_mvp_blocker_anchor_digest_review: true`, `mvp_blocker_anchor_digest_review_step_index: 10`, `expected_mvp_blocker_anchor_digest_review_step_index: 10`, `mvp_blocker_anchor_digest_review_step_index_matches_contract: true`, `mvp_blocker_anchor_digest_review_step_index_matches_operator_sequence: true`, `operator_sequence_mvp_blocker_anchor_digest_review_step_index: 10`, `expected_operator_sequence_mvp_blocker_anchor_digest_review_step_index: 10`, `operator_sequence_mvp_blocker_anchor_digest_review_step_index_matches_contract: true`, `operator_sequence_mvp_blocker_anchor_digest_review_step` including `mvp_blocker_anchor`, `mvp_blocker_anchor_sha256`, and `current first open blocker`, `operator_sequence_has_supervised_prestart_gate: true`, `operator_sequence_supervised_prestart_gate_is_final: true`, `operator_sequence_supervised_prestart_gate_step_index: 11`, `expected_operator_sequence_supervised_prestart_gate_step_index: 11`, `operator_sequence_supervised_prestart_gate_step_index_matches_contract: true`, `operator_sequence_supervised_prestart_gate_step` including `accepted_decision_ready`, `valid accepted-candidate prestart trace`, `valid accepted-candidate artifact-root trace`, `valid accepted-candidate request-sequence trace`, `expected-artifact-bound accepted source id`, and `WING_SUPERVISED_FADER_GAIN_PROOF`, `required_supervised_prestart_gate_markers` exactly listing `accepted_decision_ready`, `valid accepted-candidate prestart trace`, `valid accepted-candidate artifact-root trace`, `valid accepted-candidate request-sequence trace`, `expected-artifact-bound accepted source id`, and `WING_SUPERVISED_FADER_GAIN_PROOF`, `supervised_prestart_gate` exactly matching `expected_supervised_prestart_gate`, `supervised_prestart_gate.grants_live_authority: false`, `supervised_prestart_gate.operator_sequence_step_index: 11`, `supervised_prestart_gate.operator_sequence_step_must_be_final: true`, `supervised_prestart_gate.requires_operator_decision_status: accepted_decision_ready`, `supervised_prestart_gate.unlocks_runbook: Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md`, and `supervised_prestart_gate.requires_accepted_candidate_prestart_trace: true`; it must also show `supervised_prestart_gate.requires_accepted_candidate_artifact_root_trace: true`; it must also show `supervised_prestart_gate.requires_accepted_candidate_request_sequence_trace: true`; accepted review source ids must also have a valid accepted-candidate prestart trace, the no-WING handoff packet must expose `required_supervised_proof_plan_summary_fields` and `required_supervised_proof_plan_summary_fields_match_contract: true`, `required_supervised_proof_plan_summary_fields` must include `readonly_preflight_review_decision_expected_candidate_prestart_source_ids`, `readonly_preflight_review_decision_expected_candidate_prestart_contract_violations_by_source`, `readonly_preflight_review_decision_candidate_artifact_root_contract_valid`, `readonly_preflight_review_decision_candidate_artifact_roots`, `readonly_preflight_review_decision_candidate_artifact_root_contract_violations_by_source`, `readonly_preflight_review_decision_expected_candidate_artifact_roots`, and `readonly_preflight_review_decision_expected_candidate_artifact_root_contract_violations_by_source`, and the no-WING proof plan summary must expose those canonical helper fields, `required_supervised_proof_plan_summary_fields`, and `missing_required_supervised_proof_plan_summary_fields` computed from the canonical helpers; `next_wing_proof_plan_schema` must fail closed with `next_wing_proof_plan_required_summary_fields_missing` when any required summary field is absent; the no-WING proof plan summary must also expose `next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_summary_surface_self_check_required: true`, `next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_summary_surface_required_fields`, `next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_summary_surface_mismatches: []`, `next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_summary_surface_match: true`, `next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_summary_surface_readiness_closure_allowed: false`, and `next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_summary_surface_live_authority_granted: false`; the no-WING proof plan summary must also expose `next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_summary_surface_self_check_required: true`, `next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_summary_surface_required_fields`, `next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_summary_surface_mismatches: []`, `next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_summary_surface_match: true`, `next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_summary_surface_readiness_closure_allowed: false`, and `next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_summary_surface_live_authority_granted: false`; the canonical helper fields must be computed with `expected_artifacts`, `expected_summary_artifact`, `expected_events_artifact`, `summary_path_matches_expected_artifact: true`, `events_path_matches_expected_artifact: true`, and a bound no-WING prestart `summary.json` path and SHA256 showing `backend_port_prestart_summary_required: true`, `backend_port_prestart_summary_type: no_wing_release_check`, `backend_port_prestart_summary_connection_policy: no_hardware_no_network_no_live_writes`, `backend_port_prestart_summary_status: passed`, `backend_port_prestart_summary_read_only: true`, `backend_port_prestart_summary_live_write: false`, `backend_port_prestart_summary_contacts_backend: false`, `backend_port_prestart_summary_contacts_wing: false`, `backend_port_prestart_summary_mutates_console_state: false`, `backend_port_prestart_summary_queries_console: false`, `backend_port_prestart_summary_executes_tests: false`, and `backend_port_prestart_check_status: passed`, `safe_to_stop_listener_by_gate_matches_plan: true`, `live_authority_granted_matches_plan: true`, `backend_port_prestart_gate_execution_allowed_matches_plan: true`, `backend_port_prestart_gate_readiness_closure_allowed_matches_plan: true`, and `backend_port_prestart_gate_live_authority_granted_matches_plan: true`, otherwise keep `accepted_readonly_preflight_candidate_prestart_contract_required_before_supervised_proof` or `accepted_readonly_preflight_candidate_artifact_root_required_before_supervised_proof` in the proof plan blockers. |
| Physical console readback proof | tooling_ready_waits_real_wing | `wing_readonly_preflight.v1` schema requires physical readback; event-log crosscheck and `tools/review_wing_readonly_preflight_artifacts.py` review exist | Run read-only preflight with `--require-physical-readback` and review summary plus `events.jsonl`. |
| Supervised fader write proof | blocked_by_readonly_preflight | `Docs/reports/AUTOMIXER_SUPERVISED_WING_PROOF_READONLY_PREFLIGHT_GATE_20260606.md`; `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_DECISION_GATED_PROOF_PLAN_20260607.md` | After clean read-only preflight review, an accepted decision JSON, `backend_port_prestart_gate` exactly matching `expected_backend_port_prestart_gate` without live authority, the final handoff operator-sequence step with valid accepted-candidate prestart trace, valid accepted-candidate artifact-root trace, valid accepted-candidate request-sequence trace, `expected-artifact-bound accepted source id`, and `supervised_prestart_gate` exactly matching `expected_supervised_prestart_gate`, run `Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md` for `--value-type fader`. |
| Supervised gain-down / trim proof | blocked_by_readonly_preflight | `Docs/reports/AUTOMIXER_SUPERVISED_WING_PROOF_READONLY_PREFLIGHT_GATE_20260606.md`; `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_DECISION_GATED_PROOF_PLAN_20260607.md` | After clean read-only preflight review, an accepted decision JSON, `backend_port_prestart_gate` exactly matching `expected_backend_port_prestart_gate` without live authority, the final handoff operator-sequence step with valid accepted-candidate prestart trace, valid accepted-candidate artifact-root trace, valid accepted-candidate request-sequence trace, `expected-artifact-bound accepted source id`, and `supervised_prestart_gate` exactly matching `expected_supervised_prestart_gate`, run `Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md` for `--value-type gain` with a negative delta only. |
| Rollback proof | blocked_by_readonly_preflight | `tools/wing_operator_proposal_validation_runner.py` supports rollback through `rollback_last_manual_write` | Preserve rollback request, rollback readback, and artifact review from the supervised fader/gain run. |
| Emergency-stop proof | blocked_by_readonly_preflight | runner supports `--prove-emergency-stop`; server blocks later apply with `emergency_stop_active` | Use `--prove-emergency-stop` only inside the explicit supervised proof run. |
| Connection-loss / timeout fail-closed proof | tooling_ready_waits_real_wing | `Docs/reports/AUTOMIXER_WING_TIMEOUT_FAIL_CLOSED_PROOF_20260606.md` | Capture a controlled real-run timeout/connection-loss artifact and verify fail-closed semantics. |
| Operator-visible evidence report | tooling_ready_waits_real_wing | `.paperclip/reports/`; `Docs/reports/`; supervised proof report embeds autonomy readiness | Save preflight/proof artifacts and a clear operator-visible report for manual review. |
| Readiness matrix update based on real proof | blocked_by_real_artifact_review | `backend/autonomy_promotion_readiness.py`; `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_READINESS_BRIDGE_20260606.md` | Update readiness only after manual review of real WING artifacts; do not close rows from offline evidence. |

## Current First Open Blocker

WING read-only connection proof.

## Current First Command

Use `PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick` before
starting the intended backend/WING session, then use
`Docs/runbooks/WING_READONLY_PREFLIGHT.md`. Before opening the backend/WING
session, run
`PYTHONPATH=backend python3 tools/verify_no_wing_artifact_manifest.py <no_wing_artifact_dir>`
on the same passed no-WING artifact directory and require `status: passed`,
`failed_checks: []`, `artifact_manifest_self_consistency_match: true`,
`artifact_manifest_contract_cross_surface_match: true`,
`artifact_manifest_self_consistency_summary_surface_match: true`,
`read_only: true`, `live_write: false`, `contacts_wing: false`,
`contacts_backend: false`, `mutates_console_state: false`,
`queries_console: false`, `executes_tests: false`,
`readiness_closure_allowed: false`, and `live_authority_granted: false`. This
verifier is read-only and grants no live authority. The no-WING summary must
also expose `artifact_manifest_verifier_required: true`,
`artifact_manifest_verifier_command`,
`artifact_manifest_verifier_status_required: passed`,
`artifact_manifest_verifier_failed_checks_required: []`,
`artifact_manifest_verifier_report_required_fields`,
`artifact_manifest_verifier_contacts_wing: false`,
`artifact_manifest_verifier_mutates_console_state: false`,
`artifact_manifest_verifier_queries_console: false`,
`artifact_manifest_verifier_readiness_closure_allowed: false`, and
`artifact_manifest_verifier_live_authority_granted: false`. The detached
`artifact-manifest.json` sidecar must mirror the same verifier/no-authority
fields, and the verifier report must show
`manifest_artifact_manifest_verifier_fields_present: true` and
`manifest_artifact_manifest_verifier_command_matches_expected: true`,
`summary_json_artifact_manifest_verifier_report_required_fields_matches_expected: true`,
`summary_md_artifact_manifest_verifier_report_required_fields_matches_expected: true`,
`manifest_artifact_manifest_verifier_report_required_fields_matches_expected: true`,
`verifier_report_required_fields_present: true`,
`verifier_report_required_fields_missing: []`, and
`verifier_report_required_fields_match_contract: true`, and
`failed_checks_have_check_entries: true`, and
`failed_checks_missing_check_entries: []`. The no-WING handoff operator
sequence must place this verifier before preview and expose
`artifact_manifest_verifier_operator_sequence_step_index: 1`,
`expected_artifact_manifest_verifier_operator_sequence_step_index: 1`,
`artifact_manifest_verifier_operator_sequence_step_index_matches_contract: true`,
`artifact_manifest_verifier_operator_sequence_step`,
`expected_artifact_manifest_verifier_operator_sequence_step`,
`artifact_manifest_verifier_operator_sequence_step_matches_contract: true`,
`artifact_manifest_verifier_operator_sequence_before_preview: true`, and
`artifact_manifest_verifier_required_markers`. Before any read-only preflight
WebSocket opens, `tools/wing_readonly_preflight_runner.py` must also enforce
this verifier against the artifact directory containing `--prestart-summary`
and fail closed if the detached manifest verifier fails. Before any
supervised proof,
the no-WING handoff summary must show `operator_sequence_count: 12`;
it must also show `expected_operator_sequence_count: 12`;
it must also show top-level `operator_sequence`;
it must also show top-level `expected_operator_sequence`;
it must also show
`wing_readonly_operator_sequence_list_summary_surface_self_check_required: true`;
it must also show
`wing_readonly_operator_sequence_list_summary_surface_mismatches: []`;
it must also show
`wing_readonly_operator_sequence_list_summary_surface_match: true`;
it must also show
`wing_readonly_operator_sequence_list_summary_surface_readiness_closure_allowed: false`;
it must also show
`wing_readonly_operator_sequence_list_summary_surface_live_authority_granted: false`;
it must also show
`wing_readonly_operator_sequence_list_markdown_surface_required: true`;
it must also show
`wing_readonly_operator_sequence_list_markdown_surface_required_field_lines`;
it must also show
`wing_readonly_operator_sequence_list_markdown_surface_required_field_lines_expected_value`;
it must also show
`wing_readonly_operator_sequence_list_markdown_surface_required_field_lines_actual_value`;
it must also show
`wing_readonly_operator_sequence_list_markdown_surface_required_field_lines_match_contract: true`;
it must also show
`wing_readonly_operator_sequence_list_markdown_surface_missing_field_lines: []`;
it must also show
`wing_readonly_operator_sequence_list_markdown_surface_field_lines_rendered: true`;
it must also show
`wing_readonly_operator_sequence_list_markdown_surface_readiness_closure_allowed: false`;
it must also show
`wing_readonly_operator_sequence_list_markdown_surface_live_authority_granted: false`;
it must also show `operator_sequence_count_matches_contract: true`;
it must also show `operator_sequence_sha256`;
it must also show `actual_operator_sequence_sha256`;
it must also show `operator_sequence_sha256_matches_actual: true`;
it must also show `expected_operator_sequence_sha256`;
it must also show `actual_expected_operator_sequence_sha256`;
it must also show `operator_sequence_matches_contract: true`;
it must also show `operator_sequence_sha256_matches_contract: true`;
it must also show `declared_expected_operator_sequence_sha256_matches_contract: true`;
it must also show `declared_operator_sequence_matches_contract_flag: true`;
it must also show `declared_operator_sequence_matches_contract_flag_matches_actual: true`;
it must also show `declared_operator_sequence_sha256_matches_contract_flag: true`;
it must also show `declared_operator_sequence_sha256_matches_contract_flag_matches_actual: true`;
it must also show `declared_operator_sequence_count: 12`;
it must also show `declared_expected_operator_sequence_count: 12`;
it must also show `declared_operator_sequence_count_contract_flag: true`;
it must also show `declared_operator_sequence_count_matches_actual: true`;
it must also show `declared_operator_sequence_count_matches_contract: true`;
it must also show `declared_expected_operator_sequence_count_matches_contract: true`;
it must also show `declared_operator_sequence_count_contract_flag_matches_declared_count: true`;
it must also show `backend_port_prestart_gate` exactly
matching `expected_backend_port_prestart_gate`, with
`backend_port_prestart_gate.grants_live_authority: false` and
`backend_port_prestart_gate.must_pass_before_backend_wing_session: true`;
it must also show `backend_port_prestart_gate_matches_expected: true`;
it must also show `declared_backend_port_prestart_gate_matches_expected: true`;
it must also show `backend_port_prestart_gate.operator_sequence_step_index: 0`;
it must also show
`backend_port_prestart_gate.operator_sequence_step_must_be_first: true`;
it must also show
`backend_port_prestart_gate.runs_command: prestart_backend_port_check`;
it must also show `backend_port_prestart_gate.required_markers` exactly listing
`Before starting the intended backend/WING session`,
`prestart_backend_port_check`, and `unexpected listeners`;
the failed busy-port no-WING summary and no-WING handoff summary must both show
`backend_port_prestart_check_command` exactly matching
`PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick`;
the same summaries must also show `backend_port_prestart_check_command_text`
exactly matching
`PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick`;
the same summaries must also show `backend_port_prestart_check_required: true`
and `backend_port_prestart_check_must_pass_before_starting_backend: true`;
the same summaries must also show
`backend_port_prestart_gate_execution_allowed: false`,
`backend_port_prestart_gate_readiness_closure_allowed: false`, and
`backend_port_prestart_gate_live_authority_granted: false`;
the same summaries must also show `backend_port_prestart_check_failure_detail_fields` exactly listing
`listener_processes`, `listener_process_error`, `listener_confirmation_checklist`,
`preflight_blocked_until_listener_confirmed`,
`intended_backend_confirmation_required`, `safe_to_stop_listener_by_gate`,
`listener_confirmation_template_status`,
`listener_confirmation_template_path`,
`listener_confirmation_template_acceptance_status`,
`listener_confirmation_template_safe_to_stop_listener_by_gate`,
`listener_confirmation_template_live_authority_granted`,
`listener_confirmation_template_accepted_by_template`,
`listener_confirmation_template_operator_placeholder_fields`,
`listener_confirmation_template_execution_allowed_by_template`,
`listener_confirmation_template_readiness_closure_allowed_by_template`,
`listener_confirmation_template_required_edits`,
`backend_port_prestart_gate_execution_allowed`,
`backend_port_prestart_gate_readiness_closure_allowed`,
`backend_port_prestart_gate_live_authority_granted`, `fail_closed_reason`, and
`operator_next_step`;
if no-WING quick reports `environment_blocked_checks`, the no-WING summary must
also expose `environment_blocked_check_details` with
`local_dev_server_bind_denied`, `127.0.0.1:3000`, log paths, and
`operator_next_step`, and the gate remains failed until rerun in an environment
that can bind the frontend dev server;
if a backend listener is already present, the failed no-WING summary must
expose `listener_confirmation_template_path`,
`listener_confirmation_template_status: pending_operator_edit`,
`listener_confirmation_template_safe_to_stop_listener_by_gate: false`, and
`listener_confirmation_template_live_authority_granted: false`;
it must also show `listener_confirmation_template_accepted_by_template: false`;
it must also show `listener_confirmation_template_operator_placeholder_fields`
listing `operator_reviewed_at` and `operator_name`;
it must also show
`listener_confirmation_template_execution_allowed_by_template: false`;
it must also show
`listener_confirmation_template_readiness_closure_allowed_by_template: false`;
the generated template required edits must include
`set template_status to operator_reviewed_and_edited if using the generated
template` and `set accepted_by_template to true if using the generated
template`, and any template-derived accepted confirmation must replace
`REPLACE_WITH_*` operator placeholders, set
`template_status: operator_reviewed_and_edited`, and set
`accepted_by_template: true`;
it must also show `manual_review_required_before_readiness_update: true`;
it must also show `readiness_matrix_update_allowed_by_packet: false`;
it must also show `readiness_closure_allowed_by_packet: false`;
it must also show `read_only: true`;
it must also show `live_write: false`;
it must also show `mutates_console_state: false`;
it must also show `queries_console: false`;
it must also show `executes_tests: false`;
it must also show `operator_required: true`;
it must also show `expected_plan_required_for_real_run: true`;
it must also show `physical_preflight_requires_existing_backend_connection: true`;
it must also show `backend_connection_confirmation_step_index: 3`;
it must also show `expected_backend_connection_confirmation_step_index: 3`;
it must also show `operator_sequence_has_backend_connection_confirmation: true`;
it must also show `operator_sequence_backend_connection_confirmation_step_index: 3`;
it must also show
`operator_sequence_backend_connection_confirmation_before_capture: true`;
the `operator_sequence_backend_connection_confirmation_step` must include
`existing backend connection`, `intended WING Rack`, and
`read-only inspection`;
it must also show `operator_sequence_validate_readonly_preflight_inputs_step_index: 5`;
it must also show
`expected_operator_sequence_validate_readonly_preflight_inputs_step_index: 5`;
it must also show
`operator_sequence_validate_readonly_preflight_inputs_step_index_matches_contract: true`;
it must also show
`operator_sequence_validate_readonly_preflight_inputs_step: Run validate_readonly_preflight_inputs before capture; it must validate real-run inputs and expected-plan bindings without opening transport.`;
it must also show
`expected_operator_sequence_validate_readonly_preflight_inputs_step: Run validate_readonly_preflight_inputs before capture; it must validate real-run inputs and expected-plan bindings without opening transport.`;
it must also show
`operator_sequence_validate_readonly_preflight_inputs_step_matches_contract: true`;
it must also show `operator_sequence_capture_readonly_preflight_step_index: 6`;
it must also show
`expected_operator_sequence_capture_readonly_preflight_step_index: 6`;
it must also show
`operator_sequence_capture_readonly_preflight_step_index_matches_contract: true`;
it must also show
`operator_sequence_capture_readonly_preflight_step: Run capture_readonly_preflight.`;
it must also show
`expected_operator_sequence_capture_readonly_preflight_step: Run capture_readonly_preflight.`;
it must also show
`operator_sequence_capture_readonly_preflight_step_matches_contract: true`;
the no-WING handoff summary must also show the same
`backend_port_prestart_check_required: true` and
`backend_port_prestart_check_must_pass_before_starting_backend: true`;
it must also show `backend_port_prestart_summary_required: true`;
it must also show `backend_port_prestart_summary_path_placeholder` pointing
to the passed no-WING prestart `summary.json`;
it must also show `backend_port_prestart_summary_sha256_matches_plan: true`;
it must also show `backend_port_prestart_summary_sha256_available: true`;
it must also show
`backend_port_prestart_summary_sha256_available_matches_digest: true`;
it must also show `preview_command_prestart_summary` matching that path;
it must also show `preview_command_prestart_summary_matches_plan: true`;
it must also show `validate_inputs_command_prestart_summary` matching that path;
it must also show `validate_inputs_command_prestart_summary_matches_plan: true`;
it must also show `capture_command_prestart_summary` matching that path;
it must also show `capture_command_prestart_summary_matches_plan: true`;
the no-WING handoff summary must also show
`backend_port_prestart_summary_type_matches_plan: true`;
it must also show
`backend_port_prestart_summary_status_matches_plan: true`;
it must also show
`backend_port_prestart_summary_connection_policy_matches_plan: true`;
it must also show
`backend_port_prestart_summary_read_only_matches_plan: true`;
it must also show
`backend_port_prestart_summary_live_write_matches_plan: true`;
it must also show
`backend_port_prestart_summary_contacts_backend_matches_plan: true`;
it must also show
`backend_port_prestart_summary_contacts_wing_matches_plan: true`;
it must also show
`backend_port_prestart_summary_mutates_console_state_matches_plan: true`;
it must also show
`backend_port_prestart_summary_queries_console_matches_plan: true`;
it must also show
`backend_port_prestart_summary_executes_tests_matches_plan: true`;
it must also show `plan_safe_to_stop_listener_by_gate: false`;
it must also show `plan_live_authority_granted: false`;
it must also show
`plan_backend_port_prestart_gate_execution_allowed: false`;
it must also show
`plan_backend_port_prestart_gate_readiness_closure_allowed: false`;
it must also show
`plan_backend_port_prestart_gate_live_authority_granted: false`;
it must also show `backend_port_prestart_check_status_matches_plan: true`;
it must also show `safe_to_stop_listener_by_gate_matches_plan: true`;
it must also show `live_authority_granted_matches_plan: true`;
it must also show
`backend_port_prestart_gate_execution_allowed_matches_plan: true`;
it must also show
`backend_port_prestart_gate_readiness_closure_allowed_matches_plan: true`;
it must also show
`backend_port_prestart_gate_live_authority_granted_matches_plan: true`;
it must also show `declared_safe_to_stop_listener_by_gate_matches_plan: true`;
it must also show `declared_live_authority_granted_matches_plan: true`;
it must also show
`declared_backend_port_prestart_gate_execution_allowed_matches_plan: true`;
it must also show
`declared_backend_port_prestart_gate_readiness_closure_allowed_matches_plan: true`;
it must also show
`declared_backend_port_prestart_gate_live_authority_granted_matches_plan: true`;
it must also show `execution_allowed_by_packet: false`;
it must also show `live_authority_granted: false`;
it must also show `full_autonomous_ready: false`;
it must also show `next_manual_artifact_required: real wing_readonly_preflight.v1 summary.json`;
it must also show `expected_artifacts` exactly matching
`expected_artifact_globs` and `plan_expected_artifacts`;
it must also show `expected_artifacts_match_plan: true`,
`summary_glob_matches_plan: true`, and `event_log_glob_matches_plan: true`;
it must also show `preview_command_artifact_root_matches_packet: true`;
it must also show `preview_command_plan_output_matches_packet: true`;
it must also show `validate_inputs_command_uses_validate_run_inputs_only: true`;
it must also show `validate_inputs_command_uses_plan_only: false`;
it must also show `validate_inputs_command_artifact_root_matches_packet: true`;
it must also show `validate_inputs_command_expected_plan_matches_packet: true`;
it must also show `capture_command_artifact_root_matches_packet: true`;
it must also show `capture_command_expected_plan_matches_packet: true`;
it must also show `review_command_artifact_root_matches_packet: true`;
it must also show `decision_review_command_artifact_root_matches_packet: true`;
it must also show `decision_review_command_decision_root_matches_packet: true`;
it must also show `backend_port_prestart_check_operator_sequence_step_index: 0`;
it must also show `backend_port_prestart_check_operator_sequence_step`
including `prestart_backend_port_check` and `unexpected listeners`;
it must also show
`backend_port_prestart_check_must_be_first_operator_step: true`;
it must also show
`backend_port_prestart_check_operator_sequence_step_is_first: true`;
it must also show `mvp_blocker_anchor` matching the current first open blocker
row and `mvp_blocker_anchor_matches_current_first_open: true`;
the no-WING handoff summary must also show
`mvp_blocker_anchor_sha256_matches_current_first_open: true`;
the no-WING handoff summary must also show
`mvp_blocker_anchor.packet_does_not_close_blocker: true`;
the no-WING handoff summary must also show
`project_director_selected_action_anchor`,
`expected_project_director_selected_action_anchor`,
`project_director_selected_action_anchor_matches_current_directive: true`,
`project_director_selected_action_source: Current First Command`,
`expected_project_director_selected_action_source: Current First Command`,
`project_director_selected_action_source_matches_current_directive: true`,
`project_director_selected_action_sha256`,
`actual_project_director_selected_action_sha256`,
`expected_project_director_selected_action_sha256`,
`project_director_selected_action_sha256_available: true`,
`project_director_selected_action_sha256_available_matches_digest: true`,
`project_director_selected_action_sha256_matches_anchor: true`,
`project_director_selected_action_sha256_matches_text: true`,
`project_director_selected_action_sha256_matches_current_directive: true`,
`project_director_current_first_command_baseline_coverage_required: true`,
`project_director_current_first_command_baseline_coverage_passed: true`,
`project_director_current_first_command_baseline_coverage_marker_count`, and
`project_director_current_first_command_baseline_missing_markers: []`;
the no-WING handoff summary must also show
`project_director_selected_action_value_gate_self_check_required: true`;
it must also show
`project_director_selected_action_value_gate_self_check_required_markers`;
it must also show
`project_director_selected_action_value_gate_self_check_missing_markers: []`;
it must also show
`project_director_selected_action_value_gate_self_check_markers_present: true`;
it must also show
`project_director_selected_action_value_gate_self_check_readiness_closure_allowed: false`;
it must also show
`project_director_selected_action_value_gate_self_check_live_authority_granted: false`;
the no-WING handoff summary must also show
`project_director_selected_action_value_gate_self_check_markdown_surface_required: true`;
it must also show
`project_director_selected_action_value_gate_self_check_markdown_surface_required_field_lines`;
it must also show
`project_director_selected_action_value_gate_self_check_markdown_surface_missing_field_lines: []`;
it must also show
`project_director_selected_action_value_gate_self_check_markdown_surface_field_lines_rendered: true`;
it must also show
`project_director_selected_action_value_gate_self_check_markdown_surface_readiness_closure_allowed: false`;
it must also show
`project_director_selected_action_value_gate_self_check_markdown_surface_live_authority_granted: false`;
the no-WING handoff summary must also show
`operator_sequence_has_mvp_blocker_anchor_digest_review: true`;
the no-WING handoff summary must also show
`mvp_blocker_anchor_digest_review_step_index: 10`;
it must also show `expected_mvp_blocker_anchor_digest_review_step_index: 10`;
it must also show `mvp_blocker_anchor_digest_review_step_index_matches_contract: true`;
it must also show `mvp_blocker_anchor_digest_review_step_index_matches_operator_sequence: true`;
the no-WING handoff summary must also show
`operator_sequence_mvp_blocker_anchor_digest_review_step_index: 10`;
it must also show `expected_operator_sequence_mvp_blocker_anchor_digest_review_step_index: 10`;
it must also show `operator_sequence_mvp_blocker_anchor_digest_review_step_index_matches_contract: true`;
the digest review `operator_sequence_mvp_blocker_anchor_digest_review_step`
must include `mvp_blocker_anchor`, `mvp_blocker_anchor_sha256`, and
`current first open blocker`;
the no-WING handoff summary must also show
`operator_sequence_has_supervised_prestart_gate: true`;
the no-WING handoff summary must also show
`operator_sequence_supervised_prestart_gate_is_final: true`;
the no-WING handoff summary must also show
`operator_sequence_supervised_prestart_gate_step_index: 11`;
the no-WING handoff summary must also show
`expected_operator_sequence_supervised_prestart_gate_step_index: 11`;
the no-WING handoff summary must also show
`operator_sequence_supervised_prestart_gate_step_index_matches_contract: true`;
the final `operator_sequence_supervised_prestart_gate_step` must include
`accepted_decision_ready`, `valid accepted-candidate prestart trace`,
`valid accepted-candidate artifact-root trace`,
`valid accepted-candidate request-sequence trace`,
`expected-artifact-bound accepted source id`, and
`WING_SUPERVISED_FADER_GAIN_PROOF`;
the no-WING handoff summary must also show
`required_supervised_prestart_gate_markers` exactly listing
`accepted_decision_ready`, `valid accepted-candidate prestart trace`,
`valid accepted-candidate artifact-root trace`,
`valid accepted-candidate request-sequence trace`,
`expected-artifact-bound accepted source id`, and
`WING_SUPERVISED_FADER_GAIN_PROOF`;
the no-WING handoff summary must show `supervised_prestart_gate` exactly
matching `expected_supervised_prestart_gate`;
the no-WING handoff summary must also show
`supervised_prestart_gate_matches_expected: true`;
the no-WING handoff summary must also show
`declared_supervised_prestart_gate_matches_expected: true`;
the no-WING handoff summary must also show
`supervised_prestart_gate.grants_live_authority: false`;
the no-WING handoff summary must also show
`supervised_prestart_gate.operator_sequence_step_index: 11`;
the no-WING handoff summary must also show
`supervised_prestart_gate.operator_sequence_step_must_be_final: true`;
the no-WING handoff summary must also show top-level
`supervised_prestart_gate_required: true`,
top-level `supervised_prestart_gate_operator_sequence_step_index: 11`,
and top-level `supervised_prestart_gate_operator_sequence_step_must_be_final: true`;
the no-WING handoff summary must also show top-level
`supervised_prestart_gate_operator_sequence_step` and
`supervised_prestart_gate_operator_sequence_step_matches_summary: true`;
the no-WING handoff summary must also show top-level
`supervised_prestart_gate_required_markers`,
`expected_supervised_prestart_gate_required_markers`,
`expected_required_supervised_prestart_gate_markers`,
`required_supervised_prestart_gate_marker_count: 6`,
`expected_required_supervised_prestart_gate_marker_count: 6`,
`required_supervised_prestart_gate_marker_count_matches_contract: true`,
`required_supervised_prestart_gate_marker_count_matches_supervised_gate: true`,
`required_supervised_prestart_gate_markers_match_contract: true`,
`required_supervised_prestart_gate_markers_match_supervised_gate: true`,
`supervised_prestart_gate_required_marker_count: 6`,
`expected_supervised_prestart_gate_required_marker_count: 6`,
`supervised_prestart_gate_required_marker_count_matches_contract: true`,
`supervised_prestart_gate_required_marker_count_matches_summary: true`,
`supervised_prestart_gate_required_markers_match_contract: true`,
and `supervised_prestart_gate_required_markers_match_summary: true`;
the no-WING handoff summary must also show
`supervised_prestart_gate.requires_operator_decision_status: accepted_decision_ready`;
the no-WING handoff summary must also show
`supervised_prestart_gate.unlocks_runbook: Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md`;
the no-WING handoff summary must also show top-level
`supervised_prestart_gate_grants_live_authority: false`,
top-level `supervised_prestart_gate_requires_operator_decision_status: accepted_decision_ready`,
and top-level `supervised_prestart_gate_unlocks_runbook: Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md`;
the no-WING handoff summary must also show
`supervised_prestart_gate.requires_accepted_candidate_prestart_trace: true`; it must also show `supervised_prestart_gate.requires_accepted_candidate_artifact_root_trace: true`;
it must also show `supervised_prestart_gate.requires_accepted_candidate_request_sequence_trace: true`;
it must also show `supervised_prestart_gate.requires_accepted_candidate_expected_artifact_trace: true`;
it must also show top-level `supervised_prestart_gate_requires_accepted_candidate_prestart_trace: true`,
top-level `supervised_prestart_gate_requires_accepted_candidate_artifact_root_trace: true`,
top-level `supervised_prestart_gate_requires_accepted_candidate_request_sequence_trace: true`,
and top-level `supervised_prestart_gate_requires_accepted_candidate_expected_artifact_trace: true`;
the no-WING handoff summary must also show
`required_supervised_proof_plan_summary_fields`,
`expected_required_supervised_proof_plan_summary_fields`, and
`required_supervised_proof_plan_summary_fields_match_contract: true`;
the no-WING handoff summary must also show
`required_supervised_proof_plan_summary_fields_include_invalid_records: true`;
`required_supervised_proof_plan_summary_fields` must include
`readonly_preflight_review_decision_expected_candidate_prestart_source_ids`,
`readonly_preflight_review_decision_expected_candidate_prestart_contract_violations_by_source`,
`readonly_preflight_review_decision_candidate_artifact_root_contract_valid`,
`readonly_preflight_review_decision_candidate_artifact_roots`,
`readonly_preflight_review_decision_candidate_artifact_root_contract_violations_by_source`,
`readonly_preflight_review_decision_expected_candidate_artifact_roots`, and
`readonly_preflight_review_decision_expected_candidate_artifact_root_contract_violations_by_source`,
and `readonly_preflight_review_decision_invalid_records`;
the operator-readable no-WING `summary.md` proof plan section must render those
canonical helper fields;
the no-WING proof plan summary must also expose
`next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_summary_surface_self_check_required: true`,
`next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_summary_surface_required_fields`,
`next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_summary_surface_mismatches: []`,
`next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_summary_surface_match: true`,
`next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_summary_surface_readiness_closure_allowed: false`,
and `next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_summary_surface_live_authority_granted: false`;
the no-WING proof plan summary must also expose
`next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_summary_surface_self_check_required: true`,
`next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_summary_surface_required_fields`,
`next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_summary_surface_mismatches: []`,
`next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_summary_surface_match: true`,
`next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_summary_surface_readiness_closure_allowed: false`,
and `next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_summary_surface_live_authority_granted: false`;
the no-WING proof plan summary must also show
`operator_markdown_summary_required: true`,
`operator_markdown_summary_required_field_lines`,
`operator_markdown_summary_required_field_lines_missing_members: []`,
`operator_markdown_summary_required_field_lines_match_contract: true`,
`operator_markdown_summary_field_lines_rendered: true`, and
`operator_markdown_summary_missing_field_lines: []`;
the no-WING proof plan summary must also bind the operator markdown artifact
with `operator_markdown_summary_artifact: summary.md`,
`operator_markdown_summary_sha256`, and
`operator_markdown_summary_sha256_available: true`, and it must verify that
binding with `operator_markdown_summary_actual_sha256` and
`operator_markdown_summary_sha256_matches_actual: true`;
the no-WING `summary.json` and operator-readable `summary.md` must expose the
manifest summary surface with
`artifact_manifest_summary_surface_required: true`,
`artifact_manifest_summary_surface_missing_fields: []`,
`artifact_manifest_summary_surface_required_field_lines`,
`artifact_manifest_summary_surface_missing_field_lines: []`,
`artifact_manifest_summary_surface_field_lines_rendered: true`, and
`artifact_manifest_contract_id`,
`artifact_manifest_contract_required: true`,
`artifact_manifest_contract_fields`,
`artifact_manifest_contract_sha256`,
`artifact_manifest_contract_sha256_available: true`,
`artifact_manifest_contract_sha256_matches_expected: true`,
`artifact_manifest_contract_cross_surface_required: true`,
`artifact_manifest_contract_cross_surface_required_surfaces`,
`artifact_manifest_contract_cross_surface_required_fields`,
`artifact_manifest_contract_cross_surface_expected_match: true`,
`artifact_manifest_contract_cross_surface_readiness_closure_allowed: false`,
`artifact_manifest_contract_cross_surface_live_authority_granted: false`, and
`artifact_manifest_required_artifact_sha256_fields`;
the no-WING run artifact bundle must also write `artifact-manifest.json` with
`artifact_manifest_contract_id`,
`artifact_manifest_contract_required: true`,
`artifact_manifest_contract_fields`,
`artifact_manifest_contract_sha256`,
`artifact_manifest_contract_sha256_available: true`,
`artifact_manifest_contract_sha256_matches_expected: true`,
`artifact_manifest_contract_cross_surface_required: true`,
`artifact_manifest_contract_cross_surface_required_surfaces`,
`artifact_manifest_contract_cross_surface_required_fields`,
`artifact_manifest_contract_cross_surface_expected_match: true`,
`artifact_manifest_contract_cross_surface_match: true`,
`artifact_manifest_summary_json_contract_sha256_matches_manifest: true`,
`artifact_manifest_summary_md_contract_sha256_matches_manifest: true`,
`artifact_manifest_contract_cross_surface_mismatches: []`,
`artifact_manifest_contract_cross_surface_readiness_closure_allowed: false`,
`artifact_manifest_contract_cross_surface_live_authority_granted: false`,
`artifact_manifest_path: artifact-manifest.json`,
`artifact_manifest_artifact_count: 2`,
`artifact_manifest_required_artifact_count: 2`,
`artifact_manifest_artifact_count_matches_required: true`,
`artifact_manifest_required_artifact_ids`, `artifact_manifest_artifact_ids`,
`artifact_manifest_required_artifacts_present: true`,
`artifact_manifest_required_artifact_paths`, `artifact_manifest_artifact_paths`,
`artifact_manifest_required_artifact_paths_present: true`,
`artifact_manifest_required_artifact_sha256_available: true`,
`artifact_manifest_summary_json_sha256`,
`artifact_manifest_summary_json_sha256_available: true`,
`artifact_manifest_summary_md_sha256`,
`artifact_manifest_summary_md_sha256_available: true`,
`artifact_manifest_readiness_closure_allowed: false`, and
`artifact_manifest_live_authority_granted: false`;
Before opening a backend/WING WebSocket, the operator must also inspect
`artifact-manifest.json` in the same no-WING artifact directory as the passed
prestart `summary.json`; if any sidecar marker is missing or failing, do not
open the WING session. The sidecar checklist must show
`artifact_manifest_self_consistency_summary_surface_match: true`,
`artifact_manifest_self_consistency_summary_surface_json_mismatches: []`,
`artifact_manifest_self_consistency_summary_surface_markdown_mismatches: []`,
`artifact_manifest_self_consistency_summary_surface_mismatches: []`,
`artifact_manifest_self_consistency_summary_surface_readiness_closure_allowed: false`,
`artifact_manifest_self_consistency_summary_surface_live_authority_granted: false`,
`artifact_manifest_self_consistency_match: true`,
`artifact_manifest_self_consistency_mismatches: []`,
`artifact_manifest_contract_cross_surface_match: true`, and
`artifact_manifest_contract_cross_surface_mismatches: []`; the no-WING
operator-readable `summary.md` must also show
`wing_readonly_operator_sequence_list_markdown_surface_required_field_lines`,
`wing_readonly_operator_sequence_list_markdown_surface_required_field_lines_expected_value`,
`wing_readonly_operator_sequence_list_markdown_surface_required_field_lines_actual_value`,
`wing_readonly_operator_sequence_list_markdown_surface_required_field_lines_match_contract: true`,
`wing_readonly_operator_sequence_list_markdown_surface_missing_field_lines: []`
and
`wing_readonly_operator_sequence_list_markdown_surface_field_lines_rendered: true`.
These markers do not close readiness, approve live writes, or grant live authority;
the no-WING proof plan summary must also expose those canonical helper fields,
`required_supervised_proof_plan_summary_fields`, and
`missing_required_supervised_proof_plan_summary_fields`;
and `next_wing_proof_plan_schema` must fail closed with
`next_wing_proof_plan_required_summary_fields_missing` when any required
summary field is absent;
accepted review source ids must also have a valid accepted-candidate prestart
trace with `expected_artifacts`, `expected_summary_artifact`,
`expected_events_artifact`, `summary_path_matches_expected_artifact: true`, and
`events_path_matches_expected_artifact: true`; otherwise keep
`accepted_readonly_preflight_candidate_prestart_contract_required_before_supervised_proof`
and
`accepted_readonly_preflight_candidate_expected_artifact_trace_required_before_supervised_proof`
in the proof plan blockers.
Accepted review source ids must also have a valid accepted-candidate
`artifact_root` trace; otherwise keep
`accepted_readonly_preflight_candidate_artifact_root_required_before_supervised_proof`
in the proof plan blockers, and no-WING validation must fail with
`next_wing_proof_plan_steps_with_candidate_artifact_root_violations` if
physical proof steps are still present.
The recomputed
`readonly_preflight_review_decision_expected_candidate_artifact_root_contract_violations_by_source`
map must also stay `{}`; no-WING validation must fail with
`next_wing_proof_plan_steps_with_expected_candidate_artifact_root_violations`
if that expected map is non-empty while physical proof steps are still present.
The accepted-candidate prestart trace must also bind to a passed no-WING
prestart `summary.json` by path and SHA256, with
`backend_port_prestart_summary_required: true`,
`backend_port_prestart_summary_type: no_wing_release_check`,
`backend_port_prestart_summary_status: passed`,
`backend_port_prestart_summary_connection_policy: no_hardware_no_network_no_live_writes`,
`backend_port_prestart_summary_read_only: true`,
`backend_port_prestart_summary_live_write: false`,
`backend_port_prestart_summary_contacts_backend: false`,
`backend_port_prestart_summary_contacts_wing: false`,
`backend_port_prestart_summary_mutates_console_state: false`,
`backend_port_prestart_summary_queries_console: false`,
`backend_port_prestart_summary_executes_tests: false`, and
`backend_port_prestart_check_status: passed`.
The decision-review ledger/report must expose
`accepted_candidate_prestart_contract_violations_by_source`, and the supervised
proof plan summary must expose
`readonly_preflight_review_decision_candidate_prestart_contract_violations_by_source`.
Both maps must be `{}` before supervised proof steps are planned; a non-empty
map must trigger
`next_wing_proof_plan_steps_with_candidate_prestart_violations` if physical
proof steps are still present, and keeps
`accepted_readonly_preflight_candidate_prestart_contract_required_before_supervised_proof`
in force.
The recomputed
`readonly_preflight_review_decision_expected_candidate_prestart_contract_violations_by_source`
map must also stay `{}`; if it is non-empty while physical proof steps are
present, no-WING validation must fail with
`next_wing_proof_plan_steps_with_expected_candidate_prestart_violations`.

## Do Not Work On Instead

- New memory-trust chains.
- New queue-import chains.
- New result consumers.
- New broad report/read-model/frontend surfaces.
- Full autonomous mode.

These are blocked unless the change directly closes one row in the table above.
