# AUTO-MIXER Documentation Index

Date: 2026-05-18
Status: canonical documentation entrypoint

Start here before changing the project structure, runtime behavior, WING/OSC
paths, Product Layer, or cleanup policy.

## First Read

| Document | Purpose |
| --- | --- |
| `Docs/PROJECT_IMPLEMENTATION_PLAN.md` | Product goal, MVP phases, safety boundaries, schedule. |
| `Docs/MVP_WING_SUPERVISED.md` | Current supervised WING Rack MVP contract, blocker list, and stop-the-line rules. |
| `Docs/CORRECTION_METHODS_ROADMAP.md` | Roadmap for Gain/Trim, Pan, EQ, Comp, FX, Balance, routing, scenes, and output correction methods. |
| `Docs/PROJECT_STRUCTURE_MAP.md` | Canonical domain map and status labels. |
| `Docs/PROCESSING_METHODS_INVENTORY.md` | Canonical processing-method and pipeline-profile selection. |
| `Docs/memory_pipeline.md` | Agent Core + Project Layer + memory pipeline for Codex, Kimi, and Paperclip work. |
| `Docs/HEALTH_BASELINE_20260518.md` | Current post-cleanup test/build/audit baseline. |
| `Docs/PHASE1_SUPERVISED_WRITE_PROGRESS_20260519.md` | First Phase 1 supervised-write implementation slice and verification. |
| `Docs/reports/AUTOMIXER_HANDOFF_WING_VALIDATION_20260520.md` | Current cockpit/package handoff and narrow real WING supervised proposal proof. |
| `Docs/reports/AUTOMIXER_RELEASE_CANDIDATE_HANDOFF_20260520.md` | MVP cockpit handoff boundaries, package claim, and staging groups. |
| `Docs/reports/AUTOMIXER_PHASE1_WRITE_PATH_INVENTORY_20260520.md` | Current approved, quarantined, and proposal-only live-write surfaces. |
| `Docs/reports/AUTOMIXER_SLICE3_PROOF_READINESS_HANDOFF_20260520.md` | Proof-readiness contract before any physical `gain` validation. |
| `Docs/reports/AUTOMIXER_NEXT_ACTIONS_EXECUTION_20260521.md` | Current non-live execution report for package handoff, safety checks, and next gain-proof preconditions. |
| `Docs/reports/AUTOMIXER_GAIN_PROOF_PREFLIGHT_20260521.md` | Exact preflight, allowed flow, and success wording for the next physical `gain` proof. |
| `Docs/reports/AUTOMIXER_STAGING_GROUPS_20260521.md` | Dirty-worktree staging groups for release-candidate handoff. |
| `Docs/reports/AUTOMIXER_RC_STAGING_HANDOFF_20260521.md` | Release-candidate staging order, package evidence, and strict live boundary before the physical `gain` proof. |
| `Docs/reports/AUTOMIXER_PRE_LIVE_GAIN_REVIEW_20260521.md` | Pre-live code review, focused tests, and refreshed fake gain proof before any physical `gain` validation. |
| `Docs/reports/AUTOMIXER_OFFLINE_NO_WING_EXECUTION_20260522.md` | No-WING execution slice: safety audit refresh, fake gain proof, full checks, and package smoke. |
| `Docs/reports/AUTOMIXER_AUTOMATION_FOUNDATION_LAYER_20260527.md` | Automation foundation layer: decision traces, safety monitor, replay DTOs, idempotent supervised apply, and current validation evidence. |
| `Docs/adr/automation-foundation-layer-20260527.md` | ADR for the safe agentic-mixing foundation layer and its live-write boundary. |
| `Docs/reports/AUTOMIXER_WRITE_SURFACE_HARDENING_20260601.md` | Current write-surface inventory: canonical supervised fader/gain path, proposal-only surfaces, quarantined paths, and blocked mutation scripts. |
| `Docs/reports/AUTOMIXER_SAFETY_COVERAGE_MATRIX_20260601.md` | Safety proof matrix for live-capable action families and explicit hardware-proof gaps before any live scope expansion. |
| `Docs/reports/AUTOMIXER_WING_TIMEOUT_FAIL_CLOSED_PROOF_20260606.md` | Current timeout/connection-loss proof semantics for the supervised WING proposal runner. |
| `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_RUNNER_20260606.md` | Read-only WING preflight runner for connection/state/physical-readback artifacts before supervised writes. |
| `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_EVIDENCE_20260606.md` | Fail-closed evidence matrix for read-only WING preflight artifacts and manual readiness review. |
| `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_EVIDENCE_FAIL_CLOSED_20260606.md` | Mixed valid+invalid WING preflight artifacts fail closed before readiness-matrix review. |
| `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_EVENT_LOG_CROSSCHECK_20260606.md` | Read-only preflight summaries must match their event logs before manual readiness review. |
| `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_OPERATOR_CONFIRMATION_20260607.md` | Physical readback preflight artifacts must include operator-confirmed channels matching the queried channels. |
| `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_SESSION_CONFIRMATION_GATE_20260607.md` | Physical readback preflight artifacts must include explicit operator confirmation that the intended backend/WING session is ready for read-only inspection. |
| `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_SESSION_READY_REVIEW_CHECKLIST_20260607.md` | Operator review decisions must explicitly confirm the intended backend/WING session was ready before the read-only preflight candidate can gate supervised proof planning. |
| `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_DECISION_ARTIFACT_HASH_BINDING_20260607.md` | Operator review decisions pin the exact summary/event-log artifact bytes through SHA256 before they can gate supervised proof planning. |
| `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_REVIEW_RELEASE_GATE_20260607.md` | No-WING release summaries expose the preflight capture/review commands and fail closed if either tool is missing. |
| `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_MANUAL_REVIEW_CHECKLIST_20260607.md` | Offline artifact review exposes a required operator checklist before readiness updates. |
| `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_REVIEW_DECISION_LEDGER_20260607.md` | Operator review decisions are file-backed, validated, and still do not grant live authority. |
| `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_20260607.md` | Offline artifact review can write a pending operator decision template, and a focused decision-review CLI validates the edited JSON without granting live authority. |
| `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_REVIEW_DECISION_COLLISION_GATE_20260607.md` | Duplicate or conflicting operator decision records for the same read-only preflight candidate fail closed before supervised proof planning. |
| `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_HANDOFF_PACKET_20260607.md` | Offline operator packet captures the exact read-only preflight, review, decision-review, and validation sequence without contacting WING. |
| `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_HANDOFF_ARTIFACT_GATE_20260607.md` | No-WING release checks validate the generated handoff packet as a read-only JSON operator gate. |
| `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_OFFLINE_PLAN_PREVIEW_20260607.md` | Offline plan-only preview exposes the exact WING read-only preflight request sequence before any backend/WING session opens. |
| `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_PLAN_CONTEXT_GATE_20260607.md` | No-WING release checks require MVP blocker, manual review, fail-closed, review-command context, and a valid referenced plan for the offline preflight handoff. |
| `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_READBACK_IDENTITY_GATE_20260607.md` | Read-only preflight runner blocks physical-looking readbacks whose channel, value type, or address does not match the query. |
| `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_REQUEST_ID_CORRELATION_GATE_20260607.md` | Read-only preflight runner, backend responses, and evidence intake require request-id correlation for each queried WING readback. |
| `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_EXPECTED_PLAN_GATE_20260607.md` | Real read-only preflight capture validates against the offline plan before WebSocket and records a plan SHA256 in proof artifacts. |
| `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_EXPECTED_PLAN_REQUIRED_EVIDENCE_GATE_20260607.md` | Evidence intake requires proof that real read-only preflight runs required an offline expected plan. |
| `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_EXISTING_BACKEND_CONNECTION_EVIDENCE_GATE_20260607.md` | Evidence intake requires proof that real read-only preflight used an already-connected backend/WING session, not direct connect/disconnect authority. |
| `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_PLAN_POLICY_FLAGS_RELEASE_GATE_20260607.md` | No-WING release checks require the offline preflight plan to carry expected-plan and existing-backend policy flags. |
| `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_PLAN_SHA_FILE_EVIDENCE_GATE_20260607.md` | File-based evidence review verifies that the referenced offline preflight plan exists and matches the recorded SHA256. |
| `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_PLAN_SUMMARY_CONSISTENCY_GATE_20260607.md` | File-based evidence review verifies that the referenced offline preflight plan content matches the captured summary and request sequence. |
| `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_REQUIRE_EXPECTED_PLAN_20260607.md` | Real read-only preflight capture fails closed before WebSocket when `--expected-plan` is omitted. |
| `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_UNEXPECTED_RESPONSE_GATE_20260607.md` | Real read-only preflight capture fails closed when backend returns an unexpected response type for the request in flight. |
| `Docs/reports/AUTOMIXER_WING_MVP_DIRECTION_GUARD_SELF_REPORT_FILTER_20260607.md` | Direction guard drift scoring ignores its own policy warnings and generated evidence tables while preserving real drift detection. |
| `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_DECISION_GATED_PROOF_PLAN_20260607.md` | Next supervised fader/gain proof steps stay blocked until an accepted read-only preflight review decision exists. |
| `Docs/reports/AUTOMIXER_WING_PROOF_PLAN_REVIEW_DECISION_COLLISION_TRACE_20260607.md` | Next supervised proof plans expose duplicate/conflicting read-only preflight decision records that must be replaced before proof execution. |
| `Docs/reports/AUTOMIXER_WING_PROOF_PLAN_ACCEPTED_REVIEW_SOURCE_TRACE_20260607.md` | Next supervised proof plans preserve the accepted read-only preflight review source id that anchors physical proof planning. |
| `Docs/reports/AUTOMIXER_WING_SUPERVISED_PROOF_OPERATOR_CONFIRMATION_20260607.md` | Physical supervised WING proof commands require matching operator-confirmed channel, value type, and delta before `--execute`. |
| `Docs/reports/AUTOMIXER_WING_SUPERVISED_PROOF_PHYSICAL_READBACK_ONLY_20260607.md` | Physical supervised WING proof requires physical pre-read, apply readback, and rollback readback instead of cache evidence. |
| `Docs/reports/AUTOMIXER_SUPERVISED_PROOF_EVALUATOR_PHYSICAL_STATUS_GATE_20260607.md` | Downstream supervised proof evaluation blocks physical-looking artifacts whose pre-read, apply, or rollback status is cache-backed. |
| `Docs/reports/AUTOMIXER_WING_READONLY_PREFLIGHT_READINESS_BRIDGE_20260606.md` | Read-only preflight evidence overlay inside autonomy promotion readiness; manual review candidate only. |
| `Docs/reports/AUTOMIXER_SUPERVISED_WING_PROOF_READONLY_PREFLIGHT_GATE_20260606.md` | Next supervised WING proof plan is blocked until clean read-only preflight evidence exists. |
| `Docs/reports/AUTOMIXER_READ_SURFACE_PHYSICAL_READBACK_CAPTURE_PACKET_20260607.md` | Offline operator packet renders the next read-surface physical-readback capture target without querying or mutating a console. |
| `Docs/reports/AUTOMIXER_READ_SURFACE_CAPTURE_PACKET_ARTIFACT_GATE_20260607.md` | No-WING release checks validate the generated capture-packet JSON as a read-only operator gate. |
| `Docs/reports/AUTOMIXER_READ_SURFACE_CAPTURE_PACKET_TARGET_MANIFEST_20260607.md` | Capture packets list every remaining physical-readback target while keeping the handoff non-executing. |
| `Docs/reports/AUTOMIXER_READ_SURFACE_PHYSICAL_READBACK_CAPTURE_TEMPLATE_20260607.md` | Offline pending artifact template tells operators which physical-readback fields to fill while remaining invalid as proof and non-live. |
| `Docs/reports/AUTOMIXER_READ_SURFACE_CAPTURE_TEMPLATE_CONTRACT_SKELETON_20260607.md` | Pending capture templates include contract-derived payload skeletons while remaining invalid as proof and non-live. |
| `Docs/reports/AUTOMIXER_READ_SURFACE_ARTIFACT_REVIEW_QUEUE_ARTIFACT_GATE_20260607.md` | No-WING release checks validate the generated artifact-review queue JSON before acceptance-template generation. |
| `Docs/reports/AUTOMIXER_READ_SURFACE_REVIEW_QUEUE_SOURCE_LINK_20260607.md` | Review queues preserve physical-readback artifact source ids before operator acceptance templates are generated. |
| `Docs/reports/AUTOMIXER_READ_SURFACE_PHYSICAL_READBACK_ACCEPTANCE_TEMPLATE_20260607.md` | Offline operator template renders pending acceptance records for valid read-surface physical-readback candidates without closing fixtures or granting live authority. |
| `Docs/reports/AUTOMIXER_READ_SURFACE_ACCEPTANCE_TEMPLATE_ARTIFACT_GATE_20260607.md` | No-WING release checks validate the generated acceptance-template JSON as a read-only operator gate. |
| `Docs/reports/AUTOMIXER_READ_SURFACE_ACCEPTANCE_CANDIDATE_TRACE_20260607.md` | Acceptance records preserve and validate the review-queue candidate trace before any fixture-closure decision. |
| `Docs/reports/AUTOMIXER_READ_SURFACE_FIXTURE_CLOSURE_PLAN_ARTIFACT_GATE_20260607.md` | No-WING release checks validate the generated fixture-closure plan JSON while keeping fixture updates manual-only. |
| `Docs/reports/AUTOMIXER_WING_SUPERVISED_MVP_RAILS_20260606.md` | MVP contract, blocker table, runbooks, progress updater, and Project Director Agent. |
| `Docs/reports/AUTOMIXER_WING_SUPERVISED_RUNBOOK_PRESTART_TRACE_GATE_20260612.md` | Supervised fader/gain proof runbook and read-only handoff packet require a valid accepted-candidate prestart trace before moving from accepted review decision to supervised proof. |
| `Docs/reports/AUTOMIXER_WING_READONLY_RUNBOOK_FINAL_PRESTART_GATE_20260612.md` | Read-only preflight runbook requires the final handoff operator-sequence step to carry the accepted-candidate prestart gate before supervised fader/gain proof. |
| `Docs/reports/AUTOMIXER_WING_HANDOFF_STRUCTURED_PRESTART_GATE_20260612.md` | WING read-only handoff packet carries a structured `supervised_prestart_gate` contract before supervised fader/gain proof. |
| `Docs/reports/AUTOMIXER_WING_HANDOFF_STRUCTURED_GATE_SUMMARY_20260612.md` | No-WING release summaries expose `supervised_prestart_gate` and `expected_supervised_prestart_gate` for reviewer inspection. |
| `Docs/reports/AUTOMIXER_WING_READONLY_RUNBOOK_STRUCTURED_GATE_20260612.md` | Read-only preflight runbook requires `supervised_prestart_gate` to match `expected_supervised_prestart_gate` without granting live authority. |
| `Docs/reports/AUTOMIXER_WING_READONLY_HANDOFF_STRUCTURED_PRESTART_GATE_20260612.md` | WING read-only handoff packet carries structured `backend_port_prestart_gate` and `expected_backend_port_prestart_gate` contracts before any backend/WING session. |
| `Docs/reports/AUTOMIXER_WING_READONLY_HANDOFF_MVP_BLOCKER_ANCHOR_20260612.md` | WING read-only handoff packets carry and validate the current first-open MVP blocker anchor before supervised proof planning. |
| `Docs/reports/AUTOMIXER_WING_READONLY_HANDOFF_MVP_BLOCKER_ANCHOR_DIGEST_20260612.md` | WING read-only handoff packets carry a SHA256 digest for the current first-open MVP blocker anchor. |
| `Docs/reports/AUTOMIXER_WING_HANDOFF_ANCHOR_DIGEST_OPERATOR_REVIEW_STEP_20260612.md` | WING read-only handoff packets require an operator-sequence step to review the current blocker anchor and SHA256 digest before supervised proof planning. |
| `Docs/reports/AUTOMIXER_WING_MVP_BLOCKER_BACKEND_PRESTART_NEXT_ACTION_20260612.md` | WING MVP blocker table and generated latest progress carry the structured backend-port prestart gate in their next-action text. |
| `Docs/reports/AUTOMIXER_WING_MVP_BACKEND_PRESTART_FIRST_STEP_NEXT_ACTION_20260612.md` | WING MVP blocker table, generated progress, and autoloop docs require the backend-port prestart check to be the first operator step. |
| `Docs/reports/AUTOMIXER_WING_MVP_BACKEND_PRESTART_GATE_STEP_CONTRACT_NEXT_ACTION_20260612.md` | WING MVP blocker table, MVP doc, generated progress, and autoloop docs require the structured backend prestart gate to prove the first operator step. |
| `Docs/reports/AUTOMIXER_WING_MVP_BLOCKER_HANDOFF_ANCHOR_NEXT_ACTION_20260612.md` | WING MVP blocker table and generated latest progress require the handoff `mvp_blocker_anchor` to match the current first open row. |
| `Docs/reports/AUTOMIXER_WING_MVP_BLOCKER_HANDOFF_ANCHOR_DIGEST_NEXT_ACTION_20260612.md` | WING MVP blocker table and generated latest progress require the handoff `mvp_blocker_anchor_sha256` digest to match the current first open row. |
| `Docs/reports/AUTOMIXER_WING_MVP_BLOCKER_ANCHOR_DOES_NOT_CLOSE_NEXT_ACTION_20260612.md` | WING MVP blocker table and generated latest progress require the handoff blocker anchor to state that offline packets do not close the blocker. |
| `Docs/reports/AUTOMIXER_WING_MVP_BLOCKER_HANDOFF_DIGEST_REVIEW_NEXT_ACTION_20260612.md` | WING MVP blocker table and generated latest progress require the handoff operator sequence to include the anchor digest review step. |
| `Docs/reports/AUTOMIXER_WING_MVP_BLOCKER_DIGEST_REVIEW_STEP_MARKERS_20260612.md` | WING MVP blocker table, generated progress, and autoloop docs require the anchor digest review step to name the anchor, digest, and current first blocker. |
| `Docs/reports/AUTOMIXER_WING_MVP_BLOCKER_SUPERVISED_GATE_NEXT_ACTION_20260612.md` | WING MVP blocker table and generated latest progress require the final supervised prestart gate to be present and structured before supervised proof planning. |
| `Docs/reports/AUTOMIXER_WING_MVP_BLOCKER_SUPERVISED_TRACE_GATE_NEXT_ACTION_20260612.md` | WING MVP blocker table and generated latest progress require the supervised prestart gate to require a valid accepted-candidate prestart trace. |
| `Docs/reports/AUTOMIXER_WING_MVP_BLOCKER_SUPERVISED_NO_LIVE_AUTHORITY_NEXT_ACTION_20260612.md` | WING MVP blocker table and generated latest progress require the supervised prestart gate to show it grants no live authority. |
| `Docs/reports/AUTOMIXER_WING_MVP_BLOCKER_SUPERVISED_STEP_MARKERS_NEXT_ACTION_20260612.md` | WING MVP blocker table and generated latest progress require the final supervised operator step to carry the accepted-decision, prestart-trace, and proof-runbook markers. |
| `Docs/reports/AUTOMIXER_WING_MVP_SUPERVISED_GATE_FINAL_STEP_CONTRACT_NEXT_ACTION_20260612.md` | WING MVP blocker table, read-only runbook, MVP doc, generated progress, and autoloop docs require the structured supervised gate to prove the final operator step. |
| `Docs/reports/AUTOMIXER_WING_MVP_SUPERVISED_GATE_DECISION_RUNBOOK_CONTRACT_NEXT_ACTION_20260612.md` | WING MVP blocker table, read-only runbook, MVP doc, generated progress, and autoloop docs require the structured supervised gate to stay tied to an accepted decision and fader/gain proof runbook. |
| `Docs/reports/AUTOMIXER_WING_MVP_SUPERVISED_GATE_REQUIRED_MARKERS_NEXT_ACTION_20260612.md` | WING MVP blocker table, read-only runbook, MVP doc, generated progress, and autoloop docs require the structured supervised gate marker list before proof planning. |
| `Docs/reports/AUTOMIXER_WING_MVP_OPERATOR_SEQUENCE_STEP_INDEXES_NEXT_ACTION_20260612.md` | WING MVP blocker table, read-only runbook, MVP doc, generated progress, and autoloop docs require explicit `operator_sequence_mvp_blocker_anchor_digest_review_step_index` and `operator_sequence_supervised_prestart_gate_step_index` fields before proof planning. |
| `Docs/reports/AUTOMIXER_WING_MVP_OPERATOR_SEQUENCE_COUNT_NEXT_ACTION_20260612.md` | WING MVP blocker table, read-only runbook, MVP doc, generated progress, and autoloop docs require `operator_sequence_count: 10` before proof planning. |
| `Docs/reports/AUTOMIXER_WING_MVP_BACKEND_PRESTART_STEP_TEXT_NEXT_ACTION_20260612.md` | WING MVP blocker table, read-only runbook, MVP doc, generated progress, and autoloop docs require `backend_port_prestart_check_operator_sequence_step` to include the backend-port check and unexpected-listener markers. |
| `Docs/reports/AUTOMIXER_WING_MVP_BACKEND_PRESTART_RUNS_COMMAND_NEXT_ACTION_20260612.md` | WING MVP blocker table, read-only runbook, MVP doc, generated progress, and autoloop docs require `backend_port_prestart_gate.runs_command: prestart_backend_port_check` before proof planning. |
| `Docs/reports/AUTOMIXER_WING_MVP_BACKEND_PRESTART_REQUIRED_MARKERS_NEXT_ACTION_20260612.md` | WING MVP blocker table, read-only runbook, MVP doc, generated progress, and autoloop docs require `backend_port_prestart_gate.required_markers` to list the backend/WING boundary, prestart command, and unexpected-listener markers. |
| `Docs/reports/AUTOMIXER_WING_MVP_BACKEND_PRESTART_CHECK_COMMAND_NEXT_ACTION_20260612.md` | WING MVP blocker table, read-only runbook, MVP doc, generated progress, and autoloop docs require `backend_port_prestart_check_command` to match the no-WING release check command. |
| `Docs/reports/AUTOMIXER_WING_MVP_BACKEND_PRESTART_FAILURE_DETAIL_FIELDS_NEXT_ACTION_20260612.md` | WING MVP blocker table, read-only runbook, MVP doc, generated progress, and autoloop docs require `backend_port_prestart_check_failure_detail_fields` to list fail-closed listener-review fields. |
| `Docs/reports/AUTOMIXER_WING_MVP_MANUAL_REVIEW_BEFORE_READINESS_NEXT_ACTION_20260612.md` | WING MVP blocker table, read-only runbook, MVP doc, generated progress, and autoloop docs require `manual_review_required_before_readiness_update: true` before readiness updates. |
| `Docs/reports/AUTOMIXER_WING_MVP_READINESS_UPDATE_DISALLOWED_BY_PACKET_NEXT_ACTION_20260612.md` | WING MVP blocker table, read-only runbook, MVP doc, generated progress, and autoloop docs require `readiness_matrix_update_allowed_by_packet: false` before readiness updates. |
| `Docs/reports/AUTOMIXER_WING_MVP_READINESS_CLOSURE_DISALLOWED_BY_PACKET_NEXT_ACTION_20260612.md` | WING MVP blocker table, read-only runbook, MVP doc, generated progress, and autoloop docs require `readiness_closure_allowed_by_packet: false` before readiness closure. |
| `Docs/reports/AUTOMIXER_WING_MVP_READ_ONLY_REQUIRED_BY_PACKET_NEXT_ACTION_20260613.md` | WING MVP blocker table, read-only runbook, MVP doc, generated progress, and autoloop docs require `read_only: true` before supervised proof planning. |
| `Docs/reports/AUTOMIXER_WING_MVP_LIVE_WRITE_DISALLOWED_BY_PACKET_NEXT_ACTION_20260613.md` | WING MVP blocker table, read-only runbook, MVP doc, generated progress, and autoloop docs require `live_write: false` before supervised proof planning. |
| `Docs/reports/AUTOMIXER_WING_MVP_CONSOLE_MUTATION_DISALLOWED_BY_PACKET_NEXT_ACTION_20260613.md` | WING MVP blocker table, read-only runbook, MVP doc, generated progress, and autoloop docs require `mutates_console_state: false` before supervised proof planning. |
| `Docs/reports/AUTOMIXER_WING_MVP_CONSOLE_QUERY_DISALLOWED_BY_PACKET_NEXT_ACTION_20260613.md` | WING MVP blocker table, read-only runbook, MVP doc, generated progress, and autoloop docs require `queries_console: false` before supervised proof planning. |
| `Docs/reports/AUTOMIXER_WING_MVP_TEST_EXECUTION_DISALLOWED_BY_PACKET_NEXT_ACTION_20260613.md` | WING MVP blocker table, read-only runbook, MVP doc, generated progress, and autoloop docs require `executes_tests: false` before supervised proof planning. |
| `Docs/reports/AUTOMIXER_WING_MVP_OPERATOR_REQUIRED_BY_PACKET_NEXT_ACTION_20260613.md` | WING MVP blocker table, read-only runbook, MVP doc, generated progress, and autoloop docs require `operator_required: true` before supervised proof planning. |
| `Docs/reports/AUTOMIXER_WING_MVP_EXPECTED_PLAN_REQUIRED_BY_PACKET_NEXT_ACTION_20260613.md` | WING MVP blocker table, read-only runbook, MVP doc, generated progress, and autoloop docs require `expected_plan_required_for_real_run: true` before supervised proof planning. |
| `Docs/reports/AUTOMIXER_WING_MVP_EXISTING_BACKEND_REQUIRED_BY_PACKET_NEXT_ACTION_20260613.md` | WING MVP blocker table, read-only runbook, MVP doc, generated progress, and autoloop docs require `physical_preflight_requires_existing_backend_connection: true` before supervised proof planning. |
| `Docs/reports/AUTOMIXER_WING_MVP_BACKEND_PRESTART_CHECK_REQUIRED_BY_PACKET_NEXT_ACTION_20260613.md` | WING MVP blocker table, read-only runbook, MVP doc, generated progress, and autoloop docs require `backend_port_prestart_check_required: true` before supervised proof planning. |
| `Docs/reports/AUTOMIXER_WING_MVP_BACKEND_PRESTART_MUST_PASS_BY_PACKET_NEXT_ACTION_20260613.md` | WING MVP blocker table, read-only runbook, MVP doc, generated progress, and autoloop docs require `backend_port_prestart_check_must_pass_before_starting_backend: true` before supervised proof planning. |
| `Docs/reports/AUTOMIXER_WING_MVP_EXECUTION_DISALLOWED_BY_PACKET_NEXT_ACTION_20260612.md` | WING MVP blocker table, read-only runbook, MVP doc, generated progress, and autoloop docs require `execution_allowed_by_packet: false` before proof execution. |
| `Docs/reports/AUTOMIXER_WING_MVP_LIVE_AUTHORITY_DISALLOWED_BY_PACKET_NEXT_ACTION_20260613.md` | WING MVP blocker table, read-only runbook, MVP doc, generated progress, and autoloop docs require `live_authority_granted: false` before supervised proof planning. |
| `Docs/reports/AUTOMIXER_WING_MVP_FULL_AUTONOMOUS_DISALLOWED_BY_PACKET_NEXT_ACTION_20260613.md` | WING MVP blocker table, read-only runbook, MVP doc, generated progress, and autoloop docs require `full_autonomous_ready: false` before supervised proof planning. |
| `Docs/reports/AUTOMIXER_WING_MVP_NEXT_MANUAL_ARTIFACT_REQUIRED_BY_PACKET_NEXT_ACTION_20260613.md` | Handoff packet Markdown, no-WING release summary, WING MVP blocker table, read-only runbook, MVP doc, generated progress, and autoloop docs require `next_manual_artifact_required: real wing_readonly_preflight.v1 summary.json` before supervised proof planning. |
| `Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_EXPECTED_ARTIFACTS_CONTRACT_20260613.md` | WING handoff packet and no-WING release summary bind `expected_artifacts` to the named summary/events globs and expected plan artifact list. |
| `Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_EXPECTED_ARTIFACTS_NEXT_ACTION_20260613.md` | WING MVP blocker table, read-only runbook, MVP doc, generated progress, and autoloop docs require `expected_artifacts_match_plan: true`, `summary_glob_matches_plan: true`, and `event_log_glob_matches_plan: true` before supervised proof planning. |
| `Docs/reports/AUTOMIXER_WING_MVP_READONLY_ARTIFACT_CONTRACT_CONSTANTS_20260613.md` | WING read-only preflight runner, handoff packet, no-WING release check, evidence intake, supervised proof plan, and focused tests use shared artifact identity constants for `wing_readonly_preflight.v1`, `summary.json`, and `events.jsonl`. |
| `Docs/reports/AUTOMIXER_WING_MVP_DECISION_REVIEW_PRESTART_READY_GATE_20260613.md` | WING read-only preflight decision review reports `accepted_decision_ready` only after accepted source ids also have a valid accepted-candidate backend prestart trace. |
| `Docs/reports/AUTOMIXER_WING_MVP_DECISION_REVIEW_PRESTART_SUMMARY_FIELDS_20260614.md` | WING read-only decision review reports now render the accepted-candidate no-WING prestart summary binding and no-contact fields. |
| `Docs/reports/AUTOMIXER_WING_MVP_DECISION_REVIEW_PRESTART_BOOL_RENDERING_20260614.md` | WING read-only decision review reports now keep missing candidate-prestart booleans empty instead of rendering them as safe `false`. |
| `Docs/reports/AUTOMIXER_WING_MVP_DIRECTION_GUARD_SKIP_SELF_REPORTS_20260614.md` | Direction guard now skips its own generated reports so derived diagnostics cannot create false mixed evidence. |
| `Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_PRESTART_SUMMARY_SHA_BINDING_20260614.md` | WING read-only handoff packets now bind an existing no-WING prestart summary by SHA256 and no-WING validation fails closed on packet/plan digest mismatch. |
| `Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_PRESTART_SUMMARY_SHA_AVAILABILITY_20260614.md` | no-WING handoff validation now fails closed when the prestart summary SHA availability flag contradicts the packet digest. |
| `Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_OPERATOR_SEQUENCE_CARDINALITY_20260614.md` | no-WING handoff validation now fails closed when the operator sequence count or supervised prestart gate index drifts from the contract. |
| `Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_ANCHOR_DIGEST_REVIEW_STEP_INDEX_20260614.md` | no-WING handoff validation now fails closed unless `expected_operator_sequence_mvp_blocker_anchor_digest_review_step_index` and `operator_sequence_mvp_blocker_anchor_digest_review_step_index_matches_contract` prove the anchor digest review operator step is at its expected contract index. |
| `Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_DECLARED_ANCHOR_DIGEST_REVIEW_INDEX_20260614.md` | no-WING handoff validation now fails closed unless `expected_mvp_blocker_anchor_digest_review_step_index`, `mvp_blocker_anchor_digest_review_step_index_matches_contract`, and `mvp_blocker_anchor_digest_review_step_index_matches_operator_sequence` prove the packet-declared anchor digest review index matches the contract and actual operator sequence. |
| `Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_DECLARED_OPERATOR_SEQUENCE_COUNT_20260614.md` | no-WING handoff validation now fails closed unless `declared_operator_sequence_count`, `declared_expected_operator_sequence_count`, `declared_operator_sequence_count_contract_flag`, `declared_operator_sequence_count_matches_actual`, `declared_operator_sequence_count_matches_contract`, `declared_expected_operator_sequence_count_matches_contract`, and `declared_operator_sequence_count_contract_flag_matches_declared_count` prove packet-declared operator sequence count metadata matches the actual sequence and contract. |
| `Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_OPERATOR_SEQUENCE_SHA256_20260614.md` | no-WING handoff validation now fails closed unless `operator_sequence_sha256`, `actual_operator_sequence_sha256`, and `operator_sequence_sha256_matches_actual` prove the packet-declared digest matches the actual ordered operator sequence. |
| `Docs/reports/AUTOMIXER_WING_MVP_CURRENT_EVIDENCE_LATEST_REPORTS_20260613.md` | The current first WING MVP blocker evidence cites the latest read-only artifact contract and decision prestart ready gate reports for Project Director context. |
| `Docs/reports/AUTOMIXER_WING_MVP_CURRENT_EVIDENCE_LISTENER_PROBE_ERROR_REPORT_20260613.md` | The current first WING MVP blocker evidence cites the backend-port listener-probe error field report for Project Director context. |
| `Docs/reports/AUTOMIXER_WING_MVP_NOWING_QUICK_BUSY_PORT_FAIL_CLOSED_20260613.md` | The current no-WING quick gate failed closed on a busy backend port and requires operator confirmation before WING read-only preflight. |
| `Docs/reports/AUTOMIXER_WING_MVP_BACKEND_LISTENER_CONFIRMATION_ARTIFACT_20260613.md` | The no-WING backend-port prestart gate accepts only a matching operator confirmation JSON for an intended backend listener. |
| `Docs/reports/AUTOMIXER_WING_MVP_BACKEND_LISTENER_CONFIRMATION_TEMPLATE_20260613.md` | Busy backend-port fail-closed summaries now write a pending operator confirmation template without granting live authority. |
| `Docs/reports/AUTOMIXER_WING_MVP_BACKEND_LISTENER_TEMPLATE_EDIT_GATE_20260613.md` | Template-derived backend-listener confirmations fail closed until placeholders and pending-template markers are replaced by operator review. |
| `Docs/reports/AUTOMIXER_WING_MVP_BACKEND_LISTENER_TEMPLATE_EDIT_REQUIREMENT_CANON_20260613.md` | The WING MVP blocker table, Project Director, generated progress report, MVP doc, and autoloop docs now carry the backend-listener template edit gate. |
| `Docs/reports/AUTOMIXER_WING_MVP_BACKEND_LISTENER_TEMPLATE_PENDING_SUMMARY_FIELDS_20260613.md` | Busy backend-port summaries now expose generated-template accepted-by-template and operator placeholder markers directly. |
| `Docs/reports/AUTOMIXER_WING_MVP_BACKEND_LISTENER_TEMPLATE_AUTHORITY_SUMMARY_FIELDS_20260613.md` | Busy backend-port summaries now expose generated-template execution and readiness-closure denial markers directly. |
| `Docs/reports/AUTOMIXER_WING_MVP_BACKEND_PRESTART_TEMPLATE_FAILURE_DETAIL_FIELDS_20260613.md` | Backend-port prestart failure-detail contracts now require generated-template pending and authority-denial fields. |
| `Docs/reports/AUTOMIXER_WING_MVP_BACKEND_PRESTART_TEMPLATE_REVIEW_FIELDS_20260613.md` | Backend-port prestart failure-detail contracts now require generated-template review fields for safe-to-stop, live authority, placeholders, and required edits. |
| `Docs/reports/AUTOMIXER_WING_MVP_BUSY_PORT_PRESTART_DETAIL_FIELDS_SUMMARY_20260613.md` | Failed busy-port no-WING summaries now surface the shared backend-port prestart failure-detail field list in JSON and Markdown. |
| `Docs/reports/AUTOMIXER_WING_MVP_BUSY_PORT_PRESTART_COMMAND_SUMMARY_20260613.md` | The failed busy-port no-WING summary and no-WING handoff summary must both show the shared backend-port prestart check command in JSON and Markdown. |
| `Docs/reports/AUTOMIXER_WING_MVP_BUSY_PORT_PRESTART_REQUIRED_FLAGS_SUMMARY_20260613.md` | The failed busy-port no-WING summary and no-WING handoff summary must both show the backend-port prestart required and must-pass flags. |
| `Docs/reports/AUTOMIXER_WING_MVP_BUSY_PORT_PRESTART_GATE_AUTHORITY_SUMMARY_20260613.md` | Failed busy-port no-WING summaries now surface `backend_port_prestart_gate_execution_allowed: false`, `backend_port_prestart_gate_readiness_closure_allowed: false`, and `backend_port_prestart_gate_live_authority_granted: false` in JSON and Markdown. |
| `Docs/reports/AUTOMIXER_WING_MVP_BACKEND_LISTENER_PENDING_SUMMARY_FIELDS_CANON_20260613.md` | The WING MVP blocker table, Project Director, generated progress report, MVP doc, and autoloop docs now require the pending listener-template summary fields. |
| `Docs/reports/AUTOMIXER_WING_MVP_BACKEND_PRESTART_LISTENER_PROBE_ERROR_FIELD_20260613.md` | Backend-port prestart failure-detail contracts now require `listener_process_error` so owner-probe failures remain visible before WING read-only preflight. |
| `Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_PRESTART_SUMMARY_PLACEHOLDER_20260613.md` | WING read-only handoff packets now carry the no-WING prestart summary placeholder and pass it to plan/capture commands. |
| `Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_PRESTART_SUMMARY_COMMAND_BINDING_20260614.md` | WING read-only handoff summaries now prove preview and capture commands use the validated plan prestart summary path. |
| `Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_BACKEND_PRESTART_COMMAND_SUMMARY_20260614.md` | WING read-only handoff summaries now expose the packet-declared backend-port prestart command beside the extracted command-map value. |
| `Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_DENIAL_FLAG_SUMMARY_TRUTH_20260614.md` | Failed WING read-only handoff summaries now preserve unsafe packet denial/live-authority flag values instead of hardcoding safe false values. |
| `Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_SAFETY_FLAG_SUMMARY_TRUTH_20260614.md` | Failed WING read-only handoff summaries now preserve unsafe packet safety flags instead of hardcoding safe false values. |
| `Docs/reports/AUTOMIXER_WING_MVP_PRESTART_SUMMARY_CONTACT_BOUNDARY_20260613.md` | Accepted WING read-only prestart traces now require bound no-WING summary `contacts_backend`, `queries_console`, and `executes_tests` fields to be false. |
| `Docs/reports/AUTOMIXER_WING_MVP_PRESTART_FAIL_CLOSED_CONDITION_SYNC_20260613.md` | The no-WING release-check validator now matches the WING readonly runner/evidence prestart summary fail-closed conditions. |
| `Docs/reports/AUTOMIXER_WING_MVP_FAIL_CLOSED_CONTRACT_SOURCE_20260613.md` | WING readonly expected-plan fail-closed conditions now have one shared contract source consumed by runner, evidence, and release-check tooling. |
| `Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_GATE_CONTRACT_SOURCE_20260613.md` | WING read-only handoff structured prestart gates now have one shared contract source consumed by the handoff builder and no-WING validator. |
| `Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_SAFETY_CONTRACT_SOURCE_20260613.md` | WING read-only handoff forbidden request types and not-closed blocker ids now have one shared contract source. |
| `Docs/reports/AUTOMIXER_WING_MVP_REVIEW_CHECKLIST_CONTRACT_SOURCE_20260613.md` | WING read-only manual-review checklist ids now have one shared contract source consumed by review, handoff, and no-WING tooling. |
| `Docs/reports/AUTOMIXER_WING_MVP_REVIEW_DECISION_TEMPLATE_CONTRACT_SOURCE_20260613.md` | WING read-only review decision template required fields now have one shared contract source consumed by review and no-WING tooling. |
| `Docs/reports/AUTOMIXER_WING_MVP_REVIEW_TEMPLATE_STATUS_CONTRACT_20260613.md` | WING read-only review decision template pending status now has one shared contract source and no-WING drift check. |
| `Docs/reports/AUTOMIXER_WING_MVP_REVIEW_TEMPLATE_OUTPUT_CONTRACT_20260613.md` | WING read-only review decision template output path now has one shared contract source for runner, handoff, review, and no-WING tooling. |
| `Docs/reports/AUTOMIXER_WING_AUTOLOOP_HANDOFF_GATES_DOCS_20260612.md` | Codex autoloop prompt and README carry the current WING handoff blocker-anchor, digest-review, and supervised trace gate fields. |
| `Docs/reports/AUTOMIXER_WING_AUTOLOOP_CURRENT_HANDOFF_GATES_DOCS_20260612.md` | Codex autoloop prompt and README carry the current handoff nonclosure and final supervised operator-step marker fields. |
| `Docs/reports/AUTOMIXER_WING_AUTOLOOP_BACKEND_PRESTART_DIRECTOR_DOCS_20260612.md` | Codex autoloop README and Project Director tests carry the structured backend-port prestart gate before supervised proof planning. |
| `Docs/reports/AUTOMIXER_WING_PROJECT_DIRECTOR_HANDOFF_REQUIRED_FILE_20260612.md` | Project Director required files include the WING read-only handoff packet builder that emits structured prestart gates. |
| `Docs/reports/AUTOMIXER_WING_PROJECT_DIRECTOR_DECISION_PROOF_REQUIRED_FILES_20260612.md` | Project Director required files include decision review and supervised proof-plan gates for accepted-candidate prestart validation. |
| `Docs/reports/AUTOMIXER_WING_PROJECT_DIRECTOR_SELECTED_EVIDENCE_FILES_20260612.md` | Project Director exposes selected blocker evidence files so future cycles inspect the WING read-only proof surface named by the blocker table. |
| `Docs/reports/AUTOMIXER_PROJECT_GOAL_SUPERVISED_AUTOMIX_STRUCTURED_GATE_20260612.md` | Project goal capability matrix aligns Supervised AutoMix widening with the structured WING handoff gate. |
| `Docs/reports/MVP_PROGRESS_LATEST.md` | Latest generated supervised WING MVP blocker status and first required action. |
| `Docs/reports/AUTOMIXER_STREAMING_ALERTS_20260601.md` | Read-only streaming telemetry alert artifacts for dashboards, QA, and future learning labels. |
| `Docs/STRUCTURING_COMPLETION_20260518.md` | What was moved, what remains, and the next work boundary. |
| `Docs/PROJECT_CLEANUP_BACKLOG.md` | Remaining cleanup backlog and rules. |

## Runtime And Safety

| Area | Documents |
| --- | --- |
| Runtime hygiene | `Docs/RUNTIME_HYGIENE.md` |
| Live readiness and safety reports | `Docs/reports/tub/` |
| Write-surface and proof-gap reports | `Docs/reports/AUTOMIXER_WRITE_SURFACE_HARDENING_20260601.md`, `Docs/reports/AUTOMIXER_SAFETY_COVERAGE_MATRIX_20260601.md` |
| Read-only telemetry alert reports | `Docs/reports/AUTOMIXER_STREAMING_ALERTS_20260601.md`, `Docs/telemetry_learning_data.md` |
| WING manuals and protocol PDFs | `Docs/manuals/wing/` |
| Secret rotation | `Docs/secrets-rotation.md` |

## Backend

| Area | Documents |
| --- | --- |
| Backend docs | `Docs/backend/` |
| Backend entrypoint guide | `Docs/backend/README.md` |
| Manual probes and legacy checks | `Docs/backend/MANUAL_PROBES.md` |
| Backend source package | `backend/` |
| Legacy backup quarantine | `backend/lab_only/legacy_backups/` |

## Product And Agents

| Area | Documents |
| --- | --- |
| Product Layer plan | `Docs/PROJECT_IMPLEMENTATION_PLAN.md` |
| Current WING MVP rails | `Docs/MVP_WING_SUPERVISED.md`, `.ai/project-layer/wing_mvp_blockers.md`, `Docs/runbooks/WING_READONLY_PREFLIGHT.md`, `Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md`, `Docs/reports/AUTOMIXER_WING_READONLY_RUNBOOK_FINAL_PRESTART_GATE_20260612.md`, `Docs/reports/AUTOMIXER_WING_READONLY_RUNBOOK_STRUCTURED_GATE_20260612.md`, `Docs/reports/AUTOMIXER_WING_READONLY_HANDOFF_STRUCTURED_PRESTART_GATE_20260612.md`, `Docs/reports/AUTOMIXER_WING_READONLY_HANDOFF_MVP_BLOCKER_ANCHOR_20260612.md`, `Docs/reports/AUTOMIXER_WING_READONLY_HANDOFF_MVP_BLOCKER_ANCHOR_DIGEST_20260612.md`, `Docs/reports/AUTOMIXER_WING_HANDOFF_ANCHOR_DIGEST_OPERATOR_REVIEW_STEP_20260612.md`, `Docs/reports/AUTOMIXER_WING_MVP_BLOCKER_BACKEND_PRESTART_NEXT_ACTION_20260612.md`, `Docs/reports/AUTOMIXER_WING_MVP_BACKEND_PRESTART_FIRST_STEP_NEXT_ACTION_20260612.md`, `Docs/reports/AUTOMIXER_WING_MVP_BACKEND_PRESTART_GATE_STEP_CONTRACT_NEXT_ACTION_20260612.md`, `Docs/reports/AUTOMIXER_WING_MVP_BLOCKER_HANDOFF_ANCHOR_NEXT_ACTION_20260612.md`, `Docs/reports/AUTOMIXER_WING_MVP_BLOCKER_HANDOFF_ANCHOR_DIGEST_NEXT_ACTION_20260612.md`, `Docs/reports/AUTOMIXER_WING_MVP_BLOCKER_ANCHOR_DOES_NOT_CLOSE_NEXT_ACTION_20260612.md`, `Docs/reports/AUTOMIXER_WING_MVP_BLOCKER_HANDOFF_DIGEST_REVIEW_NEXT_ACTION_20260612.md`, `Docs/reports/AUTOMIXER_WING_MVP_BLOCKER_DIGEST_REVIEW_STEP_MARKERS_20260612.md`, `Docs/reports/AUTOMIXER_WING_MVP_BLOCKER_SUPERVISED_GATE_NEXT_ACTION_20260612.md`, `Docs/reports/AUTOMIXER_WING_MVP_BLOCKER_SUPERVISED_TRACE_GATE_NEXT_ACTION_20260612.md`, `Docs/reports/AUTOMIXER_WING_MVP_BLOCKER_SUPERVISED_NO_LIVE_AUTHORITY_NEXT_ACTION_20260612.md`, `Docs/reports/AUTOMIXER_WING_MVP_BLOCKER_SUPERVISED_STEP_MARKERS_NEXT_ACTION_20260612.md`, `Docs/reports/AUTOMIXER_WING_MVP_SUPERVISED_GATE_FINAL_STEP_CONTRACT_NEXT_ACTION_20260612.md`, `Docs/reports/AUTOMIXER_WING_MVP_SUPERVISED_GATE_DECISION_RUNBOOK_CONTRACT_NEXT_ACTION_20260612.md`, `Docs/reports/AUTOMIXER_WING_MVP_SUPERVISED_GATE_REQUIRED_MARKERS_NEXT_ACTION_20260612.md`, `Docs/reports/AUTOMIXER_WING_MVP_OPERATOR_SEQUENCE_STEP_INDEXES_NEXT_ACTION_20260612.md`, `Docs/reports/AUTOMIXER_WING_MVP_OPERATOR_SEQUENCE_COUNT_NEXT_ACTION_20260612.md`, `Docs/reports/AUTOMIXER_WING_MVP_BACKEND_PRESTART_STEP_TEXT_NEXT_ACTION_20260612.md`, `Docs/reports/AUTOMIXER_WING_MVP_BACKEND_PRESTART_RUNS_COMMAND_NEXT_ACTION_20260612.md`, `Docs/reports/AUTOMIXER_WING_MVP_BACKEND_PRESTART_REQUIRED_MARKERS_NEXT_ACTION_20260612.md`, `Docs/reports/AUTOMIXER_WING_MVP_BACKEND_PRESTART_CHECK_COMMAND_NEXT_ACTION_20260612.md`, `Docs/reports/AUTOMIXER_WING_MVP_BACKEND_PRESTART_FAILURE_DETAIL_FIELDS_NEXT_ACTION_20260612.md`, `Docs/reports/AUTOMIXER_WING_MVP_MANUAL_REVIEW_BEFORE_READINESS_NEXT_ACTION_20260612.md`, `Docs/reports/AUTOMIXER_WING_MVP_READINESS_UPDATE_DISALLOWED_BY_PACKET_NEXT_ACTION_20260612.md`, `Docs/reports/AUTOMIXER_WING_MVP_READINESS_CLOSURE_DISALLOWED_BY_PACKET_NEXT_ACTION_20260612.md`, `Docs/reports/AUTOMIXER_WING_MVP_READ_ONLY_REQUIRED_BY_PACKET_NEXT_ACTION_20260613.md`, `Docs/reports/AUTOMIXER_WING_MVP_LIVE_WRITE_DISALLOWED_BY_PACKET_NEXT_ACTION_20260613.md`, `Docs/reports/AUTOMIXER_WING_MVP_CONSOLE_MUTATION_DISALLOWED_BY_PACKET_NEXT_ACTION_20260613.md`, `Docs/reports/AUTOMIXER_WING_MVP_CONSOLE_QUERY_DISALLOWED_BY_PACKET_NEXT_ACTION_20260613.md`, `Docs/reports/AUTOMIXER_WING_MVP_TEST_EXECUTION_DISALLOWED_BY_PACKET_NEXT_ACTION_20260613.md`, `Docs/reports/AUTOMIXER_WING_MVP_OPERATOR_REQUIRED_BY_PACKET_NEXT_ACTION_20260613.md`, `Docs/reports/AUTOMIXER_WING_MVP_EXPECTED_PLAN_REQUIRED_BY_PACKET_NEXT_ACTION_20260613.md`, `Docs/reports/AUTOMIXER_WING_MVP_EXISTING_BACKEND_REQUIRED_BY_PACKET_NEXT_ACTION_20260613.md`, `Docs/reports/AUTOMIXER_WING_MVP_BACKEND_PRESTART_CHECK_REQUIRED_BY_PACKET_NEXT_ACTION_20260613.md`, `Docs/reports/AUTOMIXER_WING_MVP_BACKEND_PRESTART_MUST_PASS_BY_PACKET_NEXT_ACTION_20260613.md`, `Docs/reports/AUTOMIXER_WING_MVP_EXECUTION_DISALLOWED_BY_PACKET_NEXT_ACTION_20260612.md`, `Docs/reports/AUTOMIXER_WING_MVP_LIVE_AUTHORITY_DISALLOWED_BY_PACKET_NEXT_ACTION_20260613.md`, `Docs/reports/AUTOMIXER_WING_MVP_FULL_AUTONOMOUS_DISALLOWED_BY_PACKET_NEXT_ACTION_20260613.md`, `Docs/reports/AUTOMIXER_WING_MVP_NEXT_MANUAL_ARTIFACT_REQUIRED_BY_PACKET_NEXT_ACTION_20260613.md`, `Docs/reports/AUTOMIXER_WING_MVP_READONLY_ARTIFACT_CONTRACT_CONSTANTS_20260613.md`, `Docs/reports/AUTOMIXER_WING_AUTOLOOP_HANDOFF_GATES_DOCS_20260612.md`, `Docs/reports/AUTOMIXER_WING_AUTOLOOP_CURRENT_HANDOFF_GATES_DOCS_20260612.md`, `Docs/reports/AUTOMIXER_WING_AUTOLOOP_BACKEND_PRESTART_DIRECTOR_DOCS_20260612.md`, `Docs/reports/AUTOMIXER_WING_PROJECT_DIRECTOR_HANDOFF_REQUIRED_FILE_20260612.md`, `Docs/reports/AUTOMIXER_WING_PROJECT_DIRECTOR_DECISION_PROOF_REQUIRED_FILES_20260612.md`, `Docs/reports/AUTOMIXER_WING_PROJECT_DIRECTOR_SELECTED_EVIDENCE_FILES_20260612.md`, `Docs/reports/AUTOMIXER_WING_HANDOFF_STRUCTURED_GATE_SUMMARY_20260612.md`, `Docs/reports/MVP_PROGRESS_LATEST.md` |
| Current WING decision review prestart gate | `Docs/reports/AUTOMIXER_WING_MVP_DECISION_REVIEW_PRESTART_READY_GATE_20260613.md` |
| Current WING decision review prestart summary fields | `Docs/reports/AUTOMIXER_WING_MVP_DECISION_REVIEW_PRESTART_SUMMARY_FIELDS_20260614.md` |
| Current WING decision review prestart boolean rendering | `Docs/reports/AUTOMIXER_WING_MVP_DECISION_REVIEW_PRESTART_BOOL_RENDERING_20260614.md` |
| Current WING direction guard self-report skip | `Docs/reports/AUTOMIXER_WING_MVP_DIRECTION_GUARD_SKIP_SELF_REPORTS_20260614.md` |
| Current WING handoff prestart summary SHA binding | `Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_PRESTART_SUMMARY_SHA_BINDING_20260614.md` |
| Current WING handoff prestart summary SHA availability | `Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_PRESTART_SUMMARY_SHA_AVAILABILITY_20260614.md` |
| Current WING handoff operator sequence cardinality | `Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_OPERATOR_SEQUENCE_CARDINALITY_20260614.md` |
| Current WING handoff anchor digest review step index | `Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_ANCHOR_DIGEST_REVIEW_STEP_INDEX_20260614.md` |
| Current WING handoff declared anchor digest review index | `Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_DECLARED_ANCHOR_DIGEST_REVIEW_INDEX_20260614.md` |
| Current WING handoff declared operator sequence count | `Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_DECLARED_OPERATOR_SEQUENCE_COUNT_20260614.md` |
| Current WING handoff operator sequence SHA256 | `Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_OPERATOR_SEQUENCE_SHA256_20260614.md` |
| Current WING blocker evidence refresh | `Docs/reports/AUTOMIXER_WING_MVP_CURRENT_EVIDENCE_LATEST_REPORTS_20260613.md` |
| Current WING blocker listener-probe evidence refresh | `Docs/reports/AUTOMIXER_WING_MVP_CURRENT_EVIDENCE_LISTENER_PROBE_ERROR_REPORT_20260613.md` |
| Current WING no-WING busy-port fail-closed evidence | `Docs/reports/AUTOMIXER_WING_MVP_NOWING_QUICK_BUSY_PORT_FAIL_CLOSED_20260613.md` |
| Current WING backend-listener confirmation artifact | `Docs/reports/AUTOMIXER_WING_MVP_BACKEND_LISTENER_CONFIRMATION_ARTIFACT_20260613.md` |
| Current WING backend-listener confirmation template | `Docs/reports/AUTOMIXER_WING_MVP_BACKEND_LISTENER_CONFIRMATION_TEMPLATE_20260613.md` |
| Current WING backend-listener template edit gate | `Docs/reports/AUTOMIXER_WING_MVP_BACKEND_LISTENER_TEMPLATE_EDIT_GATE_20260613.md` |
| Current WING backend-listener template edit requirement canon | `Docs/reports/AUTOMIXER_WING_MVP_BACKEND_LISTENER_TEMPLATE_EDIT_REQUIREMENT_CANON_20260613.md` |
| Current WING backend-listener template pending summary fields | `Docs/reports/AUTOMIXER_WING_MVP_BACKEND_LISTENER_TEMPLATE_PENDING_SUMMARY_FIELDS_20260613.md` |
| Current WING backend-listener template authority summary fields | `Docs/reports/AUTOMIXER_WING_MVP_BACKEND_LISTENER_TEMPLATE_AUTHORITY_SUMMARY_FIELDS_20260613.md` |
| Current WING backend prestart template failure detail fields | `Docs/reports/AUTOMIXER_WING_MVP_BACKEND_PRESTART_TEMPLATE_FAILURE_DETAIL_FIELDS_20260613.md` |
| Current WING backend prestart template review fields | `Docs/reports/AUTOMIXER_WING_MVP_BACKEND_PRESTART_TEMPLATE_REVIEW_FIELDS_20260613.md` |
| Current WING busy-port prestart detail fields summary | `Docs/reports/AUTOMIXER_WING_MVP_BUSY_PORT_PRESTART_DETAIL_FIELDS_SUMMARY_20260613.md` |
| Current WING busy-port prestart command summary | `Docs/reports/AUTOMIXER_WING_MVP_BUSY_PORT_PRESTART_COMMAND_SUMMARY_20260613.md` |
| Current WING busy-port prestart required flags summary | `Docs/reports/AUTOMIXER_WING_MVP_BUSY_PORT_PRESTART_REQUIRED_FLAGS_SUMMARY_20260613.md` |
| Current WING busy-port prestart gate authority summary | `Docs/reports/AUTOMIXER_WING_MVP_BUSY_PORT_PRESTART_GATE_AUTHORITY_SUMMARY_20260613.md` |
| Current WING backend-listener pending summary fields canon | `Docs/reports/AUTOMIXER_WING_MVP_BACKEND_LISTENER_PENDING_SUMMARY_FIELDS_CANON_20260613.md` |
| Current WING read-only prestart summary binding | `Docs/reports/AUTOMIXER_WING_MVP_READONLY_PRESTART_SUMMARY_BINDING_20260613.md` |
| Current WING handoff prestart summary placeholder | `Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_PRESTART_SUMMARY_PLACEHOLDER_20260613.md` |
| Current WING handoff prestart summary command binding | `Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_PRESTART_SUMMARY_COMMAND_BINDING_20260614.md` |
| Current WING handoff backend prestart command summary | `Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_BACKEND_PRESTART_COMMAND_SUMMARY_20260614.md` |
| Current WING handoff denial flag summary truth | `Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_DENIAL_FLAG_SUMMARY_TRUTH_20260614.md` |
| Current WING handoff safety flag summary truth | `Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_SAFETY_FLAG_SUMMARY_TRUTH_20260614.md` |
| Current WING prestart summary contact boundary | `Docs/reports/AUTOMIXER_WING_MVP_PRESTART_SUMMARY_CONTACT_BOUNDARY_20260613.md` |
| Current WING prestart fail-closed condition sync | `Docs/reports/AUTOMIXER_WING_MVP_PRESTART_FAIL_CLOSED_CONDITION_SYNC_20260613.md` |
| Current WING fail-closed contract source | `Docs/reports/AUTOMIXER_WING_MVP_FAIL_CLOSED_CONTRACT_SOURCE_20260613.md` |
| Current WING backend prestart owner diagnostics | `Docs/reports/AUTOMIXER_WING_MVP_BACKEND_PRESTART_LISTENER_PROBE_ERROR_FIELD_20260613.md` |
| Correction method roadmap | `Docs/CORRECTION_METHODS_ROADMAP.md`, `.ai/project-layer/correction_methods_backlog.md` |
| Processing pipeline selection | `Docs/PROCESSING_METHODS_INVENTORY.md`, `Docs/adr/processing-pipeline-selection.md` |
| Automation foundation layer | `Docs/adr/automation-foundation-layer-20260527.md`, `Docs/reports/AUTOMIXER_AUTOMATION_FOUNDATION_LAYER_20260527.md` |
| Paperclip/GPT workflow | `Docs/PAPERCLIP_DIRECTOR_WORKFLOW.md` |
| Markdown memory pipeline | `Docs/memory_pipeline.md`, `.ai/agent-core/`, `.ai/project-layer/`, `.ai/pipeline/` |
| Operational memory | `Docs/OPERATIONAL_MEMORY.md` |
| Safety coverage and write surfaces | `Docs/reports/AUTOMIXER_SAFETY_COVERAGE_MATRIX_20260601.md`, `Docs/reports/AUTOMIXER_WRITE_SURFACE_HARDENING_20260601.md` |
| Telemetry learning and alerts | `Docs/telemetry_learning_data.md`, `Docs/reports/AUTOMIXER_STREAMING_ALERTS_20260601.md` |
| Tooling overview | `Docs/TOOLS.md` |
| Local Paperclip-readable reports | `.paperclip/reports/` |

## Archive

| Area | Path |
| --- | --- |
| Old root notes | `Docs/archive/root-notes/` |
| Generated legacy snapshots | `Docs/archive/generated/` |
| Old TUB reports | `Docs/reports/tub/` |

## Rules

- `backend/` is for source code and runtime-adjacent scripts, not report
  storage.
- `Docs/` is the source of truth for project knowledge.
- `Docs/reports/tub/` keeps historical issue reports searchable without
  cluttering runtime folders.
- `backend/lab_only/` is for quarantine, legacy, and manual tooling that should
  not be imported by live runtime.
- No live WING/OSC action is approved by documentation changes.
