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
| `Docs/reports/AUTOMIXER_WING_MVP_BLOCKER_HANDOFF_ANCHOR_NEXT_ACTION_20260612.md` | WING MVP blocker table and generated latest progress require the handoff `mvp_blocker_anchor` to match the current first open row. |
| `Docs/reports/AUTOMIXER_WING_MVP_BLOCKER_HANDOFF_ANCHOR_DIGEST_NEXT_ACTION_20260612.md` | WING MVP blocker table and generated latest progress require the handoff `mvp_blocker_anchor_sha256` digest to match the current first open row. |
| `Docs/reports/AUTOMIXER_WING_MVP_BLOCKER_ANCHOR_DOES_NOT_CLOSE_NEXT_ACTION_20260612.md` | WING MVP blocker table and generated latest progress require the handoff blocker anchor to state that offline packets do not close the blocker. |
| `Docs/reports/AUTOMIXER_WING_MVP_BLOCKER_HANDOFF_DIGEST_REVIEW_NEXT_ACTION_20260612.md` | WING MVP blocker table and generated latest progress require the handoff operator sequence to include the anchor digest review step. |
| `Docs/reports/AUTOMIXER_WING_MVP_BLOCKER_SUPERVISED_GATE_NEXT_ACTION_20260612.md` | WING MVP blocker table and generated latest progress require the final supervised prestart gate to be present and structured before supervised proof planning. |
| `Docs/reports/AUTOMIXER_WING_MVP_BLOCKER_SUPERVISED_TRACE_GATE_NEXT_ACTION_20260612.md` | WING MVP blocker table and generated latest progress require the supervised prestart gate to require a valid accepted-candidate prestart trace. |
| `Docs/reports/AUTOMIXER_WING_MVP_BLOCKER_SUPERVISED_NO_LIVE_AUTHORITY_NEXT_ACTION_20260612.md` | WING MVP blocker table and generated latest progress require the supervised prestart gate to show it grants no live authority. |
| `Docs/reports/AUTOMIXER_WING_MVP_BLOCKER_SUPERVISED_STEP_MARKERS_NEXT_ACTION_20260612.md` | WING MVP blocker table and generated latest progress require the final supervised operator step to carry the accepted-decision, prestart-trace, and proof-runbook markers. |
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
| Current WING MVP rails | `Docs/MVP_WING_SUPERVISED.md`, `.ai/project-layer/wing_mvp_blockers.md`, `Docs/runbooks/WING_READONLY_PREFLIGHT.md`, `Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md`, `Docs/reports/AUTOMIXER_WING_READONLY_RUNBOOK_FINAL_PRESTART_GATE_20260612.md`, `Docs/reports/AUTOMIXER_WING_READONLY_RUNBOOK_STRUCTURED_GATE_20260612.md`, `Docs/reports/AUTOMIXER_WING_READONLY_HANDOFF_STRUCTURED_PRESTART_GATE_20260612.md`, `Docs/reports/AUTOMIXER_WING_READONLY_HANDOFF_MVP_BLOCKER_ANCHOR_20260612.md`, `Docs/reports/AUTOMIXER_WING_READONLY_HANDOFF_MVP_BLOCKER_ANCHOR_DIGEST_20260612.md`, `Docs/reports/AUTOMIXER_WING_HANDOFF_ANCHOR_DIGEST_OPERATOR_REVIEW_STEP_20260612.md`, `Docs/reports/AUTOMIXER_WING_MVP_BLOCKER_BACKEND_PRESTART_NEXT_ACTION_20260612.md`, `Docs/reports/AUTOMIXER_WING_MVP_BLOCKER_HANDOFF_ANCHOR_NEXT_ACTION_20260612.md`, `Docs/reports/AUTOMIXER_WING_MVP_BLOCKER_HANDOFF_ANCHOR_DIGEST_NEXT_ACTION_20260612.md`, `Docs/reports/AUTOMIXER_WING_MVP_BLOCKER_ANCHOR_DOES_NOT_CLOSE_NEXT_ACTION_20260612.md`, `Docs/reports/AUTOMIXER_WING_MVP_BLOCKER_HANDOFF_DIGEST_REVIEW_NEXT_ACTION_20260612.md`, `Docs/reports/AUTOMIXER_WING_MVP_BLOCKER_SUPERVISED_GATE_NEXT_ACTION_20260612.md`, `Docs/reports/AUTOMIXER_WING_MVP_BLOCKER_SUPERVISED_TRACE_GATE_NEXT_ACTION_20260612.md`, `Docs/reports/AUTOMIXER_WING_MVP_BLOCKER_SUPERVISED_NO_LIVE_AUTHORITY_NEXT_ACTION_20260612.md`, `Docs/reports/AUTOMIXER_WING_MVP_BLOCKER_SUPERVISED_STEP_MARKERS_NEXT_ACTION_20260612.md`, `Docs/reports/AUTOMIXER_WING_AUTOLOOP_HANDOFF_GATES_DOCS_20260612.md`, `Docs/reports/AUTOMIXER_WING_AUTOLOOP_CURRENT_HANDOFF_GATES_DOCS_20260612.md`, `Docs/reports/AUTOMIXER_WING_AUTOLOOP_BACKEND_PRESTART_DIRECTOR_DOCS_20260612.md`, `Docs/reports/AUTOMIXER_WING_PROJECT_DIRECTOR_HANDOFF_REQUIRED_FILE_20260612.md`, `Docs/reports/AUTOMIXER_WING_PROJECT_DIRECTOR_DECISION_PROOF_REQUIRED_FILES_20260612.md`, `Docs/reports/AUTOMIXER_WING_PROJECT_DIRECTOR_SELECTED_EVIDENCE_FILES_20260612.md`, `Docs/reports/AUTOMIXER_WING_HANDOFF_STRUCTURED_GATE_SUMMARY_20260612.md`, `Docs/reports/MVP_PROGRESS_LATEST.md` |
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
