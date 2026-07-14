import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

from backend.autonomy_promotion_readiness import build_autonomy_promotion_readiness_snapshot
from backend.evaluation_integration_bundle_contract import (
    REQUIRED_INTEGRATION_CHECKS,
    evaluate_integration_bundle_contract,
)
from backend.fault_injection_recovery_contract import (
    REQUIRED_AUDIT_FIELDS as RECOVERY_REQUIRED_AUDIT_FIELDS,
    REQUIRED_EVALUATION_TYPES as RECOVERY_REQUIRED_EVALUATION_TYPES,
    REQUIRED_FAULT_SCENARIOS as RECOVERY_REQUIRED_FAULT_SCENARIOS,
    REQUIRED_RECOVERY_OUTCOMES,
    evaluate_fault_injection_recovery_contract,
)
from backend.integration_policy_contract import (
    REQUIRED_INTEGRATION_CONTROLS as POLICY_REQUIRED_INTEGRATION_CONTROLS,
    REQUIRED_INTEGRATION_EVIDENCE as POLICY_REQUIRED_INTEGRATION_EVIDENCE,
    evaluate_integration_policy_contract,
)
from backend.no_live_transport_contract import (
    REQUIRED_TRANSPORT_CONTROLS,
    REQUIRED_TRANSPORT_EVIDENCE,
    REQUIRED_ZERO_RUNTIME_OBSERVATIONS,
    evaluate_no_live_transport_contract,
)
from backend.offline_replay_evaluation_contract import (
    REQUIRED_EVALUATION_TYPES,
    REQUIRED_QUALITY_AXES,
    REQUIRED_SAFETY_CHECKS,
    REQUIRED_SCENARIOS,
    evaluate_offline_replay_evaluation_contract,
)
from backend.operator_report_contract import (
    REQUIRED_AUDIT_FIELDS as OPERATOR_REPORT_REQUIRED_AUDIT_FIELDS,
    REQUIRED_AUTHORITY_DENIAL_FLAGS as OPERATOR_REPORT_REQUIRED_AUTHORITY_DENIAL_FLAGS,
    REQUIRED_REPORT_BINDINGS,
    REQUIRED_REPORT_SECTIONS,
    evaluate_operator_report_contract,
)
from backend.quality_threshold_policy_contract import (
    REQUIRED_POLICY_CHECKS as THRESHOLD_REQUIRED_POLICY_CHECKS,
    REQUIRED_THRESHOLD_AXES,
    evaluate_quality_threshold_policy_contract,
)
from backend.recovery_policy_contract import (
    REQUIRED_RECOVERY_CONTROLS as POLICY_REQUIRED_RECOVERY_CONTROLS,
    REQUIRED_RECOVERY_EVIDENCE as POLICY_REQUIRED_RECOVERY_EVIDENCE,
    evaluate_recovery_policy_contract,
)
from backend.safety_kernel_trace_contract import (
    REQUIRED_AUDIT_FIELDS as SAFETY_KERNEL_REQUIRED_AUDIT_FIELDS,
    REQUIRED_GUARD_CHECKS as SAFETY_KERNEL_REQUIRED_GUARD_CHECKS,
    REQUIRED_TRACE_SCENARIOS as SAFETY_KERNEL_REQUIRED_TRACE_SCENARIOS,
    evaluate_safety_kernel_trace_contract,
)
from backend import wing_readonly_preflight_contract as readonly_contract
from tools import build_wing_readonly_preflight_handoff_packet as handoff_packet
from tools import no_wing_release_check as release_check
from tools import verify_no_wing_artifact_manifest as manifest_verifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MVP_DOC = PROJECT_ROOT / "Docs" / "MVP_WING_SUPERVISED.md"
CORRECTION_ROADMAP = PROJECT_ROOT / "Docs" / "CORRECTION_METHODS_ROADMAP.md"
BLOCKERS_DOC = PROJECT_ROOT / ".ai" / "project-layer" / "wing_mvp_blockers.md"
CORRECTION_BACKLOG = PROJECT_ROOT / ".ai" / "project-layer" / "correction_methods_backlog.md"
READONLY_RUNBOOK = PROJECT_ROOT / "Docs" / "runbooks" / "WING_READONLY_PREFLIGHT.md"
SUPERVISED_RUNBOOK = PROJECT_ROOT / "Docs" / "runbooks" / "WING_SUPERVISED_FADER_GAIN_PROOF.md"
DOCS_INDEX = PROJECT_ROOT / "Docs" / "INDEX.md"
CURRENT_STATE_DOC = PROJECT_ROOT / "Docs" / "CURRENT_STATE.md"
ROADMAP_DOC = PROJECT_ROOT / "Docs" / "ROADMAP.md"
ACTIVE_MILESTONE_DOC = PROJECT_ROOT / "Docs" / "ACTIVE_MILESTONE.md"
PROJECT_EXECUTION_CONTRACT = PROJECT_ROOT / "Docs" / "PROJECT_EXECUTION_CONTRACT.md"
PROJECT_GOAL = PROJECT_ROOT / "Docs" / "PROJECT_GOAL.md"
PROJECT_COMPLETION_GATE = PROJECT_ROOT / "Docs" / "PROJECT_COMPLETION_GATE.md"
SAFETY_CONTRACT = PROJECT_ROOT / "Docs" / "SAFETY_CONTRACT.md"
CAPABILITY_MATRIX = PROJECT_ROOT / "Docs" / "CAPABILITY_MATRIX.md"
CONSOLE_CAPABILITY_MATRIX = PROJECT_ROOT / "Docs" / "CONSOLE_CAPABILITY_MATRIX.md"
M001_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M001-wing-readonly-preflight-evidence-gate.md"
M001_REAL_WING_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M001-real-wing-interaction-plan.md"
M110_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M110-attended-wing-reset-soundcheck.md"
CURRENT_TASKS = PROJECT_ROOT / ".ai" / "project-layer" / "current_tasks.md"
M110_CP05_EVIDENCE = PROJECT_ROOT / "Docs" / "reports" / "M110_CP05_SOFTWARE_VALIDATION.json"
M110_CP05_GATE = PROJECT_ROOT / "ops" / "m110" / "CP05_HARDWARE_GATE.json"
M011_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M011-offline-replay-evaluation-contract.md"
M011_ADR = PROJECT_ROOT / "Docs" / "adr" / "offline-replay-evaluation-contract-20260619.md"
M012_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M012-fault-injection-recovery-contract.md"
M012_ADR = PROJECT_ROOT / "Docs" / "adr" / "fault-injection-recovery-contract-20260619.md"
M013_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M013-integration-evaluation-bundle-contract.md"
M013_ADR = PROJECT_ROOT / "Docs" / "adr" / "integration-evaluation-bundle-contract-20260619.md"
M014_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M014-quality-threshold-policy-contract.md"
M014_ADR = PROJECT_ROOT / "Docs" / "adr" / "quality-threshold-policy-contract-20260619.md"
M015_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M015-replay-threshold-policy-binding.md"
M015_ADR = PROJECT_ROOT / "Docs" / "adr" / "replay-threshold-policy-binding-20260619.md"
M016_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M016-recovery-policy-contract.md"
M016_ADR = PROJECT_ROOT / "Docs" / "adr" / "recovery-policy-contract-20260619.md"
M017_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M017-fault-recovery-policy-binding.md"
M017_ADR = PROJECT_ROOT / "Docs" / "adr" / "fault-recovery-policy-binding-20260619.md"
M018_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M018-integration-policy-contract.md"
M018_ADR = PROJECT_ROOT / "Docs" / "adr" / "integration-policy-contract-20260619.md"
M019_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M019-integration-policy-binding.md"
M019_ADR = PROJECT_ROOT / "Docs" / "adr" / "integration-policy-binding-20260619.md"
M020_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M020-safety-kernel-trace-contract.md"
M020_ADR = PROJECT_ROOT / "Docs" / "adr" / "safety-kernel-trace-contract-20260619.md"
M021_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M021-safety-kernel-trace-binding.md"
M021_ADR = PROJECT_ROOT / "Docs" / "adr" / "safety-kernel-trace-binding-20260619.md"
M022_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M022-no-live-transport-contract.md"
M022_ADR = PROJECT_ROOT / "Docs" / "adr" / "no-live-transport-contract-20260619.md"
M023_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M023-no-live-transport-binding.md"
M023_ADR = PROJECT_ROOT / "Docs" / "adr" / "no-live-transport-binding-20260619.md"
M024_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M024-operator-report-contract.md"
M024_ADR = PROJECT_ROOT / "Docs" / "adr" / "operator-report-contract-20260619.md"
M025_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M025-operator-report-binding.md"
M025_ADR = PROJECT_ROOT / "Docs" / "adr" / "operator-report-binding-20260619.md"
M026_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M026-autonomy-readiness-integration-bundle-overlay.md"
M026_ADR = PROJECT_ROOT / "Docs" / "adr" / "autonomy-readiness-integration-bundle-overlay-20260619.md"
M027_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M027-operator-adapter-integration-bundle-readiness-summary.md"
M027_ADR = PROJECT_ROOT / "Docs" / "adr" / "operator-adapter-integration-bundle-readiness-summary-20260619.md"
M028_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M028-main-pipeline-preview-integration-bundle-readiness-summary.md"
M028_ADR = PROJECT_ROOT / "Docs" / "adr" / "main-pipeline-preview-integration-bundle-readiness-summary-20260619.md"
M029_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M029-director-runtime-integration-bundle-readiness-summary.md"
M029_ADR = PROJECT_ROOT / "Docs" / "adr" / "director-runtime-integration-bundle-readiness-summary-20260619.md"
M030_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M030-director-event-integration-bundle-blocker-summary.md"
M030_ADR = PROJECT_ROOT / "Docs" / "adr" / "director-event-integration-bundle-blocker-summary-20260619.md"
M031_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M031-director-guard-integration-bundle-blocker-banner.md"
M031_ADR = PROJECT_ROOT / "Docs" / "adr" / "director-guard-integration-bundle-blocker-banner-20260619.md"
M032_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M032-director-goal-integration-bundle-blocker-summary.md"
M032_ADR = PROJECT_ROOT / "Docs" / "adr" / "director-goal-integration-bundle-blocker-summary-20260619.md"
M033_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M033-director-queue-integration-bundle-blocker-badge.md"
M033_ADR = PROJECT_ROOT / "Docs" / "adr" / "director-queue-integration-bundle-blocker-badge-20260619.md"
M034_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M034-director-suggestions-integration-bundle-blocker-notice.md"
M034_ADR = PROJECT_ROOT / "Docs" / "adr" / "director-suggestions-integration-bundle-blocker-notice-20260619.md"
M035_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M035-task-queue-integration-bundle-blocker-notice.md"
M035_ADR = PROJECT_ROOT / "Docs" / "adr" / "task-queue-integration-bundle-blocker-notice-20260619.md"
M036_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M036-supervised-apply-integration-bundle-blocker-notice.md"
M036_ADR = PROJECT_ROOT / "Docs" / "adr" / "supervised-apply-integration-bundle-blocker-notice-20260619.md"
M037_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M037-supervised-apply-integration-bundle-blocker-enforcement.md"
M037_ADR = PROJECT_ROOT / "Docs" / "adr" / "supervised-apply-integration-bundle-blocker-enforcement-20260619.md"
M038_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M038-supervised-apply-confirm-callback-blocker-guard.md"
M038_ADR = PROJECT_ROOT / "Docs" / "adr" / "supervised-apply-confirm-callback-blocker-guard-20260619.md"
M039_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M039-task-apply-request-helper-blocker-guard.md"
M039_ADR = PROJECT_ROOT / "Docs" / "adr" / "task-apply-request-helper-blocker-guard-20260619.md"
M040_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M040-task-details-apply-entrypoint-blocker-guard.md"
M040_ADR = PROJECT_ROOT / "Docs" / "adr" / "task-details-apply-entrypoint-blocker-guard-20260619.md"
M041_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M041-task-queue-metrics-blocker-guard.md"
M041_ADR = PROJECT_ROOT / "Docs" / "adr" / "task-queue-metrics-blocker-guard-20260619.md"
M042_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M042-task-queue-review-filter-blocker-guard.md"
M042_ADR = PROJECT_ROOT / "Docs" / "adr" / "task-queue-review-filter-blocker-guard-20260619.md"
M043_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M043-task-queue-row-badge-blocker-guard.md"
M043_ADR = PROJECT_ROOT / "Docs" / "adr" / "task-queue-row-badge-blocker-guard-20260619.md"
M044_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M044-task-details-badge-blocker-guard.md"
M044_ADR = PROJECT_ROOT / "Docs" / "adr" / "task-details-badge-blocker-guard-20260619.md"
M045_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M045-task-queue-review-only-metric-blocker-guard.md"
M045_ADR = PROJECT_ROOT / "Docs" / "adr" / "task-queue-review-only-metric-blocker-guard-20260620.md"
M046_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M046-task-details-blocker-list-blocker-guard.md"
M046_ADR = PROJECT_ROOT / "Docs" / "adr" / "task-details-blocker-list-blocker-guard-20260620.md"
M047_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M047-apply-plan-sequence-blocker-guard.md"
M047_ADR = PROJECT_ROOT / "Docs" / "adr" / "apply-plan-sequence-blocker-guard-20260620.md"
M048_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M048-operator-decision-blocker-status-guard.md"
M048_ADR = PROJECT_ROOT / "Docs" / "adr" / "operator-decision-blocker-status-guard-20260620.md"
M049_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M049-supervised-apply-readiness-blocker-guard.md"
M049_ADR = PROJECT_ROOT / "Docs" / "adr" / "supervised-apply-readiness-blocker-guard-20260620.md"
M050_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M050-supervised-apply-backend-result-blocker-guard.md"
M050_ADR = PROJECT_ROOT / "Docs" / "adr" / "supervised-apply-backend-result-blocker-guard-20260620.md"
M051_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M051-supervised-apply-gate-row-blocker-guard.md"
M051_ADR = PROJECT_ROOT / "Docs" / "adr" / "supervised-apply-gate-row-blocker-guard-20260620.md"
M052_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M052-task-details-apply-button-blocker-guard.md"
M052_ADR = PROJECT_ROOT / "Docs" / "adr" / "task-details-apply-button-blocker-guard-20260620.md"
M053_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M053-task-queue-row-reject-blocker-guard.md"
M053_ADR = PROJECT_ROOT / "Docs" / "adr" / "task-queue-row-reject-blocker-guard-20260620.md"
M054_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M054-task-details-reject-button-blocker-guard.md"
M054_ADR = PROJECT_ROOT / "Docs" / "adr" / "task-details-reject-button-blocker-guard-20260620.md"
M055_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M055-operator-decision-apply-result-blocker-guard.md"
M055_ADR = PROJECT_ROOT / "Docs" / "adr" / "operator-decision-apply-result-blocker-guard-20260620.md"
M056_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M056-safety-monitor-policy-blocker-guard.md"
M056_ADR = PROJECT_ROOT / "Docs" / "adr" / "safety-monitor-policy-blocker-guard-20260620.md"
M057_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M057-safety-monitor-goal-readiness-blocker-guard.md"
M057_ADR = PROJECT_ROOT / "Docs" / "adr" / "safety-monitor-goal-readiness-blocker-guard-20260620.md"
M058_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M058-safety-monitor-blocked-surfaces-blocker-guard.md"
M058_ADR = PROJECT_ROOT / "Docs" / "adr" / "safety-monitor-blocked-surfaces-blocker-guard-20260620.md"
M059_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M059-replay-evidence-decision-trace-status-blocker-guard.md"
M059_ADR = PROJECT_ROOT / "Docs" / "adr" / "replay-evidence-decision-trace-status-blocker-guard-20260620.md"
M060_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M060-replay-evidence-latest-run-status-blocker-guard.md"
M060_ADR = PROJECT_ROOT / "Docs" / "adr" / "replay-evidence-latest-run-status-blocker-guard-20260620.md"
M061_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M061-replay-evidence-policy-row-blocker-guard.md"
M061_ADR = PROJECT_ROOT / "Docs" / "adr" / "replay-evidence-policy-row-blocker-guard-20260620.md"
M062_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M062-replay-evidence-channel-status-blocker-guard.md"
M062_ADR = PROJECT_ROOT / "Docs" / "adr" / "replay-evidence-channel-status-blocker-guard-20260620.md"
M063_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M063-replay-evidence-report-path-blocker-guard.md"
M063_ADR = PROJECT_ROOT / "Docs" / "adr" / "replay-evidence-report-path-blocker-guard-20260620.md"
M064_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M064-task-queue-status-column-blocker-guard.md"
M064_ADR = PROJECT_ROOT / "Docs" / "adr" / "task-queue-status-column-blocker-guard-20260620.md"
M065_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M065-task-details-status-value-blocker-guard.md"
M065_ADR = PROJECT_ROOT / "Docs" / "adr" / "task-details-status-value-blocker-guard-20260620.md"
M066_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M066-task-details-proposal-risk-blocker-guard.md"
M066_ADR = PROJECT_ROOT / "Docs" / "adr" / "task-details-proposal-risk-blocker-guard-20260620.md"
M067_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M067-task-queue-move-delta-blocker-guard.md"
M067_ADR = PROJECT_ROOT / "Docs" / "adr" / "task-queue-move-delta-blocker-guard-20260620.md"
M068_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M068-task-details-proposal-delta-blocker-guard.md"
M068_ADR = PROJECT_ROOT / "Docs" / "adr" / "task-details-proposal-delta-blocker-guard-20260620.md"
M069_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M069-supervised-apply-new-value-blocker-guard.md"
M069_ADR = PROJECT_ROOT / "Docs" / "adr" / "supervised-apply-new-value-blocker-guard-20260620.md"
M070_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M070-supervised-apply-rollback-blocker-guard.md"
M070_ADR = PROJECT_ROOT / "Docs" / "adr" / "supervised-apply-rollback-blocker-guard-20260620.md"
M071_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M071-task-details-proposal-rollback-blocker-guard.md"
M071_ADR = PROJECT_ROOT / "Docs" / "adr" / "task-details-proposal-rollback-blocker-guard-20260620.md"
M072_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M072-task-details-proposal-proof-blocker-guard.md"
M072_ADR = PROJECT_ROOT / "Docs" / "adr" / "task-details-proposal-proof-blocker-guard-20260620.md"
M073_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M073-task-details-proposal-parameter-blocker-guard.md"
M073_ADR = PROJECT_ROOT / "Docs" / "adr" / "task-details-proposal-parameter-blocker-guard-20260620.md"
M074_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M074-supervised-apply-parameter-blocker-guard.md"
M074_ADR = PROJECT_ROOT / "Docs" / "adr" / "supervised-apply-parameter-blocker-guard-20260620.md"
M075_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M075-supervised-apply-old-value-blocker-guard.md"
M075_ADR = PROJECT_ROOT / "Docs" / "adr" / "supervised-apply-old-value-blocker-guard-20260620.md"
M076_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M076-replay-evidence-start-action-blocker-guard.md"
M076_ADR = PROJECT_ROOT / "Docs" / "adr" / "replay-evidence-start-action-blocker-guard-20260620.md"
M077_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M077-replay-evidence-latest-run-id-blocker-guard.md"
M077_ADR = PROJECT_ROOT / "Docs" / "adr" / "replay-evidence-latest-run-id-blocker-guard-20260620.md"
M078_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M078-replay-evidence-trace-detail-blocker-guard.md"
M078_ADR = PROJECT_ROOT / "Docs" / "adr" / "replay-evidence-trace-detail-blocker-guard-20260620.md"
M079_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M079-replay-evidence-channel-identity-blocker-guard.md"
M079_ADR = PROJECT_ROOT / "Docs" / "adr" / "replay-evidence-channel-identity-blocker-guard-20260620.md"
M080_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M080-replay-evidence-report-row-blocker-visibility-guard.md"
M080_ADR = PROJECT_ROOT / "Docs" / "adr" / "replay-evidence-report-row-blocker-visibility-guard-20260620.md"
M081_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M081-wing-review-decision-prestart-proof-binding.md"
M081_ADR = PROJECT_ROOT / "Docs" / "adr" / "wing-review-decision-prestart-proof-binding-20260620.md"
M082_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M082-wing-handoff-decision-template-proof-fields.md"
M082_ADR = PROJECT_ROOT / "Docs" / "adr" / "wing-handoff-decision-template-proof-fields-20260620.md"
M083_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M083-source-arrangement-readonly-context-gates.md"
M083_ADR = PROJECT_ROOT / "Docs" / "adr" / "source-arrangement-readonly-context-gates-20260620.md"
M084_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M084-unmasking-frequency-balance-context-gates.md"
M084_ADR = PROJECT_ROOT / "Docs" / "adr" / "unmasking-frequency-balance-context-gates-20260620.md"
M085_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M085-fx-send-monitor-return-safety-gates.md"
M085_ADR = PROJECT_ROOT / "Docs" / "adr" / "fx-send-monitor-return-safety-gates-20260620.md"
M086_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M086-master-bus-pa-feed-reference-safety-gates.md"
M086_ADR = PROJECT_ROOT / "Docs" / "adr" / "master-bus-pa-feed-reference-safety-gates-20260620.md"
M087_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M087-performance-adaptation-readonly-context-gates.md"
M087_ADR = PROJECT_ROOT / "Docs" / "adr" / "performance-adaptation-readonly-context-gates-20260620.md"
M088_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M088-routing-management-readonly-context-gates.md"
M088_ADR = PROJECT_ROOT / "Docs" / "adr" / "routing-management-readonly-context-gates-20260620.md"
M089_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M089-mute-management-readonly-context-gates.md"
M089_ADR = PROJECT_ROOT / "Docs" / "adr" / "mute-management-readonly-context-gates-20260620.md"
M090_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M090-pan-spatial-readonly-context-gates.md"
M090_ADR = PROJECT_ROOT / "Docs" / "adr" / "pan-spatial-readonly-context-gates-20260620.md"
M091_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M091-scene-preset-readonly-context-gates.md"
M091_ADR = PROJECT_ROOT / "Docs" / "adr" / "scene-preset-readonly-context-gates-20260620.md"
M092_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M092-mix-variant-readonly-context-gates.md"
M092_ADR = PROJECT_ROOT / "Docs" / "adr" / "mix-variant-readonly-context-gates-20260620.md"
M093_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M093-decision-evidence-readonly-context-gates.md"
M093_ADR = PROJECT_ROOT / "Docs" / "adr" / "decision-evidence-readonly-context-gates-20260620.md"
M094_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M094-project-director-external-validation-scope-guard.md"
M094_ADR = PROJECT_ROOT / "Docs" / "adr" / "project-director-external-validation-scope-guard-20260620.md"
M095_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M095-wing-readonly-reserved-root-intake-guard.md"
M095_ADR = PROJECT_ROOT / "Docs" / "adr" / "wing-readonly-reserved-root-intake-guard-20260620.md"
M096_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M096-wing-readonly-declared-path-intake-guard.md"
M096_ADR = PROJECT_ROOT / "Docs" / "adr" / "wing-readonly-declared-path-intake-guard-20260620.md"
M097_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M097-wing-readonly-canonical-root-intake-guard.md"
M097_ADR = PROJECT_ROOT / "Docs" / "adr" / "wing-readonly-canonical-root-intake-guard-20260620.md"
M098_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M098-wing-readonly-source-path-intake-guard.md"
M098_ADR = PROJECT_ROOT / "Docs" / "adr" / "wing-readonly-source-path-intake-guard-20260620.md"
M099_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M099-wing-readonly-run-id-path-segment-intake-guard.md"
M099_ADR = PROJECT_ROOT / "Docs" / "adr" / "wing-readonly-run-id-path-segment-intake-guard-20260620.md"
M100_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M100-wing-readonly-artifact-root-parent-traversal-intake-guard.md"
M100_ADR = PROJECT_ROOT / "Docs" / "adr" / "wing-readonly-artifact-root-parent-traversal-intake-guard-20260620.md"
M101_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M101-wing-readonly-artifact-root-whitespace-intake-guard.md"
M101_ADR = PROJECT_ROOT / "Docs" / "adr" / "wing-readonly-artifact-root-whitespace-intake-guard-20260620.md"
M102_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M102-wing-readonly-run-id-format-intake-guard.md"
M102_ADR = PROJECT_ROOT / "Docs" / "adr" / "wing-readonly-run-id-format-intake-guard-20260620.md"
M103_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M103-wing-readonly-run-id-timestamp-intake-guard.md"
M103_ADR = PROJECT_ROOT / "Docs" / "adr" / "wing-readonly-run-id-timestamp-intake-guard-20260620.md"
M104_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M104-wing-readonly-capture-time-order-intake-guard.md"
M104_ADR = PROJECT_ROOT / "Docs" / "adr" / "wing-readonly-capture-time-order-intake-guard-20260620.md"
M105_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M105-project-director-post-m104-intake-loop-guard.md"
M105_ADR = PROJECT_ROOT / "Docs" / "adr" / "project-director-post-m104-intake-loop-guard-20260620.md"
M106_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M106-project-director-post-m105-intake-loop-persistence.md"
M106_ADR = PROJECT_ROOT / "Docs" / "adr" / "project-director-post-m105-intake-loop-persistence-20260620.md"
M107_PLAN = PROJECT_ROOT / "Docs" / "plans" / "M107-project-director-intake-loop-lookback-floor.md"
M107_ADR = PROJECT_ROOT / "Docs" / "adr" / "project-director-intake-loop-lookback-floor-20260620.md"
PROCESSING_PIPELINE_PREVIEW = PROJECT_ROOT / "backend" / "processing_pipeline_preview.py"
PROCESSING_PIPELINE_PREVIEW_INTEGRATION_BUNDLE_TEST = (
    PROJECT_ROOT
    / "tests"
    / "test_processing_pipeline_preview_integration_bundle_summary.py"
)
FRONTEND_OPERATOR_ADAPTER = PROJECT_ROOT / "frontend" / "src" / "services" / "operatorAdapters.ts"
FRONTEND_OPERATOR_ADAPTER_TEST = (
    PROJECT_ROOT
    / "frontend"
    / "src"
    / "services"
    / "__tests__"
    / "operatorAdaptersContract.test.mjs"
)
FRONTEND_DIRECTOR_PANEL_MODEL = (
    PROJECT_ROOT
    / "frontend"
    / "src"
    / "components"
    / "dashboard"
    / "directorPanelModel.ts"
)
FRONTEND_RIGHT_DIRECTOR_PANEL = PROJECT_ROOT / "frontend" / "src" / "views" / "DirectorPanel.tsx"
FRONTEND_TASK_QUEUE_PANEL = (
    PROJECT_ROOT
    / "frontend"
    / "src"
    / "components"
    / "prototype"
    / "TaskQueuePanels.tsx"
)
FRONTEND_TASK_QUEUE_VIEW = PROJECT_ROOT / "frontend" / "src" / "views" / "TaskQueueView.tsx"
AUTOPILOT_SKILL = PROJECT_ROOT / ".agents" / "skills" / "automixer-autopilot" / "SKILL.md"
AUTOPILOT_BLOCKERS = PROJECT_ROOT / "ops" / "autopilot" / "BLOCKERS.md"
AUTOPILOT_OPERATIONS = PROJECT_ROOT / "ops" / "autopilot" / "OPERATIONS.md"
AUTOPILOT_HEARTBEAT = PROJECT_ROOT / "ops" / "autopilot" / "HEARTBEAT.md"
AUTOPILOT_LAST_REPORT = PROJECT_ROOT / "ops" / "autopilot" / "LAST_REPORT.md"
AUTOPILOT_STATE = PROJECT_ROOT / "ops" / "autopilot" / "STATE.json"
AUTOPILOT_RUN_LEDGER = PROJECT_ROOT / "ops" / "autopilot" / "RUN_LEDGER.jsonl"
M006_OWNER_REVIEW_PACKET_SUMMARY = (
    PROJECT_ROOT
    / "artifacts"
    / "wing_owner_readiness_review"
    / "20260704T191101Z_m006_owner_readiness_review_packet"
    / "summary.json"
)
M006_OWNER_REVIEW_PACKET_REPORT = (
    PROJECT_ROOT / "Docs" / "reports" / "WING_M006_OWNER_READINESS_REVIEW_PACKET_20260704.md"
)
REAL_WING_ARTIFACT_ROOT = PROJECT_ROOT / "artifacts" / "wing_readonly_preflight"
VALIDATE_INPUTS_READY_SUMMARY = (
    PROJECT_ROOT
    / "ops"
    / "autopilot"
    / "evidence"
    / "M001_CP04_validate_inputs_ready_for_websocket"
    / "summary.json"
)
VALIDATE_INPUTS_MISMATCH_SUMMARY = (
    PROJECT_ROOT
    / "ops"
    / "autopilot"
    / "evidence"
    / "M001_CP04_validate_inputs_expected_plan_mismatch"
    / "summary.json"
)
CP04_HANDOFF_REPORT = (
    PROJECT_ROOT
    / "Docs"
    / "reports"
    / "AUTOMIXER_AUTOPILOT_M001_CP04_WING_READONLY_HANDOFF_20260618.md"
)
READONLY_HANDOFF_LATEST = (
    PROJECT_ROOT / "Docs" / "reports" / "WING_READONLY_PREFLIGHT_HANDOFF_PACKET_LATEST.md"
)
MVP_PROGRESS_REPORT = PROJECT_ROOT / "Docs" / "reports" / "MVP_PROGRESS_LATEST.md"
PROGRESS_TOOL = PROJECT_ROOT / "tools" / "codex_autoloop" / "update_mvp_progress.py"
PROJECT_DIRECTOR_TOOL = PROJECT_ROOT / "tools" / "codex_autoloop" / "project_director_agent.py"
RUNNER_TOOL = PROJECT_ROOT / "tools" / "wing_readonly_preflight_runner.py"
PROMPT = PROJECT_ROOT / "tools" / "codex_autoloop" / "codex_cycle_prompt.md"

DECLARED_OPERATOR_SEQUENCE_COUNT_MARKERS = (
    "declared_operator_sequence_count: 12",
    "declared_expected_operator_sequence_count: 12",
    "declared_operator_sequence_count_contract_flag: true",
    "declared_operator_sequence_count_matches_actual: true",
    "declared_operator_sequence_count_matches_contract: true",
    "declared_expected_operator_sequence_count_matches_contract: true",
    "declared_operator_sequence_count_contract_flag_matches_declared_count: true",
)

OPERATOR_SEQUENCE_SHA_MARKERS = (
    "operator_sequence_sha256",
    "actual_operator_sequence_sha256",
    "operator_sequence_sha256_matches_actual: true",
)

OPERATOR_SEQUENCE_CONTRACT_MARKERS = (
    "expected_operator_sequence_sha256",
    "actual_expected_operator_sequence_sha256",
    "operator_sequence_matches_contract: true",
    "operator_sequence_sha256_matches_contract: true",
    "declared_expected_operator_sequence_sha256_matches_contract: true",
    "declared_operator_sequence_matches_contract_flag: true",
    "declared_operator_sequence_matches_contract_flag_matches_actual: true",
    "declared_operator_sequence_sha256_matches_contract_flag: true",
    "declared_operator_sequence_sha256_matches_contract_flag_matches_actual: true",
)

DECLARED_OPERATOR_SEQUENCE_COUNT_FIELD_MARKERS = tuple(
    marker.split(":", 1)[0] for marker in DECLARED_OPERATOR_SEQUENCE_COUNT_MARKERS
)

ENVIRONMENT_BLOCKED_FRONTEND_MARKERS = (
    "environment_blocked_checks",
    "environment_blocked_check_details",
    "local_dev_server_bind_denied",
    "127.0.0.1:3000",
)

ARTIFACT_MANIFEST_SUMMARY_SURFACE_SIDECAR_MARKERS = (
    "artifact-manifest.json",
    "same no-WING artifact directory",
    "artifact_manifest_self_consistency_summary_surface_match: true",
    "artifact_manifest_self_consistency_summary_surface_json_mismatches: []",
    "artifact_manifest_self_consistency_summary_surface_markdown_mismatches: []",
    "artifact_manifest_self_consistency_summary_surface_mismatches: []",
    "artifact_manifest_self_consistency_summary_surface_readiness_closure_allowed: false",
    "artifact_manifest_self_consistency_summary_surface_live_authority_granted: false",
    "artifact_manifest_self_consistency_match: true",
    "artifact_manifest_self_consistency_mismatches: []",
    "artifact_manifest_contract_cross_surface_match: true",
    "artifact_manifest_contract_cross_surface_mismatches: []",
)

ARTIFACT_MANIFEST_VERIFIER_OPERATOR_RESULT_MARKERS = (
    "status: passed",
    "failed_checks: []",
    "read_only: true",
    "live_authority_granted: false",
    "readiness_closure_allowed: false",
)

ARTIFACT_MANIFEST_VERIFIER_REPORT_MARKERS = (
    "PYTHONPATH=backend python3 tools/verify_no_wing_artifact_manifest.py",
    "status: passed",
    "failed_checks: []",
    "failed_checks_missing_check_entries: []",
    "failed_checks_have_check_entries: true",
    "summary_json_exists: true",
    "summary_md_exists: true",
    "artifact_manifest_exists: true",
    "artifact_bundle_loadable: true",
    "manifest_type: true",
    "manifest_required_artifact_ids_match_contract: true",
    "manifest_required_artifact_paths_match_contract: true",
    "manifest_records_required_artifact_ids: true",
    "summary_json_path_matches_contract: true",
    "summary_json_sha256_matches_actual: true",
    "summary_md_path_matches_contract: true",
    "summary_md_sha256_matches_actual: true",
    "summary_json_sha256_matches_manifest: true",
    "summary_md_sha256_matches_manifest: true",
    "manifest_denies_readiness_matrix_update: true",
    "manifest_denies_readiness_closure: true",
    "manifest_denies_live_authority: true",
    "summary_denies_readiness_matrix_update: true",
    "summary_denies_readiness_closure: true",
    "summary_denies_live_authority: true",
    "summary_md_artifact_manifest_contract_cross_surface_readiness_matrix_update_allowed_denies_authority: true",
    "summary_json_artifact_manifest_verifier_fields_present: true",
    "manifest_artifact_manifest_verifier_fields_present: true",
    "summary_md_artifact_manifest_verifier_fields_rendered: true",
    "manifest_artifact_manifest_verifier_command_matches_expected: true",
    "summary_json_artifact_manifest_verifier_report_required_fields_matches_expected: true",
    "summary_md_artifact_manifest_verifier_report_required_fields_matches_expected: true",
    "manifest_artifact_manifest_verifier_report_required_fields_matches_expected: true",
    "verifier_report_required_fields_present: true",
    "verifier_report_required_fields_missing: []",
    "verifier_report_required_fields_match_contract: true",
    "verifier_report_readiness_closure_allowed_matches_expected: true",
    "verifier_report_live_authority_granted_matches_expected: true",
    "artifact_manifest_verifier_check_mismatches: []",
    "artifact_manifest_verifier_check_summary_json_exists: true",
    "artifact_manifest_verifier_check_summary_md_exists: true",
    "artifact_manifest_verifier_check_artifact_manifest_exists: true",
    "artifact_manifest_verifier_check_artifact_bundle_loadable: true",
    "artifact_manifest_verifier_check_manifest_type: true",
    "artifact_manifest_verifier_check_manifest_required_artifact_ids_match_contract: true",
    "artifact_manifest_verifier_check_manifest_required_artifact_paths_match_contract: true",
    "artifact_manifest_verifier_check_manifest_records_required_artifact_ids: true",
    "artifact_manifest_verifier_check_summary_json_path_matches_contract: true",
    "artifact_manifest_verifier_check_summary_json_sha256_matches_actual: true",
    "artifact_manifest_verifier_check_summary_md_path_matches_contract: true",
    "artifact_manifest_verifier_check_summary_md_sha256_matches_actual: true",
    "artifact_manifest_verifier_check_summary_json_sha256_matches_manifest: true",
    "artifact_manifest_verifier_check_summary_md_sha256_matches_manifest: true",
    "artifact_manifest_verifier_check_manifest_denies_readiness_matrix_update: true",
    "artifact_manifest_verifier_check_manifest_denies_readiness_closure: true",
    "artifact_manifest_verifier_check_manifest_denies_live_authority: true",
    "artifact_manifest_verifier_check_summary_denies_readiness_matrix_update: true",
    "artifact_manifest_verifier_check_summary_denies_readiness_closure: true",
    "artifact_manifest_verifier_check_summary_denies_live_authority: true",
    "artifact_manifest_verifier_check_summary_md_artifact_manifest_contract_cross_surface_readiness_matrix_update_allowed_denies_authority: true",
    "artifact_manifest_verifier_check_summary_json_artifact_manifest_verifier_fields_present: true",
    "artifact_manifest_verifier_check_manifest_artifact_manifest_verifier_fields_present: true",
    "artifact_manifest_verifier_check_summary_md_artifact_manifest_verifier_fields_rendered: true",
    "artifact_manifest_verifier_check_summary_json_artifact_manifest_verifier_read_only_matches_expected: true",
    "artifact_manifest_verifier_check_summary_md_artifact_manifest_verifier_read_only_matches_expected: true",
    "artifact_manifest_verifier_check_manifest_artifact_manifest_verifier_read_only_matches_expected: true",
    "artifact_manifest_verifier_check_summary_json_artifact_manifest_verifier_live_write_matches_expected: true",
    "artifact_manifest_verifier_check_summary_md_artifact_manifest_verifier_live_write_matches_expected: true",
    "artifact_manifest_verifier_check_manifest_artifact_manifest_verifier_live_write_matches_expected: true",
    "artifact_manifest_verifier_check_summary_json_artifact_manifest_verifier_contacts_backend_matches_expected: true",
    "artifact_manifest_verifier_check_summary_md_artifact_manifest_verifier_contacts_backend_matches_expected: true",
    "artifact_manifest_verifier_check_manifest_artifact_manifest_verifier_contacts_backend_matches_expected: true",
    "artifact_manifest_verifier_check_summary_json_artifact_manifest_verifier_executes_tests_matches_expected: true",
    "artifact_manifest_verifier_check_summary_md_artifact_manifest_verifier_executes_tests_matches_expected: true",
    "artifact_manifest_verifier_check_manifest_artifact_manifest_verifier_executes_tests_matches_expected: true",
    "artifact_manifest_verifier_check_summary_json_artifact_manifest_verifier_contacts_wing_matches_expected: true",
    "artifact_manifest_verifier_check_summary_md_artifact_manifest_verifier_contacts_wing_matches_expected: true",
    "artifact_manifest_verifier_check_manifest_artifact_manifest_verifier_contacts_wing_matches_expected: true",
    "artifact_manifest_verifier_check_summary_json_artifact_manifest_verifier_mutates_console_state_matches_expected: true",
    "artifact_manifest_verifier_check_summary_md_artifact_manifest_verifier_mutates_console_state_matches_expected: true",
    "artifact_manifest_verifier_check_manifest_artifact_manifest_verifier_mutates_console_state_matches_expected: true",
    "artifact_manifest_verifier_check_summary_json_artifact_manifest_verifier_queries_console_matches_expected: true",
    "artifact_manifest_verifier_check_summary_md_artifact_manifest_verifier_queries_console_matches_expected: true",
    "artifact_manifest_verifier_check_manifest_artifact_manifest_verifier_queries_console_matches_expected: true",
    "artifact_manifest_verifier_check_summary_json_artifact_manifest_verifier_readiness_closure_allowed_matches_expected: true",
    "artifact_manifest_verifier_check_summary_md_artifact_manifest_verifier_readiness_closure_allowed_matches_expected: true",
    "artifact_manifest_verifier_check_manifest_artifact_manifest_verifier_readiness_closure_allowed_matches_expected: true",
    "artifact_manifest_verifier_check_summary_json_artifact_manifest_verifier_live_authority_granted_matches_expected: true",
    "artifact_manifest_verifier_check_summary_md_artifact_manifest_verifier_live_authority_granted_matches_expected: true",
    "artifact_manifest_verifier_check_manifest_artifact_manifest_verifier_live_authority_granted_matches_expected: true",
    "artifact_manifest_verifier_check_manifest_artifact_manifest_verifier_command_matches_expected: true",
    "artifact_manifest_verifier_check_summary_json_artifact_manifest_verifier_report_required_fields_matches_expected: true",
    "artifact_manifest_verifier_check_summary_md_artifact_manifest_verifier_report_required_fields_matches_expected: true",
    "artifact_manifest_verifier_check_manifest_artifact_manifest_verifier_report_required_fields_matches_expected: true",
    "artifact_manifest_verifier_check_verifier_report_required_fields_present: true",
    "artifact_manifest_verifier_check_verifier_report_artifact_manifest_self_consistency_match_matches_expected: true",
    "artifact_manifest_verifier_check_verifier_report_artifact_manifest_contract_cross_surface_match_matches_expected: true",
    "artifact_manifest_verifier_check_verifier_report_artifact_manifest_self_consistency_summary_surface_match_matches_expected: true",
    "artifact_manifest_verifier_check_verifier_report_readiness_closure_allowed_matches_expected: true",
    "artifact_manifest_verifier_check_verifier_report_live_authority_granted_matches_expected: true",
    "artifact_manifest_verifier_check_failed_checks_have_check_entries: true",
    "artifact_manifest_verifier_report_required_fields",
    "checks",
    "verifier_report_required_fields",
    "verifier_report_required_fields_missing",
    "verifier_report_required_fields_match_contract",
)


def load_progress_module():
    spec = importlib.util.spec_from_file_location("update_mvp_progress", PROGRESS_TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_runner_module():
    spec = importlib.util.spec_from_file_location(
        "wing_readonly_preflight_runner_for_mvp_rails", RUNNER_TOOL
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_project_director_module():
    sys.path.insert(0, str(PROJECT_DIRECTOR_TOOL.parent))
    spec = importlib.util.spec_from_file_location(
        "project_director_agent_for_mvp_rails", PROJECT_DIRECTOR_TOOL
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def covered_verifier_checks(markers):
    covered: set[str] = set()
    for marker in markers:
        if not marker.endswith(": true"):
            continue
        raw_marker = marker.removesuffix(": true")
        covered.add(raw_marker)
        if raw_marker.startswith("artifact_manifest_verifier_check_"):
            covered.add(raw_marker.removeprefix("artifact_manifest_verifier_check_"))
    return covered


def current_wing_mvp_20260614_index_report_names():
    index_text = DOCS_INDEX.read_text(encoding="utf-8")
    report_paths = re.findall(
        r"`(Docs/reports/AUTOMIXER_WING_MVP_[^`]+_20260614\.md)`",
        index_text,
    )
    return sorted({Path(path).name for path in report_paths})


def markdown_table_row(text: str, first_cell: str) -> list[str]:
    for line in text.splitlines():
        if line.startswith(f"| {first_cell} |"):
            return [cell.strip() for cell in line.strip("|").split("|")]
    raise AssertionError(f"missing markdown table row for {first_cell!r}")


def files_under(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(candidate for candidate in path.rglob("*") if candidate.is_file())


def safety_kernel_trace_record_fixture(scenario: str, **overrides) -> dict:
    decision = "bounded_allow" if scenario == "supervised_fader_gain_allowed" else "reject"
    record = {
        "scenario": scenario,
        "safety_decision": decision,
        "guard_checks": {key: True for key in SAFETY_KERNEL_REQUIRED_GUARD_CHECKS},
        "audit": {
            key: f"{scenario}/{key}" for key in SAFETY_KERNEL_REQUIRED_AUDIT_FIELDS
        }
        | {"safety_decision": decision},
    }
    record.update(overrides)
    return record


def safety_kernel_trace_manifest_fixture(**overrides) -> dict:
    manifest = {
        "trace_id": "safety-kernel-trace-fixture",
        "offline_only": True,
        "live_write": False,
        "contacts_wing": False,
        "contacts_backend": False,
        "hardware_evidence": False,
        "readiness_closure_allowed": False,
        "live_authority_granted": False,
        "trace_records": [
            safety_kernel_trace_record_fixture(scenario)
            for scenario in SAFETY_KERNEL_REQUIRED_TRACE_SCENARIOS
        ],
    }
    manifest.update(overrides)
    return manifest


def no_live_transport_evidence_fixture(name: str) -> dict:
    return {
        "status": "passed",
        "artifact": f"no-live-transport/{name}.json",
        "sha256": f"{name}-sha256",
    }


def no_live_transport_manifest_fixture(**overrides) -> dict:
    manifest = {
        "transport_id": "no-live-transport-fixture",
        "offline_only": True,
        "live_write": False,
        "contacts_wing": False,
        "contacts_backend": False,
        "hardware_evidence": False,
        "readiness_closure_allowed": False,
        "live_authority_granted": False,
        "transport_controls": {key: True for key in REQUIRED_TRANSPORT_CONTROLS},
        "transport_evidence": {
            key: no_live_transport_evidence_fixture(key)
            for key in REQUIRED_TRANSPORT_EVIDENCE
        },
        "runtime_observations": {
            key: 0 for key in REQUIRED_ZERO_RUNTIME_OBSERVATIONS
        },
        "forbidden_transport_tokens": [],
    }
    manifest.update(overrides)
    return manifest


def operator_report_binding_fixture(name: str) -> dict:
    return {
        "status": "passed",
        "artifact": f"operator-report/{name}.json",
        "sha256": f"{name}-sha256",
    }


def operator_report_manifest_fixture(**overrides) -> dict:
    manifest = {
        "report_id": "operator-report-fixture",
        "offline_only": True,
        "live_write": False,
        "contacts_wing": False,
        "contacts_backend": False,
        "hardware_evidence": False,
        "readiness_closure_allowed": False,
        "live_authority_granted": False,
        "report_sections": {key: True for key in REQUIRED_REPORT_SECTIONS},
        "report_bindings": {
            key: operator_report_binding_fixture(key)
            for key in REQUIRED_REPORT_BINDINGS
        },
        "authority_denial": {
            key: False for key in OPERATOR_REPORT_REQUIRED_AUTHORITY_DENIAL_FLAGS
        },
        "visible_blockers": [
            "M001 waits for real WING read-only artifact",
            "operator review required",
        ],
        "next_operator_action": "Run WING read-only preflight with operator.",
        "audit": {
            key: f"operator-report/{key}"
            for key in OPERATOR_REPORT_REQUIRED_AUDIT_FIELDS
        },
    }
    manifest.update(overrides)
    return manifest


def test_progress_baseline_covers_runner_verifier_checks():
    progress = load_progress_module()
    runner = load_runner_module()
    missing_checks = sorted(
        set(runner.ARTIFACT_MANIFEST_VERIFIER_REQUIRED_CHECKS)
        - covered_verifier_checks(progress.CURRENT_FIRST_COMMAND_BASELINE_COVERAGE_MARKERS)
    )
    assert missing_checks == []


def test_current_state_doc_matches_autopilot_state_evidence():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    summary = state["last_validation_summary"]
    authority = summary["authority"]
    text = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    ledger_rows = [
        json.loads(line)
        for line in AUTOPILOT_RUN_LEDGER.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    latest_commit = re.search(
        r"Latest verified product/runtime commit:\s*`([0-9a-f]+) ",
        text,
    )
    assert latest_commit is not None
    assert latest_commit.group(1) == state["last_verified_commit"]

    latest_full_suite = re.search(
        r"latest verified full-suite run for `([0-9a-f]+)` passed with\s*`([^`]+)`",
        text,
    )
    assert latest_full_suite is not None
    assert latest_full_suite.group(1) == state["last_verified_commit"]
    assert summary["pytest"].rsplit(": ", 1)[-1] == latest_full_suite.group(2)
    ledger_full_suite_results = [
        result
        for row in ledger_rows
        for result in row.get("results", [])
        if re.fullmatch(r"\d+ passed, 1 skipped(?:, 1 warning)?(?: in [0-9.]+s)?", result)
    ]
    assert ledger_full_suite_results
    latest_ledger_full_suite = re.sub(r" in [0-9.]+s$", "", ledger_full_suite_results[-1])
    assert summary["pytest"].rsplit(": ", 1)[-1] == latest_ledger_full_suite

    assert summary["no_wing_artifact"] in text
    assert "tools/verify_no_wing_artifact_manifest.py" in text
    assert "failed_checks: []" in text

    for flag in ("live_authority_granted", "contacts_wing", "contacts_backend"):
        expected = f"{flag}: {str(authority[flag]).lower()}"
        assert expected in text

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert "M110-HW01 has not started" in text
    assert "current aggregate ledger is `operator_review_required`" in text
    assert "No real hardware evidence was produced" in text


def test_current_no_wing_quick_evidence_is_repo_local_and_fail_closed():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    summary_text = state["last_validation_summary"]["no_wing_quick"]
    artifact_dir = state["last_validation_summary"]["no_wing_artifact"]

    assert "/tmp/" not in summary_text
    assert "--artifact-root artifacts/no_wing_release_check" in summary_text
    summary_match = re.search(
        r"summary (artifacts/no_wing_release_check/[^;]+/summary\.json)",
        summary_text,
    )
    assert summary_match is not None

    summary_rel = summary_match.group(1)
    assert summary_rel == f"{artifact_dir}/summary.json"
    assert summary_rel.startswith("artifacts/no_wing_release_check/")
    summary_path = PROJECT_ROOT / summary_rel
    assert summary_path.exists()
    artifact_path = PROJECT_ROOT / artifact_dir
    manifest_path = artifact_path / "artifact-manifest.json"
    summary_md_path = artifact_path / "summary.md"
    assert manifest_path.exists()
    assert summary_md_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    verifier_report = manifest_verifier.verify_no_wing_artifact_manifest(artifact_path)

    assert verifier_report["status"] == "passed"
    assert verifier_report["failed_checks"] == []
    assert verifier_report["live_authority_granted"] is False
    assert verifier_report["readiness_closure_allowed"] is False
    assert verifier_report["readiness_matrix_update_allowed"] is False
    assert verifier_report["contacts_wing"] is False
    assert verifier_report["contacts_backend"] is False

    summary_sha256 = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    summary_md_sha256 = hashlib.sha256(summary_md_path.read_bytes()).hexdigest()
    manifest_artifacts = {
        row["artifact_id"]: row for row in manifest["artifacts"]
    }
    assert manifest["artifact_manifest_summary_json_sha256"] == summary_sha256
    assert manifest_artifacts["summary_json"]["sha256"] == summary_sha256
    assert manifest["artifact_manifest_summary_md_sha256"] == summary_md_sha256
    assert manifest_artifacts["summary_md"]["sha256"] == summary_md_sha256
    assert manifest["artifact_manifest_self_consistency_match"] is True
    assert manifest["artifact_manifest_self_consistency_mismatches"] == []
    assert manifest["artifact_manifest_contract_cross_surface_match"] is True
    assert manifest["artifact_manifest_contract_cross_surface_mismatches"] == []
    assert manifest["artifact_manifest_readiness_matrix_update_allowed"] is False
    assert manifest["artifact_manifest_readiness_closure_allowed"] is False
    assert manifest["artifact_manifest_live_authority_granted"] is False

    assert summary["status"] == "passed"
    assert summary["failed_checks"] == []
    assert summary["live_authority_granted"] is False
    assert summary["readiness_closure_allowed"] is False
    assert summary["readiness_matrix_update_allowed"] is False
    assert summary["artifact_manifest_readiness_matrix_update_allowed"] is False
    assert summary["wing_readonly_handoff_plan_verifier_summary_surface_match"] is True
    assert (
        summary["wing_readonly_handoff_plan_verifier_summary_surface_mismatches"] == []
    )
    assert (
        summary[
            "wing_readonly_handoff_plan_verifier_summary_surface_live_authority_granted"
        ]
        is False
    )
    assert (
        summary[
            "wing_readonly_handoff_plan_verifier_summary_surface_readiness_closure_allowed"
        ]
        is False
    )
    assert (
        summary[
            "wing_readonly_handoff_plan_verifier_summary_surface_readiness_matrix_update_allowed"
        ]
        is False
    )

    decision_ledger = summary["wing_readonly_preflight_review_decision_ledger"]
    decided_count = (
        decision_ledger["accepted_record_count"]
        + decision_ledger["rejected_record_count"]
        + decision_ledger["invalid_record_count"]
    )
    assert decided_count == decision_ledger["decision_record_count"]
    assert (
        decided_count + decision_ledger["unreviewed_candidate_count"]
        == decision_ledger["review_candidate_count"]
    )
    assert decision_ledger["unreviewed_candidate_count"] == len(
        decision_ledger["unreviewed_candidate_source_ids"]
    )
    if decision_ledger["unreviewed_candidate_count"]:
        assert decision_ledger["ledger_status"] == "operator_review_required"
        assert "operator_must_record_readonly_preflight_review_decision" in (
            decision_ledger["blocked_by"]
        )
        assert decision_ledger["manual_readiness_update_candidate_available"] is False
    assert decision_ledger["live_authority_granted"] is False
    assert decision_ledger["readiness_closure_allowed_by_decision"] is False
    assert decision_ledger["readiness_matrix_update_allowed_by_decision"] is False


def test_milestone_and_roadmap_stay_waiting_for_real_wing_evidence():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    latest_m005_guard = state["latest_m005_physical_reconnect_target_guard_commit"]
    active = ACTIVE_MILESTONE_DOC.read_text(encoding="utf-8")
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    console = CONSOLE_CAPABILITY_MATRIX.read_text(encoding="utf-8")
    blockers = AUTOPILOT_BLOCKERS.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["active_milestone"] == "M001"
    assert state["checkpoint_status"] == "waiting_for_external_validation"
    assert state["last_validation_summary"]["hardware_evidence"] is False
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False

    assert "ID: M001" in active
    assert "Status: WAITING_FOR_EXTERNAL_VALIDATION" in active
    assert "current aggregate ledger remains\n`operator_review_required`" in active
    assert "historical exact 2026-07-02" in active
    assert "M110: CP00-CP05 are complete" in active
    assert "Real hardware proof now requires WING Rack" in active
    assert "rollback_not_applied" in active
    assert latest_m005_guard["commit"] in active
    assert "Latest M005 physical reconnect target guard" in active

    assert "Exactly one milestone is active: `M001`" in roadmap
    assert "`WAITING_FOR_EXTERNAL_VALIDATION`" in roadmap
    assert "Current M001 waiting reason" in roadmap
    assert "accepted_decision_ready" in roadmap
    assert "rollback_not_applied" in roadmap
    assert latest_m005_guard["commit"] in roadmap
    assert "Latest M005 physical reconnect target guard" in roadmap
    assert markdown_table_row(roadmap, "M001")[-1] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert markdown_table_row(roadmap, "M002")[-1] == "WAITING_FOR_EXTERNAL_VALIDATION"
    for milestone in ("M003", "M004", "M005", "M006"):
        assert markdown_table_row(roadmap, milestone)[-1] == "BLOCKED_BY_M002"
    assert "Real read-only WING artifact is reviewed" in markdown_table_row(
        roadmap, "M002"
    )[2]

    assert "Full Autonomous Mixing Mode | NOT_STARTED" in capability
    assert "Locked until all safety and hardware gates pass" in capability
    assert "Supervised AutoMix Mode | SPECIFIED" in capability
    assert "WING live write scope is only `fader`/`gain`; real proof missing" in capability

    raw_read = markdown_table_row(console, "Raw state/read-only connection")
    physical = markdown_table_row(console, "Physical readback")
    fader = markdown_table_row(console, "Channel fader")
    gain = markdown_table_row(console, "Preamp trim/gain-down")
    assert raw_read[-2:] == ["M001/M002 real artifact", "none in setup"]
    assert physical[-2:] == ["accepted real artifact review", "none"]
    assert fader[-2:] == ["supervised fader proof", "none"]
    assert gain[-2:] == ["supervised gain-down proof", "none"]
    assert fader[3] == "no live write"
    assert gain[3] == "no live write"

    normalized_blockers = re.sub(r"\s+", " ", blockers)
    assert "real WING Rack read-only artifact" in normalized_blockers
    assert "WING_REAL_HARDWARE_READONLY_20260702.md" in blockers
    assert "temporary-root capture only" in blockers
    assert "cannot close readiness" in blockers
    assert latest_m005_guard["commit"] in blockers
    assert "Latest M005 physical reconnect target guard" in blockers
    assert "Real WING Rack hardware evidence is not available from this setup run" not in blockers
    assert "Control-plane docs/state can proceed without live hardware contact" in blockers


def test_autopilot_stop_lease_and_recovery_contract_stays_bound_to_state():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    contract = PROJECT_EXECUTION_CONTRACT.read_text(encoding="utf-8")
    skill = AUTOPILOT_SKILL.read_text(encoding="utf-8")
    plan = M001_PLAN.read_text(encoding="utf-8")
    active = ACTIVE_MILESTONE_DOC.read_text(encoding="utf-8")
    operations = AUTOPILOT_OPERATIONS.read_text(encoding="utf-8")
    heartbeat = AUTOPILOT_HEARTBEAT.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    blockers = AUTOPILOT_BLOCKERS.read_text(encoding="utf-8")

    assert state["stop_file"] == "ops/autopilot/STOP"
    assert state["writer_lease_dir"] == "ops/autopilot/WRITER_LEASE"
    assert state["writer_lease_stale_after_minutes"] == 30
    assert state["current_writer_lease"] is None

    stop_path = state["stop_file"]
    lease_path = state["writer_lease_dir"]
    owner_path = f"{lease_path}/owner.json"
    gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")

    for text in (contract, skill, operations):
        assert stop_path in text
        assert lease_path in text

    assert owner_path in operations
    assert lease_path + "/" in gitignore
    assert not (PROJECT_ROOT / owner_path).exists()
    assert ".codex_autoloop.lock" in contract
    assert ".codex_autoloop.lock" in skill
    assert "RUN_LEDGER.jsonl" in contract
    assert "RUN_LEDGER.jsonl" in operations
    assert "STOPPED" in contract
    assert "STOPPED" in skill
    assert "30 minutes" in operations
    assert "30" in contract

    assert "STOP, lock, stale-run and recovery behavior remain documented" in plan
    assert "Stop if `ops/autopilot/STOP` exists" in active
    assert "another writer owns a non-stale lease" in active
    assert "Writer lease: none" in heartbeat
    assert "`ops/autopilot/STOP` does not exist after setup" in blockers

    assert "## Stop procedure" in report
    assert "Create `ops/autopilot/STOP`" in report
    assert re.search(r"release\s+lease", report)
    assert "set status to `STOPPED`" in report
    assert "## Recovery procedure" in report
    for marker in ("STATE.json", "RUN_LEDGER.jsonl", "active plan", "git\ndiff"):
        assert marker in report
    assert "Resume from the first checkpoint whose required evidence is missing" in report


def test_autopilot_latest_ledger_timestamp_matches_current_state_surfaces():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    heartbeat = AUTOPILOT_HEARTBEAT.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    ledger_rows = [
        json.loads(line)
        for line in AUTOPILOT_RUN_LEDGER.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    latest_ledger_ts = ledger_rows[-1]["recorded_at_utc"]
    assert state["last_heartbeat_utc"] == latest_ledger_ts
    assert f"Last heartbeat UTC: {latest_ledger_ts}" in heartbeat
    assert f"- Latest heartbeat: {latest_ledger_ts}." in report
    assert f"Generated: {latest_ledger_ts}" in current


def test_autopilot_latest_ledger_authority_stays_fail_closed_while_external_blocked():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    ledger_rows = [
        json.loads(line)
        for line in AUTOPILOT_RUN_LEDGER.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    latest = ledger_rows[-1]
    latest_authority = latest["authority"]
    state_summary = state["last_validation_summary"]
    state_authority = state_summary["authority"]

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert latest["status"] == state["program_status"]
    assert latest["checkpoint"] == state["active_checkpoint"]
    assert latest["recorded_at_utc"] == state["last_heartbeat_utc"]
    operator_approved_proof_row = latest_authority.get("operator_approved") is True
    backend_only_validation_row = (
        latest_authority.get("backend_only_validation") is True
    )
    if operator_approved_proof_row:
        # Operator-approved supervised proof rows may contact hardware and
        # perform bounded writes, but must never widen standing authority.
        assert latest["event"].startswith("supervised_")
        for flag in ("live_authority_granted", "readiness_closure_allowed"):
            assert latest_authority[flag] is False
        assert state_authority["live_authority_granted"] is False
        assert state_authority["readiness_closure_allowed"] is False
    elif backend_only_validation_row:
        assert latest["event"].startswith("m110_")
        assert latest["hardware_contacted"] is False
        assert latest["live_writes"] is False
        assert latest_authority["contacts_backend"] is True
        assert latest_authority["contacts_wing"] is False
        for flag in (
            "live_authority_granted",
            "live_write",
            "readiness_closure_allowed",
        ):
            assert latest_authority[flag] is False
        # The canonical no-WING summary remains a separate backend-free
        # validation artifact; do not rewrite its observed contact flags.
        assert state_authority["contacts_backend"] is False
        assert state_authority["contacts_wing"] is False
    else:
        assert latest["hardware_contacted"] is False
        assert latest["live_writes"] is False
        for flag in (
            "contacts_backend",
            "contacts_wing",
            "live_authority_granted",
            "live_write",
            "readiness_closure_allowed",
        ):
            assert latest_authority[flag] is state_authority[flag] is False

    assert latest_authority["readiness_matrix_update_allowed"] is False
    assert (
        latest_authority["release_ready_live_family_count"]
        == state_summary["wing_correction_apply_readiness_release_ready_live_family_count"]
        == 0
    )
    assert (
        latest_authority["safety_coverage_live_allowed_row_count"]
        == len(state_summary["safety_coverage_matrix_live_allowed_row_ids"])
        == 0
    )
    assert (
        latest_authority["supervised_current_live_write_kind_count"]
        == len(state_summary["autonomy_mode_transition_supervised_current_live_write_kinds"])
        == 0
    )
    assert latest_authority["unrestricted_live_or_soundcheck_authority"] is False


def test_m110_current_truth_surfaces_track_cp05_launch_and_missing_usb_blocker():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    work = state["active_work"]
    evidence = json.loads(M110_CP05_EVIDENCE.read_text(encoding="utf-8"))
    gate = json.loads(M110_CP05_GATE.read_text(encoding="utf-8"))

    assert work["id"] == "M110"
    assert work["software_checkpoints"]["M110-CP05"] == (
        "complete_current_markers_verified"
    )
    assert work["hardware_checkpoints"]["M110-HW01"] == (
        "blocked_before_start_missing_wing_usb_coreaudio_device"
    )
    assert all(
        work["hardware_checkpoints"][checkpoint] == "pending"
        for checkpoint in ("M110-HW02", "M110-HW03", "M110-HW04", "M110-HW05")
    )
    assert evidence["status"] == "passed"
    assert all(evidence["checks"].values())
    assert gate["checkpoint"] == "M110-CP05"
    assert gate["hardware_enabled"] is True

    launch = work["launch_only"]
    assert launch["status"] == "verified"
    assert launch["reason"] == "proof_owned_backend_launch_verified"
    assert launch["exit_code"] == 0
    assert launch["contacts_backend"] is True
    assert launch["contacts_wing"] is False
    assert launch["opens_udp_or_osc"] is False
    assert launch["live_write"] is False

    capture = work["physical_capture_prerequisite"]
    assert capture["evidence_class"] == "ephemeral_local_enumeration"
    assert capture["reason"] == "coreaudio_capture_name_not_unique"
    assert capture["exact_name_candidate_count"] == 0
    assert capture["matching_wing_usb_device_count"] == 0
    assert capture["dante_virtual_soundcard_seen"] is True
    assert capture["dante_virtual_soundcard_accepted_as_substitute"] is False
    assert capture["audio_stream_opened"] is False

    assert work["backend_contacted_this_cycle"] is True
    assert work["hardware_contacted_this_cycle"] is False
    assert work["usb_capture_surface_contacted_this_cycle"] is False
    assert work["osc_or_udp_opened_this_cycle"] is False
    assert work["live_writes_this_cycle"] is False
    assert work["reset_applied_this_cycle"] is False
    assert work["corrections_applied_this_cycle"] == 0
    assert work["normal_live_allowlist_count"] == 0
    assert work["standing_live_authority_granted"] is False
    assert work["strict_real_live_score"] == 88
    assert work["target_achieved"] is False

    assert not (PROJECT_ROOT / "ops" / "autopilot" / "WRITER_LEASE").exists()
    expected_blocker = re.sub(r"\s+", " ", state["current_blocker"])
    expected_action = re.sub(r"\s+", " ", state["next_operator_action"])
    for surface_path in (
        CURRENT_STATE_DOC,
        ACTIVE_MILESTONE_DOC,
        ROADMAP_DOC,
        M001_PLAN,
        AUTOPILOT_BLOCKERS,
        AUTOPILOT_HEARTBEAT,
        AUTOPILOT_LAST_REPORT,
        BLOCKERS_DOC,
        MVP_PROGRESS_REPORT,
    ):
        surface = surface_path.read_text(encoding="utf-8")
        normalized = re.sub(r"\s+", " ", surface)
        assert expected_blocker in normalized
        assert expected_action in normalized
        for requirement in state["external_validation_requirements"]:
            assert requirement in surface

    for surface_path in (M110_PLAN, CURRENT_TASKS):
        surface = surface_path.read_text(encoding="utf-8")
        assert "CP05" in surface
        assert "physical WING" in surface
        assert "HW01" in surface
        assert "HW05" in surface
        assert "48x48/48 kHz" in surface or "48 inputs/48 outputs at 48 kHz" in surface

    current_tasks = CURRENT_TASKS.read_text(encoding="utf-8")
    assert "active task from `ops/autopilot/STATE.json`" in current_tasks
    assert "not a task selector" in current_tasks

    latest_ledger = [
        json.loads(line)
        for line in AUTOPILOT_RUN_LEDGER.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ][-1]
    expected_latest = re.sub(
        r"\s+",
        " ",
        (
            f"Latest ledger handoff row: `{latest_ledger['event']}` at "
            f"`{latest_ledger['recorded_at_utc']}`"
        ),
    )

    for surface_path in (
        CURRENT_STATE_DOC,
        ACTIVE_MILESTONE_DOC,
        ROADMAP_DOC,
        M001_PLAN,
        AUTOPILOT_BLOCKERS,
        AUTOPILOT_HEARTBEAT,
        AUTOPILOT_LAST_REPORT,
        BLOCKERS_DOC,
    ):
        surface = surface_path.read_text(encoding="utf-8")
        current_prefix = re.split(r"^## Historical", surface, maxsplit=1, flags=re.MULTILINE)[0]
        normalized_prefix = re.sub(r"\s+", " ", current_prefix)
        assert current_prefix.count("Next operator action from `STATE.json`:") == 1
        assert current_prefix.count("Latest ledger handoff row:") == 1
        assert expected_action in normalized_prefix
        assert expected_latest in normalized_prefix


def test_autopilot_latest_ledger_records_external_validation_handoff():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    heartbeat = AUTOPILOT_HEARTBEAT.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    ledger_rows = [
        json.loads(line)
        for line in AUTOPILOT_RUN_LEDGER.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    latest = ledger_rows[-1]
    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert latest["status"] == state["program_status"]
    assert latest["next_operator_action"] == state["next_operator_action"]
    assert latest["external_validation_requirements"] == state[
        "external_validation_requirements"
    ]
    assert latest["current_blocker"] == state["current_blocker"]
    for marker in (
        "Do not widen live authority",
        "M005",
        "disconnect/timeout/stale-state",
        "M006",
        "readiness review",
    ):
        assert marker in latest["next_operator_action"]
    for requirement in latest["external_validation_requirements"]:
        assert requirement in heartbeat
        assert requirement in report
        assert requirement in current


def test_autopilot_blockers_doc_records_external_validation_handoff():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    blockers = AUTOPILOT_BLOCKERS.read_text(encoding="utf-8")
    ledger_rows = [
        json.loads(line)
        for line in AUTOPILOT_RUN_LEDGER.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    latest = ledger_rows[-1]
    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert latest["status"] == state["program_status"]
    assert latest["current_blocker"] == state["current_blocker"]
    assert latest["next_operator_action"] == state["next_operator_action"]
    assert latest["external_validation_requirements"] == state[
        "external_validation_requirements"
    ]

    assert "## Current external validation requirements" in blockers
    for requirement in state["external_validation_requirements"]:
        assert f"- {requirement}" in blockers
    assert "Next operator action from `STATE.json`:" in blockers
    normalized_blockers = re.sub(r"\s+", " ", blockers)
    assert re.sub(r"\s+", " ", state["next_operator_action"]) in normalized_blockers
    assert re.sub(
        r"\s+",
        " ",
        (
            f"Latest ledger handoff row: `{latest['event']}` at "
            f"`{latest['recorded_at_utc']}`"
        ),
    ) in normalized_blockers
    for marker in (
        "accepted_decision_ready",
        "rollback_not_applied",
        "release-ready live correction families remain `0`",
        "live-allowed safety rows remain `0`",
        "current supervised live-write kinds remain `0`",
        "Live authority remains denied",
    ):
        assert marker in normalized_blockers


def test_active_milestone_and_m001_plan_record_external_validation_handoff():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    active = ACTIVE_MILESTONE_DOC.read_text(encoding="utf-8")
    plan = M001_PLAN.read_text(encoding="utf-8")
    ledger_rows = [
        json.loads(line)
        for line in AUTOPILOT_RUN_LEDGER.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    latest = ledger_rows[-1]
    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert latest["status"] == state["program_status"]
    assert latest["current_blocker"] == state["current_blocker"]
    assert latest["next_operator_action"] == state["next_operator_action"]
    assert latest["external_validation_requirements"] == state[
        "external_validation_requirements"
    ]

    expected_latest = re.sub(
        r"\s+",
        " ",
        (
            f"Latest ledger handoff row: `{latest['event']}` at "
            f"`{latest['recorded_at_utc']}`"
        ),
    )
    expected_next_action = re.sub(r"\s+", " ", state["next_operator_action"])
    for surface in (active, plan):
        normalized_surface = re.sub(r"\s+", " ", surface)
        assert "## Current external validation requirements" in surface
        for requirement in state["external_validation_requirements"]:
            assert f"- {requirement}" in surface
        assert "Next operator action from `STATE.json`:" in surface
        assert expected_next_action in normalized_surface
        assert expected_latest in normalized_surface
        for marker in (
            "accepted_decision_ready",
            "rollback_not_applied",
            "release-ready live correction families remain `0`",
            "live-allowed safety rows remain `0`",
            "current supervised live-write kinds remain `0`",
            "No live authority changed",
        ):
            assert marker in normalized_surface


def test_roadmap_records_external_validation_handoff():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    ledger_rows = [
        json.loads(line)
        for line in AUTOPILOT_RUN_LEDGER.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    latest = ledger_rows[-1]
    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert latest["status"] == state["program_status"]
    assert latest["current_blocker"] == state["current_blocker"]
    assert latest["next_operator_action"] == state["next_operator_action"]
    assert latest["external_validation_requirements"] == state[
        "external_validation_requirements"
    ]

    expected_latest = re.sub(
        r"\s+",
        " ",
        (
            f"Latest ledger handoff row: `{latest['event']}` at "
            f"`{latest['recorded_at_utc']}`"
        ),
    )
    normalized_roadmap = re.sub(r"\s+", " ", roadmap)
    assert "## Current external validation requirements" in roadmap
    for requirement in state["external_validation_requirements"]:
        assert f"- {requirement}" in roadmap
    assert "Next operator action from `STATE.json`:" in roadmap
    assert re.sub(r"\s+", " ", state["next_operator_action"]) in normalized_roadmap
    assert expected_latest in normalized_roadmap
    for marker in (
        "accepted_decision_ready",
        "rollback_not_applied",
        "release-ready live correction families remain `0`",
        "live-allowed safety rows remain `0`",
        "current supervised live-write kinds remain `0`",
        "No live authority changed",
    ):
        assert marker in normalized_roadmap


def test_project_wing_mvp_blockers_doc_records_external_validation_handoff():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    project_blockers = BLOCKERS_DOC.read_text(encoding="utf-8")
    ledger_rows = [
        json.loads(line)
        for line in AUTOPILOT_RUN_LEDGER.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    latest = ledger_rows[-1]
    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert latest["status"] == state["program_status"]
    assert latest["current_blocker"] == state["current_blocker"]
    assert latest["next_operator_action"] == state["next_operator_action"]
    assert latest["external_validation_requirements"] == state[
        "external_validation_requirements"
    ]

    expected_latest = re.sub(
        r"\s+",
        " ",
        (
            f"Latest ledger handoff row: `{latest['event']}` at "
            f"`{latest['recorded_at_utc']}`"
        ),
    )
    normalized_project_blockers = re.sub(r"\s+", " ", project_blockers)
    assert "## Current external validation handoff" in project_blockers
    for requirement in state["external_validation_requirements"]:
        assert f"- {requirement}" in project_blockers
    assert "Next operator action from `STATE.json`:" in project_blockers
    assert (
        re.sub(r"\s+", " ", state["next_operator_action"])
        in normalized_project_blockers
    )
    assert expected_latest in normalized_project_blockers
    for marker in (
        "accepted_decision_ready",
        "rollback_not_applied",
        "release-ready live correction families remain `0`",
        "live-allowed safety rows remain `0`",
        "current supervised live-write kinds remain `0`",
        "No live authority changed",
    ):
        assert marker in normalized_project_blockers


def test_mvp_progress_report_records_external_validation_handoff():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    progress_report = MVP_PROGRESS_REPORT.read_text(encoding="utf-8")
    ledger_rows = [
        json.loads(line)
        for line in AUTOPILOT_RUN_LEDGER.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    latest = ledger_rows[-1]
    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert latest["status"] == state["program_status"]
    assert latest["current_blocker"] == state["current_blocker"]
    assert latest["next_operator_action"] == state["next_operator_action"]
    assert latest["external_validation_requirements"] == state[
        "external_validation_requirements"
    ]

    normalized_progress = re.sub(r"\s+", " ", progress_report)
    assert "## Current External Validation Handoff" in progress_report
    assert "Program status from `STATE.json`: `WAITING_FOR_EXTERNAL_VALIDATION`" in progress_report
    score_gate = state["strict_pre_release_review"]
    assert (
        f"Strict pre-release review score: `{score_gate['score']}/100`"
        in progress_report
    )
    assert score_gate["score"] < score_gate["release_threshold"]
    assert score_gate["score_ceiling_reason"] in normalized_progress
    assert "frozen legacy M001 packet contract" in progress_report
    for requirement in state["external_validation_requirements"]:
        assert f"  - {requirement}" in progress_report
    for requirement in score_gate["required_before_95"]:
        assert f"  - {requirement}" in progress_report
    assert (
        re.sub(r"\s+", " ", state["current_blocker"])
        in normalized_progress
    )
    assert (
        re.sub(r"\s+", " ", state["next_operator_action"])
        in normalized_progress
    )
    assert re.sub(
        r"\s+",
        " ",
        (
            f"Latest ledger handoff row: `{latest['event']}` at "
            f"`{latest['recorded_at_utc']}`"
        ),
    ) in normalized_progress
    for marker in (
        "accepted_decision_ready",
        "rollback_not_applied",
        "release-ready live correction families remain `0`",
        "live-allowed safety rows remain `0`",
        "current supervised live-write kinds remain `0`",
        "No live authority changed",
    ):
        assert marker in normalized_progress


def test_m006_owner_decision_handoff_surfaces_track_latest_verifier_commit():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    latest_verifier = state["latest_m006_owner_decision_verifier_commit"]
    latest_packet_check = state["latest_m006_owner_decision_stale_event_packet_check_commit"]
    expected_commit = latest_verifier["commit"]

    assert expected_commit == latest_packet_check["commit"]
    assert latest_verifier["requires_packet_verifier_m005_stale_event_checks"] is True
    assert latest_verifier["requires_packet_verifier_m005_reconnect_event_checks"] is True
    assert latest_verifier["requires_packet_verifier_m005_events_path_binding"] is True
    assert latest_packet_check["requires_packet_verifier_m005_reconnect_event_checks"] is True
    assert latest_packet_check["requires_packet_verifier_m005_events_path_binding"] is True

    for surface_path in (
        ACTIVE_MILESTONE_DOC,
        M001_PLAN,
        ROADMAP_DOC,
        BLOCKERS_DOC,
    ):
        surface = surface_path.read_text(encoding="utf-8")
        latest_mentions = re.findall(
            r"Latest M006 owner decision verifier: `([0-9a-f]+)`",
            surface,
        )
        assert latest_mentions == [expected_commit]
        normalized_surface = re.sub(r"\s+", " ", surface)
        assert "current M005 stale-readback event-evidence checks" in normalized_surface
        assert "matching reconnect-recovery physical readback event checks" in normalized_surface
        assert "accepted M005 summary sibling `events.jsonl`" in normalized_surface
        assert "packet-verifier" in normalized_surface
        assert "reason=m006_owner_decision_failed_closed" in normalized_surface
        assert "readiness-matrix update" in normalized_surface


def test_strict_pre_release_score_gate_is_state_backed_and_fail_closed():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    progress_report = MVP_PROGRESS_REPORT.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    score_gate = state["strict_pre_release_review"]

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert score_gate["score"] == 88
    assert score_gate["release_threshold"] == 95
    assert score_gate["score"] < score_gate["release_threshold"]
    assert score_gate["public_release_ready"] is False
    assert "not public-release ready" in score_gate["public_release_scope"]
    assert "normal supervised transaction live kinds are empty" in score_gate["public_release_scope"]
    assert score_gate["required_before_95"]
    assert any(
        "operator-present canonical backend transaction proof on the exact WING Rack"
        in requirement
        for requirement in score_gate["required_before_95"]
    )
    assert any(
        "real current-tree arm64 packaged-app launch smoke"
        in requirement
        for requirement in score_gate["required_before_95"]
    )
    assert any(
        "M006 technical narrow promotion for the already recorded owner decision"
        in requirement
        for requirement in score_gate["required_before_95"]
    )
    assert score_gate["evidence"][0] == (
        "Docs/reports/FULL_PROJECT_READINESS_AUDIT_20260711.md"
    )
    assert any(
        path.startswith("artifacts/no_wing_release_check/")
        for path in score_gate["evidence"]
    )
    assert any("wing_operator_proposal_ch3_fader" in path for path in score_gate["evidence"])
    assert any("wing_operator_proposal_ch3_gain" in path for path in score_gate["evidence"])
    for artifact in score_gate["evidence"]:
        assert (PROJECT_ROOT / artifact).is_file()
    for artifact in score_gate["evidence"]:
        if "wing_operator_proposal_validation" not in artifact:
            continue
        artifact_summary = json.loads(
            (PROJECT_ROOT / artifact).read_text(encoding="utf-8")
        )
        assert artifact_summary["status"] == "passed"
        assert artifact_summary["rollback_result"]["status"] == "applied"
        assert (
            artifact_summary["post_emergency_apply_result"]["status"]
            == "apply_blocked"
        )

    for marker in (
        "canonical proposal-only WING WebSocket surfaces",
        "strict address-bound physical readback and rollback CAS",
        "one per-user writer lock and recovery WAL",
        "atomic proposal claims",
        "exact browser Origin and memory-only Electron token bootstrap",
        "nested unbound physical claims",
        "runtime identity v2",
        "initial non-observation connection before baseline plus observation-only reconnect",
        "self-hashed embedded source manifest",
        "authenticated launcher-token and observed backend PID binding",
        "all four canonical stores unchanged",
    ):
        assert marker in score_gate["score_ceiling_reason"]

    normalized_progress = re.sub(r"\s+", " ", progress_report)
    normalized_current = re.sub(r"\s+", " ", current)
    normalized_report = re.sub(r"\s+", " ", report)
    assert (
        f"Strict pre-release review score: `{score_gate['score']}/100`"
        in progress_report
    )
    assert (
        f"Pre-release score threshold: `{score_gate['release_threshold']}/100`"
        in progress_report
    )
    assert (
        "Public release ready from score gate: `false` "
        "(no standing supervised transaction authority)"
        in progress_report
    )
    assert score_gate["score_ceiling_reason"] in normalized_progress
    for surface in (normalized_current, normalized_report):
        assert f"strict pre-release score is `{score_gate['score']}/100`" in surface
        assert f"threshold remains `{score_gate['release_threshold']}/100`" in surface
        assert "public release ready: `false`" in surface
        assert "no standing supervised transaction authority" in surface
        assert score_gate["score_ceiling_reason"] in surface

    assert state["truth_revision"] == (
        "m110_cp05_launch_verified_wing_usb_missing_strict_live_88_zero_authority_20260714"
    )
    assert state["full_release_truth_source"] == "strict_pre_release_review"
    assert state["full_release_requirements_source"] == (
        "strict_pre_release_review.required_before_95"
    )
    scoped_run = state["latest_m006_scoped_live_soundcheck"]
    assert scoped_run["evidence_class"] == "historical_operator_approved_execution"
    assert scoped_run["accepted_for_current_release_readiness"] is False
    assert scoped_run["standing_live_authority_granted"] is False
    assert scoped_run["superseded_by"] == "strict_pre_release_review"
    assert "live_allowed_write_kinds" not in scoped_run
    assert scoped_run["historical_bounded_write_kinds"]
    assert "Historical checkpoints — superseded" in report


def test_autopilot_operator_reports_match_latest_full_suite_evidence():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    heartbeat = AUTOPILOT_HEARTBEAT.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    ledger_rows = [
        json.loads(line)
        for line in AUTOPILOT_RUN_LEDGER.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    expected_result = state["last_validation_summary"]["pytest"].rsplit(": ", 1)[-1]
    ledger_full_suite_results = [
        result
        for row in ledger_rows
        for result in row.get("results", [])
        if re.fullmatch(r"\d+ passed, 1 skipped(?:, 1 warning)?(?: in [0-9.]+s)?", result)
    ]
    assert ledger_full_suite_results
    latest_ledger_result = re.sub(r" in [0-9.]+s$", "", ledger_full_suite_results[-1])
    assert latest_ledger_result == expected_result

    assert f"Latest verified full pytest passed with `{expected_result}`." in heartbeat
    assert re.search(
        rf"latest verified full suite passed with\s*`{re.escape(expected_result)}`",
        report,
        re.IGNORECASE,
    )


def test_goal_transition_fallback_contract_stays_report_only_without_native_goal_tools():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    contract = PROJECT_EXECUTION_CONTRACT.read_text(encoding="utf-8")
    skill = AUTOPILOT_SKILL.read_text(encoding="utf-8")
    blockers = AUTOPILOT_BLOCKERS.read_text(encoding="utf-8")
    heartbeat = AUTOPILOT_HEARTBEAT.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["active_milestone"] == "M001"
    assert state["checkpoint_status"] == "waiting_for_external_validation"
    assert state["last_validation_summary"]["hardware_evidence"] is False
    assert state["last_validation_summary"]["product_runtime_changed"] is True
    assert "SmartDSP strategy scoring" in state["last_validation_summary"][
        "product_runtime_change_scope"
    ]
    assert "offline/dry-run decision evidence" in state["last_validation_summary"][
        "product_runtime_change_scope"
    ]
    assert "DSP parameter calculation summaries" in state["last_validation_summary"][
        "product_runtime_change_scope"
    ]
    assert "problem-first action decision summaries" in state[
        "last_validation_summary"
    ]["product_runtime_change_scope"]
    assert (
        "weighted score contributions"
        in state["last_validation_summary"]["smart_dsp_weighted_score_evidence"]
    )
    assert (
        "wing_native_strategy_fit"
        in state["last_validation_summary"][
            "smart_dsp_wing_native_strategy_fit_evidence"
        ]
    )
    assert (
        "no lower safety score"
        in state["last_validation_summary"][
            "smart_dsp_wing_native_strategy_fit_evidence"
        ]
    )
    assert (
        "dsp_parameter_calculation_summary"
        in state["last_validation_summary"][
            "smart_dsp_parameter_calculation_summary_evidence"
        ]
    )
    assert (
        "proposal_only=true"
        in state["last_validation_summary"][
            "smart_dsp_parameter_calculation_summary_evidence"
        ]
    )
    assert (
        "problem_action_decision_summary"
        in state["last_validation_summary"][
            "smart_dsp_problem_action_decision_evidence"
        ]
    )
    assert (
        "selection_basis"
        in state["last_validation_summary"][
            "smart_dsp_problem_action_decision_evidence"
        ]
    )
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    if (
        state["last_validation_summary"].get("wing_readonly_preflight_review_status")
        == "candidate_ready_for_manual_review"
    ):
        if (
            state["last_validation_summary"].get(
                "wing_readonly_preflight_decision_status"
            )
            == "accepted_decision_ready"
        ):
            assert "Do not widen live authority" in state["next_action"]
            assert "M005" in state["next_action"]
            assert "M006" in state["next_action"]
        else:
            assert (
                "Operator reviews artifacts/wing_readonly_preflight" in state["next_action"]
                or "Operator uses the Operator Decision Queue" in state["next_action"]
            )
            if "Operator Decision Queue" in state["next_action"]:
                assert (
                    "Docs/reports/WING_READONLY_PREFLIGHT_DECISION_REVIEW_LATEST.md"
                    in state["next_action"]
                )
                assert (
                    state["last_validation_summary"].get(
                        "wing_readonly_preflight_operator_decision_queue_report"
                    )
                    == "Docs/reports/WING_READONLY_PREFLIGHT_DECISION_REVIEW_LATEST.md"
                )
            else:
                assert (
                    "artifacts/wing_readonly_preflight_review_decisions/"
                    "pending_review_decision_template.json"
                ) in state["next_action"]
    else:
        assert "operator-supervised WING read-only preflight" in state["next_action"]

    assert "## Goal transition" in contract
    assert "Native thread Goal management" in contract
    assert "goal tools" in contract
    assert "native Goal update unavailable" in contract
    assert "`LAST_REPORT.md`" in contract
    assert "`/goal`" in contract
    assert "state\nchecks" in contract
    assert "Codex threads/goals" in contract
    assert "Automixer product files" in contract
    assert "hardware" in contract
    assert "single-writer serialization" in contract

    assert "active Goal" in contract
    assert "RUN_LEDGER.jsonl" in contract
    assert "Do not mark a checkpoint complete while required checks fail" in skill
    assert re.search(
        r"Always name exact files, commands, evidence paths, blockers and next\s+action",
        skill,
    )

    assert "Future heartbeat wake-ups must verify native Goal transition tools before" in blockers
    assert "replacing a thread Goal" in blockers
    assert "Control-plane docs/state can proceed without live hardware contact" in blockers

    assert "Writer lease: none" in heartbeat
    assert "This heartbeat includes one local product/runtime change" in heartbeat
    assert "SmartDSP strategy\nsearch now reports mode-specific weighted score" in heartbeat
    assert "WING-native\nstrategy-fit evidence" in heartbeat
    assert "`dsp_parameter_calculation_summary`" in heartbeat
    assert "`problem_action_decision_summary`" in heartbeat
    assert "no live console authority changed" in heartbeat
    if (
        state["last_validation_summary"].get("wing_readonly_preflight_review_status")
        == "candidate_ready_for_manual_review"
    ):
        if (
            state["last_validation_summary"].get(
                "wing_readonly_preflight_decision_status"
            )
            == "accepted_decision_ready"
        ):
            assert "accepted_decision_ready" in heartbeat
            assert "rollback_not_applied" in heartbeat
            assert "release readiness remains\nblocked" in heartbeat
        else:
            assert "latest real WING read-only candidate" in heartbeat
            assert "operator_review_required" in heartbeat
            if "Operator Decision Queue" in heartbeat:
                assert "unreviewed candidates" in heartbeat
                assert "accepted or rejected operator decision records" in heartbeat
            else:
                assert "pending_operator_decision" in heartbeat
                assert "required_edits" in heartbeat
    else:
        assert "M001 still has no real WING read-only artifact" in heartbeat

    assert "## First Goal" in report
    assert "The next finite Goal should only be created after a real\nWING read-only artifact" in report
    assert "another explicitly safe\nindependent roadmap item" in report
    assert "Do not contact\nWING from unattended automation" in report
    assert "There is no dependency-ready autonomous\nroadmap milestone" in report


def test_project_completion_gate_stays_closed_without_real_hardware_and_owner_evidence():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    summary = state["last_validation_summary"]
    gate = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    goal = PROJECT_GOAL.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    console = CONSOLE_CAPABILITY_MATRIX.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["checkpoint_status"] == "waiting_for_external_validation"
    assert summary["hardware_evidence"] is False
    assert summary["authority"]["live_authority_granted"] is False
    assert summary["authority"]["readiness_closure_allowed"] is False
    assert summary["authority"]["contacts_wing"] is False
    assert summary["authority"]["contacts_backend"] is False

    assert "Status: authoritative completion gate" in gate
    assert "evidence-backed" in gate
    assert "hardware, listening или owner validation" in gate
    assert "WAITING_FOR_EXTERNAL_VALIDATION`, а не `COMPLETE`" in gate
    assert "## Gate Conditions" in gate
    assert "Full\n   Autonomous Mixing Mode" in gate
    assert "staged real-hardware evidence" in gate
    assert "High-risk console actions" in gate
    assert "human supervision" in gate
    assert "Threshold Policy" in gate
    assert "PENDING_APPROVAL" in gate
    assert "COMPLETE` запрещен" in gate
    assert "UNKNOWN" in gate

    assert "Full Autonomous Mixing Mode не может быть объявлен готовым" in goal
    assert "зеленому unit test" in goal
    assert "Offline, shadow, replay, mock и simulator evidence" in goal
    assert "не являются hardware proof" in goal
    assert "supervised `fader`/`gain` rail" in goal

    full_auto = markdown_table_row(capability, "Full Autonomous Mixing Mode")
    supervised = markdown_table_row(capability, "Supervised AutoMix Mode")
    multichannel = markdown_table_row(capability, "Multichannel audio analysis")
    assert full_auto[1] == "NOT_STARTED"
    assert full_auto[-1] == "none"
    assert "all safety and hardware gates pass" in full_auto[4]
    assert full_auto[-2] == "completion gate review"
    assert supervised[1] == "SPECIFIED"
    assert "real proof missing" in supervised[4]
    assert supervised[-1] == "none"
    assert multichannel[-1] == "UNKNOWN"
    assert "Replay, mock and simulator evidence must not be labeled as hardware evidence" in capability

    assert markdown_table_row(roadmap, "M001")[-1] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert markdown_table_row(roadmap, "M002")[-1] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert markdown_table_row(roadmap, "M010")[-1] == "PENDING_APPROVAL"
    assert "Full autonomous eligibility review" in markdown_table_row(roadmap, "M010")[1]
    assert "M001-M009" in markdown_table_row(roadmap, "M010")[3]
    assert "AUTONOMOUS_ELIGIBLE only if all gates pass" in markdown_table_row(
        roadmap, "M010"
    )[5]

    for category in (
        "Raw state/read-only connection",
        "Physical readback",
        "Channel fader",
        "Preamp trim/gain-down",
        "Rollback",
        "Emergency stop",
    ):
        assert markdown_table_row(console, category)[-1] in {"none", "none in setup"}

    assert "Full Autonomous Mixing Mode is not ready" in current
    assert "No real hardware evidence was produced" in current
    assert "Goal transition fallback behavior is now covered" in current
    assert "Hardware input required now: yes" in report
    assert "These commits do not create hardware proof" in report
    assert "grant live authority" in report
    assert "Do not contact\nWING from unattended automation" in report


def test_validate_only_preflight_evidence_stays_out_of_real_wing_artifact_root():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    summary = state["last_validation_summary"]
    authority = summary["authority"]
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    heartbeat = AUTOPILOT_HEARTBEAT.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    handoff = CP04_HANDOFF_REPORT.read_text(encoding="utf-8")
    ready = json.loads(VALIDATE_INPUTS_READY_SUMMARY.read_text(encoding="utf-8"))
    mismatch = json.loads(VALIDATE_INPUTS_MISMATCH_SUMMARY.read_text(encoding="utf-8"))

    validate_only_artifacts = summary["cp04_validate_input_artifacts"]
    expected_artifacts = [
        "ops/autopilot/evidence/M001_CP04_validate_inputs_expected_plan_mismatch/summary.json",
        "ops/autopilot/evidence/M001_CP04_validate_inputs_ready_for_websocket/summary.json",
    ]

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert summary["hardware_evidence"] is False
    assert summary["product_runtime_changed"] is True
    assert "SmartDSP strategy scoring" in summary["product_runtime_change_scope"]
    assert "offline/dry-run decision evidence" in summary["product_runtime_change_scope"]
    assert "DSP parameter calculation summaries" in summary["product_runtime_change_scope"]
    assert "problem-first action decision summaries" in summary[
        "product_runtime_change_scope"
    ]
    assert "weighted score contributions" in summary["smart_dsp_weighted_score_evidence"]
    assert (
        "wing_native_strategy_fit"
        in summary["smart_dsp_wing_native_strategy_fit_evidence"]
    )
    assert "no lower safety score" in summary[
        "smart_dsp_wing_native_strategy_fit_evidence"
    ]
    assert (
        "selected_dsp_parameter_calculation_summary"
        in summary["smart_dsp_parameter_calculation_summary_evidence"]
    )
    assert "live_write=false" in summary[
        "smart_dsp_parameter_calculation_summary_evidence"
    ]
    assert (
        "selected_problem_action_decision_summary"
        in summary["smart_dsp_problem_action_decision_evidence"]
    )
    assert "proposal_only=true" in summary[
        "smart_dsp_problem_action_decision_evidence"
    ]
    assert summary["milestone_decision"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert authority["contacts_wing"] is False
    assert authority["contacts_backend"] is False
    assert authority["live_write"] is False
    assert authority["live_authority_granted"] is False
    assert authority["readiness_closure_allowed"] is False

    assert summary["evidence_placement"] == (
        "validate-only summaries are stored under ops/autopilot/evidence, "
        "not artifacts/wing_readonly_preflight"
    )
    assert validate_only_artifacts == expected_artifacts
    assert all(
        not artifact.startswith("artifacts/wing_readonly_preflight")
        for artifact in validate_only_artifacts
    )
    real_wing_artifacts = [
        path.relative_to(PROJECT_ROOT).as_posix()
        for path in files_under(REAL_WING_ARTIFACT_ROOT)
    ]
    assert not set(validate_only_artifacts).intersection(real_wing_artifacts)
    assert all("M001_CP04_validate_inputs" not in path for path in real_wing_artifacts)

    for payload in (ready, mismatch):
        assert payload["validation_only"] is True
        assert payload["read_only"] is True
        assert payload["websocket_opened"] is False
        assert payload["write_request_sent"] is False
        assert payload["contacts_wing"] is False
        assert payload["contacts_backend"] is False
        assert payload["live_write"] is False
        assert payload["mutates_console_state"] is False
        assert payload["queries_console"] is False
        assert payload["readiness_closure_allowed"] is False
        assert payload["readiness_matrix_update_allowed"] is False
        assert payload["live_authority_granted"] is False
        assert payload["artifact_root"] == "artifacts/wing_readonly_preflight"

    assert ready["status"] == "ready_for_websocket"
    assert ready["reason"] == "run_inputs_validated_no_websocket"
    assert ready["expected_plan_match"] is True
    assert ready["execution_allowed_by_validation"] is False
    assert mismatch["status"] == "blocked"
    assert mismatch["reason"] == "expected_plan_mismatch"
    assert mismatch["expected_plan_match"] is False

    for artifact in expected_artifacts:
        assert artifact in report
        assert artifact in handoff
    assert "Validate-only summaries were moved to `ops/autopilot/evidence/`" in report
    assert "not treated as real `artifacts/wing_readonly_preflight` candidates" in report
    if summary.get("wing_readonly_preflight_review_status") == (
        "candidate_ready_for_manual_review"
    ):
        if summary.get("wing_readonly_preflight_decision_status") == (
            "accepted_decision_ready"
        ):
            assert "accepted_decision_ready" in heartbeat
            assert "rollback_not_applied" in heartbeat
            assert "canonical read-only decision ledger is accepted-ready" in heartbeat
        else:
            normalized_heartbeat = re.sub(r"\s+", " ", heartbeat)
            assert "latest real WING read-only candidate" in heartbeat
            assert "operator_review_required" in heartbeat
            assert "accepted hardware evidence" in normalized_heartbeat
    else:
        assert "M001 still has no real WING read-only artifact under\n" in heartbeat
        assert "`artifacts/wing_readonly_preflight/`" in heartbeat
    assert "No real hardware evidence was produced" in current
    assert "`artifacts/wing_readonly_preflight` root" in current
    assert "websocket_opened: false" in handoff
    assert "write_request_sent: false" in handoff


def test_m002_physical_readback_waits_for_accepted_real_wing_review():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    active = ACTIVE_MILESTONE_DOC.read_text(encoding="utf-8")
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    autopilot_blockers = AUTOPILOT_BLOCKERS.read_text(encoding="utf-8")
    project_blockers = BLOCKERS_DOC.read_text(encoding="utf-8")
    plan = M001_PLAN.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["active_checkpoint"] == "M001-EXTERNAL-WING-READONLY-ARTIFACT"
    assert state["last_validation_summary"]["hardware_evidence"] is False
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    if (
        state["last_validation_summary"].get("wing_readonly_preflight_review_status")
        == "candidate_ready_for_manual_review"
    ):
        if (
            state["last_validation_summary"].get(
                "wing_readonly_preflight_decision_status"
            )
            == "accepted_decision_ready"
        ):
            assert state["last_validation_summary"][
                "wing_readonly_preflight_decision_ledger_status"
            ] == "operator_review_decision_records_available"
            assert state["last_validation_summary"][
                "wing_readonly_preflight_unreviewed_candidate_count"
            ] == 0
            assert state["last_validation_summary"][
                "wing_readonly_preflight_unreviewed_candidate_source_ids"
            ] == []
            assert state["last_validation_summary"][
                "wing_operator_proposal_validation_latest_status"
            ] == "passed"
            assert state["last_validation_summary"][
                "wing_operator_proposal_validation_latest_reason"
            ] == "supervised_wing_mvp_gain_down_proof_ch3"
            assert state["last_validation_summary"][
                "supervised_ch3_fader_proof_rerun"
            ].startswith("passed: ")
            assert state["last_validation_summary"][
                "supervised_ch3_gain_down_proof"
            ].startswith("passed: ")
            assert state["next_operator_action"] == state["next_action"]
            assert "Do not widen live authority" in state["next_action"]
            for marker in (
                "M005",
                "disconnect/timeout/stale-state",
                "M006",
                "readiness review",
            ):
                assert marker in state["next_operator_action"]
        else:
            assert state["last_validation_summary"][
                "wing_readonly_preflight_decision_ledger_status"
            ] == "operator_review_required"
            assert state["last_validation_summary"][
                "wing_readonly_preflight_review_template_required_edits_present"
            ] is True
            unreviewed_source_ids = state["last_validation_summary"][
                "wing_readonly_preflight_unreviewed_candidate_source_ids"
            ]
            assert unreviewed_source_ids
            assert state["last_validation_summary"][
                "wing_readonly_preflight_unreviewed_candidate_count"
            ] == len(unreviewed_source_ids)
            assert state["last_validation_summary"][
                "wing_readonly_preflight_latest_valid_candidate_source_id"
            ] == unreviewed_source_ids[-1]
            assert state["last_validation_summary"][
                "wing_readonly_preflight_latest_unreviewed_candidate_source_id"
            ] == unreviewed_source_ids[-1]
            assert state["last_validation_summary"][
                "wing_readonly_preflight_manual_readiness_update_candidate_available"
            ] is False
            assert state["last_validation_summary"][
                "wing_readonly_preflight_operator_decision_queue_report"
            ] == "Docs/reports/WING_READONLY_PREFLIGHT_DECISION_REVIEW_LATEST.md"
            assert state["next_operator_action"] == state["next_action"]
            assert "Do not widen live authority" in state["next_action"]
    else:
        assert state["next_operator_action"] == state["next_action"]
        assert state["next_action"] == (
            "Wait for operator-supervised WING read-only preflight using "
            "ops/autopilot/M001_CP04_wing_readonly_preflight_plan.json."
        )
    for marker in (
        "Physical WING USB/CoreAudio device named WING",
        "M110-HW01 through M110-HW05",
        "two valid 2026-07-13 read-only preflight candidates",
        "Current-release provenance rebind for the closed M005",
        "Technical M006 readiness review promotion",
    ):
        assert any(marker in item for item in state["external_validation_requirements"])

    m002 = markdown_table_row(roadmap, "M002")
    m003 = markdown_table_row(roadmap, "M003")
    m004 = markdown_table_row(roadmap, "M004")
    m005 = markdown_table_row(roadmap, "M005")
    m006 = markdown_table_row(roadmap, "M006")
    assert m002[1] == "Physical console readback intake"
    assert m002[2] == (
        "Real read-only WING artifact is reviewed and bound to accepted "
        "decision records"
    )
    assert m002[3] == "M001 plus WING hardware/operator"
    assert m002[-1] == "WAITING_FOR_EXTERNAL_VALIDATION"
    for row in (m003, m004, m005, m006):
        assert row[-1] == "BLOCKED_BY_M002"
    assert m006[2] == "MVP readiness matrix updates only from accepted real proof"

    readonly = markdown_table_row(project_blockers, "WING read-only connection proof")
    physical_readback = markdown_table_row(project_blockers, "Physical console readback proof")
    readiness = markdown_table_row(project_blockers, "Readiness matrix update based on real proof")
    assert readonly[1] == "tooling_ready_waits_real_wing"
    assert physical_readback[1] == "tooling_ready_waits_real_wing"
    assert "Run read-only preflight with `--require-physical-readback`" in physical_readback[3]
    assert readiness[1] == "blocked_by_real_artifact_review"
    assert "Update readiness only after manual review of real WING artifacts" in readiness[3]
    assert "do not close rows from offline evidence" in readiness[3]

    if (
        state["last_validation_summary"].get("wing_readonly_preflight_decision_status")
        == "accepted_decision_ready"
    ):
        normalized_autopilot_blockers = re.sub(r"\s+", " ", autopilot_blockers)
        assert "real WING Rack read-only artifact operator decision gate" in (
            normalized_autopilot_blockers
        )
        assert "failed rollback" in autopilot_blockers
        assert "release readiness" in autopilot_blockers
    else:
        assert "operator_review_required" in autopilot_blockers
        assert "later candidates inherit no authority" in autopilot_blockers
    assert "CH3 fader state restoration/verification" in active
    assert "current aggregate machine status is `operator_review_required`" in plan
    assert "historical exact 2026-07-02 source retains `accepted_decision_ready`" in plan
    assert "failed CH3 fader rollback proof" in plan
    assert "operator captures and reviews a real WING read-only artifact" not in plan
    assert "M110 CP05/launch checkpoint and physical USB blocker" in current
    assert "current aggregate ledger is `operator_review_required`" in current
    assert re.search(r"CH3 fader state\s+restoration/verification", current)
    assert "physical WING USB missing" in report
    if (
        state["last_validation_summary"].get("wing_readonly_preflight_decision_status")
        == "accepted_decision_ready"
    ):
        assert "read-only decision gate is accepted-ready" in report
        assert "rollback_not_applied" in report
    else:
        assert "operator_review_required" in report
        assert "Two valid 2026-07-13 read-only candidates" in report
    assert "There is no dependency-ready autonomous\nroadmap milestone" in report
    assert "until that external validation exists" in report


def test_real_wing_interaction_plan_matches_accepted_decision_rollback_blocker():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    summary = state["last_validation_summary"]
    plan = M001_REAL_WING_PLAN.read_text(encoding="utf-8")

    assert summary["wing_readonly_preflight_decision_status"] == "operator_review_required"
    assert summary["wing_readonly_preflight_accepted_record_count"] == 1
    assert summary["wing_readonly_preflight_decision_record_count"] == (
        summary["wing_readonly_preflight_accepted_record_count"]
        + summary["wing_readonly_preflight_rejected_record_count"]
        + summary["wing_readonly_preflight_invalid_record_count"]
    )
    assert summary["wing_readonly_preflight_unreviewed_candidate_count"] == len(
        summary["wing_readonly_preflight_unreviewed_candidate_source_ids"]
    )
    assert summary["wing_readonly_preflight_unreviewed_candidate_count"] > 0
    assert (
        summary["wing_readonly_preflight_decision_record_count"]
        + summary["wing_readonly_preflight_unreviewed_candidate_count"]
        == summary["wing_readonly_preflight_review_candidate_count"]
    )
    assert summary["wing_operator_proposal_validation_latest_status"] == "passed"
    assert summary["wing_operator_proposal_validation_latest_reason"] == (
        "supervised_wing_mvp_gain_down_proof_ch3"
    )
    assert "current aggregate operator_review_required" in summary["wing_decision_review"]
    assert "20260702T095758Z_wing_readonly_preflight" in summary["wing_decision_review"]
    assert "current aggregate operator_review_required" in summary[
        "active_real_wing_interaction_plan_sync"
    ]

    assert "current aggregate ledger is `operator_review_required`" in plan
    assert "accepted-source snapshot, not\nthe current global ledger status" in plan
    assert "decision review status: `accepted_decision_ready`" in plan
    assert "accepted decision records: `1`" in plan
    assert "rejected decision records: `2`" in plan
    assert "invalid decision records: `0`" in plan
    assert "unreviewed read-only source ids: `0`" in plan
    assert "latest supervised proof status: `passed`" in plan
    assert (
        "latest supervised proof reason: `supervised_wing_mvp_gain_down_proof_ch3`"
        in plan
    )
    assert "earlier failed CH3 fader rollback" in plan
    assert "closed by the passed rerun" in plan
    assert "20260704T045741Z_wing_operator_proposal_ch3_fader" in plan
    assert "20260704T045820Z_wing_operator_proposal_ch3_gain" in plan
    assert "Codex must not execute or retry the supervised proof unattended" in plan
    assert "M005" in plan
    assert "M006" in plan
    assert "allowed current WING live-write proof types: `fader` and `gain` only" in plan
    assert "allowed current write direction: small negative move only" in plan

    assert "They do not inherit the older decision, readiness or live authority" in plan
    assert "accepted decision records: `0`" not in plan
    assert "unreviewed read-only source ids: `4`" not in plan
    assert "Because decision review is still blocked" not in plan


def test_latest_readonly_handoff_packet_declares_scope_after_accepted_decision():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    summary = state["last_validation_summary"]
    handoff = READONLY_HANDOFF_LATEST.read_text(encoding="utf-8")

    assert summary["wing_readonly_preflight_decision_status"] == "operator_review_required"
    assert summary["wing_readonly_preflight_accepted_record_count"] == 1
    assert summary["wing_readonly_preflight_unreviewed_candidate_count"] > 0
    assert summary["wing_operator_proposal_validation_latest_reason"] == (
        "supervised_wing_mvp_gain_down_proof_ch3"
    )
    assert "## Packet Scope" in handoff
    assert "packet_scope: `read_only_preflight_evidence_handoff_only`" in handoff
    assert "Docs/CURRENT_STATE.md" in handoff
    assert "Docs/plans/M001-real-wing-interaction-plan.md" in handoff
    assert "ops/autopilot/STATE.json" in handoff
    assert "superseded_after_accepted_decision_ready: `true`" in handoff
    assert (
        "packet_does_not_select_current_next_action_after_decision_ready: `true`"
        in handoff
    )
    assert "does not authorize repeating hardware capture" in handoff
    assert "live_authority_granted: `false`" in handoff


def test_supervised_fader_gain_proof_stays_blocked_until_m002_and_operator_approval():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    goal = PROJECT_GOAL.read_text(encoding="utf-8")
    console = CONSOLE_CAPABILITY_MATRIX.read_text(encoding="utf-8")
    project_blockers = BLOCKERS_DOC.read_text(encoding="utf-8")
    supervised_runbook = SUPERVISED_RUNBOOK.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["hardware_evidence"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False

    m003 = markdown_table_row(roadmap, "M003")
    m004 = markdown_table_row(roadmap, "M004")
    m009 = markdown_table_row(roadmap, "M009")
    assert m003[1] == "Supervised fader proof"
    assert m003[2] == "Exact bounded fader move is approved, applied, read back and rolled back"
    assert m003[3] == "M002"
    assert m003[4] == "critical live write"
    assert m003[5] == "SUPERVISED_LIVE_VERIFIED for fader only"
    assert m003[-1] == "BLOCKED_BY_M002"
    assert m004[1] == "Supervised gain-down/trim proof"
    assert m004[2] == (
        "Exact bounded gain-down/trim move is approved, applied, read back "
        "and rolled back"
    )
    assert m004[3] == "M002"
    assert m004[4] == "critical live write"
    assert m004[5] == "SUPERVISED_LIVE_VERIFIED for gain-down only"
    assert m004[-1] == "BLOCKED_BY_M002"
    assert m009[1] == "Supervised non-fader/gain live expansion"
    assert "separately approved ADR" in m009[2]
    assert m009[-1] == "PENDING_APPROVAL"

    fader = markdown_table_row(console, "Channel fader")
    gain = markdown_table_row(console, "Preamp trim/gain-down")
    pan = markdown_table_row(console, "Pan")
    rollback = markdown_table_row(console, "Rollback")
    assert fader[3] == "no live write"
    assert fader[4] == "yes only after explicit approval"
    assert fader[-2:] == ["supervised fader proof", "none"]
    assert gain[3] == "no live write"
    assert gain[4] == "yes only after explicit approval and gain-down bounds"
    assert gain[-2:] == ["supervised gain-down proof", "none"]
    assert pan[4] == "no for WING MVP"
    assert "fresh read-only/no-authority pan and mono/phase context" in pan[5]
    assert rollback[4] == "required before closure"
    assert rollback[-1] == "none"

    fader_blocker = markdown_table_row(project_blockers, "Supervised fader write proof")
    gain_blocker = markdown_table_row(project_blockers, "Supervised gain-down / trim proof")
    rollback_blocker = markdown_table_row(project_blockers, "Rollback proof")
    for row in (fader_blocker, gain_blocker, rollback_blocker):
        assert row[1] == "blocked_by_readonly_preflight"
    assert "run `Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md` for `--value-type fader`" in fader_blocker[3]
    assert (
        "run `Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md` for "
        "`--value-type gain` with a negative delta only"
    ) in gain_blocker[3]
    assert "Preserve rollback request, rollback readback, and artifact review" in rollback_blocker[3]

    assert "Status: blocked until a clean read-only preflight candidate" in supervised_runbook
    assert "Allowed write kinds for this MVP:" in supervised_runbook
    assert "- `fader`" in supervised_runbook
    assert "- `gain` with a negative delta only" in supervised_runbook
    for forbidden in ("EQ", "compressor", "gate", "routing", "mute/unmute", "FX", "scenes"):
        assert f"- {forbidden}" in supervised_runbook
    normalized_supervised_runbook = " ".join(supervised_runbook.split())
    assert "The operator explicitly approves the exact channel, value type, and delta" in supervised_runbook
    assert "`--accepted-readonly-source-id`" in supervised_runbook
    assert (
        "the runner must re-check the decision review before opening a backend WebSocket"
        in normalized_supervised_runbook
    )
    assert "preflight decision gate status is `accepted_decision_ready`" in supervised_runbook
    assert "The selected delta is small and negative" in supervised_runbook
    assert "rollback went through `rollback_last_manual_write`" in supervised_runbook
    assert "rollback readback status is `verified_from_physical_readback`" in supervised_runbook
    assert "the delta is positive" in supervised_runbook
    assert "the value type is not `fader` or `gain`" in supervised_runbook
    assert "Do not claim full autonomous readiness" in supervised_runbook

    assert "Supervised fader write proof with explicit operator approval" in goal
    assert "Supervised gain-down / trim proof with explicit operator approval" in goal
    assert "supervised `fader`/`gain` rail с явным approval, readback и rollback" in goal


def test_emergency_stop_timeout_proof_stays_blocked_until_m002_and_real_evidence():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    goal = PROJECT_GOAL.read_text(encoding="utf-8")
    safety = SAFETY_CONTRACT.read_text(encoding="utf-8")
    console = CONSOLE_CAPABILITY_MATRIX.read_text(encoding="utf-8")
    project_blockers = BLOCKERS_DOC.read_text(encoding="utf-8")
    supervised_runbook = SUPERVISED_RUNBOOK.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["hardware_evidence"] is False
    authority = state["last_validation_summary"]["authority"]
    for key in (
        "live_authority_granted",
        "readiness_closure_allowed",
        "contacts_wing",
        "contacts_backend",
        "live_write",
    ):
        assert authority[key] is False

    m005 = markdown_table_row(roadmap, "M005")
    m006 = markdown_table_row(roadmap, "M006")
    assert m005[1] == "Emergency stop and timeout proof"
    assert m005[2] == (
        "Emergency stop, disconnect, timeout and stale-state protections "
        "are verified"
    )
    assert m005[3] == "M002"
    assert m005[4] == "critical safety"
    assert m005[5] == "SIMULATOR_VERIFIED plus supervised hardware evidence"
    assert m005[-1] == "BLOCKED_BY_M002"
    assert m006[1] == "Operator evidence and readiness update"
    assert m006[3] == "M002-M005"
    assert "accepted real proof" in m006[2]
    assert m006[-1] == "BLOCKED_BY_M002"

    rollback = markdown_table_row(console, "Rollback")
    emergency = markdown_table_row(console, "Emergency stop")
    stale_disconnect = markdown_table_row(console, "Stale state / disconnect fail-closed")
    assert rollback[3] == "no live action unattended"
    assert rollback[4] == "required before closure"
    assert rollback[-2:] == ["supervised rollback proof", "none"]
    assert emergency[3] == "no live action unattended"
    assert emergency[4] == "required before closure"
    assert emergency[-2:] == ["emergency-stop proof", "none"]
    assert stale_disconnect[3] == "no hardware unattended"
    assert stale_disconnect[4] == "required"
    assert stale_disconnect[-2:] == ["timeout/fail-closed proof", "none"]

    rollback_blocker = markdown_table_row(project_blockers, "Rollback proof")
    emergency_blocker = markdown_table_row(project_blockers, "Emergency-stop proof")
    timeout_blocker = markdown_table_row(
        project_blockers, "Connection-loss / timeout fail-closed proof"
    )
    assert rollback_blocker[1] == "blocked_by_readonly_preflight"
    assert emergency_blocker[1] == "blocked_by_readonly_preflight"
    assert timeout_blocker[1] == "tooling_ready_waits_real_wing"
    assert "rollback through `rollback_last_manual_write`" in rollback_blocker[2]
    assert "runner supports `--prove-emergency-stop`" in emergency_blocker[2]
    assert "blocks later apply with `emergency_stop_active`" in emergency_blocker[2]
    assert (
        "Use `--prove-emergency-stop` only inside the explicit supervised proof run."
        in emergency_blocker[3]
    )
    assert "controlled real-run timeout/connection-loss artifact" in timeout_blocker[3]
    assert "verify fail-closed semantics" in timeout_blocker[3]

    assert "Rollback proof." in goal
    assert "Emergency stop proof." in goal
    assert "Connection-loss / timeout fail-closed proof." in goal
    assert "Offline, shadow, replay, mock и simulator evidence не являются hardware proof." in goal
    assert "Full Autonomous Mixing Mode не может быть объявлен готовым" in goal

    assert "Unknown and insufficient evidence are valid safe outcomes." in safety
    assert "send live OSC writes;" in safety
    assert "control a real console;" in safety
    assert "modify the user's WING scene;" in safety
    assert "stale-state protection" in safety
    assert "readback, rollback or audit logging" in safety
    assert "describe simulated validation as real hardware validation;" in safety
    assert "weaken safety tests to pass a change." in safety

    assert supervised_runbook.count("--prove-emergency-stop") == 2
    assert "rollback went through `rollback_last_manual_write`" in supervised_runbook
    assert "rollback readback status is `verified_from_physical_readback`" in supervised_runbook
    assert "emergency stop was applied" in supervised_runbook
    assert "later apply was blocked with `emergency_stop_active`" in supervised_runbook
    assert "emergency stop does not block a later apply" in supervised_runbook
    assert "timeout happens after `apply_operator_proposal`" in supervised_runbook
    assert "physical state is not\n  manually inspected" in supervised_runbook
    assert "any direct write path is used instead of `apply_operator_proposal`" in supervised_runbook
    assert "Do not claim full autonomous readiness." in supervised_runbook


def test_m006_readiness_update_stays_report_only_until_real_m002_m005_proof():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    goal = PROJECT_GOAL.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    console = CONSOLE_CAPABILITY_MATRIX.read_text(encoding="utf-8")
    project_blockers = BLOCKERS_DOC.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    snapshot = build_autonomy_promotion_readiness_snapshot()
    preflight = snapshot["summary"]["wing_readonly_preflight_evidence"]

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["hardware_evidence"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False

    m006 = markdown_table_row(roadmap, "M006")
    assert m006[1] == "Operator evidence and readiness update"
    assert m006[2] == "MVP readiness matrix updates only from accepted real proof"
    assert m006[3] == "M002-M005"
    assert m006[4] == "high audit"
    assert m006[5] == "HARDWARE_READ_VERIFIED or SUPERVISED_LIVE_VERIFIED evidence surfaced"
    assert m006[-1] == "BLOCKED_BY_M002"

    operator_report = markdown_table_row(project_blockers, "Operator-visible evidence report")
    readiness = markdown_table_row(project_blockers, "Readiness matrix update based on real proof")
    assert operator_report[1] == "tooling_ready_waits_real_wing"
    assert "Save preflight/proof artifacts" in operator_report[3]
    assert "manual review" in operator_report[3]
    assert readiness[1] == "blocked_by_real_artifact_review"
    assert "Update readiness only after manual review of real WING artifacts" in readiness[3]
    assert "do not close rows from offline evidence" in readiness[3]

    supervised = markdown_table_row(capability, "Supervised AutoMix Mode")
    full_auto = markdown_table_row(capability, "Full Autonomous Mixing Mode")
    audit_logging = markdown_table_row(console, "Audit logging")
    assert supervised[4] == "WING live write scope is only `fader`/`gain`; real proof missing"
    assert supervised[7] == "M001-M006"
    assert supervised[8] == "none"
    assert full_auto[1] == "NOT_STARTED"
    assert "Locked until all safety and hardware gates pass" in full_auto[4]
    assert full_auto[8] == "none"
    assert audit_logging[4] == "required"
    assert audit_logging[-2:] == ["artifact review", "none"]

    assert snapshot["safety"] == {
        "read_only": True,
        "live_write": False,
        "live_widening_allowed": False,
        "full_autonomous_ready": False,
        "supervised_only": True,
    }
    assert snapshot["summary"]["autonomous_ready_target_ids"] == []
    assert "real_wing_and_emergency_stop_proof_incomplete" in snapshot["summary"]["global_blockers"]
    assert preflight["read_only"] is True
    assert preflight["live_write"] is False
    assert preflight["live_widening_allowed"] is False
    assert preflight["automatic_safety_update_allowed"] is False
    assert preflight["readiness_closure_allowed_by_evidence"] is False
    assert preflight["live_authority_granted"] is False
    assert preflight["full_autonomous_ready"] is False
    assert preflight["readiness_update_candidate_available"] is True
    assert preflight["readiness_matrix_update_allowed"] is False
    assert (
        "operator_must_review_readonly_preflight_candidate_before_readiness_update"
        in preflight["blocked_by"]
    )
    assert preflight["closed_blocker_ids"] == []
    assert "Manually review the read-only preflight candidate before updating readiness rows." in (
        preflight["next_required_actions"]
    )
    assert "manual readiness review only" in preflight["claim"]

    assert "Operator-visible evidence report." in goal
    assert "Readiness matrix update based only on real proof." in goal
    assert "Offline, shadow, replay, mock и simulator evidence не являются hardware proof." in goal
    assert "MVP readiness/operator evidence update is now covered" in current
    assert "readiness/operator evidence update gate rail is now automated" in report


def test_m006_owner_review_packet_requires_complete_m005_summary_contract():
    packet = json.loads(M006_OWNER_REVIEW_PACKET_SUMMARY.read_text(encoding="utf-8"))
    report = M006_OWNER_REVIEW_PACKET_REPORT.read_text(encoding="utf-8")

    assert packet["status"] == "ready_for_owner_review_after_m005"
    assert packet["scope"] == "owner_review_packet_only_no_readiness_update"
    assert packet["live_authority_granted"] is False
    assert packet["readiness_closure_allowed"] is False
    assert packet["readiness_matrix_update_allowed"] is False
    assert packet["contacts_wing"] is False
    assert packet["contacts_backend"] is False
    assert packet["live_write"] is False

    contract = packet["m005_completion_contract"]
    assert contract["required_summary_glob"] == (
        "artifacts/wing_timeout_disconnect_proof/*/summary.json"
    )
    assert contract["required_schema"] == "wing_timeout_disconnect_proof.v1"
    assert contract["required_status"] == "passed"
    assert contract["required_m005_complete"] is True
    assert contract["required_missing_m005_evidence"] == []
    assert contract["required_completed_m005_evidence"] == [
        "physical_baseline_readback",
        "operator_confirmed_physical_cable_loss",
        "stale_readback_unavailable",
        "blocked_write_while_disconnected",
        "reconnect_recovery_verified",
    ]
    assert contract["required_readback_target_binding"] == [
        "baseline_readback",
        "stale_readback",
        "reconnect_readback",
        "events_stale_readback",
        "events_reconnect_readback",
    ]
    assert contract["must_keep_authority_denied"] == {
        "live_authority_granted": False,
        "readiness_closure_allowed": False,
        "readiness_matrix_update_allowed": False,
    }

    prerequisites = packet["owner_decision_template"]["acceptance_prerequisites"]
    assert "M005 artifact exists under artifacts/wing_timeout_disconnect_proof" in prerequisites
    assert "M005 summary has m005_complete=true and missing_m005_evidence=[]" in prerequisites
    assert (
        "M005 proves physical cable-loss, stale readback denial, blocked write while "
        "disconnected and reconnect recovery"
    ) in prerequisites
    assert any(
        "channel" in item and "value_type" in item and "address" in item
        for item in prerequisites
    )

    assert "M005 summary has `m005_complete=true` and `missing_m005_evidence=[]`." in report
    assert "baseline, stale-state or\nreconnect readback payloads" in report
    assert "`live_authority_granted=false`" in report
    assert "`readiness_closure_allowed=false`" in report
    assert "`readiness_matrix_update_allowed=false`" in report
    assert "python3 tools/verify_m006_owner_readiness_packet.py" in report
    assert "--packet-summary artifacts/wing_owner_readiness_review/" in report
    assert "--m005-summary artifacts/wing_timeout_disconnect_proof/<accepted_m005_run>/summary.json" in report
    assert "--write-artifact" in report
    assert "artifacts/wing_owner_readiness_review_verifier/<run_id>/" in report
    assert "the verifier must return `blocked`" in report
    assert "does not close M005 or M006" in report
    assert "does not update\nthe readiness matrix and does not grant live authority" in report


def test_m007_offline_promotion_framework_stays_replay_bound_before_live_expansion():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    goal = PROJECT_GOAL.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    console = CONSOLE_CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    snapshot = build_autonomy_promotion_readiness_snapshot()

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["hardware_evidence"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False

    m007 = markdown_table_row(roadmap, "M007")
    m008 = markdown_table_row(roadmap, "M008")
    m009 = markdown_table_row(roadmap, "M009")
    m010 = markdown_table_row(roadmap, "M010")
    assert m007[1] == "Offline capability promotion framework"
    assert m007[2] == (
        "Long-term processing families have replay/fault/evaluation gates before any live expansion"
    )
    assert m007[3] == "M001-M006"
    assert m007[4] == "high offline"
    assert m007[5] == "PROTOTYPE/UNIT_VERIFIED to REPLAY_VERIFIED"
    assert m007[-1] == "PENDING"
    assert m008[3] == "M007"
    assert m008[-1] == "PENDING"
    assert m009[2] == "A separately approved ADR widens supervised live writes one family at a time"
    assert m009[3] == "M006-M008 plus hardware"
    assert m009[4] == "critical live write"
    assert m009[-1] == "PENDING_APPROVAL"
    assert m010[5] == "AUTONOMOUS_ELIGIBLE only if all gates pass"
    assert m010[-1] == "PENDING_APPROVAL"

    assert snapshot["safety"] == {
        "read_only": True,
        "live_write": False,
        "live_widening_allowed": False,
        "full_autonomous_ready": False,
        "supervised_only": True,
    }
    assert snapshot["summary"]["autonomous_ready_target_ids"] == []
    assert set(snapshot["summary"]["global_blockers"]) >= {
        "mode_full_autonomous_blocked",
        "only_supervised_fader_gain_executor_is_registered",
        "read_coverage_is_partial_or_missing",
        "real_wing_and_emergency_stop_proof_incomplete",
    }
    assert snapshot["summary"]["next_global_step"] == (
        "Run proof gates per action family before widening supervised or autonomous live writes."
    )

    replay_bound_capabilities = {
        "Assisted Mixing Mode": ("INTEGRATION_VERIFIED", "replay accepted/rejected proposals", "none"),
        "Full Autonomous Mixing Mode": ("NOT_STARTED", "completion gate review", "none"),
        "Multichannel audio analysis": (
            "UNIT_VERIFIED",
            "replay and simulator validation",
            "UNKNOWN",
        ),
        "Source and instrument classification": ("UNIT_VERIFIED", "replay calibration gate", "none"),
        "Arrangement-role detection": ("PROTOTYPE", "replay accepted role decisions", "none"),
        "Mix balance": ("PROTOTYPE", "replay bus/stem validation", "none"),
        "EQ": ("PROTOTYPE", "assisted replay gate", "none"),
        "Compression": ("PROTOTYPE", "replay and safety audit", "none"),
        "Gating": ("PROTOTYPE", "replay and fault injection", "none"),
        "Panning": ("PROTOTYPE", "assisted replay only", "none"),
        "Frequency separation": ("PROTOTYPE", "replay validation", "none"),
        "Cross-channel unmasking": ("PROTOTYPE", "replay/fault validation", "none"),
        "Effects and sends": ("PROTOTYPE", "assisted replay", "none"),
        "Master-bus processing": ("PROTOTYPE", "offline and listening gate", "none"),
        "Adaptive live mixing": ("SPECIFIED", "simulator and supervised staged proof", "none"),
        "Long-term memory and evaluation": ("PROTOTYPE", "lint plus replay promotion", "none"),
    }
    for capability_name, (evidence_level, next_gate, hardware_evidence) in replay_bound_capabilities.items():
        row = markdown_table_row(capability, capability_name)
        assert row[1] == evidence_level
        assert row[7] == next_gate
        assert row[8] == hardware_evidence

    assert "Replay, mock and simulator evidence must not be labeled as hardware evidence." in capability
    assert "Proposal-only, not apply authority" in markdown_table_row(
        capability, "Arrangement-role detection"
    )[4]
    assert "Live EQ writes forbidden" in markdown_table_row(capability, "EQ")[4]
    assert "Live FX writes forbidden" in markdown_table_row(capability, "Effects and sends")[4]
    assert "No autonomous live authority" in markdown_table_row(
        capability, "Adaptive live mixing"
    )[4]
    assert "Memory cannot mutate console directly" in markdown_table_row(
        capability, "Long-term memory and evaluation"
    )[4]

    live_forbidden_console_categories = {
        "Pan": "no for WING MVP",
        "EQ / HPF": "no",
        "Compressor / dynamics": "no",
        "Gate / expander": "no",
        "FX sends / returns": "no",
        "Routing / patch": "no",
        "Scenes / snapshots / presets": "no",
    }
    for console_category, supervised_live_state in live_forbidden_console_categories.items():
        row = markdown_table_row(console, console_category)
        assert row[3] == "no"
        assert row[4] == supervised_live_state
        assert row[6] == "none"

    assert markdown_table_row(console, "Rate limits / bounds / allowlists")[3] == "docs/tests only"
    assert markdown_table_row(console, "Rate limits / bounds / allowlists")[4] == "required"
    assert markdown_table_row(console, "Rate limits / bounds / allowlists")[5] == (
        "fault-injection validation"
    )

    assert "offline, replay," in completion
    assert "fault-injection и integration evaluations" in completion
    assert "High-risk console actions прошли staged real-hardware testing" in completion
    assert "PENDING_APPROVAL" in completion
    assert "Offline, shadow, replay, mock и simulator evidence не являются hardware proof." in goal
    assert "EQ, compressor, gate, routing, mute, FX, scenes, matrices, DCA" in goal
    assert "Full Autonomous Mixing Mode не может быть объявлен готовым" in goal

    assert "Offline capability promotion framework entry is now covered" in current
    assert "M007 offline capability promotion framework rail is now automated" in report


def test_m008_assisted_processing_expansion_stays_proposal_only_with_identity_gates():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    goal = PROJECT_GOAL.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    console = CONSOLE_CAPABILITY_MATRIX.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    snapshot = build_autonomy_promotion_readiness_snapshot()

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["hardware_evidence"] is False
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m008 = markdown_table_row(roadmap, "M008")
    m009 = markdown_table_row(roadmap, "M009")
    assert m008[1] == "Assisted processing family expansion"
    assert m008[2] == (
        "EQ, dynamics, pan, FX and balance stay proposal-only with source identity and safety gates"
    )
    assert m008[3] == "M007"
    assert m008[4] == "high assisted"
    assert m008[5] == "UNIT_VERIFIED to INTEGRATION_VERIFIED"
    assert m008[-1] == "PENDING"
    assert m009[2] == "A separately approved ADR widens supervised live writes one family at a time"
    assert m009[-1] == "PENDING_APPROVAL"

    assisted = markdown_table_row(capability, "Assisted Mixing Mode")
    source_identity = markdown_table_row(capability, "Source and instrument classification")
    arrangement_role = markdown_table_row(capability, "Arrangement-role detection")
    assert assisted[1] == "INTEGRATION_VERIFIED"
    assert assisted[4] == "Proposal freshness and source identity vary by family"
    assert assisted[7] == "replay accepted/rejected proposals"
    assert assisted[8] == "none"
    assert source_identity[7] == "replay calibration gate"
    assert arrangement_role[4] == "Proposal-only, not apply authority"
    assert arrangement_role[7] == "replay accepted role decisions"

    assisted_family_rows = {
        "Mix balance": ("PROTOTYPE", "source/stem identity", "replay bus/stem validation"),
        "EQ": ("PROTOTYPE", "feedback, band identity", "assisted replay gate"),
        "Compression": ("PROTOTYPE", "true peak, GR readback", "replay and safety audit"),
        "Gating": ("PROTOTYPE", "source identity, hysteresis", "replay and fault injection"),
        "Panning": ("PROTOTYPE", "mono/phase safety", "assisted replay only"),
        "Effects and sends": ("PROTOTYPE", "routing/readback", "assisted replay"),
        "Cross-channel unmasking": ("PROTOTYPE", "arrangement role, EQ safety", "replay/fault validation"),
    }
    for capability_name, (evidence_level, dependency, next_gate) in assisted_family_rows.items():
        row = markdown_table_row(capability, capability_name)
        assert row[1] == evidence_level
        assert row[6] == dependency
        assert row[7] == next_gate
        assert row[8] == "none"

    assert "Live EQ writes forbidden" in markdown_table_row(capability, "EQ")[4]
    assert "Makeup cannot bypass headroom/feedback" in markdown_table_row(
        capability, "Compression"
    )[4]
    assert "Chatter/bleed risk in live mode" in markdown_table_row(capability, "Gating")[4]
    assert "WING live scope is not approved for pan" in markdown_table_row(
        capability, "Panning"
    )[4]
    assert "Live FX writes forbidden" in markdown_table_row(capability, "Effects and sends")[4]

    assisted_snapshot_rows = {
        "eq_supervised_promotion": "read_coverage_blocked",
        "dynamics_supervised_promotion": "read_coverage_blocked",
        "gate_supervised_promotion": "read_coverage_blocked",
        "pan_phase_supervised_promotion": "manual_only_blocked",
        "fx_supervised_promotion": "manual_only_blocked",
        "bus_group_matrix_supervised_promotion": "manual_only_blocked",
    }
    rows_by_id = {row["id"]: row for row in snapshot["rows"]}
    for row_id, expected_status in assisted_snapshot_rows.items():
        row = rows_by_id[row_id]
        assert row["status"] == expected_status
        assert row["supervised_live_allowed"] is False
        assert row["autonomous_ready"] is False
        assert set(row["capability_statuses"]) == {"proposal_only"}
        assert "canonical_supervised" not in row["write_surface_statuses"]
        assert set(row["write_surface_statuses"]) <= {"proposal_only", "quarantined"}
        assert "safety_live_not_allowed" in row["blockers"]
        assert "not_autonomous_ready" in row["blockers"]

    console_assisted_boundaries = {
        "EQ / HPF": ("no", "no", "protected-band, feedback, readback, rollback proof"),
        "Compressor / dynamics": ("no", "no", "headroom, GR and readback proof"),
        "Gate / expander": ("no", "no", "chatter/bleed replay plus hardware proof"),
        "Pan": (
            "no",
            "no for WING MVP",
            "fresh read-only/no-authority pan and mono/phase context, rollback and physical readback first",
        ),
        "FX sends / returns": ("no", "no", "readback and rollback proof"),
        "Bus/group/main/DCA faders": (
            "no",
            "no except later approved supervised proof",
            "physical readback",
        ),
    }
    for category, (unattended, supervised, next_gate) in console_assisted_boundaries.items():
        row = markdown_table_row(console, category)
        assert row[3] == unattended
        assert row[4] == supervised
        assert row[5] == next_gate
        assert row[6] == "none"

    assert "EQ, compressor, gate, routing, mute, FX, scenes, matrices, DCA" in goal
    assert "Offline, shadow, replay, mock и simulator evidence не являются hardware proof." in goal
    assert snapshot["summary"]["autonomous_ready_target_ids"] == []
    assert snapshot["safety"]["live_write"] is False
    assert snapshot["safety"]["live_widening_allowed"] is False
    assert "Assisted processing family expansion entry is now covered" in current
    assert "M008 assisted processing family expansion rail is now automated" in report


def test_m009_non_fader_gain_live_expansion_requires_adr_hardware_and_family_proof():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    goal = PROJECT_GOAL.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    console = CONSOLE_CAPABILITY_MATRIX.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    snapshot = build_autonomy_promotion_readiness_snapshot()

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["hardware_evidence"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m009 = markdown_table_row(roadmap, "M009")
    m010 = markdown_table_row(roadmap, "M010")
    assert m009[1] == "Supervised non-fader/gain live expansion"
    assert m009[2] == "A separately approved ADR widens supervised live writes one family at a time"
    assert m009[3] == "M006-M008 plus hardware"
    assert m009[4] == "critical live write"
    assert m009[5] == "SUPERVISED_LIVE_VERIFIED per family"
    assert m009[-1] == "PENDING_APPROVAL"
    assert m010[3] == "M001-M009"
    assert m010[-1] == "PENDING_APPROVAL"

    supervised = markdown_table_row(capability, "Supervised AutoMix Mode")
    full_auto = markdown_table_row(capability, "Full Autonomous Mixing Mode")
    assert supervised[1] == "SPECIFIED"
    assert supervised[4] == "WING live write scope is only `fader`/`gain`; real proof missing"
    assert supervised[7] == "M001-M006"
    assert supervised[8] == "none"
    assert full_auto[1] == "NOT_STARTED"
    assert "Locked until all safety and hardware gates pass" in full_auto[4]
    assert full_auto[8] == "none"

    assert snapshot["safety"] == {
        "read_only": True,
        "live_write": False,
        "live_widening_allowed": False,
        "full_autonomous_ready": False,
        "supervised_only": True,
    }
    assert snapshot["summary"]["supervised_live_allowed_target_ids"] == []
    assert snapshot["summary"]["autonomous_ready_target_ids"] == []
    assert set(snapshot["summary"]["global_blockers"]) >= {
        "mode_full_autonomous_blocked",
        "only_supervised_fader_gain_executor_is_registered",
        "read_coverage_is_partial_or_missing",
        "real_wing_and_emergency_stop_proof_incomplete",
    }

    rows_by_id = {row["id"]: row for row in snapshot["rows"]}
    non_fader_gain_rows = {
        "eq_supervised_promotion": "eq",
        "dynamics_supervised_promotion": "compressor",
        "gate_supervised_promotion": "gate",
        "pan_phase_supervised_promotion": "pan_phase_delay",
        "fx_supervised_promotion": "fx_send_insert",
        "bus_group_matrix_supervised_promotion": "bus_group_matrix",
        "mute_supervised_promotion": "mute",
        "routing_supervised_promotion": "routing",
        "scene_preset_supervised_promotion": "scene_preset",
    }
    for row_id, action_family in non_fader_gain_rows.items():
        row = rows_by_id[row_id]
        assert row["action_family"] == action_family
        assert row["supervised_live_allowed"] is False
        assert row["autonomous_ready"] is False
        assert "canonical_supervised" not in row["write_surface_statuses"]
        assert set(row["write_surface_statuses"]) <= {"proposal_only", "quarantined"}
        assert "no_allowed_live_write_surface" in row["blockers"]
        assert "safety_live_not_allowed" in row["blockers"]
        assert any(
            blocker in row["blockers"]
            for blocker in ("real_wing_proof_missing", "real_wing_proof_not_allowed")
        )
        assert "rollback_readback_missing" in row["blockers"]
        assert "emergency_stop_proof_missing" in row["blockers"]

    allowed_fader = rows_by_id["fader_supervised_promotion"]
    gain_trim = rows_by_id["gain_trim_supervised_promotion"]
    assert allowed_fader["action_family"] == "fader"
    assert allowed_fader["supervised_live_allowed"] is False
    assert allowed_fader["safety_status"] == "hardware_proven_narrow"
    assert "rollback_readback_failed" not in allowed_fader["blockers"]
    assert "safety_live_not_allowed" in allowed_fader["blockers"]
    assert gain_trim["action_family"] == "gain"
    assert gain_trim["supervised_live_allowed"] is False
    assert gain_trim["safety_status"] == "hardware_proven_narrow"
    assert "real_wing_proof_missing" not in gain_trim["blockers"]
    assert "apply_readback_missing" not in gain_trim["blockers"]
    assert "safety_live_not_allowed" in gain_trim["blockers"]

    console_live_expansion_rows = {
        "Pan": (
            "no",
            "no for WING MVP",
            "fresh read-only/no-authority pan and mono/phase context, rollback and physical readback first",
        ),
        "Mute/unmute": (
            "no",
            "no",
            "fresh read-only/no-authority mute context, rollback and physical readback first",
        ),
        "EQ / HPF": ("no", "no", "protected-band, feedback, readback, rollback proof"),
        "Compressor / dynamics": ("no", "no", "headroom, GR and readback proof"),
        "Gate / expander": ("no", "no", "chatter/bleed replay plus hardware proof"),
        "De-esser": ("no", "no", "unit/replay first"),
        "FX sends / returns": ("no", "no", "readback and rollback proof"),
        "Routing / patch": (
            "no",
            "no",
            "fresh read-only/no-authority route context, rollback and physical readback first",
        ),
        "Bus/group/main/DCA faders": (
            "no",
            "no except later approved supervised proof",
            "physical readback",
        ),
        "Matrices": ("no", "no", "physical readback"),
        "Scenes / snapshots / presets": (
            "no",
            "no",
            "fresh read-only/no-authority scene/preset context, scoped diffs, rollback and destructive-action proof first",
        ),
    }
    for category, (unattended, supervised_live, next_gate) in console_live_expansion_rows.items():
        row = markdown_table_row(console, category)
        assert row[3] == unattended
        assert row[4] == supervised_live
        assert row[5] == next_gate
        assert row[6] == "none"

    fader = markdown_table_row(console, "Channel fader")
    gain = markdown_table_row(console, "Preamp trim/gain-down")
    assert fader[4] == "yes only after explicit approval"
    assert fader[5:] == ["supervised fader proof", "none"]
    assert gain[4] == "yes only after explicit approval and gain-down bounds"
    assert gain[5:] == ["supervised gain-down proof", "none"]

    assert "High-risk console actions прошли staged real-hardware testing" in completion
    assert "Recovery, rollback, emergency-stop and audit behavior verified" in completion
    assert "Любая live WING запись до отдельного ADR и hardware proof ограничена только" in goal
    assert "EQ, compressor, gate, routing, mute, FX, scenes, matrices, DCA" in goal
    assert "Supervised non-fader/gain live expansion entry is now covered" in current
    assert "M009 supervised non-fader/gain live expansion rail is now automated" in report


def test_m010_full_autonomous_eligibility_stays_locked_until_completion_gates_pass():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    goal = PROJECT_GOAL.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    console = CONSOLE_CAPABILITY_MATRIX.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    snapshot = build_autonomy_promotion_readiness_snapshot()

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["checkpoint_status"] == "waiting_for_external_validation"
    assert state["last_validation_summary"]["hardware_evidence"] is False
    assert state["last_validation_summary"]["authority"] == {
        "contacts_backend": False,
        "contacts_wing": False,
        "live_authority_granted": False,
        "live_write": False,
        "readiness_closure_allowed": False,
    }

    m010 = markdown_table_row(roadmap, "M010")
    assert m010[1] == "Full autonomous eligibility review"
    assert m010[2] == "Full Autonomous Mixing Mode eligibility is reviewed against all completion gates"
    assert m010[3] == "M001-M009"
    assert m010[4] == "critical autonomous"
    assert m010[5] == "AUTONOMOUS_ELIGIBLE only if all gates pass"
    assert m010[-1] == "PENDING_APPROVAL"

    full_auto = markdown_table_row(capability, "Full Autonomous Mixing Mode")
    supervised = markdown_table_row(capability, "Supervised AutoMix Mode")
    multichannel = markdown_table_row(capability, "Multichannel audio analysis")
    assert full_auto == [
        "Full Autonomous Mixing Mode",
        "NOT_STARTED",
        "`backend/operator_mode_policy.py`, `Docs/PROJECT_COMPLETION_GATE.md`",
        "none sufficient",
        "Locked until all safety and hardware gates pass",
        "critical",
        "all mandatory capabilities",
        "completion gate review",
        "none",
    ]
    assert supervised[1] == "SPECIFIED"
    assert supervised[4] == "WING live write scope is only `fader`/`gain`; real proof missing"
    assert supervised[7] == "M001-M006"
    assert supervised[8] == "none"
    assert multichannel[1] == "UNIT_VERIFIED"
    assert multichannel[-1] == "UNKNOWN"

    for capability_name in (
        "Gain staging",
        "Mix balance",
        "EQ",
        "Compression",
        "Gating",
        "Panning",
        "Effects and sends",
        "Master-bus processing",
        "Long-term memory and evaluation",
    ):
        row = markdown_table_row(capability, capability_name)
        assert row[1] != "AUTONOMOUS_ELIGIBLE"
        assert row[-1] == "none"

    assert "Replay, mock and simulator evidence must not be labeled as hardware evidence" in capability
    assert "Все mandatory capabilities" in completion
    assert "Analysis Mode, Assisted Mixing Mode, Supervised AutoMix Mode и Full" in completion
    assert "Autonomous Mixing Mode" in completion
    assert "WING Rack read/write protocol coverage" in completion
    assert "deterministic Safety Kernel" in completion
    assert "offline, replay,\n   simulator, fault-injection" in completion
    assert "staged real-hardware testing" in completion
    assert "PENDING_APPROVAL" in completion
    assert "COMPLETE` запрещен" in completion

    assert "Full Autonomous Mixing Mode не может быть объявлен готовым" in goal
    assert "mock и simulator evidence не являются hardware proof" in goal
    assert "Любая live WING запись до отдельного ADR и hardware proof" in goal

    assert snapshot["safety"] == {
        "read_only": True,
        "live_write": False,
        "live_widening_allowed": False,
        "full_autonomous_ready": False,
        "supervised_only": True,
    }
    assert snapshot["summary"]["autonomous_ready_target_ids"] == []
    assert snapshot["summary"]["supervised_live_allowed_target_ids"] == []
    assert snapshot["summary"]["next_global_step"] == (
        "Run proof gates per action family before widening supervised or "
        "autonomous live writes."
    )
    assert set(snapshot["summary"]["global_blockers"]) >= {
        "mode_full_autonomous_blocked",
        "only_supervised_fader_gain_executor_is_registered",
        "read_coverage_is_partial_or_missing",
        "real_wing_and_emergency_stop_proof_incomplete",
    }

    for row in snapshot["rows"]:
        assert row["autonomous_ready"] is False
        assert "not_autonomous_ready" in row["blockers"]

    for category in (
        "EQ / HPF",
        "Compressor / dynamics",
        "Gate / expander",
        "FX sends / returns",
        "Routing / patch",
        "Scenes / snapshots / presets",
        "Matrices",
    ):
        row = markdown_table_row(console, category)
        assert row[3] == "no"
        assert row[4] == "no"
        assert row[-1] == "none"

    assert "Full autonomous eligibility review entry is now covered" in current
    assert "completion gate cannot close from docs/unit/mock/replay/no-WING evidence" in current
    assert "M010 full autonomous eligibility rail is now automated" in report
    assert "no autonomous target is reported ready" in report


def test_m011_offline_replay_evaluation_contract_stays_offline_and_threshold_gated():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M011_PLAN.read_text(encoding="utf-8")
    adr = M011_ADR.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False

    m011 = markdown_table_row(roadmap, "M011")
    assert m011[1] == "Offline replay evaluation contract"
    assert m011[2] == (
        "Soundcheck, studio and concert replay manifests fail closed until "
        "approved thresholds and complete scenario evidence exist"
    )
    assert m011[3] == "M007-M010"
    assert m011[4] == "high offline"
    assert (
        m011[5]
        == "SPECIFIED to UNIT_VERIFIED for gate contract; replay evidence remains pending"
    )
    assert m011[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert memory_eval[1] == "PROTOTYPE"
    assert "`backend/offline_replay_evaluation_contract.py`" in memory_eval[2]
    assert "tests/test_offline_replay_evaluation_contract.py" in memory_eval[3]
    assert "approved thresholds are still required" in memory_eval[4]
    assert memory_eval[-1] == "none"

    pending_manifest = {
        "manifest_id": "pending-thresholds",
        "offline_only": True,
        "live_write": False,
        "contacts_wing": False,
        "contacts_backend": False,
        "hardware_evidence": False,
        "threshold_policy": {"status": "PENDING_APPROVAL"},
        "scenarios": [
            {
                "scenario": "soundcheck",
                "evaluation_types": list(REQUIRED_EVALUATION_TYPES),
                "safety_checks": {key: True for key in REQUIRED_SAFETY_CHECKS},
                "quality_axes": {
                    key: {"status": "passed"} for key in REQUIRED_QUALITY_AXES
                },
            }
        ],
    }
    def complete_scenario(name: str) -> dict:
        return {
            "scenario": name,
            "evaluation_types": list(REQUIRED_EVALUATION_TYPES),
            "safety_checks": {key: True for key in REQUIRED_SAFETY_CHECKS},
            "quality_axes": {
                key: {"status": "passed", "artifact": f"{name}/{key}.json"}
                for key in REQUIRED_QUALITY_AXES
            },
        }

    def complete_threshold(axis: str) -> dict:
        return {
            "axis": axis,
            "metric_name": f"{axis}_score",
            "direction": "min",
            "candidate_value": 0.8,
            "evidence_artifact": f"thresholds/{axis}.json",
            "rationale": f"Replay promotion threshold for {axis}",
            "rollback_policy": f"Revert {axis} threshold on regression",
        }

    def complete_threshold_policy_manifest(**overrides) -> dict:
        manifest = {
            "policy_id": "offline-replay-threshold-policy",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "readiness_closure_allowed": False,
            "live_authority_granted": False,
            "policy_checks": {key: True for key in THRESHOLD_REQUIRED_POLICY_CHECKS},
            "threshold_policy": {
                "status": "approved",
                "approved_by": "owner",
                "approved_decision_id": "adr:offline-replay-thresholds",
                "approved_at": "2026-06-19T17:32:45Z",
            },
            "thresholds": [complete_threshold(axis) for axis in REQUIRED_THRESHOLD_AXES],
        }
        manifest.update(overrides)
        return manifest

    pending_manifest["quality_threshold_policy_manifest"] = (
        complete_threshold_policy_manifest(
            threshold_policy={"status": "PENDING_APPROVAL"}
        )
    )
    pending_result = evaluate_offline_replay_evaluation_contract(pending_manifest)
    assert pending_result["ready_for_replay_promotion"] is False
    assert pending_result["authority"]["live_authority_granted"] is False
    assert pending_result["authority"]["readiness_closure_allowed"] is False
    assert "approved_quality_threshold_policy_required" in pending_result["blockers"]
    assert "quality_threshold_policy_contract_not_ready" in pending_result["blockers"]
    assert (
        "quality_threshold_policy:owner_approved_quality_threshold_policy_required"
        in pending_result["blockers"]
    )
    assert "missing_required_scenario:studio" in pending_result["blockers"]
    assert "missing_required_scenario:concert" in pending_result["blockers"]

    complete_result = evaluate_offline_replay_evaluation_contract(
        {
            "manifest_id": "complete-offline-replay",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "readiness_closure_allowed": False,
            "live_authority_granted": False,
            "quality_threshold_policy_manifest": complete_threshold_policy_manifest(),
            "scenarios": [complete_scenario(name) for name in REQUIRED_SCENARIOS],
        }
    )
    assert complete_result["ready_for_replay_promotion"] is True
    assert complete_result["hardware_evidence"] is False
    assert complete_result["authority"]["live_authority_granted"] is False
    assert complete_result["authority"]["readiness_closure_allowed"] is False
    assert complete_result["summary"]["covered_scenarios"] == list(REQUIRED_SCENARIOS)
    assert complete_result["summary"]["threshold_policy_contract_status"] == (
        "ready_for_threshold_policy_use"
    )

    assert "offline, replay,\n   simulator, fault-injection" in completion
    assert "approved thresholds" in completion
    assert "M011-CP01" in plan
    assert "M011-CP02" in plan
    assert "Offline replay evidence cannot grant live authority" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "always returns `live_authority_granted: false`" in adr
    assert "M011 offline replay evaluation contract is now covered" in current
    assert "M011 offline replay evaluation contract rail is now automated" in report


def test_m012_fault_injection_recovery_contract_stays_simulator_only_and_audit_gated():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    console = CONSOLE_CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M012_PLAN.read_text(encoding="utf-8")
    adr = M012_ADR.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m012 = markdown_table_row(roadmap, "M012")
    assert m012[1] == "Fault-injection recovery contract"
    assert m012[2] == (
        "Simulator fault bundles fail closed until required failure modes, "
        "recovery outcomes and audit evidence exist"
    )
    assert m012[3] == "M011"
    assert m012[4] == "high offline"
    assert (
        m012[5]
        == "SPECIFIED to UNIT_VERIFIED for recovery gate; simulator evidence remains pending"
    )
    assert m012[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "`backend/fault_injection_recovery_contract.py`" in memory_eval[2]
    assert "tests/test_fault_injection_recovery_contract.py" in memory_eval[3]
    assert "fault-recovery simulator evidence cannot close live readiness" in memory_eval[4]
    assert memory_eval[-1] == "none"

    rate_limits = markdown_table_row(console, "Rate limits / bounds / allowlists")
    stale_state = markdown_table_row(console, "Stale state / disconnect fail-closed")
    audit_logging = markdown_table_row(console, "Audit logging")
    assert "`backend/fault_injection_recovery_contract.py`" in rate_limits[2]
    assert rate_limits[3] == "docs/tests only"
    assert rate_limits[4] == "required"
    assert rate_limits[5] == "fault-injection validation"
    assert stale_state[5] == "timeout/fail-closed proof"
    assert audit_logging[5] == "artifact review"
    assert rate_limits[6] == "none"
    assert stale_state[6] == "none"
    assert audit_logging[6] == "none"

    pending_manifest = {
        "manifest_id": "pending-recovery-policy",
        "offline_only": True,
        "live_write": False,
        "contacts_wing": False,
        "contacts_backend": False,
        "hardware_evidence": False,
        "recovery_policy_manifest": {
            "policy_id": "pending-recovery-policy",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "recovery_controls": {
                key: True for key in POLICY_REQUIRED_RECOVERY_CONTROLS
            },
            "recovery_evidence": {
                key: True for key in POLICY_REQUIRED_RECOVERY_EVIDENCE
            },
            "recovery_policy": {"status": "PENDING_APPROVAL"},
        },
        "fault_scenarios": [
            {
                "fault_scenario": "transport_timeout",
                "evaluation_types": list(RECOVERY_REQUIRED_EVALUATION_TYPES),
                "recovery_outcomes": {
                    key: True for key in REQUIRED_RECOVERY_OUTCOMES
                },
                "audit": {
                    key: f"transport-timeout-{key}"
                    for key in RECOVERY_REQUIRED_AUDIT_FIELDS
                },
            }
        ],
    }
    pending_result = evaluate_fault_injection_recovery_contract(pending_manifest)
    assert pending_result["ready_for_fault_recovery_promotion"] is False
    assert pending_result["authority"]["live_authority_granted"] is False
    assert pending_result["authority"]["readiness_closure_allowed"] is False
    assert "approved_recovery_policy_required" in pending_result["blockers"]
    assert "missing_required_fault_scenario:connection_loss" in pending_result["blockers"]
    assert "missing_required_fault_scenario:emergency_stop" in pending_result["blockers"]

    def complete_fault(name: str) -> dict:
        return {
            "fault_scenario": name,
            "evaluation_types": list(RECOVERY_REQUIRED_EVALUATION_TYPES),
            "recovery_outcomes": {key: True for key in REQUIRED_RECOVERY_OUTCOMES},
            "audit": {key: f"{name}/{key}" for key in RECOVERY_REQUIRED_AUDIT_FIELDS},
        }

    complete_result = evaluate_fault_injection_recovery_contract(
        {
            "manifest_id": "complete-fault-recovery",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "readiness_closure_allowed": False,
            "live_authority_granted": False,
            "recovery_policy_manifest": {
                "policy_id": "approved-recovery-policy",
                "offline_only": True,
                "live_write": False,
                "contacts_wing": False,
                "contacts_backend": False,
                "hardware_evidence": False,
                "readiness_closure_allowed": False,
                "live_authority_granted": False,
                "recovery_controls": {
                    key: True for key in POLICY_REQUIRED_RECOVERY_CONTROLS
                },
                "recovery_evidence": {
                    key: True for key in POLICY_REQUIRED_RECOVERY_EVIDENCE
                },
                "recovery_policy": {
                    "status": "approved",
                    "approved_by": "owner",
                    "approved_decision_id": "adr:fault-recovery-policy",
                    "approved_at": "2026-06-19T18:00:10Z",
                },
            },
            "fault_scenarios": [
                complete_fault(name) for name in RECOVERY_REQUIRED_FAULT_SCENARIOS
            ],
        }
    )
    assert complete_result["ready_for_fault_recovery_promotion"] is True
    assert complete_result["hardware_evidence"] is False
    assert complete_result["authority"]["live_authority_granted"] is False
    assert complete_result["authority"]["readiness_closure_allowed"] is False
    assert complete_result["summary"]["covered_fault_scenarios"] == list(
        RECOVERY_REQUIRED_FAULT_SCENARIOS
    )

    assert "Recovery, rollback, emergency-stop and audit behavior verified" in completion
    assert "fault-injection и integration evaluations" in completion
    assert "M012-CP01" in plan
    assert "M012-CP02" in plan
    assert "Simulator fault evidence cannot grant live authority" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "always returns `live_authority_granted: false`" in adr
    assert "M012 fault-injection recovery contract is now covered" in current
    assert "M012 fault-injection recovery contract rail is now automated" in report


def test_m013_integration_evaluation_bundle_contract_stays_offline_and_child_gated():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    safety = SAFETY_CONTRACT.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M013_PLAN.read_text(encoding="utf-8")
    adr = M013_ADR.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m013 = markdown_table_row(roadmap, "M013")
    assert m013[1] == "Integration evaluation bundle contract"
    assert m013[2] == (
        "Replay and fault-recovery manifests are composed into an offline "
        "integration bundle before promotion"
    )
    assert m013[3] == "M011-M012"
    assert m013[4] == "high offline"
    assert (
        m013[5]
        == "SPECIFIED to UNIT_VERIFIED for integration gate; integration evidence remains pending"
    )
    assert m013[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "`backend/evaluation_integration_bundle_contract.py`" in memory_eval[2]
    assert "tests/test_evaluation_integration_bundle_contract.py" in memory_eval[3]
    assert "integration bundle evidence cannot close live readiness" in memory_eval[4]
    assert memory_eval[-1] == "none"

    def complete_scenario(name: str) -> dict:
        return {
            "scenario": name,
            "evaluation_types": list(REQUIRED_EVALUATION_TYPES),
            "safety_checks": {key: True for key in REQUIRED_SAFETY_CHECKS},
            "quality_axes": {
                key: {"status": "passed", "artifact": f"{name}/{key}.json"}
                for key in REQUIRED_QUALITY_AXES
            },
        }

    def complete_threshold(axis: str) -> dict:
        return {
            "axis": axis,
            "metric_name": f"{axis}_score",
            "direction": "min",
            "candidate_value": 0.8,
            "evidence_artifact": f"thresholds/{axis}.json",
            "rationale": f"Replay promotion threshold for {axis}",
            "rollback_policy": f"Revert {axis} threshold on regression",
        }

    def complete_threshold_policy_manifest(**overrides) -> dict:
        manifest = {
            "policy_id": "offline-replay-threshold-policy",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "readiness_closure_allowed": False,
            "live_authority_granted": False,
            "policy_checks": {key: True for key in THRESHOLD_REQUIRED_POLICY_CHECKS},
            "threshold_policy": {
                "status": "approved",
                "approved_by": "owner",
                "approved_decision_id": "adr:offline-replay-thresholds",
                "approved_at": "2026-06-19T17:32:45Z",
            },
            "thresholds": [complete_threshold(axis) for axis in REQUIRED_THRESHOLD_AXES],
        }
        manifest.update(overrides)
        return manifest

    def complete_replay_manifest(**overrides) -> dict:
        manifest = {
            "manifest_id": "complete-offline-replay",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "readiness_closure_allowed": False,
            "live_authority_granted": False,
            "quality_threshold_policy_manifest": complete_threshold_policy_manifest(),
            "scenarios": [complete_scenario(name) for name in REQUIRED_SCENARIOS],
        }
        manifest.update(overrides)
        return manifest

    def complete_fault(name: str) -> dict:
        return {
            "fault_scenario": name,
            "evaluation_types": list(RECOVERY_REQUIRED_EVALUATION_TYPES),
            "recovery_outcomes": {key: True for key in REQUIRED_RECOVERY_OUTCOMES},
            "audit": {key: f"{name}/{key}" for key in RECOVERY_REQUIRED_AUDIT_FIELDS},
        }

    def complete_fault_manifest(**overrides) -> dict:
        manifest = {
            "manifest_id": "complete-fault-recovery",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "readiness_closure_allowed": False,
            "live_authority_granted": False,
            "recovery_policy_manifest": {
                "policy_id": "approved-recovery-policy",
                "offline_only": True,
                "live_write": False,
                "contacts_wing": False,
                "contacts_backend": False,
                "hardware_evidence": False,
                "readiness_closure_allowed": False,
                "live_authority_granted": False,
                "recovery_controls": {
                    key: True for key in POLICY_REQUIRED_RECOVERY_CONTROLS
                },
                "recovery_evidence": {
                    key: True for key in POLICY_REQUIRED_RECOVERY_EVIDENCE
                },
                "recovery_policy": {
                    "status": "approved",
                    "approved_by": "owner",
                    "approved_decision_id": "adr:fault-recovery-policy",
                    "approved_at": "2026-06-19T18:00:10Z",
                },
            },
            "fault_scenarios": [
                complete_fault(name) for name in RECOVERY_REQUIRED_FAULT_SCENARIOS
            ],
        }
        manifest.update(overrides)
        return manifest

    def complete_integration_policy_manifest(**overrides) -> dict:
        manifest = {
            "policy_id": "complete-integration-policy",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "readiness_closure_allowed": False,
            "live_authority_granted": False,
            "integration_controls": {
                key: True for key in POLICY_REQUIRED_INTEGRATION_CONTROLS
            },
            "integration_evidence": {
                key: True for key in POLICY_REQUIRED_INTEGRATION_EVIDENCE
            },
            "integration_policy": {
                "status": "approved",
                "approved_by": "owner",
                "approved_decision_id": "adr:integration-evaluation-policy",
                "approved_at": "2026-06-19T18:10:23Z",
            },
        }
        manifest.update(overrides)
        return manifest

    pending_result = evaluate_integration_bundle_contract(
        {
            "manifest_id": "pending-integration-bundle",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "integration_policy_manifest": complete_integration_policy_manifest(
                integration_policy={"status": "PENDING_APPROVAL"}
            ),
            "integration_checks": {"analysis_trace_bound": True},
            "offline_replay_manifest": complete_replay_manifest(
                quality_threshold_policy_manifest=complete_threshold_policy_manifest(
                    threshold_policy={"status": "PENDING_APPROVAL"}
                ),
                scenarios=[complete_scenario("soundcheck")],
            ),
            "fault_recovery_manifest": complete_fault_manifest(
                fault_scenarios=[complete_fault("transport_timeout")]
            ),
        }
    )
    assert pending_result["ready_for_integration_evaluation_promotion"] is False
    assert pending_result["authority"]["live_authority_granted"] is False
    assert pending_result["authority"]["readiness_closure_allowed"] is False
    assert "approved_integration_policy_required" in pending_result["blockers"]
    assert "integration_policy_contract_not_ready" in pending_result["blockers"]
    assert (
        "integration_policy:owner_approved_integration_policy_required"
        in pending_result["blockers"]
    )
    assert "offline_replay_contract_not_ready" in pending_result["blockers"]
    assert "fault_recovery_contract_not_ready" in pending_result["blockers"]
    assert "offline_replay:approved_quality_threshold_policy_required" in pending_result[
        "blockers"
    ]
    assert "offline_replay:quality_threshold_policy_contract_not_ready" in pending_result[
        "blockers"
    ]
    assert (
        "offline_replay:quality_threshold_policy:"
        "owner_approved_quality_threshold_policy_required"
        in pending_result["blockers"]
    )
    assert "fault_recovery:missing_required_fault_scenario:connection_loss" in pending_result[
        "blockers"
    ]
    assert "missing_integration_check:fault_recovery_result_bound" in pending_result[
        "blockers"
    ]

    complete_result = evaluate_integration_bundle_contract(
        {
            "manifest_id": "complete-integration-bundle",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "readiness_closure_allowed": False,
            "live_authority_granted": False,
            "integration_policy_manifest": complete_integration_policy_manifest(),
            "safety_kernel_trace_manifest": safety_kernel_trace_manifest_fixture(),
            "no_live_transport_manifest": no_live_transport_manifest_fixture(),
            "operator_report_manifest": operator_report_manifest_fixture(),
            "integration_checks": {key: True for key in REQUIRED_INTEGRATION_CHECKS},
            "offline_replay_manifest": complete_replay_manifest(),
            "fault_recovery_manifest": complete_fault_manifest(),
        }
    )
    assert complete_result["ready_for_integration_evaluation_promotion"] is True
    assert complete_result["hardware_evidence"] is False
    assert complete_result["authority"]["live_authority_granted"] is False
    assert complete_result["authority"]["readiness_closure_allowed"] is False
    assert complete_result["summary"]["offline_replay_ready"] is True
    assert complete_result["summary"]["fault_recovery_ready"] is True
    assert complete_result["summary"]["missing_integration_checks"] == []

    assert "offline, replay,\n   simulator, fault-injection и integration evaluations" in completion
    assert "Unvalidated candidates cannot alter live concert behavior" in safety
    assert "M013-CP01" in plan
    assert "M013-CP02" in plan
    assert "Integration bundle evidence cannot grant live authority" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "always returns `live_authority_granted: false`" in adr
    assert "M013 integration evaluation bundle contract is now covered" in current
    assert "M013 integration evaluation bundle contract rail is now automated" in report


def test_m014_quality_threshold_policy_contract_stays_owner_approved_and_offline():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    safety = SAFETY_CONTRACT.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M014_PLAN.read_text(encoding="utf-8")
    adr = M014_ADR.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m014 = markdown_table_row(roadmap, "M014")
    assert m014[1] == "Quality threshold policy contract"
    assert m014[2] == (
        "Candidate quality thresholds are complete, evidence-bound and "
        "owner-approved before replay or integration promotion"
    )
    assert m014[3] == "M011-M013"
    assert m014[4] == "high offline"
    assert (
        m014[5]
        == "SPECIFIED to UNIT_VERIFIED for threshold gate; owner approval remains required"
    )
    assert m014[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "`backend/quality_threshold_policy_contract.py`" in memory_eval[2]
    assert "tests/test_quality_threshold_policy_contract.py" in memory_eval[3]
    assert "quality threshold policy evidence cannot grant live authority" in memory_eval[4]
    assert memory_eval[-1] == "none"

    def complete_threshold(axis: str) -> dict:
        return {
            "axis": axis,
            "metric_name": f"{axis}_score",
            "direction": "min",
            "candidate_value": 0.8,
            "evidence_artifact": f"artifacts/evaluation/{axis}.json",
            "rationale": f"Candidate threshold for {axis}",
            "rollback_policy": f"Revert {axis} threshold if regression appears",
        }

    candidate_result = evaluate_quality_threshold_policy_contract(
        {
            "policy_id": "candidate-quality-thresholds",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "readiness_closure_allowed": False,
            "live_authority_granted": False,
            "policy_checks": {key: True for key in THRESHOLD_REQUIRED_POLICY_CHECKS},
            "threshold_policy": {"status": "candidate"},
            "thresholds": [complete_threshold(axis) for axis in REQUIRED_THRESHOLD_AXES],
        }
    )
    assert candidate_result["ready_for_owner_review"] is True
    assert candidate_result["ready_for_threshold_policy_use"] is False
    assert candidate_result["authority"]["live_authority_granted"] is False
    assert candidate_result["authority"]["readiness_closure_allowed"] is False
    assert candidate_result["review_blockers"] == []
    assert candidate_result["blockers"] == [
        "owner_approved_quality_threshold_policy_required"
    ]

    approved_result = evaluate_quality_threshold_policy_contract(
        {
            "policy_id": "approved-quality-thresholds",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "readiness_closure_allowed": False,
            "live_authority_granted": False,
            "policy_checks": {key: True for key in THRESHOLD_REQUIRED_POLICY_CHECKS},
            "threshold_policy": {
                "status": "approved",
                "approved_by": "owner",
                "approved_decision_id": "adr:quality-threshold-policy",
                "approved_at": "2026-06-19T17:20:04Z",
            },
            "thresholds": [complete_threshold(axis) for axis in REQUIRED_THRESHOLD_AXES],
        }
    )
    assert approved_result["ready_for_threshold_policy_use"] is True
    assert approved_result["threshold_policy_approved"] is True
    assert approved_result["hardware_evidence"] is False
    assert approved_result["authority"]["live_authority_granted"] is False
    assert approved_result["authority"]["readiness_closure_allowed"] is False
    assert approved_result["summary"]["covered_threshold_axes"] == list(
        REQUIRED_THRESHOLD_AXES
    )

    blocked_result = evaluate_quality_threshold_policy_contract(
        {
            "policy_id": "blocked-quality-thresholds",
            "offline_only": True,
            "live_write": True,
            "contacts_wing": True,
            "contacts_backend": True,
            "hardware_evidence": True,
            "readiness_closure_allowed": True,
            "live_authority_granted": True,
            "policy_checks": {"baseline_dataset_bound": True},
            "threshold_policy": {"status": "PENDING_APPROVAL"},
            "thresholds": [complete_threshold("objective_audio_metrics")],
        }
    )
    assert blocked_result["ready_for_owner_review"] is False
    assert "quality_threshold_policy_forbids_live_write" in blocked_result["blockers"]
    assert "quality_threshold_policy_forbids_wing_contact" in blocked_result["blockers"]
    assert "threshold_policy_evidence_cannot_grant_live_authority" in blocked_result[
        "blockers"
    ]
    assert "missing_threshold_axis:reference_mix_comparison" in blocked_result[
        "blockers"
    ]

    assert REQUIRED_THRESHOLD_AXES == REQUIRED_QUALITY_AXES
    assert "Candidate threshold policies must be evidence-bound and owner-approved" in completion
    assert "Unvalidated candidates cannot alter live concert behavior" in safety
    assert "M014-CP01" in plan
    assert "M014-CP02" in plan
    assert "Candidate threshold policy evidence cannot grant live authority" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "always returns `live_authority_granted: false`" in adr
    assert "M014 quality threshold policy contract is now covered" in current
    assert "M014 quality threshold policy contract rail is now automated" in report


def test_m015_replay_threshold_policy_binding_requires_m014_manifest():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M015_PLAN.read_text(encoding="utf-8")
    adr = M015_ADR.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m015 = markdown_table_row(roadmap, "M015")
    assert m015[1] == "Replay threshold policy binding"
    assert m015[2] == (
        "Offline replay promotion consumes the quality threshold policy contract "
        "instead of loose threshold status fields"
    )
    assert m015[3] == "M011-M014"
    assert m015[4] == "high offline"
    assert m015[5] == "UNIT_VERIFIED gate binding; replay evidence remains pending"
    assert m015[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "`backend/offline_replay_evaluation_contract.py`" in memory_eval[2]
    assert "`backend/quality_threshold_policy_contract.py`" in memory_eval[2]
    assert "tests/test_offline_replay_evaluation_contract.py" in memory_eval[3]
    assert "replay promotion now requires a bound quality threshold policy manifest" in memory_eval[4]
    assert memory_eval[-1] == "none"

    def complete_scenario(name: str) -> dict:
        return {
            "scenario": name,
            "evaluation_types": list(REQUIRED_EVALUATION_TYPES),
            "safety_checks": {key: True for key in REQUIRED_SAFETY_CHECKS},
            "quality_axes": {
                key: {"status": "passed", "artifact": f"{name}/{key}.json"}
                for key in REQUIRED_QUALITY_AXES
            },
        }

    def complete_threshold(axis: str) -> dict:
        return {
            "axis": axis,
            "metric_name": f"{axis}_score",
            "direction": "min",
            "candidate_value": 0.8,
            "evidence_artifact": f"thresholds/{axis}.json",
            "rationale": f"Replay promotion threshold for {axis}",
            "rollback_policy": f"Revert {axis} threshold on regression",
        }

    def threshold_policy_manifest(**overrides) -> dict:
        manifest = {
            "policy_id": "offline-replay-threshold-policy",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "readiness_closure_allowed": False,
            "live_authority_granted": False,
            "policy_checks": {key: True for key in THRESHOLD_REQUIRED_POLICY_CHECKS},
            "threshold_policy": {
                "status": "approved",
                "approved_by": "owner",
                "approved_decision_id": "adr:offline-replay-thresholds",
                "approved_at": "2026-06-19T17:32:45Z",
            },
            "thresholds": [complete_threshold(axis) for axis in REQUIRED_THRESHOLD_AXES],
        }
        manifest.update(overrides)
        return manifest

    def integration_policy_manifest(**overrides) -> dict:
        manifest = {
            "policy_id": "integration-policy",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "readiness_closure_allowed": False,
            "live_authority_granted": False,
            "integration_controls": {
                key: True for key in POLICY_REQUIRED_INTEGRATION_CONTROLS
            },
            "integration_evidence": {
                key: True for key in POLICY_REQUIRED_INTEGRATION_EVIDENCE
            },
            "integration_policy": {
                "status": "approved",
                "approved_by": "owner",
                "approved_decision_id": "adr:integration-policy",
                "approved_at": "2026-06-19T18:10:23Z",
            },
        }
        manifest.update(overrides)
        return manifest

    legacy_only_replay = {
        "manifest_id": "legacy-threshold-policy",
        "offline_only": True,
        "live_write": False,
        "contacts_wing": False,
        "contacts_backend": False,
        "hardware_evidence": False,
        "readiness_closure_allowed": False,
        "live_authority_granted": False,
        "threshold_policy": {
            "status": "approved",
            "approved_by": "owner",
            "approved_decision_id": "adr:legacy-threshold-policy",
        },
        "scenarios": [complete_scenario(name) for name in REQUIRED_SCENARIOS],
    }
    legacy_result = evaluate_offline_replay_evaluation_contract(legacy_only_replay)
    assert legacy_result["ready_for_replay_promotion"] is False
    assert "approved_quality_threshold_policy_required" in legacy_result["blockers"]
    assert "quality_threshold_policy_contract_not_ready" in legacy_result["blockers"]
    assert "legacy_threshold_policy_without_contract_manifest" in legacy_result["blockers"]
    assert "quality_threshold_policy:offline_only_required" in legacy_result["blockers"]
    assert legacy_result["summary"]["legacy_threshold_policy_status"] == "approved"
    assert legacy_result["summary"]["thresholds_approved"] is False
    assert legacy_result["authority"]["live_authority_granted"] is False

    complete_replay = {
        "manifest_id": "bound-threshold-policy",
        "offline_only": True,
        "live_write": False,
        "contacts_wing": False,
        "contacts_backend": False,
        "hardware_evidence": False,
        "readiness_closure_allowed": False,
        "live_authority_granted": False,
        "quality_threshold_policy_manifest": threshold_policy_manifest(),
        "scenarios": [complete_scenario(name) for name in REQUIRED_SCENARIOS],
    }
    complete_result = evaluate_offline_replay_evaluation_contract(complete_replay)
    assert complete_result["ready_for_replay_promotion"] is True
    assert complete_result["summary"]["threshold_policy_contract_status"] == (
        "ready_for_threshold_policy_use"
    )
    assert complete_result["threshold_policy_result"]["ready_for_threshold_policy_use"] is True
    assert complete_result["authority"]["readiness_closure_allowed"] is False

    integration_result = evaluate_integration_bundle_contract(
        {
            "manifest_id": "legacy-threshold-integration-bundle",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "readiness_closure_allowed": False,
            "live_authority_granted": False,
            "integration_policy_manifest": integration_policy_manifest(),
            "integration_checks": {key: True for key in REQUIRED_INTEGRATION_CHECKS},
            "offline_replay_manifest": legacy_only_replay,
            "fault_recovery_manifest": {
                "manifest_id": "missing-fault-recovery-for-prefix-check",
                "offline_only": True,
            },
        }
    )
    assert integration_result["ready_for_integration_evaluation_promotion"] is False
    assert "offline_replay:quality_threshold_policy_contract_not_ready" in integration_result[
        "blockers"
    ]
    assert "offline_replay:legacy_threshold_policy_without_contract_manifest" in integration_result[
        "blockers"
    ]

    assert "Replay manifests must consume the quality threshold policy contract result" in completion
    assert "M015-CP01" in plan
    assert "M015-CP02" in plan
    assert "Legacy `threshold_policy` fields without the M014 contract manifest" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "always returns `live_authority_granted: false`" in adr
    assert "M015 replay threshold policy binding is now covered" in current
    assert "M015 replay threshold policy binding rail is now automated" in report


def test_m016_recovery_policy_contract_stays_owner_approved_and_offline():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    safety = SAFETY_CONTRACT.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M016_PLAN.read_text(encoding="utf-8")
    adr = M016_ADR.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m016 = markdown_table_row(roadmap, "M016")
    assert m016[1] == "Recovery policy contract"
    assert m016[2] == (
        "Candidate recovery policies are complete, evidence-bound and "
        "owner-approved before simulator fault-recovery promotion"
    )
    assert m016[3] == "M012-M015"
    assert m016[4] == "high offline"
    assert (
        m016[5]
        == "SPECIFIED to UNIT_VERIFIED for recovery-policy gate; owner approval remains required"
    )
    assert m016[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "`backend/recovery_policy_contract.py`" in memory_eval[2]
    assert "tests/test_recovery_policy_contract.py" in memory_eval[3]
    assert "recovery policy evidence cannot grant live authority" in memory_eval[4]
    assert memory_eval[-1] == "none"

    candidate_result = evaluate_recovery_policy_contract(
        {
            "policy_id": "candidate-recovery-policy",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "readiness_closure_allowed": False,
            "live_authority_granted": False,
            "recovery_controls": {key: True for key in POLICY_REQUIRED_RECOVERY_CONTROLS},
            "recovery_evidence": {key: True for key in POLICY_REQUIRED_RECOVERY_EVIDENCE},
            "recovery_policy": {"status": "candidate"},
        }
    )
    assert candidate_result["ready_for_owner_review"] is True
    assert candidate_result["ready_for_recovery_policy_use"] is False
    assert candidate_result["authority"]["live_authority_granted"] is False
    assert candidate_result["authority"]["readiness_closure_allowed"] is False
    assert candidate_result["review_blockers"] == []
    assert candidate_result["blockers"] == [
        "owner_approved_recovery_policy_required"
    ]

    approved_result = evaluate_recovery_policy_contract(
        {
            "policy_id": "approved-recovery-policy",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "readiness_closure_allowed": False,
            "live_authority_granted": False,
            "recovery_controls": {key: True for key in POLICY_REQUIRED_RECOVERY_CONTROLS},
            "recovery_evidence": {key: True for key in POLICY_REQUIRED_RECOVERY_EVIDENCE},
            "recovery_policy": {
                "status": "approved",
                "approved_by": "owner",
                "approved_decision_id": "adr:recovery-policy",
                "approved_at": "2026-06-19T17:51:02Z",
            },
        }
    )
    assert approved_result["ready_for_recovery_policy_use"] is True
    assert approved_result["recovery_policy_approved"] is True
    assert approved_result["hardware_evidence"] is False
    assert approved_result["authority"]["live_authority_granted"] is False
    assert approved_result["authority"]["readiness_closure_allowed"] is False
    assert approved_result["summary"]["missing_recovery_controls"] == []
    assert approved_result["summary"]["missing_recovery_evidence"] == []

    blocked_result = evaluate_recovery_policy_contract(
        {
            "policy_id": "blocked-recovery-policy",
            "offline_only": True,
            "live_write": True,
            "contacts_wing": True,
            "contacts_backend": True,
            "hardware_evidence": True,
            "readiness_closure_allowed": True,
            "live_authority_granted": True,
            "recovery_controls": {"fault_detection_policy": True},
            "recovery_evidence": {"fault_scenario_matrix_bound": True},
            "recovery_policy": {"status": "PENDING_APPROVAL"},
        }
    )
    assert blocked_result["ready_for_owner_review"] is False
    assert "recovery_policy_forbids_live_write" in blocked_result["blockers"]
    assert "recovery_policy_forbids_wing_contact" in blocked_result["blockers"]
    assert "recovery_policy_evidence_cannot_grant_live_authority" in blocked_result[
        "blockers"
    ]
    assert "missing_recovery_control:fail_closed_policy" in blocked_result["blockers"]

    assert "Recovery policy candidates must be evidence-bound and owner-approved" in completion
    assert "Unvalidated candidates cannot alter live concert behavior" in safety
    assert "M016-CP01" in plan
    assert "M016-CP02" in plan
    assert "Candidate recovery policy evidence cannot grant live authority" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "always returns `live_authority_granted: false`" in adr
    assert "M016 recovery policy contract is now covered" in current
    assert "M016 recovery policy contract rail is now automated" in report


def test_m017_fault_recovery_policy_binding_requires_m016_manifest():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M017_PLAN.read_text(encoding="utf-8")
    adr = M017_ADR.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m017 = markdown_table_row(roadmap, "M017")
    assert m017[1] == "Fault recovery policy binding"
    assert m017[2] == (
        "Simulator fault-recovery promotion consumes the recovery policy "
        "contract instead of loose recovery_policy status fields"
    )
    assert m017[3] == "M012-M016"
    assert m017[4] == "high offline"
    assert m017[5] == "UNIT_VERIFIED gate binding; simulator evidence remains pending"
    assert m017[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "`backend/fault_injection_recovery_contract.py`" in memory_eval[2]
    assert "`backend/recovery_policy_contract.py`" in memory_eval[2]
    assert "tests/test_fault_injection_recovery_contract.py" in memory_eval[3]
    assert "fault-recovery promotion now requires a bound recovery policy manifest" in memory_eval[4]
    assert memory_eval[-1] == "none"

    def complete_fault(name: str) -> dict:
        return {
            "fault_scenario": name,
            "evaluation_types": list(RECOVERY_REQUIRED_EVALUATION_TYPES),
            "recovery_outcomes": {key: True for key in REQUIRED_RECOVERY_OUTCOMES},
            "audit": {key: f"{name}/{key}" for key in RECOVERY_REQUIRED_AUDIT_FIELDS},
        }

    def recovery_policy_manifest(**overrides) -> dict:
        manifest = {
            "policy_id": "approved-recovery-policy",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "readiness_closure_allowed": False,
            "live_authority_granted": False,
            "recovery_controls": {
                key: True for key in POLICY_REQUIRED_RECOVERY_CONTROLS
            },
            "recovery_evidence": {
                key: True for key in POLICY_REQUIRED_RECOVERY_EVIDENCE
            },
            "recovery_policy": {
                "status": "approved",
                "approved_by": "owner",
                "approved_decision_id": "adr:fault-recovery-policy",
                "approved_at": "2026-06-19T18:00:10Z",
            },
        }
        manifest.update(overrides)
        return manifest

    legacy_only_fault_recovery = {
        "manifest_id": "legacy-recovery-policy",
        "offline_only": True,
        "live_write": False,
        "contacts_wing": False,
        "contacts_backend": False,
        "hardware_evidence": False,
        "readiness_closure_allowed": False,
        "live_authority_granted": False,
        "recovery_policy": {
            "status": "approved",
            "approved_by": "owner",
            "approved_decision_id": "adr:legacy-recovery-policy",
        },
        "fault_scenarios": [
            complete_fault(name) for name in RECOVERY_REQUIRED_FAULT_SCENARIOS
        ],
    }
    legacy_result = evaluate_fault_injection_recovery_contract(legacy_only_fault_recovery)
    assert legacy_result["ready_for_fault_recovery_promotion"] is False
    assert "approved_recovery_policy_required" in legacy_result["blockers"]
    assert "recovery_policy_contract_not_ready" in legacy_result["blockers"]
    assert "legacy_recovery_policy_without_contract_manifest" in legacy_result[
        "blockers"
    ]
    assert "recovery_policy:offline_only_required" in legacy_result["blockers"]
    assert legacy_result["summary"]["legacy_recovery_policy_status"] == "approved"
    assert legacy_result["summary"]["recovery_policy_approved"] is False
    assert legacy_result["authority"]["live_authority_granted"] is False

    complete_fault_recovery = {
        "manifest_id": "bound-recovery-policy",
        "offline_only": True,
        "live_write": False,
        "contacts_wing": False,
        "contacts_backend": False,
        "hardware_evidence": False,
        "readiness_closure_allowed": False,
        "live_authority_granted": False,
        "recovery_policy_manifest": recovery_policy_manifest(),
        "fault_scenarios": [
            complete_fault(name) for name in RECOVERY_REQUIRED_FAULT_SCENARIOS
        ],
    }
    complete_result = evaluate_fault_injection_recovery_contract(complete_fault_recovery)
    assert complete_result["ready_for_fault_recovery_promotion"] is True
    assert complete_result["summary"]["recovery_policy_contract_status"] == (
        "ready_for_recovery_policy_use"
    )
    assert complete_result["recovery_policy_result"]["ready_for_recovery_policy_use"] is True
    assert complete_result["authority"]["readiness_closure_allowed"] is False

    def complete_scenario(name: str) -> dict:
        return {
            "scenario": name,
            "evaluation_types": list(REQUIRED_EVALUATION_TYPES),
            "safety_checks": {key: True for key in REQUIRED_SAFETY_CHECKS},
            "quality_axes": {
                key: {"status": "passed", "artifact": f"{name}/{key}.json"}
                for key in REQUIRED_QUALITY_AXES
            },
        }

    def complete_threshold(axis: str) -> dict:
        return {
            "axis": axis,
            "metric_name": f"{axis}_score",
            "direction": "min",
            "candidate_value": 0.8,
            "evidence_artifact": f"thresholds/{axis}.json",
            "rationale": f"Replay promotion threshold for {axis}",
            "rollback_policy": f"Revert {axis} threshold on regression",
        }

    def threshold_policy_manifest(**overrides) -> dict:
        manifest = {
            "policy_id": "offline-replay-threshold-policy",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "readiness_closure_allowed": False,
            "live_authority_granted": False,
            "policy_checks": {key: True for key in THRESHOLD_REQUIRED_POLICY_CHECKS},
            "threshold_policy": {
                "status": "approved",
                "approved_by": "owner",
                "approved_decision_id": "adr:offline-replay-thresholds",
                "approved_at": "2026-06-19T17:32:45Z",
            },
            "thresholds": [complete_threshold(axis) for axis in REQUIRED_THRESHOLD_AXES],
        }
        manifest.update(overrides)
        return manifest

    def integration_policy_manifest(**overrides) -> dict:
        manifest = {
            "policy_id": "integration-policy",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "readiness_closure_allowed": False,
            "live_authority_granted": False,
            "integration_controls": {
                key: True for key in POLICY_REQUIRED_INTEGRATION_CONTROLS
            },
            "integration_evidence": {
                key: True for key in POLICY_REQUIRED_INTEGRATION_EVIDENCE
            },
            "integration_policy": {
                "status": "approved",
                "approved_by": "owner",
                "approved_decision_id": "adr:integration-policy",
                "approved_at": "2026-06-19T18:10:23Z",
            },
        }
        manifest.update(overrides)
        return manifest

    integration_result = evaluate_integration_bundle_contract(
        {
            "manifest_id": "legacy-recovery-policy-integration-bundle",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "readiness_closure_allowed": False,
            "live_authority_granted": False,
            "integration_policy_manifest": integration_policy_manifest(),
            "integration_checks": {key: True for key in REQUIRED_INTEGRATION_CHECKS},
            "offline_replay_manifest": {
                "manifest_id": "bound-threshold-policy",
                "offline_only": True,
                "live_write": False,
                "contacts_wing": False,
                "contacts_backend": False,
                "hardware_evidence": False,
                "readiness_closure_allowed": False,
                "live_authority_granted": False,
                "quality_threshold_policy_manifest": threshold_policy_manifest(),
                "scenarios": [complete_scenario(name) for name in REQUIRED_SCENARIOS],
            },
            "fault_recovery_manifest": legacy_only_fault_recovery,
        }
    )
    assert integration_result["ready_for_integration_evaluation_promotion"] is False
    assert "fault_recovery:recovery_policy_contract_not_ready" in integration_result[
        "blockers"
    ]
    assert "fault_recovery:legacy_recovery_policy_without_contract_manifest" in integration_result[
        "blockers"
    ]

    assert "Fault-recovery manifests must consume the recovery policy contract result" in completion
    assert "M017-CP01" in plan
    assert "M017-CP02" in plan
    assert "Legacy `recovery_policy` fields without the M016 contract manifest" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "always returns `live_authority_granted: false`" in adr
    assert "M017 fault recovery policy binding is now covered" in current
    assert "M017 fault recovery policy binding rail is now automated" in report


def test_m018_integration_policy_contract_stays_owner_approved_and_offline():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    safety = SAFETY_CONTRACT.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M018_PLAN.read_text(encoding="utf-8")
    adr = M018_ADR.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m018 = markdown_table_row(roadmap, "M018")
    assert m018[1] == "Integration policy contract"
    assert m018[2] == (
        "Candidate integration policies are complete, evidence-bound and "
        "owner-approved before offline integration promotion"
    )
    assert m018[3] == "M013-M017"
    assert m018[4] == "high offline"
    assert (
        m018[5]
        == "SPECIFIED to UNIT_VERIFIED for integration-policy gate; owner approval remains required"
    )
    assert m018[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "`backend/integration_policy_contract.py`" in memory_eval[2]
    assert "tests/test_integration_policy_contract.py" in memory_eval[3]
    assert "integration policy evidence cannot grant live authority" in memory_eval[4]
    assert memory_eval[-1] == "none"

    candidate_result = evaluate_integration_policy_contract(
        {
            "policy_id": "candidate-integration-policy",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "readiness_closure_allowed": False,
            "live_authority_granted": False,
            "integration_controls": {
                key: True for key in POLICY_REQUIRED_INTEGRATION_CONTROLS
            },
            "integration_evidence": {
                key: True for key in POLICY_REQUIRED_INTEGRATION_EVIDENCE
            },
            "integration_policy": {"status": "candidate"},
        }
    )
    assert candidate_result["ready_for_owner_review"] is True
    assert candidate_result["ready_for_integration_policy_use"] is False
    assert candidate_result["authority"]["live_authority_granted"] is False
    assert candidate_result["authority"]["readiness_closure_allowed"] is False
    assert candidate_result["review_blockers"] == []
    assert candidate_result["blockers"] == [
        "owner_approved_integration_policy_required"
    ]

    approved_result = evaluate_integration_policy_contract(
        {
            "policy_id": "approved-integration-policy",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "readiness_closure_allowed": False,
            "live_authority_granted": False,
            "integration_controls": {
                key: True for key in POLICY_REQUIRED_INTEGRATION_CONTROLS
            },
            "integration_evidence": {
                key: True for key in POLICY_REQUIRED_INTEGRATION_EVIDENCE
            },
            "integration_policy": {
                "status": "approved",
                "approved_by": "owner",
                "approved_decision_id": "adr:integration-policy",
                "approved_at": "2026-06-19T18:10:23Z",
            },
        }
    )
    assert approved_result["ready_for_integration_policy_use"] is True
    assert approved_result["integration_policy_approved"] is True
    assert approved_result["hardware_evidence"] is False
    assert approved_result["authority"]["live_authority_granted"] is False
    assert approved_result["authority"]["readiness_closure_allowed"] is False
    assert approved_result["summary"]["missing_integration_controls"] == []
    assert approved_result["summary"]["missing_integration_evidence"] == []

    blocked_result = evaluate_integration_policy_contract(
        {
            "policy_id": "blocked-integration-policy",
            "offline_only": True,
            "live_write": True,
            "contacts_wing": True,
            "contacts_backend": True,
            "hardware_evidence": True,
            "readiness_closure_allowed": True,
            "live_authority_granted": True,
            "integration_controls": {"analysis_trace_policy": True},
            "integration_evidence": {"analysis_trace_bound": True},
            "integration_policy": {"status": "PENDING_APPROVAL"},
        }
    )
    assert blocked_result["ready_for_owner_review"] is False
    assert "integration_policy_forbids_live_write" in blocked_result["blockers"]
    assert "integration_policy_forbids_wing_contact" in blocked_result["blockers"]
    assert "integration_policy_evidence_cannot_grant_live_authority" in blocked_result[
        "blockers"
    ]
    assert "missing_integration_control:offline_replay_binding_policy" in blocked_result[
        "blockers"
    ]

    assert "Integration policy candidates must be evidence-bound and owner-approved" in completion
    assert "Unvalidated candidates cannot alter live concert behavior" in safety
    assert "M018-CP01" in plan
    assert "M018-CP02" in plan
    assert "Candidate integration policy evidence cannot grant live authority" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "always returns `live_authority_granted: false`" in adr
    assert "M018 integration policy contract is now covered" in current
    assert "M018 integration policy contract rail is now automated" in report


def test_m019_integration_policy_binding_requires_m018_manifest():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M019_PLAN.read_text(encoding="utf-8")
    adr = M019_ADR.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m019 = markdown_table_row(roadmap, "M019")
    assert m019[1] == "Integration policy binding"
    assert m019[2] == (
        "Offline integration promotion consumes the integration policy contract "
        "instead of loose integration_policy status fields"
    )
    assert m019[3] == "M013-M018"
    assert m019[4] == "high offline"
    assert m019[5] == "UNIT_VERIFIED gate binding; integration evidence remains pending"
    assert m019[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "tests/test_evaluation_integration_bundle_contract.py" in memory_eval[3]
    assert "integration promotion now requires a bound integration policy manifest" in (
        memory_eval[4]
    )
    assert memory_eval[-1] == "none"

    def complete_scenario(name: str) -> dict:
        return {
            "scenario": name,
            "evaluation_types": list(REQUIRED_EVALUATION_TYPES),
            "safety_checks": {key: True for key in REQUIRED_SAFETY_CHECKS},
            "quality_axes": {
                key: {"status": "passed", "artifact": f"{name}/{key}.json"}
                for key in REQUIRED_QUALITY_AXES
            },
        }

    def complete_threshold(axis: str) -> dict:
        return {
            "axis": axis,
            "metric_name": f"{axis}_score",
            "direction": "min",
            "candidate_value": 0.8,
            "evidence_artifact": f"thresholds/{axis}.json",
            "rationale": f"Replay promotion threshold for {axis}",
            "rollback_policy": f"Revert {axis} threshold on regression",
        }

    def threshold_policy_manifest(**overrides) -> dict:
        manifest = {
            "policy_id": "offline-replay-threshold-policy",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "readiness_closure_allowed": False,
            "live_authority_granted": False,
            "policy_checks": {key: True for key in THRESHOLD_REQUIRED_POLICY_CHECKS},
            "threshold_policy": {
                "status": "approved",
                "approved_by": "owner",
                "approved_decision_id": "adr:offline-replay-thresholds",
                "approved_at": "2026-06-19T17:32:45Z",
            },
            "thresholds": [complete_threshold(axis) for axis in REQUIRED_THRESHOLD_AXES],
        }
        manifest.update(overrides)
        return manifest

    def replay_manifest() -> dict:
        return {
            "manifest_id": "bound-threshold-policy",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "readiness_closure_allowed": False,
            "live_authority_granted": False,
            "quality_threshold_policy_manifest": threshold_policy_manifest(),
            "scenarios": [complete_scenario(name) for name in REQUIRED_SCENARIOS],
        }

    def complete_fault(name: str) -> dict:
        return {
            "fault_scenario": name,
            "evaluation_types": list(RECOVERY_REQUIRED_EVALUATION_TYPES),
            "recovery_outcomes": {key: True for key in REQUIRED_RECOVERY_OUTCOMES},
            "audit": {key: f"{name}/{key}" for key in RECOVERY_REQUIRED_AUDIT_FIELDS},
        }

    def recovery_policy_manifest(**overrides) -> dict:
        manifest = {
            "policy_id": "fault-recovery-policy",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "readiness_closure_allowed": False,
            "live_authority_granted": False,
            "recovery_controls": {key: True for key in POLICY_REQUIRED_RECOVERY_CONTROLS},
            "recovery_evidence": {key: True for key in POLICY_REQUIRED_RECOVERY_EVIDENCE},
            "recovery_policy": {
                "status": "approved",
                "approved_by": "owner",
                "approved_decision_id": "adr:fault-recovery-policy",
                "approved_at": "2026-06-19T18:00:10Z",
            },
        }
        manifest.update(overrides)
        return manifest

    def fault_manifest() -> dict:
        return {
            "manifest_id": "bound-recovery-policy",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "readiness_closure_allowed": False,
            "live_authority_granted": False,
            "recovery_policy_manifest": recovery_policy_manifest(),
            "fault_scenarios": [
                complete_fault(name) for name in RECOVERY_REQUIRED_FAULT_SCENARIOS
            ],
        }

    def integration_policy_manifest(**overrides) -> dict:
        manifest = {
            "policy_id": "integration-policy",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "readiness_closure_allowed": False,
            "live_authority_granted": False,
            "integration_controls": {
                key: True for key in POLICY_REQUIRED_INTEGRATION_CONTROLS
            },
            "integration_evidence": {
                key: True for key in POLICY_REQUIRED_INTEGRATION_EVIDENCE
            },
            "integration_policy": {
                "status": "approved",
                "approved_by": "owner",
                "approved_decision_id": "adr:integration-policy",
                "approved_at": "2026-06-19T18:10:23Z",
            },
        }
        manifest.update(overrides)
        return manifest

    def integration_bundle(**overrides) -> dict:
        bundle = {
            "manifest_id": "integration-policy-binding",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "readiness_closure_allowed": False,
            "live_authority_granted": False,
            "integration_policy_manifest": integration_policy_manifest(),
            "safety_kernel_trace_manifest": safety_kernel_trace_manifest_fixture(),
            "no_live_transport_manifest": no_live_transport_manifest_fixture(),
            "operator_report_manifest": operator_report_manifest_fixture(),
            "integration_checks": {key: True for key in REQUIRED_INTEGRATION_CHECKS},
            "offline_replay_manifest": replay_manifest(),
            "fault_recovery_manifest": fault_manifest(),
        }
        bundle.update(overrides)
        return bundle

    legacy_result = evaluate_integration_bundle_contract(
        integration_bundle(
            integration_policy_manifest={},
            integration_policy={
                "status": "approved",
                "approved_by": "owner",
                "approved_decision_id": "adr:legacy-integration-policy",
            },
        )
    )
    assert legacy_result["ready_for_integration_evaluation_promotion"] is False
    assert "approved_integration_policy_required" in legacy_result["blockers"]
    assert "integration_policy_contract_not_ready" in legacy_result["blockers"]
    assert "legacy_integration_policy_without_contract_manifest" in legacy_result[
        "blockers"
    ]
    assert "integration_policy:offline_only_required" in legacy_result["blockers"]
    assert legacy_result["summary"]["legacy_integration_policy_status"] == "approved"
    assert legacy_result["summary"]["integration_policy_approved"] is False
    assert legacy_result["authority"]["live_authority_granted"] is False

    pending_manifest_result = evaluate_integration_bundle_contract(
        integration_bundle(
            integration_policy_manifest=integration_policy_manifest(
                integration_policy={"status": "PENDING_APPROVAL"}
            )
        )
    )
    assert (
        "integration_policy:owner_approved_integration_policy_required"
        in pending_manifest_result["blockers"]
    )
    assert pending_manifest_result["child_results"]["integration_policy"][
        "ready_for_owner_review"
    ] is True
    assert pending_manifest_result["child_results"]["integration_policy"][
        "ready_for_integration_policy_use"
    ] is False

    bound_result = evaluate_integration_bundle_contract(integration_bundle())
    assert bound_result["ready_for_integration_evaluation_promotion"] is True
    assert bound_result["summary"]["integration_policy_contract_status"] == (
        "ready_for_integration_policy_use"
    )
    assert bound_result["child_results"]["integration_policy"][
        "ready_for_integration_policy_use"
    ] is True
    assert bound_result["authority"]["readiness_closure_allowed"] is False

    assert "Integration bundle manifests must consume the integration policy contract result" in completion
    assert "M019-CP01" in plan
    assert "M019-CP02" in plan
    assert "Legacy `integration_policy` fields without the M018 contract manifest" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "always returns `live_authority_granted: false`" in adr
    assert "M019 integration policy binding is now covered" in current
    assert "M019 integration policy binding rail is now automated" in report


def test_m020_safety_kernel_trace_contract_stays_offline_and_non_bypassable():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    safety = SAFETY_CONTRACT.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M020_PLAN.read_text(encoding="utf-8")
    adr = M020_ADR.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m020 = markdown_table_row(roadmap, "M020")
    assert m020[1] == "Safety kernel trace contract"
    assert m020[2] == (
        "Deterministic safety-kernel traces are complete, audited and "
        "non-bypassable before integration use"
    )
    assert m020[3] == "M013-M019"
    assert m020[4] == "high offline"
    assert (
        m020[5]
        == "SPECIFIED to UNIT_VERIFIED for safety-kernel trace gate; integration binding remains pending"
    )
    assert m020[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "`backend/safety_kernel_trace_contract.py`" in memory_eval[2]
    assert "tests/test_safety_kernel_trace_contract.py" in memory_eval[3]
    assert "deterministic safety-kernel trace evidence cannot grant live authority" in (
        memory_eval[4]
    )
    assert memory_eval[-1] == "none"

    def trace_record(scenario: str, **overrides) -> dict:
        decision = (
            "bounded_allow"
            if scenario == "supervised_fader_gain_allowed"
            else "reject"
        )
        record = {
            "scenario": scenario,
            "safety_decision": decision,
            "guard_checks": {
                key: True for key in SAFETY_KERNEL_REQUIRED_GUARD_CHECKS
            },
            "audit": {
                "decision_id": f"{scenario}-decision",
                "checked_at": "2026-06-19T18:39:18Z",
                "candidate_action": f"{scenario}-candidate",
                "safety_decision": decision,
                "reason": f"{scenario} exercised deterministic safety kernel",
                "policy_version": "safety-kernel-offline-v1",
                "audit_reference": f"artifacts/safety-kernel/{scenario}.json",
            },
        }
        record.update(overrides)
        return record

    def trace_manifest(**overrides) -> dict:
        manifest = {
            "trace_id": "safety-kernel-trace",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "readiness_closure_allowed": False,
            "live_authority_granted": False,
            "trace_records": [
                trace_record(scenario)
                for scenario in SAFETY_KERNEL_REQUIRED_TRACE_SCENARIOS
            ],
        }
        manifest.update(overrides)
        return manifest

    incomplete_result = evaluate_safety_kernel_trace_contract(
        trace_manifest(
            trace_records=[
                trace_record(
                    "supervised_fader_gain_allowed",
                    guard_checks={"mode_permission_checked": True},
                )
            ]
        )
    )
    assert incomplete_result["ready_for_safety_kernel_trace_use"] is False
    assert "missing_required_trace_scenario:unsupported_live_write_rejected" in (
        incomplete_result["blockers"]
    )
    assert (
        "supervised_fader_gain_allowed:missing_guard_check:"
        "live_write_allowlist_checked"
        in incomplete_result["blockers"]
    )
    assert incomplete_result["authority"]["live_authority_granted"] is False

    complete_result = evaluate_safety_kernel_trace_contract(trace_manifest())
    assert complete_result["ready_for_safety_kernel_trace_use"] is True
    assert complete_result["hardware_evidence"] is False
    assert complete_result["summary"]["missing_trace_scenarios"] == []
    assert complete_result["authority"]["live_authority_granted"] is False
    assert complete_result["authority"]["readiness_closure_allowed"] is False

    blocked_result = evaluate_safety_kernel_trace_contract(
        trace_manifest(
            offline_only=False,
            live_write=True,
            contacts_wing=True,
            contacts_backend=True,
            hardware_evidence=True,
            readiness_closure_allowed=True,
            live_authority_granted=True,
        )
    )
    assert blocked_result["ready_for_safety_kernel_trace_use"] is False
    assert "offline_only_required" in blocked_result["blockers"]
    assert "safety_kernel_trace_forbids_live_write" in blocked_result["blockers"]
    assert "safety_kernel_trace_forbids_wing_contact" in blocked_result["blockers"]
    assert "safety_kernel_trace_cannot_grant_live_authority" in blocked_result[
        "blockers"
    ]
    assert blocked_result["authority"]["live_authority_granted"] is False

    wrong_decision_result = evaluate_safety_kernel_trace_contract(
        trace_manifest(
            trace_records=[
                trace_record(
                    scenario,
                    safety_decision="allow",
                    audit={
                        key: f"audit-{key}"
                        for key in SAFETY_KERNEL_REQUIRED_AUDIT_FIELDS
                    }
                    | {"safety_decision": "allow"},
                )
                if scenario == "unsupported_live_write_rejected"
                else trace_record(scenario)
                for scenario in SAFETY_KERNEL_REQUIRED_TRACE_SCENARIOS
            ]
        )
    )
    assert wrong_decision_result["ready_for_safety_kernel_trace_use"] is False
    assert "unsupported_live_write_rejected:unexpected_safety_decision" in (
        wrong_decision_result["blockers"]
    )

    assert "Safety-kernel trace candidates must be offline, complete and audited" in (
        completion
    )
    assert "deterministic safety kernel is the non-bypassable authority" in safety
    assert "M020-CP01" in plan
    assert "M020-CP02" in plan
    assert "Safety-kernel trace evidence cannot grant live authority" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "always returns `live_authority_granted: false`" in adr
    assert "M020 safety kernel trace contract is now covered" in current
    assert "M020 safety kernel trace contract rail is now automated" in report


def test_m021_safety_kernel_trace_binding_requires_m020_manifest():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M021_PLAN.read_text(encoding="utf-8")
    adr = M021_ADR.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m021 = markdown_table_row(roadmap, "M021")
    assert m021[1] == "Safety kernel trace binding"
    assert m021[2] == (
        "Offline integration promotion consumes the safety-kernel trace "
        "contract instead of loose safety_kernel_trace_bound checks"
    )
    assert m021[3] == "M013-M020"
    assert m021[4] == "high offline"
    assert m021[5] == "UNIT_VERIFIED gate binding; integration evidence remains pending"
    assert m021[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "tests/test_evaluation_integration_bundle_contract.py" in memory_eval[3]
    assert "integration promotion now requires a bound safety-kernel trace manifest" in (
        memory_eval[4]
    )
    assert memory_eval[-1] == "none"

    def trace_record(scenario: str, **overrides) -> dict:
        decision = (
            "bounded_allow"
            if scenario == "supervised_fader_gain_allowed"
            else "reject"
        )
        record = {
            "scenario": scenario,
            "safety_decision": decision,
            "guard_checks": {
                key: True for key in SAFETY_KERNEL_REQUIRED_GUARD_CHECKS
            },
            "audit": {
                key: f"{scenario}/{key}"
                for key in SAFETY_KERNEL_REQUIRED_AUDIT_FIELDS
            }
            | {"safety_decision": decision},
        }
        record.update(overrides)
        return record

    def trace_manifest(**overrides) -> dict:
        manifest = {
            "trace_id": "m021-safety-kernel-trace",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "readiness_closure_allowed": False,
            "live_authority_granted": False,
            "trace_records": [
                trace_record(scenario)
                for scenario in SAFETY_KERNEL_REQUIRED_TRACE_SCENARIOS
            ],
        }
        manifest.update(overrides)
        return manifest

    def partial_integration_bundle(**overrides) -> dict:
        bundle = {
            "manifest_id": "m021-integration-bundle",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "readiness_closure_allowed": False,
            "live_authority_granted": False,
            "integration_checks": {key: True for key in REQUIRED_INTEGRATION_CHECKS},
        }
        bundle.update(overrides)
        return bundle

    legacy_result = evaluate_integration_bundle_contract(
        partial_integration_bundle(safety_kernel_trace_manifest={})
    )
    assert legacy_result["ready_for_integration_evaluation_promotion"] is False
    assert "safety_kernel_trace_contract_not_ready" in legacy_result["blockers"]
    assert "legacy_safety_kernel_trace_bound_without_contract_manifest" in (
        legacy_result["blockers"]
    )
    assert "safety_kernel_trace:offline_only_required" in legacy_result["blockers"]
    assert legacy_result["summary"]["safety_kernel_trace_ready"] is False
    assert legacy_result["child_results"]["safety_kernel_trace"]["status"] == "blocked"

    bound_trace_result = evaluate_integration_bundle_contract(
        partial_integration_bundle(safety_kernel_trace_manifest=trace_manifest())
    )
    assert "safety_kernel_trace_contract_not_ready" not in (
        bound_trace_result["blockers"]
    )
    assert not any(
        blocker.startswith("safety_kernel_trace:")
        for blocker in bound_trace_result["blockers"]
    )
    assert bound_trace_result["summary"]["safety_kernel_trace_ready"] is True
    assert (
        bound_trace_result["child_results"]["safety_kernel_trace"][
            "ready_for_safety_kernel_trace_use"
        ]
        is True
    )
    assert bound_trace_result["authority"]["live_authority_granted"] is False

    assert "Integration bundle manifests must consume the safety-kernel trace contract" in (
        completion
    )
    assert "M021-CP01" in plan
    assert "M021-CP02" in plan
    assert "loose `safety_kernel_trace_bound` integration checks" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "always returns `live_authority_granted: false`" in adr
    assert "M021 safety kernel trace binding is now covered" in current
    assert "M021 safety kernel trace binding rail is now automated" in report


def test_m022_no_live_transport_contract_stays_offline_and_artifact_only():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M022_PLAN.read_text(encoding="utf-8")
    adr = M022_ADR.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m022 = markdown_table_row(roadmap, "M022")
    assert m022[1] == "No-live-transport contract"
    assert m022[2] == (
        "Offline integration candidates prove no live transport, backend "
        "contact, console contact or live write occurred"
    )
    assert m022[3] == "M013-M021"
    assert m022[4] == "high offline"
    assert (
        m022[5]
        == "SPECIFIED to UNIT_VERIFIED for no-live-transport gate; integration binding remains pending"
    )
    assert m022[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "`backend/no_live_transport_contract.py`" in memory_eval[2]
    assert "tests/test_no_live_transport_contract.py" in memory_eval[3]
    assert "no-live-transport evidence cannot grant live authority" in memory_eval[4]
    assert memory_eval[-1] == "none"

    def evidence(name: str) -> dict:
        return {
            "status": "passed",
            "artifact": f"artifacts/no-live-transport/{name}.json",
            "sha256": f"{name}-sha256",
        }

    def transport_manifest(**overrides) -> dict:
        manifest = {
            "transport_id": "m022-no-live-transport",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "readiness_closure_allowed": False,
            "live_authority_granted": False,
            "transport_controls": {
                key: True for key in REQUIRED_TRANSPORT_CONTROLS
            },
            "transport_evidence": {
                key: evidence(key) for key in REQUIRED_TRANSPORT_EVIDENCE
            },
            "runtime_observations": {
                key: False if key == "websocket_opened" else 0
                for key in REQUIRED_ZERO_RUNTIME_OBSERVATIONS
            },
            "forbidden_transport_tokens": [],
        }
        manifest.update(overrides)
        return manifest

    incomplete_result = evaluate_no_live_transport_contract(
        transport_manifest(
            transport_controls={"offline_executor_only": True},
            transport_evidence={"source_audit_artifact": evidence("source")},
            runtime_observations={"websocket_opened": False},
        )
    )
    assert incomplete_result["ready_for_no_live_transport_use"] is False
    assert "missing_transport_control:no_wing_client" in incomplete_result["blockers"]
    assert "missing_transport_evidence:runtime_config_artifact" in (
        incomplete_result["blockers"]
    )
    assert "missing_runtime_observation:osc_messages_sent" in (
        incomplete_result["blockers"]
    )
    assert incomplete_result["authority"]["live_authority_granted"] is False

    complete_result = evaluate_no_live_transport_contract(transport_manifest())
    assert complete_result["ready_for_no_live_transport_use"] is True
    assert complete_result["hardware_evidence"] is False
    assert complete_result["summary"]["missing_transport_controls"] == []
    assert complete_result["summary"]["missing_transport_evidence"] == []
    assert complete_result["summary"]["nonzero_runtime_observations"] == []
    assert complete_result["authority"]["readiness_closure_allowed"] is False

    blocked_result = evaluate_no_live_transport_contract(
        transport_manifest(
            offline_only=False,
            live_write=True,
            contacts_wing=True,
            contacts_backend=True,
            hardware_evidence=True,
            readiness_closure_allowed=True,
            live_authority_granted=True,
            runtime_observations={
                "websocket_opened": True,
                "osc_messages_sent": 1,
                "live_write_requests_sent": 1,
                "mixer_client_calls": 1,
                "network_connections_opened": 1,
            },
            forbidden_transport_tokens=["WingClient"],
        )
    )
    assert blocked_result["ready_for_no_live_transport_use"] is False
    assert "offline_only_required" in blocked_result["blockers"]
    assert "no_live_transport_forbids_live_write" in blocked_result["blockers"]
    assert "no_live_transport_forbids_wing_contact" in blocked_result["blockers"]
    assert "no_live_transport_cannot_grant_live_authority" in (
        blocked_result["blockers"]
    )
    assert "live_transport_observed:websocket_opened" in blocked_result["blockers"]
    assert "forbidden_transport_token:WingClient" in blocked_result["blockers"]
    assert blocked_result["authority"]["live_authority_granted"] is False

    assert "No-live-transport evidence candidates must be offline" in completion
    assert "M022-CP01" in plan
    assert "M022-CP02" in plan
    assert "No-live-transport evidence cannot grant live authority" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "always returns `live_authority_granted: false`" in adr
    assert "M022 no-live-transport contract is now covered" in current
    assert "M022 no-live-transport contract rail is now automated" in report


def test_m023_no_live_transport_binding_requires_m022_manifest():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M023_PLAN.read_text(encoding="utf-8")
    adr = M023_ADR.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m023 = markdown_table_row(roadmap, "M023")
    assert m023[1] == "No-live-transport binding"
    assert m023[2] == (
        "Offline integration promotion consumes the no-live-transport "
        "contract instead of loose no_live_transport checks"
    )
    assert m023[3] == "M013-M022"
    assert m023[4] == "high offline"
    assert m023[5] == "UNIT_VERIFIED gate binding; integration evidence remains pending"
    assert m023[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "tests/test_evaluation_integration_bundle_contract.py" in memory_eval[3]
    assert "integration promotion now requires a bound no-live-transport manifest" in (
        memory_eval[4]
    )
    assert memory_eval[-1] == "none"

    def integration_bundle(**overrides) -> dict:
        bundle = {
            "manifest_id": "m023-integration-bundle",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "readiness_closure_allowed": False,
            "live_authority_granted": False,
            "integration_checks": {key: True for key in REQUIRED_INTEGRATION_CHECKS},
        }
        bundle.update(overrides)
        return bundle

    legacy_result = evaluate_integration_bundle_contract(
        integration_bundle(no_live_transport_manifest={})
    )
    assert legacy_result["ready_for_integration_evaluation_promotion"] is False
    assert "no_live_transport_contract_not_ready" in legacy_result["blockers"]
    assert "legacy_no_live_transport_without_contract_manifest" in legacy_result[
        "blockers"
    ]
    assert "no_live_transport:offline_only_required" in legacy_result["blockers"]
    assert (
        "no_live_transport:missing_transport_control:offline_executor_only"
        in legacy_result["blockers"]
    )
    assert legacy_result["summary"]["no_live_transport_ready"] is False
    assert legacy_result["child_results"]["no_live_transport"]["status"] == "blocked"

    bound_transport_result = evaluate_integration_bundle_contract(
        integration_bundle(
            no_live_transport_manifest=no_live_transport_manifest_fixture()
        )
    )
    assert "no_live_transport_contract_not_ready" not in (
        bound_transport_result["blockers"]
    )
    assert not any(
        blocker.startswith("no_live_transport:")
        for blocker in bound_transport_result["blockers"]
    )
    assert bound_transport_result["summary"]["no_live_transport_ready"] is True
    assert (
        bound_transport_result["child_results"]["no_live_transport"][
            "ready_for_no_live_transport_use"
        ]
        is True
    )
    assert bound_transport_result["authority"]["live_authority_granted"] is False

    assert "Integration bundle manifests must consume the no-live-transport contract" in (
        completion
    )
    assert "M023-CP01" in plan
    assert "M023-CP02" in plan
    assert "loose `no_live_transport` integration checks" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "always returns `live_authority_granted: false`" in adr
    assert "M023 no-live-transport binding is now covered" in current
    assert "M023 no-live-transport binding rail is now automated" in report


def test_m024_operator_report_contract_stays_offline_and_review_bound():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M024_PLAN.read_text(encoding="utf-8")
    adr = M024_ADR.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m024 = markdown_table_row(roadmap, "M024")
    assert m024[1] == "Operator report contract"
    assert m024[2] == (
        "Offline integration reports prove operator-visible status, blockers, "
        "authority denial, child evidence and next action before review use"
    )
    assert m024[3] == "M013-M023"
    assert m024[4] == "high offline"
    assert (
        m024[5]
        == "SPECIFIED to UNIT_VERIFIED for operator-report gate; integration binding remains pending"
    )
    assert m024[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "`backend/operator_report_contract.py`" in memory_eval[2]
    assert "tests/test_operator_report_contract.py" in memory_eval[3]
    assert "operator-report evidence cannot grant live authority" in memory_eval[4]
    assert memory_eval[-1] == "none"

    def binding(name: str) -> dict:
        return {
            "status": "passed",
            "artifact": f"operator-report/{name}.json",
            "sha256": f"{name}-sha256",
        }

    def operator_report(**overrides) -> dict:
        manifest = {
            "report_id": "m024-operator-report",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "readiness_closure_allowed": False,
            "live_authority_granted": False,
            "report_sections": {
                key: True for key in REQUIRED_REPORT_SECTIONS
            },
            "report_bindings": {
                key: binding(key) for key in REQUIRED_REPORT_BINDINGS
            },
            "authority_denial": {
                key: False for key in OPERATOR_REPORT_REQUIRED_AUTHORITY_DENIAL_FLAGS
            },
            "visible_blockers": [
                "M001 waits for real WING read-only artifact",
                "operator review required",
            ],
            "next_operator_action": "Run WING read-only preflight with operator.",
            "audit": {
                key: f"operator-report/{key}"
                for key in OPERATOR_REPORT_REQUIRED_AUDIT_FIELDS
            },
        }
        manifest.update(overrides)
        return manifest

    incomplete_result = evaluate_operator_report_contract(
        operator_report(
            report_sections={"current_status": True},
            report_bindings={"integration_bundle_artifact": binding("bundle")},
            authority_denial={"live_authority_granted": False},
            visible_blockers=[],
            next_operator_action="",
            audit={"report_id": "m024-operator-report"},
        )
    )
    assert incomplete_result["ready_for_operator_report_use"] is False
    assert "missing_report_section:authority_denial" in incomplete_result["blockers"]
    assert "missing_report_binding:offline_replay_result" in (
        incomplete_result["blockers"]
    )
    assert "missing_authority_denial_flag:readiness_closure_allowed" in (
        incomplete_result["blockers"]
    )
    assert "visible_blockers_required" in incomplete_result["blockers"]
    assert "next_operator_action_required" in incomplete_result["blockers"]
    assert incomplete_result["authority"]["live_authority_granted"] is False

    complete_result = evaluate_operator_report_contract(operator_report())
    assert complete_result["ready_for_operator_report_use"] is True
    assert complete_result["hardware_evidence"] is False
    assert complete_result["summary"]["missing_report_sections"] == []
    assert complete_result["summary"]["missing_report_bindings"] == []
    assert complete_result["summary"]["non_denied_authority_flags"] == []
    assert complete_result["authority"]["readiness_closure_allowed"] is False
    assert complete_result["authority"]["live_authority_granted"] is False

    blocked_result = evaluate_operator_report_contract(
        operator_report(
            offline_only=False,
            live_write=True,
            contacts_wing=True,
            contacts_backend=True,
            hardware_evidence=True,
            readiness_closure_allowed=True,
            live_authority_granted=True,
            authority_denial={
                key: False for key in OPERATOR_REPORT_REQUIRED_AUTHORITY_DENIAL_FLAGS
            }
            | {"live_authority_granted": True},
        )
    )
    assert blocked_result["ready_for_operator_report_use"] is False
    assert "operator_report_forbids_live_write" in blocked_result["blockers"]
    assert "operator_report_forbids_wing_contact" in blocked_result["blockers"]
    assert "operator_report_cannot_grant_live_authority" in blocked_result[
        "blockers"
    ]
    assert "authority_not_denied:live_authority_granted" in blocked_result[
        "blockers"
    ]

    assert "Operator report candidates must be offline" in completion
    assert "M024-CP01" in plan
    assert "M024-CP02" in plan
    assert "Operator-report evidence cannot grant live authority" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "always returns `live_authority_granted: false`" in adr
    assert "M024 operator report contract is now covered" in current
    assert "M024 operator report contract rail is now automated" in report


def test_m025_operator_report_binding_requires_m024_manifest():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M025_PLAN.read_text(encoding="utf-8")
    adr = M025_ADR.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m025 = markdown_table_row(roadmap, "M025")
    assert m025[1] == "Operator report binding"
    assert m025[2] == (
        "Offline integration promotion consumes the operator-report contract "
        "instead of loose operator_report_bound checks"
    )
    assert m025[3] == "M013-M024"
    assert m025[4] == "high offline"
    assert m025[5] == "UNIT_VERIFIED gate binding; integration evidence remains pending"
    assert m025[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "tests/test_evaluation_integration_bundle_contract.py" in memory_eval[3]
    assert "integration promotion now requires a bound operator-report manifest" in (
        memory_eval[4]
    )
    assert memory_eval[-1] == "none"

    def binding(name: str) -> dict:
        return {
            "status": "passed",
            "artifact": f"operator-report/{name}.json",
            "sha256": f"{name}-sha256",
        }

    def operator_report_manifest(**overrides) -> dict:
        manifest = {
            "report_id": "m025-operator-report",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "readiness_closure_allowed": False,
            "live_authority_granted": False,
            "report_sections": {key: True for key in REQUIRED_REPORT_SECTIONS},
            "report_bindings": {key: binding(key) for key in REQUIRED_REPORT_BINDINGS},
            "authority_denial": {
                key: False for key in OPERATOR_REPORT_REQUIRED_AUTHORITY_DENIAL_FLAGS
            },
            "visible_blockers": [
                "M001 waits for real WING read-only artifact",
                "operator review required",
            ],
            "next_operator_action": "Run WING read-only preflight with operator.",
            "audit": {
                key: f"operator-report/{key}"
                for key in OPERATOR_REPORT_REQUIRED_AUDIT_FIELDS
            },
        }
        manifest.update(overrides)
        return manifest

    def integration_bundle(**overrides) -> dict:
        bundle = {
            "manifest_id": "m025-integration-bundle",
            "offline_only": True,
            "live_write": False,
            "contacts_wing": False,
            "contacts_backend": False,
            "hardware_evidence": False,
            "readiness_closure_allowed": False,
            "live_authority_granted": False,
            "integration_checks": {key: True for key in REQUIRED_INTEGRATION_CHECKS},
        }
        bundle.update(overrides)
        return bundle

    legacy_result = evaluate_integration_bundle_contract(
        integration_bundle(operator_report_manifest={})
    )
    assert legacy_result["ready_for_integration_evaluation_promotion"] is False
    assert "operator_report_contract_not_ready" in legacy_result["blockers"]
    assert "legacy_operator_report_bound_without_contract_manifest" in legacy_result[
        "blockers"
    ]
    assert "operator_report:offline_only_required" in legacy_result["blockers"]
    assert "operator_report:missing_report_section:current_status" in legacy_result[
        "blockers"
    ]
    assert legacy_result["summary"]["operator_report_ready"] is False
    assert legacy_result["child_results"]["operator_report"]["status"] == "blocked"

    bound_report_result = evaluate_integration_bundle_contract(
        integration_bundle(operator_report_manifest=operator_report_manifest())
    )
    assert "operator_report_contract_not_ready" not in bound_report_result["blockers"]
    assert not any(
        blocker.startswith("operator_report:")
        for blocker in bound_report_result["blockers"]
    )
    assert bound_report_result["summary"]["operator_report_ready"] is True
    assert (
        bound_report_result["child_results"]["operator_report"][
            "ready_for_operator_report_use"
        ]
        is True
    )
    assert bound_report_result["authority"]["live_authority_granted"] is False

    assert "Integration bundle manifests must consume the operator-report contract" in (
        completion
    )
    assert "M025-CP01" in plan
    assert "M025-CP02" in plan
    assert "loose `operator_report_bound` integration checks" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "always returns `live_authority_granted: false`" in adr
    assert "M025 operator report binding is now covered" in current
    assert "M025 operator report binding rail is now automated" in report


def test_m026_autonomy_readiness_consumes_integration_bundle_status():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M026_PLAN.read_text(encoding="utf-8")
    adr = M026_ADR.read_text(encoding="utf-8")
    snapshot = build_autonomy_promotion_readiness_snapshot()
    integration = snapshot["summary"]["offline_integration_bundle"]

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m026 = markdown_table_row(roadmap, "M026")
    assert m026[1] == "Autonomy readiness integration bundle overlay"
    assert m026[2] == (
        "Autonomy promotion readiness consumes offline integration bundle "
        "contract status before planning supervised or autonomous promotion"
    )
    assert m026[3] == "M007-M025"
    assert m026[4] == "high offline"
    assert m026[5] == "UNIT_VERIFIED readiness overlay; live authority remains blocked"
    assert m026[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "`backend/autonomy_promotion_readiness.py`" in memory_eval[2]
    assert "tests/test_autonomy_promotion_readiness.py" in memory_eval[3]
    assert (
        "autonomy promotion readiness now requires offline integration bundle contract status"
        in memory_eval[4]
    )
    assert memory_eval[-1] == "none"

    assert integration["read_only"] is True
    assert integration["live_write"] is False
    assert integration["hardware_evidence"] is False
    assert integration["readiness_closure_allowed"] is False
    assert integration["live_authority_granted"] is False
    assert integration["ready_for_integration_promotion"] is False
    assert "offline_only_required" in integration["blockers"]
    assert "operator_report_contract_not_ready" in integration["blockers"]
    assert "offline_integration_bundle_contract_not_ready" in (
        snapshot["summary"]["global_blockers"]
    )
    assert snapshot["summary"]["autonomous_ready_target_ids"] == []
    assert snapshot["safety"]["full_autonomous_ready"] is False
    assert "promotion planning only" in integration["claim"]

    assert "Autonomy promotion readiness snapshots must consume the offline" in (
        completion
    )
    assert "M026-CP01" in plan
    assert "M026-CP02" in plan
    assert "offline integration bundle evidence can inform promotion planning only" in (
        plan
    )
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "always returns `live_authority_granted: false`" in adr
    assert "M026 autonomy readiness integration-bundle overlay is now covered" in (
        current
    )
    assert "M026 autonomy readiness integration-bundle overlay rail is now automated" in (
        report
    )


def test_m027_operator_adapter_surfaces_integration_bundle_readiness_summary():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M027_PLAN.read_text(encoding="utf-8")
    adr = M027_ADR.read_text(encoding="utf-8")
    adapter = FRONTEND_OPERATOR_ADAPTER.read_text(encoding="utf-8")
    adapter_test = FRONTEND_OPERATOR_ADAPTER_TEST.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m027 = markdown_table_row(roadmap, "M027")
    assert m027[1] == "Operator adapter integration bundle readiness summary"
    assert m027[2] == (
        "Operator-facing module status rows surface offline integration bundle "
        "readiness, blockers and authority-denial status from autonomy promotion readiness"
    )
    assert m027[3] == "M026"
    assert m027[4] == "high offline frontend"
    assert m027[5] == "UNIT_VERIFIED read-only operator surface; live authority remains blocked"
    assert m027[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "`frontend/src/services/operatorAdapters.ts`" in memory_eval[2]
    assert "npm --prefix frontend test -- src/services/__tests__/operatorAdaptersContract.test.mjs" in (
        memory_eval[3]
    )
    assert "operator-facing adapter summary now surfaces offline integration bundle readiness" in (
        memory_eval[4]
    )
    assert memory_eval[-1] == "none"

    assert "summary.offline_integration_bundle" in plan
    assert "M027-CP01" in plan
    assert "M027-CP02" in plan
    assert "cannot grant live authority" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "It keeps `canWriteLive: false` and `liveWriteKinds: []`." in adr

    assert "offlineIntegrationBundle" in adapter
    assert "integration_bundle_ready:" in adapter
    assert "integration_bundle_status:" in adapter
    assert "integration_bundle_source_authority_denied:" in adapter
    assert "integration_bundle_blocker:" in adapter
    assert "integration_bundle_missing_check:" in adapter
    assert "integration_bundle_child_status:" in adapter
    assert "offline_integration_bundle_contract_not_ready" in adapter_test
    assert "integration_bundle_source_authority_denied:true" in adapter_test
    assert "Operator-facing module status adapters must surface offline integration bundle" in (
        completion
    )
    assert "M027 operator adapter integration bundle readiness summary is now covered" in (
        current
    )
    assert "M027 operator adapter integration bundle readiness summary rail is now automated" in (
        report
    )


def test_m028_main_pipeline_preview_surfaces_integration_bundle_readiness_summary():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M028_PLAN.read_text(encoding="utf-8")
    adr = M028_ADR.read_text(encoding="utf-8")
    preview = PROCESSING_PIPELINE_PREVIEW.read_text(encoding="utf-8")
    preview_test = PROCESSING_PIPELINE_PREVIEW_INTEGRATION_BUNDLE_TEST.read_text(
        encoding="utf-8"
    )

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m028 = markdown_table_row(roadmap, "M028")
    assert m028[1] == "Main pipeline preview integration bundle readiness summary"
    assert m028[2] == (
        "Main pipeline preview summary surfaces offline integration bundle "
        "readiness, blockers and authority-denial status from autonomy promotion readiness"
    )
    assert m028[3] == "M026-M027"
    assert m028[4] == "high offline preview"
    assert m028[5] == "UNIT_VERIFIED read-only preview summary; live authority remains blocked"
    assert m028[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "`backend/processing_pipeline_preview.py`" in memory_eval[2]
    assert (
        "tests/test_processing_pipeline_preview_integration_bundle_summary.py"
        in memory_eval[3]
    )
    assert "main pipeline preview summary now surfaces offline integration bundle readiness" in (
        memory_eval[4]
    )
    assert memory_eval[-1] == "none"

    assert "main_pipeline_preview.summary" in plan
    assert "M028-CP01" in plan
    assert "M028-CP02" in plan
    assert "No queue mutation" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "They do not queue work" in adr

    for field in (
        "autonomy_promotion_offline_integration_bundle_ready",
        "autonomy_promotion_offline_integration_bundle_status",
        "autonomy_promotion_offline_integration_bundle_blockers",
        "autonomy_promotion_offline_integration_bundle_missing_checks",
        "autonomy_promotion_offline_integration_bundle_source_authority_denied",
        "autonomy_promotion_offline_integration_bundle_live_authority_granted",
        "autonomy_promotion_offline_integration_bundle_readiness_closure_allowed",
        "autonomy_promotion_offline_integration_bundle_live_write",
        "autonomy_promotion_offline_integration_bundle_hardware_evidence",
    ):
        assert field in preview
        assert field in preview_test
    assert "payload[\"summary\"][\"queue_mutation\"] is False" in preview_test
    assert "Main pipeline preview summaries must surface offline integration bundle" in (
        completion
    )
    assert "M028 main pipeline preview integration bundle readiness summary is now covered" in (
        current
    )
    assert "M028 main pipeline preview integration bundle readiness summary rail is now automated" in (
        report
    )


def test_m029_director_runtime_surfaces_integration_bundle_readiness_summary():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M029_PLAN.read_text(encoding="utf-8")
    adr = M029_ADR.read_text(encoding="utf-8")
    adapter = FRONTEND_OPERATOR_ADAPTER.read_text(encoding="utf-8")
    adapter_test = FRONTEND_OPERATOR_ADAPTER_TEST.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m029 = markdown_table_row(roadmap, "M029")
    assert m029[1] == "Director runtime integration bundle readiness summary"
    assert m029[2] == (
        "Director runtime surfaces main pipeline preview integration bundle "
        "readiness, blockers and authority-denial status for the runtime guard"
    )
    assert m029[3] == "M027-M028"
    assert m029[4] == "high offline frontend"
    assert m029[5] == "UNIT_VERIFIED read-only Director surface; live authority remains blocked"
    assert m029[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "`frontend/src/services/operatorAdapters.ts`" in memory_eval[2]
    assert "npm --prefix frontend test -- src/services/__tests__/operatorAdaptersContract.test.mjs" in (
        memory_eval[3]
    )
    assert "Director runtime guard now surfaces offline integration bundle readiness" in (
        memory_eval[4]
    )
    assert memory_eval[-1] == "none"

    assert "DirectorRuntime" in plan
    assert "M029-CP01" in plan
    assert "M029-CP02" in plan
    assert "No queue mutation" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "does not add a live apply path" in adr

    for field in (
        "autonomy_promotion_offline_integration_bundle_ready",
        "autonomy_promotion_offline_integration_bundle_status",
        "autonomy_promotion_offline_integration_bundle_blockers",
        "autonomy_promotion_offline_integration_bundle_missing_checks",
        "autonomy_promotion_offline_integration_bundle_source_authority_denied",
        "autonomy_promotion_offline_integration_bundle_live_authority_granted",
        "autonomy_promotion_offline_integration_bundle_readiness_closure_allowed",
    ):
        assert field in adapter
        assert field in adapter_test
    assert "Integration bundle" in adapter
    assert "integration_bundle_live_authority_granted:false" in adapter_test
    assert "Director runtime guards must surface offline integration bundle" in (
        completion
    )
    assert "M029 Director runtime integration bundle readiness summary is now covered" in (
        current
    )
    assert "M029 Director runtime integration bundle readiness summary rail is now automated" in (
        report
    )


def test_m030_director_events_render_integration_bundle_blocker_summary():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M030_PLAN.read_text(encoding="utf-8")
    adr = M030_ADR.read_text(encoding="utf-8")
    director_model = FRONTEND_DIRECTOR_PANEL_MODEL.read_text(encoding="utf-8")
    adapter_test = FRONTEND_OPERATOR_ADAPTER_TEST.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m030 = markdown_table_row(roadmap, "M030")
    assert m030[1] == "Director event integration bundle blocker summary"
    assert m030[2] == (
        "Director event rows render integration bundle blocker tags as "
        "operator-readable read-only events"
    )
    assert m030[3] == "M029"
    assert m030[4] == "high offline frontend"
    assert m030[5] == "UNIT_VERIFIED read-only Director event surface; live authority remains blocked"
    assert m030[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "`frontend/src/components/dashboard/directorPanelModel.ts`" in memory_eval[2]
    assert "npm --prefix frontend test -- src/services/__tests__/operatorAdaptersContract.test.mjs" in (
        memory_eval[3]
    )
    assert "Director event rows now render offline integration bundle blockers" in (
        memory_eval[4]
    )
    assert memory_eval[-1] == "none"

    assert "directorEventRows" in plan
    assert "M030-CP01" in plan
    assert "M030-CP02" in plan
    assert "No queue mutation" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "This does not queue proposals" in adr

    assert "integrationBundleBlockedReasonMessage" in director_model
    assert "Integration bundle" in director_model
    assert "integration_bundle_missing_check:" in director_model
    assert "integration_bundle_live_authority_granted:false" in director_model
    assert "directorEventRows renders integration bundle blockers as operator-readable events" in (
        adapter_test
    )
    assert "Integration bundle blocked; 1 blocker(s); missing: operator_report_bound" in (
        adapter_test
    )
    assert "Director event rows must render offline integration bundle blocker" in (
        completion
    )
    assert "M030 Director event integration bundle blocker summary is now covered" in (
        current
    )
    assert "M030 Director event integration bundle blocker summary rail is now automated" in (
        report
    )


def test_m031_director_guard_banner_renders_integration_bundle_blocker_summary():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M031_PLAN.read_text(encoding="utf-8")
    adr = M031_ADR.read_text(encoding="utf-8")
    director_model = FRONTEND_DIRECTOR_PANEL_MODEL.read_text(encoding="utf-8")
    right_panel = FRONTEND_RIGHT_DIRECTOR_PANEL.read_text(encoding="utf-8")
    adapter_test = FRONTEND_OPERATOR_ADAPTER_TEST.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m031 = markdown_table_row(roadmap, "M031")
    assert m031[1] == "Director guard integration bundle blocker banner"
    assert m031[2] == (
        "Runtime Guard blocker banner renders integration bundle blocker tags "
        "as operator-readable read-only messages"
    )
    assert m031[3] == "M030"
    assert m031[4] == "high offline frontend"
    assert m031[5] == "UNIT_VERIFIED read-only Director guard banner; live authority remains blocked"
    assert m031[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "`frontend/src/components/dashboard/directorPanelModel.ts`" in memory_eval[2]
    assert "Director Runtime Guard blocker banners now render offline integration bundle blockers" in (
        memory_eval[4]
    )
    assert memory_eval[-1] == "none"

    assert "M031-CP01" in plan
    assert "M031-CP02" in plan
    assert "No queue mutation" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "This does not queue proposals" in adr

    assert "directorBlockedReasonMessages" in director_model
    assert "directorBlockedReasonMessages(runtime.blockedReasons)" in right_panel
    assert "runtime.blockedReasons.slice(0, 2).join(' · ')" not in right_panel
    assert "director runtime guard banner uses readable integration bundle blocker messages" in (
        adapter_test
    )
    assert "Director Runtime Guard banners must render offline integration bundle" in (
        completion
    )
    assert "M031 Director guard integration bundle blocker banner is now covered" in (
        current
    )
    assert "M031 Director guard integration bundle blocker banner rail is now automated" in (
        report
    )


def test_m032_director_goal_surfaces_integration_bundle_blocker_summary():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M032_PLAN.read_text(encoding="utf-8")
    adr = M032_ADR.read_text(encoding="utf-8")
    adapter = FRONTEND_OPERATOR_ADAPTER.read_text(encoding="utf-8")
    adapter_test = FRONTEND_OPERATOR_ADAPTER_TEST.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m032 = markdown_table_row(roadmap, "M032")
    assert m032[1] == "Director goal integration bundle blocker summary"
    assert m032[2] == (
        "Director current goal names the offline integration bundle blocker "
        "before generic promotion-planning fallback"
    )
    assert m032[3] == "M031"
    assert m032[4] == "high offline frontend"
    assert m032[5] == "UNIT_VERIFIED read-only Director goal; live authority remains blocked"
    assert m032[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "`frontend/src/services/operatorAdapters.ts`" in memory_eval[2]
    assert "Director current goal now names offline integration bundle blockers" in (
        memory_eval[4]
    )
    assert memory_eval[-1] == "none"

    assert "M032-CP01" in plan
    assert "M032-CP02" in plan
    assert "No queue mutation" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "This does not queue proposals" in adr

    goal_text = (
        "Resolve the offline integration bundle blocker before promotion planning. "
        "Keep proposals review-only; no live authority, readiness closure or "
        "console apply is granted."
    )
    assert "integrationBundleGoal" in adapter
    assert goal_text in adapter
    assert "currentGoal: emergencyStopped" in adapter
    assert goal_text in adapter_test
    assert "director runtime surfaces preview integration bundle readiness without live authority" in (
        adapter_test
    )
    assert "Director current-goal text must render offline integration bundle" in (
        completion
    )
    assert "M032 Director goal integration bundle blocker summary is now covered" in (
        current
    )
    assert "M032 Director goal integration bundle blocker summary rail is now automated" in (
        report
    )


def test_m033_director_queue_badge_surfaces_integration_bundle_blocker():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M033_PLAN.read_text(encoding="utf-8")
    adr = M033_ADR.read_text(encoding="utf-8")
    adapter = FRONTEND_OPERATOR_ADAPTER.read_text(encoding="utf-8")
    adapter_test = FRONTEND_OPERATOR_ADAPTER_TEST.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m033 = markdown_table_row(roadmap, "M033")
    assert m033[1] == "Director queue integration bundle blocker badge"
    assert m033[2] == (
        "Director proposal queue badge shows blocked promotion planning when "
        "the offline integration bundle is blocked"
    )
    assert m033[3] == "M032"
    assert m033[4] == "high offline frontend"
    assert m033[5] == "UNIT_VERIFIED read-only Director queue badge; live authority remains blocked"
    assert m033[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "`frontend/src/services/operatorAdapters.ts`" in memory_eval[2]
    assert "Director proposal queue badge now shows offline integration bundle" in (
        memory_eval[4]
    )
    assert memory_eval[-1] == "none"

    assert "M033-CP01" in plan
    assert "M033-CP02" in plan
    assert "No queue mutation" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "This does not queue proposals" in adr

    assert "integrationBundleBlocksPromotionPlanning" in adapter
    assert "Promotion planning blocked" in adapter
    assert "proposalQueueTone" in adapter
    assert "Promotion planning blocked" in adapter_test
    assert "runtime.proposalQueueTone, 'red'" in adapter_test
    assert "Director proposal queue badges must render offline integration bundle" in (
        completion
    )
    assert "M033 Director queue integration bundle blocker badge is now covered" in (
        current
    )
    assert "M033 Director queue integration bundle blocker badge rail is now automated" in (
        report
    )


def test_m034_director_suggestions_show_integration_bundle_blocker_notice():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M034_PLAN.read_text(encoding="utf-8")
    adr = M034_ADR.read_text(encoding="utf-8")
    right_director = FRONTEND_RIGHT_DIRECTOR_PANEL.read_text(encoding="utf-8")
    adapter_test = FRONTEND_OPERATOR_ADAPTER_TEST.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m034 = markdown_table_row(roadmap, "M034")
    assert m034[1] == "Director suggestions integration bundle blocker notice"
    assert m034[2] == (
        "Agent Suggestions panel shows a read-only integration bundle blocker "
        "notice while suggestions remain review-only"
    )
    assert m034[3] == "M033"
    assert m034[4] == "high offline frontend"
    assert m034[5] == "UNIT_VERIFIED read-only Director suggestions notice; live authority remains blocked"
    assert m034[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "Director Agent Suggestions now show offline integration bundle blocker notices" in (
        memory_eval[4]
    )
    assert memory_eval[-1] == "none"

    assert "M034-CP01" in plan
    assert "M034-CP02" in plan
    assert "No queue mutation" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "This does not queue proposals" in adr

    assert "suggestionBlockerNotice" in right_director
    assert "Promotion planning blocked. Suggestions remain review-only" in right_director
    assert "blockedReasonMessages[0]" in right_director
    assert "suggestionBlockerNotice" in adapter_test
    assert "Promotion planning blocked\\. Suggestions remain review-only" in adapter_test
    assert "blockedReasonMessages\\[0\\]" in adapter_test
    assert "Director Agent Suggestions panels must render offline integration bundle" in (
        completion
    )
    assert "M034 Director suggestions integration bundle blocker notice is now covered" in (
        current
    )
    assert "M034 Director suggestions integration bundle blocker notice rail is now automated" in (
        report
    )


def test_m035_task_queue_shows_integration_bundle_blocker_notice():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M035_PLAN.read_text(encoding="utf-8")
    adr = M035_ADR.read_text(encoding="utf-8")
    task_queue = FRONTEND_TASK_QUEUE_PANEL.read_text(encoding="utf-8")
    task_queue_view = FRONTEND_TASK_QUEUE_VIEW.read_text(encoding="utf-8")
    adapter_test = FRONTEND_OPERATOR_ADAPTER_TEST.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m035 = markdown_table_row(roadmap, "M035")
    assert m035[1] == "Task Queue integration bundle blocker notice"
    assert m035[2] == (
        "Task Queue shows a read-only integration bundle blocker notice before "
        "proposal actions when promotion planning is blocked"
    )
    assert m035[3] == "M034"
    assert m035[4] == "high offline frontend"
    assert m035[5] == "UNIT_VERIFIED read-only Task Queue notice; live authority remains blocked"
    assert m035[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "Task Queue now shows offline integration bundle blocker notices" in (
        memory_eval[4]
    )
    assert memory_eval[-1] == "none"

    assert "M035-CP01" in plan
    assert "M035-CP02" in plan
    assert "No queue mutation" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "This does not queue proposals" in adr

    assert "taskQueueBlockerNotice" in task_queue
    assert "directorBlockedReasonMessages(runtime.blockedReasons)" in task_queue
    assert "Promotion planning blocked. Task Queue evidence remains review-only" in (
        task_queue
    )
    assert "runtime={data.directorRuntime}" in task_queue_view
    assert "taskQueueBlockerNotice" in adapter_test
    assert "Promotion planning blocked\\. Task Queue evidence remains review-only" in (
        adapter_test
    )
    assert "Task Queue panels must render offline integration bundle" in completion
    assert "M035 Task Queue integration bundle blocker notice is now covered" in (
        current
    )
    assert "M035 Task Queue integration bundle blocker notice rail is now automated" in (
        report
    )


def test_m036_supervised_apply_modal_shows_integration_bundle_blocker_notice():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M036_PLAN.read_text(encoding="utf-8")
    adr = M036_ADR.read_text(encoding="utf-8")
    task_queue = FRONTEND_TASK_QUEUE_PANEL.read_text(encoding="utf-8")
    adapter_test = FRONTEND_OPERATOR_ADAPTER_TEST.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m036 = markdown_table_row(roadmap, "M036")
    assert m036[1] == "Supervised Apply integration bundle blocker notice"
    assert m036[2] == (
        "Supervised Apply modal shows a read-only integration bundle blocker "
        "notice before supervised apply review"
    )
    assert m036[3] == "M035"
    assert m036[4] == "high offline frontend"
    assert m036[5] == "UNIT_VERIFIED read-only Supervised Apply notice; live authority remains blocked"
    assert m036[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "Supervised Apply modal now shows offline integration bundle blocker notices" in (
        memory_eval[4]
    )
    assert memory_eval[-1] == "none"

    assert "M036-CP01" in plan
    assert "M036-CP02" in plan
    assert "No queue mutation" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "This does not queue proposals" in adr

    assert "supervisedApplyBlockerNotice" in task_queue
    assert "integrationBundleBlockerNotice" in task_queue
    assert "Promotion planning blocked. Supervised apply remains blocked" in task_queue
    assert "supervisedApplyBlockerNotice" in adapter_test
    assert "integrationBundleBlockerNotice" in adapter_test
    assert "Promotion planning blocked\\. Supervised apply remains blocked" in (
        adapter_test
    )
    assert "Supervised Apply modals must render offline integration bundle" in (
        completion
    )
    assert "M036 Supervised Apply integration bundle blocker notice is now covered" in (
        current
    )
    assert "M036 Supervised Apply integration bundle blocker notice rail is now automated" in (
        report
    )


def test_m037_supervised_apply_blocker_enforcement_blocks_apply_confirm():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M037_PLAN.read_text(encoding="utf-8")
    adr = M037_ADR.read_text(encoding="utf-8")
    task_queue = FRONTEND_TASK_QUEUE_PANEL.read_text(encoding="utf-8")
    adapter_test = FRONTEND_OPERATOR_ADAPTER_TEST.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m037 = markdown_table_row(roadmap, "M037")
    assert m037[1] == "Supervised Apply integration bundle blocker enforcement"
    assert m037[2] == (
        "Supervised Apply modal and task actions block apply confirm when "
        "promotion planning is blocked"
    )
    assert m037[3] == "M036"
    assert m037[4] == "high offline frontend"
    assert m037[5] == "UNIT_VERIFIED read-only Supervised Apply enforcement; live authority remains blocked"
    assert m037[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "Supervised Apply modal now blocks apply confirm when offline integration bundle blockers are visible" in (
        memory_eval[4]
    )
    assert memory_eval[-1] == "none"

    assert "M037-CP01" in plan
    assert "M037-CP02" in plan
    assert "No backend contact" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "This does not queue proposals" in adr

    assert "integrationBundleApplyBlocked" in task_queue
    assert "&& !integrationBundleApplyBlocked" in task_queue
    assert "Blocked by integration bundle" in task_queue
    assert "disabled={!canRequestBackendGate}" in task_queue
    assert "integrationBundleApplyBlocked || task.canApply === false || task.reviewOnly" in (
        task_queue
    )
    assert "canRequestBackendGate ? 'Request Supervised Apply' : 'Apply Blocked'" in (
        task_queue
    )
    assert "integrationBundleApplyBlocked" in adapter_test
    assert "disabled=\\{!canRequestBackendGate\\}" in adapter_test
    assert "Supervised Apply blockers must enforce the blocked state" in completion
    assert "M037 Supervised Apply integration bundle blocker enforcement is now covered" in (
        current
    )
    assert "M037 Supervised Apply integration bundle blocker enforcement rail is now automated" in (
        report
    )


def test_m038_supervised_apply_confirm_callback_is_guarded_by_blocker():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M038_PLAN.read_text(encoding="utf-8")
    adr = M038_ADR.read_text(encoding="utf-8")
    task_queue = FRONTEND_TASK_QUEUE_PANEL.read_text(encoding="utf-8")
    adapter_test = FRONTEND_OPERATOR_ADAPTER_TEST.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m038 = markdown_table_row(roadmap, "M038")
    assert m038[1] == "Supervised Apply confirm callback blocker guard"
    assert m038[2] == (
        "Supervised Apply confirm callback returns before dispatch while "
        "promotion planning is blocked"
    )
    assert m038[3] == "M037"
    assert m038[4] == "high offline frontend"
    assert m038[5] == "UNIT_VERIFIED guarded confirm callback; live authority remains blocked"
    assert m038[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "Supervised Apply confirm callback now returns before dispatch when offline integration bundle blockers are visible" in (
        memory_eval[4]
    )
    assert memory_eval[-1] == "none"

    assert "M038-CP01" in plan
    assert "M038-CP02" in plan
    assert "No backend contact" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "This does not queue proposals" in adr

    assert "guardedConfirmSupervisedApply" in task_queue
    assert "if (integrationBundleApplyBlocked) return" in task_queue
    assert "actions.confirmSupervisedApply(taskId)" in task_queue
    assert "onConfirm={() => guardedConfirmSupervisedApply(modalTask.id)}" in (
        task_queue
    )
    assert "guardedConfirmSupervisedApply" in adapter_test
    assert "if \\(integrationBundleApplyBlocked\\) return" in adapter_test
    assert "Supervised Apply confirm callbacks must fail closed" in completion
    assert "M038 Supervised Apply confirm callback blocker guard is now covered" in (
        current
    )
    assert "M038 Supervised Apply confirm callback blocker guard rail is now automated" in (
        report
    )


def test_m039_task_apply_request_helper_is_guarded_by_blocker():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M039_PLAN.read_text(encoding="utf-8")
    adr = M039_ADR.read_text(encoding="utf-8")
    task_queue = FRONTEND_TASK_QUEUE_PANEL.read_text(encoding="utf-8")
    adapter_test = FRONTEND_OPERATOR_ADAPTER_TEST.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m039 = markdown_table_row(roadmap, "M039")
    assert m039[1] == "Task apply request helper blocker guard"
    assert m039[2] == (
        "Task apply request helper returns before opening Supervised Apply "
        "while promotion planning is blocked"
    )
    assert m039[3] == "M038"
    assert m039[4] == "high offline frontend"
    assert m039[5] == "UNIT_VERIFIED guarded task apply request; live authority remains blocked"
    assert m039[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "Task apply request helper now returns before opening Supervised Apply when offline integration bundle blockers are visible" in (
        memory_eval[4]
    )
    assert memory_eval[-1] == "none"

    assert "M039-CP01" in plan
    assert "M039-CP02" in plan
    assert "No backend contact" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "This does not open Supervised Apply" in adr

    assert "integrationBundleApplyBlocked = false" in task_queue
    assert "if (integrationBundleApplyBlocked) {" in task_queue
    assert "action: 'Apply blocked'" in task_queue
    assert "offline integration bundle blockers are visible" in task_queue
    assert "requestTaskApply(task, ui, actions, integrationBundleApplyBlocked)" in (
        task_queue
    )
    assert "actions.openSupervisedProposal(task.id)" in task_queue
    assert "integrationBundleApplyBlocked = false" in adapter_test
    assert "Task apply request helpers must fail closed" in completion
    assert "M039 Task apply request helper blocker guard is now covered" in (
        current
    )
    assert "M039 Task apply request helper blocker guard rail is now automated" in (
        report
    )


def test_m040_task_details_apply_entrypoint_is_guarded_by_blocker():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M040_PLAN.read_text(encoding="utf-8")
    adr = M040_ADR.read_text(encoding="utf-8")
    task_queue = FRONTEND_TASK_QUEUE_PANEL.read_text(encoding="utf-8")
    task_queue_view = FRONTEND_TASK_QUEUE_VIEW.read_text(encoding="utf-8")
    adapter_test = FRONTEND_OPERATOR_ADAPTER_TEST.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m040 = markdown_table_row(roadmap, "M040")
    assert m040[1] == "Task Details apply entrypoint blocker guard"
    assert m040[2] == (
        "Task Details apply button stays blocked before opening Supervised "
        "Apply while promotion planning is blocked"
    )
    assert m040[3] == "M039"
    assert m040[4] == "high offline frontend"
    assert m040[5] == "UNIT_VERIFIED guarded task details apply entrypoint; live authority remains blocked"
    assert m040[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "Task Details apply entrypoint now stays blocked before opening Supervised Apply when offline integration bundle blockers are visible" in (
        memory_eval[4]
    )
    assert memory_eval[-1] == "none"

    assert "M040-CP01" in plan
    assert "M040-CP02" in plan
    assert "No backend contact" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "This does not open Supervised Apply" in adr

    assert "integrationBundleApplyBlocked?: boolean" in task_queue
    assert "const applyBlocked = integrationBundleApplyBlocked || task.canApply === false || task.reviewOnly" in (
        task_queue
    )
    assert "requestTaskApply(task, ui, actions, integrationBundleApplyBlocked)" in (
        task_queue
    )
    assert "directorBlockedReasonMessages(data.directorRuntime.blockedReasons).length > 0" in (
        task_queue_view
    )
    assert "integrationBundleApplyBlocked={integrationBundleApplyBlocked}" in (
        task_queue_view
    )
    assert "task details apply entrypoint is guarded" in adapter_test
    assert "Task Details apply entrypoints must fail closed" in completion
    assert "M040 Task Details apply entrypoint blocker guard is now covered" in (
        current
    )
    assert "M040 Task Details apply entrypoint blocker guard rail is now automated" in (
        report
    )


def test_m041_task_queue_metrics_reflect_integration_bundle_blocker():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M041_PLAN.read_text(encoding="utf-8")
    adr = M041_ADR.read_text(encoding="utf-8")
    task_queue = FRONTEND_TASK_QUEUE_PANEL.read_text(encoding="utf-8")
    adapter_test = FRONTEND_OPERATOR_ADAPTER_TEST.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m041 = markdown_table_row(roadmap, "M041")
    assert m041[1] == "Task Queue metrics blocker guard"
    assert m041[2] == (
        "Task Queue metrics show blocked apply capacity while promotion "
        "planning is blocked"
    )
    assert m041[3] == "M040"
    assert m041[4] == "high offline frontend"
    assert m041[5] == "UNIT_VERIFIED blocked Task Queue metrics; live authority remains blocked"
    assert m041[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "Task Queue metrics now show blocked apply capacity when offline integration bundle blockers are visible" in (
        memory_eval[4]
    )
    assert memory_eval[-1] == "none"

    assert "M041-CP01" in plan
    assert "M041-CP02" in plan
    assert "No backend contact" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "This does not queue proposals" in adr

    assert "function taskQueueMetrics(" in task_queue
    assert "integrationBundleApplyBlocked = false" in task_queue
    assert "const applyableCount = integrationBundleApplyBlocked" in task_queue
    assert "? 0" in task_queue
    assert "label: integrationBundleApplyBlocked ? 'Apply Blocked' : 'Applyable'" in (
        task_queue
    )
    assert "sub: integrationBundleApplyBlocked ? 'integration bundle' : 'fader / gain only'" in (
        task_queue
    )
    assert "tone: integrationBundleApplyBlocked ? 'red' : 'green'" in task_queue
    assert "taskQueueMetrics(taskRows, integrationBundleApplyBlocked)" in task_queue
    assert "task queue metrics show apply blocked" in adapter_test
    assert "Task Queue apply metrics must fail closed" in completion
    assert "M041 Task Queue metrics blocker guard is now covered" in current
    assert "M041 Task Queue metrics blocker guard rail is now automated" in (
        report
    )


def test_m042_task_queue_review_filter_reflects_integration_bundle_blocker():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M042_PLAN.read_text(encoding="utf-8")
    adr = M042_ADR.read_text(encoding="utf-8")
    task_queue = FRONTEND_TASK_QUEUE_PANEL.read_text(encoding="utf-8")
    adapter_test = FRONTEND_OPERATOR_ADAPTER_TEST.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m042 = markdown_table_row(roadmap, "M042")
    assert m042[1] == "Task Queue review filter blocker guard"
    assert m042[2] == (
        "Task Queue Review only filter includes all rows while promotion "
        "planning is blocked"
    )
    assert m042[3] == "M041"
    assert m042[4] == "high offline frontend"
    assert m042[5] == "UNIT_VERIFIED blocked Task Queue review filter; live authority remains blocked"
    assert m042[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "Task Queue Review only filter now includes all rows when offline integration bundle blockers are visible" in (
        memory_eval[4]
    )
    assert memory_eval[-1] == "none"

    assert "M042-CP01" in plan
    assert "M042-CP02" in plan
    assert "No backend contact" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "This does not hide rows" in adr

    assert "label: 'Review only'" in task_queue
    assert "predicate: task => integrationBundleApplyBlocked || Boolean(task.reviewOnly || task.canApply === false)" in (
        task_queue
    )
    assert "integrationBundleApplyBlocked || task.canApply === false || task.reviewOnly" in (
        task_queue
    )
    assert "task queue review-only filter includes all rows" in adapter_test
    assert "Task Queue review filters must fail closed" in completion
    assert "M042 Task Queue review filter blocker guard is now covered" in current
    assert "M042 Task Queue review filter blocker guard rail is now automated" in (
        report
    )


def test_m043_task_queue_row_badge_reflects_integration_bundle_blocker():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M043_PLAN.read_text(encoding="utf-8")
    adr = M043_ADR.read_text(encoding="utf-8")
    task_queue = FRONTEND_TASK_QUEUE_PANEL.read_text(encoding="utf-8")
    adapter_test = FRONTEND_OPERATOR_ADAPTER_TEST.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m043 = markdown_table_row(roadmap, "M043")
    assert m043[1] == "Task Queue row badge blocker guard"
    assert m043[2] == (
        "Task Queue rows show blocked badges while promotion planning is "
        "blocked"
    )
    assert m043[3] == "M042"
    assert m043[4] == "high offline frontend"
    assert m043[5] == "UNIT_VERIFIED blocked Task Queue row badges; live authority remains blocked"
    assert m043[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "Task Queue rows now show blocked badges when offline integration bundle blockers are visible" in (
        memory_eval[4]
    )
    assert memory_eval[-1] == "none"

    assert "M043-CP01" in plan
    assert "M043-CP02" in plan
    assert "No backend contact" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "This does not change row data" in adr

    assert "(integrationBundleApplyBlocked || task.reviewOnly)" in task_queue
    assert "integrationBundleApplyBlocked ? 'Blocked' : 'Review'" in task_queue
    assert "predicate: task => integrationBundleApplyBlocked || Boolean(task.reviewOnly || task.canApply === false)" in (
        task_queue
    )
    assert "integrationBundleApplyBlocked || task.canApply === false || task.reviewOnly" in (
        task_queue
    )
    assert "task queue row badge shows blocked state" in adapter_test
    assert "Task Queue row badges must fail closed" in completion
    assert "M043 Task Queue row badge blocker guard is now covered" in current
    assert "M043 Task Queue row badge blocker guard rail is now automated" in (
        report
    )


def test_m044_task_details_badge_reflects_integration_bundle_blocker():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M044_PLAN.read_text(encoding="utf-8")
    adr = M044_ADR.read_text(encoding="utf-8")
    task_queue = FRONTEND_TASK_QUEUE_PANEL.read_text(encoding="utf-8")
    adapter_test = FRONTEND_OPERATOR_ADAPTER_TEST.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m044 = markdown_table_row(roadmap, "M044")
    assert m044[1] == "Task Details badge blocker guard"
    assert m044[2] == (
        "Task Details status badges show blocked state while promotion "
        "planning is blocked"
    )
    assert m044[3] == "M043"
    assert m044[4] == "high offline frontend"
    assert m044[5] == "UNIT_VERIFIED blocked Task Details badges; live authority remains blocked"
    assert m044[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "Task Details now shows blocked badges when offline integration bundle blockers are visible" in (
        memory_eval[4]
    )
    assert memory_eval[-1] == "none"

    assert "M044-CP01" in plan
    assert "M044-CP02" in plan
    assert "No backend contact" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "This does not change task data" in adr

    assert "const applyBlocked = integrationBundleApplyBlocked || task.canApply === false || task.reviewOnly" in (
        task_queue
    )
    assert "(integrationBundleApplyBlocked || task.reviewOnly)" in task_queue
    assert "integrationBundleApplyBlocked ? 'Blocked' : 'Review only'" in task_queue
    assert "requestTaskApply(task, ui, actions, integrationBundleApplyBlocked)" in (
        task_queue
    )
    assert "task details badge shows blocked state" in adapter_test
    assert "Task Details status badges must fail closed" in completion
    assert "M044 Task Details badge blocker guard is now covered" in current
    assert "M044 Task Details badge blocker guard rail is now automated" in (
        report
    )


def test_m045_task_queue_review_only_metric_reflects_integration_bundle_blocker():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M045_PLAN.read_text(encoding="utf-8")
    adr = M045_ADR.read_text(encoding="utf-8")
    task_queue = FRONTEND_TASK_QUEUE_PANEL.read_text(encoding="utf-8")
    adapter_test = FRONTEND_OPERATOR_ADAPTER_TEST.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m045 = markdown_table_row(roadmap, "M045")
    assert m045[1] == "Task Queue review-only metric blocker guard"
    assert m045[2] == (
        "Task Queue Review Only metric counts all rows while promotion "
        "planning is blocked"
    )
    assert m045[3] == "M044"
    assert m045[4] == "high offline frontend"
    assert m045[5] == "UNIT_VERIFIED blocked Task Queue review-only metric; live authority remains blocked"
    assert m045[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "Task Queue Review Only metric now counts all rows when offline integration bundle blockers are visible" in (
        memory_eval[4]
    )
    assert memory_eval[-1] == "none"

    assert "M045-CP01" in plan
    assert "M045-CP02" in plan
    assert "No backend contact" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "This does not change row data" in adr

    assert "const reviewOnlyCount = integrationBundleApplyBlocked" in task_queue
    assert "? taskRows.length" in task_queue
    assert ": taskRows.filter(task => task.reviewOnly || task.canApply === false).length" in (
        task_queue
    )
    assert "label: 'Review Only', value: String(reviewOnlyCount)" in task_queue
    assert "taskQueueMetrics(taskRows, integrationBundleApplyBlocked)" in task_queue
    assert "task queue review-only metric counts all rows" in adapter_test
    assert "Task Queue review-only metrics must fail closed" in completion
    assert "M045 Task Queue review-only metric blocker guard is now covered" in current
    assert "M045 Task Queue review-only metric blocker guard rail is now automated" in (
        report
    )


def test_m046_task_details_blocker_list_reflects_integration_bundle_blocker():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M046_PLAN.read_text(encoding="utf-8")
    adr = M046_ADR.read_text(encoding="utf-8")
    task_queue = FRONTEND_TASK_QUEUE_PANEL.read_text(encoding="utf-8")
    adapter_test = FRONTEND_OPERATOR_ADAPTER_TEST.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m046 = markdown_table_row(roadmap, "M046")
    assert m046[1] == "Task Details blocker list blocker guard"
    assert m046[2] == (
        "Task Details Apply Blockers list shows integration bundle blockers "
        "while promotion planning is blocked"
    )
    assert m046[3] == "M045"
    assert m046[4] == "high offline frontend"
    assert m046[5] == "UNIT_VERIFIED blocked Task Details blocker list; live authority remains blocked"
    assert m046[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "Task Details Apply Blockers list now includes offline integration bundle blockers when visible" in (
        memory_eval[4]
    )
    assert memory_eval[-1] == "none"

    assert "M046-CP01" in plan
    assert "M046-CP02" in plan
    assert "No backend contact" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "This does not change task data" in adr

    assert "const detailApplyBlockers = integrationBundleApplyBlocked" in task_queue
    assert "offline_integration_bundle_blockers_visible" in task_queue
    assert "detailApplyBlockers.length > 0" in task_queue
    assert "detailApplyBlockers.slice(0, 4).map(blocker =>" in task_queue
    assert "task details blocker list includes integration bundle blocker" in adapter_test
    assert "Task Details blocker lists must fail closed" in completion
    assert "M046 Task Details blocker list blocker guard is now covered" in current
    assert "M046 Task Details blocker list blocker guard rail is now automated" in (
        report
    )


def test_m047_apply_plan_sequence_reflects_integration_bundle_blocker():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M047_PLAN.read_text(encoding="utf-8")
    adr = M047_ADR.read_text(encoding="utf-8")
    task_queue = FRONTEND_TASK_QUEUE_PANEL.read_text(encoding="utf-8")
    adapter_test = FRONTEND_OPERATOR_ADAPTER_TEST.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m047 = markdown_table_row(roadmap, "M047")
    assert m047[1] == "Apply Plan sequence blocker guard"
    assert m047[2] == (
        "Apply Plan and modal Gate Sequence show blocked integration bundle "
        "steps while promotion planning is blocked"
    )
    assert m047[3] == "M046"
    assert m047[4] == "high offline frontend"
    assert m047[5] == "UNIT_VERIFIED blocked Apply Plan sequence; live authority remains blocked"
    assert m047[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "Apply Plan sequences now show blocked integration bundle steps when visible" in (
        memory_eval[4]
    )
    assert memory_eval[-1] == "none"

    assert "M047-CP01" in plan
    assert "M047-CP02" in plan
    assert "No backend contact" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "This does not mutate apply plan input" in adr

    assert "function applyPlanForIntegrationBundleBlocker(" in task_queue
    assert "if (!integrationBundleApplyBlocked) return plan" in task_queue
    assert "value: 'Bundle blocked', tone: 'red'" in task_queue
    assert "value: 'No contact', tone: 'red'" in task_queue
    assert "const guardedApplyPlan = applyPlanForIntegrationBundleBlocker(applyPlan, integrationBundleApplyBlocked)" in (
        task_queue
    )
    assert "const modalApplyPlan = applyPlanForIntegrationBundleBlocker(" in task_queue
    assert "<ApplyPlanPanel plan={guardedApplyPlan} />" in task_queue
    assert "apply plan sequence is guarded" in adapter_test
    assert "Apply Plan sequences must fail closed" in completion
    assert "M047 Apply Plan sequence blocker guard is now covered" in current
    assert "M047 Apply Plan sequence blocker guard rail is now automated" in report


def test_m048_operator_decision_reflects_integration_bundle_blocker():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M048_PLAN.read_text(encoding="utf-8")
    adr = M048_ADR.read_text(encoding="utf-8")
    task_queue = FRONTEND_TASK_QUEUE_PANEL.read_text(encoding="utf-8")
    adapter_test = FRONTEND_OPERATOR_ADAPTER_TEST.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m048 = markdown_table_row(roadmap, "M048")
    assert m048[1] == "Operator Decision blocker status guard"
    assert m048[2] == (
        "Operator Decision panel shows integration bundle blocker status while "
        "promotion planning is blocked"
    )
    assert m048[3] == "M047"
    assert m048[4] == "high offline frontend"
    assert m048[5] == "UNIT_VERIFIED blocked Operator Decision status; live authority remains blocked"
    assert m048[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "Operator Decision panel now shows integration bundle blocker status when visible" in (
        memory_eval[4]
    )
    assert memory_eval[-1] == "none"

    assert "M048-CP01" in plan
    assert "M048-CP02" in plan
    assert "No backend contact" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "This does not mutate backend status input" in adr

    assert "function OperatorDecisionPanel({" in task_queue
    assert "integrationBundleApplyBlocked?: boolean" in task_queue
    assert "const writePathStatus = integrationBundleApplyBlocked" in task_queue
    assert "value: 'Integration bundle blocked', tone: 'red' as Tone" in task_queue
    assert "const backendStatus = integrationBundleApplyBlocked" in task_queue
    assert "value: 'No contact', tone: 'red' as Tone" in task_queue
    assert "value: writePathStatus.value, tone: writePathStatus.tone" in task_queue
    assert "value: backendStatus.value, tone: backendStatus.tone" in task_queue
    assert (
        "<OperatorDecisionPanel ui={ui} policy={policy} integrationBundleApplyBlocked={integrationBundleApplyBlocked} />"
        in task_queue
    )
    assert "operator decision status is guarded" in adapter_test
    assert "Operator Decision statuses must fail closed" in completion
    assert "M048 Operator Decision blocker status guard is now covered" in current
    assert "M048 Operator Decision blocker status guard rail is now automated" in report


def test_m049_supervised_apply_final_readiness_reflects_integration_bundle_blocker():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M049_PLAN.read_text(encoding="utf-8")
    adr = M049_ADR.read_text(encoding="utf-8")
    task_queue = FRONTEND_TASK_QUEUE_PANEL.read_text(encoding="utf-8")
    adapter_test = FRONTEND_OPERATOR_ADAPTER_TEST.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m049 = markdown_table_row(roadmap, "M049")
    assert m049[1] == "Supervised Apply readiness blocker guard"
    assert m049[2] == (
        "Supervised Apply modal Final readiness shows integration bundle "
        "blocker status while promotion planning is blocked"
    )
    assert m049[3] == "M048"
    assert m049[4] == "high offline frontend"
    assert m049[5] == "UNIT_VERIFIED blocked Supervised Apply readiness; live authority remains blocked"
    assert m049[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "Supervised Apply Final readiness now shows integration bundle blocker status when visible" in (
        memory_eval[4]
    )
    assert memory_eval[-1] == "none"

    assert "M049-CP01" in plan
    assert "M049-CP02" in plan
    assert "No backend contact" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "This does not mutate backend status input" in adr

    assert "function SupervisedApplyModal({" in task_queue
    assert "const finalReadinessStatus = integrationBundleApplyBlocked" in task_queue
    assert "value: 'Blocked by integration bundle', tone: 'red' as Tone" in task_queue
    assert "value: ui.backend.statusLabel, tone: ui.backend.statusTone" in task_queue
    assert "label: 'Final readiness', value: finalReadinessStatus.value, tone: finalReadinessStatus.tone" in (
        task_queue
    )
    assert "disabled={!canRequestBackendGate}" in task_queue
    assert "supervised apply final readiness is guarded" in adapter_test
    assert "Supervised Apply readiness statuses must fail closed" in completion
    assert "M049 Supervised Apply readiness blocker guard is now covered" in current
    assert "M049 Supervised Apply readiness blocker guard rail is now automated" in report


def test_m050_supervised_apply_backend_result_is_hidden_by_integration_bundle_blocker():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M050_PLAN.read_text(encoding="utf-8")
    adr = M050_ADR.read_text(encoding="utf-8")
    task_queue = FRONTEND_TASK_QUEUE_PANEL.read_text(encoding="utf-8")
    adapter_test = FRONTEND_OPERATOR_ADAPTER_TEST.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m050 = markdown_table_row(roadmap, "M050")
    assert m050[1] == "Supervised Apply backend result blocker guard"
    assert m050[2] == (
        "Supervised Apply modal hides matched backend results while promotion "
        "planning is blocked"
    )
    assert m050[3] == "M049"
    assert m050[4] == "high offline frontend"
    assert m050[5] == "UNIT_VERIFIED hidden Supervised Apply backend result; live authority remains blocked"
    assert m050[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "Supervised Apply matched backend results now hide when integration bundle blockers are visible" in (
        memory_eval[4]
    )
    assert memory_eval[-1] == "none"

    assert "M050-CP01" in plan
    assert "M050-CP02" in plan
    assert "No backend contact" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "This does not mutate backend result input" in adr

    assert "const visibleBackendResultMatches = resultMatches && !integrationBundleApplyBlocked" in (
        task_queue
    )
    assert ": visibleBackendResultMatches ? backendResult?.applyRequest ?? applyRequest : applyRequest" in (
        task_queue
    )
    assert ": visibleBackendResultMatches ? backendResult?.readbackStatus ?? 'Required after backend apply' : 'Required after backend apply'" in (
        task_queue
    )
    assert ": visibleBackendResultMatches ? backendResult?.rollbackStatus ?? 'Snapshot required' : 'Snapshot required'" in (
        task_queue
    )
    assert "const backendResultTone = visibleBackendResultMatches ? backendResult?.tone ?? approvalTone : approvalTone" in (
        task_queue
    )
    assert "const readbackTone = modalGateBlockerStatus" in task_queue
    assert "backendResult?.readbackTone ?? 'yellow' as Tone" in task_queue
    assert "const rollbackTone = modalGateBlockerStatus" in task_queue
    assert "backendResult?.rollbackTone ?? 'yellow' as Tone" in task_queue
    assert "{ label: 'Readback status', value: readbackLabel, tone: readbackTone }" in task_queue
    assert "{ label: 'Rollback status', value: rollbackLabel, tone: rollbackTone }" in task_queue
    assert "{visibleBackendResultMatches && (" in task_queue
    assert "supervised apply backend result is hidden" in adapter_test
    assert "Supervised Apply backend result panels must fail closed" in completion
    assert "M050 Supervised Apply backend result blocker guard is now covered" in current
    assert "M050 Supervised Apply backend result blocker guard rail is now automated" in report


def test_m051_supervised_apply_gate_rows_reflect_integration_bundle_blocker():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M051_PLAN.read_text(encoding="utf-8")
    adr = M051_ADR.read_text(encoding="utf-8")
    task_queue = FRONTEND_TASK_QUEUE_PANEL.read_text(encoding="utf-8")
    adapter_test = FRONTEND_OPERATOR_ADAPTER_TEST.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m051 = markdown_table_row(roadmap, "M051")
    assert m051[1] == "Supervised Apply gate row blocker guard"
    assert m051[2] == (
        "Supervised Apply modal Apply Gate rows show no-contact and no-live "
        "status while promotion planning is blocked"
    )
    assert m051[3] == "M050"
    assert m051[4] == "high offline frontend"
    assert m051[5] == "UNIT_VERIFIED blocked Supervised Apply gate rows; live authority remains blocked"
    assert m051[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "Supervised Apply gate rows now show no-contact and no-live status when integration bundle blockers are visible" in (
        memory_eval[4]
    )
    assert memory_eval[-1] == "none"

    assert "M051-CP01" in plan
    assert "M051-CP02" in plan
    assert "No backend contact" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "This does not mutate backend result input" in adr

    assert "const modalGateBlockerStatus = integrationBundleApplyBlocked" in task_queue
    assert "applyRequest: 'Blocked'" in task_queue
    assert "readback: 'No backend contact'" in task_queue
    assert "rollback: 'No live apply'" in task_queue
    assert "readbackTone: 'red' as Tone" in task_queue
    assert "rollbackTone: 'red' as Tone" in task_queue
    assert "const applyRequestLabel = modalGateBlockerStatus" in task_queue
    assert "? modalGateBlockerStatus.applyRequest" in task_queue
    assert "const readbackLabel = modalGateBlockerStatus" in task_queue
    assert "? modalGateBlockerStatus.readback" in task_queue
    assert "const rollbackLabel = modalGateBlockerStatus" in task_queue
    assert "? modalGateBlockerStatus.rollback" in task_queue
    assert "label: 'Readback status', value: readbackLabel, tone: readbackTone" in task_queue
    assert "label: 'Rollback status', value: rollbackLabel, tone: rollbackTone" in task_queue
    assert "supervised apply gate rows are blocked" in adapter_test
    assert "Supervised Apply gate rows must fail closed" in completion
    assert "M051 Supervised Apply gate row blocker guard is now covered" in current
    assert "M051 Supervised Apply gate row blocker guard rail is now automated" in report


def test_m052_task_details_apply_button_reflects_integration_bundle_blocker():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M052_PLAN.read_text(encoding="utf-8")
    adr = M052_ADR.read_text(encoding="utf-8")
    task_queue = FRONTEND_TASK_QUEUE_PANEL.read_text(encoding="utf-8")
    adapter_test = FRONTEND_OPERATOR_ADAPTER_TEST.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["contacts_backend"] is False
    assert state["last_validation_summary"]["authority"]["contacts_wing"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m052 = markdown_table_row(roadmap, "M052")
    assert m052[1] == "Task Details apply button blocker guard"
    assert m052[2] == (
        "Task Details apply button shows blocked state while promotion "
        "planning is blocked"
    )
    assert m052[3] == "M051"
    assert m052[4] == "high offline frontend"
    assert m052[5] == "UNIT_VERIFIED blocked Task Details apply button; live authority remains blocked"
    assert m052[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "Task Details apply button now shows blocked state when integration bundle blockers are visible" in (
        memory_eval[4]
    )
    assert memory_eval[-1] == "none"

    assert "M052-CP01" in plan
    assert "M052-CP02" in plan
    assert "No backend contact" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "This does not mutate task data" in adr

    assert "const detailApplyButtonLabel = integrationBundleApplyBlocked" in task_queue
    assert "? 'Apply Blocked'" in task_queue
    assert "const detailApplyButtonClassName = integrationBundleApplyBlocked" in task_queue
    assert "applyBlocked && !integrationBundleApplyBlocked" in task_queue
    assert "disabled={integrationBundleApplyBlocked}" in task_queue
    assert "{detailApplyButtonLabel}" in task_queue
    assert "requestTaskApply(task, ui, actions, integrationBundleApplyBlocked)" in (
        task_queue
    )
    assert "task details apply button is visibly blocked" in adapter_test
    assert "Task Details apply buttons must fail closed" in completion
    assert "M052 Task Details apply button blocker guard is now covered" in current
    assert "M052 Task Details apply button blocker guard rail is now automated" in report


def test_m053_task_queue_row_reject_action_reflects_integration_bundle_blocker():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M053_PLAN.read_text(encoding="utf-8")
    adr = M053_ADR.read_text(encoding="utf-8")
    task_queue = FRONTEND_TASK_QUEUE_PANEL.read_text(encoding="utf-8")
    adapter_test = FRONTEND_OPERATOR_ADAPTER_TEST.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["contacts_backend"] is False
    assert state["last_validation_summary"]["authority"]["contacts_wing"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m053 = markdown_table_row(roadmap, "M053")
    assert m053[1] == "Task Queue row reject blocker guard"
    assert m053[2] == (
        "Task Queue row actions hide reject while promotion planning is blocked"
    )
    assert m053[3] == "M052"
    assert m053[4] == "high offline frontend"
    assert m053[5] == "UNIT_VERIFIED blocked Task Queue row reject action; live authority remains blocked"
    assert m053[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "Task Queue row actions now hide reject when integration bundle blockers are visible" in (
        memory_eval[4]
    )
    assert memory_eval[-1] == "none"

    assert "M053-CP01" in plan
    assert "M053-CP02" in plan
    assert "No backend contact" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "This does not mutate task data" in adr

    assert "function taskQueueRowActions(" in task_queue
    assert "if (integrationBundleApplyBlocked) return ['Detail']" in task_queue
    assert "if (task.canApply === false || task.reviewOnly) return ['Detail', 'Reject']" in (
        task_queue
    )
    assert "return ['Detail', 'Apply', 'Reject']" in task_queue
    assert "actions={taskQueueRowActions(task, integrationBundleApplyBlocked)}" in (
        task_queue
    )
    assert "task queue row reject action is hidden" in adapter_test
    assert "Task Queue row reject actions must fail closed" in completion
    assert "M053 Task Queue row reject blocker guard is now covered" in current
    assert "M053 Task Queue row reject blocker guard rail is now automated" in report


def test_m054_task_details_reject_button_reflects_integration_bundle_blocker():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M054_PLAN.read_text(encoding="utf-8")
    adr = M054_ADR.read_text(encoding="utf-8")
    task_queue = FRONTEND_TASK_QUEUE_PANEL.read_text(encoding="utf-8")
    adapter_test = FRONTEND_OPERATOR_ADAPTER_TEST.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["contacts_backend"] is False
    assert state["last_validation_summary"]["authority"]["contacts_wing"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m054 = markdown_table_row(roadmap, "M054")
    assert m054[1] == "Task Details reject button blocker guard"
    assert m054[2] == (
        "Task Details reject button shows blocked state while promotion "
        "planning is blocked"
    )
    assert m054[3] == "M053"
    assert m054[4] == "high offline frontend"
    assert m054[5] == "UNIT_VERIFIED blocked Task Details reject button; live authority remains blocked"
    assert m054[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "Task Details reject button now shows blocked state when integration bundle blockers are visible" in (
        memory_eval[4]
    )
    assert memory_eval[-1] == "none"

    assert "M054-CP01" in plan
    assert "M054-CP02" in plan
    assert "No backend contact" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "This does not mutate task data" in adr

    assert "const detailRejectButtonLabel = integrationBundleApplyBlocked" in task_queue
    assert "? 'Reject Blocked'" in task_queue
    assert "const detailRejectButtonClassName = integrationBundleApplyBlocked" in task_queue
    assert "className={detailRejectButtonClassName}" in task_queue
    assert "disabled={integrationBundleApplyBlocked}" in task_queue
    assert "{detailRejectButtonLabel}" in task_queue
    assert "actions.setTaskStatus(task.id, 'Rejected')" in task_queue
    assert "task details reject button is visibly blocked" in adapter_test
    assert "Task Details reject buttons must fail closed" in completion
    assert "M054 Task Details reject button blocker guard is now covered" in current
    assert "M054 Task Details reject button blocker guard rail is now automated" in report


def test_m055_operator_decision_apply_result_reflects_integration_bundle_blocker():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M055_PLAN.read_text(encoding="utf-8")
    adr = M055_ADR.read_text(encoding="utf-8")
    task_queue = FRONTEND_TASK_QUEUE_PANEL.read_text(encoding="utf-8")
    adapter_test = FRONTEND_OPERATOR_ADAPTER_TEST.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["contacts_backend"] is False
    assert state["last_validation_summary"]["authority"]["contacts_wing"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m055 = markdown_table_row(roadmap, "M055")
    assert m055[1] == "Operator Decision apply result blocker guard"
    assert m055[2] == (
        "Operator Decision Apply result row shows blocked state while "
        "promotion planning is blocked"
    )
    assert m055[3] == "M054"
    assert m055[4] == "high offline frontend"
    assert m055[5] == "UNIT_VERIFIED blocked Operator Decision apply result; live authority remains blocked"
    assert m055[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "Operator Decision Apply result now shows integration bundle blocker status when visible" in (
        memory_eval[4]
    )
    assert memory_eval[-1] == "none"

    assert "M055-CP01" in plan
    assert "M055-CP02" in plan
    assert "No backend contact" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "This does not mutate backend status input" in adr

    assert "const applyResultStatus = integrationBundleApplyBlocked" in task_queue
    assert "value: 'Apply blocked', tone: 'red' as Tone" in task_queue
    assert "value: policy.apply, tone: toneForMode(ui.mode)" in task_queue
    assert "label: 'Apply result', value: applyResultStatus.value, tone: applyResultStatus.tone" in task_queue
    assert "value: writePathStatus.value, tone: writePathStatus.tone" in task_queue
    assert "value: backendStatus.value, tone: backendStatus.tone" in task_queue
    assert "operator decision apply result is guarded" in adapter_test
    assert "Operator Decision apply result statuses must fail closed" in completion
    assert "M055 Operator Decision apply result blocker guard is now covered" in current
    assert "M055 Operator Decision apply result blocker guard rail is now automated" in report


def test_m056_safety_monitor_policy_reflects_integration_bundle_blocker():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M056_PLAN.read_text(encoding="utf-8")
    adr = M056_ADR.read_text(encoding="utf-8")
    task_queue = FRONTEND_TASK_QUEUE_PANEL.read_text(encoding="utf-8")
    adapter_test = FRONTEND_OPERATOR_ADAPTER_TEST.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["contacts_backend"] is False
    assert state["last_validation_summary"]["authority"]["contacts_wing"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m056 = markdown_table_row(roadmap, "M056")
    assert m056[1] == "Safety Monitor policy blocker guard"
    assert m056[2] == (
        "Safety Monitor Policy row shows blocked state while promotion "
        "planning is blocked"
    )
    assert m056[3] == "M055"
    assert m056[4] == "high offline frontend"
    assert m056[5] == "UNIT_VERIFIED blocked Safety Monitor policy; live authority remains blocked"
    assert m056[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "Safety Monitor Policy now shows integration bundle blocker status when visible" in (
        memory_eval[4]
    )
    assert memory_eval[-1] == "none"

    assert "M056-CP01" in plan
    assert "M056-CP02" in plan
    assert "No backend contact" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "This does not mutate safety monitor input" in adr

    assert "function SafetyMonitorPanel({" in task_queue
    assert "const policyStatus = integrationBundleApplyBlocked" in task_queue
    assert "value: 'Integration bundle blocked', tone: 'red' as const" in task_queue
    assert "value: safety.liveWritePolicy, tone: safety.liveWritePolicy.includes('fader_gain') ? 'green' as const : 'yellow' as const" in task_queue
    assert "label: 'Policy', value: policyStatus.value, tone: policyStatus.tone" in task_queue
    assert "<SafetyMonitorPanel safety={safetyMonitor} integrationBundleApplyBlocked={integrationBundleApplyBlocked} />" in task_queue
    assert "safety monitor policy is guarded" in adapter_test
    assert "Safety Monitor policy statuses must fail closed" in completion
    assert "M056 Safety Monitor policy blocker guard is now covered" in current
    assert "M056 Safety Monitor policy blocker guard rail is now automated" in report


def test_m057_safety_monitor_goal_readiness_reflects_integration_bundle_blocker():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M057_PLAN.read_text(encoding="utf-8")
    adr = M057_ADR.read_text(encoding="utf-8")
    task_queue = FRONTEND_TASK_QUEUE_PANEL.read_text(encoding="utf-8")
    adapter_test = FRONTEND_OPERATOR_ADAPTER_TEST.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["contacts_backend"] is False
    assert state["last_validation_summary"]["authority"]["contacts_wing"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m057 = markdown_table_row(roadmap, "M057")
    assert m057[1] == "Safety Monitor goal readiness blocker guard"
    assert m057[2] == (
        "Safety Monitor Goal readiness row shows blocked state while promotion "
        "planning is blocked"
    )
    assert m057[3] == "M056"
    assert m057[4] == "high offline frontend"
    assert m057[5] == "UNIT_VERIFIED blocked Safety Monitor goal readiness; live authority remains blocked"
    assert m057[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "Safety Monitor Goal readiness now shows integration bundle blocker status when visible" in (
        memory_eval[4]
    )
    assert memory_eval[-1] == "none"

    assert "M057-CP01" in plan
    assert "M057-CP02" in plan
    assert "No backend contact" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "This does not mutate readiness input" in adr

    assert "const goalReadinessStatus = integrationBundleApplyBlocked" in task_queue
    assert "value: 'Blocked by integration bundle', tone: 'red' as const" in task_queue
    assert "value: `${safety.goalCapabilityObjectiveReadinessScore}%`" in task_queue
    assert "tone: toneForReadinessScore(safety.goalCapabilityObjectiveReadinessScore)" in task_queue
    assert "label: 'Goal readiness', value: goalReadinessStatus.value, tone: goalReadinessStatus.tone" in task_queue
    assert "<SafetyMonitorPanel safety={safetyMonitor} integrationBundleApplyBlocked={integrationBundleApplyBlocked} />" in task_queue
    assert "safety monitor goal readiness is guarded" in adapter_test
    assert "Safety Monitor goal readiness statuses must fail closed" in completion
    assert "M057 Safety Monitor goal readiness blocker guard is now covered" in current
    assert "M057 Safety Monitor goal readiness blocker guard rail is now automated" in report


def test_m058_safety_monitor_blocked_surfaces_reflects_integration_bundle_blocker():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M058_PLAN.read_text(encoding="utf-8")
    adr = M058_ADR.read_text(encoding="utf-8")
    task_queue = FRONTEND_TASK_QUEUE_PANEL.read_text(encoding="utf-8")
    adapter_test = FRONTEND_OPERATOR_ADAPTER_TEST.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["contacts_backend"] is False
    assert state["last_validation_summary"]["authority"]["contacts_wing"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m058 = markdown_table_row(roadmap, "M058")
    assert m058[1] == "Safety Monitor blocked surfaces blocker guard"
    assert m058[2] == (
        "Safety Monitor Blocked footer shows integration bundle blocker while "
        "promotion planning is blocked"
    )
    assert m058[3] == "M057"
    assert m058[4] == "high offline frontend"
    assert m058[5] == "UNIT_VERIFIED blocked Safety Monitor blocked surfaces; live authority remains blocked"
    assert m058[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "Safety Monitor Blocked footer now shows integration bundle blocker status when visible" in (
        memory_eval[4]
    )
    assert memory_eval[-1] == "none"

    assert "M058-CP01" in plan
    assert "M058-CP02" in plan
    assert "No backend contact" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "This does not mutate safety monitor input" in adr

    assert "const blockedSurfaces = integrationBundleApplyBlocked" in task_queue
    assert "Array.from(new Set(['offline_integration_bundle_blockers_visible', ...safety.blockedSurfaces]))" in task_queue
    assert ": safety.blockedSurfaces" in task_queue
    assert "const blockedSurfaceText = blockedSurfaces.slice(0, 4).join(', ') || 'none'" in task_queue
    assert "Blocked: {blockedSurfaceText}" in task_queue
    assert "Blocked: {safety.blockedSurfaces.slice(0, 4).join(', ') || 'none'}" not in task_queue
    assert "<SafetyMonitorPanel safety={safetyMonitor} integrationBundleApplyBlocked={integrationBundleApplyBlocked} />" in task_queue
    assert "safety monitor blocked surfaces include integration bundle blocker" in adapter_test
    assert "Safety Monitor blocked-surface summaries must fail closed" in completion
    assert "M058 Safety Monitor blocked surfaces blocker guard is now covered" in current
    assert "M058 Safety Monitor blocked surfaces blocker guard rail is now automated" in report


def test_m059_replay_evidence_decision_trace_status_reflects_integration_bundle_blocker():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M059_PLAN.read_text(encoding="utf-8")
    adr = M059_ADR.read_text(encoding="utf-8")
    task_queue = FRONTEND_TASK_QUEUE_PANEL.read_text(encoding="utf-8")
    adapter_test = FRONTEND_OPERATOR_ADAPTER_TEST.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["contacts_backend"] is False
    assert state["last_validation_summary"]["authority"]["contacts_wing"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m059 = markdown_table_row(roadmap, "M059")
    assert m059[1] == "Replay Evidence decision trace status blocker guard"
    assert m059[2] == (
        "Replay Evidence Trace row shows integration bundle blocker status "
        "while promotion planning is blocked"
    )
    assert m059[3] == "M058"
    assert m059[4] == "high offline frontend"
    assert m059[5] == "UNIT_VERIFIED blocked Replay Evidence decision trace status; live authority remains blocked"
    assert m059[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "Replay Evidence decision trace status now shows integration bundle blocker status when visible" in (
        memory_eval[4]
    )
    assert memory_eval[-1] == "none"

    assert "M059-CP01" in plan
    assert "M059-CP02" in plan
    assert "No backend contact" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "This does not mutate decision trace input" in adr

    assert "function ReplayEvidencePanel({" in task_queue
    assert "integrationBundleApplyBlocked = false" in task_queue
    assert "integrationBundleApplyBlocked?: boolean" in task_queue
    assert "const traceStatusLabel = integrationBundleApplyBlocked" in task_queue
    assert "? 'bundle blocked'" in task_queue
    assert "? latestTrace.blockedReason ? 'blocked' : latestTrace.proofStatus" in task_queue
    assert "const traceStatusTone = integrationBundleApplyBlocked" in task_queue
    assert "? latestTrace.blockedReason ? 'red' as const : latestTrace.tone" in task_queue
    assert "<StatusBadge tone={traceStatusTone}>{traceStatusLabel}</StatusBadge>" in task_queue
    assert "integrationBundleApplyBlocked={integrationBundleApplyBlocked}" in task_queue
    assert "replay evidence decision trace status is guarded" in adapter_test
    assert "Replay Evidence decision trace statuses must fail closed" in completion
    assert "M059 Replay Evidence decision trace status blocker guard is now covered" in current
    assert "M059 Replay Evidence decision trace status blocker guard rail is now automated" in report


def test_m060_replay_evidence_latest_run_status_reflects_integration_bundle_blocker():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M060_PLAN.read_text(encoding="utf-8")
    adr = M060_ADR.read_text(encoding="utf-8")
    task_queue = FRONTEND_TASK_QUEUE_PANEL.read_text(encoding="utf-8")
    adapter_test = FRONTEND_OPERATOR_ADAPTER_TEST.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["contacts_backend"] is False
    assert state["last_validation_summary"]["authority"]["contacts_wing"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m060 = markdown_table_row(roadmap, "M060")
    assert m060[1] == "Replay Evidence latest run status blocker guard"
    assert m060[2] == (
        "Replay Evidence Latest run badge shows integration bundle blocker "
        "status while promotion planning is blocked"
    )
    assert m060[3] == "M059"
    assert m060[4] == "high offline frontend"
    assert m060[5] == "UNIT_VERIFIED blocked Replay Evidence latest run status; live authority remains blocked"
    assert m060[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "Replay Evidence latest run status now shows integration bundle blocker status when visible" in (
        memory_eval[4]
    )
    assert memory_eval[-1] == "none"

    assert "M060-CP01" in plan
    assert "M060-CP02" in plan
    assert "No backend contact" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "This does not mutate replay run input" in adr

    assert "function ReplayEvidencePanel({" in task_queue
    assert "const latestRunStatusLabel = integrationBundleApplyBlocked" in task_queue
    assert "? 'bundle blocked'" in task_queue
    assert ": latestRun?.status ?? 'idle'" in task_queue
    assert "const latestRunStatusTone = integrationBundleApplyBlocked" in task_queue
    assert ": toneForReplayStatus(latestRun?.status ?? 'idle')" in task_queue
    assert "<StatusBadge tone={latestRunStatusTone}>{latestRunStatusLabel}</StatusBadge>" in task_queue
    assert "toneForReplayStatus(latestRun?.status ?? 'idle')}>{latestRun?.status ?? 'idle'}" not in task_queue
    assert "integrationBundleApplyBlocked={integrationBundleApplyBlocked}" in task_queue
    assert "replay evidence latest run status is guarded" in adapter_test
    assert "Replay Evidence latest run statuses must fail closed" in completion
    assert "M060 Replay Evidence latest run status blocker guard is now covered" in current
    assert "M060 Replay Evidence latest run status blocker guard rail is now automated" in report


def test_m061_replay_evidence_policy_row_reflects_integration_bundle_blocker():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M061_PLAN.read_text(encoding="utf-8")
    adr = M061_ADR.read_text(encoding="utf-8")
    task_queue = FRONTEND_TASK_QUEUE_PANEL.read_text(encoding="utf-8")
    adapter_test = FRONTEND_OPERATOR_ADAPTER_TEST.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["contacts_backend"] is False
    assert state["last_validation_summary"]["authority"]["contacts_wing"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m061 = markdown_table_row(roadmap, "M061")
    assert m061[1] == "Replay Evidence policy row blocker guard"
    assert m061[2] == (
        "Replay Evidence Policy row shows integration bundle blocker status "
        "while promotion planning is blocked"
    )
    assert m061[3] == "M060"
    assert m061[4] == "high offline frontend"
    assert m061[5] == "UNIT_VERIFIED blocked Replay Evidence policy row; live authority remains blocked"
    assert m061[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "Replay Evidence Policy row now shows integration bundle blocker status when visible" in (
        memory_eval[4]
    )
    assert memory_eval[-1] == "none"

    assert "M061-CP01" in plan
    assert "M061-CP02" in plan
    assert "No backend contact" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "This does not mutate replay run input" in adr

    assert "function ReplayEvidencePanel({" in task_queue
    assert "const replayPolicyLabel = integrationBundleApplyBlocked" in task_queue
    assert "? 'bundle blocked'" in task_queue
    assert ": latestRun?.offlineOnly === false ? 'review' : 'offline only'" in task_queue
    assert "const replayPolicyDetail = integrationBundleApplyBlocked" in task_queue
    assert "? 'no live apply'" in task_queue
    assert ": latestRun?.liveWrite ? 'live write' : 'no live write'" in task_queue
    assert "const replayPolicyDetailClassName = integrationBundleApplyBlocked || latestRun?.liveWrite" in task_queue
    assert "{replayPolicyLabel} ·{' '}" in task_queue
    assert "<span className={replayPolicyDetailClassName}>" in task_queue
    assert "{replayPolicyDetail}" in task_queue
    assert "{latestRun?.offlineOnly === false ? 'review' : 'offline only'} ·" not in task_queue
    assert "integrationBundleApplyBlocked={integrationBundleApplyBlocked}" in task_queue
    assert "replay evidence policy row is guarded" in adapter_test
    assert "Replay Evidence policy rows must fail closed" in completion
    assert "M061 Replay Evidence policy row blocker guard is now covered" in current
    assert "M061 Replay Evidence policy row blocker guard rail is now automated" in report


def test_m062_replay_evidence_channel_status_reflects_integration_bundle_blocker():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M062_PLAN.read_text(encoding="utf-8")
    adr = M062_ADR.read_text(encoding="utf-8")
    task_queue = FRONTEND_TASK_QUEUE_PANEL.read_text(encoding="utf-8")
    adapter_test = FRONTEND_OPERATOR_ADAPTER_TEST.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["contacts_backend"] is False
    assert state["last_validation_summary"]["authority"]["contacts_wing"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m062 = markdown_table_row(roadmap, "M062")
    assert m062[1] == "Replay Evidence channel status blocker guard"
    assert m062[2] == (
        "Replay Evidence channel status badge shows integration bundle blocker "
        "status while promotion planning is blocked"
    )
    assert m062[3] == "M061"
    assert m062[4] == "high offline frontend"
    assert m062[5] == "UNIT_VERIFIED blocked Replay Evidence channel status; live authority remains blocked"
    assert m062[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "Replay Evidence channel status now shows integration bundle blocker status when visible" in (
        memory_eval[4]
    )
    assert memory_eval[-1] == "none"

    assert "M062-CP01" in plan
    assert "M062-CP02" in plan
    assert "No backend contact" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "This does not mutate channel evidence input" in adr

    assert "function ReplayEvidencePanel({" in task_queue
    assert "const channelStatusLabel = integrationBundleApplyBlocked" in task_queue
    assert "? 'bundle blocked'" in task_queue
    assert ": channel?.gainRideState ?? 'shadow'" in task_queue
    assert "const channelStatusTone = integrationBundleApplyBlocked" in task_queue
    assert ": channel?.liveApplyAllowed ? 'yellow' as const : 'blue' as const" in task_queue
    assert "<StatusBadge tone={channelStatusTone}>{channelStatusLabel}</StatusBadge>" in task_queue
    assert "channel?.liveApplyAllowed ? 'yellow' : 'blue'}>{channel?.gainRideState ?? 'shadow'}" not in task_queue
    assert "integrationBundleApplyBlocked={integrationBundleApplyBlocked}" in task_queue
    assert "replay evidence channel status is guarded" in adapter_test
    assert "Replay Evidence channel statuses must fail closed" in completion
    assert "M062 Replay Evidence channel status blocker guard is now covered" in current
    assert "M062 Replay Evidence channel status blocker guard rail is now automated" in report


def test_m063_replay_evidence_report_path_reflects_integration_bundle_blocker():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M063_PLAN.read_text(encoding="utf-8")
    adr = M063_ADR.read_text(encoding="utf-8")
    task_queue = FRONTEND_TASK_QUEUE_PANEL.read_text(encoding="utf-8")
    adapter_test = FRONTEND_OPERATOR_ADAPTER_TEST.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["contacts_backend"] is False
    assert state["last_validation_summary"]["authority"]["contacts_wing"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m063 = markdown_table_row(roadmap, "M063")
    assert m063[1] == "Replay Evidence report path blocker guard"
    assert m063[2] == (
        "Replay Evidence report path text shows integration bundle blocker "
        "status while promotion planning is blocked"
    )
    assert m063[3] == "M062"
    assert m063[4] == "high offline frontend"
    assert m063[5] == "UNIT_VERIFIED blocked Replay Evidence report path; live authority remains blocked"
    assert m063[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "Replay Evidence report path now shows integration bundle blocker status when visible" in (
        memory_eval[4]
    )
    assert memory_eval[-1] == "none"

    assert "M063-CP01" in plan
    assert "M063-CP02" in plan
    assert "No backend contact" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "This does not mutate replay report input" in adr

    assert "function ReplayEvidencePanel({" in task_queue
    assert "const replayReportLabel = integrationBundleApplyBlocked" in task_queue
    assert "? 'bundle blocked'" in task_queue
    assert ": latestRun?.reportPath || 'pending'" in task_queue
    assert "const replayReportClassName = integrationBundleApplyBlocked" in task_queue
    assert "? 'mt-0.5 truncate text-[8px] text-danger'" in task_queue
    assert ": 'mt-0.5 truncate text-[8px] text-muted'" in task_queue
    assert "<div className={replayReportClassName}>" in task_queue
    assert "Report: {replayReportLabel}" in task_queue
    assert "Report: {latestRun.reportPath || 'pending'}" not in task_queue
    assert "integrationBundleApplyBlocked={integrationBundleApplyBlocked}" in task_queue
    assert "replay evidence report path is guarded" in adapter_test
    assert "Replay Evidence report paths must fail closed" in completion
    assert "M063 Replay Evidence report path blocker guard is now covered" in current
    assert "M063 Replay Evidence report path blocker guard rail is now automated" in report


def test_m064_task_queue_status_column_reflects_integration_bundle_blocker():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M064_PLAN.read_text(encoding="utf-8")
    adr = M064_ADR.read_text(encoding="utf-8")
    task_queue = FRONTEND_TASK_QUEUE_PANEL.read_text(encoding="utf-8")
    adapter_test = FRONTEND_OPERATOR_ADAPTER_TEST.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["contacts_backend"] is False
    assert state["last_validation_summary"]["authority"]["contacts_wing"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m064 = markdown_table_row(roadmap, "M064")
    assert m064[1] == "Task Queue status column blocker guard"
    assert m064[2] == (
        "Task Queue Status column shows integration bundle blocker status "
        "while promotion planning is blocked"
    )
    assert m064[3] == "M063"
    assert m064[4] == "high offline frontend"
    assert m064[5] == "UNIT_VERIFIED blocked Task Queue status column; live authority remains blocked"
    assert m064[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "Task Queue Status column now shows integration bundle blocker status when visible" in (
        memory_eval[4]
    )
    assert memory_eval[-1] == "none"

    assert "M064-CP01" in plan
    assert "M064-CP02" in plan
    assert "No backend contact" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "This does not mutate task status input" in adr

    assert "function taskStatusDisplay(" in task_queue
    assert "const label = integrationBundleApplyBlocked ? 'Blocked' : task.status" in task_queue
    assert "const tone = integrationBundleApplyBlocked ? 'red' as const : toneFromState(task.status)" in task_queue
    assert "const status = taskStatusDisplay(task, integrationBundleApplyBlocked)" in task_queue
    assert "<StatusBadge tone={status.tone}>{status.label}</StatusBadge>" in task_queue
    assert "key: 'status'" in task_queue
    assert "task queue status column shows blocked state" in adapter_test
    assert "Task Queue status columns must fail closed" in completion
    assert "M064 Task Queue status column blocker guard is now covered" in current
    assert "M064 Task Queue status column blocker guard rail is now automated" in report


def test_m065_task_details_status_value_reflects_integration_bundle_blocker():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M065_PLAN.read_text(encoding="utf-8")
    adr = M065_ADR.read_text(encoding="utf-8")
    task_queue = FRONTEND_TASK_QUEUE_PANEL.read_text(encoding="utf-8")
    adapter_test = FRONTEND_OPERATOR_ADAPTER_TEST.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["contacts_backend"] is False
    assert state["last_validation_summary"]["authority"]["contacts_wing"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m065 = markdown_table_row(roadmap, "M065")
    assert m065[1] == "Task Details status value blocker guard"
    assert m065[2] == (
        "Task Details primary status badge shows integration bundle blocker "
        "status while promotion planning is blocked"
    )
    assert m065[3] == "M064"
    assert m065[4] == "high offline frontend"
    assert m065[5] == "UNIT_VERIFIED blocked Task Details status value; live authority remains blocked"
    assert m065[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "Task Details status value now shows integration bundle blocker status when visible" in (
        memory_eval[4]
    )
    assert memory_eval[-1] == "none"

    assert "M065-CP01" in plan
    assert "M065-CP02" in plan
    assert "No backend contact" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "This does not mutate task status input" in adr

    assert "function taskStatusDisplay(" in task_queue
    assert "const detailStatus = taskStatusDisplay(task, integrationBundleApplyBlocked)" in task_queue
    assert "<StatusBadge tone={detailStatus.tone}>{detailStatus.label}</StatusBadge>" in task_queue
    assert "const label = integrationBundleApplyBlocked ? 'Blocked' : task.status" in task_queue
    assert "const tone = integrationBundleApplyBlocked ? 'red' as const : toneFromState(task.status)" in task_queue
    assert "<StatusBadge tone={toneFromState(task.status)}>{task.status}</StatusBadge>" not in task_queue
    assert "task details status value shows blocked state" in adapter_test
    assert "Task Details status values must fail closed" in completion
    assert "M065 Task Details status value blocker guard is now covered" in current
    assert "M065 Task Details status value blocker guard rail is now automated" in report


def test_m066_task_details_proposal_risk_reflects_integration_bundle_blocker():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M066_PLAN.read_text(encoding="utf-8")
    adr = M066_ADR.read_text(encoding="utf-8")
    task_queue = FRONTEND_TASK_QUEUE_PANEL.read_text(encoding="utf-8")
    adapter_test = FRONTEND_OPERATOR_ADAPTER_TEST.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["contacts_backend"] is False
    assert state["last_validation_summary"]["authority"]["contacts_wing"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m066 = markdown_table_row(roadmap, "M066")
    assert m066[1] == "Task Details proposal risk blocker guard"
    assert m066[2] == (
        "Task Details Proposal Preview risk badge shows integration bundle "
        "blocker status while promotion planning is blocked"
    )
    assert m066[3] == "M065"
    assert m066[4] == "high offline frontend"
    assert m066[5] == "UNIT_VERIFIED blocked Task Details proposal risk; live authority remains blocked"
    assert m066[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "Task Details proposal risk now shows integration bundle blocker status when visible" in (
        memory_eval[4]
    )
    assert memory_eval[-1] == "none"

    assert "M066-CP01" in plan
    assert "M066-CP02" in plan
    assert "No backend contact" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "This does not mutate proposal data" in adr

    assert "function taskProposalRiskDisplay(" in task_queue
    assert "proposal: ProposalPreview" in task_queue
    assert "const label = integrationBundleApplyBlocked ? 'Blocked' : proposal.risk" in task_queue
    assert "const tone = integrationBundleApplyBlocked ? 'red' as const : proposal.tone" in task_queue
    assert "const detailProposalRisk = taskProposalRiskDisplay(proposal, integrationBundleApplyBlocked)" in task_queue
    assert "<StatusBadge tone={detailProposalRisk.tone}>{detailProposalRisk.label}</StatusBadge>" in task_queue
    assert "<StatusBadge tone={proposal.tone}>{proposal.risk}</StatusBadge>" not in task_queue
    assert "task details proposal risk badge shows blocked state" in adapter_test
    assert "Task Details proposal risk badges must fail closed" in completion
    assert "M066 Task Details proposal risk blocker guard is now covered" in current
    assert "M066 Task Details proposal risk blocker guard rail is now automated" in report


def test_m067_task_queue_move_delta_reflects_integration_bundle_blocker():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M067_PLAN.read_text(encoding="utf-8")
    adr = M067_ADR.read_text(encoding="utf-8")
    task_queue = FRONTEND_TASK_QUEUE_PANEL.read_text(encoding="utf-8")
    adapter_test = FRONTEND_OPERATOR_ADAPTER_TEST.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["contacts_backend"] is False
    assert state["last_validation_summary"]["authority"]["contacts_wing"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m067 = markdown_table_row(roadmap, "M067")
    assert m067[1] == "Task Queue move delta blocker guard"
    assert m067[2] == (
        "Task Queue Target / Move delta shows integration bundle blocker "
        "status while promotion planning is blocked"
    )
    assert m067[3] == "M066"
    assert m067[4] == "high offline frontend"
    assert m067[5] == "UNIT_VERIFIED blocked Task Queue move delta; live authority remains blocked"
    assert m067[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "Task Queue move delta now shows integration bundle blocker status when visible" in (
        memory_eval[4]
    )
    assert memory_eval[-1] == "none"

    assert "M067-CP01" in plan
    assert "M067-CP02" in plan
    assert "No backend contact" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "This does not mutate proposal data" in adr

    assert "function taskMoveDisplay(" in task_queue
    assert "const label = integrationBundleApplyBlocked ? 'Blocked' : task.proposal.delta" in task_queue
    assert "const className = integrationBundleApplyBlocked ? 'text-danger' : 'text-audio'" in task_queue
    assert "const move = taskMoveDisplay(task, integrationBundleApplyBlocked)" in task_queue
    assert "className={`ml-2 font-mono ${move.className}`}" in task_queue
    assert "{move.label}" in task_queue
    assert '<span className="ml-2 font-mono text-audio">{task.proposal.delta}</span>' not in task_queue
    assert "task queue move column shows blocked state" in adapter_test
    assert "Task Queue move deltas must fail closed" in completion
    assert "M067 Task Queue move delta blocker guard is now covered" in current
    assert "M067 Task Queue move delta blocker guard rail is now automated" in report


def test_m068_task_details_proposal_delta_reflects_integration_bundle_blocker():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M068_PLAN.read_text(encoding="utf-8")
    adr = M068_ADR.read_text(encoding="utf-8")
    task_queue = FRONTEND_TASK_QUEUE_PANEL.read_text(encoding="utf-8")
    adapter_test = FRONTEND_OPERATOR_ADAPTER_TEST.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["contacts_backend"] is False
    assert state["last_validation_summary"]["authority"]["contacts_wing"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m068 = markdown_table_row(roadmap, "M068")
    assert m068[1] == "Task Details proposal delta blocker guard"
    assert m068[2] == (
        "Task Details Proposal Preview delta shows integration bundle blocker "
        "status while promotion planning is blocked"
    )
    assert m068[3] == "M067"
    assert m068[4] == "high offline frontend"
    assert m068[5] == "UNIT_VERIFIED blocked Task Details proposal delta; live authority remains blocked"
    assert m068[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "Task Details proposal delta now shows integration bundle blocker status when visible" in (
        memory_eval[4]
    )
    assert memory_eval[-1] == "none"

    assert "M068-CP01" in plan
    assert "M068-CP02" in plan
    assert "No backend contact" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "This does not mutate proposal data" in adr

    assert "function taskProposalDeltaDisplay(" in task_queue
    assert "proposal: ProposalPreview" in task_queue
    assert "const label = integrationBundleApplyBlocked ? 'Blocked' : proposal.delta" in task_queue
    assert "const className = integrationBundleApplyBlocked ? 'text-danger' : 'text-audio'" in task_queue
    assert "const detailProposalDelta = taskProposalDeltaDisplay(proposal, integrationBundleApplyBlocked)" in task_queue
    assert "className={`truncate font-bold ${detailProposalDelta.className}`}" in task_queue
    assert "{detailProposalDelta.label}" in task_queue
    assert '<span className="truncate font-bold text-audio">{proposal.delta}</span>' not in task_queue
    assert "task details proposal delta shows blocked state" in adapter_test
    assert "Task Details proposal deltas must fail closed" in completion
    assert "M068 Task Details proposal delta blocker guard is now covered" in current
    assert "M068 Task Details proposal delta blocker guard rail is now automated" in report


def test_m069_supervised_apply_new_value_reflects_integration_bundle_blocker():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M069_PLAN.read_text(encoding="utf-8")
    adr = M069_ADR.read_text(encoding="utf-8")
    task_queue = FRONTEND_TASK_QUEUE_PANEL.read_text(encoding="utf-8")
    adapter_test = FRONTEND_OPERATOR_ADAPTER_TEST.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["contacts_backend"] is False
    assert state["last_validation_summary"]["authority"]["contacts_wing"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m069 = markdown_table_row(roadmap, "M069")
    assert m069[1] == "Supervised Apply new value blocker guard"
    assert m069[2] == (
        "Supervised Apply Proposal Target New value shows integration bundle "
        "blocker status while promotion planning is blocked"
    )
    assert m069[3] == "M068"
    assert m069[4] == "high offline frontend"
    assert m069[5] == "UNIT_VERIFIED blocked Supervised Apply new value; live authority remains blocked"
    assert m069[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "Supervised Apply new value now shows integration bundle blocker status when visible" in (
        memory_eval[4]
    )
    assert memory_eval[-1] == "none"

    assert "M069-CP01" in plan
    assert "M069-CP02" in plan
    assert "No backend contact" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "This does not mutate proposal data" in adr

    assert "const supervisedProposalNewValue = taskProposalDeltaDisplay(proposal, integrationBundleApplyBlocked)" in task_queue
    assert "className={`truncate ${supervisedProposalNewValue.className}`}" in task_queue
    assert "{supervisedProposalNewValue.label}" in task_queue
    assert '<span className="text-muted">New value</span><b className="truncate text-audio">{proposal.delta}</b>' not in task_queue
    assert "supervised apply new value shows blocked state" in adapter_test
    assert "Supervised Apply new values must fail closed" in completion
    assert "M069 Supervised Apply new value blocker guard is now covered" in current
    assert "M069 Supervised Apply new value blocker guard rail is now automated" in report


def test_m070_supervised_apply_rollback_reflects_integration_bundle_blocker():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M070_PLAN.read_text(encoding="utf-8")
    adr = M070_ADR.read_text(encoding="utf-8")
    task_queue = FRONTEND_TASK_QUEUE_PANEL.read_text(encoding="utf-8")
    adapter_test = FRONTEND_OPERATOR_ADAPTER_TEST.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["contacts_backend"] is False
    assert state["last_validation_summary"]["authority"]["contacts_wing"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m070 = markdown_table_row(roadmap, "M070")
    assert m070[1] == "Supervised Apply rollback blocker guard"
    assert m070[2] == (
        "Supervised Apply Proposal Target Rollback shows no-live-apply "
        "status while promotion planning is blocked"
    )
    assert m070[3] == "M069"
    assert m070[4] == "high offline frontend"
    assert m070[5] == "UNIT_VERIFIED blocked Supervised Apply rollback; live authority remains blocked"
    assert m070[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "Supervised Apply rollback now shows no-live-apply blocker status when visible" in (
        memory_eval[4]
    )
    assert memory_eval[-1] == "none"

    assert "M070-CP01" in plan
    assert "M070-CP02" in plan
    assert "No backend contact" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "This does not mutate proposal data" in adr

    assert "function taskProposalRollbackDisplay(" in task_queue
    assert "const label = integrationBundleApplyBlocked ? 'No live apply' : proposal.rollback" in task_queue
    assert "const className = integrationBundleApplyBlocked ? 'text-danger' : 'text-text'" in task_queue
    assert "const supervisedProposalRollback = taskProposalRollbackDisplay(proposal, integrationBundleApplyBlocked)" in task_queue
    assert "className={`truncate ${supervisedProposalRollback.className}`}" in task_queue
    assert "{supervisedProposalRollback.label}" in task_queue
    assert '<span className="text-muted">Rollback</span><b className="truncate text-text">{proposal.rollback}</b>' not in task_queue
    assert "supervised apply rollback shows no live apply" in adapter_test
    assert "Supervised Apply rollback values must fail closed" in completion
    assert "M070 Supervised Apply rollback blocker guard is now covered" in current
    assert "M070 Supervised Apply rollback blocker guard rail is now automated" in report


def test_m071_task_details_proposal_rollback_reflects_integration_bundle_blocker():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M071_PLAN.read_text(encoding="utf-8")
    adr = M071_ADR.read_text(encoding="utf-8")
    task_queue = FRONTEND_TASK_QUEUE_PANEL.read_text(encoding="utf-8")
    adapter_test = FRONTEND_OPERATOR_ADAPTER_TEST.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["contacts_backend"] is False
    assert state["last_validation_summary"]["authority"]["contacts_wing"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m071 = markdown_table_row(roadmap, "M071")
    assert m071[1] == "Task Details proposal rollback blocker guard"
    assert m071[2] == (
        "Task Details Proposal Preview Rollback shows no-live-apply "
        "status while promotion planning is blocked"
    )
    assert m071[3] == "M070"
    assert m071[4] == "high offline frontend"
    assert m071[5] == "UNIT_VERIFIED blocked Task Details proposal rollback; live authority remains blocked"
    assert m071[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "Task Details proposal rollback now shows no-live-apply blocker status when visible" in (
        memory_eval[4]
    )
    assert memory_eval[-1] == "none"

    assert "M071-CP01" in plan
    assert "M071-CP02" in plan
    assert "No backend contact" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "This does not mutate proposal data" in adr

    assert "function taskProposalRollbackDisplay(" in task_queue
    assert "const label = integrationBundleApplyBlocked ? 'No live apply' : proposal.rollback" in task_queue
    assert "const className = integrationBundleApplyBlocked ? 'text-danger' : 'text-text'" in task_queue
    assert "const detailProposalRollback = taskProposalRollbackDisplay(proposal, integrationBundleApplyBlocked)" in task_queue
    assert "className={`truncate ${detailProposalRollback.className}`}" in task_queue
    assert "{detailProposalRollback.label}" in task_queue
    assert '<span className="text-muted">Rollback</span><span className="truncate text-text">{proposal.rollback}</span>' not in task_queue
    assert "task details proposal rollback shows no live apply" in adapter_test
    assert "Task Details proposal rollback values must fail closed" in completion
    assert "M071 Task Details proposal rollback blocker guard is now covered" in current
    assert "M071 Task Details proposal rollback blocker guard rail is now automated" in report


def test_m072_task_details_proposal_proof_reflects_integration_bundle_blocker():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M072_PLAN.read_text(encoding="utf-8")
    adr = M072_ADR.read_text(encoding="utf-8")
    task_queue = FRONTEND_TASK_QUEUE_PANEL.read_text(encoding="utf-8")
    adapter_test = FRONTEND_OPERATOR_ADAPTER_TEST.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["contacts_backend"] is False
    assert state["last_validation_summary"]["authority"]["contacts_wing"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m072 = markdown_table_row(roadmap, "M072")
    assert m072[1] == "Task Details proposal proof blocker guard"
    assert m072[2] == (
        "Task Details Proposal Preview Proof shows no-live-proof "
        "status while promotion planning is blocked"
    )
    assert m072[3] == "M071"
    assert m072[4] == "high offline frontend"
    assert m072[5] == "UNIT_VERIFIED blocked Task Details proposal proof; live authority remains blocked"
    assert m072[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "Task Details proposal proof now shows no-live-proof blocker status when visible" in (
        memory_eval[4]
    )
    assert memory_eval[-1] == "none"

    assert "M072-CP01" in plan
    assert "M072-CP02" in plan
    assert "No backend contact" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "This does not mutate proposal data" in adr

    assert "function taskProposalProofDisplay(" in task_queue
    assert "const label = integrationBundleApplyBlocked ? 'No live proof' : proposal.proof" in task_queue
    assert "const className = integrationBundleApplyBlocked ? 'text-danger' : 'text-text'" in task_queue
    assert "const detailProposalProof = taskProposalProofDisplay(proposal, integrationBundleApplyBlocked)" in task_queue
    assert "className={`truncate ${detailProposalProof.className}`}" in task_queue
    assert "{detailProposalProof.label}" in task_queue
    assert '<span className="text-muted">Proof</span><span className="truncate text-text">{proposal.proof}</span>' not in task_queue
    assert "task details proposal proof shows no live proof" in adapter_test
    assert "Task Details proposal proof values must fail closed" in completion
    assert "M072 Task Details proposal proof blocker guard is now covered" in current
    assert "M072 Task Details proposal proof blocker guard rail is now automated" in report


def test_m073_task_details_proposal_parameter_reflects_integration_bundle_blocker():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M073_PLAN.read_text(encoding="utf-8")
    adr = M073_ADR.read_text(encoding="utf-8")
    task_queue = FRONTEND_TASK_QUEUE_PANEL.read_text(encoding="utf-8")
    adapter_test = FRONTEND_OPERATOR_ADAPTER_TEST.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["contacts_backend"] is False
    assert state["last_validation_summary"]["authority"]["contacts_wing"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m073 = markdown_table_row(roadmap, "M073")
    assert m073[1] == "Task Details proposal parameter blocker guard"
    assert m073[2] == (
        "Task Details Proposal Preview Parameter shows blocked-parameter "
        "status while promotion planning is blocked"
    )
    assert m073[3] == "M072"
    assert m073[4] == "high offline frontend"
    assert m073[5] == "UNIT_VERIFIED blocked Task Details proposal parameter; live authority remains blocked"
    assert m073[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "Task Details proposal parameter now shows blocked-parameter status when visible" in (
        memory_eval[4]
    )
    assert memory_eval[-1] == "none"

    assert "M073-CP01" in plan
    assert "M073-CP02" in plan
    assert "No backend contact" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "This does not mutate proposal data" in adr

    assert "function taskProposalParameterDisplay(" in task_queue
    assert "const label = integrationBundleApplyBlocked ? 'Blocked parameter' : proposal.parameter" in task_queue
    assert "const className = integrationBundleApplyBlocked ? 'text-danger' : 'text-text'" in task_queue
    assert "const detailProposalParameter = taskProposalParameterDisplay(proposal, integrationBundleApplyBlocked)" in task_queue
    assert "className={`truncate ${detailProposalParameter.className}`}" in task_queue
    assert "{detailProposalParameter.label}" in task_queue
    assert '<span className="text-muted">Parameter</span><span className="truncate text-text">{proposal.parameter}</span>' not in task_queue
    assert "task details proposal parameter shows blocked parameter" in adapter_test
    assert "Task Details proposal parameter values must fail closed" in completion
    assert "M073 Task Details proposal parameter blocker guard is now covered" in current
    assert "M073 Task Details proposal parameter blocker guard rail is now automated" in report


def test_m074_supervised_apply_parameter_reflects_integration_bundle_blocker():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M074_PLAN.read_text(encoding="utf-8")
    adr = M074_ADR.read_text(encoding="utf-8")
    task_queue = FRONTEND_TASK_QUEUE_PANEL.read_text(encoding="utf-8")
    adapter_test = FRONTEND_OPERATOR_ADAPTER_TEST.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["contacts_backend"] is False
    assert state["last_validation_summary"]["authority"]["contacts_wing"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m074 = markdown_table_row(roadmap, "M074")
    assert m074[1] == "Supervised Apply parameter blocker guard"
    assert m074[2] == (
        "Supervised Apply Proposal Target Parameter shows blocked-parameter "
        "status while promotion planning is blocked"
    )
    assert m074[3] == "M073"
    assert m074[4] == "high offline frontend"
    assert m074[5] == "UNIT_VERIFIED blocked Supervised Apply parameter; live authority remains blocked"
    assert m074[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "Supervised Apply parameter now shows blocked-parameter status when visible" in (
        memory_eval[4]
    )
    assert memory_eval[-1] == "none"

    assert "M074-CP01" in plan
    assert "M074-CP02" in plan
    assert "No backend contact" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "This does not mutate proposal data" in adr

    assert "function taskProposalParameterDisplay(" in task_queue
    assert "const label = integrationBundleApplyBlocked ? 'Blocked parameter' : proposal.parameter" in task_queue
    assert "const supervisedProposalParameter = taskProposalParameterDisplay(proposal, integrationBundleApplyBlocked)" in task_queue
    assert "className={`truncate ${supervisedProposalParameter.className}`}" in task_queue
    assert "{supervisedProposalParameter.label}" in task_queue
    assert '<span className="text-muted">Parameter</span><b className="truncate text-text">{proposal.parameter}</b>' not in task_queue
    assert "supervised apply parameter shows blocked parameter" in adapter_test
    assert "Supervised Apply parameter values must fail closed" in completion
    assert "M074 Supervised Apply parameter blocker guard is now covered" in current
    assert "M074 Supervised Apply parameter blocker guard rail is now automated" in report


def test_m075_supervised_apply_old_value_reflects_integration_bundle_blocker():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M075_PLAN.read_text(encoding="utf-8")
    adr = M075_ADR.read_text(encoding="utf-8")
    task_queue = FRONTEND_TASK_QUEUE_PANEL.read_text(encoding="utf-8")
    adapter_test = FRONTEND_OPERATOR_ADAPTER_TEST.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["contacts_backend"] is False
    assert state["last_validation_summary"]["authority"]["contacts_wing"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m075 = markdown_table_row(roadmap, "M075")
    assert m075[1] == "Supervised Apply old value blocker guard"
    assert m075[2] == (
        "Supervised Apply Proposal Target Old value shows no-readback "
        "status while promotion planning is blocked"
    )
    assert m075[3] == "M074"
    assert m075[4] == "high offline frontend"
    assert m075[5] == "UNIT_VERIFIED blocked Supervised Apply old value; live authority remains blocked"
    assert m075[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "Supervised Apply old value now shows no-readback status when visible" in (
        memory_eval[4]
    )
    assert memory_eval[-1] == "none"

    assert "M075-CP01" in plan
    assert "M075-CP02" in plan
    assert "No backend contact" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "This does not mutate proposal data" in adr

    assert "function supervisedProposalOldValueDisplay(" in task_queue
    assert "const label = integrationBundleApplyBlocked ? 'No readback' : 'Current console snapshot'" in task_queue
    assert "const className = integrationBundleApplyBlocked ? 'text-danger' : 'text-muted'" in task_queue
    assert "const supervisedProposalOldValue = supervisedProposalOldValueDisplay(integrationBundleApplyBlocked)" in task_queue
    assert "className={`truncate ${supervisedProposalOldValue.className}`}" in task_queue
    assert "{supervisedProposalOldValue.label}" in task_queue
    assert '<span className="text-muted">Old value</span><b className="truncate text-muted">Current console snapshot</b>' not in task_queue
    assert "supervised apply old value shows no readback" in adapter_test
    assert "Supervised Apply old values must fail closed" in completion
    assert "M075 Supervised Apply old value blocker guard is now covered" in current
    assert "M075 Supervised Apply old value blocker guard rail is now automated" in report


def test_m076_replay_evidence_start_action_is_blocked_by_integration_bundle():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M076_PLAN.read_text(encoding="utf-8")
    adr = M076_ADR.read_text(encoding="utf-8")
    task_queue = FRONTEND_TASK_QUEUE_PANEL.read_text(encoding="utf-8")
    adapter_test = FRONTEND_OPERATOR_ADAPTER_TEST.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["contacts_backend"] is False
    assert state["last_validation_summary"]["authority"]["contacts_wing"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m076 = markdown_table_row(roadmap, "M076")
    assert m076[1] == "Replay Evidence start action blocker guard"
    assert m076[2] == (
        "Replay Evidence Start Offline action is disabled while promotion "
        "planning is blocked"
    )
    assert m076[3] == "M075"
    assert m076[4] == "high offline frontend"
    assert m076[5] == "UNIT_VERIFIED blocked Replay Evidence start action; live authority remains blocked"
    assert m076[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "Replay Evidence Start Offline action now stays disabled when visible blockers exist" in (
        memory_eval[4]
    )
    assert memory_eval[-1] == "none"

    assert "M076-CP01" in plan
    assert "M076-CP02" in plan
    assert "No backend contact" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "This does not mutate replay run input" in adr

    assert "function ReplayEvidencePanel({" in task_queue
    assert "const replayStartBlocked = integrationBundleApplyBlocked" in task_queue
    assert "const replayStartLabel = replayStartBlocked ? 'Replay Blocked' : 'Start Offline'" in task_queue
    assert "const replayStartButtonClassName = replayStartBlocked" in task_queue
    assert "const handleStartReplay = () => {" in task_queue
    assert "if (replayStartBlocked) return" in task_queue
    assert "onStartReplay()" in task_queue
    assert (
        '<button className={replayStartButtonClassName} disabled={replayStartBlocked} '
        'onClick={handleStartReplay} type="button">{replayStartLabel}</button>'
    ) in task_queue
    assert "onClick={onStartReplay}" not in task_queue
    assert "replay evidence start action is blocked" in adapter_test
    assert "Replay Evidence start actions must fail closed" in completion
    assert "M076 Replay Evidence start action blocker guard is now covered" in current
    assert "M076 Replay Evidence start action blocker guard rail is now automated" in report


def test_m077_replay_evidence_latest_run_id_reflects_integration_bundle_blocker():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M077_PLAN.read_text(encoding="utf-8")
    adr = M077_ADR.read_text(encoding="utf-8")
    task_queue = FRONTEND_TASK_QUEUE_PANEL.read_text(encoding="utf-8")
    adapter_test = FRONTEND_OPERATOR_ADAPTER_TEST.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["contacts_backend"] is False
    assert state["last_validation_summary"]["authority"]["contacts_wing"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m077 = markdown_table_row(roadmap, "M077")
    assert m077[1] == "Replay Evidence latest run id blocker guard"
    assert m077[2] == (
        "Replay Evidence Latest run id shows no-replay-start status while "
        "promotion planning is blocked"
    )
    assert m077[3] == "M076"
    assert m077[4] == "high offline frontend"
    assert m077[5] == "UNIT_VERIFIED blocked Replay Evidence latest run id; live authority remains blocked"
    assert m077[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "Replay Evidence latest run id now shows no-replay-start status when visible" in (
        memory_eval[4]
    )
    assert memory_eval[-1] == "none"

    assert "M077-CP01" in plan
    assert "M077-CP02" in plan
    assert "No backend contact" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "This does not mutate replay run input" in adr

    assert "function ReplayEvidencePanel({" in task_queue
    assert "const latestRunIdLabel = integrationBundleApplyBlocked" in task_queue
    assert "? 'No replay start'" in task_queue
    assert ": latestRun?.runId ?? 'No replay yet'" in task_queue
    assert "const latestRunIdClassName = integrationBundleApplyBlocked" in task_queue
    assert "? 'truncate text-[9px] font-bold text-danger'" in task_queue
    assert ": 'truncate text-[9px] font-bold text-text'" in task_queue
    assert "<div className={latestRunIdClassName}>{latestRunIdLabel}</div>" in task_queue
    assert '<div className="truncate text-[9px] font-bold text-text">{latestRun?.runId ?? \'No replay yet\'}</div>' not in task_queue
    assert "replay evidence latest run id is guarded" in adapter_test
    assert "Replay Evidence latest run ids must fail closed" in completion
    assert "M077 Replay Evidence latest run id blocker guard is now covered" in current
    assert "M077 Replay Evidence latest run id blocker guard rail is now automated" in report


def test_m078_replay_evidence_trace_detail_reflects_integration_bundle_blocker():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M078_PLAN.read_text(encoding="utf-8")
    adr = M078_ADR.read_text(encoding="utf-8")
    task_queue = FRONTEND_TASK_QUEUE_PANEL.read_text(encoding="utf-8")
    adapter_test = FRONTEND_OPERATOR_ADAPTER_TEST.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["contacts_backend"] is False
    assert state["last_validation_summary"]["authority"]["contacts_wing"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m078 = markdown_table_row(roadmap, "M078")
    assert m078[1] == "Replay Evidence trace detail blocker guard"
    assert m078[2] == (
        "Replay Evidence Trace detail shows no-trace-proof status while "
        "promotion planning is blocked"
    )
    assert m078[3] == "M077"
    assert m078[4] == "high offline frontend"
    assert m078[5] == "UNIT_VERIFIED blocked Replay Evidence trace detail; live authority remains blocked"
    assert m078[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "Replay Evidence trace detail now shows no-trace-proof status when visible" in (
        memory_eval[4]
    )
    assert memory_eval[-1] == "none"

    assert "M078-CP01" in plan
    assert "M078-CP02" in plan
    assert "No backend contact" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "This does not mutate decision trace input" in adr

    assert "function ReplayEvidencePanel({" in task_queue
    assert "const traceDetailLabel = integrationBundleApplyBlocked" in task_queue
    assert "? 'No trace proof'" in task_queue
    assert "? `${latestRun.traceCount} rows`" in task_queue
    assert "? `${latestTrace.actionType} ${latestTrace.delta ? `${latestTrace.delta} dB` : ''}`" in task_queue
    assert ": 'No decision trace'" in task_queue
    assert "const traceDetailClassName = integrationBundleApplyBlocked" in task_queue
    assert "? 'truncate text-[9px] font-bold text-danger'" in task_queue
    assert ": 'truncate text-[9px] font-bold text-text'" in task_queue
    assert "<div className={traceDetailClassName}>{traceDetailLabel}</div>" in task_queue
    assert '<div className="truncate text-[9px] font-bold text-text">' not in task_queue
    assert "replay evidence trace detail is guarded" in adapter_test
    assert "Replay Evidence trace details must fail closed" in completion
    assert "M078 Replay Evidence trace detail blocker guard is now covered" in current
    assert "M078 Replay Evidence trace detail blocker guard rail is now automated" in report


def test_m079_replay_evidence_channel_identity_reflects_integration_bundle_blocker():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M079_PLAN.read_text(encoding="utf-8")
    adr = M079_ADR.read_text(encoding="utf-8")
    task_queue = FRONTEND_TASK_QUEUE_PANEL.read_text(encoding="utf-8")
    adapter_test = FRONTEND_OPERATOR_ADAPTER_TEST.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["contacts_backend"] is False
    assert state["last_validation_summary"]["authority"]["contacts_wing"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m079 = markdown_table_row(roadmap, "M079")
    assert m079[1] == "Replay Evidence channel identity blocker guard"
    assert m079[2] == (
        "Replay Evidence channel identity shows no-channel-proof status while "
        "promotion planning is blocked"
    )
    assert m079[3] == "M078"
    assert m079[4] == "high offline frontend"
    assert m079[5] == "UNIT_VERIFIED blocked Replay Evidence channel identity; live authority remains blocked"
    assert m079[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "Replay Evidence channel identity now shows no-channel-proof status when visible" in (
        memory_eval[4]
    )
    assert memory_eval[-1] == "none"

    assert "M079-CP01" in plan
    assert "M079-CP02" in plan
    assert "No backend contact" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "This does not mutate channel evidence input" in adr

    assert "function ReplayEvidencePanel({" in task_queue
    assert "const channelEvidenceLabel = integrationBundleApplyBlocked" in task_queue
    assert "? 'No channel proof'" in task_queue
    assert "? `${channel.channel} ${channel.name}`" in task_queue
    assert ": 'No channel evidence'" in task_queue
    assert "const channelEvidenceClassName = integrationBundleApplyBlocked" in task_queue
    assert "? 'truncate text-danger'" in task_queue
    assert ": 'truncate text-text'" in task_queue
    assert "<span className={channelEvidenceClassName}>{channelEvidenceLabel}</span>" in task_queue
    assert (
        '<span className="truncate text-text">{channel ? `${channel.channel} '
        "${channel.name}` : 'No channel evidence'}</span>"
    ) not in task_queue
    assert "replay evidence channel identity is guarded" in adapter_test
    assert "Replay Evidence channel identities must fail closed" in completion
    assert "M079 Replay Evidence channel identity blocker guard is now covered" in current
    assert "M079 Replay Evidence channel identity blocker guard rail is now automated" in report


def test_m080_replay_evidence_report_row_visible_under_integration_bundle_blocker():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M080_PLAN.read_text(encoding="utf-8")
    adr = M080_ADR.read_text(encoding="utf-8")
    task_queue = FRONTEND_TASK_QUEUE_PANEL.read_text(encoding="utf-8")
    adapter_test = FRONTEND_OPERATOR_ADAPTER_TEST.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["contacts_backend"] is False
    assert state["last_validation_summary"]["authority"]["contacts_wing"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m080 = markdown_table_row(roadmap, "M080")
    assert m080[1] == "Replay Evidence report row blocker visibility guard"
    assert m080[2] == (
        "Replay Evidence report row remains visible with blocker status while "
        "promotion planning is blocked"
    )
    assert m080[3] == "M079"
    assert m080[4] == "high offline frontend"
    assert m080[5] == "UNIT_VERIFIED visible blocked Replay Evidence report row; live authority remains blocked"
    assert m080[-1] == "PENDING"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "Replay Evidence report row now remains visible with blocker status when visible blockers exist" in (
        memory_eval[4]
    )
    assert memory_eval[-1] == "none"

    assert "M080-CP01" in plan
    assert "M080-CP02" in plan
    assert "No backend contact" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "This does not mutate replay report input" in adr

    assert "function ReplayEvidencePanel({" in task_queue
    assert "const replayReportVisible = integrationBundleApplyBlocked || Boolean(latestRun)" in task_queue
    assert "{replayReportVisible && (" in task_queue
    assert "<div className={replayReportClassName}>" in task_queue
    assert "Report: {replayReportLabel}" in task_queue
    assert "{latestRun && (" not in task_queue
    assert "replay evidence report row remains visible" in adapter_test
    assert "Replay Evidence report rows must remain visible" in completion
    assert "M080 Replay Evidence report row blocker visibility guard is now covered" in current
    assert "M080 Replay Evidence report row blocker visibility guard rail is now automated" in report


def test_m081_wing_review_decision_binds_prestart_proof():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    completion = PROJECT_COMPLETION_GATE.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M081_PLAN.read_text(encoding="utf-8")
    adr = M081_ADR.read_text(encoding="utf-8")
    contract = (PROJECT_ROOT / "backend" / "wing_readonly_preflight_contract.py").read_text(
        encoding="utf-8"
    )
    review_tool = (PROJECT_ROOT / "tools" / "review_wing_readonly_preflight_artifacts.py").read_text(
        encoding="utf-8"
    )
    review_decision = (
        PROJECT_ROOT / "backend" / "wing_readonly_preflight_review_decision.py"
    ).read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["contacts_backend"] is False
    assert state["last_validation_summary"]["authority"]["contacts_wing"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m081 = markdown_table_row(roadmap, "M081")
    assert m081[1] == "WING review decision prestart proof binding"
    assert m081[2] == (
        "Review decision templates and ledgers bind accepted WING candidates "
        "to backend prestart and manifest-verifier proof fields"
    )
    assert m081[3] == "M002"
    assert m081[4] == "critical read-only hardware review"
    assert m081[5] == (
        "UNIT_VERIFIED stricter WING review decision proof binding; hardware remains waiting"
    )
    assert m081[-1] == "WAITING_FOR_EXTERNAL_VALIDATION"

    memory_eval = markdown_table_row(capability, "Long-term memory and evaluation")
    assert "WING review decision records now bind backend prestart and manifest-verifier proof fields" in (
        memory_eval[4]
    )
    assert memory_eval[-1] == "none"

    assert "M081-CP01" in plan
    assert "M081-CP02" in plan
    assert "M081-CP03" in plan
    assert "WING_READONLY_PREFLIGHT_REVIEW_DECISION_PRESTART_PROOF_FIELDS" in contract
    assert "decision_template_candidate_prestart_proof_incomplete" in review_tool
    assert "review_decision_{field}_missing" in review_decision
    assert "review_decision_{field}_mismatch" in review_decision
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "No live transport" in adr

    assert "WING review decision proof binding must remain offline" in completion
    assert "M081 WING review decision prestart proof binding is now covered" in current
    assert "M081 WING review decision prestart proof binding rail is now automated" in report


def test_m082_wing_handoff_surfaces_decision_template_proof_fields():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    runbook = READONLY_RUNBOOK.read_text(encoding="utf-8")
    plan = M082_PLAN.read_text(encoding="utf-8")
    adr = M082_ADR.read_text(encoding="utf-8")
    handoff_tool = (
        PROJECT_ROOT / "tools" / "build_wing_readonly_preflight_handoff_packet.py"
    ).read_text(encoding="utf-8")
    packet = handoff_packet.build_handoff_packet(generated_at="2026-06-20T00:00:00Z")
    rendered = handoff_packet.render_markdown(packet)

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["hardware_evidence"] is False
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["contacts_backend"] is False
    assert state["last_validation_summary"]["authority"]["contacts_wing"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m082 = markdown_table_row(roadmap, "M082")
    assert m082[1] == "WING handoff decision template proof fields"
    assert m082[2] == (
        "Operator handoff packets surface the review-decision template "
        "required fields and prestart/verifier proof-field subset"
    )
    assert m082[3] == "M002"
    assert m082[4] == "critical read-only hardware handoff"
    assert m082[5] == (
        "UNIT_VERIFIED handoff proof-field contract; hardware remains waiting"
    )
    assert m082[-1] == "WAITING_FOR_EXTERNAL_VALIDATION"

    assert "M082-CP01" in plan
    assert "M082-CP02" in plan
    assert "M082-CP03" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "No live transport" in adr

    assert "REVIEW_DECISION_TEMPLATE_REQUIRED_FIELDS" in handoff_tool
    assert "REVIEW_DECISION_PRESTART_PROOF_FIELDS" in handoff_tool
    assert packet["decision_template_required_fields"] == list(
        readonly_contract.WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_REQUIRED_FIELDS
    )
    assert packet["expected_decision_template_required_fields"] == packet[
        "decision_template_required_fields"
    ]
    assert packet["decision_template_required_fields_match_contract"] is True
    assert packet["decision_template_prestart_proof_fields"] == list(
        readonly_contract.WING_READONLY_PREFLIGHT_REVIEW_DECISION_PRESTART_PROOF_FIELDS
    )
    assert packet["expected_decision_template_prestart_proof_fields"] == packet[
        "decision_template_prestart_proof_fields"
    ]
    assert packet["decision_template_prestart_proof_fields_match_contract"] is True
    assert packet["decision_template_prestart_proof_fields_required"] is True
    assert packet[
        "decision_template_required_fields_include_prestart_proof_fields"
    ] is True
    assert (
        packet["decision_template_preserve_prestart_proof_fields_before_operator_edit"]
        is True
    )
    assert packet["readiness_closure_allowed_by_packet"] is False
    assert packet["live_authority_granted"] is False

    for field_name in (
        "backend_port_prestart_summary_path",
        "backend_port_prestart_summary_sha256",
        "backend_port_prestart_summary_status",
        "backend_port_prestart_summary_read_only",
        "backend_port_prestart_summary_live_write",
        "backend_port_prestart_summary_contacts_backend",
        "backend_port_prestart_summary_contacts_wing",
        "artifact_manifest_verifier_must_pass_before_websocket",
        "artifact_manifest_verifier_passed_before_websocket",
        "artifact_manifest_verifier_websocket_opened_after_pass",
    ):
        assert field_name in packet["decision_template_prestart_proof_fields"]
        assert field_name in rendered

    assert "decision_template_required_fields" in rendered
    assert "decision_template_prestart_proof_fields" in rendered
    assert "decision_template_prestart_proof_fields_required: `true`" in rendered
    assert (
        "decision_template_preserve_prestart_proof_fields_before_operator_edit: `true`"
        in rendered
    )
    assert "decision_template_prestart_proof_fields" in runbook
    assert "Removing or editing" in runbook
    assert "those generated proof fields makes the accepted decision stale or invalid" in runbook
    assert "M082 WING handoff decision template proof fields is now covered" in current
    assert "M082 WING handoff decision template proof fields rail is now automated" in report


def test_m083_source_arrangement_context_gates_stay_readonly():
    from backend.arrangement_role_proposal import (
        ArrangementRoleAnalysisInput,
        analyze_arrangement_role_need,
    )
    from backend.source_identity_confidence_proposal import (
        SourceIdentityObservation,
        analyze_source_identity_confidence,
    )

    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M083_PLAN.read_text(encoding="utf-8")
    adr = M083_ADR.read_text(encoding="utf-8")
    source_identity_module = (
        PROJECT_ROOT / "backend" / "source_identity_confidence_proposal.py"
    ).read_text(encoding="utf-8")
    arrangement_module = (
        PROJECT_ROOT / "backend" / "arrangement_role_proposal.py"
    ).read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["hardware_evidence"] is False
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["contacts_backend"] is False
    assert state["last_validation_summary"]["authority"]["contacts_wing"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m083 = markdown_table_row(roadmap, "M083")
    assert m083[1] == "Source identity and arrangement read-only context gates"
    assert m083[2] == (
        "Source-identity and arrangement-role proposal candidates require fresh "
        "read-only/no-authority context before downstream mix decisions can trust them"
    )
    assert m083[3] == "M008"
    assert m083[4] == "high offline analysis"
    assert m083[5] == (
        "UNIT_VERIFIED read-only context gate; live authority remains blocked"
    )
    assert m083[-1] == "PENDING"

    assert "M083-CP01" in plan
    assert "M083-CP02" in plan
    assert "M083-CP03" in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "No live transport" in adr

    assert "_freshness_is_no_authority" in source_identity_module
    assert "source_identity_context_cache_live_authority_present" in source_identity_module
    assert "_freshness_is_no_authority" in arrangement_module
    assert "arrangement_role_context_cache_live_authority_present" in arrangement_module

    fresh_context = {
        "status": "fresh",
        "field_count": 5,
        "fresh_field_count": 5,
        "read_only": True,
        "live_write": False,
        "queries_console": False,
    }
    live_authority_context = {
        "status": "fresh",
        "field_count": 5,
        "fresh_field_count": 5,
        "read_only": False,
        "live_write": True,
        "queries_console": True,
    }
    observation = SourceIdentityObservation(
        source_id="ch12",
        channel_id=12,
        channel_label="Mystery Pod",
        predicted_source_type="unknown",
        source_type_confidence=0.32,
        match_type="unknown",
        signal_present=True,
        activity_confidence=0.72,
    )

    source_identity_ok = analyze_source_identity_confidence(
        (observation,),
        current_state_cache_freshness=fresh_context,
    )
    assert source_identity_ok.status == "proposal_only"
    assert (
        source_identity_ok.metadata[
            "current_source_identity_context_cache_no_live_authority"
        ]
        is True
    )
    assert source_identity_ok.metadata["live_write_allowed"] is False

    source_identity_missing = analyze_source_identity_confidence((observation,))
    assert source_identity_missing.status == "analysis_only"
    assert "source_identity_context_cache_not_fresh" in source_identity_missing.metadata[
        "blockers"
    ]
    source_identity_live = analyze_source_identity_confidence(
        (observation,),
        current_state_cache_freshness=live_authority_context,
    )
    assert source_identity_live.status == "analysis_only"
    assert "source_identity_context_cache_live_authority_present" in (
        source_identity_live.metadata["blockers"]
    )
    assert (
        source_identity_live.metadata[
            "current_source_identity_context_cache_no_live_authority"
        ]
        is False
    )

    arrangement_input = ArrangementRoleAnalysisInput(
        source_id="ch:1",
        source_type="lead vocal",
        channel_label="Lead Vox",
        current_arrangement_role="unknown",
        section_label="verse",
        source_type_confidence=0.92,
        role_confidence=0.9,
        context_confidence=0.86,
        arrangement_context_available=True,
        inter_channel_context_available=True,
        reference_role_available=True,
        accepted_history_available=True,
        activity_persistence_cycles=4,
        competing_role_count=0,
        recent_role_switches=0,
        lead_salience_score=0.9,
        masking_impact_score=0.55,
        current_state_cache_freshness=fresh_context,
    )
    arrangement_ok = analyze_arrangement_role_need(arrangement_input)
    assert arrangement_ok.status == "proposal_only"
    assert (
        arrangement_ok.evidence[
            "current_arrangement_role_context_cache_no_live_authority"
        ]
        is True
    )
    assert arrangement_ok.to_operator_proposal()["can_apply_live"] is False

    arrangement_missing = analyze_arrangement_role_need(
        ArrangementRoleAnalysisInput(
            **{
                **arrangement_input.__dict__,
                "current_state_cache_freshness": None,
            }
        )
    )
    assert arrangement_missing.status == "analysis_only"
    assert "arrangement_role_context_cache_not_fresh" in arrangement_missing.blockers
    arrangement_live = analyze_arrangement_role_need(
        ArrangementRoleAnalysisInput(
            **{
                **arrangement_input.__dict__,
                "current_state_cache_freshness": live_authority_context,
            }
        )
    )
    assert arrangement_live.status == "analysis_only"
    assert "arrangement_role_context_cache_live_authority_present" in (
        arrangement_live.blockers
    )
    assert (
        arrangement_live.evidence[
            "current_arrangement_role_context_cache_no_live_authority"
        ]
        is False
    )

    assert "M083 source identity and arrangement read-only context gates is now covered" in current
    assert "M083 source identity and arrangement read-only context gates rail is now" in report
    assert "automated: source-identity review candidates" in report


def test_m084_unmasking_frequency_balance_context_gates_stay_readonly():
    from backend.bus_group_balance_proposal import (
        BusGroupBalanceAnalysisInput,
        analyze_bus_group_balance_need,
    )
    from backend.cross_adaptive_eq import CrossAdaptiveEQ
    from backend.frequency_separation_proposal import (
        FrequencySeparationSource,
        analyze_frequency_separation_need,
    )

    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M084_PLAN.read_text(encoding="utf-8")
    adr = M084_ADR.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["hardware_evidence"] is False
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["contacts_backend"] is False
    assert state["last_validation_summary"]["authority"]["contacts_wing"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m084 = markdown_table_row(roadmap, "M084")
    assert m084[1] == "Unmasking, frequency and balance context gates"
    assert m084[2] == (
        "Cross-channel unmasking, frequency separation and bus/group balance "
        "proposal candidates require fresh read-only/no-authority context before "
        "downstream mix decisions can trust them"
    )
    assert m084[3] == "M083"
    assert m084[4] == "high offline mix decision"
    assert m084[5] == (
        "UNIT_VERIFIED downstream context gate; live authority remains blocked"
    )
    assert m084[-1] == "PENDING"

    for marker in ("M084-CP01", "M084-CP02", "M084-CP03", "M084-CP04"):
        assert marker in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "No live transport" in adr

    assert "fresh read-only/no-authority bus/DCA/matrix context" in capability
    assert "fresh read-only/no-authority spectrum context" in capability
    assert "fresh read-only/no-authority masking and arrangement context" in capability

    fresh_context = {
        "status": "fresh",
        "field_count": 6,
        "fresh_field_count": 6,
        "read_only": True,
        "live_write": False,
        "queries_console": False,
    }
    live_authority_context = {
        "status": "fresh",
        "field_count": 6,
        "fresh_field_count": 6,
        "read_only": False,
        "live_write": True,
        "queries_console": True,
    }
    arrangement_context = {
        1: {
            "arrangement_role": "lead",
            "source_role": "lead_vocal",
            "role_confidence": 0.86,
        },
        2: {
            "arrangement_role": "harmonic_support",
            "source_role": "guitar",
            "role_confidence": 0.84,
        },
    }

    unmasking_ok = CrossAdaptiveEQ(overlap_tolerance_db=6.0).analyze_unmasking_proposals(
        channel_band_energy={1: {"mid": -20.0}, 2: {"mid": -21.0}},
        channel_priorities={},
        current_state_cache_freshness=fresh_context,
        arrangement_context_cache_freshness=fresh_context,
        channel_arrangement_context=arrangement_context,
    )
    assert unmasking_ok.suggested_actions
    assert unmasking_ok.metadata["live_write_allowed"] is False
    assert unmasking_ok.metadata["current_masking_context_cache_no_live_authority"] is True
    assert unmasking_ok.metadata["current_arrangement_context_cache_no_live_authority"] is True
    assert unmasking_ok.metadata["effective_channel_priorities"] == {"1": 1, "2": 3}

    unmasking_missing = CrossAdaptiveEQ().analyze_unmasking_proposals(
        channel_band_energy={1: {"mid": -20.0}, 2: {"mid": -21.0}},
        channel_priorities={1: 1, 2: 3},
        current_state_cache_freshness=fresh_context,
        arrangement_context_cache_freshness=fresh_context,
    )
    assert unmasking_missing.suggested_actions == ()
    assert "cross_adaptive_eq_arrangement_context_missing_for_channel" in (
        unmasking_missing.metadata["blockers"]
    )
    unmasking_live = CrossAdaptiveEQ().analyze_unmasking_proposals(
        channel_band_energy={1: {"mid": -20.0}, 2: {"mid": -21.0}},
        channel_priorities={1: 1, 2: 3},
        current_state_cache_freshness=live_authority_context,
        arrangement_context_cache_freshness=fresh_context,
        channel_arrangement_context=arrangement_context,
    )
    assert unmasking_live.suggested_actions == ()
    assert "cross_adaptive_eq_context_cache_live_authority_present" in (
        unmasking_live.metadata["blockers"]
    )

    sources = (
        FrequencySeparationSource(
            source_id="lead_vox",
            channel_id=1,
            label="Lead Vox",
            source_type="lead_vocal",
            arrangement_role="lead",
            stem_id="band",
            priority=1,
            band_energy_db={"high_mid": -20.0},
            stem_contribution={"high_mid": 0.44},
            confidence=0.88,
        ),
        FrequencySeparationSource(
            source_id="gtr_l",
            channel_id=7,
            label="GTR L",
            source_type="guitar",
            arrangement_role="harmonic_support",
            stem_id="band",
            priority=3,
            band_energy_db={"high_mid": -21.0},
            stem_contribution={"high_mid": 0.37},
            confidence=0.81,
        ),
    )
    frequency_ok = analyze_frequency_separation_need(
        sources,
        current_state_cache_freshness=fresh_context,
    )
    assert frequency_ok.status == "proposal_only"
    assert frequency_ok.metadata["live_write_allowed"] is False
    assert frequency_ok.metadata["current_frequency_context_cache_no_live_authority"] is True
    assert frequency_ok.recommendations
    frequency_live = analyze_frequency_separation_need(
        sources,
        current_state_cache_freshness=live_authority_context,
    )
    assert frequency_live.status == "analysis_only"
    assert frequency_live.recommendations == ()
    assert "frequency_separation_context_cache_live_authority_present" in (
        frequency_live.metadata["blockers"]
    )

    balance_input = BusGroupBalanceAnalysisInput(
        bus_id="group:drums",
        bus_kind="group",
        bus_label="DRUMS",
        source_role="drums",
        current_fader_db=-5.0,
        integrated_lufs=-26.1,
        short_term_lufs=-25.8,
        true_peak_dbfs=-8.0,
        summed_child_lufs=-25.6,
        child_count=8,
        dominant_child_share=0.35,
        main_bus_headroom_db=6.0,
        feedback_risk=0.1,
        assignment_confidence=0.9,
        confidence=0.9,
        persistence_cycles=3,
        current_state_cache_freshness=fresh_context,
    )
    balance_ok = analyze_bus_group_balance_need(balance_input)
    assert balance_ok.status == "proposal_only"
    assert balance_ok.evidence["current_bus_group_dca_matrix_cache_no_live_authority"] is True
    assert balance_ok.to_operator_proposal()["can_apply_live"] is False

    balance_live = analyze_bus_group_balance_need(
        BusGroupBalanceAnalysisInput(
            **{
                **balance_input.__dict__,
                "current_state_cache_freshness": live_authority_context,
            }
        )
    )
    assert balance_live.status == "analysis_only"
    assert "bus_group_current_state_cache_live_authority_present" in balance_live.blockers
    assert balance_live.to_operator_proposal()["can_apply_live"] is False

    assert "M084 unmasking, frequency and balance context gates is now covered" in current
    assert "M084 unmasking, frequency and balance context gates rail is now" in report
    assert "This is a dependency chain toward the project goal" in report


def test_m085_fx_send_monitor_return_safety_gates_stay_readonly():
    from backend.fx_send_proposal import FXSendAnalysisInput, analyze_fx_send_need

    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M085_PLAN.read_text(encoding="utf-8")
    adr = M085_ADR.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["hardware_evidence"] is False
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["contacts_backend"] is False
    assert state["last_validation_summary"]["authority"]["contacts_wing"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m085 = markdown_table_row(roadmap, "M085")
    assert m085[1] == "FX send monitor and return safety gates"
    assert m085[2] == (
        "FX send proposal candidates require fresh read-only/no-authority "
        "send/return context, mapped no-authority FX schema hints, "
        "monitor-safety evidence and rollback/readback sources"
    )
    assert m085[3] == "M083"
    assert m085[4] == "high offline FX decision"
    assert m085[5] == "UNIT_VERIFIED FX send safety gate; live authority remains blocked"
    assert m085[-1] == "PENDING"

    for marker in ("M085-CP01", "M085-CP02", "M085-CP03", "M085-CP04"):
        assert marker in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "No live transport" in adr

    assert "fresh read-only/no-authority send/return context" in capability
    assert "mapped FX schema hints" in capability
    assert "monitor safety and return-level safety" in capability

    fresh_context = {
        "status": "fresh",
        "field_count": 6,
        "fresh_field_count": 6,
        "read_only": True,
        "live_write": False,
        "queries_console": False,
    }
    live_authority_context = {
        "status": "fresh",
        "field_count": 6,
        "fresh_field_count": 6,
        "read_only": False,
        "live_write": True,
        "queries_console": True,
    }
    mapped_schema = {
        "status": "mapped",
        "schema_family": "reverb",
        "schema_source": "automixer_generic_fx_parameter_hints",
        "mapped_parameter_count": 6,
        "generic_parameter_count": 0,
        "physical_model_schema_verified": False,
        "live_write_authority": False,
    }
    live_schema = {**mapped_schema, "live_write_authority": True}

    base = dict(
        channel_id=6,
        channel_label="Lead Vox",
        source_role="lead_vocal",
        current_send_db=-28.0,
        arrangement_role="lead_space",
        contribution_share=0.85,
        return_headroom_db=6.0,
        feedback_risk=0.1,
        intelligibility_risk=0.2,
        source_confidence=0.92,
        persistence_cycles=3,
    )

    fx_ok = analyze_fx_send_need(
        FXSendAnalysisInput(
            **base,
            current_state_cache_freshness=fresh_context,
            current_fx_parameter_schema=mapped_schema,
        )
    )
    assert fx_ok.status == "proposal_only"
    assert fx_ok.evidence["current_fx_send_return_cache_no_live_authority"] is True
    assert fx_ok.evidence["fx_parameter_schema_no_live_authority"] is True
    assert fx_ok.evidence["monitor_safety"]["status"] == "passed"
    assert fx_ok.evidence["return_level_safety"]["status"] == "passed"
    assert fx_ok.to_operator_proposal()["can_apply_live"] is False

    fx_missing_context = analyze_fx_send_need(
        FXSendAnalysisInput(**base, current_fx_parameter_schema=mapped_schema)
    )
    assert fx_missing_context.status == "analysis_only"
    assert "fx_current_state_cache_not_fresh" in fx_missing_context.blockers
    assert fx_missing_context.to_operator_proposal()["can_apply_live"] is False

    fx_live_context = analyze_fx_send_need(
        FXSendAnalysisInput(
            **base,
            current_state_cache_freshness=live_authority_context,
            current_fx_parameter_schema=mapped_schema,
        )
    )
    assert fx_live_context.status == "analysis_only"
    assert "fx_current_state_cache_live_authority_present" in fx_live_context.blockers
    assert fx_live_context.evidence["current_fx_send_return_cache_no_live_authority"] is False

    fx_live_schema = analyze_fx_send_need(
        FXSendAnalysisInput(
            **base,
            current_state_cache_freshness=fresh_context,
            current_fx_parameter_schema=live_schema,
        )
    )
    assert fx_live_schema.status == "analysis_only"
    assert "fx_parameter_schema_live_authority_present" in fx_live_schema.blockers
    assert fx_live_schema.evidence["fx_parameter_schema_no_live_authority"] is False

    fx_monitor_blocked = analyze_fx_send_need(
        FXSendAnalysisInput(
            **base,
            monitor_send_exposure=True,
            monitor_safety_risk=0.72,
            monitor_safety_freshness={"status": "fresh"},
            current_state_cache_freshness=fresh_context,
            current_fx_parameter_schema=mapped_schema,
        )
    )
    assert fx_monitor_blocked.status == "analysis_only"
    assert "monitor_safety_risk_high" in fx_monitor_blocked.blockers
    assert fx_monitor_blocked.evidence["monitor_safety"]["status"] == "blocked"

    assert "M085 FX send monitor and return safety gates is now covered" in current
    assert "M085 FX send monitor and return safety gates rail is now" in report
    assert "effects and sends, a separate capability required by the project goal" in report


def test_m086_master_bus_pa_feed_reference_safety_gates_stay_readonly():
    from backend.master_bus_optimization_proposal import (
        MasterBusAnalysisInput,
        analyze_master_bus_need,
    )

    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M086_PLAN.read_text(encoding="utf-8")
    adr = M086_ADR.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["hardware_evidence"] is False
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["contacts_backend"] is False
    assert state["last_validation_summary"]["authority"]["contacts_wing"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m086 = markdown_table_row(roadmap, "M086")
    assert m086[1] == "Master bus PA feed and reference safety gates"
    assert m086[2] == (
        "Master-bus proposal candidates require fresh read-only/no-authority "
        "main-bus and PA-feed context, explicit PA-feed safety and "
        "reference-feed exclusion"
    )
    assert m086[3] == "M084"
    assert m086[4] == "critical offline master output"
    assert m086[5] == "UNIT_VERIFIED master/output safety gate; live authority remains blocked"
    assert m086[-1] == "PENDING"

    for marker in ("M086-CP01", "M086-CP02", "M086-CP03", "M086-CP04"):
        assert marker in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "No live transport" in adr

    assert "PA-feed risk or reference-feed exclusion" in capability
    assert "fresh read-only/no-authority main-bus and PA-feed context" in capability

    fresh_main_context = {
        "status": "fresh",
        "field_count": 8,
        "fresh_field_count": 8,
        "read_only": True,
        "live_write": False,
        "queries_console": False,
    }
    fresh_pa_context = {
        "status": "fresh",
        "field_count": 6,
        "fresh_field_count": 6,
        "read_only": True,
        "live_write": False,
        "queries_console": False,
    }
    live_authority_pa_context = {
        "status": "fresh",
        "field_count": 6,
        "fresh_field_count": 6,
        "read_only": False,
        "live_write": True,
        "queries_console": True,
    }

    base = dict(
        bus_label="Main L/R",
        target_profile="live",
        integrated_lufs=-18.7,
        short_term_lufs=-18.3,
        true_peak_dbfs=-4.5,
        crest_factor_db=10.5,
        lra_db=5.0,
        spectral_tilt_db=1.8,
        low_end_excess_db=1.4,
        harshness_db=0.4,
        stereo_correlation=0.1,
        confidence=0.9,
        persistence_cycles=3,
    )

    master_ok = analyze_master_bus_need(
        MasterBusAnalysisInput(
            **base,
            pa_feed_readback_available=True,
            pa_feed_safe_for_master_change=True,
            pa_feed_cache_freshness=fresh_pa_context,
            current_state_cache_freshness=fresh_main_context,
        )
    )
    assert master_ok.status == "proposal_only"
    assert master_ok.evidence["pa_feed_cache_no_live_authority"] is True
    assert master_ok.evidence["current_main_bus_processing_cache_no_live_authority"] is True
    assert master_ok.evidence["is_reference_feed"] is False
    assert master_ok.to_operator_proposal()["can_apply_live"] is False

    master_missing_pa = analyze_master_bus_need(
        MasterBusAnalysisInput(**base, current_state_cache_freshness=fresh_main_context)
    )
    assert master_missing_pa.status == "analysis_only"
    assert "pa_feed_readback_missing" in master_missing_pa.blockers
    assert "pa_feed_safety_not_confirmed" in master_missing_pa.blockers
    assert "pa_feed_cache_not_fresh" in master_missing_pa.blockers

    master_reference = analyze_master_bus_need(
        MasterBusAnalysisInput(
            **{**base, "bus_label": "Master Reference Capture"},
            bus_id="dante_reference",
            output_feed_role="master_reference_capture",
            is_reference_feed=True,
            pa_feed_readback_available=True,
            pa_feed_safe_for_master_change=True,
            pa_feed_cache_freshness=fresh_pa_context,
            current_state_cache_freshness=fresh_main_context,
        )
    )
    assert master_reference.status == "analysis_only"
    assert "master_reference_feed_excluded" in master_reference.blockers
    assert master_reference.evidence["is_reference_feed"] is True

    master_live_pa = analyze_master_bus_need(
        MasterBusAnalysisInput(
            **base,
            pa_feed_readback_available=True,
            pa_feed_safe_for_master_change=True,
            pa_feed_cache_freshness=live_authority_pa_context,
            current_state_cache_freshness=fresh_main_context,
        )
    )
    assert master_live_pa.status == "analysis_only"
    assert "pa_feed_cache_live_authority_present" in master_live_pa.blockers
    assert master_live_pa.evidence["pa_feed_cache_no_live_authority"] is False

    master_missing_main = analyze_master_bus_need(
        MasterBusAnalysisInput(
            **base,
            pa_feed_readback_available=True,
            pa_feed_safe_for_master_change=True,
            pa_feed_cache_freshness=fresh_pa_context,
        )
    )
    assert master_missing_main.status == "analysis_only"
    assert "master_bus_current_state_cache_not_fresh" in master_missing_main.blockers
    assert master_missing_main.to_operator_proposal()["can_apply_live"] is False

    assert "M086 master bus PA feed and reference safety gates is now covered" in current
    assert "M086 master bus PA feed and reference safety gates rail is now" in report
    assert "master-bus/output optimization, a separate capability required by the project goal" in report


def test_m087_performance_adaptation_readonly_context_gates_stay_readonly():
    from backend.performance_adaptation_proposal import (
        PerformanceAdaptationAnalysisInput,
        analyze_performance_adaptation_need,
        build_performance_adaptation_proposals,
    )

    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M087_PLAN.read_text(encoding="utf-8")
    adr = M087_ADR.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["hardware_evidence"] is False
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["contacts_backend"] is False
    assert state["last_validation_summary"]["authority"]["contacts_wing"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m087 = markdown_table_row(roadmap, "M087")
    assert m087[1] == "Performance adaptation read-only context gates"
    assert m087[2] == (
        "Performance-adaptation proposal candidates require fresh "
        "read-only/no-authority current performance context before concert-time "
        "level adaptation decisions can trust them"
    )
    assert m087[3] == "M083"
    assert m087[4] == "high offline adaptive mixing"
    assert m087[5] == "UNIT_VERIFIED performance-adaptation context gate; live authority remains blocked"
    assert m087[-1] == "PENDING"

    for marker in ("M087-CP01", "M087-CP02", "M087-CP03", "M087-CP04"):
        assert marker in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "No live transport" in adr

    assert "fresh read-only/no-authority current performance context" in capability
    assert "rate, latency, rollback, headroom and safety evidence" in capability

    fresh_context = {
        "status": "fresh",
        "field_count": 8,
        "fresh_field_count": 8,
        "read_only": True,
        "live_write": False,
        "queries_console": False,
    }
    live_authority_context = {
        "status": "fresh",
        "field_count": 8,
        "fresh_field_count": 8,
        "read_only": False,
        "live_write": True,
        "queries_console": True,
    }

    base = dict(
        target_id="ch:7",
        target_kind="channel",
        target_label="Lead Guitar",
        source_role="electric_guitar",
        arrangement_role="support",
        baseline_level_lufs=-25.0,
        current_level_lufs=-21.6,
        short_term_lufs=-21.2,
        current_fader_db=-6.0,
        true_peak_dbfs=-7.0,
        main_bus_headroom_db=5.0,
        feedback_risk=0.12,
        bleed_risk=0.2,
        baseline_available=True,
        current_readback_available=True,
        rollback_available=True,
        operator_intent="operator_adaptation_review",
        adaptation_window="performance",
        active_performance=True,
        cooldown_elapsed_sec=12.0,
        persistence_cycles=4,
        latency_budget_ms=180.0,
        latency_budget_met=True,
        source_confidence=0.9,
        trend_confidence=0.88,
        safety_confidence=0.86,
    )

    adaptation_ok = analyze_performance_adaptation_need(
        PerformanceAdaptationAnalysisInput(
            **base,
            current_state_cache_freshness=fresh_context,
        )
    )
    assert adaptation_ok.status == "proposal_only"
    assert adaptation_ok.evidence["current_performance_context_cache_no_live_authority"] is True
    assert adaptation_ok.evidence["rate_limited"] is True
    assert adaptation_ok.to_operator_proposal()["can_apply_live"] is False

    adaptation_missing_context = analyze_performance_adaptation_need(
        PerformanceAdaptationAnalysisInput(**base)
    )
    assert adaptation_missing_context.status == "analysis_only"
    assert "performance_adaptation_current_state_cache_not_fresh" in adaptation_missing_context.blockers
    assert adaptation_missing_context.to_operator_proposal()["can_apply_live"] is False
    assert build_performance_adaptation_proposals([PerformanceAdaptationAnalysisInput(**base)]) == ()

    adaptation_live_context = analyze_performance_adaptation_need(
        PerformanceAdaptationAnalysisInput(
            **base,
            current_state_cache_freshness=live_authority_context,
        )
    )
    assert adaptation_live_context.status == "analysis_only"
    assert "performance_adaptation_current_state_cache_live_authority_present" in adaptation_live_context.blockers
    assert adaptation_live_context.evidence["current_performance_context_cache_no_live_authority"] is False
    assert adaptation_live_context.to_operator_proposal()["can_apply_live"] is False

    assert "M087 performance adaptation read-only context gates is now covered" in current
    assert "M087 performance adaptation read-only context gates rail is now" in report
    assert "concert-time performance adaptation, a separate capability required by the project goal" in report


def test_m088_routing_management_readonly_context_gates_stay_readonly():
    from backend.routing_management_proposal import (
        RoutingManagementAnalysisInput,
        analyze_routing_management_need,
        build_routing_management_proposals,
    )

    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    console = CONSOLE_CAPABILITY_MATRIX.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M088_PLAN.read_text(encoding="utf-8")
    adr = M088_ADR.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["hardware_evidence"] is False
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["contacts_backend"] is False
    assert state["last_validation_summary"]["authority"]["contacts_wing"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m088 = markdown_table_row(roadmap, "M088")
    assert m088[1] == "Routing management read-only context gates"
    assert m088[2] == (
        "Routing proposal candidates require fresh read-only/no-authority "
        "current route context before console-routing decisions can trust them"
    )
    assert m088[3] == "M083"
    assert m088[4] == "critical offline console routing"
    assert m088[5] == "UNIT_VERIFIED routing context gate; live authority remains blocked"
    assert m088[-1] == "PENDING"

    routing_row = markdown_table_row(console, "Routing / patch")
    assert "`backend/routing_management_proposal.py`" in routing_row[2]
    assert routing_row[3] == "no"
    assert routing_row[4] == "no"
    assert routing_row[5] == "fresh read-only/no-authority route context, rollback and physical readback first"
    assert routing_row[-1] == "none"

    for marker in ("M088-CP01", "M088-CP02", "M088-CP03", "M088-CP04"):
        assert marker in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "No live transport" in adr

    fresh_context = {
        "status": "fresh",
        "field_count": 6,
        "fresh_field_count": 6,
        "read_only": True,
        "live_write": False,
        "queries_console": False,
    }
    live_authority_context = {
        "status": "fresh",
        "field_count": 6,
        "fresh_field_count": 6,
        "read_only": False,
        "live_write": True,
        "queries_console": True,
    }

    base = dict(
        route_id="ch:10:input",
        route_kind="channel_input",
        target_label="Lead Vox input",
        current_source_group="LOCAL",
        current_source_channel=10,
        current_source_label="Local 10",
        proposed_source_group="DANTE",
        proposed_source_channel=10,
        proposed_source_label="Dante Vox 10",
        expected_source_role="lead_vocal",
        route_readback_available=True,
        rollback_route_available=True,
        snapshot_baseline_id="soundcheck_pre_patch",
        operator_intent="operator_patch_request",
        route_change_window="soundcheck",
        active_performance=False,
        current_route_signal_dbfs=-82.0,
        proposed_source_signal_dbfs=-28.0,
        proposed_source_verified=True,
        destructive_scope_score=0.1,
        affected_target_count=1,
        source_confidence=0.9,
        route_confidence=0.92,
        persistence_cycles=3,
    )

    route_ok = analyze_routing_management_need(
        RoutingManagementAnalysisInput(
            **base,
            current_state_cache_freshness=fresh_context,
        )
    )
    assert route_ok.status == "proposal_only"
    assert route_ok.evidence["current_route_cache_no_live_authority"] is True
    assert route_ok.evidence["destructive_console_state"] is True
    assert route_ok.to_operator_proposal()["can_apply_live"] is False

    route_missing_context = analyze_routing_management_need(
        RoutingManagementAnalysisInput(**base)
    )
    assert route_missing_context.status == "analysis_only"
    assert "routing_current_state_cache_not_fresh" in route_missing_context.blockers
    assert route_missing_context.to_operator_proposal()["can_apply_live"] is False
    assert build_routing_management_proposals([RoutingManagementAnalysisInput(**base)]) == ()

    route_live_context = analyze_routing_management_need(
        RoutingManagementAnalysisInput(
            **base,
            current_state_cache_freshness=live_authority_context,
        )
    )
    assert route_live_context.status == "analysis_only"
    assert "routing_current_state_cache_live_authority_present" in route_live_context.blockers
    assert route_live_context.evidence["current_route_cache_no_live_authority"] is False
    assert route_live_context.to_operator_proposal()["can_apply_live"] is False

    assert "M088 routing management read-only context gates is now covered" in current
    assert "M088 routing management read-only context gates rail is now" in report
    assert "console routing, a separate console-control capability required by the project goal" in report


def test_m089_mute_management_readonly_context_gates_stay_readonly():
    from backend.mute_management_proposal import (
        MuteManagementAnalysisInput,
        analyze_mute_management_need,
        build_mute_management_proposals,
    )

    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    console = CONSOLE_CAPABILITY_MATRIX.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M089_PLAN.read_text(encoding="utf-8")
    adr = M089_ADR.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["hardware_evidence"] is False
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["contacts_backend"] is False
    assert state["last_validation_summary"]["authority"]["contacts_wing"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m089 = markdown_table_row(roadmap, "M089")
    assert m089[1] == "Mute management read-only context gates"
    assert m089[2] == (
        "Mute/unmute proposal candidates require fresh read-only/no-authority "
        "current mute context before console-mute decisions can trust them"
    )
    assert m089[3] == "M083"
    assert m089[4] == "critical offline console mute"
    assert m089[5] == "UNIT_VERIFIED mute context gate; live authority remains blocked"
    assert m089[-1] == "PENDING"

    mute_row = markdown_table_row(console, "Mute/unmute")
    assert "`backend/mute_management_proposal.py`" in mute_row[2]
    assert mute_row[3] == "no"
    assert mute_row[4] == "no"
    assert mute_row[5] == "fresh read-only/no-authority mute context, rollback and physical readback first"
    assert mute_row[-1] == "none"

    for marker in ("M089-CP01", "M089-CP02", "M089-CP03", "M089-CP04"):
        assert marker in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "No live transport" in adr

    fresh_context = {
        "status": "fresh",
        "field_count": 5,
        "fresh_field_count": 5,
        "read_only": True,
        "live_write": False,
        "queries_console": False,
    }
    live_authority_context = {
        "status": "fresh",
        "field_count": 5,
        "fresh_field_count": 5,
        "read_only": False,
        "live_write": True,
        "queries_console": True,
    }

    base = dict(
        channel_id=12,
        channel_label="Floor Tom",
        source_role="floor_tom",
        current_muted=False,
        allow_auto_mute_candidate=True,
        rms_dbfs=-67.0,
        peak_dbfs=-55.0,
        noise_floor_dbfs=-66.0,
        idle_duration_sec=48.0,
        source_confidence=0.9,
        bleed_risk=0.25,
        feedback_risk=0.05,
        dropout_risk=0.1,
        expected_next_cue_sec=40.0,
        persistence_cycles=3,
    )

    mute_ok = analyze_mute_management_need(
        MuteManagementAnalysisInput(
            **base,
            current_state_cache_freshness=fresh_context,
        )
    )
    assert mute_ok.status == "proposal_only"
    assert mute_ok.evidence["current_mute_cache_no_live_authority"] is True
    assert mute_ok.to_operator_proposal()["can_apply_live"] is False

    mute_missing_context = analyze_mute_management_need(
        MuteManagementAnalysisInput(**base)
    )
    assert mute_missing_context.status == "analysis_only"
    assert "mute_current_state_cache_not_fresh" in mute_missing_context.blockers
    assert mute_missing_context.to_operator_proposal()["can_apply_live"] is False
    assert build_mute_management_proposals([MuteManagementAnalysisInput(**base)]) == ()

    mute_live_context = analyze_mute_management_need(
        MuteManagementAnalysisInput(
            **base,
            current_state_cache_freshness=live_authority_context,
        )
    )
    assert mute_live_context.status == "analysis_only"
    assert "mute_current_state_cache_live_authority_present" in mute_live_context.blockers
    assert mute_live_context.evidence["current_mute_cache_no_live_authority"] is False
    assert mute_live_context.to_operator_proposal()["can_apply_live"] is False

    assert "M089 mute management read-only context gates is now covered" in current
    assert "M089 mute management read-only context gates rail is now" in report
    assert "console mute/unmute, a separate console-control capability required by the project goal" in report


def test_m090_pan_spatial_readonly_context_gates_stay_readonly():
    from backend.pan_spatial_proposal import (
        PanAnalysisInput,
        analyze_pan_need,
        build_pan_proposals,
    )

    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    console = CONSOLE_CAPABILITY_MATRIX.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M090_PLAN.read_text(encoding="utf-8")
    adr = M090_ADR.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["hardware_evidence"] is False
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["contacts_backend"] is False
    assert state["last_validation_summary"]["authority"]["contacts_wing"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m090 = markdown_table_row(roadmap, "M090")
    assert m090[1] == "Pan spatial read-only context gates"
    assert m090[2] == (
        "Pan proposal candidates require fresh read-only/no-authority current "
        "pan and mono/phase context before pan/stereo decisions can trust them"
    )
    assert m090[3] == "M083"
    assert m090[4] == "critical offline console pan"
    assert m090[5] == "UNIT_VERIFIED pan context gate; live authority remains blocked"
    assert m090[-1] == "PENDING"

    pan_row = markdown_table_row(console, "Pan")
    assert "`backend/pan_spatial_proposal.py`" in pan_row[2]
    assert pan_row[3] == "no"
    assert pan_row[4] == "no for WING MVP"
    assert pan_row[5] == (
        "fresh read-only/no-authority pan and mono/phase context, rollback and physical readback first"
    )
    assert pan_row[-1] == "none"

    for marker in ("M090-CP01", "M090-CP02", "M090-CP03", "M090-CP04"):
        assert marker in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "No live transport" in adr

    fresh_pan_context = {
        "status": "fresh",
        "field_count": 4,
        "fresh_field_count": 4,
        "read_only": True,
        "live_write": False,
        "queries_console": False,
    }
    fresh_mono_phase_context = {
        "status": "fresh",
        "field_count": 3,
        "fresh_field_count": 3,
        "read_only": True,
        "live_write": False,
        "queries_console": False,
    }
    live_authority_context = {
        "status": "fresh",
        "field_count": 4,
        "fresh_field_count": 4,
        "read_only": False,
        "live_write": True,
        "queries_console": True,
    }

    center_base = dict(
        channel_id=4,
        channel_label="Lead Vox",
        source_role="lead_vocal",
        current_pan_percent=-24.0,
        mono_compatibility_risk=0.15,
        contribution_share=0.9,
        role_confidence=0.95,
        persistence_cycles=3,
    )
    stereo_base = dict(
        channel_id=9,
        channel_label="GTR L",
        source_role="guitar",
        current_pan_percent=0.0,
        stereo_pair_id="gtr_pair",
        stereo_pair_side="left",
        mono_compatibility_risk=0.2,
        phase_correlation=0.84,
        mono_sum_delta_db=-0.4,
        contribution_share=0.65,
        role_confidence=0.8,
        persistence_cycles=3,
    )

    pan_ok = analyze_pan_need(
        PanAnalysisInput(
            **stereo_base,
            current_state_cache_freshness=fresh_pan_context,
            mono_phase_check_freshness=fresh_mono_phase_context,
        )
    )
    assert pan_ok.status == "proposal_only"
    assert pan_ok.evidence["current_pan_cache_no_live_authority"] is True
    assert pan_ok.evidence["mono_phase_check_no_live_authority"] is True
    assert pan_ok.evidence["mono_phase_safety"]["status"] == "passed"
    assert pan_ok.to_operator_proposal()["can_apply_live"] is False

    missing_pan_context = analyze_pan_need(PanAnalysisInput(**center_base))
    assert missing_pan_context.status == "analysis_only"
    assert "pan_current_state_cache_not_fresh" in missing_pan_context.blockers
    assert missing_pan_context.to_operator_proposal()["can_apply_live"] is False
    assert build_pan_proposals([PanAnalysisInput(**center_base)]) == ()

    live_pan_context = analyze_pan_need(
        PanAnalysisInput(
            **center_base,
            current_state_cache_freshness=live_authority_context,
        )
    )
    assert live_pan_context.status == "analysis_only"
    assert "pan_current_state_cache_live_authority_present" in live_pan_context.blockers
    assert live_pan_context.evidence["current_pan_cache_no_live_authority"] is False
    assert live_pan_context.to_operator_proposal()["can_apply_live"] is False

    live_mono_phase_context = analyze_pan_need(
        PanAnalysisInput(
            **stereo_base,
            current_state_cache_freshness=fresh_pan_context,
            mono_phase_check_freshness=live_authority_context,
        )
    )
    assert live_mono_phase_context.status == "analysis_only"
    assert "pan_mono_phase_check_live_authority_present" in live_mono_phase_context.blockers
    assert live_mono_phase_context.evidence["mono_phase_check_no_live_authority"] is False
    assert live_mono_phase_context.to_operator_proposal()["can_apply_live"] is False

    assert "M090 pan spatial read-only context gates is now covered" in current
    assert "M090 pan spatial read-only context gates rail is now" in report
    assert "pan/stereo placement, a separate mixing and console-control capability required by the project goal" in report


def test_m091_scene_preset_readonly_context_gates_stay_readonly():
    from backend.scene_preset_management_proposal import (
        ScenePresetManagementAnalysisInput,
        analyze_scene_preset_management_need,
        build_scene_preset_management_proposals,
    )

    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    console = CONSOLE_CAPABILITY_MATRIX.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M091_PLAN.read_text(encoding="utf-8")
    adr = M091_ADR.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["hardware_evidence"] is False
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["contacts_backend"] is False
    assert state["last_validation_summary"]["authority"]["contacts_wing"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m091 = markdown_table_row(roadmap, "M091")
    assert m091[1] == "Scene/preset management read-only context gates"
    assert m091[2] == (
        "Scene/preset proposal candidates require fresh read-only/no-authority "
        "scene/preset context, scoped diffs and rollback evidence before "
        "destructive scene decisions can trust them"
    )
    assert m091[3] == "M083"
    assert m091[4] == "critical offline console scene/preset"
    assert m091[5] == "UNIT_VERIFIED scene/preset context gate; live authority remains blocked"
    assert m091[-1] == "PENDING"

    scene_row = markdown_table_row(console, "Scenes / snapshots / presets")
    assert "`backend/scene_preset_management_proposal.py`" in scene_row[2]
    assert scene_row[3] == "no"
    assert scene_row[4] == "no"
    assert scene_row[5] == (
        "fresh read-only/no-authority scene/preset context, scoped diffs, "
        "rollback and destructive-action proof first"
    )
    assert scene_row[-1] == "none"

    for marker in ("M091-CP01", "M091-CP02", "M091-CP03", "M091-CP04"):
        assert marker in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "No live transport" in adr

    fresh_context = {
        "status": "fresh",
        "field_count": 7,
        "fresh_field_count": 7,
        "read_only": True,
        "live_write": False,
        "queries_console": False,
    }
    stale_context = {
        "status": "stale",
        "field_count": 7,
        "fresh_field_count": 4,
        "stale_field_count": 3,
        "read_only": True,
        "live_write": False,
        "queries_console": False,
    }
    live_authority_context = {
        "status": "fresh",
        "field_count": 7,
        "fresh_field_count": 7,
        "read_only": False,
        "live_write": True,
        "queries_console": True,
    }
    base_scene = dict(
        artifact_id="scene:12",
        artifact_name="Band A vocal strip",
        artifact_type="scene",
        action_type="recall",
        target_scope="channel_strip",
        target_label="Lead Vox scene recall",
        active_scene_id="scene:pre_show",
        active_scene_name="Pre-show baseline",
        artifact_exists=True,
        target_name_match=True,
        current_state_readback_available=True,
        target_preview_available=True,
        scoped_diff_available=True,
        rollback_snapshot_available=True,
        rollback_snapshot_id="rollback:pre_scene_recall",
        operator_intent="operator_scene_request",
        change_window="soundcheck",
        active_performance=False,
        affected_channel_count=1,
        affected_bus_count=0,
        affected_fx_slot_count=0,
        estimated_changed_parameter_count=12,
        destructive_scope_score=0.08,
        current_mix_signal_dbfs=-82.0,
        affected_sources_active=False,
        target_confidence=0.92,
        scope_confidence=0.9,
        rollback_confidence=0.88,
        persistence_cycles=3,
    )

    scene_ok = analyze_scene_preset_management_need(
        ScenePresetManagementAnalysisInput(
            **base_scene,
            current_state_cache_freshness=fresh_context,
        )
    )
    assert scene_ok.status == "proposal_only"
    assert scene_ok.evidence["current_scene_preset_cache_no_live_authority"] is True
    assert scene_ok.evidence["current_state_cache_freshness_status"] == "fresh"
    assert scene_ok.to_operator_proposal()["can_apply_live"] is False

    missing_context = analyze_scene_preset_management_need(
        ScenePresetManagementAnalysisInput(**base_scene)
    )
    assert missing_context.status == "analysis_only"
    assert "scene_preset_current_state_cache_not_fresh" in missing_context.blockers
    assert missing_context.to_operator_proposal()["can_apply_live"] is False
    assert build_scene_preset_management_proposals(
        [ScenePresetManagementAnalysisInput(**base_scene)]
    ) == ()

    stale_context_result = analyze_scene_preset_management_need(
        ScenePresetManagementAnalysisInput(
            **base_scene,
            current_state_cache_freshness=stale_context,
        )
    )
    assert stale_context_result.status == "analysis_only"
    assert "scene_preset_current_state_cache_not_fresh" in stale_context_result.blockers
    assert stale_context_result.to_operator_proposal()["can_apply_live"] is False

    live_context_result = analyze_scene_preset_management_need(
        ScenePresetManagementAnalysisInput(
            **base_scene,
            current_state_cache_freshness=live_authority_context,
        )
    )
    assert live_context_result.status == "analysis_only"
    assert (
        "scene_preset_current_state_cache_live_authority_present"
        in live_context_result.blockers
    )
    assert live_context_result.evidence["current_scene_preset_cache_no_live_authority"] is False
    assert live_context_result.to_operator_proposal()["can_apply_live"] is False

    assert "M091 scene preset read-only context gates is now covered" in current
    assert "M091 scene preset read-only context gates rail is now" in report
    assert "scene/snapshot/preset management, a separate console-control capability required by the project goal" in report


def test_m092_mix_variant_readonly_context_gates_stay_readonly():
    from backend.mix_variant_profile_proposal import (
        MixVariantProfileAnalysisInput,
        analyze_mix_variant_profile_need,
        build_mix_variant_profile_proposals,
    )

    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M092_PLAN.read_text(encoding="utf-8")
    adr = M092_ADR.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["hardware_evidence"] is False
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["contacts_backend"] is False
    assert state["last_validation_summary"]["authority"]["contacts_wing"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m092 = markdown_table_row(roadmap, "M092")
    assert m092[1] == "Mix variant profile read-only context gates"
    assert m092[2] == (
        "Mix-variant proposal candidates require fresh read-only/no-authority "
        "genre/task context, trusted source identity, reference metrics and "
        "arrangement context before genre/task variants can fan out"
    )
    assert m092[3] == "M083"
    assert m092[4] == "high offline genre/task mix decision"
    assert m092[5] == "UNIT_VERIFIED mix-variant context gate; live authority remains blocked"
    assert m092[-1] == "PENDING"

    genre_row = markdown_table_row(capability, "Genre and task context")
    assert "`backend/mix_variant_profile_proposal.py`" in genre_row[2]
    assert "fresh read-only/no-authority genre/task context" in genre_row[4]
    assert "trusted source identity" in genre_row[4]
    assert "reference metrics" in genre_row[4]
    assert "arrangement context" in genre_row[4]
    assert genre_row[6] == "source identity"
    assert genre_row[7] == "replay validation with no-authority context evidence"
    assert genre_row[-1] == "none"

    for marker in ("M092-CP01", "M092-CP02", "M092-CP03", "M092-CP04"):
        assert marker in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "No live transport" in adr

    fresh_context = {
        "status": "fresh",
        "field_count": 8,
        "fresh_field_count": 8,
        "read_only": True,
        "live_write": False,
        "queries_console": False,
    }
    stale_context = {
        "status": "stale",
        "field_count": 8,
        "fresh_field_count": 5,
        "stale_field_count": 3,
        "read_only": True,
        "live_write": False,
        "queries_console": False,
    }
    live_authority_context = {
        "status": "fresh",
        "field_count": 8,
        "fresh_field_count": 8,
        "read_only": False,
        "live_write": True,
        "queries_console": True,
    }
    base_variant = dict(
        variant_id="variant:pop-rock-live",
        genre="pop rock",
        task="live",
        current_profile_id="custom:live_foh",
        reference_id="ref:house_pop_rock",
        reference_lufs=-17.8,
        reference_metrics_available=True,
        arrangement_context_available=True,
        arrangement_roles=("lead_vocal", "drums", "bass", "guitar", "keys"),
        detected_source_count=18,
        stem_count=5,
        operator_intent="operator_variant_request",
        mix_window="soundcheck",
        current_integrated_lufs=-18.5,
        current_true_peak_dbfs=-5.5,
        main_bus_headroom_db=4.5,
        feedback_risk=0.18,
        genre_confidence=0.92,
        task_confidence=0.9,
        reference_confidence=0.84,
        arrangement_confidence=0.88,
        persistence_cycles=3,
    )

    variant_ok = analyze_mix_variant_profile_need(
        MixVariantProfileAnalysisInput(
            **base_variant,
            current_state_cache_freshness=fresh_context,
        )
    )
    assert variant_ok.status == "proposal_only"
    assert variant_ok.evidence["current_mix_variant_context_cache_no_live_authority"] is True
    assert variant_ok.evidence["current_state_cache_freshness_status"] == "fresh"
    assert variant_ok.to_operator_proposal()["can_apply_live"] is False

    missing_context = analyze_mix_variant_profile_need(
        MixVariantProfileAnalysisInput(**base_variant)
    )
    assert missing_context.status == "analysis_only"
    assert "mix_variant_context_cache_not_fresh" in missing_context.blockers
    assert missing_context.to_operator_proposal()["can_apply_live"] is False
    assert build_mix_variant_profile_proposals(
        [MixVariantProfileAnalysisInput(**base_variant)]
    ) == ()

    stale_context_result = analyze_mix_variant_profile_need(
        MixVariantProfileAnalysisInput(
            **base_variant,
            current_state_cache_freshness=stale_context,
        )
    )
    assert stale_context_result.status == "analysis_only"
    assert "mix_variant_context_cache_not_fresh" in stale_context_result.blockers
    assert stale_context_result.to_operator_proposal()["can_apply_live"] is False

    live_context_result = analyze_mix_variant_profile_need(
        MixVariantProfileAnalysisInput(
            **base_variant,
            current_state_cache_freshness=live_authority_context,
        )
    )
    assert live_context_result.status == "analysis_only"
    assert "mix_variant_context_cache_live_authority_present" in live_context_result.blockers
    assert live_context_result.evidence["current_mix_variant_context_cache_no_live_authority"] is False
    assert live_context_result.to_operator_proposal()["can_apply_live"] is False

    assert "M092 mix variant read-only context gates is now covered" in current
    assert "M092 mix variant read-only context gates rail is now" in report
    assert "genre/task mix-variant profiles, a separate mix-decision capability required by the project goal" in report


def test_m093_decision_evidence_readonly_context_gates_stay_readonly():
    from backend.decision_evidence_packet_proposal import (
        DecisionEvidenceAnalysisInput,
        analyze_decision_evidence_packet_need,
        build_decision_evidence_packet_proposals,
    )

    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    capability = CAPABILITY_MATRIX.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M093_PLAN.read_text(encoding="utf-8")
    adr = M093_ADR.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["hardware_evidence"] is False
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["contacts_backend"] is False
    assert state["last_validation_summary"]["authority"]["contacts_wing"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m093 = markdown_table_row(roadmap, "M093")
    assert m093[1] == "Decision evidence packet read-only context gates"
    assert m093[2] == (
        "Decision-evidence packet candidates require fresh read-only/no-authority "
        "decision context, trusted source identity, readback state and rollback "
        "planning before evidence packets can rank downstream proposals"
    )
    assert m093[3] == "M083"
    assert m093[4] == "high offline decision basis"
    assert m093[5] == (
        "UNIT_VERIFIED decision-evidence context gate; live authority remains blocked"
    )
    assert m093[-1] == "PENDING"

    decision_row = markdown_table_row(capability, "Decision evidence basis")
    assert "`backend/decision_evidence_packet_proposal.py`" in decision_row[2]
    assert "`backend/source_identity_downstream_gate.py`" in decision_row[2]
    assert "fresh read-only/no-authority decision context" in decision_row[4]
    assert "trusted source identity" in decision_row[4]
    assert "readback state and rollback planning" in decision_row[4]
    assert decision_row[5] == "high"
    assert decision_row[7] == "replay validation with no-authority decision context evidence"
    assert decision_row[-1] == "none"

    for marker in ("M093-CP01", "M093-CP02", "M093-CP03", "M093-CP04"):
        assert marker in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "No live transport" in adr

    fresh_context = {
        "status": "fresh",
        "field_count": 9,
        "fresh_field_count": 9,
        "read_only": True,
        "live_write": False,
        "queries_console": False,
    }
    stale_context = {
        "status": "stale",
        "field_count": 9,
        "fresh_field_count": 6,
        "stale_field_count": 3,
        "stale_fields": ["audio_analysis", "project_history", "genre_context"],
        "read_only": True,
        "live_write": False,
        "queries_console": False,
    }
    live_authority_context = {
        "status": "fresh",
        "field_count": 9,
        "fresh_field_count": 9,
        "read_only": False,
        "live_write": True,
        "queries_console": True,
    }
    source_identity_context = {
        "status": "fresh",
        "field_count": 4,
        "fresh_field_count": 4,
        "read_only": True,
        "live_write": False,
        "queries_console": False,
    }
    base_packet = dict(
        decision_id="proposal:lead_vocal:fader_hold",
        target_id="ch:1",
        decision_kind="proposal",
        proposed_action_family="gain_fader",
        operator_intent="soundcheck lead vocal balance",
        operator_review_enabled=True,
        audio_analysis_available=True,
        audio_analysis_confidence=0.92,
        psychoacoustic_model_available=True,
        psychoacoustic_model_confidence=0.84,
        statistical_model_available=True,
        statistical_model_confidence=0.82,
        project_history_available=True,
        project_history_confidence=0.78,
        reference_comparison_available=True,
        reference_comparison_confidence=0.81,
        previous_decision_outcomes_available=True,
        previous_decision_outcomes_confidence=0.76,
        inter_channel_context_available=True,
        inter_channel_context_confidence=0.88,
        arrangement_context_available=True,
        arrangement_context_confidence=0.86,
        genre_context_available=True,
        genre_context_confidence=0.74,
        current_state_readback_available=True,
        rollback_plan_available=True,
        true_peak_headroom_db=5.5,
        feedback_risk=0.12,
        latency_budget_ms=120.0,
        latency_budget_met=True,
        decision_persistence_cycles=4,
        conflicting_evidence_count=0,
        manual_override_active=False,
        auto_apply_requested=False,
        active_performance=False,
        source_identity_trust_state="trusted_for_downstream_priority",
        source_identity_trust_confidence=0.91,
        source_identity_replay_outcome_count=8,
        source_identity_calibration_freshness=source_identity_context,
    )

    packet_ok = analyze_decision_evidence_packet_need(
        DecisionEvidenceAnalysisInput(
            **base_packet,
            current_state_cache_freshness=fresh_context,
        )
    )
    assert packet_ok.status == "proposal_only"
    assert packet_ok.evidence["current_decision_evidence_context_cache_no_live_authority"] is True
    assert packet_ok.evidence["current_state_cache_freshness_status"] == "fresh"
    assert packet_ok.evidence["source_identity_downstream_trust"]["usable_for_downstream"] is True
    assert packet_ok.to_operator_proposal()["can_apply_live"] is False

    missing_context = analyze_decision_evidence_packet_need(
        DecisionEvidenceAnalysisInput(**base_packet)
    )
    assert missing_context.status == "analysis_only"
    assert "decision_evidence_context_cache_not_fresh" in missing_context.blockers
    assert missing_context.to_operator_proposal()["can_apply_live"] is False
    assert build_decision_evidence_packet_proposals(
        [DecisionEvidenceAnalysisInput(**base_packet)]
    ) == ()

    stale_context_result = analyze_decision_evidence_packet_need(
        DecisionEvidenceAnalysisInput(
            **base_packet,
            current_state_cache_freshness=stale_context,
        )
    )
    assert stale_context_result.status == "analysis_only"
    assert "decision_evidence_context_cache_not_fresh" in stale_context_result.blockers
    assert stale_context_result.to_operator_proposal()["can_apply_live"] is False

    live_context_result = analyze_decision_evidence_packet_need(
        DecisionEvidenceAnalysisInput(
            **base_packet,
            current_state_cache_freshness=live_authority_context,
        )
    )
    assert live_context_result.status == "analysis_only"
    assert (
        "decision_evidence_context_cache_live_authority_present"
        in live_context_result.blockers
    )
    assert live_context_result.evidence[
        "current_decision_evidence_context_cache_no_live_authority"
    ] is False
    assert live_context_result.to_operator_proposal()["can_apply_live"] is False

    assert "M093 decision evidence read-only context gates is now covered" in current
    assert "M093 decision evidence read-only context gates rail is now" in report
    assert "decision evidence and explainability, a separate decision-basis capability required by the project goal" in report


def test_m094_project_director_external_validation_scope_guard_stays_readonly():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    heartbeat = AUTOPILOT_HEARTBEAT.read_text(encoding="utf-8")
    plan = M094_PLAN.read_text(encoding="utf-8")
    adr = M094_ADR.read_text(encoding="utf-8")
    director = load_project_director_module()

    directive = director.build_directive(
        project_root=PROJECT_ROOT,
        lookback_commits=1,
        report_limit=1,
    )

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["hardware_evidence"] is False
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["contacts_backend"] is False
    assert state["last_validation_summary"]["authority"]["contacts_wing"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m094 = markdown_table_row(roadmap, "M094")
    assert m094[1] == "Project Director external-validation scope guard"
    assert m094[2] == (
        "Project Director narrows autonomous work to WING external-validation "
        "handoff/intake or control-plane state while M001 waits for real "
        "WING/operator evidence"
    )
    assert m094[3] == "M001"
    assert m094[4] == "critical autopilot direction"
    assert m094[5] == (
        "UNIT_VERIFIED external-validation scope guard; live authority remains blocked"
    )
    assert m094[-1] == "WAITING_FOR_EXTERNAL_VALIDATION"

    for marker in ("M094-CP01", "M094-CP02", "M094-CP03"):
        assert marker in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "No live transport" in adr

    assert directive["selected_blocker"] == "WING read-only connection proof"
    assert directive["selected_blocker_status"] in {
        "tooling_ready_waits_real_wing",
        "blocked_by_real_artifact_review",
        "blocked_by_wing_fader_rollback_proof",
    }
    if directive["selected_blocker_status"] == "blocked_by_real_artifact_review":
        assert directive["selected_action_source"] == "WING Read-Only Decision Review"
    elif (
        directive["selected_blocker_status"]
        == "blocked_by_wing_fader_rollback_proof"
    ):
        assert directive["selected_action_source"] == "WING Supervised Fader Rollback Proof"
    assert directive["selected_action_requires_external_validation"] is True
    assert directive["external_validation_wait"] is True
    assert directive["stop_broad_work"] is True
    assert (
        directive["allowed_autonomous_scope"]
        == "external_validation_handoff_or_control_plane_only"
    )
    assert any("control-plane docs/state/rails" in item for item in directive["allowed_scope"])
    assert not any("stay offline/proposal-only" in item for item in directive["allowed_scope"])
    assert any(
        "No additional proposal-only context-gate milestone" in item
        for item in directive["forbidden_scope"]
    )
    assert any("real WING artifact" in item for item in directive["forbidden_scope"])

    assert "M094 Project Director external-validation scope guard is now covered" in current
    assert "M094 Project Director external-validation scope guard rail is now" in report
    assert "proposal-only context-gate" in current
    assert "milestones are not allowed" in current
    assert "external_validation_handoff_or_control_plane_only" in report
    assert "Writer lease: none" in heartbeat
    assert (
        "No WING/backend contact was made" in heartbeat
        or "No WING contact was made by this cycle" in heartbeat
    )
    normalized_heartbeat = re.sub(r"\s+", " ", heartbeat)
    assert "No OSC write path was opened by this cycle" in normalized_heartbeat


def test_m095_wing_readonly_reserved_root_intake_guard_stays_readonly():
    from backend.wing_readonly_preflight_evidence import (
        validate_wing_readonly_preflight_artifact,
    )

    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M095_PLAN.read_text(encoding="utf-8")
    adr = M095_ADR.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["hardware_evidence"] is False
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["contacts_backend"] is False
    assert state["last_validation_summary"]["authority"]["contacts_wing"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m095 = markdown_table_row(roadmap, "M095")
    assert m095[1] == "WING read-only reserved-root intake guard"
    assert m095[2] == (
        "Real WING read-only intake rejects spoofed candidates from reserved "
        "no-WING and autopilot evidence roots before manual review can accept them"
    )
    assert m095[3] == "M001"
    assert m095[4] == "critical read-only hardware intake"
    assert m095[5] == (
        "UNIT_VERIFIED reserved-root intake guard; live authority remains blocked"
    )
    assert m095[-1] == "WAITING_FOR_EXTERNAL_VALIDATION"

    for marker in ("M095-CP01", "M095-CP02", "M095-CP03"):
        assert marker in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "No live transport" in adr

    validation = validate_wing_readonly_preflight_artifact(
        {
            "proof_artifact_schema": "wing_readonly_preflight.v1",
            "proof_runner": "wing_readonly_preflight_runner",
            "artifact_root": "artifacts/no_wing_release_check/spoofed-wing",
            "summary_path": (
                "artifacts/no_wing_release_check/spoofed-wing/"
                "20260606T120000Z_wing_readonly_preflight/summary.json"
            ),
            "events_path": (
                "artifacts/no_wing_release_check/spoofed-wing/"
                "20260606T120000Z_wing_readonly_preflight/events.jsonl"
            ),
        }
    )
    assert validation["accepted_as_readonly_preflight_candidate"] is False
    assert "artifact_root_reserved_for_non_wing_evidence" in validation["violations"]
    assert "summary_path_reserved_for_non_wing_evidence" in validation["violations"]
    assert "events_path_reserved_for_non_wing_evidence" in validation["violations"]

    assert "M095 WING read-only reserved-root intake guard is now covered" in current
    assert "M095 WING read-only reserved-root intake guard rail is now" in report
    assert "artifacts/no_wing_release_check" in current
    assert "ops/autopilot/evidence" in report


def test_m096_wing_readonly_declared_path_intake_guard_stays_readonly():
    from backend.wing_readonly_preflight_evidence import (
        validate_wing_readonly_preflight_artifact,
    )

    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M096_PLAN.read_text(encoding="utf-8")
    adr = M096_ADR.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["hardware_evidence"] is False
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["contacts_backend"] is False
    assert state["last_validation_summary"]["authority"]["contacts_wing"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m096 = markdown_table_row(roadmap, "M096")
    assert m096[1] == "WING read-only declared path intake guard"
    assert m096[2] == (
        "Real WING read-only intake rejects candidates whose summary/events "
        "paths do not match declared artifact_root and run_id before manual "
        "review can accept them"
    )
    assert m096[3] == "M001"
    assert m096[4] == "critical read-only hardware intake"
    assert m096[5] == (
        "UNIT_VERIFIED declared artifact path-shape guard; live authority remains blocked"
    )
    assert m096[-1] == "WAITING_FOR_EXTERNAL_VALIDATION"

    for marker in ("M096-CP01", "M096-CP02", "M096-CP03"):
        assert marker in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "No live transport" in adr

    validation = validate_wing_readonly_preflight_artifact(
        {
            "proof_artifact_schema": "wing_readonly_preflight.v1",
            "proof_runner": "wing_readonly_preflight_runner",
            "artifact_root": "artifacts/wing_readonly_preflight",
            "run_id": "20260606T120000Z_wing_readonly_preflight",
            "summary_path": "artifacts/wing_readonly_preflight/stale/summary.json",
            "events_path": "artifacts/wing_readonly_preflight/stale/events.jsonl",
        }
    )
    assert validation["accepted_as_readonly_preflight_candidate"] is False
    assert (
        "summary_path_declared_artifact_path_mismatch" in validation["violations"]
    )
    assert (
        "events_path_declared_artifact_path_mismatch" in validation["violations"]
    )

    assert "M096 WING read-only declared path intake guard is now covered" in current
    assert "M096 WING read-only declared path intake guard rail is now" in report
    assert "summary_path_declared_artifact_path_mismatch" in current
    assert "events_path_declared_artifact_path_mismatch" in report


def test_m097_wing_readonly_canonical_root_intake_guard_stays_readonly():
    from backend.wing_readonly_preflight_evidence import (
        validate_wing_readonly_preflight_artifact,
    )

    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M097_PLAN.read_text(encoding="utf-8")
    adr = M097_ADR.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["hardware_evidence"] is False
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["contacts_backend"] is False
    assert state["last_validation_summary"]["authority"]["contacts_wing"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m097 = markdown_table_row(roadmap, "M097")
    assert m097[1] == "WING read-only canonical root intake guard"
    assert m097[2] == (
        "Real WING read-only intake rejects candidates whose artifact_root is "
        "not the canonical WING read-only capture root before manual review can "
        "accept them"
    )
    assert m097[3] == "M001"
    assert m097[4] == "critical read-only hardware intake"
    assert m097[5] == (
        "UNIT_VERIFIED canonical artifact-root guard; live authority remains blocked"
    )
    assert m097[-1] == "WAITING_FOR_EXTERNAL_VALIDATION"

    for marker in ("M097-CP01", "M097-CP02", "M097-CP03"):
        assert marker in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "No live transport" in adr

    validation = validate_wing_readonly_preflight_artifact(
        {
            "proof_artifact_schema": "wing_readonly_preflight.v1",
            "proof_runner": "wing_readonly_preflight_runner",
            "artifact_root": "artifacts/stale_wing_readonly_preflight",
            "run_id": "20260606T120000Z_wing_readonly_preflight",
            "summary_path": (
                "artifacts/stale_wing_readonly_preflight/"
                "20260606T120000Z_wing_readonly_preflight/summary.json"
            ),
            "events_path": (
                "artifacts/stale_wing_readonly_preflight/"
                "20260606T120000Z_wing_readonly_preflight/events.jsonl"
            ),
        }
    )
    assert validation["accepted_as_readonly_preflight_candidate"] is False
    assert "artifact_root_not_wing_readonly_capture_root" in validation["violations"]

    assert "M097 WING read-only canonical root intake guard is now covered" in current
    assert "M097 WING read-only canonical root intake guard rail is now" in report
    assert "artifact_root_not_wing_readonly_capture_root" in current
    assert "artifact_root_not_wing_readonly_capture_root" in report


def test_m098_wing_readonly_source_path_intake_guard_stays_readonly():
    from backend.wing_readonly_preflight_evidence import (
        validate_wing_readonly_preflight_artifact,
    )

    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M098_PLAN.read_text(encoding="utf-8")
    adr = M098_ADR.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["hardware_evidence"] is False
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["contacts_backend"] is False
    assert state["last_validation_summary"]["authority"]["contacts_wing"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m098 = markdown_table_row(roadmap, "M098")
    assert m098[1] == "WING read-only source path intake guard"
    assert m098[2] == (
        "Real WING read-only intake rejects copied or stale-source summary "
        "files whose source path does not match the declared summary_path "
        "before manual review can accept them"
    )
    assert m098[3] == "M001"
    assert m098[4] == "critical read-only hardware intake"
    assert m098[5] == (
        "UNIT_VERIFIED source-path binding guard; live authority remains blocked"
    )
    assert m098[-1] == "WAITING_FOR_EXTERNAL_VALIDATION"

    for marker in ("M098-CP01", "M098-CP02", "M098-CP03"):
        assert marker in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "No live transport" in adr

    validation = validate_wing_readonly_preflight_artifact(
        {
            "proof_artifact_schema": "wing_readonly_preflight.v1",
            "proof_runner": "wing_readonly_preflight_runner",
            "artifact_root": "artifacts/wing_readonly_preflight",
            "run_id": "20260606T120000Z_wing_readonly_preflight",
            "summary_path": (
                "artifacts/wing_readonly_preflight/"
                "20260606T120000Z_wing_readonly_preflight/summary.json"
            ),
            "events_path": (
                "artifacts/wing_readonly_preflight/"
                "20260606T120000Z_wing_readonly_preflight/events.jsonl"
            ),
        },
        summary_path=Path("artifacts/wing_readonly_preflight/stale-source/summary.json"),
    )
    assert validation["accepted_as_readonly_preflight_candidate"] is False
    assert "source_path_declared_summary_path_mismatch" in validation["violations"]

    assert "M098 WING read-only source path intake guard is now covered" in current
    assert "M098 WING read-only source path intake guard rail is now" in report
    assert "source_path_declared_summary_path_mismatch" in current
    assert "source_path_declared_summary_path_mismatch" in report


def test_m099_wing_readonly_run_id_path_segment_intake_guard_stays_readonly():
    from backend.wing_readonly_preflight_evidence import (
        validate_wing_readonly_preflight_artifact,
    )

    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M099_PLAN.read_text(encoding="utf-8")
    adr = M099_ADR.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["hardware_evidence"] is False
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["contacts_backend"] is False
    assert state["last_validation_summary"]["authority"]["contacts_wing"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m099 = markdown_table_row(roadmap, "M099")
    assert m099[1] == "WING read-only run id path segment intake guard"
    assert m099[2] == (
        "Real WING read-only intake rejects candidates whose run_id is not a "
        "single safe path segment before manual review can accept them"
    )
    assert m099[3] == "M001"
    assert m099[4] == "critical read-only hardware intake"
    assert m099[5] == (
        "UNIT_VERIFIED run-id path-segment guard; live authority remains blocked"
    )
    assert m099[-1] == "WAITING_FOR_EXTERNAL_VALIDATION"

    for marker in ("M099-CP01", "M099-CP02", "M099-CP03"):
        assert marker in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "No live transport" in adr

    validation = validate_wing_readonly_preflight_artifact(
        {
            "proof_artifact_schema": "wing_readonly_preflight.v1",
            "proof_runner": "wing_readonly_preflight_runner",
            "artifact_root": "artifacts/wing_readonly_preflight",
            "run_id": "../stale_wing_readonly_preflight",
            "summary_path": (
                "artifacts/wing_readonly_preflight/../stale_wing_readonly_preflight/"
                "summary.json"
            ),
            "events_path": (
                "artifacts/wing_readonly_preflight/../stale_wing_readonly_preflight/"
                "events.jsonl"
            ),
        }
    )
    assert validation["accepted_as_readonly_preflight_candidate"] is False
    assert "run_id_not_single_safe_path_segment" in validation["violations"]

    assert "M099 WING read-only run id path segment intake guard is now covered" in current
    assert "M099 WING read-only run id path segment intake guard rail is now" in report
    assert "run_id_not_single_safe_path_segment" in current
    assert "run_id_not_single_safe_path_segment" in report


def test_m100_wing_readonly_artifact_root_parent_traversal_intake_guard_stays_readonly():
    from backend.wing_readonly_preflight_evidence import (
        validate_wing_readonly_preflight_artifact,
    )

    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M100_PLAN.read_text(encoding="utf-8")
    adr = M100_ADR.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["hardware_evidence"] is False
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["contacts_backend"] is False
    assert state["last_validation_summary"]["authority"]["contacts_wing"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m100 = markdown_table_row(roadmap, "M100")
    assert m100[1] == "WING read-only artifact root parent traversal intake guard"
    assert m100[2] == (
        "Real WING read-only intake rejects candidates whose artifact_root "
        "contains parent traversal before manual review can accept them"
    )
    assert m100[3] == "M001"
    assert m100[4] == "critical read-only hardware intake"
    assert m100[5] == (
        "UNIT_VERIFIED artifact-root traversal guard; live authority remains blocked"
    )
    assert m100[-1] == "WAITING_FOR_EXTERNAL_VALIDATION"

    for marker in ("M100-CP01", "M100-CP02", "M100-CP03"):
        assert marker in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "No live transport" in adr

    validation = validate_wing_readonly_preflight_artifact(
        {
            "proof_artifact_schema": "wing_readonly_preflight.v1",
            "proof_runner": "wing_readonly_preflight_runner",
            "artifact_root": "tmp/container/../artifacts/wing_readonly_preflight",
            "run_id": "20260606T120000Z_wing_readonly_preflight",
            "summary_path": (
                "tmp/container/../artifacts/wing_readonly_preflight/"
                "20260606T120000Z_wing_readonly_preflight/summary.json"
            ),
            "events_path": (
                "tmp/container/../artifacts/wing_readonly_preflight/"
                "20260606T120000Z_wing_readonly_preflight/events.jsonl"
            ),
        }
    )
    assert validation["accepted_as_readonly_preflight_candidate"] is False
    assert "artifact_root_contains_parent_traversal" in validation["violations"]

    assert (
        "M100 WING read-only artifact root parent traversal intake guard is now covered"
        in current
    )
    assert (
        "M100 WING read-only artifact root parent traversal intake guard rail is now"
        in report
    )
    assert "artifact_root_contains_parent_traversal" in current
    assert "artifact_root_contains_parent_traversal" in report


def test_m101_wing_readonly_artifact_root_whitespace_intake_guard_stays_readonly():
    from backend.wing_readonly_preflight_evidence import (
        validate_wing_readonly_preflight_artifact,
    )

    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M101_PLAN.read_text(encoding="utf-8")
    adr = M101_ADR.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["hardware_evidence"] is False
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["contacts_backend"] is False
    assert state["last_validation_summary"]["authority"]["contacts_wing"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m101 = markdown_table_row(roadmap, "M101")
    assert m101[1] == "WING read-only artifact root whitespace intake guard"
    assert m101[2] == (
        "Real WING read-only intake rejects candidates whose artifact_root has "
        "surrounding whitespace before manual review can accept them"
    )
    assert m101[3] == "M001"
    assert m101[4] == "critical read-only hardware intake"
    assert m101[5] == (
        "UNIT_VERIFIED artifact-root whitespace guard; live authority remains blocked"
    )
    assert m101[-1] == "WAITING_FOR_EXTERNAL_VALIDATION"

    for marker in ("M101-CP01", "M101-CP02", "M101-CP03"):
        assert marker in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "No live transport" in adr

    validation = validate_wing_readonly_preflight_artifact(
        {
            "proof_artifact_schema": "wing_readonly_preflight.v1",
            "proof_runner": "wing_readonly_preflight_runner",
            "artifact_root": "artifacts/wing_readonly_preflight ",
            "run_id": "20260606T120000Z_wing_readonly_preflight",
            "summary_path": (
                "artifacts/wing_readonly_preflight /"
                "20260606T120000Z_wing_readonly_preflight/summary.json"
            ),
            "events_path": (
                "artifacts/wing_readonly_preflight /"
                "20260606T120000Z_wing_readonly_preflight/events.jsonl"
            ),
        }
    )
    assert validation["accepted_as_readonly_preflight_candidate"] is False
    assert "artifact_root_has_surrounding_whitespace" in validation["violations"]

    assert "M101 WING read-only artifact root whitespace intake guard is now covered" in current
    assert "M101 WING read-only artifact root whitespace intake guard rail is now" in report
    assert "artifact_root_has_surrounding_whitespace" in current
    assert "artifact_root_has_surrounding_whitespace" in report


def test_m102_wing_readonly_run_id_format_intake_guard_stays_readonly():
    from backend.wing_readonly_preflight_evidence import (
        validate_wing_readonly_preflight_artifact,
    )

    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M102_PLAN.read_text(encoding="utf-8")
    adr = M102_ADR.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["hardware_evidence"] is False
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["contacts_backend"] is False
    assert state["last_validation_summary"]["authority"]["contacts_wing"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m102 = markdown_table_row(roadmap, "M102")
    assert m102[1] == "WING read-only run id format intake guard"
    assert m102[2] == (
        "Real WING read-only intake rejects candidates whose run_id is not the "
        "runner-generated UTC capture format before manual review can accept them"
    )
    assert m102[3] == "M001"
    assert m102[4] == "critical read-only hardware intake"
    assert m102[5] == (
        "UNIT_VERIFIED run-id format guard; live authority remains blocked"
    )
    assert m102[-1] == "WAITING_FOR_EXTERNAL_VALIDATION"

    for marker in ("M102-CP01", "M102-CP02", "M102-CP03"):
        assert marker in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "No live transport" in adr

    validation = validate_wing_readonly_preflight_artifact(
        {
            "proof_artifact_schema": "wing_readonly_preflight.v1",
            "proof_runner": "wing_readonly_preflight_runner",
            "artifact_root": "artifacts/wing_readonly_preflight",
            "run_id": "manual_wing_readonly_preflight",
            "summary_path": (
                "artifacts/wing_readonly_preflight/"
                "manual_wing_readonly_preflight/summary.json"
            ),
            "events_path": (
                "artifacts/wing_readonly_preflight/"
                "manual_wing_readonly_preflight/events.jsonl"
            ),
        }
    )
    assert validation["accepted_as_readonly_preflight_candidate"] is False
    assert "run_id_not_wing_readonly_capture_format" in validation["violations"]

    assert "M102 WING read-only run id format intake guard is now covered" in current
    assert "M102 WING read-only run id format intake guard rail is now" in report
    assert "run_id_not_wing_readonly_capture_format" in current
    assert "run_id_not_wing_readonly_capture_format" in report


def test_m103_wing_readonly_run_id_timestamp_intake_guard_stays_readonly():
    from backend.wing_readonly_preflight_evidence import (
        validate_wing_readonly_preflight_artifact,
    )

    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M103_PLAN.read_text(encoding="utf-8")
    adr = M103_ADR.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["hardware_evidence"] is False
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["contacts_backend"] is False
    assert state["last_validation_summary"]["authority"]["contacts_wing"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m103 = markdown_table_row(roadmap, "M103")
    assert m103[1] == "WING read-only run id timestamp intake guard"
    assert m103[2] == (
        "Real WING read-only intake rejects candidates whose run_id timestamp "
        "is not a valid UTC capture time before manual review can accept them"
    )
    assert m103[3] == "M001"
    assert m103[4] == "critical read-only hardware intake"
    assert m103[5] == (
        "UNIT_VERIFIED run-id timestamp guard; live authority remains blocked"
    )
    assert m103[-1] == "WAITING_FOR_EXTERNAL_VALIDATION"

    for marker in ("M103-CP01", "M103-CP02", "M103-CP03"):
        assert marker in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "No live transport" in adr

    validation = validate_wing_readonly_preflight_artifact(
        {
            "proof_artifact_schema": "wing_readonly_preflight.v1",
            "proof_runner": "wing_readonly_preflight_runner",
            "artifact_root": "artifacts/wing_readonly_preflight",
            "run_id": "20261340T250000Z_wing_readonly_preflight",
            "summary_path": (
                "artifacts/wing_readonly_preflight/"
                "20261340T250000Z_wing_readonly_preflight/summary.json"
            ),
            "events_path": (
                "artifacts/wing_readonly_preflight/"
                "20261340T250000Z_wing_readonly_preflight/events.jsonl"
            ),
        }
    )
    assert validation["accepted_as_readonly_preflight_candidate"] is False
    assert "run_id_not_valid_utc_timestamp" in validation["violations"]

    assert "M103 WING read-only run id timestamp intake guard is now covered" in current
    assert "M103 WING read-only run id timestamp intake guard rail is now" in report
    assert "run_id_not_valid_utc_timestamp" in current
    assert "run_id_not_valid_utc_timestamp" in report


def test_m104_wing_readonly_capture_time_order_intake_guard_stays_readonly():
    from backend.wing_readonly_preflight_evidence import (
        validate_wing_readonly_preflight_artifact,
    )

    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M104_PLAN.read_text(encoding="utf-8")
    adr = M104_ADR.read_text(encoding="utf-8")

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["hardware_evidence"] is False
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["contacts_backend"] is False
    assert state["last_validation_summary"]["authority"]["contacts_wing"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m104 = markdown_table_row(roadmap, "M104")
    assert m104[1] == "WING read-only capture time order intake guard"
    assert m104[2] == (
        "Real WING read-only intake rejects candidates whose captured_at is "
        "earlier than the UTC capture timestamp encoded in run_id"
    )
    assert m104[3] == "M001"
    assert m104[4] == "critical read-only hardware intake"
    assert m104[5] == (
        "UNIT_VERIFIED capture-time ordering guard; live authority remains blocked"
    )
    assert m104[-1] == "WAITING_FOR_EXTERNAL_VALIDATION"

    for marker in ("M104-CP01", "M104-CP02", "M104-CP03"):
        assert marker in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "No live transport" in adr

    validation = validate_wing_readonly_preflight_artifact(
        {
            "proof_artifact_schema": "wing_readonly_preflight.v1",
            "proof_runner": "wing_readonly_preflight_runner",
            "artifact_root": "artifacts/wing_readonly_preflight",
            "run_id": "20260608T010000Z_wing_readonly_preflight",
            "captured_at": "2026-06-07T01:00:00Z",
            "summary_path": (
                "artifacts/wing_readonly_preflight/"
                "20260608T010000Z_wing_readonly_preflight/summary.json"
            ),
            "events_path": (
                "artifacts/wing_readonly_preflight/"
                "20260608T010000Z_wing_readonly_preflight/events.jsonl"
            ),
        }
    )
    assert validation["accepted_as_readonly_preflight_candidate"] is False
    assert "captured_at_before_run_id_timestamp" in validation["violations"]

    assert "M104 WING read-only capture time order intake guard is now covered" in current
    assert "M104 WING read-only capture time order intake guard rail is now" in report
    assert "captured_at_before_run_id_timestamp" in current
    assert "captured_at_before_run_id_timestamp" in report


def test_m105_project_director_post_m104_intake_loop_guard_stays_readonly():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M105_PLAN.read_text(encoding="utf-8")
    adr = M105_ADR.read_text(encoding="utf-8")
    director = load_project_director_module()

    directive = director.build_directive(
        project_root=PROJECT_ROOT,
        lookback_commits=12,
        report_limit=1,
    )

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["hardware_evidence"] is False
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["contacts_backend"] is False
    assert state["last_validation_summary"]["authority"]["contacts_wing"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m105 = markdown_table_row(roadmap, "M105")
    assert m105[1] == "Project Director post-M104 intake loop guard"
    assert m105[2] == (
        "Project Director detects the post-M104 WING read-only intake-hardening "
        "streak and narrows further autonomous work to real external validation "
        "or a named unverified contract gap"
    )
    assert m105[3] == "M001"
    assert m105[4] == "critical autopilot direction"
    assert m105[5] == (
        "UNIT_VERIFIED post-M104 intake-loop guard; live authority remains blocked"
    )
    assert m105[-1] == "WAITING_FOR_EXTERNAL_VALIDATION"

    for marker in ("M105-CP01", "M105-CP02", "M105-CP03"):
        assert marker in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "No live transport" in adr

    if directive["selected_blocker_status"] in {
        "blocked_by_real_artifact_review",
        "blocked_by_wing_fader_rollback_proof",
    }:
        assert directive["selected_action_source"] in {
            "WING Read-Only Decision Review",
            "WING Supervised Fader Rollback Proof",
        }
        assert directive["pre_hardware_intake_hardening_loop_guard"] is False
        assert (
            directive["allowed_autonomous_scope"]
            == "external_validation_handoff_or_control_plane_only"
        )
        assert any(
            "selected external WING validation handoff/intake" in item
            for item in directive["allowed_scope"]
        )
    else:
        assert directive["pre_hardware_intake_hardening_loop_guard"] is True
        assert directive["pre_hardware_intake_guard_recent_streak_count"] >= 10
        assert (
            directive["allowed_autonomous_scope"]
            == "external_validation_wait_or_named_gap_only"
        )
        assert any("After M104" in item for item in directive["allowed_scope"])
        assert any(
            "No further generic WING read-only intake guard milestone after M104" in item
            for item in directive["forbidden_scope"]
        )
        assert any(
            "Do not keep improving the same WING read-only intake function" in item
            for item in directive["forbidden_scope"]
        )

    assert "M105 Project Director post-M104 intake loop guard is now covered" in current
    assert "M105 Project Director post-M104 intake loop guard rail is now" in report
    assert "external_validation_wait_or_named_gap_only" in current
    assert "external_validation_wait_or_named_gap_only" in report


def test_m106_project_director_post_m105_intake_loop_guard_persists():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M106_PLAN.read_text(encoding="utf-8")
    adr = M106_ADR.read_text(encoding="utf-8")
    director = load_project_director_module()

    directive = director.build_directive(
        project_root=PROJECT_ROOT,
        lookback_commits=12,
        report_limit=1,
    )

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["hardware_evidence"] is False
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["contacts_backend"] is False
    assert state["last_validation_summary"]["authority"]["contacts_wing"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m106 = markdown_table_row(roadmap, "M106")
    assert m106[1] == "Project Director post-M105 intake loop persistence"
    assert m106[2] == (
        "Project Director keeps the post-M104 intake-loop guard active even "
        "after direction-guard commits sit above the WING read-only streak"
    )
    assert m106[3] == "M001"
    assert m106[4] == "critical autopilot direction"
    assert m106[5] == (
        "UNIT_VERIFIED post-M105 loop persistence; live authority remains blocked"
    )
    assert m106[-1] == "WAITING_FOR_EXTERNAL_VALIDATION"

    for marker in ("M106-CP01", "M106-CP02", "M106-CP03"):
        assert marker in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "No live transport" in adr

    if directive["selected_blocker_status"] in {
        "blocked_by_real_artifact_review",
        "blocked_by_wing_fader_rollback_proof",
    }:
        assert directive["selected_action_source"] in {
            "WING Read-Only Decision Review",
            "WING Supervised Fader Rollback Proof",
        }
        assert directive["pre_hardware_intake_hardening_loop_guard"] is False
        assert (
            directive["allowed_autonomous_scope"]
            == "external_validation_handoff_or_control_plane_only"
        )
    else:
        assert directive["pre_hardware_intake_hardening_loop_guard"] is True
        assert directive["pre_hardware_intake_guard_recent_streak_count"] >= 10
        assert directive["pre_hardware_intake_guard_recent_subjects"][0] == (
            "Guard WING readonly capture time order"
        )
        assert (
            directive["allowed_autonomous_scope"]
            == "external_validation_wait_or_named_gap_only"
        )
        assert any(
            "No further generic WING read-only intake guard milestone after M104" in item
            for item in directive["forbidden_scope"]
        )

    assert "M106 Project Director post-M105 intake loop persistence is now covered" in current
    assert "M106 Project Director post-M105 intake loop persistence rail is now" in report
    assert "direction-guard commits sit above the WING read-only streak" in current
    assert "direction-guard commits sit above the WING read-only streak" in report


def test_m107_project_director_intake_loop_guard_lookback_floor_persists():
    state = json.loads(AUTOPILOT_STATE.read_text(encoding="utf-8"))
    roadmap = ROADMAP_DOC.read_text(encoding="utf-8")
    current = CURRENT_STATE_DOC.read_text(encoding="utf-8")
    report = AUTOPILOT_LAST_REPORT.read_text(encoding="utf-8")
    plan = M107_PLAN.read_text(encoding="utf-8")
    adr = M107_ADR.read_text(encoding="utf-8")
    director = load_project_director_module()

    directive = director.build_directive(
        project_root=PROJECT_ROOT,
        lookback_commits=12,
        report_limit=1,
    )

    assert state["program_status"] == "WAITING_FOR_EXTERNAL_VALIDATION"
    assert state["last_validation_summary"]["hardware_evidence"] is False
    assert state["last_validation_summary"]["authority"]["live_authority_granted"] is False
    assert state["last_validation_summary"]["authority"]["readiness_closure_allowed"] is False
    assert state["last_validation_summary"]["authority"]["contacts_backend"] is False
    assert state["last_validation_summary"]["authority"]["contacts_wing"] is False
    assert state["last_validation_summary"]["authority"]["live_write"] is False

    m107 = markdown_table_row(roadmap, "M107")
    assert m107[1] == "Project Director intake loop lookback floor"
    assert m107[2] == (
        "Project Director keeps the post-M104 intake-loop guard active even "
        "when additional direction-guard commits would otherwise push the WING "
        "read-only streak outside the requested lookback window"
    )
    assert m107[3] == "M001"
    assert m107[4] == "critical autopilot direction"
    assert m107[5] == (
        "UNIT_VERIFIED intake-loop lookback floor; live authority remains blocked"
    )
    assert m107[-1] == "WAITING_FOR_EXTERNAL_VALIDATION"

    for marker in ("M107-CP01", "M107-CP02", "M107-CP03"):
        assert marker in plan
    for heading in (
        "## Context",
        "## Options Considered",
        "## Decision",
        "## Why This Won",
        "## Rejected Alternatives",
        "## Implementation Plan",
        "## Test Plan",
        "## Risks and Rollback",
    ):
        assert heading in adr
    assert "No live transport" in adr

    assert directive["pre_hardware_intake_guard_requested_lookback_commits"] == 12
    assert directive["pre_hardware_intake_guard_effective_lookback_commits"] >= (
        directive["pre_hardware_intake_guard_lookback_floor"]
    )
    assert directive["pre_hardware_intake_guard_lookback_floor"] == 20
    if directive["selected_blocker_status"] in {
        "blocked_by_real_artifact_review",
        "blocked_by_wing_fader_rollback_proof",
    }:
        assert directive["selected_action_source"] in {
            "WING Read-Only Decision Review",
            "WING Supervised Fader Rollback Proof",
        }
        assert directive["pre_hardware_intake_hardening_loop_guard"] is False
        assert (
            directive["allowed_autonomous_scope"]
            == "external_validation_handoff_or_control_plane_only"
        )
    else:
        assert directive["pre_hardware_intake_hardening_loop_guard"] is True
        assert directive["pre_hardware_intake_guard_recent_streak_count"] >= 10
        assert (
            directive["allowed_autonomous_scope"]
            == "external_validation_wait_or_named_gap_only"
        )
        assert any(
            "Do not keep improving the same WING read-only intake function" in item
            for item in directive["forbidden_scope"]
        )

    assert "M107 Project Director intake loop lookback floor is now covered" in current
    assert "M107 Project Director intake loop lookback floor rail is now" in report
    assert "requested lookback window" in current
    assert "requested lookback window" in report


def test_mvp_doc_declares_scope_and_stop_rules():
    text = MVP_DOC.read_text(encoding="utf-8")

    assert "Supervised WING Rack Proof" in text
    assert "Full autonomous live mixing" in text
    assert "Stop-The-Line Rules" in text
    assert ".ai/project-layer/wing_mvp_blockers.md" in text
    assert "Docs/runbooks/WING_READONLY_PREFLIGHT.md" in text
    assert "tools/wing_readonly_preflight_runner.py" in text
    assert "tools/review_wing_readonly_preflight_artifacts.py" in text
    assert "tools/no_wing_release_check.py" in text
    assert "readiness candidates are not enough" in text
    assert "backend_port_prestart_gate" in text
    assert "expected_backend_port_prestart_gate" in text
    assert "backend_port_prestart_gate.grants_live_authority: false" in text
    assert (
        "backend_port_prestart_gate.must_pass_before_backend_wing_session: true"
        in text
    )
    assert "backend_port_prestart_gate.operator_sequence_step_index: 0" in text
    assert "backend_port_prestart_gate.operator_sequence_step_must_be_first: true" in text
    assert "backend_port_prestart_gate.runs_command: prestart_backend_port_check" in text
    assert "backend_port_prestart_gate.required_markers" in text
    assert "Before starting the intended backend/WING session" in text
    assert "backend_port_prestart_summary_sha256_matches_plan: true" in text
    assert "backend_port_prestart_summary_sha256_available: true" in text
    assert (
        "backend_port_prestart_summary_sha256_available_matches_digest: true"
        in text
    )
    assert "expected_operator_sequence_count: 12" in text
    assert "operator_sequence_count_matches_contract: true" in text
    for marker in OPERATOR_SEQUENCE_SHA_MARKERS:
        assert marker in text
    for marker in OPERATOR_SEQUENCE_CONTRACT_MARKERS:
        assert marker in text
    for marker in DECLARED_OPERATOR_SEQUENCE_COUNT_MARKERS:
        assert marker in text
    assert "expected_mvp_blocker_anchor_digest_review_step_index: 10" in text
    assert "mvp_blocker_anchor_digest_review_step_index_matches_contract: true" in text
    assert (
        "mvp_blocker_anchor_digest_review_step_index_matches_operator_sequence: true"
        in text
    )
    assert (
        "expected_operator_sequence_mvp_blocker_anchor_digest_review_step_index: 10"
        in text
    )
    assert (
        "operator_sequence_mvp_blocker_anchor_digest_review_step_index_matches_contract: true"
        in text
    )
    assert (
        "expected_operator_sequence_supervised_prestart_gate_step_index: 11"
        in text
    )
    assert (
        "operator_sequence_supervised_prestart_gate_step_index_matches_contract: true"
        in text
    )
    assert "backend_port_prestart_check_command" in text
    assert "backend_port_prestart_check_command_text" in text
    assert "PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick" in text
    assert "backend_port_prestart_check_failure_detail_fields" in text
    assert "listener_processes" in text
    assert "listener_process_error" in text
    assert "safe_to_stop_listener_by_gate" in text
    assert "operator_next_step" in text
    for marker in ENVIRONMENT_BLOCKED_FRONTEND_MARKERS:
        assert marker in text
    assert "listener_confirmation_template_path" in text
    assert "listener_confirmation_template_acceptance_status" in text
    assert "listener_confirmation_template_status: pending_operator_edit" in text
    assert "listener_confirmation_template_safe_to_stop_listener_by_gate: false" in text
    assert "listener_confirmation_template_live_authority_granted: false" in text
    assert "listener_confirmation_template_accepted_by_template: false" in text
    assert "listener_confirmation_template_execution_allowed_by_template: false" in text
    assert (
        "listener_confirmation_template_readiness_closure_allowed_by_template: false"
        in text
    )
    assert "listener_confirmation_template_operator_placeholder_fields" in text
    assert "listener_confirmation_template_required_edits" in text
    assert "set template_status to operator_reviewed_and_edited" in text
    assert "set accepted_by_template to true" in text
    assert "REPLACE_WITH_*" in text
    assert "template_status: operator_reviewed_and_edited" in text
    assert "accepted_by_template: true" in text
    assert "manual_review_required_before_readiness_update: true" in text
    assert "readiness_matrix_update_allowed_by_packet: false" in text
    assert "readiness_closure_allowed_by_packet: false" in text
    assert "unsafe packet values" in text
    assert "unsafe safety flags" in text
    assert "full_autonomous_ready" in text
    assert "mutates_console_state" in text
    assert "read_only: true" in text
    assert "live_write: false" in text
    assert "mutates_console_state: false" in text
    assert "queries_console: false" in text
    assert "executes_tests: false" in text
    assert "operator_required: true" in text
    assert "expected_plan_required_for_real_run: true" in text
    assert "backend_port_prestart_summary_required: true" in text
    assert "backend_port_prestart_summary_path_placeholder" in text
    assert "preview_command_prestart_summary" in text
    assert "preview_command_prestart_summary_matches_plan: true" in text
    assert "validate_inputs_command_prestart_summary" in text
    assert "validate_inputs_command_prestart_summary_matches_plan: true" in text
    assert "capture_command_prestart_summary" in text
    assert "capture_command_prestart_summary_matches_plan: true" in text
    assert "backend_port_prestart_summary_type: no_wing_release_check" in text
    assert "backend_port_prestart_summary_status: passed" in text
    assert (
        "backend_port_prestart_summary_connection_policy: "
        "no_hardware_no_network_no_live_writes"
    ) in text
    assert "backend_port_prestart_summary_read_only: true" in text
    assert "backend_port_prestart_summary_live_write: false" in text
    assert "backend_port_prestart_summary_contacts_backend: false" in text
    assert "backend_port_prestart_summary_contacts_wing: false" in text
    assert "backend_port_prestart_summary_mutates_console_state: false" in text
    assert "backend_port_prestart_summary_queries_console: false" in text
    assert "backend_port_prestart_summary_executes_tests: false" in text
    assert "backend_port_prestart_check_status: passed" in text
    for marker in (
        "backend_port_prestart_summary_type_matches_plan: true",
        "backend_port_prestart_summary_status_matches_plan: true",
        "backend_port_prestart_summary_connection_policy_matches_plan: true",
        "backend_port_prestart_summary_read_only_matches_plan: true",
        "backend_port_prestart_summary_live_write_matches_plan: true",
        "backend_port_prestart_summary_contacts_backend_matches_plan: true",
        "backend_port_prestart_summary_contacts_wing_matches_plan: true",
        "backend_port_prestart_summary_mutates_console_state_matches_plan: true",
        "backend_port_prestart_summary_queries_console_matches_plan: true",
        "backend_port_prestart_summary_executes_tests_matches_plan: true",
        "plan_safe_to_stop_listener_by_gate: false",
        "plan_live_authority_granted: false",
        "plan_backend_port_prestart_gate_execution_allowed: false",
        "plan_backend_port_prestart_gate_readiness_closure_allowed: false",
        "plan_backend_port_prestart_gate_live_authority_granted: false",
        "backend_port_prestart_check_status_matches_plan: true",
        "safe_to_stop_listener_by_gate_matches_plan: true",
        "live_authority_granted_matches_plan: true",
        "backend_port_prestart_gate_execution_allowed_matches_plan: true",
        "backend_port_prestart_gate_readiness_closure_allowed_matches_plan: true",
        "backend_port_prestart_gate_live_authority_granted_matches_plan: true",
        "declared_safe_to_stop_listener_by_gate_matches_plan: true",
        "declared_live_authority_granted_matches_plan: true",
        "declared_backend_port_prestart_gate_execution_allowed_matches_plan: true",
        "declared_backend_port_prestart_gate_readiness_closure_allowed_matches_plan: true",
        "declared_backend_port_prestart_gate_live_authority_granted_matches_plan: true",
    ):
        assert marker in text
    assert "physical_preflight_requires_existing_backend_connection: true" in text
    assert "backend_port_prestart_check_required: true" in text
    assert "backend_port_prestart_check_must_pass_before_starting_backend: true" in text
    assert "backend_port_prestart_gate_execution_allowed: false" in text
    assert "backend_port_prestart_gate_readiness_closure_allowed: false" in text
    assert "backend_port_prestart_gate_live_authority_granted: false" in text
    assert "execution_allowed_by_packet: false" in text
    assert "live_authority_granted: false" in text
    assert "full_autonomous_ready: false" in text
    assert (
        "next_manual_artifact_required: real wing_readonly_preflight.v1 summary.json"
        in text
    )
    assert "expected_artifacts" in text
    assert "expected_artifact_globs" in text
    assert "plan_expected_artifacts" in text
    assert "expected_artifacts_match_plan: true" in text
    assert "summary_glob_matches_plan: true" in text
    assert "event_log_glob_matches_plan: true" in text
    assert "backend_port_prestart_check_operator_sequence_step" in text
    assert "prestart_backend_port_check" in text
    assert "unexpected listeners" in text
    assert "supervised_prestart_gate" in text
    assert "expected_supervised_prestart_gate" in text
    assert "required_supervised_prestart_gate_markers" in text
    assert "required_supervised_prestart_gate_marker_count: 9" in text
    assert "expected_required_supervised_prestart_gate_marker_count: 9" in text
    assert "supervised_prestart_gate_required_marker_count: 9" in text
    assert "expected_supervised_prestart_gate_required_marker_count: 9" in text
    assert "supervised_prestart_gate.grants_live_authority: false" in text
    assert "supervised_prestart_gate.operator_sequence_step_index: 11" in text
    assert "supervised_prestart_gate.operator_sequence_step_must_be_final: true" in text
    assert "valid accepted-candidate contact trace" in text
    assert "valid accepted-candidate expected-artifact trace" in text
    assert (
        "accepted_readonly_preflight_candidate_expected_artifact_trace_required_before_supervised_proof"
        in text
    )
    assert (
        "accepted_readonly_preflight_candidate_contact_trace_required_before_supervised_proof"
        in text
    )
    assert "operator_sequence_count: 12" in text
    for marker in OPERATOR_SEQUENCE_SHA_MARKERS:
        assert marker in text
    for marker in OPERATOR_SEQUENCE_CONTRACT_MARKERS:
        assert marker in text
    for marker in DECLARED_OPERATOR_SEQUENCE_COUNT_MARKERS:
        assert marker in text
    assert "mvp_blocker_anchor_digest_review_step_index: 10" in text
    assert "expected_mvp_blocker_anchor_digest_review_step_index: 10" in text
    assert "mvp_blocker_anchor_digest_review_step_index_matches_contract: true" in text
    assert (
        "mvp_blocker_anchor_digest_review_step_index_matches_operator_sequence: true"
        in text
    )
    assert "operator_sequence_mvp_blocker_anchor_digest_review_step_index: 10" in text
    assert (
        "expected_operator_sequence_mvp_blocker_anchor_digest_review_step_index: 10"
        in text
    )
    assert (
        "operator_sequence_mvp_blocker_anchor_digest_review_step_index_matches_contract: true"
        in text
    )
    assert "operator_sequence_supervised_prestart_gate_step_index: 11" in text
    assert (
        "supervised_prestart_gate.requires_operator_decision_status: accepted_decision_ready"
        in text
    )
    assert (
        "supervised_prestart_gate.unlocks_runbook: Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md"
        in text
    )
    assert (
        "supervised_prestart_gate.requires_accepted_candidate_contact_trace: true"
        in text
    )
    assert (
        "supervised_prestart_gate_requires_accepted_candidate_contact_trace: true"
        in text
    )
    assert "grants_live_authority: true" in text
    assert "Docs/CORRECTION_METHODS_ROADMAP.md" in text


def test_blocker_table_has_exact_nine_rows_and_no_closed_baseline():
    progress = load_progress_module()
    rows = progress.parse_blocker_table(BLOCKERS_DOC)

    assert len(rows) == 9
    assert rows[0]["blocker"] == "WING read-only connection proof"
    assert rows[0]["status"] == "tooling_ready_waits_real_wing"
    missing_index_reports = [
        report_name
        for report_name in current_wing_mvp_20260614_index_report_names()
        if report_name not in rows[0]["evidence"]
    ]
    assert missing_index_reports == []
    assert "tools/no_wing_release_check.py" in rows[0]["evidence"]
    assert (
        "AUTOMIXER_WING_READONLY_PREFLIGHT_REVIEW_RELEASE_GATE_20260607.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_READONLY_PREFLIGHT_OPERATOR_CONFIRMATION_20260607.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_READONLY_ARTIFACT_CONTRACT_CONSTANTS_20260613.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_DECISION_REVIEW_PRESTART_READY_GATE_20260613.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_DECISION_REVIEW_PRESTART_SUMMARY_FIELDS_20260614.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_DECISION_REVIEW_PRESTART_BOOL_RENDERING_20260614.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_DIRECTION_GUARD_SKIP_SELF_REPORTS_20260614.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_HANDOFF_PRESTART_SUMMARY_SHA_BINDING_20260614.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_HANDOFF_PRESTART_SUMMARY_SHA_AVAILABILITY_20260614.md"
        in rows[0]["evidence"]
    )
    for report_name in (
        "AUTOMIXER_WING_MVP_HANDOFF_GATE_MATCH_VERDICTS_20260614.md",
        "AUTOMIXER_WING_MVP_HANDOFF_PRESTART_SUMMARY_CONTRACT_20260614.md",
        "AUTOMIXER_WING_MVP_CURRENT_FIRST_PRESTART_SUMMARY_MATCH_FLAGS_20260614.md",
    ):
        assert report_name in rows[0]["evidence"]
    assert (
        "AUTOMIXER_WING_MVP_HANDOFF_OPERATOR_SEQUENCE_CARDINALITY_20260614.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_HANDOFF_ANCHOR_DIGEST_REVIEW_STEP_INDEX_20260614.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_HANDOFF_DECLARED_ANCHOR_DIGEST_REVIEW_INDEX_20260614.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_HANDOFF_DECLARED_OPERATOR_SEQUENCE_COUNT_20260614.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_HANDOFF_OPERATOR_SEQUENCE_SHA256_20260614.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_HANDOFF_OPERATOR_SEQUENCE_CONTRACT_20260614.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_PROOF_PLAN_PRESTART_VIOLATIONS_MAP_SUMMARY_20260614.md"
        in rows[0]["evidence"]
    )
    assert "backend_port_prestart_summary_sha256_matches_plan: true" in rows[0][
        "next action"
    ]
    assert "backend_port_prestart_summary_sha256_available: true" in rows[0][
        "next action"
    ]
    assert "backend_port_prestart_summary_sha256_available_matches_digest: true" in rows[
        0
    ]["next action"]
    assert "expected_operator_sequence_count: 12" in rows[0]["next action"]
    assert "operator_sequence_count_matches_contract: true" in rows[0][
        "next action"
    ]
    for marker in OPERATOR_SEQUENCE_SHA_MARKERS:
        assert marker in rows[0]["next action"]
    for marker in OPERATOR_SEQUENCE_CONTRACT_MARKERS:
        assert marker in rows[0]["next action"]
    for marker in DECLARED_OPERATOR_SEQUENCE_COUNT_MARKERS:
        assert marker in rows[0]["next action"]
    assert "expected_mvp_blocker_anchor_digest_review_step_index: 10" in rows[0][
        "next action"
    ]
    assert "mvp_blocker_anchor_digest_review_step_index_matches_contract: true" in rows[
        0
    ]["next action"]
    assert "mvp_blocker_anchor_digest_review_step_index_matches_operator_sequence: true" in rows[
        0
    ]["next action"]
    assert (
        "expected_operator_sequence_mvp_blocker_anchor_digest_review_step_index: 10"
        in rows[0]["next action"]
    )
    assert (
        "operator_sequence_mvp_blocker_anchor_digest_review_step_index_matches_contract: true"
        in rows[0]["next action"]
    )
    assert (
        "expected_operator_sequence_supervised_prestart_gate_step_index: 11"
        in rows[0]["next action"]
    )
    assert (
        "operator_sequence_supervised_prestart_gate_step_index_matches_contract: true"
        in rows[0]["next action"]
    )
    assert (
        "AUTOMIXER_WING_MVP_BACKEND_PRESTART_LISTENER_PROBE_ERROR_FIELD_20260613.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_NOWING_QUICK_BUSY_PORT_FAIL_CLOSED_20260613.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_BACKEND_LISTENER_CONFIRMATION_ARTIFACT_20260613.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_BACKEND_LISTENER_CONFIRMATION_TEMPLATE_20260613.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_BACKEND_LISTENER_TEMPLATE_EDIT_GATE_20260613.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_BACKEND_LISTENER_TEMPLATE_PENDING_SUMMARY_FIELDS_20260613.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_BACKEND_LISTENER_TEMPLATE_AUTHORITY_SUMMARY_FIELDS_20260613.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_BACKEND_PRESTART_TEMPLATE_FAILURE_DETAIL_FIELDS_20260613.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_BACKEND_PRESTART_TEMPLATE_REVIEW_FIELDS_20260613.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_BUSY_PORT_PRESTART_DETAIL_FIELDS_SUMMARY_20260613.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_BUSY_PORT_PRESTART_COMMAND_SUMMARY_20260613.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_BUSY_PORT_PRESTART_REQUIRED_FLAGS_SUMMARY_20260613.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_BUSY_PORT_PRESTART_GATE_AUTHORITY_SUMMARY_20260613.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_READONLY_PRESTART_SUMMARY_BINDING_20260613.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_HANDOFF_PRESTART_SUMMARY_PLACEHOLDER_20260613.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_HANDOFF_EXPECTED_ARTIFACTS_CONTRACT_20260613.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_HANDOFF_EXPECTED_ARTIFACTS_NEXT_ACTION_20260613.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_HANDOFF_DENIAL_FLAG_SUMMARY_TRUTH_20260614.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_HANDOFF_SAFETY_FLAG_SUMMARY_TRUTH_20260614.md"
        in rows[0]["evidence"]
    )
    assert (
        "AUTOMIXER_WING_MVP_HANDOFF_BACKEND_PRESTART_COMMAND_SUMMARY_20260614.md"
        in rows[0]["evidence"]
    )
    assert "accepted-candidate prestart trace" in rows[0]["next action"]
    assert "backend_port_prestart_gate" in rows[0]["next action"]
    assert "expected_backend_port_prestart_gate" in rows[0]["next action"]
    assert "backend_port_prestart_gate.grants_live_authority: false" in rows[0][
        "next action"
    ]
    assert "backend_port_prestart_gate.must_pass_before_backend_wing_session: true" in rows[
        0
    ]["next action"]
    assert "backend_port_prestart_gate.operator_sequence_step_index: 0" in rows[
        0
    ]["next action"]
    assert (
        "backend_port_prestart_gate.operator_sequence_step_must_be_first: true"
        in rows[0]["next action"]
    )
    assert (
        "backend_port_prestart_gate.runs_command: prestart_backend_port_check"
        in rows[0]["next action"]
    )
    assert "backend_port_prestart_gate.required_markers" in rows[0]["next action"]
    assert "Before starting the intended backend/WING session" in rows[0][
        "next action"
    ]
    assert "backend_port_prestart_check_command" in rows[0]["next action"]
    assert "PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick" in rows[
        0
    ]["next action"]
    assert (
        "failed busy-port no-WING summary and no-WING handoff summary must both show"
        in rows[0]["next action"]
    )
    assert "backend_port_prestart_check_required: true" in rows[0]["next action"]
    assert (
        "backend_port_prestart_check_must_pass_before_starting_backend: true"
        in rows[0]["next action"]
    )
    assert "backend_port_prestart_gate_execution_allowed: false" in rows[0][
        "next action"
    ]
    assert "backend_port_prestart_gate_readiness_closure_allowed: false" in rows[0][
        "next action"
    ]
    assert "backend_port_prestart_gate_live_authority_granted: false" in rows[0][
        "next action"
    ]
    assert "backend_port_prestart_check_failure_detail_fields" in rows[0][
        "next action"
    ]
    assert "failed busy-port no-WING summary" in rows[0]["next action"]
    assert "no-WING handoff summary" in rows[0]["next action"]
    assert "listener_processes" in rows[0]["next action"]
    assert "listener_process_error" in rows[0]["next action"]
    assert "safe_to_stop_listener_by_gate" in rows[0]["next action"]
    assert "operator_next_step" in rows[0]["next action"]
    assert "listener_confirmation_template_path" in rows[0]["next action"]
    assert "listener_confirmation_template_acceptance_status" in rows[0][
        "next action"
    ]
    assert "listener_confirmation_template_status: pending_operator_edit" in rows[0][
        "next action"
    ]
    assert "listener_confirmation_template_safe_to_stop_listener_by_gate: false" in rows[
        0
    ]["next action"]
    assert "listener_confirmation_template_live_authority_granted: false" in rows[0][
        "next action"
    ]
    assert "listener_confirmation_template_accepted_by_template: false" in rows[0][
        "next action"
    ]
    assert (
        "listener_confirmation_template_execution_allowed_by_template: false"
        in rows[0]["next action"]
    )
    assert (
        "listener_confirmation_template_readiness_closure_allowed_by_template: false"
        in rows[0]["next action"]
    )
    assert "listener_confirmation_template_operator_placeholder_fields" in rows[0][
        "next action"
    ]
    assert "listener_confirmation_template_required_edits" in rows[0]["next action"]
    assert "set template_status to operator_reviewed_and_edited" in rows[0][
        "next action"
    ]
    assert "set accepted_by_template to true" in rows[0]["next action"]
    assert "REPLACE_WITH_*" in rows[0]["next action"]
    assert "template_status: operator_reviewed_and_edited" in rows[0][
        "next action"
    ]
    assert "accepted_by_template: true" in rows[0]["next action"]
    assert "manual_review_required_before_readiness_update: true" in rows[0][
        "next action"
    ]
    assert "readiness_matrix_update_allowed_by_packet: false" in rows[0][
        "next action"
    ]
    assert "readiness_closure_allowed_by_packet: false" in rows[0][
        "next action"
    ]
    assert "read_only: true" in rows[0]["next action"]
    assert "live_write: false" in rows[0]["next action"]
    assert "mutates_console_state: false" in rows[0]["next action"]
    assert "queries_console: false" in rows[0]["next action"]
    assert "executes_tests: false" in rows[0]["next action"]
    assert "operator_required: true" in rows[0]["next action"]
    assert "expected_plan_required_for_real_run: true" in rows[0]["next action"]
    assert "backend_port_prestart_summary_required: true" in rows[0][
        "next action"
    ]
    assert "backend_port_prestart_summary_type: no_wing_release_check" in rows[0][
        "next action"
    ]
    assert (
        "backend_port_prestart_summary_connection_policy: "
        "no_hardware_no_network_no_live_writes"
    ) in rows[0]["next action"]
    assert "backend_port_prestart_summary_path_placeholder" in rows[0][
        "next action"
    ]
    assert "preview_command_prestart_summary" in rows[0]["next action"]
    assert "preview_command_prestart_summary_matches_plan: true" in rows[0][
        "next action"
    ]
    assert "capture_command_prestart_summary" in rows[0]["next action"]
    assert "capture_command_prestart_summary_matches_plan: true" in rows[0][
        "next action"
    ]
    assert "backend_port_prestart_summary_status: passed" in rows[0][
        "next action"
    ]
    assert "backend_port_prestart_summary_read_only: true" in rows[0][
        "next action"
    ]
    assert "backend_port_prestart_summary_live_write: false" in rows[0][
        "next action"
    ]
    assert "backend_port_prestart_summary_contacts_backend: false" in rows[0][
        "next action"
    ]
    assert "backend_port_prestart_summary_contacts_wing: false" in rows[0][
        "next action"
    ]
    assert "backend_port_prestart_summary_mutates_console_state: false" in rows[0][
        "next action"
    ]
    assert "backend_port_prestart_summary_queries_console: false" in rows[0][
        "next action"
    ]
    assert "backend_port_prestart_summary_executes_tests: false" in rows[0][
        "next action"
    ]
    assert "backend_port_prestart_check_status: passed" in rows[0]["next action"]
    assert (
        "physical_preflight_requires_existing_backend_connection: true"
        in rows[0]["next action"]
    )
    assert "backend_port_prestart_check_required: true" in rows[0]["next action"]
    assert (
        "backend_port_prestart_check_must_pass_before_starting_backend: true"
        in rows[0]["next action"]
    )
    assert "backend_port_prestart_gate_execution_allowed: false" in rows[0][
        "next action"
    ]
    assert "backend_port_prestart_gate_readiness_closure_allowed: false" in rows[0][
        "next action"
    ]
    assert "backend_port_prestart_gate_live_authority_granted: false" in rows[0][
        "next action"
    ]
    assert "execution_allowed_by_packet: false" in rows[0]["next action"]
    assert "live_authority_granted: false" in rows[0]["next action"]
    assert "full_autonomous_ready: false" in rows[0]["next action"]
    assert (
        "next_manual_artifact_required: real wing_readonly_preflight.v1 summary.json"
        in rows[0]["next action"]
    )
    assert "expected_artifacts" in rows[0]["next action"]
    assert "expected_artifact_globs" in rows[0]["next action"]
    assert "plan_expected_artifacts" in rows[0]["next action"]
    assert "expected_artifacts_match_plan: true" in rows[0]["next action"]
    assert "summary_glob_matches_plan: true" in rows[0]["next action"]
    assert "event_log_glob_matches_plan: true" in rows[0]["next action"]
    for marker in (
        "preview_command_artifact_root_matches_packet: true",
        "preview_command_plan_output_matches_packet: true",
        "validate_inputs_command_uses_validate_run_inputs_only: true",
        "validate_inputs_command_uses_plan_only: false",
        "validate_inputs_command_artifact_root_matches_packet: true",
        "validate_inputs_command_expected_plan_matches_packet: true",
        "capture_command_artifact_root_matches_packet: true",
        "capture_command_expected_plan_matches_packet: true",
        "review_command_artifact_root_matches_packet: true",
        "decision_review_command_artifact_root_matches_packet: true",
        "decision_review_command_decision_root_matches_packet: true",
    ):
        assert marker in rows[0]["next action"]
    assert "backend_port_prestart_check_operator_sequence_step_index: 0" in rows[0][
        "next action"
    ]
    assert "backend_port_prestart_check_operator_sequence_step" in rows[0][
        "next action"
    ]
    assert "prestart_backend_port_check" in rows[0]["next action"]
    assert "unexpected listeners" in rows[0]["next action"]
    assert "backend_port_prestart_check_must_be_first_operator_step: true" in rows[
        0
    ]["next action"]
    assert "backend_port_prestart_check_operator_sequence_step_is_first: true" in rows[
        0
    ]["next action"]
    assert (
        "backend_connection_confirmation_step_index: 3"
        in rows[0]["next action"]
    )
    assert (
        "operator_sequence_has_backend_connection_confirmation: true"
        in rows[0]["next action"]
    )
    assert (
        "operator_sequence_backend_connection_confirmation_step_index: 3"
        in rows[0]["next action"]
    )
    assert (
        "operator_sequence_backend_connection_confirmation_before_capture: true"
        in rows[0]["next action"]
    )
    assert "operator_sequence_backend_connection_confirmation_step" in rows[0][
        "next action"
    ]
    assert "existing backend connection" in rows[0]["next action"]
    assert "intended WING Rack" in rows[0]["next action"]
    assert "read-only inspection" in rows[0]["next action"]
    assert (
        "operator_sequence_validate_readonly_preflight_inputs_step_index: 5"
        in rows[0]["next action"]
    )
    assert (
        "operator_sequence_validate_readonly_preflight_inputs_step_matches_contract: true"
        in rows[0]["next action"]
    )
    assert (
        "operator_sequence_capture_readonly_preflight_step_index: 6"
        in rows[0]["next action"]
    )
    assert "mvp_blocker_anchor" in rows[0]["next action"]
    assert "mvp_blocker_anchor_matches_current_first_open: true" in rows[0][
        "next action"
    ]
    assert "mvp_blocker_anchor_sha256_matches_current_first_open: true" in rows[0][
        "next action"
    ]
    assert "mvp_blocker_anchor.packet_does_not_close_blocker: true" in rows[0][
        "next action"
    ]
    assert "operator_sequence_has_mvp_blocker_anchor_digest_review: true" in rows[0][
        "next action"
    ]
    assert "operator_sequence_count: 12" in rows[0]["next action"]
    for marker in OPERATOR_SEQUENCE_SHA_MARKERS:
        assert marker in rows[0]["next action"]
    for marker in OPERATOR_SEQUENCE_CONTRACT_MARKERS:
        assert marker in rows[0]["next action"]
    for marker in DECLARED_OPERATOR_SEQUENCE_COUNT_MARKERS:
        assert marker in rows[0]["next action"]
    assert "mvp_blocker_anchor_digest_review_step_index: 10" in rows[0][
        "next action"
    ]
    assert "expected_mvp_blocker_anchor_digest_review_step_index: 10" in rows[0][
        "next action"
    ]
    assert "mvp_blocker_anchor_digest_review_step_index_matches_contract: true" in rows[
        0
    ]["next action"]
    assert "mvp_blocker_anchor_digest_review_step_index_matches_operator_sequence: true" in rows[
        0
    ]["next action"]
    assert "operator_sequence_mvp_blocker_anchor_digest_review_step" in rows[0][
        "next action"
    ]
    assert "operator_sequence_mvp_blocker_anchor_digest_review_step_index: 10" in rows[
        0
    ]["next action"]
    assert (
        "expected_operator_sequence_mvp_blocker_anchor_digest_review_step_index: 10"
        in rows[0]["next action"]
    )
    assert (
        "operator_sequence_mvp_blocker_anchor_digest_review_step_index_matches_contract: true"
        in rows[0]["next action"]
    )
    assert "mvp_blocker_anchor_sha256" in rows[0]["next action"]
    assert "current first open blocker" in rows[0]["next action"]
    assert "operator_sequence_supervised_prestart_gate_is_final: true" in rows[0][
        "next action"
    ]
    assert "operator_sequence_has_supervised_prestart_gate: true" in rows[0][
        "next action"
    ]
    assert "operator_sequence_supervised_prestart_gate_step" in rows[0][
        "next action"
    ]
    assert "operator_sequence_supervised_prestart_gate_step_index: 11" in rows[0][
        "next action"
    ]
    assert "required_supervised_prestart_gate_markers" in rows[0]["next action"]
    assert "accepted_decision_ready" in rows[0]["next action"]
    assert "valid accepted-candidate prestart trace" in rows[0]["next action"]
    assert "valid accepted-candidate verifier WebSocket proof" in rows[0][
        "next action"
    ]
    assert "valid accepted-candidate artifact-root trace" in rows[0]["next action"]
    assert "valid accepted-candidate request-sequence trace" in rows[0][
        "next action"
    ]
    assert "expected-artifact-bound accepted source id" in rows[0]["next action"]
    assert "WING_SUPERVISED_FADER_GAIN_PROOF" in rows[0]["next action"]
    assert "supervised_prestart_gate" in rows[0]["next action"]
    assert "expected_supervised_prestart_gate" in rows[0]["next action"]
    assert "supervised_prestart_gate.grants_live_authority: false" in rows[0][
        "next action"
    ]
    assert "supervised_prestart_gate.operator_sequence_step_index: 11" in rows[0][
        "next action"
    ]
    assert (
        "supervised_prestart_gate.operator_sequence_step_must_be_final: true"
        in rows[0]["next action"]
    )
    assert (
        "supervised_prestart_gate.requires_operator_decision_status: accepted_decision_ready"
        in rows[0]["next action"]
    )
    assert (
        "supervised_prestart_gate.unlocks_runbook: Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md"
        in rows[0]["next action"]
    )
    assert (
        "supervised_prestart_gate.requires_accepted_candidate_prestart_trace: true"
        in rows[0]["next action"]
    )
    assert (
        "supervised_prestart_gate.requires_accepted_candidate_contact_trace: true"
        in rows[0]["next action"]
    )
    assert (
        "accepted_readonly_preflight_candidate_prestart_contract_required_before_supervised_proof"
        in rows[0]["next action"]
    )
    assert (
        "accepted_readonly_preflight_candidate_verifier_ws_proof_required_before_supervised_proof"
        in rows[0]["next action"]
    )
    assert (
        "accepted_readonly_preflight_candidate_request_sequence_required_before_supervised_proof"
        in rows[0]["next action"]
    )
    assert (
        "accepted_readonly_preflight_candidate_expected_artifact_trace_required_before_supervised_proof"
        in rows[0]["next action"]
    )
    assert (
        "accepted_readonly_preflight_candidate_contact_trace_required_before_supervised_proof"
        in rows[0]["next action"]
    )
    assert "valid accepted-candidate contact trace" in rows[0]["next action"]
    fader_row = next(row for row in rows if row["blocker"] == "Supervised fader write proof")
    gain_row = next(
        row for row in rows if row["blocker"] == "Supervised gain-down / trim proof"
    )
    for row in (fader_row, gain_row):
        assert row["status"] == "blocked_by_readonly_preflight"
        assert "accepted decision JSON" in row["next action"]
        assert "backend_port_prestart_gate" in row["next action"]
        assert "expected_backend_port_prestart_gate" in row["next action"]
        assert "without live authority" in row["next action"]
        assert "final handoff operator-sequence step" in row["next action"]
        assert "valid accepted-candidate prestart trace" in row["next action"]
        assert "valid accepted-candidate verifier WebSocket proof" in row[
            "next action"
        ]
        assert "valid accepted-candidate artifact-root trace" in row["next action"]
        assert "valid accepted-candidate request-sequence trace" in row["next action"]
        assert "valid accepted-candidate contact trace" in row["next action"]
        assert (
            "valid accepted-candidate expected-artifact trace" in row["next action"]
        )
        assert "expected-artifact-bound accepted source id" in row["next action"]
        assert "supervised_prestart_gate" in row["next action"]
        assert "expected_supervised_prestart_gate" in row["next action"]
    assert all(row["status"] != "closed" for row in rows)
    assert progress.first_open_blocker(rows)["blocker"] == "WING read-only connection proof"


def test_runbooks_keep_readonly_before_supervised_execute():
    readonly = READONLY_RUNBOOK.read_text(encoding="utf-8")
    supervised = SUPERVISED_RUNBOOK.read_text(encoding="utf-8")
    missing_sidecar_markers = [
        marker
        for marker in (
            release_check.PROJECT_DIRECTOR_SELECTED_ACTION_MANIFEST_SIDECAR_REQUIRED_MARKERS
        )
        if marker not in readonly
    ]
    assert missing_sidecar_markers == []
    missing_value_gate_self_check_markers = [
        marker
        for marker in (
            release_check.PROJECT_DIRECTOR_SELECTED_ACTION_VALUE_GATE_SELF_CHECK_REQUIRED_MARKERS
        )
        if marker not in readonly
    ]
    assert missing_value_gate_self_check_markers == []
    missing_operator_sequence_markdown_markers = [
        marker
        for marker in (
            release_check.WING_READONLY_OPERATOR_SEQUENCE_LIST_MARKDOWN_SURFACE_REQUIRED_FIELD_LINES
        )
        if marker not in readonly
    ]
    assert missing_operator_sequence_markdown_markers == []
    missing_artifact_manifest_contract_markers = [
        marker
        for marker in release_check.NO_WING_RELEASE_CHECK_ARTIFACT_MANIFEST_CONTRACT_FIELDS
        if marker not in readonly
    ]
    assert missing_artifact_manifest_contract_markers == []
    missing_artifact_manifest_cross_surface_markers = [
        marker
        for marker in release_check.NO_WING_RELEASE_CHECK_ARTIFACT_MANIFEST_CONTRACT_CROSS_SURFACE_FIELDS
        if marker not in readonly
    ]
    assert missing_artifact_manifest_cross_surface_markers == []
    missing_artifact_manifest_summary_surface_markers = [
        marker
        for marker in release_check.NO_WING_RELEASE_CHECK_ARTIFACT_MANIFEST_SUMMARY_SURFACE_REQUIRED_FIELDS
        if marker not in readonly
    ]
    assert missing_artifact_manifest_summary_surface_markers == []
    missing_artifact_manifest_summary_surface_field_lines = [
        marker
        for marker in release_check.NO_WING_RELEASE_CHECK_ARTIFACT_MANIFEST_SUMMARY_SURFACE_REQUIRED_FIELD_LINES
        if marker not in readonly
    ]
    assert missing_artifact_manifest_summary_surface_field_lines == []

    assert "wing_readonly_preflight_runner.py" in readonly
    assert "review_wing_readonly_preflight_artifacts.py" in readonly
    assert "--require-physical-readback" in readonly
    assert "--operator-confirm-channels" in readonly
    assert "--url ws://localhost:8765" in readonly
    assert "--value-types fader,gain,pan" in readonly
    assert "--artifact-root artifacts/wing_readonly_preflight" in readonly
    assert "--value-types fader,gain \\" not in readonly
    assert "--plan-output artifacts/wing_readonly_preflight_plan/latest_plan.json" in readonly
    assert "--validate-run-inputs-only" in readonly
    assert "status: ready_for_websocket" in readonly
    assert "reason: run_inputs_validated_no_websocket" in readonly
    assert "validation_only: true" in readonly
    assert "websocket_opened: false" in readonly
    assert "contacts_backend: false" in readonly
    assert "contacts_wing: false" in readonly
    assert "expected_plan_match: true" in readonly
    assert "expected_plan_violations: []" in readonly
    assert "sent_request_types: []" in readonly
    assert "--execute" not in readonly
    assert "read_only: true" in readonly
    assert "live_write: false" in readonly
    assert "write_request_sent: false" in readonly
    assert "proposal_request_sent: false" in readonly
    assert "readiness_matrix_update_allowed: false" in readonly
    assert "readiness_closure_allowed: false" in readonly
    assert "live_authority_granted: false" in readonly
    assert "operator_confirmed_channels == channels" in readonly
    assert "strict `send`/`recv` pairs" in readonly
    assert "channel, value type, and address" in readonly
    assert "final handoff operator-sequence step" in readonly
    assert "operator_sequence_has_supervised_prestart_gate" in readonly
    assert "operator_sequence_supervised_prestart_gate_is_final" in readonly
    assert "supervised_prestart_gate" in readonly
    assert "expected_supervised_prestart_gate" in readonly
    missing_supervised_prestart_gate_markers = [
        marker
        for marker in (
            readonly_contract.WING_READONLY_PREFLIGHT_SUPERVISED_PRESTART_GATE_REQUIRED_MARKERS
        )
        if marker not in readonly
    ]
    assert missing_supervised_prestart_gate_markers == []
    assert "artifact_operator_next_steps" in readonly
    assert "before opening a backend/WING WebSocket" in readonly
    assert "they do not approve live writes or close readiness rows" in readonly
    assert "artifact_manifest_verifier_required: true" in readonly
    assert "artifact_manifest_verifier_command" in readonly
    assert "artifact_manifest_verifier_target_artifact_dir" in readonly
    assert "artifact_manifest_verifier_status_required: passed" in readonly
    assert "artifact_manifest_verifier_failed_checks_required: []" in readonly
    assert "artifact_manifest_verifier_required_fields_missing: []" in readonly
    assert "artifact_manifest_verifier_contacts_wing: false" in readonly
    assert "artifact_manifest_verifier_mutates_console_state: false" in readonly
    assert "artifact_manifest_verifier_queries_console: false" in readonly
    assert "artifact_manifest_verifier_readiness_closure_allowed: false" in readonly
    assert "artifact_manifest_verifier_live_authority_granted: false" in readonly
    assert "summary_json_artifact_manifest_verifier_fields_present: true" in readonly
    assert "manifest_artifact_manifest_verifier_fields_present: true" in readonly
    assert "summary_md_artifact_manifest_verifier_fields_rendered: true" in readonly
    assert (
        "summary_json_artifact_manifest_verifier_command_matches_expected: true"
        in readonly
    )
    assert (
        "summary_md_artifact_manifest_verifier_command_matches_expected: true"
        in readonly
    )
    assert "manifest_artifact_manifest_verifier_command_matches_expected: true" in readonly
    assert (
        "summary_json_artifact_manifest_verifier_report_required_fields_matches_expected: true"
        in readonly
    )
    assert (
        "summary_md_artifact_manifest_verifier_report_required_fields_matches_expected: true"
        in readonly
    )
    assert (
        "manifest_artifact_manifest_verifier_report_required_fields_matches_expected: true"
        in readonly
    )
    missing_artifact_manifest_verifier_operator_markers = [
        marker
        for marker in (
            readonly_contract.WING_READONLY_PREFLIGHT_ARTIFACT_MANIFEST_VERIFIER_REQUIRED_MARKERS
        )
        if marker not in readonly
    ]
    assert missing_artifact_manifest_verifier_operator_markers == []
    for marker in (
        "summary_json_wing_readonly_preflight_plan_artifact_source_authority_emission_fields_present: true",
        "summary_md_wing_readonly_preflight_plan_artifact_source_authority_emission_fields_rendered: true",
        "summary_json_wing_readonly_preflight_plan_artifact_artifact_manifest_verifier_source_authority_check_fields_optional_matches_expected: true",
        "summary_json_wing_readonly_preflight_plan_artifact_artifact_manifest_verifier_source_authority_check_emission_status_matches_expected: true",
        "summary_json_wing_readonly_preflight_plan_artifact_artifact_manifest_verifier_source_authority_check_field_count_matches_expected: true",
        "summary_json_wing_readonly_preflight_plan_artifact_artifact_manifest_verifier_source_authority_check_missing_count_matches_expected: true",
        "summary_json_wing_readonly_preflight_plan_artifact_artifact_manifest_verifier_source_authority_check_fields_present_matches_expected: true",
        "summary_json_wing_readonly_preflight_plan_artifact_artifact_manifest_verifier_source_authority_check_fields_missing_matches_expected: true",
        "summary_md_wing_readonly_preflight_plan_artifact_artifact_manifest_verifier_source_authority_check_fields_optional_matches_expected: true",
        "summary_md_wing_readonly_preflight_plan_artifact_artifact_manifest_verifier_source_authority_check_emission_status_matches_expected: true",
        "summary_md_wing_readonly_preflight_plan_artifact_artifact_manifest_verifier_source_authority_check_field_count_matches_expected: true",
        "summary_md_wing_readonly_preflight_plan_artifact_artifact_manifest_verifier_source_authority_check_fields_present_matches_expected: true",
        "summary_md_wing_readonly_preflight_plan_artifact_artifact_manifest_verifier_source_authority_check_fields_missing_matches_expected: true",
        "summary_md_wing_readonly_preflight_plan_artifact_artifact_manifest_verifier_source_authority_check_missing_count_matches_expected: true",
        "artifact_manifest_verifier_check_summary_json_wing_readonly_preflight_plan_artifact_source_authority_emission_fields_present: true",
        "artifact_manifest_verifier_check_summary_md_wing_readonly_preflight_plan_artifact_source_authority_emission_fields_rendered: true",
        "artifact_manifest_verifier_check_summary_json_wing_readonly_preflight_plan_artifact_artifact_manifest_verifier_source_authority_check_fields_optional_matches_expected: true",
        "artifact_manifest_verifier_check_summary_json_wing_readonly_preflight_plan_artifact_artifact_manifest_verifier_source_authority_check_emission_status_matches_expected: true",
        "artifact_manifest_verifier_check_summary_json_wing_readonly_preflight_plan_artifact_artifact_manifest_verifier_source_authority_check_field_count_matches_expected: true",
        "artifact_manifest_verifier_check_summary_json_wing_readonly_preflight_plan_artifact_artifact_manifest_verifier_source_authority_check_missing_count_matches_expected: true",
        "artifact_manifest_verifier_check_summary_json_wing_readonly_preflight_plan_artifact_artifact_manifest_verifier_source_authority_check_fields_present_matches_expected: true",
        "artifact_manifest_verifier_check_summary_json_wing_readonly_preflight_plan_artifact_artifact_manifest_verifier_source_authority_check_fields_missing_matches_expected: true",
        "artifact_manifest_verifier_check_summary_md_wing_readonly_preflight_plan_artifact_artifact_manifest_verifier_source_authority_check_fields_optional_matches_expected: true",
        "artifact_manifest_verifier_check_summary_md_wing_readonly_preflight_plan_artifact_artifact_manifest_verifier_source_authority_check_emission_status_matches_expected: true",
        "artifact_manifest_verifier_check_summary_md_wing_readonly_preflight_plan_artifact_artifact_manifest_verifier_source_authority_check_field_count_matches_expected: true",
        "artifact_manifest_verifier_check_summary_md_wing_readonly_preflight_plan_artifact_artifact_manifest_verifier_source_authority_check_fields_present_matches_expected: true",
        "artifact_manifest_verifier_check_summary_md_wing_readonly_preflight_plan_artifact_artifact_manifest_verifier_source_authority_check_fields_missing_matches_expected: true",
        "artifact_manifest_verifier_check_summary_md_wing_readonly_preflight_plan_artifact_artifact_manifest_verifier_source_authority_check_missing_count_matches_expected: true",
    ):
        assert marker in readonly
    assert "verifier_report_read_only_matches_expected: true" in readonly
    assert "verifier_report_live_write_matches_expected: true" in readonly
    assert "verifier_report_contacts_wing_matches_expected: true" in readonly
    assert "verifier_report_contacts_backend_matches_expected: true" in readonly
    assert "verifier_report_mutates_console_state_matches_expected: true" in readonly
    assert "verifier_report_queries_console_matches_expected: true" in readonly
    assert "verifier_report_executes_tests_matches_expected: true" in readonly
    assert (
        "verifier_report_readiness_closure_allowed_matches_expected: true"
        in readonly
    )
    assert "verifier_report_live_authority_granted_matches_expected: true" in readonly
    assert "artifact_manifest_verifier_check_mismatches: []" in readonly
    assert "artifact_manifest_verifier_check_<check_name>: true" in readonly
    assert "artifact_manifest_verifier_check_manifest_type: true" in readonly
    assert (
        "artifact_manifest_verifier_check_summary_json_sha256_matches_manifest: true"
        in readonly
    )
    assert (
        "artifact_manifest_verifier_check_summary_denies_live_authority: true"
        in readonly
    )
    assert "artifact_manifest_verifier_must_pass_before_websocket: true" in readonly
    assert "artifact_manifest_verifier_passed_before_websocket: true" in readonly
    assert "artifact_manifest_verifier_websocket_opened_after_pass: true" in readonly
    assert "artifact_manifest_verifier_operator_sequence_step_index: 1" in readonly
    assert (
        "expected_artifact_manifest_verifier_operator_sequence_step_index: 1"
        in readonly
    )
    assert (
        "artifact_manifest_verifier_operator_sequence_step_index_matches_contract: true"
        in readonly
    )
    assert "artifact_manifest_verifier_operator_sequence_step" in readonly
    assert "expected_artifact_manifest_verifier_operator_sequence_step" in readonly
    assert (
        "artifact_manifest_verifier_operator_sequence_step_matches_contract: true"
        in readonly
    )
    assert "artifact_manifest_verifier_required_markers" in readonly
    for marker in ARTIFACT_MANIFEST_VERIFIER_OPERATOR_RESULT_MARKERS:
        assert marker in readonly
    assert "artifact_manifest_verifier_operator_sequence_before_preview: true" in readonly
    assert "backend_port_prestart_check_operator_sequence_step_index: 0" in readonly
    assert "backend_port_prestart_check_operator_sequence_step" in readonly
    assert "prestart_backend_port_check" in readonly
    assert "unexpected listeners" in readonly
    assert "backend_port_prestart_check_must_be_first_operator_step: true" in readonly
    assert "backend_port_prestart_check_operator_sequence_step_is_first: true" in readonly
    assert "backend_connection_confirmation_step_index: 3" in readonly
    assert "operator_sequence_has_backend_connection_confirmation: true" in readonly
    assert "operator_sequence_backend_connection_confirmation_step_index: 3" in readonly
    assert (
        "operator_sequence_backend_connection_confirmation_before_capture: true"
        in readonly
    )
    assert "operator_sequence_backend_connection_confirmation_step" in readonly
    assert "existing backend connection" in readonly
    assert "intended WING Rack" in readonly
    assert "read-only inspection" in readonly
    assert "operator_sequence_validate_readonly_preflight_inputs_step_index: 5" in readonly
    assert (
        "operator_sequence_validate_readonly_preflight_inputs_step_matches_contract: true"
        in readonly
    )
    assert "operator_sequence_capture_readonly_preflight_step_index: 6" in readonly
    assert "mvp_blocker_anchor" in readonly
    assert "mvp_blocker_anchor_matches_current_first_open: true" in readonly
    assert "mvp_blocker_anchor_sha256" in readonly
    assert "mvp_blocker_anchor_sha256_matches_current_first_open: true" in readonly
    assert "mvp_blocker_anchor.packet_does_not_close_blocker: true" in readonly
    assert "operator_sequence_has_mvp_blocker_anchor_digest_review: true" in readonly
    assert "operator_sequence_count: 12" in readonly
    for marker in OPERATOR_SEQUENCE_SHA_MARKERS:
        assert marker in readonly
    for marker in OPERATOR_SEQUENCE_CONTRACT_MARKERS:
        assert marker in readonly
    for marker in DECLARED_OPERATOR_SEQUENCE_COUNT_MARKERS:
        assert marker in readonly
    assert "mvp_blocker_anchor_digest_review_step_index: 10" in readonly
    assert "expected_mvp_blocker_anchor_digest_review_step_index: 10" in readonly
    assert "mvp_blocker_anchor_digest_review_step_index_matches_contract: true" in readonly
    assert (
        "mvp_blocker_anchor_digest_review_step_index_matches_operator_sequence: true"
        in readonly
    )
    assert "operator_sequence_mvp_blocker_anchor_digest_review_step" in readonly
    assert "operator_sequence_mvp_blocker_anchor_digest_review_step_index: 10" in readonly
    assert (
        "expected_operator_sequence_mvp_blocker_anchor_digest_review_step_index: 10"
        in readonly
    )
    assert (
        "operator_sequence_mvp_blocker_anchor_digest_review_step_index_matches_contract: true"
        in readonly
    )
    assert "current first open blocker" in readonly
    assert "backend_port_prestart_gate" in readonly
    assert "expected_backend_port_prestart_gate" in readonly
    assert "backend_port_prestart_gate.grants_live_authority: false" in readonly
    assert (
        "backend_port_prestart_gate.must_pass_before_backend_wing_session: true"
        in readonly
    )
    assert "backend_port_prestart_gate.operator_sequence_step_index: 0" in readonly
    assert (
        "backend_port_prestart_gate.operator_sequence_step_must_be_first: true"
        in readonly
    )
    assert "backend_port_prestart_gate.runs_command: prestart_backend_port_check" in readonly
    assert "backend_port_prestart_gate.required_markers" in readonly
    assert "Before starting the intended backend/WING session" in readonly
    assert "backend_port_prestart_check_command" in readonly
    assert "backend_port_prestart_check_command_text" in readonly
    assert "PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick" in readonly
    assert (
        "failed busy-port no-WING summary and no-WING handoff summary both show"
        in readonly
    )
    assert "backend_port_prestart_check_required: true" in readonly
    assert (
        "backend_port_prestart_check_must_pass_before_starting_backend: true"
        in readonly
    )
    assert "backend_port_prestart_gate_execution_allowed: false" in readonly
    assert "backend_port_prestart_gate_readiness_closure_allowed: false" in readonly
    assert "backend_port_prestart_gate_live_authority_granted: false" in readonly
    assert "backend_port_prestart_check_failure_detail_fields" in readonly
    assert "listener_processes" in readonly
    assert "listener_process_error" in readonly
    assert "safe_to_stop_listener_by_gate" in readonly
    assert "operator_next_step" in readonly
    for marker in ENVIRONMENT_BLOCKED_FRONTEND_MARKERS:
        assert marker in readonly
    assert "listener_confirmation_template_accepted_by_template: false" in readonly
    assert (
        "listener_confirmation_template_execution_allowed_by_template: false"
        in readonly
    )
    assert (
        "listener_confirmation_template_readiness_closure_allowed_by_template: false"
        in readonly
    )
    assert "listener_confirmation_template_operator_placeholder_fields" in readonly
    assert "REPLACE_WITH_*" in readonly
    assert "manual_review_required_before_readiness_update: true" in readonly
    assert "readiness_matrix_update_allowed_by_packet: false" in readonly
    assert "readiness_closure_allowed_by_packet: false" in readonly
    assert "read_only: true" in readonly
    assert "live_write: false" in readonly
    assert "mutates_console_state: false" in readonly
    assert "queries_console: false" in readonly
    assert "executes_tests: false" in readonly
    assert "operator_required: true" in readonly
    assert "expected_plan_required_for_real_run: true" in readonly
    assert "physical_preflight_requires_existing_backend_connection: true" in readonly
    assert "backend_port_prestart_check_required: true" in readonly
    assert (
        "backend_port_prestart_check_must_pass_before_starting_backend: true"
        in readonly
    )
    assert "backend_port_prestart_gate_execution_allowed: false" in readonly
    assert "backend_port_prestart_gate_readiness_closure_allowed: false" in readonly
    assert "backend_port_prestart_gate_live_authority_granted: false" in readonly
    assert "execution_allowed_by_packet: false" in readonly
    assert "live_authority_granted: false" in readonly
    assert "full_autonomous_ready: false" in readonly
    assert (
        "next_manual_artifact_required: real wing_readonly_preflight.v1 summary.json"
        in readonly
    )
    assert "expected_artifacts" in readonly
    assert "expected_artifact_globs" in readonly
    assert "plan_expected_artifacts" in readonly
    assert "expected_artifacts_match_plan: true" in readonly
    assert "summary_glob_matches_plan: true" in readonly
    assert "event_log_glob_matches_plan: true" in readonly
    for marker in (
        "preview_command_artifact_root_matches_packet: true",
        "preview_command_plan_output_matches_packet: true",
        "validate_inputs_command_uses_validate_run_inputs_only: true",
        "validate_inputs_command_uses_plan_only: false",
        "validate_inputs_command_artifact_root_matches_packet: true",
        "validate_inputs_command_expected_plan_matches_packet: true",
        "capture_command_artifact_root_matches_packet: true",
        "capture_command_expected_plan_matches_packet: true",
        "review_command_artifact_root_matches_packet: true",
        "decision_review_command_artifact_root_matches_packet: true",
        "decision_review_command_decision_root_matches_packet: true",
    ):
        assert marker in readonly
    assert "supervised_prestart_gate.grants_live_authority: false" in readonly
    assert "required_supervised_prestart_gate_markers" in readonly
    assert "supervised_prestart_gate.operator_sequence_step_index: 11" in readonly
    assert "operator_sequence_supervised_prestart_gate_step_index: 11" in readonly
    assert (
        "supervised_prestart_gate.operator_sequence_step_must_be_final: true"
        in readonly
    )
    assert "supervised_prestart_gate.requires_operator_decision_status:" in readonly
    assert "accepted_decision_ready" in readonly
    assert "supervised_prestart_gate.unlocks_runbook:" in readonly
    assert "Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md" in readonly
    assert "--backend-listener-confirmation" in readonly
    assert "wing_backend_port_listener_confirmation.v1" in readonly
    assert "operator_confirmed_intended_backend" in readonly
    assert "confirms_intended_backend" in readonly
    assert "reviewed_listener_pid" in readonly
    assert "reviewed_listener_command_line" in readonly
    assert "safe_to_stop_listener_by_gate" in readonly
    assert "live_authority_granted" in readonly
    assert '"artifact_root": "artifacts/wing_readonly_preflight"' in readonly
    assert "`review_id` field must remain present" in readonly
    assert (
        "keeps the record invalid even if the artifact hashes still match"
        in readonly
    )
    assert (
        "`template_status: operator_must_replace_placeholders_before_ledger_acceptance`"
        in readonly
    )
    assert "unedited pending template and fails closed" in readonly
    assert (
        "`artifact_captured_at`, `artifact_root`, `summary_sha256`, `events_sha256`"
        in readonly
    )
    assert (
        "`artifact_captured_at`, `artifact_root`, `summary_path`, `events_path`"
        in readonly
    )
    assert (
        "supervised_prestart_gate.requires_accepted_candidate_prestart_trace: true"
        in readonly
    )
    assert "valid accepted-candidate prestart trace" in readonly
    assert "expected-artifact-bound accepted source id" in readonly
    assert "wing_operator_proposal_validation_runner.py" in supervised
    assert "--execute" in supervised
    assert "blocked until a clean read-only preflight" in supervised
    assert "valid accepted-candidate prestart trace" in supervised
    assert "valid accepted-candidate verifier WebSocket proof" in supervised
    assert "valid accepted-candidate artifact-root trace" in supervised
    assert "valid accepted-candidate request-sequence trace" in supervised
    assert "valid accepted-candidate contact trace" in supervised
    assert "valid accepted-candidate expected-artifact trace" in supervised
    assert "expected-artifact-bound accepted source id" in supervised
    assert "supervised_prestart_gate" in supervised
    assert "expected_supervised_prestart_gate" in supervised
    assert "WING_SUPERVISED_FADER_GAIN_PROOF" in supervised
    assert (
        "readonly_preflight_review_decision_candidate_verifier_ws_proof_complete: true"
        in supervised
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_request_sequence_trace_complete: true"
        in supervised
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_contact_trace_complete: true"
        in supervised
    )
    assert (
        "readonly_preflight_review_decision_candidate_artifact_root_contract_valid: true"
        in supervised
    )
    assert (
        "accepted_readonly_preflight_candidate_prestart_contract_required_before_supervised_proof"
        in supervised
    )
    assert (
        "accepted_readonly_preflight_candidate_verifier_ws_proof_required_before_supervised_proof"
        in supervised
    )
    assert (
        "accepted_readonly_preflight_candidate_artifact_root_required_before_supervised_proof"
        in supervised
    )
    assert (
        "accepted_readonly_preflight_candidate_request_sequence_required_before_supervised_proof"
        in supervised
    )
    assert (
        "accepted_readonly_preflight_candidate_expected_artifact_trace_required_before_supervised_proof"
        in supervised
    )
    assert (
        "accepted_readonly_preflight_candidate_contact_trace_required_before_supervised_proof"
        in supervised
    )
    assert "accepted_candidate_prestart_contract_violations_by_source" in readonly
    assert (
        "readonly_preflight_review_decision_candidate_prestart_contract_violations_by_source"
        in readonly
    )
    assert "accepted_candidate_prestart_contract_violations_by_source: {}" in readonly
    assert (
        "readonly_preflight_review_decision_candidate_prestart_contract_violations_by_source: {}"
        in readonly
    )
    assert "negative delta" in supervised
    assert "Full autonomous" not in supervised


def test_readonly_runbook_requires_no_wing_manifest_summary_surface_sidecar_gate():
    readonly = READONLY_RUNBOOK.read_text(encoding="utf-8")
    normalized_readonly = " ".join(readonly.split())

    assert "Before opening a backend/WING WebSocket" in readonly
    assert "do not open the WING session" in normalized_readonly
    assert "do not close readiness" in readonly
    assert "approve live writes" in readonly
    assert "grant live authority" in readonly
    for marker in ARTIFACT_MANIFEST_SUMMARY_SURFACE_SIDECAR_MARKERS:
        assert marker in readonly
    for marker in ARTIFACT_MANIFEST_VERIFIER_REPORT_MARKERS:
        assert marker in readonly


def test_progress_tool_generates_latest_report_from_blocker_table(tmp_path):
    progress = load_progress_module()
    rows = progress.parse_blocker_table(BLOCKERS_DOC)
    current_command = progress.current_first_command(BLOCKERS_DOC)
    assert current_command is not None
    assert current_command.startswith("Use `PYTHONPATH=backend python3")
    assert "required_supervised_proof_plan_summary_fields" in current_command
    assert "expected_required_supervised_proof_plan_summary_fields" in current_command
    assert (
        "required_supervised_proof_plan_summary_fields_match_contract: true"
        in current_command
    )
    assert (
        "required_supervised_proof_plan_summary_fields_include_invalid_records: true"
        in current_command
    )
    assert (
        "the final `operator_sequence_supervised_prestart_gate_step` must include `accepted_decision_ready`, `valid accepted-candidate prestart trace`, `valid accepted-candidate verifier WebSocket proof`, `valid accepted-candidate artifact-root trace`, `valid accepted-candidate request-sequence trace`, `valid accepted-candidate contact trace`, `valid accepted-candidate expected-artifact trace`, `expected-artifact-bound accepted source id`, and `WING_SUPERVISED_FADER_GAIN_PROOF"
        in current_command
    )
    assert (
        'required_supervised_prestart_gate_markers: ["accepted_decision_ready", "valid accepted-candidate prestart trace", "valid accepted-candidate verifier WebSocket proof", "valid accepted-candidate artifact-root trace", "valid accepted-candidate request-sequence trace", "valid accepted-candidate contact trace", "valid accepted-candidate expected-artifact trace", "expected-artifact-bound accepted source id", "WING_SUPERVISED_FADER_GAIN_PROOF"]'
        in current_command
    )
    assert (
        'supervised_prestart_gate_required_markers: ["accepted_decision_ready", "valid accepted-candidate prestart trace", "valid accepted-candidate verifier WebSocket proof", "valid accepted-candidate artifact-root trace", "valid accepted-candidate request-sequence trace", "valid accepted-candidate contact trace", "valid accepted-candidate expected-artifact trace", "expected-artifact-bound accepted source id", "WING_SUPERVISED_FADER_GAIN_PROOF"]'
        in current_command
    )
    assert (
        'expected_supervised_prestart_gate_required_markers: ["accepted_decision_ready", "valid accepted-candidate prestart trace", "valid accepted-candidate verifier WebSocket proof", "valid accepted-candidate artifact-root trace", "valid accepted-candidate request-sequence trace", "valid accepted-candidate contact trace", "valid accepted-candidate expected-artifact trace", "expected-artifact-bound accepted source id", "WING_SUPERVISED_FADER_GAIN_PROOF"]'
        in current_command
    )
    assert (
        'expected_required_supervised_prestart_gate_markers: ["accepted_decision_ready", "valid accepted-candidate prestart trace", "valid accepted-candidate verifier WebSocket proof", "valid accepted-candidate artifact-root trace", "valid accepted-candidate request-sequence trace", "valid accepted-candidate contact trace", "valid accepted-candidate expected-artifact trace", "expected-artifact-bound accepted source id", "WING_SUPERVISED_FADER_GAIN_PROOF"]'
        in current_command
    )
    assert (
        'readonly_preflight_review_decision_accepted_candidate_queries_console_by_source: {"artifacts/wing_readonly_preflight/run/summary.json": false}'
        in current_command
    )
    for marker in (
        'readonly_preflight_review_decision_candidate_verifier_ws_proof_source_ids: ["artifacts/wing_readonly_preflight/run/summary.json"]',
        "readonly_preflight_review_decision_candidate_verifier_ws_proof_violations_by_source: {}",
        "readonly_preflight_review_decision_candidate_verifier_ws_proof_complete: true",
    ):
        assert marker in current_command
    assert "operator-readable no-WING `summary.md`" in current_command
    assert "operator_markdown_summary_required: true" in current_command
    assert "operator_markdown_summary_field_lines_rendered: true" in current_command
    assert "operator_markdown_summary_missing_field_lines: []" in current_command
    assert "operator_markdown_summary_artifact: summary.md" in current_command
    assert "operator_markdown_summary_sha256" in current_command
    assert "operator_markdown_summary_sha256_available: true" in current_command
    assert "operator_markdown_summary_actual_sha256" in current_command
    assert "operator_markdown_summary_sha256_matches_actual: true" in current_command
    assert "artifact_manifest_summary_surface_required: true" in current_command
    assert "artifact_manifest_summary_surface_missing_fields: []" in current_command
    assert "artifact_manifest_summary_surface_required_field_lines" in current_command
    assert (
        "artifact_manifest_summary_surface_missing_field_lines: []"
        in current_command
    )
    assert (
        "artifact_manifest_summary_surface_field_lines_rendered: true"
        in current_command
    )
    assert "artifact_manifest_contract_id" in current_command
    assert "artifact_manifest_contract_required: true" in current_command
    assert "artifact_manifest_contract_fields" in current_command
    assert "artifact_manifest_contract_sha256" in current_command
    assert "artifact_manifest_contract_sha256_available: true" in current_command
    assert (
        "artifact_manifest_contract_sha256_matches_expected: true"
        in current_command
    )
    assert "artifact_manifest_contract_cross_surface_required: true" in current_command
    assert "artifact_manifest_contract_cross_surface_required_surfaces" in current_command
    assert "artifact_manifest_contract_cross_surface_required_fields" in current_command
    assert (
        "artifact_manifest_contract_cross_surface_expected_match: true"
        in current_command
    )
    assert "artifact_manifest_contract_cross_surface_match: true" in current_command
    assert (
        "artifact_manifest_summary_json_contract_sha256_matches_manifest: true"
        in current_command
    )
    assert (
        "artifact_manifest_summary_md_contract_sha256_matches_manifest: true"
        in current_command
    )
    assert "artifact_manifest_contract_cross_surface_mismatches: []" in current_command
    assert (
        "artifact_manifest_contract_cross_surface_readiness_closure_allowed: false"
        in current_command
    )
    assert (
        "artifact_manifest_contract_cross_surface_live_authority_granted: false"
        in current_command
    )
    assert "artifact_manifest_required_artifact_sha256_fields" in current_command
    assert "artifact_manifest_path: artifact-manifest.json" in current_command
    assert "artifact_manifest_artifact_count: 2" in current_command
    assert "artifact_manifest_required_artifact_count: 2" in current_command
    assert "artifact_manifest_artifact_count_matches_required: true" in current_command
    assert "artifact_manifest_required_artifact_ids" in current_command
    assert "artifact_manifest_artifact_ids" in current_command
    assert "artifact_manifest_required_artifacts_present: true" in current_command
    assert "artifact_manifest_required_artifact_paths" in current_command
    assert "artifact_manifest_artifact_paths" in current_command
    assert "artifact_manifest_required_artifact_paths_present: true" in current_command
    assert (
        "artifact_manifest_required_artifact_sha256_available: true"
        in current_command
    )
    assert "artifact_manifest_summary_json_sha256" in current_command
    assert "artifact_manifest_summary_json_sha256_available: true" in current_command
    assert "artifact_manifest_summary_md_sha256" in current_command
    assert "artifact_manifest_summary_md_sha256_available: true" in current_command
    assert "artifact_manifest_readiness_closure_allowed: false" in current_command
    assert "artifact_manifest_live_authority_granted: false" in current_command
    assert "Before opening a backend/WING WebSocket" in current_command
    assert "do not open the WING session" in current_command
    assert "do not close readiness" in current_command
    assert "approve live writes" in current_command
    assert "grant live authority" in current_command
    for marker in ARTIFACT_MANIFEST_SUMMARY_SURFACE_SIDECAR_MARKERS:
        assert marker in current_command
    for marker in ARTIFACT_MANIFEST_VERIFIER_REPORT_MARKERS:
        assert marker in current_command
    assert "backend_port_prestart_gate_matches_expected: true" in current_command
    assert (
        "declared_backend_port_prestart_gate_matches_expected: true"
        in current_command
    )
    assert "supervised_prestart_gate_matches_expected: true" in current_command
    assert (
        "declared_supervised_prestart_gate_matches_expected: true"
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_candidate_prestart_contract_valid: true"
        in current_command
    )
    assert (
        'readonly_preflight_review_decision_candidate_prestart_source_ids: ["artifacts/wing_readonly_preflight/run/summary.json"]'
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_candidate_prestart_contract_violations_by_source: {}"
        in current_command
    )
    assert (
        'readonly_preflight_review_decision_expected_candidate_prestart_source_ids: ["artifacts/wing_readonly_preflight/run/summary.json"]'
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_expected_candidate_prestart_contract_violations_by_source: {}"
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_candidate_artifact_root_contract_valid: true"
        in current_command
    )
    assert (
        'readonly_preflight_review_decision_candidate_artifact_roots: ["artifacts/wing_readonly_preflight"]'
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_candidate_artifact_root_contract_violations_by_source: {}"
        in current_command
    )
    assert (
        'readonly_preflight_review_decision_expected_candidate_artifact_roots: ["artifacts/wing_readonly_preflight"]'
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_expected_candidate_artifact_root_contract_violations_by_source: {}"
        in current_command
    )
    assert (
        'readonly_preflight_review_decision_accepted_candidate_expected_artifact_trace_source_ids: ["artifacts/wing_readonly_preflight/run/summary.json"]'
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_expected_artifacts_by_source"
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_expected_summary_artifact_by_source"
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_expected_events_artifact_by_source"
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_summary_path_matches_expected_artifact_by_source"
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_events_path_matches_expected_artifact_by_source"
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_expected_artifact_trace_missing_source_ids: []"
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_expected_artifact_trace_complete: true"
        in current_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_required: true"
        in current_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_violations: []"
        in current_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_passed: true"
        in current_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_readiness_closure_allowed: false"
        in current_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_live_authority_granted: false"
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_candidate_request_sequence_contract_valid: true"
        in current_command
    )
    assert (
        'readonly_preflight_review_decision_candidate_request_sequence_source_ids: ["artifacts/wing_readonly_preflight/run/summary.json"]'
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_candidate_request_sequence_contract_violations_by_source: {}"
        in current_command
    )
    assert (
        'readonly_preflight_review_decision_expected_candidate_request_sequence_source_ids: ["artifacts/wing_readonly_preflight/run/summary.json"]'
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_expected_candidate_request_sequence_contract_violations_by_source: {}"
        in current_command
    )
    assert (
        'readonly_preflight_review_decision_accepted_candidate_request_sequence_trace_source_ids: ["artifacts/wing_readonly_preflight/run/summary.json"]'
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_request_sequence_by_source"
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_plan_request_sequence_by_source"
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_expected_request_sequence_by_source"
        in current_command
    )
    assert (
        'readonly_preflight_review_decision_accepted_candidate_request_sequence_matches_plan_by_source: {"artifacts/wing_readonly_preflight/run/summary.json": true}'
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_request_sequence_trace_missing_source_ids: []"
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_request_sequence_trace_complete: true"
        in current_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_required: true"
        in current_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_violations: []"
        in current_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_passed: true"
        in current_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_readiness_closure_allowed: false"
        in current_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_live_authority_granted: false"
        in current_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_summary_surface_readiness_closure_allowed: false"
        in current_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_summary_surface_live_authority_granted: false"
        in current_command
    )
    assert (
        'readonly_preflight_review_decision_accepted_candidate_contact_trace_source_ids: ["artifacts/wing_readonly_preflight/run/summary.json"]'
        in current_command
    )
    assert (
        'readonly_preflight_review_decision_accepted_candidate_expected_contacts_backend_by_source: {"artifacts/wing_readonly_preflight/run/summary.json": true}'
        in current_command
    )
    assert (
        'readonly_preflight_review_decision_accepted_candidate_expected_queries_console_by_source: {"artifacts/wing_readonly_preflight/run/summary.json": true}'
        in current_command
    )
    assert (
        'readonly_preflight_review_decision_accepted_candidate_contacts_backend_by_source: {"artifacts/wing_readonly_preflight/run/summary.json": true}'
        in current_command
    )
    assert (
        'readonly_preflight_review_decision_accepted_candidate_queries_console_by_source: {"artifacts/wing_readonly_preflight/run/summary.json": false}'
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_contact_trace_missing_source_ids: []"
        in current_command
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_contact_trace_complete: true"
        in current_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_contact_trace_value_gate_required: true"
        in current_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_contact_trace_value_gate_violations: []"
        in current_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_contact_trace_value_gate_passed: true"
        in current_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_contact_trace_value_gate_readiness_closure_allowed: false"
        in current_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_contact_trace_value_gate_live_authority_granted: false"
        in current_command
    )
    assert "project_director_selected_action_anchor" in current_command
    assert "expected_project_director_selected_action_anchor" in current_command
    assert (
        "project_director_selected_action_anchor_matches_current_directive: true"
        in current_command
    )
    assert (
        "project_director_selected_action_source: Current First Command"
        in current_command
    )
    assert (
        "expected_project_director_selected_action_source: Current First Command"
        in current_command
    )
    assert (
        "project_director_selected_action_source_matches_current_directive: true"
        in current_command
    )
    assert "project_director_selected_action_sha256" in current_command
    assert "actual_project_director_selected_action_sha256" in current_command
    assert "expected_project_director_selected_action_sha256" in current_command
    assert "project_director_selected_action_sha256_available: true" in current_command
    assert (
        "project_director_selected_action_sha256_available_matches_digest: true"
        in current_command
    )
    assert (
        "project_director_selected_action_sha256_matches_anchor: true"
        in current_command
    )
    assert (
        "project_director_selected_action_sha256_matches_text: true"
        in current_command
    )
    assert (
        "project_director_selected_action_sha256_matches_current_directive: true"
        in current_command
    )
    assert (
        "project_director_current_first_command_baseline_coverage_required: true"
        in current_command
    )
    assert (
        "project_director_current_first_command_baseline_coverage_passed: true"
        in current_command
    )
    assert (
        "project_director_current_first_command_baseline_coverage_marker_count"
        in current_command
    )
    assert (
        "project_director_current_first_command_baseline_missing_markers: []"
        in current_command
    )
    assert progress.current_first_command_missing_baseline_markers(current_command) == []
    assert progress.current_first_command_covers_baseline(current_command)
    missing_final_supervised_step_contact_marker_command = current_command.replace(
        "the final `operator_sequence_supervised_prestart_gate_step` must include `accepted_decision_ready`, `valid accepted-candidate prestart trace`, `valid accepted-candidate verifier WebSocket proof`, `valid accepted-candidate artifact-root trace`, `valid accepted-candidate request-sequence trace`, `valid accepted-candidate contact trace`, `valid accepted-candidate expected-artifact trace`, `expected-artifact-bound accepted source id`, and `WING_SUPERVISED_FADER_GAIN_PROOF",
        "the final `operator_sequence_supervised_prestart_gate_step` must include `accepted_decision_ready`, `valid accepted-candidate prestart trace`, `valid accepted-candidate verifier WebSocket proof`, `valid accepted-candidate artifact-root trace`, `valid accepted-candidate request-sequence trace`, `valid accepted-candidate expected-artifact trace`, `expected-artifact-bound accepted source id`, and `WING_SUPERVISED_FADER_GAIN_PROOF",
    )
    assert not progress.current_first_command_covers_baseline(
        missing_final_supervised_step_contact_marker_command
    )
    assert (
        "the final `operator_sequence_supervised_prestart_gate_step` must include `accepted_decision_ready`, `valid accepted-candidate prestart trace`, `valid accepted-candidate verifier WebSocket proof`, `valid accepted-candidate artifact-root trace`, `valid accepted-candidate request-sequence trace`, `valid accepted-candidate contact trace`, `valid accepted-candidate expected-artifact trace`, `expected-artifact-bound accepted source id`, and `WING_SUPERVISED_FADER_GAIN_PROOF"
        in progress.current_first_command_missing_baseline_markers(
            missing_final_supervised_step_contact_marker_command
        )
    )
    missing_final_supervised_step_expected_artifact_marker_command = (
        current_command.replace(
            "the final `operator_sequence_supervised_prestart_gate_step` must include `accepted_decision_ready`, `valid accepted-candidate prestart trace`, `valid accepted-candidate verifier WebSocket proof`, `valid accepted-candidate artifact-root trace`, `valid accepted-candidate request-sequence trace`, `valid accepted-candidate contact trace`, `valid accepted-candidate expected-artifact trace`, `expected-artifact-bound accepted source id`, and `WING_SUPERVISED_FADER_GAIN_PROOF",
            "the final `operator_sequence_supervised_prestart_gate_step` must include `accepted_decision_ready`, `valid accepted-candidate prestart trace`, `valid accepted-candidate verifier WebSocket proof`, `valid accepted-candidate artifact-root trace`, `valid accepted-candidate request-sequence trace`, `valid accepted-candidate contact trace`, `expected-artifact-bound accepted source id`, and `WING_SUPERVISED_FADER_GAIN_PROOF",
        )
    )
    assert not progress.current_first_command_covers_baseline(
        missing_final_supervised_step_expected_artifact_marker_command
    )
    assert (
        "the final `operator_sequence_supervised_prestart_gate_step` must include `accepted_decision_ready`, `valid accepted-candidate prestart trace`, `valid accepted-candidate verifier WebSocket proof`, `valid accepted-candidate artifact-root trace`, `valid accepted-candidate request-sequence trace`, `valid accepted-candidate contact trace`, `valid accepted-candidate expected-artifact trace`, `expected-artifact-bound accepted source id`, and `WING_SUPERVISED_FADER_GAIN_PROOF"
        in progress.current_first_command_missing_baseline_markers(
            missing_final_supervised_step_expected_artifact_marker_command
        )
    )
    missing_supervised_contact_gate_marker_command = current_command.replace(
        'required_supervised_prestart_gate_markers: ["accepted_decision_ready", "valid accepted-candidate prestart trace", "valid accepted-candidate verifier WebSocket proof", "valid accepted-candidate artifact-root trace", "valid accepted-candidate request-sequence trace", "valid accepted-candidate contact trace", "valid accepted-candidate expected-artifact trace", "expected-artifact-bound accepted source id", "WING_SUPERVISED_FADER_GAIN_PROOF"]',
        'required_supervised_prestart_gate_markers: ["accepted_decision_ready", "valid accepted-candidate prestart trace", "valid accepted-candidate verifier WebSocket proof", "valid accepted-candidate artifact-root trace", "valid accepted-candidate request-sequence trace", "valid accepted-candidate expected-artifact trace", "expected-artifact-bound accepted source id", "WING_SUPERVISED_FADER_GAIN_PROOF"]',
    )
    assert not progress.current_first_command_covers_baseline(
        missing_supervised_contact_gate_marker_command
    )
    assert (
        'required_supervised_prestart_gate_markers: ["accepted_decision_ready", "valid accepted-candidate prestart trace", "valid accepted-candidate verifier WebSocket proof", "valid accepted-candidate artifact-root trace", "valid accepted-candidate request-sequence trace", "valid accepted-candidate contact trace", "valid accepted-candidate expected-artifact trace", "expected-artifact-bound accepted source id", "WING_SUPERVISED_FADER_GAIN_PROOF"]'
        in progress.current_first_command_missing_baseline_markers(
            missing_supervised_contact_gate_marker_command
        )
    )
    missing_supervised_expected_artifact_gate_marker_command = (
        current_command.replace(
            'required_supervised_prestart_gate_markers: ["accepted_decision_ready", "valid accepted-candidate prestart trace", "valid accepted-candidate verifier WebSocket proof", "valid accepted-candidate artifact-root trace", "valid accepted-candidate request-sequence trace", "valid accepted-candidate contact trace", "valid accepted-candidate expected-artifact trace", "expected-artifact-bound accepted source id", "WING_SUPERVISED_FADER_GAIN_PROOF"]',
            'required_supervised_prestart_gate_markers: ["accepted_decision_ready", "valid accepted-candidate prestart trace", "valid accepted-candidate verifier WebSocket proof", "valid accepted-candidate artifact-root trace", "valid accepted-candidate request-sequence trace", "valid accepted-candidate contact trace", "expected-artifact-bound accepted source id", "WING_SUPERVISED_FADER_GAIN_PROOF"]',
        )
    )
    assert not progress.current_first_command_covers_baseline(
        missing_supervised_expected_artifact_gate_marker_command
    )
    assert (
        'required_supervised_prestart_gate_markers: ["accepted_decision_ready", "valid accepted-candidate prestart trace", "valid accepted-candidate verifier WebSocket proof", "valid accepted-candidate artifact-root trace", "valid accepted-candidate request-sequence trace", "valid accepted-candidate contact trace", "valid accepted-candidate expected-artifact trace", "expected-artifact-bound accepted source id", "WING_SUPERVISED_FADER_GAIN_PROOF"]'
        in progress.current_first_command_missing_baseline_markers(
            missing_supervised_expected_artifact_gate_marker_command
        )
    )
    missing_supervised_gate_required_contact_marker_command = current_command.replace(
        'supervised_prestart_gate_required_markers: ["accepted_decision_ready", "valid accepted-candidate prestart trace", "valid accepted-candidate verifier WebSocket proof", "valid accepted-candidate artifact-root trace", "valid accepted-candidate request-sequence trace", "valid accepted-candidate contact trace", "valid accepted-candidate expected-artifact trace", "expected-artifact-bound accepted source id", "WING_SUPERVISED_FADER_GAIN_PROOF"]',
        'supervised_prestart_gate_required_markers: ["accepted_decision_ready", "valid accepted-candidate prestart trace", "valid accepted-candidate verifier WebSocket proof", "valid accepted-candidate artifact-root trace", "valid accepted-candidate request-sequence trace", "valid accepted-candidate expected-artifact trace", "expected-artifact-bound accepted source id", "WING_SUPERVISED_FADER_GAIN_PROOF"]',
    )
    assert not progress.current_first_command_covers_baseline(
        missing_supervised_gate_required_contact_marker_command
    )
    assert (
        'supervised_prestart_gate_required_markers: ["accepted_decision_ready", "valid accepted-candidate prestart trace", "valid accepted-candidate verifier WebSocket proof", "valid accepted-candidate artifact-root trace", "valid accepted-candidate request-sequence trace", "valid accepted-candidate contact trace", "valid accepted-candidate expected-artifact trace", "expected-artifact-bound accepted source id", "WING_SUPERVISED_FADER_GAIN_PROOF"]'
        in progress.current_first_command_missing_baseline_markers(
            missing_supervised_gate_required_contact_marker_command
        )
    )
    missing_expected_supervised_gate_required_contact_marker_command = (
        current_command.replace(
            'expected_supervised_prestart_gate_required_markers: ["accepted_decision_ready", "valid accepted-candidate prestart trace", "valid accepted-candidate verifier WebSocket proof", "valid accepted-candidate artifact-root trace", "valid accepted-candidate request-sequence trace", "valid accepted-candidate contact trace", "valid accepted-candidate expected-artifact trace", "expected-artifact-bound accepted source id", "WING_SUPERVISED_FADER_GAIN_PROOF"]',
            'expected_supervised_prestart_gate_required_markers: ["accepted_decision_ready", "valid accepted-candidate prestart trace", "valid accepted-candidate verifier WebSocket proof", "valid accepted-candidate artifact-root trace", "valid accepted-candidate request-sequence trace", "valid accepted-candidate expected-artifact trace", "expected-artifact-bound accepted source id", "WING_SUPERVISED_FADER_GAIN_PROOF"]',
        )
    )
    assert not progress.current_first_command_covers_baseline(
        missing_expected_supervised_gate_required_contact_marker_command
    )
    assert (
        'expected_supervised_prestart_gate_required_markers: ["accepted_decision_ready", "valid accepted-candidate prestart trace", "valid accepted-candidate verifier WebSocket proof", "valid accepted-candidate artifact-root trace", "valid accepted-candidate request-sequence trace", "valid accepted-candidate contact trace", "valid accepted-candidate expected-artifact trace", "expected-artifact-bound accepted source id", "WING_SUPERVISED_FADER_GAIN_PROOF"]'
        in progress.current_first_command_missing_baseline_markers(
            missing_expected_supervised_gate_required_contact_marker_command
        )
    )
    missing_expected_required_supervised_contact_marker_command = (
        current_command.replace(
            'expected_required_supervised_prestart_gate_markers: ["accepted_decision_ready", "valid accepted-candidate prestart trace", "valid accepted-candidate verifier WebSocket proof", "valid accepted-candidate artifact-root trace", "valid accepted-candidate request-sequence trace", "valid accepted-candidate contact trace", "valid accepted-candidate expected-artifact trace", "expected-artifact-bound accepted source id", "WING_SUPERVISED_FADER_GAIN_PROOF"]',
            'expected_required_supervised_prestart_gate_markers: ["accepted_decision_ready", "valid accepted-candidate prestart trace", "valid accepted-candidate verifier WebSocket proof", "valid accepted-candidate artifact-root trace", "valid accepted-candidate request-sequence trace", "valid accepted-candidate expected-artifact trace", "expected-artifact-bound accepted source id", "WING_SUPERVISED_FADER_GAIN_PROOF"]',
        )
    )
    assert not progress.current_first_command_covers_baseline(
        missing_expected_required_supervised_contact_marker_command
    )
    assert (
        'expected_required_supervised_prestart_gate_markers: ["accepted_decision_ready", "valid accepted-candidate prestart trace", "valid accepted-candidate verifier WebSocket proof", "valid accepted-candidate artifact-root trace", "valid accepted-candidate request-sequence trace", "valid accepted-candidate contact trace", "valid accepted-candidate expected-artifact trace", "expected-artifact-bound accepted source id", "WING_SUPERVISED_FADER_GAIN_PROOF"]'
        in progress.current_first_command_missing_baseline_markers(
            missing_expected_required_supervised_contact_marker_command
        )
    )
    querying_console_contact_trace_command = current_command.replace(
        'readonly_preflight_review_decision_accepted_candidate_queries_console_by_source: {"artifacts/wing_readonly_preflight/run/summary.json": false}',
        'readonly_preflight_review_decision_accepted_candidate_queries_console_by_source: {"artifacts/wing_readonly_preflight/run/summary.json": true}',
    )
    assert not progress.current_first_command_covers_baseline(
        querying_console_contact_trace_command
    )
    assert (
        'readonly_preflight_review_decision_accepted_candidate_queries_console_by_source: {"artifacts/wing_readonly_preflight/run/summary.json": false}'
        in progress.current_first_command_missing_baseline_markers(
            querying_console_contact_trace_command
        )
    )
    missing_proof_plan_fields_command = current_command.replace(
        "required_supervised_proof_plan_summary_fields",
        "missing_supervised_proof_plan_summary_fields",
    )
    assert not progress.current_first_command_covers_baseline(
        missing_proof_plan_fields_command
    )
    assert "required_supervised_proof_plan_summary_fields" in (
        progress.current_first_command_missing_baseline_markers(
            missing_proof_plan_fields_command
        )
    )
    failing_candidate_prestart_command = current_command.replace(
        "readonly_preflight_review_decision_candidate_prestart_contract_valid: true",
        "readonly_preflight_review_decision_candidate_prestart_contract_valid: false",
    )
    assert not progress.current_first_command_covers_baseline(
        failing_candidate_prestart_command
    )
    assert (
        "readonly_preflight_review_decision_candidate_prestart_contract_valid: true"
        in progress.current_first_command_missing_baseline_markers(
            failing_candidate_prestart_command
        )
    )
    failing_candidate_prestart_source_command = current_command.replace(
        'readonly_preflight_review_decision_candidate_prestart_source_ids: ["artifacts/wing_readonly_preflight/run/summary.json"]',
        'readonly_preflight_review_decision_candidate_prestart_source_ids: ["artifacts/stale_preflight/run/summary.json"]',
    )
    assert not progress.current_first_command_covers_baseline(
        failing_candidate_prestart_source_command
    )
    assert (
        'readonly_preflight_review_decision_candidate_prestart_source_ids: ["artifacts/wing_readonly_preflight/run/summary.json"]'
        in progress.current_first_command_missing_baseline_markers(
            failing_candidate_prestart_source_command
        )
    )
    failing_candidate_request_sequence_command = current_command.replace(
        "readonly_preflight_review_decision_candidate_request_sequence_contract_valid: true",
        "readonly_preflight_review_decision_candidate_request_sequence_contract_valid: false",
    )
    assert not progress.current_first_command_covers_baseline(
        failing_candidate_request_sequence_command
    )
    assert (
        "readonly_preflight_review_decision_candidate_request_sequence_contract_valid: true"
        in progress.current_first_command_missing_baseline_markers(
            failing_candidate_request_sequence_command
        )
    )
    failing_candidate_request_sequence_source_command = current_command.replace(
        'readonly_preflight_review_decision_candidate_request_sequence_source_ids: ["artifacts/wing_readonly_preflight/run/summary.json"]',
        'readonly_preflight_review_decision_candidate_request_sequence_source_ids: ["artifacts/stale_preflight/run/summary.json"]',
    )
    assert not progress.current_first_command_covers_baseline(
        failing_candidate_request_sequence_source_command
    )
    assert (
        'readonly_preflight_review_decision_candidate_request_sequence_source_ids: ["artifacts/wing_readonly_preflight/run/summary.json"]'
        in progress.current_first_command_missing_baseline_markers(
            failing_candidate_request_sequence_source_command
        )
    )
    failing_candidate_artifact_root_command = current_command.replace(
        "readonly_preflight_review_decision_candidate_artifact_root_contract_valid: true",
        "readonly_preflight_review_decision_candidate_artifact_root_contract_valid: false",
    )
    assert not progress.current_first_command_covers_baseline(
        failing_candidate_artifact_root_command
    )
    assert (
        "readonly_preflight_review_decision_candidate_artifact_root_contract_valid: true"
        in progress.current_first_command_missing_baseline_markers(
            failing_candidate_artifact_root_command
        )
    )
    failing_candidate_artifact_roots_command = current_command.replace(
        'readonly_preflight_review_decision_candidate_artifact_roots: ["artifacts/wing_readonly_preflight"]',
        'readonly_preflight_review_decision_candidate_artifact_roots: ["artifacts/stale_preflight"]',
    )
    assert not progress.current_first_command_covers_baseline(
        failing_candidate_artifact_roots_command
    )
    assert (
        'readonly_preflight_review_decision_candidate_artifact_roots: ["artifacts/wing_readonly_preflight"]'
        in progress.current_first_command_missing_baseline_markers(
            failing_candidate_artifact_roots_command
        )
    )
    failing_expected_artifact_root_command = current_command.replace(
        "readonly_preflight_review_decision_expected_candidate_artifact_root_contract_violations_by_source: {}",
        "readonly_preflight_review_decision_expected_candidate_artifact_root_contract_violations_by_source: {'stale_source': ['missing artifact_root']}",
    )
    assert not progress.current_first_command_covers_baseline(
        failing_expected_artifact_root_command
    )
    assert (
        "readonly_preflight_review_decision_expected_candidate_artifact_root_contract_violations_by_source: {}"
        in progress.current_first_command_missing_baseline_markers(
            failing_expected_artifact_root_command
        )
    )
    failing_verifier_ws_proof_command = current_command.replace(
        "readonly_preflight_review_decision_candidate_verifier_ws_proof_complete: true",
        "readonly_preflight_review_decision_candidate_verifier_ws_proof_complete: false",
    )
    assert not progress.current_first_command_covers_baseline(
        failing_verifier_ws_proof_command
    )
    assert (
        "readonly_preflight_review_decision_candidate_verifier_ws_proof_complete: true"
        in progress.current_first_command_missing_baseline_markers(
            failing_verifier_ws_proof_command
        )
    )
    failing_verifier_ws_proof_source_command = current_command.replace(
        'readonly_preflight_review_decision_candidate_verifier_ws_proof_source_ids: ["artifacts/wing_readonly_preflight/run/summary.json"]',
        'readonly_preflight_review_decision_candidate_verifier_ws_proof_source_ids: ["artifacts/stale_preflight/run/summary.json"]',
    )
    assert not progress.current_first_command_covers_baseline(
        failing_verifier_ws_proof_source_command
    )
    assert (
        'readonly_preflight_review_decision_candidate_verifier_ws_proof_source_ids: ["artifacts/wing_readonly_preflight/run/summary.json"]'
        in progress.current_first_command_missing_baseline_markers(
            failing_verifier_ws_proof_source_command
        )
    )
    missing_contact_query_trace_command = current_command.replace(
        "readonly_preflight_review_decision_accepted_candidate_queries_console_by_source",
        "readonly_preflight_review_decision_accepted_candidate_queries_console_missing",
    )
    assert not progress.current_first_command_covers_baseline(
        missing_contact_query_trace_command
    )
    assert (
        'readonly_preflight_review_decision_accepted_candidate_queries_console_by_source: {"artifacts/wing_readonly_preflight/run/summary.json": false}'
        in progress.current_first_command_missing_baseline_markers(
            missing_contact_query_trace_command
        )
    )
    missing_contact_trace_command = current_command.replace(
        "readonly_preflight_review_decision_accepted_candidate_contact_trace_complete: true",
        "readonly_preflight_review_decision_accepted_candidate_contact_trace_complete: false",
    )
    assert not progress.current_first_command_covers_baseline(
        missing_contact_trace_command
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_contact_trace_complete: true"
        in progress.current_first_command_missing_baseline_markers(
            missing_contact_trace_command
        )
    )
    failing_contact_trace_source_command = current_command.replace(
        'readonly_preflight_review_decision_accepted_candidate_contact_trace_source_ids: ["artifacts/wing_readonly_preflight/run/summary.json"]',
        'readonly_preflight_review_decision_accepted_candidate_contact_trace_source_ids: ["artifacts/stale_preflight/run/summary.json"]',
    )
    assert not progress.current_first_command_covers_baseline(
        failing_contact_trace_source_command
    )
    assert (
        'readonly_preflight_review_decision_accepted_candidate_contact_trace_source_ids: ["artifacts/wing_readonly_preflight/run/summary.json"]'
        in progress.current_first_command_missing_baseline_markers(
            failing_contact_trace_source_command
        )
    )
    failing_expected_contact_backend_trace_command = current_command.replace(
        'readonly_preflight_review_decision_accepted_candidate_expected_contacts_backend_by_source: {"artifacts/wing_readonly_preflight/run/summary.json": true}',
        'readonly_preflight_review_decision_accepted_candidate_expected_contacts_backend_by_source: {"artifacts/wing_readonly_preflight/run/summary.json": false}',
    )
    assert not progress.current_first_command_covers_baseline(
        failing_expected_contact_backend_trace_command
    )
    assert (
        'readonly_preflight_review_decision_accepted_candidate_expected_contacts_backend_by_source: {"artifacts/wing_readonly_preflight/run/summary.json": true}'
        in progress.current_first_command_missing_baseline_markers(
            failing_expected_contact_backend_trace_command
        )
    )
    failing_expected_console_query_trace_command = current_command.replace(
        'readonly_preflight_review_decision_accepted_candidate_expected_queries_console_by_source: {"artifacts/wing_readonly_preflight/run/summary.json": true}',
        'readonly_preflight_review_decision_accepted_candidate_expected_queries_console_by_source: {"artifacts/wing_readonly_preflight/run/summary.json": false}',
    )
    assert not progress.current_first_command_covers_baseline(
        failing_expected_console_query_trace_command
    )
    assert (
        'readonly_preflight_review_decision_accepted_candidate_expected_queries_console_by_source: {"artifacts/wing_readonly_preflight/run/summary.json": true}'
        in progress.current_first_command_missing_baseline_markers(
            failing_expected_console_query_trace_command
        )
    )
    failing_contact_backend_trace_command = current_command.replace(
        'readonly_preflight_review_decision_accepted_candidate_contacts_backend_by_source: {"artifacts/wing_readonly_preflight/run/summary.json": true}',
        'readonly_preflight_review_decision_accepted_candidate_contacts_backend_by_source: {"artifacts/wing_readonly_preflight/run/summary.json": false}',
    )
    assert not progress.current_first_command_covers_baseline(
        failing_contact_backend_trace_command
    )
    assert (
        'readonly_preflight_review_decision_accepted_candidate_contacts_backend_by_source: {"artifacts/wing_readonly_preflight/run/summary.json": true}'
        in progress.current_first_command_missing_baseline_markers(
            failing_contact_backend_trace_command
        )
    )
    failing_console_query_trace_command = current_command.replace(
        'readonly_preflight_review_decision_accepted_candidate_queries_console_by_source: {"artifacts/wing_readonly_preflight/run/summary.json": false}',
        'readonly_preflight_review_decision_accepted_candidate_queries_console_by_source: {"artifacts/wing_readonly_preflight/run/summary.json": true}',
    )
    assert not progress.current_first_command_covers_baseline(
        failing_console_query_trace_command
    )
    assert (
        'readonly_preflight_review_decision_accepted_candidate_queries_console_by_source: {"artifacts/wing_readonly_preflight/run/summary.json": false}'
        in progress.current_first_command_missing_baseline_markers(
            failing_console_query_trace_command
        )
    )
    missing_contact_source_trace_command = current_command.replace(
        "readonly_preflight_review_decision_accepted_candidate_contact_trace_missing_source_ids: []",
        "readonly_preflight_review_decision_accepted_candidate_contact_trace_missing_source_ids: ['stale_source']",
    )
    assert not progress.current_first_command_covers_baseline(
        missing_contact_source_trace_command
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_contact_trace_missing_source_ids: []"
        in progress.current_first_command_missing_baseline_markers(
            missing_contact_source_trace_command
        )
    )
    failing_contact_value_gate_command = current_command.replace(
        "next_wing_proof_plan_accepted_candidate_contact_trace_value_gate_passed: true",
        "next_wing_proof_plan_accepted_candidate_contact_trace_value_gate_passed: false",
    )
    assert not progress.current_first_command_covers_baseline(
        failing_contact_value_gate_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_contact_trace_value_gate_passed: true"
        in progress.current_first_command_missing_baseline_markers(
            failing_contact_value_gate_command
        )
    )
    granting_contact_value_gate_live_authority_command = current_command.replace(
        "next_wing_proof_plan_accepted_candidate_contact_trace_value_gate_live_authority_granted: false",
        "next_wing_proof_plan_accepted_candidate_contact_trace_value_gate_live_authority_granted: true",
    )
    assert not progress.current_first_command_covers_baseline(
        granting_contact_value_gate_live_authority_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_contact_trace_value_gate_live_authority_granted: false"
        in progress.current_first_command_missing_baseline_markers(
            granting_contact_value_gate_live_authority_command
        )
    )
    closing_readiness_contact_value_gate_command = current_command.replace(
        "next_wing_proof_plan_accepted_candidate_contact_trace_value_gate_readiness_closure_allowed: false",
        "next_wing_proof_plan_accepted_candidate_contact_trace_value_gate_readiness_closure_allowed: true",
    )
    assert not progress.current_first_command_covers_baseline(
        closing_readiness_contact_value_gate_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_contact_trace_value_gate_readiness_closure_allowed: false"
        in progress.current_first_command_missing_baseline_markers(
            closing_readiness_contact_value_gate_command
        )
    )
    granting_contact_summary_surface_live_authority_command = current_command.replace(
        "next_wing_proof_plan_accepted_candidate_contact_trace_value_gate_summary_surface_live_authority_granted: false",
        "next_wing_proof_plan_accepted_candidate_contact_trace_value_gate_summary_surface_live_authority_granted: true",
    )
    assert not progress.current_first_command_covers_baseline(
        granting_contact_summary_surface_live_authority_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_contact_trace_value_gate_summary_surface_live_authority_granted: false"
        in progress.current_first_command_missing_baseline_markers(
            granting_contact_summary_surface_live_authority_command
        )
    )
    closing_readiness_contact_summary_surface_command = current_command.replace(
        "next_wing_proof_plan_accepted_candidate_contact_trace_value_gate_summary_surface_readiness_closure_allowed: false",
        "next_wing_proof_plan_accepted_candidate_contact_trace_value_gate_summary_surface_readiness_closure_allowed: true",
    )
    assert not progress.current_first_command_covers_baseline(
        closing_readiness_contact_summary_surface_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_contact_trace_value_gate_summary_surface_readiness_closure_allowed: false"
        in progress.current_first_command_missing_baseline_markers(
            closing_readiness_contact_summary_surface_command
        )
    )
    failing_request_sequence_value_gate_command = current_command.replace(
        "next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_passed: true",
        "next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_passed: false",
    )
    assert not progress.current_first_command_covers_baseline(
        failing_request_sequence_value_gate_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_passed: true"
        in progress.current_first_command_missing_baseline_markers(
            failing_request_sequence_value_gate_command
        )
    )
    granting_request_sequence_value_gate_live_authority_command = (
        current_command.replace(
            "next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_live_authority_granted: false",
            "next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_live_authority_granted: true",
        )
    )
    assert not progress.current_first_command_covers_baseline(
        granting_request_sequence_value_gate_live_authority_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_live_authority_granted: false"
        in progress.current_first_command_missing_baseline_markers(
            granting_request_sequence_value_gate_live_authority_command
        )
    )
    closing_readiness_request_sequence_value_gate_command = current_command.replace(
        "next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_readiness_closure_allowed: false",
        "next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_readiness_closure_allowed: true",
    )
    assert not progress.current_first_command_covers_baseline(
        closing_readiness_request_sequence_value_gate_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_readiness_closure_allowed: false"
        in progress.current_first_command_missing_baseline_markers(
            closing_readiness_request_sequence_value_gate_command
        )
    )
    granting_request_sequence_summary_surface_live_authority_command = (
        current_command.replace(
            "next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_summary_surface_live_authority_granted: false",
            "next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_summary_surface_live_authority_granted: true",
        )
    )
    assert not progress.current_first_command_covers_baseline(
        granting_request_sequence_summary_surface_live_authority_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_summary_surface_live_authority_granted: false"
        in progress.current_first_command_missing_baseline_markers(
            granting_request_sequence_summary_surface_live_authority_command
        )
    )
    closing_readiness_request_sequence_summary_surface_command = (
        current_command.replace(
            "next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_summary_surface_readiness_closure_allowed: false",
            "next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_summary_surface_readiness_closure_allowed: true",
        )
    )
    assert not progress.current_first_command_covers_baseline(
        closing_readiness_request_sequence_summary_surface_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_request_sequence_trace_value_gate_summary_surface_readiness_closure_allowed: false"
        in progress.current_first_command_missing_baseline_markers(
            closing_readiness_request_sequence_summary_surface_command
        )
    )
    failing_request_sequence_trace_source_command = current_command.replace(
        'readonly_preflight_review_decision_accepted_candidate_request_sequence_trace_source_ids: ["artifacts/wing_readonly_preflight/run/summary.json"]',
        'readonly_preflight_review_decision_accepted_candidate_request_sequence_trace_source_ids: ["artifacts/stale_preflight/run/summary.json"]',
    )
    assert not progress.current_first_command_covers_baseline(
        failing_request_sequence_trace_source_command
    )
    assert (
        'readonly_preflight_review_decision_accepted_candidate_request_sequence_trace_source_ids: ["artifacts/wing_readonly_preflight/run/summary.json"]'
        in progress.current_first_command_missing_baseline_markers(
            failing_request_sequence_trace_source_command
        )
    )
    failing_request_sequence_match_trace_command = current_command.replace(
        'readonly_preflight_review_decision_accepted_candidate_request_sequence_matches_plan_by_source: {"artifacts/wing_readonly_preflight/run/summary.json": true}',
        'readonly_preflight_review_decision_accepted_candidate_request_sequence_matches_plan_by_source: {"artifacts/wing_readonly_preflight/run/summary.json": false}',
    )
    assert not progress.current_first_command_covers_baseline(
        failing_request_sequence_match_trace_command
    )
    assert (
        'readonly_preflight_review_decision_accepted_candidate_request_sequence_matches_plan_by_source: {"artifacts/wing_readonly_preflight/run/summary.json": true}'
        in progress.current_first_command_missing_baseline_markers(
            failing_request_sequence_match_trace_command
        )
    )
    missing_request_sequence_source_trace_command = current_command.replace(
        "readonly_preflight_review_decision_accepted_candidate_request_sequence_trace_missing_source_ids: []",
        "readonly_preflight_review_decision_accepted_candidate_request_sequence_trace_missing_source_ids: ['stale_source']",
    )
    assert not progress.current_first_command_covers_baseline(
        missing_request_sequence_source_trace_command
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_request_sequence_trace_missing_source_ids: []"
        in progress.current_first_command_missing_baseline_markers(
            missing_request_sequence_source_trace_command
        )
    )
    failing_expected_artifact_value_gate_command = current_command.replace(
        "next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_passed: true",
        "next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_passed: false",
    )
    assert not progress.current_first_command_covers_baseline(
        failing_expected_artifact_value_gate_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_passed: true"
        in progress.current_first_command_missing_baseline_markers(
            failing_expected_artifact_value_gate_command
        )
    )
    granting_expected_artifact_value_gate_live_authority_command = (
        current_command.replace(
            "next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_live_authority_granted: false",
            "next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_live_authority_granted: true",
        )
    )
    assert not progress.current_first_command_covers_baseline(
        granting_expected_artifact_value_gate_live_authority_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_live_authority_granted: false"
        in progress.current_first_command_missing_baseline_markers(
            granting_expected_artifact_value_gate_live_authority_command
        )
    )
    closing_readiness_expected_artifact_value_gate_command = current_command.replace(
        "next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_readiness_closure_allowed: false",
        "next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_readiness_closure_allowed: true",
    )
    assert not progress.current_first_command_covers_baseline(
        closing_readiness_expected_artifact_value_gate_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_readiness_closure_allowed: false"
        in progress.current_first_command_missing_baseline_markers(
            closing_readiness_expected_artifact_value_gate_command
        )
    )
    granting_expected_artifact_summary_surface_live_authority_command = (
        current_command.replace(
            "next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_summary_surface_live_authority_granted: false",
            "next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_summary_surface_live_authority_granted: true",
        )
    )
    assert not progress.current_first_command_covers_baseline(
        granting_expected_artifact_summary_surface_live_authority_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_summary_surface_live_authority_granted: false"
        in progress.current_first_command_missing_baseline_markers(
            granting_expected_artifact_summary_surface_live_authority_command
        )
    )
    closing_readiness_expected_artifact_summary_surface_command = (
        current_command.replace(
            "next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_summary_surface_readiness_closure_allowed: false",
            "next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_summary_surface_readiness_closure_allowed: true",
        )
    )
    assert not progress.current_first_command_covers_baseline(
        closing_readiness_expected_artifact_summary_surface_command
    )
    assert (
        "next_wing_proof_plan_accepted_candidate_expected_artifact_trace_value_gate_summary_surface_readiness_closure_allowed: false"
        in progress.current_first_command_missing_baseline_markers(
            closing_readiness_expected_artifact_summary_surface_command
        )
    )
    failing_expected_artifact_trace_source_command = current_command.replace(
        'readonly_preflight_review_decision_accepted_candidate_expected_artifact_trace_source_ids: ["artifacts/wing_readonly_preflight/run/summary.json"]',
        'readonly_preflight_review_decision_accepted_candidate_expected_artifact_trace_source_ids: ["artifacts/stale_preflight/run/summary.json"]',
    )
    assert not progress.current_first_command_covers_baseline(
        failing_expected_artifact_trace_source_command
    )
    assert (
        'readonly_preflight_review_decision_accepted_candidate_expected_artifact_trace_source_ids: ["artifacts/wing_readonly_preflight/run/summary.json"]'
        in progress.current_first_command_missing_baseline_markers(
            failing_expected_artifact_trace_source_command
        )
    )
    missing_expected_artifact_source_trace_command = current_command.replace(
        "readonly_preflight_review_decision_accepted_candidate_expected_artifact_trace_missing_source_ids: []",
        "readonly_preflight_review_decision_accepted_candidate_expected_artifact_trace_missing_source_ids: ['stale_source']",
    )
    assert not progress.current_first_command_covers_baseline(
        missing_expected_artifact_source_trace_command
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_expected_artifact_trace_missing_source_ids: []"
        in progress.current_first_command_missing_baseline_markers(
            missing_expected_artifact_source_trace_command
        )
    )
    missing_expected_artifact_trace_command = current_command.replace(
        "readonly_preflight_review_decision_accepted_candidate_expected_artifact_trace_complete: true",
        "readonly_preflight_review_decision_accepted_candidate_expected_artifact_trace_complete: false",
    )
    assert not progress.current_first_command_covers_baseline(
        missing_expected_artifact_trace_command
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_expected_artifact_trace_complete: true"
        in progress.current_first_command_missing_baseline_markers(
            missing_expected_artifact_trace_command
        )
    )
    missing_request_sequence_trace_command = current_command.replace(
        "readonly_preflight_review_decision_accepted_candidate_request_sequence_trace_complete: true",
        "readonly_preflight_review_decision_accepted_candidate_request_sequence_trace_complete: false",
    )
    assert not progress.current_first_command_covers_baseline(
        missing_request_sequence_trace_command
    )
    assert (
        "readonly_preflight_review_decision_accepted_candidate_request_sequence_trace_complete: true"
        in progress.current_first_command_missing_baseline_markers(
            missing_request_sequence_trace_command
        )
    )
    missing_markdown_summary_command = current_command.replace(
        "operator-readable no-WING `summary.md`",
        "machine-only no-WING summary",
    )
    assert not progress.current_first_command_covers_baseline(
        missing_markdown_summary_command
    )
    assert "operator-readable no-WING `summary.md`" in (
        progress.current_first_command_missing_baseline_markers(
            missing_markdown_summary_command
        )
    )
    missing_markdown_rendered_command = current_command.replace(
        "operator_markdown_summary_field_lines_rendered: true",
        "operator_markdown_summary_field_lines_rendered: false",
    )
    assert not progress.current_first_command_covers_baseline(
        missing_markdown_rendered_command
    )
    assert "operator_markdown_summary_field_lines_rendered: true" in (
        progress.current_first_command_missing_baseline_markers(
            missing_markdown_rendered_command
        )
    )
    missing_markdown_sha_command = current_command.replace(
        "operator_markdown_summary_sha256_available: true",
        "operator_markdown_summary_sha256_available: false",
    )
    assert not progress.current_first_command_covers_baseline(
        missing_markdown_sha_command
    )
    assert "operator_markdown_summary_sha256_available: true" in (
        progress.current_first_command_missing_baseline_markers(
            missing_markdown_sha_command
        )
    )
    missing_markdown_sha_match_command = current_command.replace(
        "operator_markdown_summary_sha256_matches_actual: true",
        "operator_markdown_summary_sha256_matches_actual: false",
    )
    assert not progress.current_first_command_covers_baseline(
        missing_markdown_sha_match_command
    )
    assert "operator_markdown_summary_sha256_matches_actual: true" in (
        progress.current_first_command_missing_baseline_markers(
            missing_markdown_sha_match_command
        )
    )
    missing_artifact_manifest_command = current_command.replace(
        "artifact_manifest_path: artifact-manifest.json",
        "artifact_manifest_path: missing",
    )
    assert not progress.current_first_command_covers_baseline(
        missing_artifact_manifest_command
    )
    assert "artifact_manifest_path: artifact-manifest.json" in (
        progress.current_first_command_missing_baseline_markers(
            missing_artifact_manifest_command
        )
    )
    missing_artifact_manifest_surface_command = current_command.replace(
        "artifact_manifest_summary_surface_required: true",
        "artifact_manifest_summary_surface_required: false",
    )
    assert not progress.current_first_command_covers_baseline(
        missing_artifact_manifest_surface_command
    )
    assert "artifact_manifest_summary_surface_required: true" in (
        progress.current_first_command_missing_baseline_markers(
            missing_artifact_manifest_surface_command
        )
    )
    missing_artifact_manifest_surface_rendered_command = current_command.replace(
        "artifact_manifest_summary_surface_field_lines_rendered: true",
        "artifact_manifest_summary_surface_field_lines_rendered: false",
    )
    assert not progress.current_first_command_covers_baseline(
        missing_artifact_manifest_surface_rendered_command
    )
    assert "artifact_manifest_summary_surface_field_lines_rendered: true" in (
        progress.current_first_command_missing_baseline_markers(
            missing_artifact_manifest_surface_rendered_command
        )
    )
    missing_artifact_manifest_contract_digest_command = current_command.replace(
        "artifact_manifest_contract_sha256_matches_expected: true",
        "artifact_manifest_contract_sha256_matches_expected: false",
    )
    assert not progress.current_first_command_covers_baseline(
        missing_artifact_manifest_contract_digest_command
    )
    assert "artifact_manifest_contract_sha256_matches_expected: true" in (
        progress.current_first_command_missing_baseline_markers(
            missing_artifact_manifest_contract_digest_command
        )
    )
    missing_artifact_manifest_cross_surface_command = current_command.replace(
        "artifact_manifest_contract_cross_surface_match: true",
        "artifact_manifest_contract_cross_surface_match: false",
    )
    assert not progress.current_first_command_covers_baseline(
        missing_artifact_manifest_cross_surface_command
    )
    assert "artifact_manifest_contract_cross_surface_match: true" in (
        progress.current_first_command_missing_baseline_markers(
            missing_artifact_manifest_cross_surface_command
        )
    )
    missing_artifact_manifest_count_match_command = current_command.replace(
        "artifact_manifest_artifact_count_matches_required: true",
        "artifact_manifest_artifact_count_matches_required: false",
    )
    assert not progress.current_first_command_covers_baseline(
        missing_artifact_manifest_count_match_command
    )
    assert "artifact_manifest_artifact_count_matches_required: true" in (
        progress.current_first_command_missing_baseline_markers(
            missing_artifact_manifest_count_match_command
        )
    )
    missing_artifact_manifest_closure_command = current_command.replace(
        "artifact_manifest_readiness_closure_allowed: false",
        "artifact_manifest_readiness_closure_allowed: true",
    )
    assert not progress.current_first_command_covers_baseline(
        missing_artifact_manifest_closure_command
    )
    assert "artifact_manifest_readiness_closure_allowed: false" in (
        progress.current_first_command_missing_baseline_markers(
            missing_artifact_manifest_closure_command
        )
    )
    missing_artifact_manifest_ids_command = current_command.replace(
        "artifact_manifest_required_artifacts_present: true",
        "artifact_manifest_required_artifacts_present: false",
    )
    assert not progress.current_first_command_covers_baseline(
        missing_artifact_manifest_ids_command
    )
    assert "artifact_manifest_required_artifacts_present: true" in (
        progress.current_first_command_missing_baseline_markers(
            missing_artifact_manifest_ids_command
        )
    )
    missing_director_action_digest_command = current_command.replace(
        "project_director_selected_action_sha256",
        "missing_director_action_digest",
    )
    assert not progress.current_first_command_covers_baseline(
        missing_director_action_digest_command
    )
    assert "project_director_selected_action_sha256" in (
        progress.current_first_command_missing_baseline_markers(
            missing_director_action_digest_command
        )
    )
    missing_expected_digest_command = current_command.replace(
        "expected_project_director_selected_action_sha256",
        "missing_expected_director_action_digest",
    )
    assert not progress.current_first_command_covers_baseline(
        missing_expected_digest_command
    )
    assert "expected_project_director_selected_action_sha256" in (
        progress.current_first_command_missing_baseline_markers(
            missing_expected_digest_command
        )
    )
    report = progress.build_progress_report(
        rows,
        generated_at="2026-06-06T00:00:00Z",
        current_first_command=current_command,
    )

    assert "Total blockers: 9" in report
    assert "Closed blockers: 0" in report
    assert "Current first open blocker: WING read-only connection proof" in report
    assert "Next required action source: Current First Command" in report
    assert "Current First Command baseline coverage required: true" in report
    assert "Current First Command baseline coverage passed: true" in report
    assert (
        "Current First Command baseline coverage marker count: "
        f"{len(progress.CURRENT_FIRST_COMMAND_BASELINE_COVERAGE_MARKERS)}"
        in report
    )
    assert "Current First Command baseline missing markers: []" in report
    assert (
        f"Selected evidence file count: {len(progress.extract_evidence_files(rows[0]['evidence']))}"
        in report
    )
    assert "backend_port_prestart_gate" in report
    assert "expected_backend_port_prestart_gate" in report
    assert "backend_port_prestart_gate.grants_live_authority: false" in report
    assert (
        "backend_port_prestart_gate.must_pass_before_backend_wing_session: true"
        in report
    )
    assert "backend_port_prestart_gate.operator_sequence_step_index: 0" in report
    assert "backend_port_prestart_gate.runs_command: prestart_backend_port_check" in report
    assert "backend_port_prestart_gate.required_markers" in report
    assert "Before starting the intended backend/WING session" in report
    assert "backend_port_prestart_check_command" in report
    assert "backend_port_prestart_check_command_text" in report
    assert "PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick" in report
    assert (
        "failed busy-port no-WING summary and no-WING handoff summary must both show"
        in report
    )
    assert "backend_port_prestart_check_required: true" in report
    assert (
        "backend_port_prestart_check_must_pass_before_starting_backend: true"
        in report
    )
    assert "backend_port_prestart_gate_execution_allowed: false" in report
    assert "backend_port_prestart_gate_readiness_closure_allowed: false" in report
    assert "backend_port_prestart_gate_live_authority_granted: false" in report
    assert "backend_port_prestart_check_failure_detail_fields" in report
    assert "listener_processes" in report
    assert "listener_process_error" in report
    assert "safe_to_stop_listener_by_gate" in report
    assert "operator_next_step" in report
    for marker in ENVIRONMENT_BLOCKED_FRONTEND_MARKERS:
        assert marker in report
    assert "listener_confirmation_template_path" in report
    assert "listener_confirmation_template_acceptance_status" in report
    assert "listener_confirmation_template_accepted_by_template: false" in report
    assert (
        "listener_confirmation_template_execution_allowed_by_template: false"
        in report
    )
    assert (
        "listener_confirmation_template_readiness_closure_allowed_by_template: false"
        in report
    )
    assert "listener_confirmation_template_operator_placeholder_fields" in report
    assert "listener_confirmation_template_required_edits" in report
    assert "set template_status to operator_reviewed_and_edited" in report
    assert "set accepted_by_template to true" in report
    assert "REPLACE_WITH_*" in report
    assert "template_status: operator_reviewed_and_edited" in report
    assert "accepted_by_template: true" in report
    assert "manual_review_required_before_readiness_update: true" in report
    assert "readiness_matrix_update_allowed_by_packet: false" in report
    assert "readiness_closure_allowed_by_packet: false" in report
    assert "read_only: true" in report
    assert "live_write: false" in report
    assert "mutates_console_state: false" in report
    assert "queries_console: false" in report
    assert "executes_tests: false" in report
    assert "operator_required: true" in report
    assert "expected_plan_required_for_real_run: true" in report
    assert "backend_port_prestart_summary_type: no_wing_release_check" in report
    assert (
        "backend_port_prestart_summary_connection_policy: "
        "no_hardware_no_network_no_live_writes"
    ) in report
    assert "backend_port_prestart_summary_read_only: true" in report
    assert "backend_port_prestart_summary_live_write: false" in report
    assert "backend_port_prestart_summary_contacts_backend: false" in report
    assert "backend_port_prestart_summary_contacts_wing: false" in report
    assert "backend_port_prestart_summary_mutates_console_state: false" in report
    assert "backend_port_prestart_summary_queries_console: false" in report
    assert "backend_port_prestart_summary_executes_tests: false" in report
    for marker in (
        "backend_port_prestart_summary_type_matches_plan: true",
        "backend_port_prestart_summary_status_matches_plan: true",
        "backend_port_prestart_summary_connection_policy_matches_plan: true",
        "backend_port_prestart_summary_read_only_matches_plan: true",
        "backend_port_prestart_summary_live_write_matches_plan: true",
        "backend_port_prestart_summary_contacts_backend_matches_plan: true",
        "backend_port_prestart_summary_contacts_wing_matches_plan: true",
        "backend_port_prestart_summary_mutates_console_state_matches_plan: true",
        "backend_port_prestart_summary_queries_console_matches_plan: true",
        "backend_port_prestart_summary_executes_tests_matches_plan: true",
        "plan_safe_to_stop_listener_by_gate: false",
        "plan_live_authority_granted: false",
        "plan_backend_port_prestart_gate_execution_allowed: false",
        "plan_backend_port_prestart_gate_readiness_closure_allowed: false",
        "plan_backend_port_prestart_gate_live_authority_granted: false",
        "backend_port_prestart_check_status_matches_plan: true",
        "safe_to_stop_listener_by_gate_matches_plan: true",
        "live_authority_granted_matches_plan: true",
        "backend_port_prestart_gate_execution_allowed_matches_plan: true",
        "backend_port_prestart_gate_readiness_closure_allowed_matches_plan: true",
        "backend_port_prestart_gate_live_authority_granted_matches_plan: true",
        "declared_safe_to_stop_listener_by_gate_matches_plan: true",
        "declared_live_authority_granted_matches_plan: true",
        "declared_backend_port_prestart_gate_execution_allowed_matches_plan: true",
        "declared_backend_port_prestart_gate_readiness_closure_allowed_matches_plan: true",
        "declared_backend_port_prestart_gate_live_authority_granted_matches_plan: true",
    ):
        assert marker in report
    assert "physical_preflight_requires_existing_backend_connection: true" in report
    assert "backend_port_prestart_check_required: true" in report
    assert (
        "backend_port_prestart_check_must_pass_before_starting_backend: true"
        in report
    )
    assert "backend_port_prestart_gate_execution_allowed: false" in report
    assert "backend_port_prestart_gate_readiness_closure_allowed: false" in report
    assert "backend_port_prestart_gate_live_authority_granted: false" in report
    assert "execution_allowed_by_packet: false" in report
    assert "live_authority_granted: false" in report
    assert "full_autonomous_ready: false" in report
    assert (
        "next_manual_artifact_required: real wing_readonly_preflight.v1 summary.json"
        in report
    )
    assert "expected_artifacts" in report
    assert "expected_artifact_globs" in report
    assert "plan_expected_artifacts" in report
    assert "expected_artifacts_match_plan: true" in report
    assert "summary_glob_matches_plan: true" in report
    assert "event_log_glob_matches_plan: true" in report
    assert "accepted_candidate_prestart_contract_violations_by_source" in report
    assert (
        "readonly_preflight_review_decision_candidate_prestart_contract_violations_by_source"
        in report
    )
    assert "Both maps must be `{}` before supervised proof steps are planned" in report
    for marker in (
        "preview_command_artifact_root_matches_packet: true",
        "preview_command_plan_output_matches_packet: true",
        "validate_inputs_command_uses_validate_run_inputs_only: true",
        "validate_inputs_command_uses_plan_only: false",
        "validate_inputs_command_artifact_root_matches_packet: true",
        "validate_inputs_command_expected_plan_matches_packet: true",
        "capture_command_artifact_root_matches_packet: true",
        "capture_command_expected_plan_matches_packet: true",
        "review_command_artifact_root_matches_packet: true",
        "decision_review_command_artifact_root_matches_packet: true",
        "decision_review_command_decision_root_matches_packet: true",
    ):
        assert marker in report
    assert "backend_port_prestart_check_operator_sequence_step" in report
    assert "prestart_backend_port_check" in report
    assert "unexpected listeners" in report
    assert "backend_connection_confirmation_step_index: 3" in report
    assert "operator_sequence_has_backend_connection_confirmation: true" in report
    assert "operator_sequence_backend_connection_confirmation_step_index: 3" in report
    assert (
        "operator_sequence_backend_connection_confirmation_before_capture: true"
        in report
    )
    assert "operator_sequence_backend_connection_confirmation_step" in report
    assert "existing backend connection" in report
    assert "intended WING Rack" in report
    assert "read-only inspection" in report
    assert "operator_sequence_validate_readonly_preflight_inputs_step_index: 5" in report
    assert (
        "operator_sequence_validate_readonly_preflight_inputs_step_matches_contract: true"
        in report
    )
    assert "operator_sequence_capture_readonly_preflight_step_index: 6" in report
    assert (
        "backend_port_prestart_gate.operator_sequence_step_must_be_first: true"
        in report
    )
    assert "mvp_blocker_anchor" in report
    assert "mvp_blocker_anchor_matches_current_first_open: true" in report
    assert "mvp_blocker_anchor_sha256_matches_current_first_open: true" in report
    assert "project_director_selected_action_anchor" in report
    assert "expected_project_director_selected_action_anchor" in report
    assert (
        "project_director_selected_action_anchor_matches_current_directive: true"
        in report
    )
    assert "project_director_selected_action_source: Current First Command" in report
    assert (
        "expected_project_director_selected_action_source: Current First Command"
        in report
    )
    assert (
        "project_director_selected_action_source_matches_current_directive: true"
        in report
    )
    assert "project_director_selected_action_sha256" in report
    assert "actual_project_director_selected_action_sha256" in report
    assert "expected_project_director_selected_action_sha256" in report
    assert "project_director_selected_action_sha256_available: true" in report
    assert (
        "project_director_selected_action_sha256_available_matches_digest: true"
        in report
    )
    assert "project_director_selected_action_sha256_matches_anchor: true" in report
    assert "project_director_selected_action_sha256_matches_text: true" in report
    assert (
        "project_director_selected_action_sha256_matches_current_directive: true"
        in report
    )
    assert (
        "project_director_current_first_command_baseline_coverage_required: true"
        in report
    )
    assert (
        "project_director_current_first_command_baseline_coverage_passed: true"
        in report
    )
    assert (
        "project_director_current_first_command_baseline_coverage_marker_count"
        in report
    )
    assert "project_director_current_first_command_baseline_missing_markers: []" in report
    assert "operator_sequence_has_mvp_blocker_anchor_digest_review: true" in report
    assert "operator_sequence_count: 12" in report
    for marker in OPERATOR_SEQUENCE_SHA_MARKERS:
        assert marker in report
    for marker in OPERATOR_SEQUENCE_CONTRACT_MARKERS:
        assert marker in report
    for marker in DECLARED_OPERATOR_SEQUENCE_COUNT_MARKERS:
        assert marker in report
    assert "mvp_blocker_anchor_digest_review_step_index: 10" in report
    assert "expected_mvp_blocker_anchor_digest_review_step_index: 10" in report
    assert "mvp_blocker_anchor_digest_review_step_index_matches_contract: true" in report
    assert (
        "mvp_blocker_anchor_digest_review_step_index_matches_operator_sequence: true"
        in report
    )
    assert "operator_sequence_mvp_blocker_anchor_digest_review_step_index: 10" in report
    assert (
        "expected_operator_sequence_mvp_blocker_anchor_digest_review_step_index: 10"
        in report
    )
    assert (
        "operator_sequence_mvp_blocker_anchor_digest_review_step_index_matches_contract: true"
        in report
    )
    assert "operator_sequence_supervised_prestart_gate_is_final: true" in report
    assert "operator_sequence_supervised_prestart_gate_step_index: 11" in report
    assert "operator_sequence_has_supervised_prestart_gate: true" in report
    assert "required_supervised_prestart_gate_markers" in report
    assert "supervised_prestart_gate" in report
    assert "expected_supervised_prestart_gate" in report
    assert "supervised_prestart_gate.operator_sequence_step_index: 11" in report
    assert (
        "supervised_prestart_gate.operator_sequence_step_must_be_final: true"
        in report
    )
    assert (
        "supervised_prestart_gate.requires_operator_decision_status: accepted_decision_ready"
        in report
    )
    assert (
        "supervised_prestart_gate.unlocks_runbook: Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md"
        in report
    )
    assert "supervised_prestart_gate.requires_accepted_candidate_prestart_trace: true" in report
    assert "accepted-candidate prestart trace" in report
    assert "expected-artifact-bound accepted source id" in report
    assert (
        "accepted_readonly_preflight_candidate_prestart_contract_required_before_supervised_proof"
        in report
    )
    assert (
        "accepted_readonly_preflight_candidate_verifier_ws_proof_required_before_supervised_proof"
        in report
    )
    assert (
        "accepted_readonly_preflight_candidate_expected_artifact_trace_required_before_supervised_proof"
        in report
    )
    assert "final handoff operator-sequence step" in report
    assert "supervised_prestart_gate" in report
    assert "expected_supervised_prestart_gate" in report
    assert "Offline tooling, shadow" in report

    output = tmp_path / "MVP_PROGRESS_LATEST.md"
    progress.write_report(output, report)
    assert output.read_text(encoding="utf-8") == report


def test_correction_methods_roadmap_preserves_full_console_goal_without_live_widening():
    roadmap = CORRECTION_ROADMAP.read_text(encoding="utf-8")
    backlog = CORRECTION_BACKLOG.read_text(encoding="utf-8")

    for family in (
        "Gain / Trim",
        "Pan / Stereo Placement",
        "EQ",
        "Compressor",
        "FX / Sends",
        "Fader / Balance",
    ):
        assert family in roadmap

    assert "offline analysis" in roadmap
    assert "proposal-only" in roadmap
    assert "No method may skip stages." in roadmap
    assert "must not widen live writes" in roadmap
    assert "separate ADR after WING MVP" in backlog
    assert "Do not widen live writes" in backlog


def test_codex_prompt_requires_mvp_rails_before_new_work():
    text = PROMPT.read_text(encoding="utf-8")

    assert "Docs/MVP_WING_SUPERVISED.md" in text
    assert "Docs/CORRECTION_METHODS_ROADMAP.md" in text
    assert ".ai/project-layer/wing_mvp_blockers.md" in text
    assert "Docs/reports/MVP_PROGRESS_LATEST.md" in text
    assert "first non-closed row" in text
    assert "supervised_prestart_gate" in text
    assert "expected_supervised_prestart_gate" in text
    assert "backend_port_prestart_gate" in text
    assert "expected_backend_port_prestart_gate" in text
    assert "backend_port_prestart_gate.grants_live_authority: false" in text
    assert (
        "backend_port_prestart_gate.must_pass_before_backend_wing_session: true"
        in text
    )
    assert "backend_port_prestart_gate.operator_sequence_step_index: 0" in text
    assert "backend_port_prestart_gate.operator_sequence_step_must_be_first: true" in text
    assert "backend_port_prestart_gate.runs_command: prestart_backend_port_check" in text
    assert "backend_port_prestart_gate.required_markers" in text
    assert "Before starting the intended backend/WING session" in text
    assert "backend_port_prestart_check_command" in text
    assert "PYTHONPATH=backend python3 tools/no_wing_release_check.py --quick" in text
    assert "failed busy-port no-WING summary and no-WING handoff summary" in text
    assert "must both show" in text
    assert "backend_port_prestart_check_required: true" in text
    assert "backend_port_prestart_check_must_pass_before_starting_backend: true" in text
    assert "backend_port_prestart_gate_execution_allowed: false" in text
    assert "backend_port_prestart_gate_readiness_closure_allowed: false" in text
    assert "backend_port_prestart_gate_live_authority_granted: false" in text
    assert "backend_port_prestart_check_failure_detail_fields" in text
    assert "listener_processes" in text
    assert "listener_process_error" in text
    assert "safe_to_stop_listener_by_gate" in text
    assert "operator_next_step" in text
    for marker in ENVIRONMENT_BLOCKED_FRONTEND_MARKERS:
        assert marker in text
    assert "listener_confirmation_template_accepted_by_template: false" in text
    assert "listener_confirmation_template_execution_allowed_by_template: false" in text
    assert (
        "listener_confirmation_template_readiness_closure_allowed_by_template: false"
        in text
    )
    assert "listener_confirmation_template_operator_placeholder_fields" in text
    assert "listener_confirmation_template_required_edits" in text
    assert "set template_status to operator_reviewed_and_edited" in text
    assert "set accepted_by_template to true" in text
    assert "REPLACE_WITH_*" in text
    assert "template_status: operator_reviewed_and_edited" in text
    assert "accepted_by_template: true" in text
    assert "manual_review_required_before_readiness_update: true" in text
    assert "readiness_matrix_update_allowed_by_packet: false" in text
    assert "readiness_closure_allowed_by_packet: false" in text
    assert "unsafe `true` values" in text
    assert "failed handoff is auditable" in text
    assert "packet safety flags" in text
    assert "mutates_console_state" in text
    assert "read_only: true" in text
    assert "live_write: false" in text
    assert "mutates_console_state: false" in text
    assert "queries_console: false" in text
    assert "executes_tests: false" in text
    assert "operator_required: true" in text
    assert "expected_plan_required_for_real_run: true" in text
    assert "backend_port_prestart_summary_required: true" in text
    assert "backend_port_prestart_summary_path_placeholder" in text
    assert "preview_command_prestart_summary" in text
    assert "preview_command_prestart_summary_matches_plan: true" in text
    assert "validate_inputs_command_prestart_summary" in text
    assert "validate_inputs_command_prestart_summary_matches_plan: true" in text
    assert "capture_command_prestart_summary" in text
    assert "capture_command_prestart_summary_matches_plan: true" in text
    assert "backend_port_prestart_summary_type: no_wing_release_check" in text
    assert "backend_port_prestart_summary_status: passed" in text
    assert (
        "backend_port_prestart_summary_connection_policy: "
        "no_hardware_no_network_no_live_writes"
    ) in text
    assert "backend_port_prestart_summary_read_only: true" in text
    assert "backend_port_prestart_summary_live_write: false" in text
    assert "backend_port_prestart_summary_contacts_backend: false" in text
    assert "backend_port_prestart_summary_contacts_wing: false" in text
    assert "backend_port_prestart_summary_mutates_console_state: false" in text
    assert "backend_port_prestart_summary_queries_console: false" in text
    assert "backend_port_prestart_summary_executes_tests: false" in text
    assert "backend_port_prestart_check_status: passed" in text
    for marker in (
        "backend_port_prestart_summary_type_matches_plan: true",
        "backend_port_prestart_summary_status_matches_plan: true",
        "backend_port_prestart_summary_connection_policy_matches_plan: true",
        "backend_port_prestart_summary_read_only_matches_plan: true",
        "backend_port_prestart_summary_live_write_matches_plan: true",
        "backend_port_prestart_summary_contacts_backend_matches_plan: true",
        "backend_port_prestart_summary_contacts_wing_matches_plan: true",
        "backend_port_prestart_summary_mutates_console_state_matches_plan: true",
        "backend_port_prestart_summary_queries_console_matches_plan: true",
        "backend_port_prestart_summary_executes_tests_matches_plan: true",
        "plan_safe_to_stop_listener_by_gate: false",
        "plan_live_authority_granted: false",
        "plan_backend_port_prestart_gate_execution_allowed: false",
        "plan_backend_port_prestart_gate_readiness_closure_allowed: false",
        "plan_backend_port_prestart_gate_live_authority_granted: false",
        "backend_port_prestart_check_status_matches_plan: true",
        "safe_to_stop_listener_by_gate_matches_plan: true",
        "live_authority_granted_matches_plan: true",
        "backend_port_prestart_gate_execution_allowed_matches_plan: true",
        "backend_port_prestart_gate_readiness_closure_allowed_matches_plan: true",
        "backend_port_prestart_gate_live_authority_granted_matches_plan: true",
        "declared_safe_to_stop_listener_by_gate_matches_plan: true",
        "declared_live_authority_granted_matches_plan: true",
        "declared_backend_port_prestart_gate_execution_allowed_matches_plan: true",
        "declared_backend_port_prestart_gate_readiness_closure_allowed_matches_plan: true",
        "declared_backend_port_prestart_gate_live_authority_granted_matches_plan: true",
    ):
        assert marker in text
    assert "physical_preflight_requires_existing_backend_connection: true" in text
    assert "backend_port_prestart_check_required: true" in text
    assert "backend_port_prestart_check_must_pass_before_starting_backend: true" in text
    assert "backend_port_prestart_gate_execution_allowed: false" in text
    assert "backend_port_prestart_gate_readiness_closure_allowed: false" in text
    assert "backend_port_prestart_gate_live_authority_granted: false" in text
    assert "execution_allowed_by_packet: false" in text
    assert "live_authority_granted: false" in text
    assert "full_autonomous_ready: false" in text
    assert (
        "next_manual_artifact_required: real wing_readonly_preflight.v1 summary.json"
        in text
    )
    assert "expected_artifacts" in text
    assert "expected_artifact_globs" in text
    assert "plan_expected_artifacts" in text
    assert "expected_artifacts_match_plan: true" in text
    assert "summary_glob_matches_plan: true" in text
    assert "event_log_glob_matches_plan: true" in text
    assert "backend_port_prestart_check_operator_sequence_step" in text
    assert "prestart_backend_port_check" in text
    assert "unexpected listeners" in text
    assert "supervised_prestart_gate.grants_live_authority: false" in text
    assert "supervised_prestart_gate.operator_sequence_step_index: 11" in text
    assert "supervised_prestart_gate.operator_sequence_step_must_be_final: true" in text
    assert "operator_sequence_count: 12" in text
    for marker in OPERATOR_SEQUENCE_SHA_MARKERS:
        assert marker in text
    for marker in OPERATOR_SEQUENCE_CONTRACT_MARKERS:
        assert marker in text
    for marker in DECLARED_OPERATOR_SEQUENCE_COUNT_MARKERS:
        assert marker in text
    assert "mvp_blocker_anchor_digest_review_step_index" in text
    assert "expected_mvp_blocker_anchor_digest_review_step_index" in text
    assert "mvp_blocker_anchor_digest_review_step_index_matches_contract" in text
    assert "mvp_blocker_anchor_digest_review_step_index_matches_operator_sequence" in text
    assert "operator_sequence_mvp_blocker_anchor_digest_review_step_index" in text
    assert "expected_operator_sequence_mvp_blocker_anchor_digest_review_step_index" in text
    assert (
        "operator_sequence_mvp_blocker_anchor_digest_review_step_index_matches_contract"
        in text
    )
    assert "operator_sequence_supervised_prestart_gate_step_index" in text
    assert "supervised_prestart_gate.requires_operator_decision_status:" in text
    assert "supervised_prestart_gate.unlocks_runbook:" in text
    assert "required_supervised_prestart_gate_markers" in text
    assert "required_supervised_proof_plan_summary_fields" in text
    assert "expected_required_supervised_proof_plan_summary_fields" in text
    assert "required_supervised_proof_plan_summary_fields_match_contract: true" in text
    assert (
        "required_supervised_proof_plan_summary_fields_include_invalid_records: true"
        in text
    )
    for marker in (
        'readonly_preflight_review_decision_candidate_verifier_ws_proof_source_ids: ["artifacts/wing_readonly_preflight/run/summary.json"]',
        "readonly_preflight_review_decision_candidate_verifier_ws_proof_violations_by_source: {}",
        "readonly_preflight_review_decision_candidate_verifier_ws_proof_complete: true",
    ):
        assert marker in text
    assert "operator-readable no-WING `summary.md`" in text
    assert "operator_markdown_summary_required: true" in text
    assert "operator_markdown_summary_field_lines_rendered: true" in text
    assert "operator_markdown_summary_missing_field_lines: []" in text
    assert "operator_markdown_summary_artifact: summary.md" in text
    assert "operator_markdown_summary_sha256" in text
    assert "operator_markdown_summary_sha256_available: true" in text
    assert "operator_markdown_summary_actual_sha256" in text
    assert "operator_markdown_summary_sha256_matches_actual: true" in text
    assert "artifact_manifest_summary_surface_required: true" in text
    assert "artifact_manifest_summary_surface_missing_fields: []" in text
    assert "artifact_manifest_summary_surface_required_field_lines" in text
    assert "artifact_manifest_summary_surface_missing_field_lines: []" in text
    assert "artifact_manifest_summary_surface_field_lines_rendered: true" in text
    assert "artifact_manifest_contract_id" in text
    assert "artifact_manifest_contract_required: true" in text
    assert "artifact_manifest_contract_fields" in text
    assert "artifact_manifest_contract_sha256" in text
    assert "artifact_manifest_contract_sha256_available: true" in text
    assert "artifact_manifest_contract_sha256_matches_expected: true" in text
    assert "artifact_manifest_contract_cross_surface_required: true" in text
    assert "artifact_manifest_contract_cross_surface_required_surfaces" in text
    assert "artifact_manifest_contract_cross_surface_required_fields" in text
    assert "artifact_manifest_contract_cross_surface_expected_match: true" in text
    assert "artifact_manifest_contract_cross_surface_match: true" in text
    assert "artifact_manifest_summary_json_contract_sha256_matches_manifest: true" in text
    assert "artifact_manifest_summary_md_contract_sha256_matches_manifest: true" in text
    assert "artifact_manifest_contract_cross_surface_mismatches: []" in text
    assert (
        "artifact_manifest_contract_cross_surface_readiness_closure_allowed: false"
        in text
    )
    assert (
        "artifact_manifest_contract_cross_surface_live_authority_granted: false"
        in text
    )
    assert "artifact_manifest_required_artifact_sha256_fields" in text
    assert "artifact_manifest_path: artifact-manifest.json" in text
    assert "artifact_manifest_artifact_count: 2" in text
    assert "artifact_manifest_required_artifact_count: 2" in text
    assert "artifact_manifest_artifact_count_matches_required: true" in text
    assert "artifact_manifest_required_artifact_ids" in text
    assert "artifact_manifest_artifact_ids" in text
    assert "artifact_manifest_required_artifacts_present: true" in text
    assert "artifact_manifest_required_artifact_paths" in text
    assert "artifact_manifest_artifact_paths" in text
    assert "artifact_manifest_required_artifact_paths_present: true" in text
    assert "artifact_manifest_required_artifact_sha256_available: true" in text
    assert "artifact_manifest_summary_json_sha256" in text
    assert "artifact_manifest_summary_json_sha256_available: true" in text
    assert "artifact_manifest_summary_md_sha256" in text
    assert "artifact_manifest_summary_md_sha256_available: true" in text
    assert "artifact_manifest_readiness_closure_allowed: false" in text
    assert "artifact_manifest_live_authority_granted: false" in text
    assert "backend_port_prestart_gate_matches_expected: true" in text
    assert "declared_backend_port_prestart_gate_matches_expected: true" in text
    assert "supervised_prestart_gate_matches_expected: true" in text
    assert "declared_supervised_prestart_gate_matches_expected: true" in text
    assert (
        'readonly_preflight_review_decision_expected_candidate_prestart_source_ids: ["artifacts/wing_readonly_preflight/run/summary.json"]'
        in text
    )
    assert (
        "readonly_preflight_review_decision_expected_candidate_prestart_contract_violations_by_source: {}"
        in text
    )


def test_docs_index_points_to_current_wing_mvp_prestart_gate():
    text = DOCS_INDEX.read_text(encoding="utf-8")

    assert "Current WING MVP rails" in text
    assert "Docs/runbooks/WING_READONLY_PREFLIGHT.md" in text
    assert "Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md" in text
    assert (
        "Docs/reports/AUTOMIXER_WING_READONLY_RUNBOOK_FINAL_PRESTART_GATE_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_HANDOFF_STRUCTURED_PRESTART_GATE_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_HANDOFF_STRUCTURED_GATE_SUMMARY_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_READONLY_PRESTART_SUMMARY_BINDING_20260613.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_PRESTART_SUMMARY_PLACEHOLDER_20260613.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BACKEND_LISTENER_CONFIRMATION_TEMPLATE_20260613.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BACKEND_LISTENER_TEMPLATE_EDIT_GATE_20260613.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BACKEND_LISTENER_TEMPLATE_EDIT_REQUIREMENT_CANON_20260613.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BACKEND_LISTENER_TEMPLATE_PENDING_SUMMARY_FIELDS_20260613.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BACKEND_LISTENER_TEMPLATE_AUTHORITY_SUMMARY_FIELDS_20260613.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BACKEND_PRESTART_TEMPLATE_FAILURE_DETAIL_FIELDS_20260613.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BACKEND_PRESTART_TEMPLATE_REVIEW_FIELDS_20260613.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BUSY_PORT_PRESTART_DETAIL_FIELDS_SUMMARY_20260613.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BUSY_PORT_PRESTART_COMMAND_SUMMARY_20260613.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BUSY_PORT_PRESTART_REQUIRED_FLAGS_SUMMARY_20260613.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BUSY_PORT_PRESTART_GATE_AUTHORITY_SUMMARY_20260613.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_EXPECTED_ARTIFACTS_CONTRACT_20260613.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_EXPECTED_ARTIFACTS_NEXT_ACTION_20260613.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_DENIAL_FLAG_SUMMARY_TRUTH_20260614.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_SAFETY_FLAG_SUMMARY_TRUTH_20260614.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_BACKEND_PRESTART_COMMAND_SUMMARY_20260614.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_DECISION_REVIEW_PRESTART_SUMMARY_FIELDS_20260614.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_DECISION_REVIEW_PRESTART_BOOL_RENDERING_20260614.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_DIRECTION_GUARD_SKIP_SELF_REPORTS_20260614.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_PRESTART_SUMMARY_SHA_BINDING_20260614.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_PRESTART_SUMMARY_SHA_AVAILABILITY_20260614.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_OPERATOR_SEQUENCE_CARDINALITY_20260614.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_ANCHOR_DIGEST_REVIEW_STEP_INDEX_20260614.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_DECLARED_ANCHOR_DIGEST_REVIEW_INDEX_20260614.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_DECLARED_OPERATOR_SEQUENCE_COUNT_20260614.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_OPERATOR_SEQUENCE_SHA256_20260614.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_HANDOFF_OPERATOR_SEQUENCE_CONTRACT_20260614.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_PROOF_PLAN_PRESTART_VIOLATIONS_MAP_SUMMARY_20260614.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BACKEND_LISTENER_PENDING_SUMMARY_FIELDS_CANON_20260613.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_READONLY_RUNBOOK_STRUCTURED_GATE_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_READONLY_HANDOFF_STRUCTURED_PRESTART_GATE_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_READONLY_HANDOFF_MVP_BLOCKER_ANCHOR_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_READONLY_HANDOFF_MVP_BLOCKER_ANCHOR_DIGEST_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BLOCKER_BACKEND_PRESTART_NEXT_ACTION_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BACKEND_PRESTART_GATE_STEP_CONTRACT_NEXT_ACTION_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BLOCKER_HANDOFF_ANCHOR_NEXT_ACTION_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BLOCKER_HANDOFF_ANCHOR_DIGEST_NEXT_ACTION_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_HANDOFF_ANCHOR_DIGEST_OPERATOR_REVIEW_STEP_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BLOCKER_HANDOFF_DIGEST_REVIEW_NEXT_ACTION_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BLOCKER_SUPERVISED_GATE_NEXT_ACTION_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_BLOCKER_SUPERVISED_TRACE_GATE_NEXT_ACTION_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_SUPERVISED_GATE_FINAL_STEP_CONTRACT_NEXT_ACTION_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_SUPERVISED_GATE_DECISION_RUNBOOK_CONTRACT_NEXT_ACTION_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_SUPERVISED_GATE_REQUIRED_MARKERS_NEXT_ACTION_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_MVP_OPERATOR_SEQUENCE_STEP_INDEXES_NEXT_ACTION_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_AUTOLOOP_BACKEND_PRESTART_DIRECTOR_DOCS_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_PROJECT_DIRECTOR_HANDOFF_REQUIRED_FILE_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_PROJECT_DIRECTOR_DECISION_PROOF_REQUIRED_FILES_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_WING_PROJECT_DIRECTOR_SELECTED_EVIDENCE_FILES_20260612.md"
        in text
    )
    assert (
        "Docs/reports/AUTOMIXER_PROJECT_GOAL_SUPERVISED_AUTOMIX_STRUCTURED_GATE_20260612.md"
        in text
    )
    assert "final handoff operator-sequence step" in text
    assert "accepted-candidate prestart gate" in text
    assert "backend_port_prestart_gate" in text
    assert "expected_backend_port_prestart_gate" in text
    assert "backend_port_prestart_gate.runs_command" in text
    assert "backend_port_prestart_gate.required_markers" in text
    assert "backend_port_prestart_check_command" in text
    assert "failed busy-port no-WING summary" in text
    assert "no-WING handoff summary must both show" in text
    assert "backend_port_prestart_check_required: true" in text
    assert "backend_port_prestart_check_must_pass_before_starting_backend: true" in text
    assert "backend_port_prestart_gate_execution_allowed: false" in text
    assert "backend_port_prestart_gate_readiness_closure_allowed: false" in text
    assert "backend_port_prestart_gate_live_authority_granted: false" in text
    assert "backend_port_prestart_check_failure_detail_fields" in text
    assert "listener_confirmation_template_path" in text
    assert "listener_confirmation_template_acceptance_status" in text
    assert "manual_review_required_before_readiness_update" in text
    assert "readiness_matrix_update_allowed_by_packet" in text
    assert "readiness_closure_allowed_by_packet" in text
    assert "live_write" in text
    assert "execution_allowed_by_packet" in text
    assert "backend_port_prestart_check_operator_sequence_step" in text
    assert "supervised_prestart_gate" in text
    assert "expected_supervised_prestart_gate" in text
    assert "operator_sequence_count" in text
    for marker in OPERATOR_SEQUENCE_SHA_MARKERS:
        assert marker.split(":", 1)[0] in text
    for marker in OPERATOR_SEQUENCE_CONTRACT_MARKERS:
        assert marker.split(":", 1)[0] in text
    for marker in DECLARED_OPERATOR_SEQUENCE_COUNT_FIELD_MARKERS:
        assert marker in text
    assert "mvp_blocker_anchor_digest_review_step_index" in text
    assert "expected_mvp_blocker_anchor_digest_review_step_index" in text
    assert "mvp_blocker_anchor_digest_review_step_index_matches_contract" in text
    assert "mvp_blocker_anchor_digest_review_step_index_matches_operator_sequence" in text
    assert "operator_sequence_mvp_blocker_anchor_digest_review_step_index" in text
    assert "expected_operator_sequence_mvp_blocker_anchor_digest_review_step_index" in text
    assert (
        "operator_sequence_mvp_blocker_anchor_digest_review_step_index_matches_contract"
        in text
    )
    assert "operator_sequence_supervised_prestart_gate_step_index" in text
