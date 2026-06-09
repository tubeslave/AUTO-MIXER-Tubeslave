"""Read-only registry for processing methods and pipeline profiles.

The registry is intentionally declarative. It must not import mixer clients or
DSP modules because Product Layer snapshots need to be safe in offline and live
WING deployments.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from operator_mode_policy import (
    normalize_manual_live_write_kinds,
    CURRENT_PROVEN_LIVE_WRITE_KINDS,
)

try:
    from bus_group_main_dca_physical_readback_contract import (
        build_bus_group_main_dca_physical_readback_contract,
    )
    from bus_group_main_dca_physical_readback_artifact_intake import (
        build_bus_group_main_dca_physical_readback_artifact_intake,
    )
    from channel_basic_controls_physical_readback_contract import (
        build_channel_basic_controls_physical_readback_contract,
    )
    from channel_basic_controls_physical_readback_artifact_intake import (
        build_channel_basic_controls_physical_readback_artifact_intake,
    )
    from channel_processing_physical_readback_contract import (
        build_channel_processing_physical_readback_contract,
    )
    from channel_processing_physical_readback_artifact_intake import (
        build_channel_processing_physical_readback_artifact_intake,
    )
    from dlive_state_physical_readback_contract import (
        build_dlive_state_physical_readback_contract,
    )
    from dlive_state_physical_readback_artifact_intake import (
        build_dlive_state_physical_readback_artifact_intake,
    )
    from fx_slots_inserts_physical_readback_contract import (
        build_fx_slots_inserts_physical_readback_contract,
    )
    from fx_slots_inserts_physical_readback_artifact_intake import (
        build_fx_slots_inserts_physical_readback_artifact_intake,
    )
    from matrix_state_physical_readback_contract import (
        build_matrix_state_physical_readback_contract,
    )
    from matrix_state_physical_readback_artifact_intake import (
        build_matrix_state_physical_readback_artifact_intake,
    )
    from autonomy_mode_transition_gate import build_autonomy_mode_transition_gate
    from autonomy_promotion_readiness import build_autonomy_promotion_readiness_snapshot
    from project_goal_capability_matrix import (
        build_project_goal_capability_matrix_snapshot,
        build_project_goal_capability_progress_snapshot,
        build_project_goal_objective_alignment_snapshot,
    )
    from goal_capability_readiness_normalization import (
        normalize_goal_capability_readiness_hotspots,
        normalize_goal_capability_readiness_rows,
    )
    from read_surface_fixture_evidence import build_read_surface_fixture_evidence_snapshot
    from read_surface_inventory import build_read_surface_inventory_snapshot
    from read_surface_physical_readback_capture_plan import (
        build_read_surface_physical_readback_capture_plan,
    )
    from read_surface_remediation_plan import build_read_surface_remediation_plan
    from raw_mixer_state_physical_readback_contract import (
        build_raw_mixer_state_physical_readback_contract,
    )
    from raw_mixer_state_physical_readback_artifact_intake import (
        build_raw_mixer_state_physical_readback_artifact_intake,
    )
    from read_surface_physical_readback_artifact_review_queue import (
        build_read_surface_physical_readback_artifact_review_queue,
    )
    from read_surface_physical_readback_fixture_acceptance_ledger import (
        build_read_surface_physical_readback_fixture_acceptance_ledger,
    )
    from read_surface_physical_readback_fixture_closure_plan import (
        build_read_surface_physical_readback_fixture_closure_plan,
    )
    from routing_sends_physical_readback_contract import (
        build_routing_sends_physical_readback_contract,
    )
    from routing_sends_physical_readback_artifact_intake import (
        build_routing_sends_physical_readback_artifact_intake,
    )
    from snapshots_scenes_physical_readback_contract import (
        build_snapshots_scenes_physical_readback_contract,
    )
    from snapshots_scenes_physical_readback_artifact_intake import (
        build_snapshots_scenes_physical_readback_artifact_intake,
    )
    from user_presets_physical_readback_contract import (
        build_user_presets_physical_readback_contract,
    )
    from user_presets_physical_readback_artifact_intake import (
        build_user_presets_physical_readback_artifact_intake,
    )
    from wing_readonly_preflight_evidence import build_wing_readonly_preflight_evidence
    from safety_coverage_matrix import build_safety_coverage_matrix_snapshot
    from source_identity_calibration_acceptance_ledger import (
        build_source_identity_calibration_acceptance_ledger,
    )
    from source_identity_calibration_persistence_plan import (
        build_source_identity_calibration_persistence_plan,
    )
    from source_identity_calibration_memory_draft import (
        build_source_identity_calibration_memory_draft,
    )
    from source_identity_calibration_memory_writer import (
        build_source_identity_calibration_memory_writer,
    )
    from source_identity_calibration_memory_index import (
        build_source_identity_calibration_memory_index,
    )
    from operator_proposal_apply_sequence_policy_memory_index import (
        build_operator_proposal_apply_sequence_policy_memory_index,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_review import (
        build_operator_proposal_apply_sequence_policy_memory_trust_review,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger import (
        build_operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_plan import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_plan,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_contract import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_contract,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_executor_dry_run import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_executor_dry_run,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_executor import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_executor,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_apply_audit import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_apply_audit,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_rollback_verification import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_rollback_verification,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_sequence_policy_gate import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_sequence_policy_gate,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_policy_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_policy_consumer,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering_preview import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering_preview,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_ordering_plan import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_ordering_plan,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_importer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_importer,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_plan import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_plan,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_result_consumer,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_sequence_plan_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_sequence_plan_result_consumer,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_dry_run_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_dry_run_result_consumer,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_gate_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_gate_result_consumer,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_result_consumer,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_audit_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_audit_result_consumer,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_audit_policy_gate_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_audit_policy_gate_result_consumer,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_draft_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_draft_result_consumer,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_dry_run_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_dry_run_result_consumer,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_apply_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_apply_result_consumer,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_index_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_index_result_consumer,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_review_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_review_result_consumer,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_acceptance_ledger_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_acceptance_ledger_result_consumer,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_plan_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_plan_result_consumer,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_contract_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_contract_result_consumer,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_acceptance_ledger_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_acceptance_ledger_result_consumer,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_plan_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_plan_result_consumer,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_dry_run_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_dry_run_result_consumer,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_result_consumer,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_apply_audit_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_apply_audit_result_consumer,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_rollback_verification_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_rollback_verification_result_consumer,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_sequence_policy_gate_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_sequence_policy_gate_result_consumer,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_policy_consumer_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_policy_consumer_result_consumer,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_ordering_preview_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_ordering_preview_result_consumer,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_ordering_plan_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_ordering_plan_result_consumer,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_preflight_plan_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_preflight_plan_result_consumer,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_batch_preflight_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_batch_preflight_result_consumer,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_sequence_plan_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_sequence_plan_result_consumer,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_dry_run_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_dry_run_result_consumer,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_gate_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_gate_result_consumer,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_result_consumer,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_audit_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_audit_result_consumer,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_audit_policy_gate_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_audit_policy_gate_result_consumer,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_draft_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_draft_result_consumer,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_writer_dry_run_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_writer_dry_run_result_consumer,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_writer_apply_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_writer_apply_result_consumer,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_index_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_index_result_consumer,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_trust_review_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_trust_review_result_consumer,
    )
    from operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_trust_acceptance_ledger_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_trust_acceptance_ledger_result_consumer,
    )
    from source_identity_calibration_memory_trust_review import (
        build_source_identity_calibration_memory_trust_review,
    )
    from source_identity_calibration_memory_trust_acceptance_ledger import (
        build_source_identity_calibration_memory_trust_acceptance_ledger,
    )
    from source_identity_calibration_memory_trust_promotion_plan import (
        build_source_identity_calibration_memory_trust_promotion_plan,
    )
    from source_identity_calibration_memory_trust_promotion_contract import (
        build_source_identity_calibration_memory_trust_promotion_contract,
    )
    from source_identity_calibration_memory_trust_promotion_acceptance_ledger import (
        build_source_identity_calibration_memory_trust_promotion_acceptance_ledger,
    )
    from source_identity_calibration_memory_trust_promotion_execution_plan import (
        build_source_identity_calibration_memory_trust_promotion_execution_plan,
    )
    from source_identity_calibration_memory_trust_promotion_executor_dry_run import (
        build_source_identity_calibration_memory_trust_promotion_executor_dry_run,
    )
    from source_identity_calibration_memory_trust_promotion_executor import (
        build_source_identity_calibration_memory_trust_promotion_executor,
    )
    from source_identity_calibration_memory_trust_promotion_apply_audit import (
        build_source_identity_calibration_memory_trust_promotion_apply_audit,
    )
    from source_identity_calibration_memory_trust_promotion_rollback_verification import (
        build_source_identity_calibration_memory_trust_promotion_rollback_verification,
    )
    from source_identity_calibration_memory_trust_promotion_priority_gate import (
        build_source_identity_calibration_memory_trust_promotion_priority_gate,
    )
    from source_identity_proposal_priority_policy import (
        build_source_identity_proposal_priority_policy,
    )
    from source_identity_proposal_ordering_preview import (
        build_source_identity_proposal_ordering_preview,
    )
    from source_identity_proposal_queue_ordering_plan import (
        build_source_identity_proposal_queue_ordering_plan,
    )
    from source_identity_proposal_queue_importer import (
        build_source_identity_proposal_queue_importer,
    )
    from write_surface_inventory import build_write_surface_inventory_snapshot
except ImportError:  # pragma: no cover - package import compatibility
    from backend.bus_group_main_dca_physical_readback_contract import (
        build_bus_group_main_dca_physical_readback_contract,
    )
    from backend.bus_group_main_dca_physical_readback_artifact_intake import (
        build_bus_group_main_dca_physical_readback_artifact_intake,
    )
    from backend.channel_basic_controls_physical_readback_contract import (
        build_channel_basic_controls_physical_readback_contract,
    )
    from backend.channel_basic_controls_physical_readback_artifact_intake import (
        build_channel_basic_controls_physical_readback_artifact_intake,
    )
    from backend.channel_processing_physical_readback_contract import (
        build_channel_processing_physical_readback_contract,
    )
    from backend.channel_processing_physical_readback_artifact_intake import (
        build_channel_processing_physical_readback_artifact_intake,
    )
    from backend.dlive_state_physical_readback_contract import (
        build_dlive_state_physical_readback_contract,
    )
    from backend.dlive_state_physical_readback_artifact_intake import (
        build_dlive_state_physical_readback_artifact_intake,
    )
    from backend.fx_slots_inserts_physical_readback_contract import (
        build_fx_slots_inserts_physical_readback_contract,
    )
    from backend.fx_slots_inserts_physical_readback_artifact_intake import (
        build_fx_slots_inserts_physical_readback_artifact_intake,
    )
    from backend.matrix_state_physical_readback_contract import (
        build_matrix_state_physical_readback_contract,
    )
    from backend.matrix_state_physical_readback_artifact_intake import (
        build_matrix_state_physical_readback_artifact_intake,
    )
    from backend.autonomy_mode_transition_gate import build_autonomy_mode_transition_gate
    from backend.autonomy_promotion_readiness import build_autonomy_promotion_readiness_snapshot
    from backend.project_goal_capability_matrix import (
        build_project_goal_capability_matrix_snapshot,
        build_project_goal_capability_progress_snapshot,
        build_project_goal_objective_alignment_snapshot,
    )
    from backend.goal_capability_readiness_normalization import (
        normalize_goal_capability_readiness_hotspots,
        normalize_goal_capability_readiness_rows,
    )
    from backend.read_surface_fixture_evidence import build_read_surface_fixture_evidence_snapshot
    from backend.read_surface_inventory import build_read_surface_inventory_snapshot
    from backend.read_surface_physical_readback_capture_plan import (
        build_read_surface_physical_readback_capture_plan,
    )
    from backend.read_surface_remediation_plan import build_read_surface_remediation_plan
    from backend.raw_mixer_state_physical_readback_contract import (
        build_raw_mixer_state_physical_readback_contract,
    )
    from backend.raw_mixer_state_physical_readback_artifact_intake import (
        build_raw_mixer_state_physical_readback_artifact_intake,
    )
    from backend.read_surface_physical_readback_artifact_review_queue import (
        build_read_surface_physical_readback_artifact_review_queue,
    )
    from backend.read_surface_physical_readback_fixture_acceptance_ledger import (
        build_read_surface_physical_readback_fixture_acceptance_ledger,
    )
    from backend.read_surface_physical_readback_fixture_closure_plan import (
        build_read_surface_physical_readback_fixture_closure_plan,
    )
    from backend.routing_sends_physical_readback_contract import (
        build_routing_sends_physical_readback_contract,
    )
    from backend.routing_sends_physical_readback_artifact_intake import (
        build_routing_sends_physical_readback_artifact_intake,
    )
    from backend.snapshots_scenes_physical_readback_contract import (
        build_snapshots_scenes_physical_readback_contract,
    )
    from backend.snapshots_scenes_physical_readback_artifact_intake import (
        build_snapshots_scenes_physical_readback_artifact_intake,
    )
    from backend.user_presets_physical_readback_contract import (
        build_user_presets_physical_readback_contract,
    )
    from backend.user_presets_physical_readback_artifact_intake import (
        build_user_presets_physical_readback_artifact_intake,
    )
    from backend.wing_readonly_preflight_evidence import build_wing_readonly_preflight_evidence
    from backend.safety_coverage_matrix import build_safety_coverage_matrix_snapshot
    from backend.source_identity_calibration_acceptance_ledger import (
        build_source_identity_calibration_acceptance_ledger,
    )
    from backend.source_identity_calibration_persistence_plan import (
        build_source_identity_calibration_persistence_plan,
    )
    from backend.source_identity_calibration_memory_draft import (
        build_source_identity_calibration_memory_draft,
    )
    from backend.source_identity_calibration_memory_writer import (
        build_source_identity_calibration_memory_writer,
    )
    from backend.source_identity_calibration_memory_index import (
        build_source_identity_calibration_memory_index,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_index import (
        build_operator_proposal_apply_sequence_policy_memory_index,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_review import (
        build_operator_proposal_apply_sequence_policy_memory_trust_review,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger import (
        build_operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_plan import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_plan,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_contract import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_contract,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_executor_dry_run import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_executor_dry_run,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_executor import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_executor,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_apply_audit import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_apply_audit,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_rollback_verification import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_rollback_verification,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_sequence_policy_gate import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_sequence_policy_gate,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_policy_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_policy_consumer,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering_preview import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering_preview,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_ordering_plan import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_ordering_plan,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_importer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_importer,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_plan import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_plan,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_result_consumer,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_sequence_plan_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_sequence_plan_result_consumer,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_dry_run_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_dry_run_result_consumer,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_gate_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_gate_result_consumer,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_result_consumer,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_audit_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_audit_result_consumer,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_audit_policy_gate_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_audit_policy_gate_result_consumer,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_draft_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_draft_result_consumer,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_dry_run_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_dry_run_result_consumer,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_apply_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_apply_result_consumer,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_index_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_index_result_consumer,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_review_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_review_result_consumer,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_acceptance_ledger_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_acceptance_ledger_result_consumer,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_plan_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_plan_result_consumer,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_contract_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_contract_result_consumer,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_acceptance_ledger_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_acceptance_ledger_result_consumer,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_plan_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_plan_result_consumer,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_dry_run_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_dry_run_result_consumer,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_result_consumer,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_apply_audit_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_apply_audit_result_consumer,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_rollback_verification_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_rollback_verification_result_consumer,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_sequence_policy_gate_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_sequence_policy_gate_result_consumer,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_policy_consumer_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_policy_consumer_result_consumer,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_ordering_preview_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_ordering_preview_result_consumer,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_ordering_plan_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_ordering_plan_result_consumer,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_preflight_plan_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_preflight_plan_result_consumer,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_batch_preflight_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_batch_preflight_result_consumer,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_sequence_plan_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_sequence_plan_result_consumer,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_dry_run_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_dry_run_result_consumer,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_gate_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_gate_result_consumer,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_result_consumer,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_audit_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_audit_result_consumer,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_audit_policy_gate_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_audit_policy_gate_result_consumer,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_draft_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_draft_result_consumer,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_writer_dry_run_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_writer_dry_run_result_consumer,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_writer_apply_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_writer_apply_result_consumer,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_index_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_index_result_consumer,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_trust_review_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_trust_review_result_consumer,
    )
    from backend.operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_trust_acceptance_ledger_result_consumer import (
        build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_trust_acceptance_ledger_result_consumer,
    )
    from backend.source_identity_calibration_memory_trust_review import (
        build_source_identity_calibration_memory_trust_review,
    )
    from backend.source_identity_calibration_memory_trust_acceptance_ledger import (
        build_source_identity_calibration_memory_trust_acceptance_ledger,
    )
    from backend.source_identity_calibration_memory_trust_promotion_plan import (
        build_source_identity_calibration_memory_trust_promotion_plan,
    )
    from backend.source_identity_calibration_memory_trust_promotion_contract import (
        build_source_identity_calibration_memory_trust_promotion_contract,
    )
    from backend.source_identity_calibration_memory_trust_promotion_acceptance_ledger import (
        build_source_identity_calibration_memory_trust_promotion_acceptance_ledger,
    )
    from backend.source_identity_calibration_memory_trust_promotion_execution_plan import (
        build_source_identity_calibration_memory_trust_promotion_execution_plan,
    )
    from backend.source_identity_calibration_memory_trust_promotion_executor_dry_run import (
        build_source_identity_calibration_memory_trust_promotion_executor_dry_run,
    )
    from backend.source_identity_calibration_memory_trust_promotion_executor import (
        build_source_identity_calibration_memory_trust_promotion_executor,
    )
    from backend.source_identity_calibration_memory_trust_promotion_apply_audit import (
        build_source_identity_calibration_memory_trust_promotion_apply_audit,
    )
    from backend.source_identity_calibration_memory_trust_promotion_rollback_verification import (
        build_source_identity_calibration_memory_trust_promotion_rollback_verification,
    )
    from backend.source_identity_calibration_memory_trust_promotion_priority_gate import (
        build_source_identity_calibration_memory_trust_promotion_priority_gate,
    )
    from backend.source_identity_proposal_priority_policy import (
        build_source_identity_proposal_priority_policy,
    )
    from backend.source_identity_proposal_ordering_preview import (
        build_source_identity_proposal_ordering_preview,
    )
    from backend.source_identity_proposal_queue_ordering_plan import (
        build_source_identity_proposal_queue_ordering_plan,
    )
    from backend.source_identity_proposal_queue_importer import (
        build_source_identity_proposal_queue_importer,
    )
    from backend.write_surface_inventory import build_write_surface_inventory_snapshot


PROCESSING_STATUSES = {
    "canonical",
    "proposal_only",
    "offline_only",
    "shadow_only",
    "lab",
    "placeholder",
    "quarantined",
    "deprecated",
}

PROCESSING_DOMAINS = {
    "gain_fader",
    "source_identity",
    "eq",
    "dynamics",
    "gate",
    "pan",
    "phase",
    "fx",
    "master_bus",
    "bus_group",
    "mute",
    "routing",
    "scene_preset",
    "mix_variant",
    "adaptation",
    "arrangement",
    "decision_evidence",
    "memory_learning",
    "evaluation",
    "offline_render",
}

LIVE_ALLOWED_WRITE_KINDS = tuple(CURRENT_PROVEN_LIVE_WRITE_KINDS)


@dataclass(frozen=True)
class ProcessingMethod:
    id: str
    label: str
    domain: str
    status: str
    summary: str
    module: str
    can_analyze: bool = False
    can_propose: bool = False
    can_render_offline: bool = False
    can_write_live: bool = False
    requires_supervision: bool = True
    live_write_kinds: Tuple[str, ...] = ()
    pipeline_profiles: Tuple[str, ...] = ()
    blocked_reason: Optional[str] = None
    promotion_next: Optional[str] = None
    notes: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in PROCESSING_STATUSES:
            raise ValueError(f"Unknown processing status: {self.status}")
        if self.domain not in PROCESSING_DOMAINS:
            raise ValueError(f"Unknown processing domain: {self.domain}")
        if self.can_write_live and not self.requires_supervision:
            raise ValueError(f"{self.id} can write live without supervision")
        if self.can_write_live and any(kind not in LIVE_ALLOWED_WRITE_KINDS for kind in self.live_write_kinds):
            raise ValueError(f"{self.id} declares unsupported live write kinds: {self.live_write_kinds}")
        if self.status in {"placeholder", "quarantined", "deprecated"} and self.can_write_live:
            raise ValueError(f"{self.id} is {self.status} but can_write_live=True")

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["live_write_kinds"] = list(self.live_write_kinds)
        payload["pipeline_profiles"] = list(self.pipeline_profiles)
        payload["notes"] = list(self.notes)
        return payload


@dataclass(frozen=True)
class PipelineProfile:
    id: str
    label: str
    status: str
    summary: str
    method_ids: Tuple[str, ...]
    proposal_method_ids: Tuple[str, ...] = ()
    executor_method_ids: Tuple[str, ...] = ()
    blocked_method_ids: Tuple[str, ...] = ()
    can_write_live: bool = False
    requires_supervision: bool = True
    live_write_kinds: Tuple[str, ...] = ()
    mode: str = "read_only"
    notes: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in PROCESSING_STATUSES:
            raise ValueError(f"Unknown pipeline status: {self.status}")
        if self.can_write_live and not self.requires_supervision:
            raise ValueError(f"{self.id} can write live without supervision")
        if self.can_write_live and any(kind not in LIVE_ALLOWED_WRITE_KINDS for kind in self.live_write_kinds):
            raise ValueError(f"{self.id} declares unsupported live write kinds: {self.live_write_kinds}")

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["method_ids"] = list(self.method_ids)
        payload["proposal_method_ids"] = list(self.proposal_method_ids)
        payload["executor_method_ids"] = list(self.executor_method_ids)
        payload["blocked_method_ids"] = list(self.blocked_method_ids)
        payload["live_write_kinds"] = list(self.live_write_kinds)
        payload["notes"] = list(self.notes)
        return payload


PROCESSING_METHODS: Tuple[ProcessingMethod, ...] = (
    ProcessingMethod(
        id="product_operator_state",
        label="Product / Operator State",
        domain="evaluation",
        status="canonical",
        summary="Read-only state, readiness blockers, proposal queue and operator mode policy.",
        module="backend/operator_product_state.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("main_live_assist",),
        promotion_next="keep_as_canonical_read_model",
    ),
    ProcessingMethod(
        id="auto_soundcheck_engine",
        label="AutoSoundcheck Engine",
        domain="gain_fader",
        status="canonical",
        summary="Primary live analysis/orchestration hook; live writes must be translated to proposals or supervised writes.",
        module="backend/auto_soundcheck_engine.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("main_live_assist", "lab_dsp"),
        notes=("guarded_use_only", "do_not_bypass_product_layer"),
    ),
    ProcessingMethod(
        id="source_identity_confidence_proposal",
        label="Source Identity Confidence Proposal Layer",
        domain="source_identity",
        status="proposal_only",
        summary=(
            "Transport-free source identity confidence review layer that turns "
            "low-confidence, unknown, or ambiguous classifier results into "
            "operator-facing review records and downstream rejection reasons."
        ),
        module="backend/source_identity_confidence_proposal.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason=(
            "source_identity_review_needs_operator_acceptance_and_replay_before_"
            "downstream_processing_can_trust_low_confidence_classifications"
        ),
        promotion_next=(
            "route low-confidence or ambiguous source identities to operator "
            "review, then replay accepted/rejected labels before changing "
            "downstream proposal priority"
        ),
        notes=(
            "transport_free",
            "proposal_only",
            "classifier_confidence_gate",
            "downstream_rejection_reasons",
            "no_live_writes",
        ),
    ),
    ProcessingMethod(
        id="source_identity_replay_calibration",
        label="Source Identity Replay Calibration Layer",
        domain="source_identity",
        status="proposal_only",
        summary=(
            "Transport-free replay calibration layer that summarizes accepted, "
            "corrected, and rejected source identity outcomes before downstream "
            "proposal families trust classifier labels."
        ),
        module="backend/source_identity_replay_calibration.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason=(
            "source_identity_calibration_needs_operator_review_before_thresholds_"
            "or_downstream_priority_can_change"
        ),
        promotion_next=(
            "review replay-calibrated source identity trust states, then wire "
            "accepted calibration records into downstream proposal priority without "
            "creating live writes"
        ),
        notes=(
            "transport_free",
            "proposal_only",
            "accepted_rejected_replay_outcomes",
            "no_model_mutation",
            "no_memory_mutation",
            "no_live_writes",
        ),
    ),
    ProcessingMethod(
        id="source_identity_calibration_acceptance_ledger",
        label="Source Identity Calibration Acceptance Ledger",
        domain="source_identity",
        status="shadow_only",
        summary=(
            "Read-only ledger for operator accept/reject records tied to "
            "source identity replay-calibration candidates."
        ),
        module="backend/source_identity_calibration_acceptance_ledger.py",
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason="acceptance_ledger_only_calibration_persistence_still_separate",
        promotion_next=(
            "operator records acceptance or rejection of replay-calibrated trust "
            "states; model, memory, and downstream-priority writes remain separate"
        ),
        notes=(
            "transport_free",
            "operator_acceptance_ledger_only",
            "no_queue_mutation",
            "no_model_mutation",
            "no_memory_mutation",
            "no_downstream_priority_mutation",
            "no_live_writes",
        ),
    ),
    ProcessingMethod(
        id="source_identity_calibration_persistence_plan",
        label="Source Identity Calibration Persistence Plan",
        domain="source_identity",
        status="shadow_only",
        summary=(
            "Read-only plan that turns accepted source identity calibration "
            "ledger records into manual persistence candidates."
        ),
        module="backend/source_identity_calibration_persistence_plan.py",
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason="persistence_plan_only_memory_write_still_separate",
        promotion_next=(
            "review accepted calibration records and define a separate, audited "
            "persistence/write contract before memory or downstream priority changes"
        ),
        notes=(
            "transport_free",
            "manual_persistence_plan_only",
            "no_queue_mutation",
            "no_model_mutation",
            "no_memory_mutation",
            "no_downstream_priority_mutation",
            "no_live_writes",
        ),
    ),
    ProcessingMethod(
        id="source_identity_calibration_memory_draft",
        label="Source Identity Calibration Memory Draft",
        domain="memory_learning",
        status="shadow_only",
        summary=(
            "Read-only append-only memory draft records derived from manual "
            "source identity calibration persistence candidates."
        ),
        module="backend/source_identity_calibration_memory_draft.py",
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason="memory_draft_only_append_write_contract_still_separate",
        promotion_next=(
            "review memory draft records and route them through a separate "
            "append-only memory write contract before updating project memory"
        ),
        notes=(
            "transport_free",
            "append_only_memory_draft_only",
            "no_queue_mutation",
            "no_model_mutation",
            "no_memory_mutation",
            "no_downstream_priority_mutation",
            "no_live_writes",
        ),
    ),
    ProcessingMethod(
        id="source_identity_calibration_memory_writer",
        label="Source Identity Calibration Memory Writer",
        domain="memory_learning",
        status="shadow_only",
        summary=(
            "Append-only memory writer contract for source identity calibration "
            "drafts; preview runs dry-run and apply requires operator ack."
        ),
        module="backend/source_identity_calibration_memory_writer.py",
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason="memory_writer_requires_explicit_operator_ack_and_apply",
        promotion_next=(
            "use only with audited draft records and explicit operator ack; never "
            "change models, downstream priority, queues, or live console state"
        ),
        notes=(
            "transport_free",
            "append_only_memory_writer_contract",
            "dry_run_in_preview",
            "operator_ack_required",
            "no_queue_mutation",
            "no_model_mutation",
            "no_downstream_priority_mutation",
            "no_live_writes",
        ),
    ),
    ProcessingMethod(
        id="source_identity_calibration_memory_index",
        label="Source Identity Calibration Memory Index",
        domain="memory_learning",
        status="shadow_only",
        summary=(
            "Read-only index of append-only source identity calibration memory "
            "blocks; exposes durable memory evidence without promotion authority."
        ),
        module="backend/source_identity_calibration_memory_index.py",
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason="memory_index_is_review_evidence_not_downstream_priority_authority",
        promotion_next=(
            "use indexed records only as review evidence; separate replay/trust "
            "promotion must approve any downstream priority influence"
        ),
        notes=(
            "transport_free",
            "append_only_memory_index",
            "read_only_memory_evidence",
            "no_queue_mutation",
            "no_model_mutation",
            "no_runtime_memory_mutation",
            "no_downstream_priority_mutation",
            "no_live_writes",
        ),
    ),
    ProcessingMethod(
        id="operator_proposal_apply_sequence_policy_memory_index",
        label="Operator Proposal Apply Sequence Policy Memory Index",
        domain="memory_learning",
        status="shadow_only",
        summary=(
            "Read-only index of append-only supervised operator proposal apply "
            "sequence memory blocks; exposes durable sequence evidence without "
            "execution or policy authority."
        ),
        module="backend/operator_proposal_apply_sequence_policy_memory_index.py",
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason="sequence_memory_index_is_review_evidence_not_execution_authority",
        promotion_next=(
            "use indexed sequence memory only as review evidence; separate "
            "policy/trust promotion must approve any future execution influence"
        ),
        notes=(
            "transport_free",
            "append_only_sequence_memory_index",
            "read_only_memory_evidence",
            "no_queue_mutation",
            "no_policy_mutation",
            "no_model_mutation",
            "no_runtime_memory_mutation",
            "no_live_writes",
        ),
    ),
    ProcessingMethod(
        id="operator_proposal_apply_sequence_policy_memory_trust_review",
        label="Operator Proposal Apply Sequence Policy Memory Trust Review",
        domain="memory_learning",
        status="shadow_only",
        summary=(
            "Read-only operator-review candidates derived from indexed supervised "
            "operator proposal apply-sequence memory and fresh review context."
        ),
        module="backend/operator_proposal_apply_sequence_policy_memory_trust_review.py",
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason="sequence_memory_trust_review_requires_operator_policy_review",
        promotion_next=(
            "operator reviews indexed sequence memory with fresh context before "
            "any separate policy/trust promotion can influence execution"
        ),
        notes=(
            "transport_free",
            "operator_review_only",
            "sequence_memory_index_consumer",
            "fresh_sequence_review_context_required",
            "no_execution_authority",
            "no_queue_mutation",
            "no_policy_mutation",
            "no_model_mutation",
            "no_runtime_memory_mutation",
            "no_live_writes",
        ),
    ),
    ProcessingMethod(
        id="operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger",
        label="Operator Proposal Apply Sequence Policy Memory Trust Acceptance Ledger",
        domain="memory_learning",
        status="shadow_only",
        summary=(
            "Read-only ledger for operator accepted/rejected decisions on "
            "supervised sequence memory trust-review candidates."
        ),
        module="backend/operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger.py",
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason="sequence_memory_trust_acceptance_is_not_promotion_authority",
        promotion_next=(
            "use accepted sequence-memory trust records only as evidence for a "
            "separate promotion contract before policy or execution influence"
        ),
        notes=(
            "transport_free",
            "operator_acceptance_ledger_only",
            "sequence_memory_trust_review_consumer",
            "no_execution_authority",
            "no_queue_mutation",
            "no_policy_mutation",
            "no_model_mutation",
            "no_runtime_memory_mutation",
            "no_live_writes",
        ),
    ),
    ProcessingMethod(
        id="operator_proposal_apply_sequence_policy_memory_trust_promotion_plan",
        label="Operator Proposal Apply Sequence Policy Memory Trust Promotion Plan",
        domain="memory_learning",
        status="shadow_only",
        summary=(
            "Read-only manual promotion plan derived from accepted supervised "
            "sequence-memory trust ledger records; does not change policy or execution."
        ),
        module="backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_plan.py",
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason="sequence_memory_trust_promotion_plan_is_manual_not_authority",
        promotion_next=(
            "define a separate promotion contract with fresh policy context and "
            "rollback before accepted sequence memory can influence execution"
        ),
        notes=(
            "transport_free",
            "manual_promotion_plan_only",
            "sequence_memory_trust_acceptance_consumer",
            "no_execution_authority",
            "no_queue_mutation",
            "no_policy_mutation",
            "no_model_mutation",
            "no_runtime_memory_mutation",
            "no_live_writes",
        ),
    ),
    ProcessingMethod(
        id="operator_proposal_apply_sequence_policy_memory_trust_promotion_contract",
        label="Operator Proposal Apply Sequence Policy Memory Trust Promotion Contract",
        domain="memory_learning",
        status="shadow_only",
        summary=(
            "Read-only contract for accepted sequence-memory trust promotion "
            "candidates; requires fresh policy context and rollback evidence."
        ),
        module="backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_contract.py",
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason="sequence_memory_trust_promotion_contract_is_review_not_authority",
        promotion_next=(
            "operator reviews contract records before any separate executor can "
            "mutate sequence policy or influence supervised execution"
        ),
        notes=(
            "transport_free",
            "manual_promotion_contract_only",
            "fresh_sequence_context_required",
            "rollback_evidence_required",
            "sequence_memory_trust_promotion_plan_consumer",
            "no_execution_authority",
            "no_queue_mutation",
            "no_policy_mutation",
            "no_model_mutation",
            "no_runtime_memory_mutation",
            "no_live_writes",
        ),
    ),
    ProcessingMethod(
        id="operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger",
        label="Operator Proposal Apply Sequence Policy Memory Trust Promotion Acceptance Ledger",
        domain="memory_learning",
        status="shadow_only",
        summary=(
            "Read-only ledger for operator accepted/rejected decisions on "
            "sequence-memory trust promotion contract records."
        ),
        module=(
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger.py"
        ),
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason=(
            "sequence_memory_trust_promotion_acceptance_is_not_policy_or_execution_authority"
        ),
        promotion_next=(
            "use accepted promotion contract records only as evidence for a "
            "separate executor plan before policy or execution influence"
        ),
        notes=(
            "transport_free",
            "operator_acceptance_ledger_only",
            "sequence_memory_trust_promotion_contract_consumer",
            "no_execution_authority",
            "no_queue_mutation",
            "no_policy_mutation",
            "no_model_mutation",
            "no_runtime_memory_mutation",
            "no_live_writes",
        ),
    ),
    ProcessingMethod(
        id="operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan",
        label="Operator Proposal Apply Sequence Policy Memory Trust Promotion Execution Plan",
        domain="memory_learning",
        status="shadow_only",
        summary=(
            "Read-only execution plan that surfaces accepted sequence-memory "
            "promotion records as non-executing candidates."
        ),
        module=(
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan.py"
        ),
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason=(
            "sequence_memory_trust_promotion_execution_plan_is_not_executor_authority"
        ),
        promotion_next=(
            "build a separate acknowledged executor before sequence policy, "
            "queue, runtime memory, or live execution can be influenced"
        ),
        notes=(
            "transport_free",
            "execution_plan_only",
            "sequence_memory_trust_promotion_acceptance_consumer",
            "no_execution_authority",
            "no_queue_mutation",
            "no_policy_mutation",
            "no_model_mutation",
            "no_runtime_memory_mutation",
            "no_live_writes",
        ),
    ),
    ProcessingMethod(
        id="operator_proposal_apply_sequence_policy_memory_trust_promotion_executor_dry_run",
        label="Operator Proposal Apply Sequence Policy Memory Trust Promotion Executor Dry Run",
        domain="memory_learning",
        status="shadow_only",
        summary=(
            "Read-only dry-run executor contract that converts sequence-memory "
            "promotion execution candidates into deterministic policy apply specs."
        ),
        module=(
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_executor_dry_run.py"
        ),
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason=(
            "sequence_memory_trust_promotion_executor_dry_run_does_not_apply"
        ),
        promotion_next=(
            "validate dry-run policy specs, then add a separate operator-gated "
            "executor contract before sequence policy can be promoted"
        ),
        notes=(
            "transport_free",
            "executor_dry_run_only",
            "sequence_memory_trust_promotion_execution_plan_consumer",
            "idempotent_policy_apply_spec",
            "no_execution_authority",
            "no_queue_mutation",
            "no_policy_mutation",
            "no_model_mutation",
            "no_runtime_memory_mutation",
            "no_live_writes",
        ),
    ),
    ProcessingMethod(
        id="operator_proposal_apply_sequence_policy_memory_trust_promotion_executor",
        label="Operator Proposal Apply Sequence Policy Memory Trust Promotion Executor",
        domain="memory_learning",
        status="shadow_only",
        summary=(
            "Operator-gated executor that can promote sequence-memory trust "
            "only into an explicit caller-provided sequence policy store."
        ),
        module=(
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_executor.py"
        ),
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason=(
            "sequence_memory_trust_promotion_executor_requires_operator_ack_and_explicit_store"
        ),
        promotion_next=(
            "audit applied policy-store results and verify rollback before any "
            "sequence policy consumer can influence ordering or execution"
        ),
        notes=(
            "transport_free",
            "operator_gated_policy_store_executor",
            "sequence_memory_trust_promotion_executor_dry_run_consumer",
            "explicit_target_store_required",
            "no_live_writes",
            "no_queue_mutation",
            "no_model_mutation",
            "no_proposal_state_mutation",
            "no_direct_sequence_execution",
        ),
    ),
    ProcessingMethod(
        id="operator_proposal_apply_sequence_policy_memory_trust_promotion_apply_audit",
        label="Operator Proposal Apply Sequence Policy Memory Trust Promotion Apply Audit",
        domain="memory_learning",
        status="shadow_only",
        summary=(
            "Read-only rollback audit for applied sequence-policy trust promotion "
            "executor results."
        ),
        module=(
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_apply_audit.py"
        ),
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason=(
            "sequence_memory_trust_promotion_apply_audit_is_not_sequence_or_live_authority"
        ),
        promotion_next=(
            "verify rollback evidence before any sequence policy consumer can "
            "use promoted memory trust records"
        ),
        notes=(
            "transport_free",
            "apply_audit_only",
            "sequence_memory_trust_promotion_executor_consumer",
            "rollback_evidence_required",
            "no_live_writes",
            "no_queue_mutation",
            "no_model_mutation",
            "no_proposal_state_mutation",
            "no_direct_sequence_execution",
        ),
    ),
    ProcessingMethod(
        id="operator_proposal_apply_sequence_policy_memory_trust_promotion_rollback_verification",
        label=(
            "Operator Proposal Apply Sequence Policy Memory Trust Promotion "
            "Rollback Verification"
        ),
        domain="memory_learning",
        status="shadow_only",
        summary=(
            "Read-only operator rollback verification gate for sequence-policy "
            "memory trust promotion apply-audit records."
        ),
        module=(
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_rollback_verification.py"
        ),
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason=(
            "sequence_memory_trust_promotion_rollback_verification_is_not_sequence_or_live_authority"
        ),
        promotion_next=(
            "use verified rollback records only as evidence for a separate "
            "sequence-policy gate"
        ),
        notes=(
            "transport_free",
            "rollback_verification_only",
            "sequence_memory_trust_promotion_apply_audit_consumer",
            "operator_review_required",
            "no_live_writes",
            "no_queue_mutation",
            "no_model_mutation",
            "no_runtime_memory_mutation",
            "no_policy_mutation",
            "no_proposal_state_mutation",
            "no_direct_sequence_execution",
        ),
    ),
    ProcessingMethod(
        id="operator_proposal_apply_sequence_policy_memory_trust_promotion_sequence_policy_gate",
        label=(
            "Operator Proposal Apply Sequence Policy Memory Trust Promotion "
            "Sequence Policy Gate"
        ),
        domain="memory_learning",
        status="shadow_only",
        summary=(
            "Read-only sequence-policy eligibility gate for operator-verified "
            "sequence memory trust promotion rollback records."
        ),
        module=(
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_sequence_policy_gate.py"
        ),
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason=(
            "sequence_memory_trust_promotion_sequence_policy_gate_is_not_policy_or_live_mutation_authority"
        ),
        promotion_next=(
            "route eligible records through a separate sequence-policy consumer "
            "before any proposal ordering, queue import, or execution behavior changes"
        ),
        notes=(
            "transport_free",
            "sequence_policy_gate_only",
            "sequence_memory_trust_promotion_rollback_verification_consumer",
            "operator_review_required",
            "no_live_writes",
            "no_queue_mutation",
            "no_model_mutation",
            "no_runtime_memory_mutation",
            "no_policy_mutation",
            "no_proposal_state_mutation",
            "no_direct_sequence_execution",
        ),
    ),
    ProcessingMethod(
        id="operator_proposal_apply_sequence_policy_memory_trust_promotion_policy_consumer",
        label=(
            "Operator Proposal Apply Sequence Policy Memory Trust Promotion "
            "Policy Consumer"
        ),
        domain="decision_evidence",
        status="shadow_only",
        summary=(
            "Read-only consumer that projects trusted sequence-memory evidence "
            "onto proposal previews without mutating order, queues, policy "
            "stores, models, proposal state, or live transport."
        ),
        module=(
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_policy_consumer.py"
        ),
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason=(
            "sequence_memory_trust_promotion_policy_consumer_is_not_ordering_queue_or_live_authority"
        ),
        promotion_next=(
            "route consumer records through a separate ordering or sequence-policy "
            "consumer before any proposal order, queue import, or execution changes"
        ),
        notes=(
            "transport_free",
            "policy_evidence_only",
            "sequence_policy_gate_consumer",
            "no_proposal_order_mutation",
            "no_live_writes",
            "no_queue_mutation",
            "no_model_mutation",
            "no_runtime_memory_mutation",
            "no_policy_mutation",
            "no_proposal_state_mutation",
            "no_direct_sequence_execution",
        ),
    ),
    ProcessingMethod(
        id="operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering_preview",
        label=(
            "Operator Proposal Apply Sequence Policy Memory Trust Promotion "
            "Ordering Preview"
        ),
        domain="decision_evidence",
        status="shadow_only",
        summary=(
            "Read-only recommended proposal-order view from sequence-policy "
            "consumer records; does not mutate queues, proposal payloads, "
            "policy stores, models, proposal state, or live transport."
        ),
        module=(
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering_preview.py"
        ),
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason=(
            "sequence_memory_trust_promotion_ordering_preview_is_not_queue_execution_or_live_authority"
        ),
        promotion_next=(
            "route ordering-preview rows through a separate queue or sequence "
            "consumer before mutating queues, proposal order, or execution behavior"
        ),
        notes=(
            "transport_free",
            "ordering_preview_only",
            "sequence_policy_consumer_records",
            "proposal_order_mutation_false",
            "no_live_writes",
            "no_queue_mutation",
            "no_model_mutation",
            "no_runtime_memory_mutation",
            "no_policy_mutation",
            "no_proposal_state_mutation",
            "no_direct_sequence_execution",
        ),
    ),
    ProcessingMethod(
        id="operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_ordering_plan",
        label=(
            "Operator Proposal Apply Sequence Policy Memory Trust Promotion "
            "Queue Ordering Plan"
        ),
        domain="decision_evidence",
        status="shadow_only",
        summary=(
            "Read-only queue-ordering plan from sequence-policy ordering "
            "preview rows; does not import or reorder queues, mutate proposal "
            "payloads, policy stores, models, proposal state, or live transport."
        ),
        module=(
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_ordering_plan.py"
        ),
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason=(
            "sequence_memory_trust_promotion_queue_ordering_plan_is_not_queue_execution_or_live_authority"
        ),
        promotion_next=(
            "route queue-ordering plan records through a separate acknowledged "
            "queue import or sequence consumer before mutating queues or execution"
        ),
        notes=(
            "transport_free",
            "queue_ordering_plan_only",
            "sequence_policy_ordering_preview_consumer",
            "queue_import_allowed_by_plan_false",
            "proposal_order_mutation_false",
            "no_live_writes",
            "no_queue_mutation",
            "no_model_mutation",
            "no_runtime_memory_mutation",
            "no_policy_mutation",
            "no_proposal_state_mutation",
            "no_direct_sequence_execution",
        ),
    ),
    ProcessingMethod(
        id="operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_importer",
        label=(
            "Operator Proposal Apply Sequence Policy Memory Trust Promotion "
            "Queue Importer"
        ),
        domain="decision_evidence",
        status="shadow_only",
        summary=(
            "Operator-gated queue importer for sequence-policy queue-ordering "
            "plans; Product Layer uses dry-run only."
        ),
        module=(
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_importer.py"
        ),
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason=(
            "sequence_memory_trust_promotion_queue_importer_requires_operator_ack_and_explicit_queue"
        ),
        promotion_next=(
            "wire an operator-reviewed UI action that passes apply=true, "
            "operator_ack=true, and the explicit target queue before queue mutation; "
            "direct sequence execution still requires a separate sequence consumer"
        ),
        notes=(
            "transport_free",
            "product_layer_dry_run_only",
            "operator_ack_required",
            "explicit_target_queue_required",
            "sequence_policy_queue_ordering_plan_consumer",
            "sequence_consumer_required_before_execution",
            "no_model_mutation",
            "no_runtime_memory_mutation",
            "no_sequence_policy_trust_store_mutation",
            "no_proposal_state_mutation",
            "no_direct_sequence_execution",
            "no_live_writes",
        ),
    ),
    ProcessingMethod(
        id="operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_plan",
        label=(
            "Operator Proposal Apply Sequence Policy Memory Trust Promotion "
            "Queue Import Preflight Plan"
        ),
        domain="decision_evidence",
        status="shadow_only",
        summary=(
            "Read-only batch-preflight request plan for sequence-policy imported "
            "operator-queue proposals."
        ),
        module=(
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_plan.py"
        ),
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason=(
            "sequence_memory_trust_promotion_queue_import_preflight_plan_is_not_preflight_or_live_authority"
        ),
        promotion_next=(
            "run the existing batch preflight handler with explicit operator "
            "approval before any sequence plan, dry-run, gate, or executor"
        ),
        notes=(
            "transport_free",
            "preflight_request_plan_only",
            "sequence_policy_queue_importer_consumer",
            "batch_preflight_invoked_false",
            "idempotency_reserved_false",
            "sequence_plan_created_false",
            "sequence_consumer_required_before_execution",
            "no_queue_mutation",
            "no_model_mutation",
            "no_runtime_memory_mutation",
            "no_sequence_policy_trust_store_mutation",
            "no_proposal_state_mutation",
            "no_direct_sequence_execution",
            "no_live_writes",
        ),
    ),
    ProcessingMethod(
        id="operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_result_consumer",
        label=(
            "Operator Proposal Apply Sequence Policy Memory Trust Promotion "
            "Queue Import Preflight Result Consumer"
        ),
        domain="decision_evidence",
        status="shadow_only",
        summary=(
            "Read-only verifier that matches approved batch-preflight results "
            "back to the sequence-policy queue-import preflight plan before "
            "generic sequence planning."
        ),
        module=(
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_result_consumer.py"
        ),
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason=(
            "sequence_memory_trust_promotion_queue_import_preflight_result_consumer_is_not_sequence_plan_or_live_authority"
        ),
        promotion_next=(
            "run the emitted generic sequence-plan request only after the "
            "consumer verifies order, safety flags, and idempotency evidence; "
            "dry-run, gate, and executor remain separate steps"
        ),
        notes=(
            "transport_free",
            "preflight_result_consumer_only",
            "sequence_plan_request_template_only",
            "sequence_plan_handler_invoked_false",
            "sequence_plan_created_false",
            "idempotency_reserved_false",
            "sequence_consumer_required_before_execution",
            "no_queue_mutation",
            "no_model_mutation",
            "no_runtime_memory_mutation",
            "no_sequence_policy_trust_store_mutation",
            "no_proposal_state_mutation",
            "no_direct_sequence_execution",
            "no_live_writes",
        ),
    ),
    ProcessingMethod(
        id="operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_sequence_plan_result_consumer",
        label=(
            "Operator Proposal Apply Sequence Policy Memory Trust Promotion "
            "Queue Import Sequence Plan Result Consumer"
        ),
        domain="decision_evidence",
        status="shadow_only",
        summary=(
            "Read-only verifier that matches generic apply-sequence plans back "
            "to verified sequence-policy queue-import preflight evidence before "
            "dry-run."
        ),
        module=(
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_sequence_plan_result_consumer.py"
        ),
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason=(
            "sequence_memory_trust_promotion_queue_import_sequence_plan_result_consumer_is_not_dry_run_or_live_authority"
        ),
        promotion_next=(
            "run the emitted generic sequence dry-run request only after this "
            "consumer verifies sequence-plan order, safety flags, and "
            "idempotency evidence; gate and executor remain separate steps"
        ),
        notes=(
            "transport_free",
            "sequence_plan_result_consumer_only",
            "dry_run_request_template_only",
            "dry_run_handler_invoked_false",
            "dry_run_created_false",
            "idempotency_reserved_false",
            "sequence_consumer_required_before_execution",
            "no_queue_mutation",
            "no_model_mutation",
            "no_runtime_memory_mutation",
            "no_sequence_policy_trust_store_mutation",
            "no_proposal_state_mutation",
            "no_direct_sequence_execution",
            "no_live_writes",
        ),
    ),
    ProcessingMethod(
        id="operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_dry_run_result_consumer",
        label=(
            "Operator Proposal Apply Sequence Policy Memory Trust Promotion "
            "Queue Import Dry Run Result Consumer"
        ),
        domain="decision_evidence",
        status="shadow_only",
        summary=(
            "Read-only verifier that matches generic apply-sequence dry-run "
            "results back to verified sequence-policy queue-import provenance "
            "before execution gating."
        ),
        module=(
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_dry_run_result_consumer.py"
        ),
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason=(
            "sequence_memory_trust_promotion_queue_import_dry_run_result_consumer_is_not_gate_or_live_authority"
        ),
        promotion_next=(
            "run the emitted generic execution-gate request only after explicit "
            "operator acknowledgement; executor and live apply remain separate"
        ),
        notes=(
            "transport_free",
            "dry_run_result_consumer_only",
            "execution_gate_request_template_only",
            "execution_gate_handler_invoked_false",
            "execution_gate_created_false",
            "operator_ack_required_before_gate_ready",
            "idempotency_reserved_false",
            "sequence_consumer_required_before_execution",
            "no_queue_mutation",
            "no_model_mutation",
            "no_runtime_memory_mutation",
            "no_sequence_policy_trust_store_mutation",
            "no_proposal_state_mutation",
            "no_direct_sequence_execution",
            "no_live_writes",
        ),
    ),
    ProcessingMethod(
        id="operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_gate_result_consumer",
        label=(
            "Operator Proposal Apply Sequence Policy Memory Trust Promotion "
            "Queue Import Execution Gate Result Consumer"
        ),
        domain="decision_evidence",
        status="shadow_only",
        summary=(
            "Read-only verifier that matches generic apply-sequence execution "
            "gate results back to verified sequence-policy queue-import "
            "provenance before emitting a sequence-executor request template."
        ),
        module=(
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_gate_result_consumer.py"
        ),
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason=(
            "sequence_memory_trust_promotion_queue_import_execution_gate_result_consumer_is_not_executor_or_live_authority"
        ),
        promotion_next=(
            "run the emitted generic sequence-executor request only after "
            "explicit execute, operator acknowledgement, and approval flags; "
            "this consumer does not invoke the executor or reserve idempotency"
        ),
        notes=(
            "transport_free",
            "execution_gate_result_consumer_only",
            "executor_request_template_only",
            "sequence_executor_handler_invoked_false",
            "sequence_executor_created_false",
            "execute_true_required_before_executor",
            "operator_ack_required_before_executor",
            "approved_true_required_before_executor",
            "idempotency_reserved_false",
            "sequence_consumer_required_before_execution",
            "no_queue_mutation",
            "no_model_mutation",
            "no_runtime_memory_mutation",
            "no_sequence_policy_trust_store_mutation",
            "no_proposal_state_mutation",
            "no_direct_sequence_execution",
            "no_live_writes",
        ),
    ),
    ProcessingMethod(
        id="operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_result_consumer",
        label=(
            "Operator Proposal Apply Sequence Policy Memory Trust Promotion "
            "Queue Import Execution Result Consumer"
        ),
        domain="decision_evidence",
        status="shadow_only",
        summary=(
            "Read-only verifier that matches generic apply-sequence execution "
            "results back to verified sequence-policy queue-import provenance "
            "before emitting an execution-audit request template."
        ),
        module=(
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_result_consumer.py"
        ),
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason=(
            "sequence_memory_trust_promotion_queue_import_execution_result_consumer_is_not_audit_or_live_authority"
        ),
        promotion_next=(
            "run the emitted generic execution-audit request only after this "
            "consumer verifies executor-result provenance, delegated apply, "
            "readback, rollback, and idempotency evidence"
        ),
        notes=(
            "transport_free",
            "execution_result_consumer_only",
            "audit_request_template_only",
            "audit_handler_invoked_false",
            "audit_created_false",
            "requires_apply_operator_proposal_delegation",
            "readback_required",
            "rollback_required",
            "idempotency_reserved_false_by_consumer",
            "sequence_consumer_required_before_audit",
            "no_queue_mutation_by_consumer",
            "no_model_mutation",
            "no_runtime_memory_mutation",
            "no_sequence_policy_trust_store_mutation",
            "no_proposal_state_mutation_by_consumer",
            "no_direct_sequence_execution",
            "no_live_writes_by_consumer",
        ),
    ),
    ProcessingMethod(
        id="operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_audit_result_consumer",
        label=(
            "Operator Proposal Apply Sequence Policy Memory Trust Promotion "
            "Queue Import Execution Audit Result Consumer"
        ),
        domain="decision_evidence",
        status="shadow_only",
        summary=(
            "Read-only verifier that matches generic apply-sequence execution "
            "audits back to verified sequence-policy queue-import provenance "
            "before emitting an audit-policy-gate request template."
        ),
        module=(
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_audit_result_consumer.py"
        ),
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason=(
            "sequence_memory_trust_promotion_queue_import_execution_audit_result_consumer_is_not_policy_gate_or_live_authority"
        ),
        promotion_next=(
            "run the emitted generic audit-policy gate only after this consumer "
            "verifies audit provenance, passed rows, readback, rollback, and "
            "idempotency evidence"
        ),
        notes=(
            "transport_free",
            "execution_audit_result_consumer_only",
            "audit_policy_gate_request_template_only",
            "audit_policy_gate_handler_invoked_false",
            "policy_gate_created_false",
            "policy_mutation_false",
            "runtime_memory_mutation_false",
            "model_mutation_false",
            "idempotency_reserved_false_by_consumer",
            "sequence_consumer_required_before_policy_gate",
            "no_queue_mutation_by_consumer",
            "no_model_mutation",
            "no_runtime_memory_mutation",
            "no_sequence_policy_trust_store_mutation",
            "no_proposal_state_mutation_by_consumer",
            "no_direct_sequence_execution",
            "no_live_writes_by_consumer",
        ),
    ),
    ProcessingMethod(
        id="operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_audit_policy_gate_result_consumer",
        label=(
            "Operator Proposal Apply Sequence Policy Memory Trust Promotion "
            "Queue Import Audit Policy Gate Result Consumer"
        ),
        domain="decision_evidence",
        status="shadow_only",
        summary=(
            "Read-only verifier that matches generic apply-sequence audit-policy "
            "gate results back to verified sequence-policy queue-import provenance "
            "before emitting a policy-memory draft request template."
        ),
        module=(
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_audit_policy_gate_result_consumer.py"
        ),
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason=(
            "sequence_memory_trust_promotion_queue_import_audit_policy_gate_result_consumer_is_not_memory_writer_or_live_authority"
        ),
        promotion_next=(
            "run the emitted generic policy-memory draft request only after this "
            "consumer verifies policy evidence provenance, readback, rollback, "
            "idempotency, and fader/gain bounds"
        ),
        notes=(
            "transport_free",
            "audit_policy_gate_result_consumer_only",
            "memory_draft_request_template_only",
            "memory_draft_handler_invoked_false",
            "memory_draft_created_false",
            "memory_write_allowed_by_consumer_false",
            "policy_mutation_false",
            "runtime_memory_mutation_false",
            "model_mutation_false",
            "idempotency_reserved_false_by_consumer",
            "sequence_consumer_required_before_memory_draft",
            "no_queue_mutation_by_consumer",
            "no_model_mutation",
            "no_runtime_memory_mutation",
            "no_sequence_policy_trust_store_mutation",
            "no_proposal_state_mutation_by_consumer",
            "no_direct_sequence_execution",
            "no_live_writes_by_consumer",
        ),
    ),
    ProcessingMethod(
        id="operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_draft_result_consumer",
        label=(
            "Operator Proposal Apply Sequence Policy Memory Trust Promotion "
            "Queue Import Memory Draft Result Consumer"
        ),
        domain="decision_evidence",
        status="shadow_only",
        summary=(
            "Read-only verifier that matches generic policy-memory draft results "
            "back to verified sequence-policy queue-import provenance before "
            "emitting a memory-writer request template."
        ),
        module=(
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_draft_result_consumer.py"
        ),
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason=(
            "sequence_memory_trust_promotion_queue_import_memory_draft_result_consumer_is_not_memory_writer_or_live_authority"
        ),
        promotion_next=(
            "run the emitted generic memory-writer request only after this "
            "consumer verifies memory-draft provenance, append-only target, "
            "readback, rollback, idempotency, and fader/gain bounds"
        ),
        notes=(
            "transport_free",
            "memory_draft_result_consumer_only",
            "memory_writer_request_template_only",
            "memory_writer_handler_invoked_false",
            "memory_file_write_applied_false",
            "memory_write_allowed_by_consumer_false",
            "operator_ack_required_before_write",
            "apply_required_before_write",
            "policy_mutation_false",
            "runtime_memory_mutation_false",
            "model_mutation_false",
            "idempotency_reserved_false_by_consumer",
            "sequence_consumer_required_before_memory_writer",
            "no_queue_mutation_by_consumer",
            "no_model_mutation",
            "no_runtime_memory_mutation",
            "no_sequence_policy_trust_store_mutation",
            "no_proposal_state_mutation_by_consumer",
            "no_direct_sequence_execution",
            "no_live_writes_by_consumer",
        ),
    ),
    ProcessingMethod(
        id="operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_dry_run_result_consumer",
        label=(
            "Operator Proposal Apply Sequence Policy Memory Trust Promotion "
            "Queue Import Memory Writer Dry-Run Result Consumer"
        ),
        domain="decision_evidence",
        status="shadow_only",
        summary=(
            "Read-only verifier that matches generic policy-memory writer dry-run "
            "results back to verified sequence-policy queue-import provenance "
            "before emitting a memory-writer apply request template."
        ),
        module=(
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_dry_run_result_consumer.py"
        ),
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason=(
            "sequence_memory_trust_promotion_queue_import_memory_writer_dry_run_result_consumer_is_not_memory_writer_or_live_authority"
        ),
        promotion_next=(
            "run the emitted generic memory-writer apply request only after this "
            "consumer verifies writer dry-run provenance and the operator sets "
            "apply plus acknowledgement explicitly"
        ),
        notes=(
            "transport_free",
            "memory_writer_dry_run_result_consumer_only",
            "memory_writer_apply_request_template_only",
            "memory_writer_handler_invoked_false",
            "memory_file_write_applied_false",
            "memory_write_allowed_by_consumer_false",
            "operator_ack_required_before_write",
            "apply_required_before_write",
            "policy_mutation_false",
            "runtime_memory_mutation_false",
            "model_mutation_false",
            "idempotency_reserved_false_by_consumer",
            "sequence_consumer_required_before_memory_writer_apply",
            "no_queue_mutation_by_consumer",
            "no_model_mutation",
            "no_runtime_memory_mutation",
            "no_sequence_policy_trust_store_mutation",
            "no_proposal_state_mutation_by_consumer",
            "no_direct_sequence_execution",
            "no_live_writes_by_consumer",
        ),
    ),
    ProcessingMethod(
        id="operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_apply_result_consumer",
        label=(
            "Operator Proposal Apply Sequence Policy Memory Trust Promotion "
            "Queue Import Memory Writer Apply Result Consumer"
        ),
        domain="decision_evidence",
        status="shadow_only",
        summary=(
            "Read-only verifier that matches explicitly applied generic "
            "policy-memory writer results back to verified sequence-policy "
            "queue-import provenance before emitting a memory-index request "
            "template."
        ),
        module=(
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_apply_result_consumer.py"
        ),
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason=(
            "sequence_memory_trust_promotion_queue_import_memory_writer_apply_result_consumer_is_not_memory_index_or_live_authority"
        ),
        promotion_next=(
            "run the emitted read-only memory-index request only after this "
            "consumer verifies that a generic writer applied append-only memory "
            "with explicit operator acknowledgement"
        ),
        notes=(
            "transport_free",
            "memory_writer_apply_result_consumer_only",
            "memory_index_request_template_only",
            "memory_index_handler_invoked_false",
            "memory_index_created_false",
            "source_memory_file_write_applied_true_only_from_writer",
            "memory_file_write_applied_false_by_consumer",
            "memory_write_allowed_by_consumer_false",
            "policy_mutation_false",
            "runtime_memory_mutation_false",
            "model_mutation_false",
            "idempotency_reserved_false_by_consumer",
            "sequence_consumer_required_before_memory_index",
            "no_queue_mutation_by_consumer",
            "no_model_mutation",
            "no_runtime_memory_mutation",
            "no_sequence_policy_trust_store_mutation",
            "no_proposal_state_mutation_by_consumer",
            "no_direct_sequence_execution",
            "no_live_writes_by_consumer",
        ),
    ),
    ProcessingMethod(
        id="operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_index_result_consumer",
        label=(
            "Operator Proposal Apply Sequence Policy Memory Trust Promotion "
            "Queue Import Memory Index Result Consumer"
        ),
        domain="decision_evidence",
        status="shadow_only",
        summary=(
            "Read-only verifier that matches sequence-policy memory-index "
            "results back to verified queue-import writer-apply provenance "
            "before emitting a memory-trust-review request template."
        ),
        module=(
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_index_result_consumer.py"
        ),
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason=(
            "sequence_memory_trust_promotion_queue_import_memory_index_result_consumer_is_not_trust_review_or_live_authority"
        ),
        promotion_next=(
            "run the emitted read-only memory-trust-review request only after "
            "this consumer verifies memory-index records and fresh review "
            "context against queue-import writer-apply provenance"
        ),
        notes=(
            "transport_free",
            "memory_index_result_consumer_only",
            "memory_trust_review_request_template_only",
            "memory_trust_review_handler_invoked_false",
            "memory_trust_review_created_false",
            "source_memory_index_available_true_only_from_index",
            "fresh_sequence_review_context_required",
            "memory_write_allowed_by_consumer_false",
            "policy_mutation_false",
            "runtime_memory_mutation_false",
            "model_mutation_false",
            "idempotency_reserved_false_by_consumer",
            "sequence_consumer_required_before_memory_trust_review",
            "no_queue_mutation_by_consumer",
            "no_model_mutation",
            "no_runtime_memory_mutation",
            "no_sequence_policy_trust_store_mutation",
            "no_proposal_state_mutation_by_consumer",
            "no_direct_sequence_execution",
            "no_live_writes_by_consumer",
        ),
    ),
    ProcessingMethod(
        id="operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_review_result_consumer",
        label=(
            "Operator Proposal Apply Sequence Policy Memory Trust Promotion "
            "Queue Import Memory Trust Review Result Consumer"
        ),
        domain="decision_evidence",
        status="shadow_only",
        summary=(
            "Read-only verifier that matches sequence-policy memory "
            "trust-review results back to verified queue-import memory-index "
            "provenance before emitting an operator acceptance-ledger request "
            "template."
        ),
        module=(
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_review_result_consumer.py"
        ),
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason=(
            "sequence_memory_trust_promotion_queue_import_memory_trust_review_result_consumer_is_not_acceptance_or_live_authority"
        ),
        promotion_next=(
            "run the emitted read-only acceptance-ledger request only after "
            "this consumer verifies trust-review candidates and an operator "
            "supplies explicit accept/reject records"
        ),
        notes=(
            "transport_free",
            "memory_trust_review_result_consumer_only",
            "acceptance_ledger_request_template_only",
            "acceptance_ledger_handler_invoked_false",
            "acceptance_ledger_created_false",
            "operator_acceptance_required",
            "acceptance_records_empty_until_operator_supplies_decisions",
            "memory_write_allowed_by_consumer_false",
            "policy_mutation_false",
            "runtime_memory_mutation_false",
            "model_mutation_false",
            "idempotency_reserved_false_by_consumer",
            "sequence_consumer_required_before_acceptance_ledger",
            "no_queue_mutation_by_consumer",
            "no_model_mutation",
            "no_runtime_memory_mutation",
            "no_sequence_policy_trust_store_mutation",
            "no_proposal_state_mutation_by_consumer",
            "no_direct_sequence_execution",
            "no_live_writes_by_consumer",
        ),
    ),
    ProcessingMethod(
        id="operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_acceptance_ledger_result_consumer",
        label=(
            "Operator Proposal Apply Sequence Policy Memory Trust Promotion "
            "Queue Import Memory Trust Acceptance Ledger Result Consumer"
        ),
        domain="decision_evidence",
        status="shadow_only",
        summary=(
            "Read-only verifier that matches operator-supplied sequence-policy "
            "memory trust acceptance-ledger results back to queue-import "
            "trust-review provenance before emitting a promotion-plan request "
            "template."
        ),
        module=(
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_acceptance_ledger_result_consumer.py"
        ),
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason=(
            "sequence_memory_trust_promotion_queue_import_memory_trust_acceptance_ledger_result_consumer_is_not_promotion_or_live_authority"
        ),
        promotion_next=(
            "run the emitted read-only promotion-plan request only after this "
            "consumer verifies that every queue-import memory trust candidate "
            "has an explicit operator accepted/rejected ledger record"
        ),
        notes=(
            "transport_free",
            "memory_trust_acceptance_ledger_result_consumer_only",
            "promotion_plan_request_template_only",
            "promotion_plan_handler_invoked_false",
            "promotion_plan_created_false",
            "operator_acceptance_required",
            "accepted_rejected_records_must_be_operator_supplied",
            "blocks_empty_acceptance_ledger",
            "memory_write_allowed_by_consumer_false",
            "policy_mutation_false",
            "runtime_memory_mutation_false",
            "model_mutation_false",
            "idempotency_reserved_false_by_consumer",
            "no_queue_mutation_by_consumer",
            "no_model_mutation",
            "no_runtime_memory_mutation",
            "no_sequence_policy_trust_store_mutation",
            "no_proposal_state_mutation_by_consumer",
            "no_direct_sequence_execution",
            "no_live_writes_by_consumer",
        ),
    ),
    ProcessingMethod(
        id="operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_plan_result_consumer",
        label=(
            "Operator Proposal Apply Sequence Policy Memory Trust Promotion "
            "Queue Import Promotion Plan Result Consumer"
        ),
        domain="decision_evidence",
        status="shadow_only",
        summary=(
            "Read-only verifier that matches sequence-policy memory trust "
            "promotion plans back to queue-import acceptance-ledger provenance "
            "before emitting a promotion-contract request template."
        ),
        module=(
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_plan_result_consumer.py"
        ),
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason=(
            "sequence_memory_trust_promotion_queue_import_promotion_plan_result_consumer_is_not_contract_or_live_authority"
        ),
        promotion_next=(
            "run the emitted read-only promotion-contract request only with "
            "fresh sequence-policy context and rollback evidence; policy "
            "promotion remains a later accepted contract/executor path"
        ),
        notes=(
            "transport_free",
            "promotion_plan_result_consumer_only",
            "promotion_contract_request_template_only",
            "promotion_contract_handler_invoked_false",
            "promotion_contract_created_false",
            "fresh_sequence_policy_context_required",
            "rollback_evidence_required",
            "memory_write_allowed_by_consumer_false",
            "policy_mutation_false",
            "runtime_memory_mutation_false",
            "model_mutation_false",
            "idempotency_reserved_false_by_consumer",
            "no_queue_mutation_by_consumer",
            "no_model_mutation",
            "no_runtime_memory_mutation",
            "no_sequence_policy_trust_store_mutation",
            "no_proposal_state_mutation_by_consumer",
            "no_direct_sequence_execution",
            "no_live_writes_by_consumer",
        ),
    ),
    ProcessingMethod(
        id="operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_contract_result_consumer",
        label=(
            "Operator Proposal Apply Sequence Policy Memory Trust Promotion "
            "Queue Import Promotion Contract Result Consumer"
        ),
        domain="decision_evidence",
        status="shadow_only",
        summary=(
            "Read-only verifier that matches sequence-policy memory trust "
            "promotion contracts back to queue-import promotion-plan provenance "
            "before emitting a promotion-acceptance-ledger request template."
        ),
        module=(
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_contract_result_consumer.py"
        ),
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason=(
            "sequence_memory_trust_promotion_queue_import_promotion_contract_result_consumer_is_not_acceptance_or_live_authority"
        ),
        promotion_next=(
            "record explicit operator accept/reject decisions only through the "
            "separate read-only promotion acceptance ledger; this consumer does "
            "not accept contracts or promote sequence policy"
        ),
        notes=(
            "transport_free",
            "promotion_contract_result_consumer_only",
            "promotion_acceptance_ledger_request_template_only",
            "promotion_acceptance_ledger_handler_invoked_false",
            "promotion_acceptance_ledger_created_false",
            "operator_acceptance_required",
            "acceptance_records_empty_by_default",
            "policy_mutation_false",
            "runtime_memory_mutation_false",
            "model_mutation_false",
            "memory_file_mutation_false",
            "idempotency_reserved_false_by_consumer",
            "no_queue_mutation_by_consumer",
            "no_model_mutation",
            "no_runtime_memory_mutation",
            "no_sequence_policy_trust_store_mutation",
            "no_proposal_state_mutation_by_consumer",
            "no_direct_sequence_execution",
            "no_live_writes_by_consumer",
        ),
    ),
    ProcessingMethod(
        id="operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_acceptance_ledger_result_consumer",
        label=(
            "Operator Proposal Apply Sequence Policy Memory Trust Promotion "
            "Queue Import Promotion Acceptance Ledger Result Consumer"
        ),
        domain="decision_evidence",
        status="shadow_only",
        summary=(
            "Read-only verifier that matches sequence-policy memory trust "
            "promotion acceptance ledgers back to queue-import promotion-contract "
            "provenance before emitting an execution-plan request template."
        ),
        module=(
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_acceptance_ledger_result_consumer.py"
        ),
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason=(
            "sequence_memory_trust_promotion_queue_import_promotion_acceptance_ledger_result_consumer_is_not_execution_or_live_authority"
        ),
        promotion_next=(
            "run the separate read-only promotion execution-plan request only "
            "after this consumer verifies explicit operator acceptance against "
            "queue-import contract provenance"
        ),
        notes=(
            "transport_free",
            "promotion_acceptance_ledger_result_consumer_only",
            "promotion_execution_plan_request_template_only",
            "promotion_execution_plan_handler_invoked_false",
            "promotion_execution_plan_created_false",
            "accepted_promotion_record_required",
            "operator_acceptance_required",
            "policy_mutation_false",
            "runtime_memory_mutation_false",
            "model_mutation_false",
            "memory_file_mutation_false",
            "idempotency_reserved_false_by_consumer",
            "no_queue_mutation_by_consumer",
            "no_model_mutation",
            "no_runtime_memory_mutation",
            "no_sequence_policy_trust_store_mutation",
            "no_proposal_state_mutation_by_consumer",
            "no_direct_sequence_execution",
            "no_live_writes_by_consumer",
        ),
    ),
    ProcessingMethod(
        id="operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_plan_result_consumer",
        label=(
            "Operator Proposal Apply Sequence Policy Memory Trust Promotion "
            "Queue Import Promotion Execution Plan Result Consumer"
        ),
        domain="decision_evidence",
        status="shadow_only",
        summary=(
            "Read-only verifier that matches sequence-policy memory trust "
            "promotion execution plans back to queue-import promotion-acceptance "
            "ledger provenance before emitting an executor dry-run request template."
        ),
        module=(
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_plan_result_consumer.py"
        ),
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason=(
            "sequence_memory_trust_promotion_queue_import_promotion_execution_plan_result_consumer_is_not_execution_or_live_authority"
        ),
        promotion_next=(
            "run the separate read-only promotion executor dry-run request only "
            "after this consumer verifies execution-plan candidates against "
            "accepted queue-import promotion records"
        ),
        notes=(
            "transport_free",
            "promotion_execution_plan_result_consumer_only",
            "promotion_executor_dry_run_request_template_only",
            "promotion_executor_dry_run_handler_invoked_false",
            "promotion_executor_dry_run_created_false",
            "apply_false_by_consumer",
            "operator_ack_false_by_consumer",
            "policy_mutation_false",
            "runtime_memory_mutation_false",
            "model_mutation_false",
            "memory_file_mutation_false",
            "idempotency_reserved_false_by_consumer",
            "no_queue_mutation_by_consumer",
            "no_model_mutation",
            "no_runtime_memory_mutation",
            "no_sequence_policy_trust_store_mutation",
            "no_proposal_state_mutation_by_consumer",
            "no_direct_sequence_execution",
            "no_live_writes_by_consumer",
        ),
    ),
    ProcessingMethod(
        id="operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_dry_run_result_consumer",
        label=(
            "Operator Proposal Apply Sequence Policy Memory Trust Promotion "
            "Queue Import Promotion Executor Dry Run Result Consumer"
        ),
        domain="decision_evidence",
        status="shadow_only",
        summary=(
            "Read-only verifier that matches sequence-policy memory trust "
            "promotion executor dry-run specs back to queue-import execution-plan "
            "provenance before emitting a promotion executor request template."
        ),
        module=(
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_dry_run_result_consumer.py"
        ),
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason=(
            "sequence_memory_trust_promotion_queue_import_promotion_executor_dry_run_result_consumer_is_not_execution_or_live_authority"
        ),
        promotion_next=(
            "run the separate read-only promotion executor preview only after "
            "this consumer verifies dry-run specs against accepted execution-plan "
            "candidates"
        ),
        notes=(
            "transport_free",
            "promotion_executor_dry_run_result_consumer_only",
            "promotion_executor_request_template_only",
            "promotion_executor_handler_invoked_false",
            "promotion_executor_created_false",
            "apply_false_by_consumer",
            "operator_ack_false_by_consumer",
            "target_policy_store_required_before_apply",
            "policy_mutation_false",
            "runtime_memory_mutation_false",
            "model_mutation_false",
            "memory_file_mutation_false",
            "idempotency_reserved_false_by_consumer",
            "no_queue_mutation_by_consumer",
            "no_model_mutation",
            "no_runtime_memory_mutation",
            "no_sequence_policy_trust_store_mutation_by_consumer",
            "no_proposal_state_mutation_by_consumer",
            "no_direct_sequence_execution",
            "no_live_writes_by_consumer",
        ),
    ),
    ProcessingMethod(
        id="operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_result_consumer",
        label=(
            "Operator Proposal Apply Sequence Policy Memory Trust Promotion "
            "Queue Import Promotion Executor Result Consumer"
        ),
        domain="decision_evidence",
        status="shadow_only",
        summary=(
            "Read-only verifier that matches sequence-policy memory trust "
            "promotion executor output back to queue-import executor dry-run "
            "provenance before emitting an apply-audit request template."
        ),
        module=(
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_result_consumer.py"
        ),
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason=(
            "sequence_memory_trust_promotion_queue_import_promotion_executor_result_consumer_is_not_execution_or_live_authority"
        ),
        promotion_next=(
            "run the separate read-only promotion apply audit only after this "
            "consumer verifies applied executor results against queue-import "
            "executor dry-run provenance"
        ),
        notes=(
            "transport_free",
            "promotion_executor_result_consumer_only",
            "promotion_apply_audit_request_template_only",
            "promotion_apply_audit_handler_invoked_false",
            "promotion_apply_audit_created_false",
            "requires_applied_executor_result",
            "policy_mutation_false_by_consumer",
            "runtime_memory_mutation_false_by_consumer",
            "model_mutation_false",
            "memory_file_mutation_false",
            "idempotency_reserved_false_by_consumer",
            "no_queue_mutation_by_consumer",
            "no_model_mutation",
            "no_runtime_memory_mutation_by_consumer",
            "no_sequence_policy_trust_store_mutation_by_consumer",
            "no_proposal_state_mutation_by_consumer",
            "no_direct_sequence_execution",
            "no_live_writes_by_consumer",
        ),
    ),
    ProcessingMethod(
        id="operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_apply_audit_result_consumer",
        label=(
            "Operator Proposal Apply Sequence Policy Memory Trust Promotion "
            "Queue Import Promotion Apply Audit Result Consumer"
        ),
        domain="decision_evidence",
        status="shadow_only",
        summary=(
            "Read-only verifier that matches sequence-policy memory trust "
            "promotion apply-audit records back to queue-import executor-result "
            "provenance before emitting a rollback-verification request template."
        ),
        module=(
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_apply_audit_result_consumer.py"
        ),
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason=(
            "sequence_memory_trust_promotion_queue_import_promotion_apply_audit_result_consumer_is_not_rollback_verification_or_live_authority"
        ),
        promotion_next=(
            "run the separate read-only rollback verification only after this "
            "consumer verifies apply-audit records against queue-import "
            "executor-result provenance and the operator supplies verification records"
        ),
        notes=(
            "transport_free",
            "promotion_apply_audit_result_consumer_only",
            "promotion_rollback_verification_request_template_only",
            "promotion_rollback_verification_handler_invoked_false",
            "promotion_rollback_verification_created_false",
            "requires_apply_audit_records",
            "verification_records_required_from_operator",
            "policy_mutation_false_by_consumer",
            "runtime_memory_mutation_false_by_consumer",
            "model_mutation_false",
            "memory_file_mutation_false",
            "idempotency_reserved_false_by_consumer",
            "no_queue_mutation_by_consumer",
            "no_model_mutation",
            "no_runtime_memory_mutation_by_consumer",
            "no_sequence_policy_trust_store_mutation_by_consumer",
            "no_proposal_state_mutation_by_consumer",
            "no_direct_sequence_execution",
            "no_live_writes_by_consumer",
        ),
    ),
    ProcessingMethod(
        id="operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_rollback_verification_result_consumer",
        label=(
            "Operator Proposal Apply Sequence Policy Memory Trust Promotion "
            "Queue Import Promotion Rollback Verification Result Consumer"
        ),
        domain="decision_evidence",
        status="shadow_only",
        summary=(
            "Read-only verifier that matches sequence-policy memory trust "
            "promotion rollback-verification records back to queue-import "
            "apply-audit provenance before emitting a sequence-policy gate "
            "request template."
        ),
        module=(
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_rollback_verification_result_consumer.py"
        ),
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason=(
            "sequence_memory_trust_promotion_queue_import_promotion_rollback_verification_result_consumer_is_not_sequence_policy_gate_or_live_authority"
        ),
        promotion_next=(
            "run the separate read-only sequence-policy gate only after this "
            "consumer verifies rollback records against queue-import "
            "apply-audit provenance"
        ),
        notes=(
            "transport_free",
            "promotion_rollback_verification_result_consumer_only",
            "promotion_sequence_policy_gate_request_template_only",
            "promotion_sequence_policy_gate_handler_invoked_false",
            "promotion_sequence_policy_gate_created_false",
            "requires_verified_rollback_records",
            "policy_mutation_false_by_consumer",
            "runtime_memory_mutation_false_by_consumer",
            "model_mutation_false",
            "memory_file_mutation_false",
            "idempotency_reserved_false_by_consumer",
            "no_queue_mutation_by_consumer",
            "no_model_mutation",
            "no_runtime_memory_mutation_by_consumer",
            "no_sequence_policy_trust_store_mutation_by_consumer",
            "no_proposal_state_mutation_by_consumer",
            "no_direct_sequence_execution",
            "no_live_writes_by_consumer",
        ),
    ),
    ProcessingMethod(
        id="operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_sequence_policy_gate_result_consumer",
        label=(
            "Operator Proposal Apply Sequence Policy Memory Trust Promotion "
            "Queue Import Promotion Sequence Policy Gate Result Consumer"
        ),
        domain="decision_evidence",
        status="shadow_only",
        summary=(
            "Read-only verifier that matches sequence-policy gate records back "
            "to queue-import rollback-verification provenance before emitting "
            "a policy-consumer request template."
        ),
        module=(
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_sequence_policy_gate_result_consumer.py"
        ),
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason=(
            "sequence_memory_trust_promotion_queue_import_promotion_sequence_policy_gate_result_consumer_is_not_policy_consumer_or_live_authority"
        ),
        promotion_next=(
            "run the separate read-only policy consumer only after this consumer "
            "verifies sequence-policy gate records against queue-import "
            "rollback-verification provenance and operator proposals are supplied"
        ),
        notes=(
            "transport_free",
            "promotion_sequence_policy_gate_result_consumer_only",
            "promotion_policy_consumer_request_template_only",
            "promotion_policy_consumer_handler_invoked_false",
            "promotion_policy_consumer_created_false",
            "requires_policy_records",
            "proposals_required_from_operator",
            "policy_mutation_false_by_consumer",
            "runtime_memory_mutation_false_by_consumer",
            "model_mutation_false",
            "memory_file_mutation_false",
            "idempotency_reserved_false_by_consumer",
            "no_queue_mutation_by_consumer",
            "no_proposal_order_mutation_by_consumer",
            "no_model_mutation",
            "no_runtime_memory_mutation_by_consumer",
            "no_sequence_policy_trust_store_mutation_by_consumer",
            "no_proposal_state_mutation_by_consumer",
            "no_direct_sequence_execution",
            "no_live_writes_by_consumer",
        ),
    ),
    ProcessingMethod(
        id="operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_policy_consumer_result_consumer",
        label=(
            "Operator Proposal Apply Sequence Policy Memory Trust Promotion "
            "Queue Import Promotion Policy Consumer Result Consumer"
        ),
        domain="decision_evidence",
        status="shadow_only",
        summary=(
            "Read-only verifier that matches policy-consumer records back to "
            "queue-import sequence-policy gate provenance before emitting an "
            "ordering-preview request template."
        ),
        module=(
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_policy_consumer_result_consumer.py"
        ),
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason=(
            "sequence_memory_trust_promotion_queue_import_promotion_policy_consumer_result_consumer_is_not_ordering_preview_or_live_authority"
        ),
        promotion_next=(
            "run the separate read-only ordering preview only after this "
            "consumer verifies policy-consumer records against queue-import "
            "sequence-policy gate provenance and operator proposals are supplied"
        ),
        notes=(
            "transport_free",
            "promotion_policy_consumer_result_consumer_only",
            "promotion_ordering_preview_request_template_only",
            "promotion_ordering_preview_handler_invoked_false",
            "promotion_ordering_preview_created_false",
            "requires_policy_consumer_records",
            "proposals_required_from_operator",
            "policy_mutation_false_by_consumer",
            "runtime_memory_mutation_false_by_consumer",
            "model_mutation_false",
            "memory_file_mutation_false",
            "idempotency_reserved_false_by_consumer",
            "no_queue_mutation_by_consumer",
            "no_proposal_order_mutation_by_consumer",
            "no_model_mutation",
            "no_runtime_memory_mutation_by_consumer",
            "no_sequence_policy_trust_store_mutation_by_consumer",
            "no_proposal_state_mutation_by_consumer",
            "no_direct_sequence_execution",
            "no_live_writes_by_consumer",
        ),
    ),
    ProcessingMethod(
        id="operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_ordering_preview_result_consumer",
        label=(
            "Operator Proposal Apply Sequence Policy Memory Trust Promotion "
            "Queue Import Promotion Ordering Preview Result Consumer"
        ),
        domain="decision_evidence",
        status="shadow_only",
        summary=(
            "Read-only verifier that matches ordering-preview rows back to "
            "queue-import policy-consumer provenance before emitting a "
            "queue-ordering-plan request template."
        ),
        module=(
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_ordering_preview_result_consumer.py"
        ),
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason=(
            "sequence_memory_trust_promotion_queue_import_promotion_ordering_preview_result_consumer_is_not_queue_ordering_plan_or_live_authority"
        ),
        promotion_next=(
            "run the separate read-only queue-ordering plan only after this "
            "consumer verifies ordering-preview rows against queue-import "
            "policy-consumer provenance"
        ),
        notes=(
            "transport_free",
            "promotion_ordering_preview_result_consumer_only",
            "promotion_queue_ordering_plan_request_template_only",
            "promotion_queue_ordering_plan_handler_invoked_false",
            "promotion_queue_ordering_plan_created_false",
            "requires_ordered_preview_records",
            "policy_mutation_false_by_consumer",
            "runtime_memory_mutation_false_by_consumer",
            "model_mutation_false",
            "memory_file_mutation_false",
            "idempotency_reserved_false_by_consumer",
            "no_queue_mutation_by_consumer",
            "no_proposal_order_mutation_by_consumer",
            "no_model_mutation",
            "no_runtime_memory_mutation_by_consumer",
            "no_sequence_policy_trust_store_mutation_by_consumer",
            "no_proposal_state_mutation_by_consumer",
            "no_direct_sequence_execution",
            "no_live_writes_by_consumer",
        ),
    ),
    ProcessingMethod(
        id="operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_ordering_plan_result_consumer",
        label=(
            "Operator Proposal Apply Sequence Policy Memory Trust Promotion "
            "Queue Import Promotion Queue Ordering Plan Result Consumer"
        ),
        domain="decision_evidence",
        status="shadow_only",
        summary=(
            "Read-only verifier that matches queue-ordering-plan records back "
            "to queue-import ordering-preview provenance before emitting a "
            "queue-importer request template."
        ),
        module=(
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_ordering_plan_result_consumer.py"
        ),
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason=(
            "sequence_memory_trust_promotion_queue_import_promotion_queue_ordering_plan_result_consumer_is_not_queue_importer_or_live_authority"
        ),
        promotion_next=(
            "run the separate queue importer dry-run only after this consumer "
            "verifies queue-ordering-plan records against queue-import "
            "ordering-preview provenance and operator proposal payloads are supplied"
        ),
        notes=(
            "transport_free",
            "promotion_queue_ordering_plan_result_consumer_only",
            "promotion_queue_importer_request_template_only",
            "promotion_queue_importer_handler_invoked_false",
            "promotion_queue_importer_created_false",
            "requires_queue_ordering_plan_records",
            "proposal_payloads_required_from_operator",
            "apply_false_by_template",
            "operator_ack_false_by_template",
            "policy_mutation_false_by_consumer",
            "runtime_memory_mutation_false_by_consumer",
            "model_mutation_false",
            "memory_file_mutation_false",
            "idempotency_reserved_false_by_consumer",
            "no_queue_mutation_by_consumer",
            "no_proposal_order_mutation_by_consumer",
            "no_model_mutation",
            "no_runtime_memory_mutation_by_consumer",
            "no_sequence_policy_trust_store_mutation_by_consumer",
            "no_proposal_state_mutation_by_consumer",
            "no_direct_sequence_execution",
            "no_live_writes_by_consumer",
        ),
    ),
    ProcessingMethod(
        id="operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer",
        label=(
            "Operator Proposal Apply Sequence Policy Memory Trust Promotion "
            "Queue Import Promotion Queue Importer Result Consumer"
        ),
        domain="decision_evidence",
        status="shadow_only",
        summary=(
            "Read-only verifier that matches queue-importer results back to "
            "promotion queue-ordering-plan provenance before emitting a "
            "queue-import preflight-plan request template."
        ),
        module=(
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer.py"
        ),
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason=(
            "sequence_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer_is_not_preflight_or_live_authority"
        ),
        promotion_next=(
            "run the separate read-only queue-import preflight plan only after "
            "this consumer verifies an applied queue-importer result against "
            "queue-import queue-ordering-plan provenance"
        ),
        notes=(
            "transport_free",
            "promotion_queue_importer_result_consumer_only",
            "promotion_queue_import_preflight_plan_request_template_only",
            "promotion_queue_import_preflight_plan_handler_invoked_false",
            "promotion_queue_import_preflight_plan_created_false",
            "requires_applied_queue_importer_result",
            "pending_actions_required_from_operator_queue",
            "policy_mutation_false_by_consumer",
            "runtime_memory_mutation_false_by_consumer",
            "model_mutation_false",
            "memory_file_mutation_false",
            "idempotency_reserved_false_by_consumer",
            "no_queue_mutation_by_consumer",
            "no_proposal_order_mutation_by_consumer",
            "no_model_mutation",
            "no_runtime_memory_mutation_by_consumer",
            "no_sequence_policy_trust_store_mutation_by_consumer",
            "no_proposal_state_mutation_by_consumer",
            "no_direct_sequence_execution",
            "no_live_writes_by_consumer",
        ),
    ),
    ProcessingMethod(
        id="operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_preflight_plan_result_consumer",
        label=(
            "Operator Proposal Apply Sequence Policy Memory Trust Promotion "
            "Queue Import Promotion Preflight Plan Result Consumer"
        ),
        domain="decision_evidence",
        status="shadow_only",
        summary=(
            "Read-only verifier that matches queue-import preflight plans back "
            "to applied queue-importer provenance before emitting a "
            "batch-preflight request template."
        ),
        module=(
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_preflight_plan_result_consumer.py"
        ),
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason=(
            "sequence_memory_trust_promotion_queue_import_promotion_preflight_plan_result_consumer_is_not_batch_preflight_or_live_authority"
        ),
        promotion_next=(
            "run the separate batch-preflight handler only after this consumer "
            "verifies a queue-import preflight plan against applied "
            "queue-importer provenance and explicit operator approval is set"
        ),
        notes=(
            "transport_free",
            "promotion_preflight_plan_result_consumer_only",
            "promotion_batch_preflight_request_template_only",
            "promotion_batch_preflight_handler_invoked_false",
            "promotion_batch_preflight_created_false",
            "operator_must_set_approved_true",
            "idempotency_reserved_false_by_consumer",
            "no_queue_mutation_by_consumer",
            "no_proposal_status_mutation_by_consumer",
            "no_proposal_order_mutation_by_consumer",
            "no_model_mutation",
            "no_runtime_memory_mutation_by_consumer",
            "no_sequence_policy_trust_store_mutation_by_consumer",
            "no_proposal_state_mutation_by_consumer",
            "no_direct_sequence_execution",
            "no_live_writes_by_consumer",
        ),
    ),
    ProcessingMethod(
        id="operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_batch_preflight_result_consumer",
        label=(
            "Operator Proposal Apply Sequence Policy Memory Trust Promotion "
            "Queue Import Promotion Batch Preflight Result Consumer"
        ),
        domain="decision_evidence",
        status="shadow_only",
        summary=(
            "Read-only verifier that matches approved batch-preflight results "
            "back to promotion preflight-plan provenance before emitting a "
            "sequence-plan request template."
        ),
        module=(
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_batch_preflight_result_consumer.py"
        ),
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason=(
            "sequence_memory_trust_promotion_queue_import_promotion_batch_preflight_result_consumer_is_not_sequence_plan_or_live_authority"
        ),
        promotion_next=(
            "run the separate generic sequence-plan handler only after this "
            "consumer verifies approved batch-preflight results against "
            "promotion preflight-plan provenance"
        ),
        notes=(
            "transport_free",
            "promotion_batch_preflight_result_consumer_only",
            "promotion_sequence_plan_request_template_only",
            "sequence_plan_handler_invoked_false",
            "sequence_plan_created_false",
            "requires_all_preflight_results_ready",
            "idempotency_reserved_false_by_consumer",
            "no_queue_mutation_by_consumer",
            "no_proposal_status_mutation_by_consumer",
            "no_proposal_order_mutation_by_consumer",
            "no_model_mutation",
            "no_runtime_memory_mutation_by_consumer",
            "no_sequence_policy_trust_store_mutation_by_consumer",
            "no_proposal_state_mutation_by_consumer",
            "no_direct_sequence_execution",
            "no_live_writes_by_consumer",
        ),
    ),
    ProcessingMethod(
        id="operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_sequence_plan_result_consumer",
        label=(
            "Operator Proposal Apply Sequence Policy Memory Trust Promotion "
            "Queue Import Promotion Sequence Plan Result Consumer"
        ),
        domain="decision_evidence",
        status="shadow_only",
        summary=(
            "Read-only verifier that matches generic sequence plans back to "
            "promotion batch-preflight provenance before emitting a dry-run "
            "request template."
        ),
        module=(
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_sequence_plan_result_consumer.py"
        ),
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason=(
            "sequence_memory_trust_promotion_queue_import_promotion_sequence_plan_result_consumer_is_not_dry_run_or_live_authority"
        ),
        promotion_next=(
            "run the separate generic dry-run handler only after this consumer "
            "verifies a sequence plan against promotion batch-preflight "
            "provenance"
        ),
        notes=(
            "transport_free",
            "promotion_sequence_plan_result_consumer_only",
            "promotion_dry_run_request_template_only",
            "dry_run_handler_invoked_false",
            "dry_run_created_false",
            "requires_all_sequence_plan_steps_planned",
            "idempotency_reserved_false_by_consumer",
            "no_queue_mutation_by_consumer",
            "no_proposal_status_mutation_by_consumer",
            "no_proposal_order_mutation_by_consumer",
            "no_model_mutation",
            "no_runtime_memory_mutation_by_consumer",
            "no_sequence_policy_trust_store_mutation_by_consumer",
            "no_proposal_state_mutation_by_consumer",
            "no_direct_sequence_execution",
            "no_live_writes_by_consumer",
        ),
    ),
    ProcessingMethod(
        id="operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_dry_run_result_consumer",
        label=(
            "Operator Proposal Apply Sequence Policy Memory Trust Promotion "
            "Queue Import Promotion Dry Run Result Consumer"
        ),
        domain="decision_evidence",
        status="shadow_only",
        summary=(
            "Read-only verifier that matches generic sequence dry-runs back "
            "to promotion sequence-plan provenance before emitting an "
            "execution-gate request template."
        ),
        module=(
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_dry_run_result_consumer.py"
        ),
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason=(
            "sequence_memory_trust_promotion_queue_import_promotion_dry_run_result_consumer_is_not_execution_gate_or_live_authority"
        ),
        promotion_next=(
            "run the separate generic execution gate only after this consumer "
            "verifies dry-run evidence against promotion sequence-plan "
            "provenance and explicit operator acknowledgement is set"
        ),
        notes=(
            "transport_free",
            "promotion_dry_run_result_consumer_only",
            "promotion_execution_gate_request_template_only",
            "execution_gate_handler_invoked_false",
            "execution_gate_created_false",
            "operator_ack_required_before_gate_ready",
            "requires_all_dry_run_results_would_apply",
            "idempotency_reserved_false_by_consumer",
            "no_queue_mutation_by_consumer",
            "no_proposal_status_mutation_by_consumer",
            "no_proposal_order_mutation_by_consumer",
            "no_model_mutation",
            "no_runtime_memory_mutation_by_consumer",
            "no_sequence_policy_trust_store_mutation_by_consumer",
            "no_proposal_state_mutation_by_consumer",
            "no_direct_sequence_execution",
            "no_live_writes_by_consumer",
        ),
    ),
    ProcessingMethod(
        id="operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_gate_result_consumer",
        label=(
            "Operator Proposal Apply Sequence Policy Memory Trust Promotion "
            "Queue Import Promotion Execution Gate Result Consumer"
        ),
        domain="decision_evidence",
        status="shadow_only",
        summary=(
            "Read-only verifier that matches generic execution-gate results "
            "back to promotion dry-run provenance before emitting an executor "
            "request template."
        ),
        module=(
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_gate_result_consumer.py"
        ),
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason=(
            "sequence_memory_trust_promotion_queue_import_promotion_execution_gate_result_consumer_is_not_executor_or_live_authority"
        ),
        promotion_next=(
            "run the generic sequence executor only after this consumer verifies "
            "execution-gate evidence against promotion dry-run provenance and "
            "execute, operator acknowledgement, and approval are supplied "
            "explicitly"
        ),
        notes=(
            "transport_free",
            "promotion_execution_gate_result_consumer_only",
            "promotion_executor_request_template_only",
            "sequence_executor_handler_invoked_false",
            "sequence_executor_created_false",
            "execute_true_required_before_executor",
            "operator_ack_required_before_executor",
            "approved_true_required_before_executor",
            "requires_all_gate_rows_ready_for_executor",
            "idempotency_reserved_false_by_consumer",
            "no_queue_mutation_by_consumer",
            "no_proposal_status_mutation_by_consumer",
            "no_proposal_order_mutation_by_consumer",
            "no_policy_mutation_by_consumer",
            "no_memory_mutation_by_consumer",
            "no_model_mutation",
            "no_runtime_memory_mutation_by_consumer",
            "no_sequence_policy_trust_store_mutation_by_consumer",
            "no_proposal_state_mutation_by_consumer",
            "no_direct_sequence_execution",
            "no_live_writes_by_consumer",
        ),
    ),
    ProcessingMethod(
        id="operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_result_consumer",
        label=(
            "Operator Proposal Apply Sequence Policy Memory Trust Promotion "
            "Queue Import Promotion Execution Result Consumer"
        ),
        domain="decision_evidence",
        status="shadow_only",
        summary=(
            "Read-only verifier that matches generic sequence-executor results "
            "back to promotion execution-gate provenance before emitting an "
            "execution-audit request template."
        ),
        module=(
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_result_consumer.py"
        ),
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason=(
            "sequence_memory_trust_promotion_queue_import_promotion_execution_result_consumer_is_not_execution_audit_or_live_authority"
        ),
        promotion_next=(
            "run the generic execution audit only after this consumer verifies "
            "sequence-executor evidence against promotion execution-gate "
            "provenance"
        ),
        notes=(
            "transport_free",
            "promotion_execution_result_consumer_only",
            "promotion_execution_audit_request_template_only",
            "execution_audit_handler_invoked_false",
            "execution_audit_created_false",
            "requires_apply_operator_proposal_delegation",
            "requires_all_execution_results_applied",
            "idempotency_reserved_false_by_consumer",
            "no_queue_mutation_by_consumer",
            "no_proposal_status_mutation_by_consumer",
            "no_proposal_order_mutation_by_consumer",
            "no_policy_mutation_by_consumer",
            "no_memory_mutation_by_consumer",
            "no_model_mutation",
            "no_runtime_memory_mutation_by_consumer",
            "no_sequence_policy_trust_store_mutation_by_consumer",
            "no_proposal_state_mutation_by_consumer",
            "no_direct_sequence_execution",
            "no_live_writes_by_consumer",
        ),
    ),
    ProcessingMethod(
        id="operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_audit_result_consumer",
        label=(
            "Operator Proposal Apply Sequence Policy Memory Trust Promotion "
            "Queue Import Promotion Execution Audit Result Consumer"
        ),
        domain="decision_evidence",
        status="shadow_only",
        summary=(
            "Read-only verifier that matches generic execution-audit results "
            "back to promotion execution-result provenance before emitting an "
            "audit-policy-gate request template."
        ),
        module=(
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_audit_result_consumer.py"
        ),
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason=(
            "sequence_memory_trust_promotion_queue_import_promotion_execution_audit_result_consumer_is_not_policy_gate_or_live_authority"
        ),
        promotion_next=(
            "run the generic audit-policy gate only after this consumer "
            "verifies execution-audit evidence against promotion "
            "execution-result provenance"
        ),
        notes=(
            "transport_free",
            "promotion_execution_audit_result_consumer_only",
            "promotion_audit_policy_gate_request_template_only",
            "audit_policy_gate_handler_invoked_false",
            "policy_gate_created_false",
            "requires_all_execution_audit_rows_passed",
            "idempotency_reserved_false_by_consumer",
            "no_queue_mutation_by_consumer",
            "no_proposal_status_mutation_by_consumer",
            "no_proposal_order_mutation_by_consumer",
            "no_policy_mutation_by_consumer",
            "no_memory_mutation_by_consumer",
            "no_model_mutation",
            "no_runtime_memory_mutation_by_consumer",
            "no_sequence_policy_trust_store_mutation_by_consumer",
            "no_proposal_state_mutation_by_consumer",
            "no_direct_sequence_execution",
            "no_live_writes_by_consumer",
        ),
    ),
    ProcessingMethod(
        id="operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_audit_policy_gate_result_consumer",
        label=(
            "Operator Proposal Apply Sequence Policy Memory Trust Promotion "
            "Queue Import Promotion Audit Policy Gate Result Consumer"
        ),
        domain="decision_evidence",
        status="shadow_only",
        summary=(
            "Read-only verifier that matches generic audit-policy-gate results "
            "back to promotion execution-audit provenance before emitting a "
            "memory-draft request template."
        ),
        module=(
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_audit_policy_gate_result_consumer.py"
        ),
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason=(
            "sequence_memory_trust_promotion_queue_import_promotion_audit_policy_gate_result_consumer_is_not_memory_draft_or_live_authority"
        ),
        promotion_next=(
            "run the generic policy-memory draft only after this consumer "
            "verifies audit-policy-gate evidence against promotion "
            "execution-audit provenance"
        ),
        notes=(
            "transport_free",
            "promotion_audit_policy_gate_result_consumer_only",
            "promotion_memory_draft_request_template_only",
            "memory_draft_handler_invoked_false",
            "memory_draft_created_false",
            "memory_write_allowed_by_consumer_false",
            "requires_all_policy_evidence_ready",
            "idempotency_reserved_false_by_consumer",
            "no_queue_mutation_by_consumer",
            "no_proposal_status_mutation_by_consumer",
            "no_proposal_order_mutation_by_consumer",
            "no_policy_mutation_by_consumer",
            "no_memory_mutation_by_consumer",
            "no_memory_file_mutation_by_consumer",
            "no_model_mutation",
            "no_runtime_memory_mutation_by_consumer",
            "no_sequence_policy_trust_store_mutation_by_consumer",
            "no_proposal_state_mutation_by_consumer",
            "no_direct_sequence_execution",
            "no_live_writes_by_consumer",
        ),
    ),
    ProcessingMethod(
        id="operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_draft_result_consumer",
        label=(
            "Operator Proposal Apply Sequence Policy Memory Trust Promotion "
            "Queue Import Promotion Memory Draft Result Consumer"
        ),
        domain="decision_evidence",
        status="shadow_only",
        summary=(
            "Read-only verifier that matches generic policy-memory draft "
            "results back to promotion audit-policy-gate provenance before "
            "emitting a memory-writer request template."
        ),
        module=(
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_draft_result_consumer.py"
        ),
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason=(
            "sequence_memory_trust_promotion_queue_import_promotion_memory_draft_result_consumer_is_not_memory_writer_or_live_authority"
        ),
        promotion_next=(
            "run the generic memory-writer dry-run only after this consumer "
            "verifies memory-draft evidence against promotion audit-policy-gate "
            "provenance"
        ),
        notes=(
            "transport_free",
            "promotion_memory_draft_result_consumer_only",
            "promotion_memory_writer_request_template_only",
            "memory_writer_handler_invoked_false",
            "memory_file_write_applied_false",
            "memory_write_allowed_by_consumer_false",
            "requires_all_memory_draft_records_ready",
            "requires_operator_ack_before_writer_apply",
            "requires_apply_true_before_writer_apply",
            "idempotency_reserved_false_by_consumer",
            "no_queue_mutation_by_consumer",
            "no_proposal_status_mutation_by_consumer",
            "no_proposal_order_mutation_by_consumer",
            "no_policy_mutation_by_consumer",
            "no_memory_mutation_by_consumer",
            "no_memory_file_mutation_by_consumer",
            "no_model_mutation",
            "no_runtime_memory_mutation_by_consumer",
            "no_sequence_policy_trust_store_mutation_by_consumer",
            "no_proposal_state_mutation_by_consumer",
            "no_direct_sequence_execution",
            "no_live_writes_by_consumer",
        ),
    ),
    ProcessingMethod(
        id="operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_writer_dry_run_result_consumer",
        label=(
            "Operator Proposal Apply Sequence Policy Memory Trust Promotion "
            "Queue Import Promotion Memory Writer Dry Run Result Consumer"
        ),
        domain="decision_evidence",
        status="shadow_only",
        summary=(
            "Read-only verifier that matches generic memory-writer dry-run "
            "results back to promotion memory-draft provenance before emitting "
            "a memory-writer apply request template."
        ),
        module=(
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_writer_dry_run_result_consumer.py"
        ),
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason=(
            "sequence_memory_trust_promotion_queue_import_promotion_memory_writer_dry_run_result_consumer_is_not_memory_writer_apply_or_live_authority"
        ),
        promotion_next=(
            "prepare an explicit memory-writer apply request only after this "
            "consumer verifies dry-run evidence against promotion memory-draft "
            "provenance"
        ),
        notes=(
            "transport_free",
            "promotion_memory_writer_dry_run_result_consumer_only",
            "promotion_memory_writer_apply_request_template_only",
            "memory_writer_handler_invoked_false",
            "memory_file_write_applied_false",
            "memory_write_allowed_by_consumer_false",
            "requires_operator_ack_before_writer_apply",
            "requires_apply_true_before_writer_apply",
            "idempotency_reserved_false_by_consumer",
            "no_queue_mutation_by_consumer",
            "no_proposal_status_mutation_by_consumer",
            "no_proposal_order_mutation_by_consumer",
            "no_policy_mutation_by_consumer",
            "no_memory_mutation_by_consumer",
            "no_memory_file_mutation_by_consumer",
            "no_model_mutation",
            "no_runtime_memory_mutation_by_consumer",
            "no_sequence_policy_trust_store_mutation_by_consumer",
            "no_proposal_state_mutation_by_consumer",
            "no_direct_sequence_execution",
            "no_live_writes_by_consumer",
        ),
    ),
    ProcessingMethod(
        id="operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_writer_apply_result_consumer",
        label=(
            "Operator Proposal Apply Sequence Policy Memory Trust Promotion "
            "Queue Import Promotion Memory Writer Apply Result Consumer"
        ),
        domain="decision_evidence",
        status="shadow_only",
        summary=(
            "Read-only verifier that matches explicitly applied generic "
            "append-only memory-writer results back to promotion dry-run "
            "provenance before emitting a memory-index request template."
        ),
        module=(
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_writer_apply_result_consumer.py"
        ),
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason=(
            "sequence_memory_trust_promotion_queue_import_promotion_memory_writer_apply_result_consumer_is_not_memory_index_or_live_authority"
        ),
        promotion_next=(
            "run the emitted read-only memory-index request only after this "
            "consumer verifies that the generic memory writer applied "
            "append-only memory with explicit operator acknowledgement"
        ),
        notes=(
            "transport_free",
            "promotion_memory_writer_apply_result_consumer_only",
            "promotion_memory_index_request_template_only",
            "memory_index_handler_invoked_false",
            "memory_index_created_false",
            "source_memory_file_write_applied_true_only_from_writer",
            "memory_file_write_applied_false_by_consumer",
            "memory_write_allowed_by_consumer_false",
            "consumer_side_memory_mutation_false",
            "policy_mutation_false",
            "runtime_memory_mutation_false",
            "model_mutation_false",
            "idempotency_reserved_false_by_consumer",
            "promotion_dry_run_consumer_required_before_memory_index",
            "no_queue_mutation_by_consumer",
            "no_proposal_status_mutation_by_consumer",
            "no_proposal_order_mutation_by_consumer",
            "no_policy_mutation_by_consumer",
            "no_memory_mutation_by_consumer",
            "no_memory_file_mutation_by_consumer",
            "no_model_mutation",
            "no_runtime_memory_mutation_by_consumer",
            "no_sequence_policy_trust_store_mutation_by_consumer",
            "no_proposal_state_mutation_by_consumer",
            "no_direct_sequence_execution",
            "no_live_writes_by_consumer",
        ),
    ),
    ProcessingMethod(
        id="operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_index_result_consumer",
        label=(
            "Operator Proposal Apply Sequence Policy Memory Trust Promotion "
            "Queue Import Promotion Memory Index Result Consumer"
        ),
        domain="decision_evidence",
        status="shadow_only",
        summary=(
            "Read-only verifier that matches read-only sequence memory index "
            "results back to promotion memory-writer apply provenance before "
            "emitting a memory-trust-review request template."
        ),
        module=(
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_index_result_consumer.py"
        ),
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason=(
            "sequence_memory_trust_promotion_queue_import_promotion_memory_index_result_consumer_is_not_trust_review_or_live_authority"
        ),
        promotion_next=(
            "run the emitted read-only memory-trust-review request only after "
            "this consumer verifies that indexed sequence memory matches "
            "promotion writer-apply provenance and fresh review context"
        ),
        notes=(
            "transport_free",
            "promotion_memory_index_result_consumer_only",
            "promotion_memory_trust_review_request_template_only",
            "memory_trust_review_handler_invoked_false",
            "memory_trust_review_created_false",
            "source_memory_index_available_true_only_from_index",
            "memory_write_allowed_by_consumer_false",
            "consumer_side_memory_mutation_false",
            "policy_mutation_false",
            "runtime_memory_mutation_false",
            "model_mutation_false",
            "idempotency_reserved_false_by_consumer",
            "requires_fresh_sequence_review_context",
            "promotion_writer_apply_consumer_required_before_trust_review",
            "no_queue_mutation_by_consumer",
            "no_proposal_status_mutation_by_consumer",
            "no_proposal_order_mutation_by_consumer",
            "no_policy_mutation_by_consumer",
            "no_memory_mutation_by_consumer",
            "no_memory_file_mutation_by_consumer",
            "no_model_mutation",
            "no_runtime_memory_mutation_by_consumer",
            "no_sequence_policy_trust_store_mutation_by_consumer",
            "no_proposal_state_mutation_by_consumer",
            "no_direct_sequence_execution",
            "no_live_writes_by_consumer",
        ),
    ),
    ProcessingMethod(
        id="operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_trust_review_result_consumer",
        label=(
            "Operator Proposal Apply Sequence Policy Memory Trust Promotion "
            "Queue Import Promotion Memory Trust Review Result Consumer"
        ),
        domain="decision_evidence",
        status="shadow_only",
        summary=(
            "Read-only verifier that matches memory trust-review results back "
            "to promotion memory-index provenance before emitting an "
            "acceptance-ledger request template."
        ),
        module=(
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_trust_review_result_consumer.py"
        ),
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason=(
            "sequence_memory_trust_promotion_queue_import_promotion_memory_trust_review_result_consumer_is_not_acceptance_ledger_or_live_authority"
        ),
        promotion_next=(
            "use the emitted read-only acceptance-ledger request template only "
            "after this consumer verifies that trust-review candidates match "
            "promotion memory-index provenance"
        ),
        notes=(
            "transport_free",
            "promotion_memory_trust_review_result_consumer_only",
            "promotion_acceptance_ledger_request_template_only",
            "acceptance_ledger_handler_invoked_false",
            "acceptance_ledger_created_false",
            "operator_acceptance_required",
            "memory_write_allowed_by_consumer_false",
            "consumer_side_memory_mutation_false",
            "policy_mutation_false",
            "runtime_memory_mutation_false",
            "model_mutation_false",
            "idempotency_reserved_false_by_consumer",
            "promotion_memory_index_consumer_required_before_acceptance",
            "no_queue_mutation_by_consumer",
            "no_proposal_status_mutation_by_consumer",
            "no_proposal_order_mutation_by_consumer",
            "no_policy_mutation_by_consumer",
            "no_memory_mutation_by_consumer",
            "no_memory_file_mutation_by_consumer",
            "no_model_mutation",
            "no_runtime_memory_mutation_by_consumer",
            "no_sequence_policy_trust_store_mutation_by_consumer",
            "no_proposal_state_mutation_by_consumer",
            "no_direct_sequence_execution",
            "no_live_writes_by_consumer",
        ),
    ),
    ProcessingMethod(
        id="operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_trust_acceptance_ledger_result_consumer",
        label=(
            "Operator Proposal Apply Sequence Policy Memory Trust Promotion "
            "Queue Import Promotion Memory Trust Acceptance Ledger Result Consumer"
        ),
        domain="decision_evidence",
        status="shadow_only",
        summary=(
            "Read-only verifier that matches operator acceptance-ledger "
            "results back to promotion memory-trust-review provenance before "
            "emitting a promotion-plan request template."
        ),
        module=(
            "backend/operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_trust_acceptance_ledger_result_consumer.py"
        ),
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason=(
            "sequence_memory_trust_promotion_queue_import_promotion_memory_trust_acceptance_ledger_result_consumer_is_not_promotion_plan_or_live_authority"
        ),
        promotion_next=(
            "use the emitted read-only promotion-plan request template only "
            "after explicit operator accept/reject records match promotion "
            "memory-trust-review provenance"
        ),
        notes=(
            "transport_free",
            "promotion_memory_trust_acceptance_ledger_result_consumer_only",
            "promotion_plan_request_template_only",
            "promotion_plan_handler_invoked_false",
            "promotion_plan_created_false",
            "operator_acceptance_required",
            "memory_write_allowed_by_consumer_false",
            "consumer_side_memory_mutation_false",
            "policy_mutation_false",
            "runtime_memory_mutation_false",
            "model_mutation_false",
            "idempotency_reserved_false_by_consumer",
            "promotion_memory_trust_review_consumer_required_before_plan",
            "no_queue_mutation_by_consumer",
            "no_proposal_status_mutation_by_consumer",
            "no_proposal_order_mutation_by_consumer",
            "no_policy_mutation_by_consumer",
            "no_memory_mutation_by_consumer",
            "no_memory_file_mutation_by_consumer",
            "no_model_mutation",
            "no_runtime_memory_mutation_by_consumer",
            "no_sequence_policy_trust_store_mutation_by_consumer",
            "no_proposal_state_mutation_by_consumer",
            "no_direct_sequence_execution",
            "no_live_writes_by_consumer",
        ),
    ),
    ProcessingMethod(
        id="source_identity_calibration_memory_trust_review",
        label="Source Identity Calibration Memory Trust Review",
        domain="memory_learning",
        status="shadow_only",
        summary=(
            "Read-only operator-review candidates derived from indexed source "
            "identity calibration memory and fresh replay context."
        ),
        module="backend/source_identity_calibration_memory_trust_review.py",
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason="memory_trust_review_requires_operator_promotion_before_priority",
        promotion_next=(
            "operator reviews indexed memory candidates with fresh replay context "
            "before any separate downstream trust promotion can change priority"
        ),
        notes=(
            "transport_free",
            "operator_review_only",
            "memory_index_consumer",
            "fresh_replay_context_required",
            "no_queue_mutation",
            "no_model_mutation",
            "no_runtime_memory_mutation",
            "no_downstream_priority_mutation",
            "no_live_writes",
        ),
    ),
    ProcessingMethod(
        id="source_identity_calibration_memory_trust_acceptance_ledger",
        label="Source Identity Calibration Memory Trust Acceptance Ledger",
        domain="memory_learning",
        status="shadow_only",
        summary=(
            "Read-only ledger for operator accepted/rejected decisions on "
            "source identity memory trust-review candidates."
        ),
        module="backend/source_identity_calibration_memory_trust_acceptance_ledger.py",
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason="memory_trust_acceptance_is_not_promotion_authority",
        promotion_next=(
            "use accepted memory-trust records only as evidence for a separate "
            "promotion contract before downstream priority changes"
        ),
        notes=(
            "transport_free",
            "operator_acceptance_ledger_only",
            "memory_trust_review_consumer",
            "no_queue_mutation",
            "no_model_mutation",
            "no_runtime_memory_mutation",
            "no_downstream_priority_mutation",
            "no_live_writes",
        ),
    ),
    ProcessingMethod(
        id="source_identity_calibration_memory_trust_promotion_plan",
        label="Source Identity Calibration Memory Trust Promotion Plan",
        domain="memory_learning",
        status="shadow_only",
        summary=(
            "Read-only manual promotion plan derived from accepted memory-trust "
            "ledger records; does not change downstream priority."
        ),
        module="backend/source_identity_calibration_memory_trust_promotion_plan.py",
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason="memory_trust_promotion_plan_is_manual_not_authority",
        promotion_next=(
            "define a separate promotion contract with replay freshness and rollback "
            "before accepted memory trust can influence downstream priority"
        ),
        notes=(
            "transport_free",
            "manual_promotion_plan_only",
            "memory_trust_acceptance_consumer",
            "no_queue_mutation",
            "no_model_mutation",
            "no_runtime_memory_mutation",
            "no_downstream_priority_mutation",
            "no_live_writes",
        ),
    ),
    ProcessingMethod(
        id="source_identity_calibration_memory_trust_promotion_contract",
        label="Source Identity Calibration Memory Trust Promotion Contract",
        domain="memory_learning",
        status="shadow_only",
        summary=(
            "Read-only contract that validates manual memory-trust promotion "
            "candidates against fresh replay context and rollback evidence."
        ),
        module="backend/source_identity_calibration_memory_trust_promotion_contract.py",
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason="memory_trust_promotion_contract_is_review_not_executor",
        promotion_next=(
            "review contract-ready records manually; a separate executor is still "
            "required before downstream priority can change"
        ),
        notes=(
            "transport_free",
            "promotion_contract_review_only",
            "fresh_replay_context_required",
            "rollback_evidence_required",
            "no_queue_mutation",
            "no_model_mutation",
            "no_runtime_memory_mutation",
            "no_downstream_priority_mutation",
            "no_live_writes",
        ),
    ),
    ProcessingMethod(
        id="source_identity_calibration_memory_trust_promotion_acceptance_ledger",
        label="Source Identity Calibration Memory Trust Promotion Acceptance Ledger",
        domain="memory_learning",
        status="shadow_only",
        summary=(
            "Read-only ledger for operator accepted/rejected decisions on "
            "source identity memory trust promotion contract records."
        ),
        module=(
            "backend/source_identity_calibration_memory_trust_promotion_acceptance_ledger.py"
        ),
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason="memory_trust_promotion_acceptance_is_not_executor_authority",
        promotion_next=(
            "use accepted promotion contract records only as evidence for a "
            "separate executor before downstream priority changes"
        ),
        notes=(
            "transport_free",
            "operator_acceptance_ledger_only",
            "memory_trust_promotion_contract_consumer",
            "no_queue_mutation",
            "no_model_mutation",
            "no_runtime_memory_mutation",
            "no_downstream_priority_mutation",
            "no_live_writes",
        ),
    ),
    ProcessingMethod(
        id="source_identity_calibration_memory_trust_promotion_execution_plan",
        label="Source Identity Calibration Memory Trust Promotion Execution Plan",
        domain="memory_learning",
        status="shadow_only",
        summary=(
            "Read-only execution plan that surfaces accepted memory-trust "
            "promotion records as non-executing candidates."
        ),
        module=(
            "backend/source_identity_calibration_memory_trust_promotion_execution_plan.py"
        ),
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason="memory_trust_promotion_execution_plan_is_not_executor_authority",
        promotion_next=(
            "use execution candidates only as review evidence; a separate "
            "executor implementation and safety contract are required before "
            "downstream priority changes"
        ),
        notes=(
            "transport_free",
            "execution_plan_only",
            "memory_trust_promotion_acceptance_consumer",
            "no_queue_mutation",
            "no_model_mutation",
            "no_runtime_memory_mutation",
            "no_downstream_priority_mutation",
            "no_live_writes",
        ),
    ),
    ProcessingMethod(
        id="source_identity_calibration_memory_trust_promotion_executor_dry_run",
        label="Source Identity Calibration Memory Trust Promotion Executor Dry Run",
        domain="memory_learning",
        status="shadow_only",
        summary=(
            "Read-only dry-run executor contract that converts accepted "
            "memory-trust promotion execution candidates into deterministic "
            "apply specifications."
        ),
        module=(
            "backend/source_identity_calibration_memory_trust_promotion_executor_dry_run.py"
        ),
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason="memory_trust_promotion_executor_dry_run_does_not_apply",
        promotion_next=(
            "validate dry-run apply specs, then add a separate operator-gated "
            "executor contract before any downstream priority changes"
        ),
        notes=(
            "transport_free",
            "executor_dry_run_only",
            "memory_trust_promotion_execution_plan_consumer",
            "idempotent_apply_spec",
            "no_queue_mutation",
            "no_model_mutation",
            "no_runtime_memory_mutation",
            "no_downstream_priority_mutation",
            "no_live_writes",
        ),
    ),
    ProcessingMethod(
        id="source_identity_calibration_memory_trust_promotion_executor",
        label="Source Identity Calibration Memory Trust Promotion Executor",
        domain="memory_learning",
        status="shadow_only",
        summary=(
            "Operator-gated non-live executor that can apply source identity "
            "memory trust promotion to an explicit trust store only."
        ),
        module=(
            "backend/source_identity_calibration_memory_trust_promotion_executor.py"
        ),
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason="memory_trust_promotion_executor_is_not_live_console_authority",
        promotion_next=(
            "wire a reviewed persistent source-identity trust store and rollback "
            "audit before using applied trust to affect downstream proposal priority"
        ),
        notes=(
            "transport_free",
            "operator_ack_required",
            "explicit_trust_store_only",
            "memory_trust_promotion_executor_dry_run_consumer",
            "no_queue_mutation",
            "no_model_mutation",
            "no_live_writes",
            "not_product_layer_task_queue_executor",
        ),
    ),
    ProcessingMethod(
        id="source_identity_calibration_memory_trust_promotion_apply_audit",
        label="Source Identity Calibration Memory Trust Promotion Apply Audit",
        domain="memory_learning",
        status="shadow_only",
        summary=(
            "Read-only rollback audit for operator-gated source identity memory "
            "trust promotion apply results."
        ),
        module=(
            "backend/source_identity_calibration_memory_trust_promotion_apply_audit.py"
        ),
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason="memory_trust_promotion_apply_audit_is_not_priority_or_live_authority",
        promotion_next=(
            "use audit records only as rollback evidence; a separate reviewed "
            "downstream-priority gate is required before proposal policy changes"
        ),
        notes=(
            "transport_free",
            "rollback_audit_only",
            "memory_trust_promotion_executor_consumer",
            "no_queue_mutation",
            "no_model_mutation",
            "no_runtime_memory_mutation",
            "no_downstream_priority_mutation",
            "no_live_writes",
        ),
    ),
    ProcessingMethod(
        id="source_identity_calibration_memory_trust_promotion_rollback_verification",
        label="Source Identity Calibration Memory Trust Promotion Rollback Verification",
        domain="memory_learning",
        status="shadow_only",
        summary=(
            "Read-only operator rollback verification gate for source identity "
            "memory trust promotion apply-audit records."
        ),
        module=(
            "backend/source_identity_calibration_memory_trust_promotion_rollback_verification.py"
        ),
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason="memory_trust_promotion_rollback_verification_is_not_priority_or_live_authority",
        promotion_next=(
            "use verified rollback records only as evidence for a separate "
            "downstream-priority gate"
        ),
        notes=(
            "transport_free",
            "rollback_verification_only",
            "memory_trust_promotion_apply_audit_consumer",
            "operator_review_required",
            "no_queue_mutation",
            "no_model_mutation",
            "no_runtime_memory_mutation",
            "no_downstream_priority_mutation",
            "no_live_writes",
        ),
    ),
    ProcessingMethod(
        id="source_identity_calibration_memory_trust_promotion_priority_gate",
        label="Source Identity Calibration Memory Trust Promotion Priority Gate",
        domain="memory_learning",
        status="shadow_only",
        summary=(
            "Read-only downstream-priority eligibility gate for operator-verified "
            "source identity memory trust promotion rollback records."
        ),
        module=(
            "backend/source_identity_calibration_memory_trust_promotion_priority_gate.py"
        ),
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason="memory_trust_promotion_priority_gate_is_not_priority_or_live_mutation_authority",
        promotion_next=(
            "route eligible records through a separate proposal-priority policy "
            "consumer before any proposal ordering or live behavior changes"
        ),
        notes=(
            "transport_free",
            "priority_gate_only",
            "memory_trust_promotion_rollback_verification_consumer",
            "operator_review_required",
            "proposal_priority_consumer_required",
            "no_queue_mutation",
            "no_model_mutation",
            "no_runtime_memory_mutation",
            "no_direct_downstream_priority_mutation",
            "no_live_writes",
        ),
    ),
    ProcessingMethod(
        id="source_identity_proposal_priority_policy",
        label="Source Identity Proposal Priority Policy",
        domain="decision_evidence",
        status="shadow_only",
        summary=(
            "Read-only proposal-priority policy projection from trusted source "
            "identity priority-gate records."
        ),
        module="backend/source_identity_proposal_priority_policy.py",
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason="source_identity_proposal_priority_policy_is_not_ordering_queue_or_live_authority",
        promotion_next=(
            "feed ranked policy records into a separate proposal-priority "
            "consumer before mutating proposal order, queues, or live behavior"
        ),
        notes=(
            "transport_free",
            "policy_projection_only",
            "source_identity_priority_gate_consumer",
            "proposal_order_mutation_false",
            "no_queue_mutation",
            "no_model_mutation",
            "no_runtime_memory_mutation",
            "no_direct_downstream_priority_mutation",
            "no_live_writes",
        ),
    ),
    ProcessingMethod(
        id="source_identity_proposal_ordering_preview",
        label="Source Identity Proposal Ordering Preview",
        domain="decision_evidence",
        status="shadow_only",
        summary=(
            "Read-only recommended proposal-order view from source identity "
            "proposal-priority policy records."
        ),
        module="backend/source_identity_proposal_ordering_preview.py",
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason="source_identity_proposal_ordering_preview_is_not_queue_ordering_or_live_authority",
        promotion_next=(
            "route reviewed ordering-preview records through a separate queue "
            "ordering/import consumer before mutating queues or live behavior"
        ),
        notes=(
            "transport_free",
            "ordering_preview_only",
            "source_identity_proposal_priority_policy_consumer",
            "proposal_order_mutation_false",
            "no_queue_mutation",
            "no_model_mutation",
            "no_runtime_memory_mutation",
            "no_source_identity_trust_store_mutation",
            "no_direct_downstream_priority_mutation",
            "no_live_writes",
        ),
    ),
    ProcessingMethod(
        id="source_identity_proposal_queue_ordering_plan",
        label="Source Identity Proposal Queue Ordering Plan",
        domain="decision_evidence",
        status="shadow_only",
        summary=(
            "Read-only queue-ordering plan from source identity proposal "
            "ordering-preview records."
        ),
        module="backend/source_identity_proposal_queue_ordering_plan.py",
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason="source_identity_proposal_queue_ordering_plan_is_not_queue_import_or_live_authority",
        promotion_next=(
            "route reviewed queue-ordering plan records through a separate "
            "operator-acknowledged queue import consumer before mutating queues"
        ),
        notes=(
            "transport_free",
            "queue_ordering_plan_only",
            "source_identity_proposal_ordering_preview_consumer",
            "queue_import_allowed_by_plan_false",
            "proposal_order_mutation_false",
            "no_queue_mutation",
            "no_model_mutation",
            "no_runtime_memory_mutation",
            "no_source_identity_trust_store_mutation",
            "no_direct_downstream_priority_mutation",
            "no_live_writes",
        ),
    ),
    ProcessingMethod(
        id="source_identity_proposal_queue_importer",
        label="Source Identity Proposal Queue Importer",
        domain="decision_evidence",
        status="shadow_only",
        summary=(
            "Operator-gated queue importer for source identity queue-ordering "
            "plans; Product Layer uses dry-run only."
        ),
        module="backend/source_identity_proposal_queue_importer.py",
        can_analyze=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason="source_identity_proposal_queue_importer_requires_operator_ack_and_explicit_queue",
        promotion_next=(
            "wire an operator-reviewed UI action that passes apply=true, "
            "operator_ack=true, and the explicit target queue before queue mutation"
        ),
        notes=(
            "transport_free",
            "product_layer_dry_run_only",
            "operator_ack_required",
            "explicit_target_queue_required",
            "source_identity_proposal_queue_ordering_plan_consumer",
            "no_model_mutation",
            "no_runtime_memory_mutation",
            "no_source_identity_trust_store_mutation",
            "no_direct_downstream_priority_mutation",
            "no_live_writes",
        ),
    ),
    ProcessingMethod(
        id="supervised_manual_fader_gain",
        label="Supervised Manual Fader/Gain Gate",
        domain="gain_fader",
        status="canonical",
        summary="Only accepted WING live executor for MVP: approved channel fader/gain with cooldown, readback and rollback evidence.",
        module="backend/server.py + backend/handlers/mixer_handlers.py",
        can_write_live=True,
        requires_supervision=True,
        live_write_kinds=LIVE_ALLOWED_WRITE_KINDS,
        pipeline_profiles=("main_live_assist",),
        promotion_next="physical_console_proof_for_any_extra_write_kind",
    ),
    ProcessingMethod(
        id="supervised_proof_artifact_evaluator",
        label="Supervised Proof Artifact Evaluator",
        domain="evaluation",
        status="shadow_only",
        summary="Transport-free evaluator for dry-run and physical supervised fader/gain proof artifacts; distinguishes dry-run readiness from physical promotion candidates.",
        module="backend/supervised_proof_artifact_evaluator.py",
        can_analyze=True,
        pipeline_profiles=("shadow_quality", "lab_dsp"),
        blocked_reason="proof_evaluator_only_no_live_transport",
        promotion_next="feed physical fader/gain proof summaries through the evaluator before changing safety coverage rows",
        notes=("transport_free", "proof_evaluator", "no_live_writes", "not_autonomous_authority"),
    ),
    ProcessingMethod(
        id="supervised_promotion_evidence_bridge",
        label="Supervised Promotion Evidence Bridge",
        domain="evaluation",
        status="shadow_only",
        summary="Read-only bridge from proof artifact reports to manual safety-row review candidates; never updates live policy automatically.",
        module="backend/supervised_promotion_evidence.py",
        can_analyze=True,
        pipeline_profiles=("shadow_quality", "lab_dsp"),
        blocked_reason="evidence_bridge_only_manual_safety_review_required",
        promotion_next="use physical fader/gain proof reports to decide whether a manual safety coverage review is justified",
        notes=("transport_free", "proof_report_bridge", "no_live_writes", "no_automatic_safety_update"),
    ),
    ProcessingMethod(
        id="supervised_wing_proof_plan",
        label="Supervised WING Proof Plan",
        domain="evaluation",
        status="shadow_only",
        summary="Read-only operator checklist for the next bounded physical WING fader/gain proof; emits commands but never executes them.",
        module="backend/supervised_wing_proof_plan.py",
        can_analyze=True,
        pipeline_profiles=("shadow_quality", "lab_dsp"),
        blocked_reason="operator_must_execute_physical_proof_manually",
        promotion_next="run the generated physical proof plan only with explicit operator approval and then re-evaluate artifacts",
        notes=("transport_free", "proof_plan_only", "no_live_writes", "execution_allowed_by_plan_false"),
    ),
    ProcessingMethod(
        id="wing_readonly_preflight_evidence",
        label="WING Read-Only Preflight Evidence",
        domain="evaluation",
        status="shadow_only",
        summary="Read-only intake matrix for WING preflight runner artifacts; surfaces connection/readback candidates without closing readiness or granting live authority.",
        module="backend/wing_readonly_preflight_evidence.py",
        can_analyze=True,
        pipeline_profiles=("shadow_quality", "lab_dsp"),
        blocked_reason="operator_must_review_readonly_preflight_candidate_before_readiness_update",
        promotion_next="review a valid physical read-only preflight artifact before updating WING readiness rows",
        notes=(
            "transport_free",
            "preflight_evidence_only",
            "no_live_writes",
            "readiness_closure_allowed_by_evidence_false",
        ),
    ),
    ProcessingMethod(
        id="supervised_proof_remediation_plan",
        label="Supervised Proof Remediation Plan",
        domain="evaluation",
        status="shadow_only",
        summary="Read-only remediation checklist for blocked or legacy supervised proof artifacts; emits rerun/fix actions but never executes them.",
        module="backend/supervised_proof_remediation_plan.py",
        can_analyze=True,
        pipeline_profiles=("shadow_quality", "lab_dsp"),
        blocked_reason="proof_remediation_only_operator_must_execute_manually",
        promotion_next="rerun blocked legacy artifacts with current proof schema before reviewing promotion evidence",
        notes=("transport_free", "proof_remediation_only", "no_live_writes", "execution_allowed_by_plan_false"),
    ),
    ProcessingMethod(
        id="autonomy_evidence_overlay",
        label="Autonomy Evidence Overlay",
        domain="evaluation",
        status="shadow_only",
        summary="Read-only overlay that maps current supervised proof reports onto fader/gain autonomy promotion targets.",
        module="backend/autonomy_evidence_overlay.py",
        can_analyze=True,
        pipeline_profiles=("shadow_quality", "lab_dsp"),
        blocked_reason="overlay_only_manual_safety_review_required",
        promotion_next="use overlay rows to connect proof artifacts and remediation actions to promotion targets before manual safety review",
        notes=("transport_free", "proof_overlay_only", "no_live_writes", "no_automatic_safety_update"),
    ),
    ProcessingMethod(
        id="autonomy_mode_transition_gate",
        label="Autonomy Mode Transition Gate",
        domain="evaluation",
        status="shadow_only",
        summary="Read-only four-mode gate that reports available, limited, and blocked operating modes from current proof evidence.",
        module="backend/autonomy_mode_transition_gate.py",
        can_analyze=True,
        pipeline_profiles=("shadow_quality", "lab_dsp"),
        blocked_reason="mode_gate_only_no_runtime_mode_change",
        promotion_next="use gate rows to explain why Supervised remains narrow and Full Autonomous remains blocked",
        notes=("transport_free", "mode_gate_only", "no_live_writes", "no_automatic_mode_change"),
    ),
    ProcessingMethod(
        id="read_surface_remediation_plan",
        label="Read Surface Remediation Plan",
        domain="evaluation",
        status="shadow_only",
        summary="Read-only plan for closing partial and missing console read-surface coverage before autonomous promotion.",
        module="backend/read_surface_remediation_plan.py",
        can_analyze=True,
        pipeline_profiles=("shadow_quality", "lab_dsp"),
        blocked_reason="read_remediation_only_no_console_queries",
        promotion_next="complete read DTOs, coverage flags, and fixture tests before broad autonomous console control",
        notes=("transport_free", "read_remediation_only", "no_live_writes", "mutates_console_state_false"),
    ),
    ProcessingMethod(
        id="read_surface_physical_readback_capture_plan",
        label="Read Surface Physical Readback Capture Plan",
        domain="evaluation",
        status="shadow_only",
        summary=(
            "Read-only operator checklist for capturing the next physical readback "
            "artifact required by console read-surface remediation."
        ),
        module="backend/read_surface_physical_readback_capture_plan.py",
        can_analyze=True,
        pipeline_profiles=("shadow_quality", "lab_dsp"),
        blocked_reason="operator_capture_plan_only_no_console_queries_or_execution",
        promotion_next=(
            "operator captures the named physical readback artifact outside this "
            "plan, then release checks validate the artifact before closing the gap"
        ),
        notes=(
            "transport_free",
            "readback_capture_plan_only",
            "no_live_writes",
            "does_not_query_console",
            "execution_allowed_by_plan_false",
        ),
    ),
    ProcessingMethod(
        id="read_surface_fixture_evidence",
        label="Read Surface Fixture Evidence",
        domain="evaluation",
        status="shadow_only",
        summary=(
            "Read-only evidence map for read-surface DTO fixture coverage and "
            "remaining physical readback gaps."
        ),
        module="backend/read_surface_fixture_evidence.py",
        can_analyze=True,
        pipeline_profiles=("shadow_quality", "lab_dsp"),
        blocked_reason="fixture_evidence_only_no_console_queries_no_test_execution",
        promotion_next=(
            "add physical readback fixtures and dLive/WING parity mappings before "
            "claiming complete console read awareness"
        ),
        notes=(
            "transport_free",
            "fixture_evidence_only",
            "no_live_writes",
            "does_not_execute_tests",
        ),
    ),
    ProcessingMethod(
        id="raw_mixer_state_physical_readback_contract",
        label="Raw Mixer State Physical Readback Contract",
        domain="evaluation",
        status="shadow_only",
        summary=(
            "Transport-free schema contract for the raw mixer-state physical "
            "readback artifact targeted by read-surface remediation."
        ),
        module="backend/raw_mixer_state_physical_readback_contract.py",
        can_analyze=True,
        pipeline_profiles=("shadow_quality", "lab_dsp"),
        blocked_reason="contract_only_physical_readback_artifact_missing",
        promotion_next=(
            "capture an operator-run physical readback artifact matching this "
            "contract before closing the raw mixer-state fixture gap"
        ),
        notes=(
            "transport_free",
            "readback_contract_only",
            "no_live_writes",
            "does_not_query_console",
        ),
    ),
    ProcessingMethod(
        id="raw_mixer_state_physical_readback_artifact_intake",
        label="Raw Mixer State Physical Readback Artifact Intake",
        domain="evaluation",
        status="shadow_only",
        summary=(
            "Read-only intake snapshot for separately captured raw mixer-state "
            "physical readback artifacts."
        ),
        module="backend/raw_mixer_state_physical_readback_artifact_intake.py",
        can_analyze=True,
        pipeline_profiles=("shadow_quality", "lab_dsp"),
        blocked_reason="artifact_intake_only_operator_review_required",
        promotion_next=(
            "operator reviews any valid candidate before closing the raw "
            "mixer-state physical fixture gap"
        ),
        notes=(
            "transport_free",
            "artifact_intake_only",
            "no_live_writes",
            "does_not_query_console",
            "fixture_closure_allowed_by_intake_false",
        ),
    ),
    ProcessingMethod(
        id="channel_basic_controls_physical_readback_contract",
        label="Channel Basic Controls Physical Readback Contract",
        domain="evaluation",
        status="shadow_only",
        summary=(
            "Transport-free schema contract for channel identity and basic-control "
            "physical readback artifacts needed by fader, gain, mute, and pan proof."
        ),
        module="backend/channel_basic_controls_physical_readback_contract.py",
        can_analyze=True,
        pipeline_profiles=("shadow_quality", "lab_dsp"),
        blocked_reason="contract_only_channel_physical_readback_artifact_missing",
        promotion_next=(
            "capture an operator-run channel physical readback artifact matching "
            "this contract before closing the channel identity/basic controls gap"
        ),
        notes=(
            "transport_free",
            "readback_contract_only",
            "no_live_writes",
            "does_not_query_console",
        ),
    ),
    ProcessingMethod(
        id="channel_basic_controls_physical_readback_artifact_intake",
        label="Channel Basic Controls Physical Readback Artifact Intake",
        domain="evaluation",
        status="shadow_only",
        summary=(
            "Read-only intake snapshot for separately captured channel identity "
            "and basic-control physical readback artifacts."
        ),
        module="backend/channel_basic_controls_physical_readback_artifact_intake.py",
        can_analyze=True,
        pipeline_profiles=("shadow_quality", "lab_dsp"),
        blocked_reason="artifact_intake_only_channel_basic_controls_operator_review_required",
        promotion_next=(
            "operator reviews any valid channel identity/basic-control candidate "
            "before closing the physical fixture gap"
        ),
        notes=(
            "transport_free",
            "artifact_intake_only",
            "no_live_writes",
            "does_not_query_console",
            "fixture_closure_allowed_by_intake_false",
        ),
    ),
    ProcessingMethod(
        id="channel_processing_physical_readback_contract",
        label="Channel Processing Physical Readback Contract",
        domain="evaluation",
        status="shadow_only",
        summary=(
            "Transport-free schema contract for channel HPF, EQ, compressor, "
            "and gate physical readback artifacts needed by processing proof."
        ),
        module="backend/channel_processing_physical_readback_contract.py",
        can_analyze=True,
        pipeline_profiles=("shadow_quality", "lab_dsp"),
        blocked_reason="contract_only_channel_processing_physical_readback_artifact_missing",
        promotion_next=(
            "capture an operator-run channel processing physical readback "
            "artifact before closing the channel EQ/dynamics/gate gap"
        ),
        notes=(
            "transport_free",
            "readback_contract_only",
            "no_live_writes",
            "does_not_query_console",
        ),
    ),
    ProcessingMethod(
        id="channel_processing_physical_readback_artifact_intake",
        label="Channel Processing Physical Readback Artifact Intake",
        domain="evaluation",
        status="shadow_only",
        summary=(
            "Read-only intake snapshot for separately captured channel HPF, "
            "EQ, compressor, and gate physical readback artifacts."
        ),
        module="backend/channel_processing_physical_readback_artifact_intake.py",
        can_analyze=True,
        pipeline_profiles=("shadow_quality", "lab_dsp"),
        blocked_reason="artifact_intake_only_channel_processing_operator_review_required",
        promotion_next=(
            "operator reviews any valid channel processing candidate before "
            "closing the physical fixture gap"
        ),
        notes=(
            "transport_free",
            "artifact_intake_only",
            "no_live_writes",
            "does_not_query_console",
            "fixture_closure_allowed_by_intake_false",
        ),
    ),
    ProcessingMethod(
        id="bus_group_main_dca_physical_readback_contract",
        label="Bus / Main / DCA Physical Readback Contract",
        domain="evaluation",
        status="shadow_only",
        summary=(
            "Transport-free schema contract for bus, main, and DCA physical "
            "readback artifacts needed by bus/group balance and main-bus proof."
        ),
        module="backend/bus_group_main_dca_physical_readback_contract.py",
        can_analyze=True,
        pipeline_profiles=("shadow_quality", "lab_dsp"),
        blocked_reason="contract_only_bus_group_main_dca_physical_readback_artifact_missing",
        promotion_next=(
            "capture an operator-run bus/group/main/DCA physical readback "
            "artifact before closing the bus/group/main/DCA gap"
        ),
        notes=(
            "transport_free",
            "readback_contract_only",
            "no_live_writes",
            "does_not_query_console",
        ),
    ),
    ProcessingMethod(
        id="bus_group_main_dca_physical_readback_artifact_intake",
        label="Bus / Main / DCA Physical Readback Artifact Intake",
        domain="evaluation",
        status="shadow_only",
        summary=(
            "Read-only intake snapshot for separately captured bus, main, "
            "and DCA physical readback artifacts."
        ),
        module="backend/bus_group_main_dca_physical_readback_artifact_intake.py",
        can_analyze=True,
        pipeline_profiles=("shadow_quality", "lab_dsp"),
        blocked_reason="artifact_intake_only_bus_group_main_dca_operator_review_required",
        promotion_next=(
            "operator reviews any valid bus/group/main/DCA candidate before "
            "closing the physical fixture gap"
        ),
        notes=(
            "transport_free",
            "artifact_intake_only",
            "no_live_writes",
            "does_not_query_console",
            "fixture_closure_allowed_by_intake_false",
        ),
    ),
    ProcessingMethod(
        id="matrix_state_physical_readback_contract",
        label="Matrix State Physical Readback Contract",
        domain="evaluation",
        status="shadow_only",
        summary=(
            "Transport-free schema contract for matrix controls, EQ, dynamics, "
            "inserts, and metering physical readback artifacts."
        ),
        module="backend/matrix_state_physical_readback_contract.py",
        can_analyze=True,
        pipeline_profiles=("shadow_quality", "lab_dsp"),
        blocked_reason="contract_only_matrix_state_physical_readback_artifact_missing",
        promotion_next=(
            "capture an operator-run matrix physical readback artifact before "
            "closing the matrix-state physical fixture gap"
        ),
        notes=(
            "transport_free",
            "readback_contract_only",
            "no_live_writes",
            "does_not_query_console",
        ),
    ),
    ProcessingMethod(
        id="matrix_state_physical_readback_artifact_intake",
        label="Matrix State Physical Readback Artifact Intake",
        domain="evaluation",
        status="shadow_only",
        summary=(
            "Read-only intake snapshot for separately captured matrix controls, "
            "EQ, dynamics, inserts, and metering physical readback artifacts."
        ),
        module="backend/matrix_state_physical_readback_artifact_intake.py",
        can_analyze=True,
        pipeline_profiles=("shadow_quality", "lab_dsp"),
        blocked_reason="artifact_intake_only_matrix_state_operator_review_required",
        promotion_next=(
            "operator reviews any valid matrix-state candidate before closing "
            "the physical fixture gap"
        ),
        notes=(
            "transport_free",
            "artifact_intake_only",
            "no_live_writes",
            "does_not_query_console",
            "fixture_closure_allowed_by_intake_false",
        ),
    ),
    ProcessingMethod(
        id="fx_slots_inserts_physical_readback_contract",
        label="FX Slots / Inserts Physical Readback Contract",
        domain="evaluation",
        status="shadow_only",
        summary=(
            "Transport-free schema contract for FX slot, parameter, and insert "
            "assignment physical readback artifacts."
        ),
        module="backend/fx_slots_inserts_physical_readback_contract.py",
        can_analyze=True,
        pipeline_profiles=("shadow_quality", "lab_dsp"),
        blocked_reason="contract_only_fx_slots_inserts_physical_readback_artifact_missing",
        promotion_next=(
            "capture an operator-run FX slots/inserts physical readback artifact "
            "before closing the FX slots and insert assignment fixture gap"
        ),
        notes=(
            "transport_free",
            "readback_contract_only",
            "no_live_writes",
            "does_not_query_console",
        ),
    ),
    ProcessingMethod(
        id="fx_slots_inserts_physical_readback_artifact_intake",
        label="FX Slots / Inserts Physical Readback Artifact Intake",
        domain="evaluation",
        status="shadow_only",
        summary=(
            "Read-only intake snapshot for separately captured FX slot, "
            "parameter, and insert assignment physical readback artifacts."
        ),
        module="backend/fx_slots_inserts_physical_readback_artifact_intake.py",
        can_analyze=True,
        pipeline_profiles=("shadow_quality", "lab_dsp"),
        blocked_reason="artifact_intake_only_fx_slots_inserts_operator_review_required",
        promotion_next=(
            "operator reviews any valid FX slots/inserts candidate before "
            "closing the physical fixture gap"
        ),
        notes=(
            "transport_free",
            "artifact_intake_only",
            "no_live_writes",
            "does_not_query_console",
            "fixture_closure_allowed_by_intake_false",
        ),
    ),
    ProcessingMethod(
        id="routing_sends_physical_readback_contract",
        label="Routing / Sends Physical Readback Contract",
        domain="evaluation",
        status="shadow_only",
        summary=(
            "Transport-free schema contract for static routing, bus/main/matrix "
            "send, and output-route physical readback artifacts."
        ),
        module="backend/routing_sends_physical_readback_contract.py",
        can_analyze=True,
        pipeline_profiles=("shadow_quality", "lab_dsp"),
        blocked_reason="contract_only_routing_sends_physical_readback_artifact_missing",
        promotion_next=(
            "capture an operator-run routing/sends physical readback artifact "
            "before closing the routing and send fixture gap"
        ),
        notes=(
            "transport_free",
            "readback_contract_only",
            "no_live_writes",
            "does_not_query_console",
        ),
    ),
    ProcessingMethod(
        id="routing_sends_physical_readback_artifact_intake",
        label="Routing / Sends Physical Readback Artifact Intake",
        domain="evaluation",
        status="shadow_only",
        summary=(
            "Read-only intake snapshot for separately captured static route, "
            "send, and output-route physical readback artifacts."
        ),
        module="backend/routing_sends_physical_readback_artifact_intake.py",
        can_analyze=True,
        pipeline_profiles=("shadow_quality", "lab_dsp"),
        blocked_reason="artifact_intake_only_routing_sends_operator_review_required",
        promotion_next=(
            "operator reviews any valid routing/sends candidate before closing "
            "the physical fixture gap"
        ),
        notes=(
            "transport_free",
            "artifact_intake_only",
            "no_live_writes",
            "does_not_query_console",
            "fixture_closure_allowed_by_intake_false",
        ),
    ),
    ProcessingMethod(
        id="snapshots_scenes_physical_readback_contract",
        label="Snapshots / Scenes Physical Readback Contract",
        domain="evaluation",
        status="shadow_only",
        summary=(
            "Transport-free schema contract for snapshot catalog, active scene, "
            "scene diff, and rollback physical readback artifacts."
        ),
        module="backend/snapshots_scenes_physical_readback_contract.py",
        can_analyze=True,
        pipeline_profiles=("shadow_quality", "lab_dsp"),
        blocked_reason="contract_only_snapshots_scenes_physical_readback_artifact_missing",
        promotion_next=(
            "capture an operator-run snapshots/scenes physical readback artifact "
            "before closing scene content, diff, and rollback fixture gaps"
        ),
        notes=(
            "transport_free",
            "readback_contract_only",
            "no_live_writes",
            "does_not_query_console",
            "does_not_load_or_save_scenes",
        ),
    ),
    ProcessingMethod(
        id="snapshots_scenes_physical_readback_artifact_intake",
        label="Snapshots / Scenes Physical Readback Artifact Intake",
        domain="evaluation",
        status="shadow_only",
        summary=(
            "Read-only intake snapshot for separately captured scene catalog, "
            "active-scene, scene diff, and rollback physical readback artifacts."
        ),
        module="backend/snapshots_scenes_physical_readback_artifact_intake.py",
        can_analyze=True,
        pipeline_profiles=("shadow_quality", "lab_dsp"),
        blocked_reason="artifact_intake_only_snapshots_scenes_operator_review_required",
        promotion_next=(
            "operator reviews any valid snapshots/scenes candidate before "
            "closing the physical fixture gap"
        ),
        notes=(
            "transport_free",
            "artifact_intake_only",
            "no_live_writes",
            "does_not_query_console",
            "does_not_load_or_save_scenes",
            "fixture_closure_allowed_by_intake_false",
        ),
    ),
    ProcessingMethod(
        id="user_presets_physical_readback_contract",
        label="User Presets Physical Readback Contract",
        domain="evaluation",
        status="shadow_only",
        summary=(
            "Transport-free schema contract for user-preset catalog, scope, "
            "target, preset content, scoped diff, and rollback physical readback artifacts."
        ),
        module="backend/user_presets_physical_readback_contract.py",
        can_analyze=True,
        pipeline_profiles=("shadow_quality", "lab_dsp"),
        blocked_reason="contract_only_user_presets_physical_readback_artifact_missing",
        promotion_next=(
            "capture an operator-run user-preset physical readback artifact "
            "before closing preset content, scoped diff, and rollback fixture gaps"
        ),
        notes=(
            "transport_free",
            "readback_contract_only",
            "no_live_writes",
            "does_not_query_console",
            "does_not_recall_or_store_presets",
        ),
    ),
    ProcessingMethod(
        id="user_presets_physical_readback_artifact_intake",
        label="User Presets Physical Readback Artifact Intake",
        domain="evaluation",
        status="shadow_only",
        summary=(
            "Read-only intake snapshot for separately captured user-preset "
            "catalog, scope, content, scoped diff, and rollback physical readback artifacts."
        ),
        module="backend/user_presets_physical_readback_artifact_intake.py",
        can_analyze=True,
        pipeline_profiles=("shadow_quality", "lab_dsp"),
        blocked_reason="artifact_intake_only_user_presets_operator_review_required",
        promotion_next=(
            "operator reviews any valid user-preset candidate before closing "
            "the physical fixture gap"
        ),
        notes=(
            "transport_free",
            "artifact_intake_only",
            "no_live_writes",
            "does_not_query_console",
            "does_not_recall_or_store_presets",
            "fixture_closure_allowed_by_intake_false",
        ),
    ),
    ProcessingMethod(
        id="dlive_state_physical_readback_contract",
        label="dLive State Physical Readback Contract",
        domain="evaluation",
        status="shadow_only",
        summary=(
            "Transport-free schema contract for dLive cached state, MIDI/TCP "
            "state counts, and WING parity physical readback artifacts."
        ),
        module="backend/dlive_state_physical_readback_contract.py",
        can_analyze=True,
        pipeline_profiles=("shadow_quality", "lab_dsp"),
        blocked_reason="contract_only_dlive_state_physical_readback_artifact_missing",
        promotion_next=(
            "capture an operator-run dLive state readback and WING parity "
            "artifact before closing the dLive parity fixture gap"
        ),
        notes=(
            "transport_free",
            "readback_contract_only",
            "no_live_writes",
            "does_not_query_console",
            "midi_tcp_writes_blocked",
        ),
    ),
    ProcessingMethod(
        id="dlive_state_physical_readback_artifact_intake",
        label="dLive State Physical Readback Artifact Intake",
        domain="evaluation",
        status="shadow_only",
        summary=(
            "Read-only intake snapshot for separately captured dLive cached "
            "state, MIDI/TCP state counts, and WING parity physical readback artifacts."
        ),
        module="backend/dlive_state_physical_readback_artifact_intake.py",
        can_analyze=True,
        pipeline_profiles=("shadow_quality", "lab_dsp"),
        blocked_reason="artifact_intake_only_dlive_state_operator_review_required",
        promotion_next=(
            "operator reviews any valid dLive-state candidate before closing "
            "the dLive parity fixture gap"
        ),
        notes=(
            "transport_free",
            "artifact_intake_only",
            "no_live_writes",
            "does_not_query_console",
            "midi_tcp_writes_blocked",
            "dlive_writes_blocked",
            "fixture_closure_allowed_by_intake_false",
        ),
    ),
    ProcessingMethod(
        id="read_surface_physical_readback_artifact_review_queue",
        label="Read Surface Physical Readback Artifact Review Queue",
        domain="evaluation",
        status="shadow_only",
        summary=(
            "Read-only aggregate queue for operator review of all physical "
            "readback artifact-intake candidates."
        ),
        module="backend/read_surface_physical_readback_artifact_review_queue.py",
        can_analyze=True,
        pipeline_profiles=("shadow_quality", "lab_dsp"),
        blocked_reason="artifact_review_queue_only_operator_review_required",
        promotion_next=(
            "operator reviews valid artifact candidates and replaces invalid "
            "ones before any physical fixture gap can be closed"
        ),
        notes=(
            "transport_free",
            "artifact_review_queue_only",
            "no_live_writes",
            "does_not_query_console",
            "fixture_closure_allowed_by_queue_false",
        ),
    ),
    ProcessingMethod(
        id="read_surface_physical_readback_fixture_acceptance_ledger",
        label="Read Surface Physical Readback Fixture Acceptance Ledger",
        domain="evaluation",
        status="shadow_only",
        summary=(
            "Read-only ledger for operator accept/reject records tied to "
            "physical readback artifact review candidates."
        ),
        module="backend/read_surface_physical_readback_fixture_acceptance_ledger.py",
        can_analyze=True,
        pipeline_profiles=("shadow_quality", "lab_dsp"),
        blocked_reason="acceptance_ledger_only_fixture_closure_still_manual",
        promotion_next=(
            "operator records acceptance or rejection of valid candidates; "
            "fixture closure remains a separate manual decision"
        ),
        notes=(
            "transport_free",
            "operator_acceptance_ledger_only",
            "no_live_writes",
            "does_not_query_console",
            "fixture_closure_allowed_by_acceptance_false",
        ),
    ),
    ProcessingMethod(
        id="read_surface_physical_readback_fixture_closure_plan",
        label="Read Surface Physical Readback Fixture Closure Plan",
        domain="evaluation",
        status="shadow_only",
        summary=(
            "Read-only plan that turns accepted physical readback review records "
            "into manual fixture-closure candidates."
        ),
        module="backend/read_surface_physical_readback_fixture_closure_plan.py",
        can_analyze=True,
        pipeline_profiles=("shadow_quality", "lab_dsp"),
        blocked_reason="closure_plan_only_manual_fixture_update_required",
        promotion_next=(
            "operator performs a separate manual fixture baseline update before "
            "any physical fixture gap can be closed"
        ),
        notes=(
            "transport_free",
            "manual_fixture_closure_plan_only",
            "no_live_writes",
            "does_not_query_console",
            "fixture_update_allowed_by_plan_false",
            "fixture_closure_allowed_by_plan_false",
        ),
    ),
    ProcessingMethod(
        id="auto_fader_v1",
        label="Auto Fader v1",
        domain="gain_fader",
        status="proposal_only",
        summary="Fader/balance automation candidate; may inform proposals, not direct live execution.",
        module="backend/auto_fader.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("main_live_assist", "lab_dsp"),
        promotion_next="fake_mixer_proposal_adapter_tests",
    ),
    ProcessingMethod(
        id="lufs_gain_staging",
        label="Input Trim Autogain",
        domain="gain_fader",
        status="proposal_only",
        summary="Main input-trim autogain: headroom-first proposals/offline renders with local preamp character and distortion safeguards.",
        module="backend/lufs_gain_staging.py + tools/offline_input_trim_autogain_mix.py",
        can_analyze=True,
        can_propose=True,
        can_render_offline=True,
        pipeline_profiles=("main_live_assist", "offline_mix"),
    ),
    ProcessingMethod(
        id="live_shared_mix",
        label="Live Shared Mix Planner",
        domain="gain_fader",
        status="proposal_only",
        summary="Grouped balance planner adapted from offline shared-mix rules; no direct routing/name writes.",
        module="backend/live_shared_mix.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("main_live_assist", "lab_dsp"),
    ),
    ProcessingMethod(
        id="bus_group_balance_proposal",
        label="Bus / Group Balance Proposal Layer",
        domain="bus_group",
        status="proposal_only",
        summary="Transport-free stem, group, DCA and matrix balance analyzer using Ayaic planes, summed-bus evidence, and fresh current assignment/fader state without live writes.",
        module="backend/bus_group_balance_proposal.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason="bus_group_dca_matrix_live_writes_need_fresh_assignment_fader_readback_true_peak_rollback_and_hardware_proof",
        promotion_next="connect bus/group balance proposals to operator review only after stale assignment/fader cache states are blocked; keep DCA/group/matrix writes out of the live executor",
        notes=("transport_free", "proposal_only", "ayaic_bus_planes", "current_bus_group_freshness_gate", "no_live_writes"),
    ),
    ProcessingMethod(
        id="mute_management_proposal",
        label="Mute Management Proposal Layer",
        domain="mute",
        status="proposal_only",
        summary="Transport-free mute/unmute analyzer using source role, signal activity, operator intent, fresh current mute state and dropout/feedback risk without live writes.",
        module="backend/mute_management_proposal.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason="mute_live_writes_need_fresh_intent_readback_rollback_and_hardware_proof",
        promotion_next="connect mute-state proposals to operator review only after stale mute-state cache is blocked; keep mute writes out of the live executor",
        notes=("transport_free", "proposal_only", "manual_mute_authority", "current_mute_freshness_gate", "no_live_writes"),
    ),
    ProcessingMethod(
        id="routing_management_proposal",
        label="Routing Management Proposal Layer",
        domain="routing",
        status="proposal_only",
        summary="Transport-free routing analyzer that emits route-diff proposals with fresh route readback, rollback, intent and activity evidence without live writes.",
        module="backend/routing_management_proposal.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason="routing_live_writes_need_fresh_full_route_readback_rollback_scope_limits_and_hardware_proof",
        promotion_next="connect routing proposals to operator review only after stale route cache states are blocked; keep routing writes out of the live executor",
        notes=("transport_free", "proposal_only", "route_diff_evidence", "current_route_freshness_gate", "no_live_writes"),
    ),
    ProcessingMethod(
        id="scene_preset_management_proposal",
        label="Scene / Preset Management Proposal Layer",
        domain="scene_preset",
        status="proposal_only",
        summary="Transport-free scene/preset analyzer that emits scoped scene and user-preset proposals with fresh state readback, diff, rollback and intent evidence without live writes.",
        module="backend/scene_preset_management_proposal.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason="scene_preset_live_writes_need_fresh_full_state_readback_scoped_diff_rollback_scope_limits_and_hardware_proof",
        promotion_next="connect scene/preset proposals to operator review only after stale scene/preset cache states are blocked; keep scene, snapshot and preset writes out of the live executor",
        notes=("transport_free", "proposal_only", "scoped_scene_preset_diff", "scoped_diff_evidence_contract", "rollback_evidence_contract", "current_scene_preset_freshness_gate", "product_snapshot_freshness_bridge", "snapshots_scenes_snapshot_bridge", "user_presets_snapshot_bridge", "no_live_writes"),
    ),
    ProcessingMethod(
        id="mix_variant_profile_proposal",
        label="Mix Variant Profile Proposal Layer",
        domain="mix_variant",
        status="proposal_only",
        summary="Transport-free genre/task mix profile analyzer using fresh arrangement/reference context, Ayaic level planes and safety evidence without live writes.",
        module="backend/mix_variant_profile_proposal.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("main_live_assist", "offline_mix", "shadow_quality", "lab_dsp"),
        blocked_reason="mix_variant_profiles_need_fresh_reference_arrangement_operator_review_and_followup_proposals_before_any_live_or_preset_apply",
        promotion_next="route fresh-context mix-variant profiles to operator review, then fan out through proposal-only bus, FX, master and scene/preset layers",
        notes=("transport_free", "proposal_only", "ayaic_level_planes", "current_mix_variant_context_freshness_gate", "no_style_clone", "no_live_writes"),
    ),
    ProcessingMethod(
        id="performance_adaptation_proposal",
        label="Performance Adaptation Proposal Layer",
        domain="adaptation",
        status="proposal_only",
        summary="Transport-free level-drift analyzer for musician, concert-time and real-time adaptation proposals with fresh trend/readback, cooldown, headroom and feedback evidence without live writes.",
        module="backend/performance_adaptation_proposal.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason="performance_adaptation_needs_fresh_trend_readback_replay_hardware_proof_and_rate_limited_supervised_family_promotion_before_apply",
        promotion_next="compare fresh-state adaptation proposals against engineer actions in replay, then promote only one reversible fader/gain family at a time",
        notes=("transport_free", "proposal_only", "rate_limited", "current_performance_context_freshness_gate", "no_live_writes"),
    ),
    ProcessingMethod(
        id="arrangement_role_proposal",
        label="Arrangement Role Proposal Layer",
        domain="arrangement",
        status="proposal_only",
        summary="Transport-free arrangement-role analyzer using source type, section, fresh inter-channel/arrangement context, reference and accepted-history evidence without live writes.",
        module="backend/arrangement_role_proposal.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason="arrangement_role_learning_needs_fresh_context_operator_acceptance_and_replay_correlation_before_authority",
        promotion_next="attach fresh-context role proposals to operator review and replay accepted/rejected role decisions before feeding downstream mix actions",
        notes=("transport_free", "proposal_only", "current_arrangement_role_context_freshness_gate", "decision_evidence_packet", "no_live_writes"),
    ),
    ProcessingMethod(
        id="decision_evidence_packet_proposal",
        label="Decision Evidence Packet Proposal Layer",
        domain="decision_evidence",
        status="proposal_only",
        summary="Transport-free evidence-packet analyzer tying fresh audio, psychoacoustic, statistical, history, reference, inter-channel, arrangement and genre evidence to operator-review decisions without live writes.",
        module="backend/decision_evidence_packet_proposal.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason="decision_evidence_packets_need_fresh_context_and_are_review_contracts_not_live_apply_authority",
        promotion_next="attach fresh-context evidence packets to all proposal and rejected shadow decisions, then replay against accepted engineer actions before any authority promotion",
        notes=("transport_free", "proposal_only", "current_decision_evidence_context_freshness_gate", "decision_evidence_packet", "no_live_writes"),
    ),
    ProcessingMethod(
        id="long_term_memory_learning_proposal",
        label="Long-Term Memory Learning Proposal Layer",
        domain="memory_learning",
        status="proposal_only",
        summary="Transport-free learning-record analyzer using fresh verified decisions, offline replay, telemetry, operator outcomes, evidence packets and safety regressions without live writes.",
        module="backend/long_term_memory_learning_proposal.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason="memory_learning_records_require_fresh_context_operator_acceptance_and_replay_correlation_before_authority",
        promotion_next="attach fresh-context learning proposals to memory review, then compare future proposals against accepted engineer actions before any autonomous influence",
        notes=("transport_free", "proposal_only", "current_memory_learning_context_freshness_gate", "decision_memory", "no_live_writes"),
    ),
    ProcessingMethod(
        id="agentic_method_foundation",
        label="Agentic Feature / Recommendation Foundation",
        domain="evaluation",
        status="shadow_only",
        summary="Transport-free feature extraction, inter-channel context and dry-run recommendations for future gain/EQ/dynamics/pan/FX agents.",
        module="backend/agentic_method_foundation.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason="dry_run_recommendation_layer_no_live_transport",
        promotion_next="connect selected outputs through proposal adapters and supervised fader/gain gate only",
        notes=("no_osc_imports", "all_recommendations_dry_run_by_default"),
    ),
    ProcessingMethod(
        id="context_aware_gate_agent",
        label="Context-Aware Gate / Bleed Agent",
        domain="gate",
        status="proposal_only",
        summary=(
            "ACAD-inspired dry-run gate/expander recommendations using channel "
            "context, bleed correlation, spectral outliers, false-open/false-close "
            "risk, freshness, and source identity trust."
        ),
        module="backend/context_aware_gate.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason=(
            "gate_expander_live_writes_need_fresh_bleed_telemetry_source_identity_"
            "trust_readback_rollback_and_physical_proof"
        ),
        promotion_next=(
            "collect live/offline bleed replay evidence with trusted source identity; "
            "keep gate/expander changes proposal-only until separate physical proof exists"
        ),
        notes=(
            "no_live_transport",
            "dry_run_true",
            "decision_trace_ready",
            "current_gate_bleed_context_freshness_gate",
            "source_identity_downstream_trust",
        ),
    ),
    ProcessingMethod(
        id="auto_eq",
        label="Auto EQ",
        domain="eq",
        status="proposal_only",
        summary="Single/multi-channel EQ analysis may propose changes; direct apply/reset paths are fail-closed blocked evidence.",
        module="backend/auto_eq.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("lab_dsp",),
        blocked_reason="eq_live_writes_need_filter_audit_feedback_safety_readback_rollback_and_hardware_proof",
        promotion_next="proposal_adapter_blocked_eq_evidence: route blocked EQ evidence to operator review; require filter audit, feedback safety, readback and rollback proof before live apply",
        notes=("proposal_only", "direct_eq_apply_quarantined", "direct_eq_reset_quarantined", "blocked_apply_evidence", "no_live_writes"),
    ),
    ProcessingMethod(
        id="autoeq_audit_validation",
        label="AutoEQ Audit Validation",
        domain="eq",
        status="shadow_only",
        summary="Transport-free audit checks for EQ units, phase contracts, IIR stability, split leakage, export reproducibility and headroom.",
        module="backend/autoeq_audit.py",
        can_analyze=True,
        pipeline_profiles=("shadow_quality", "lab_dsp"),
        blocked_reason="audit_only_no_mixer_transport",
        promotion_next="require_passed_audit_before_any_future_eq_apply_promotion",
        notes=("no_osc_imports", "no_live_writes", "ci_gate_for_filter_packages"),
    ),
    ProcessingMethod(
        id="diff_deq_playbook_bridge",
        label="Diff-DEQ Dynamic EQ Playbook Bridge",
        domain="eq",
        status="shadow_only",
        summary="Imports external six-band PEQ+DRC dynamic-EQ predictions into dry-run playbook entries with safety validation.",
        module="backend/dynamic_eq_playbook.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("shadow_quality", "lab_dsp"),
        blocked_reason="research_import_only_no_model_runtime_no_live_transport",
        promotion_next="use_as_offline_fixture_source_before_any_operator_facing_eq_dynamics_proposal",
        notes=("no_model_weights", "no_torch_dependency", "no_live_writes", "peq_drc_predictions_shadow_only"),
    ),
    ProcessingMethod(
        id="cross_adaptive_eq",
        label="Cross-Adaptive EQ",
        domain="eq",
        status="proposal_only",
        summary=(
            "Masking-aware mirror EQ proposal layer that emits linked cut/boost "
            "candidates with freshness-gated masking context, source identity "
            "downstream trust, and no live EQ writes."
        ),
        module="backend/cross_adaptive_eq.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("lab_dsp", "offline_mix"),
        blocked_reason=(
            "eq_live_writes_need_fresh_masking_context_band_readback_sum_bus_check_"
            "source_identity_trust_and_hardware_proof"
        ),
        promotion_next=(
            "route fresh-context linked unmasking proposals to operator review "
            "with trusted source identity and rollback EQ-band evidence; keep direct "
            "EQ writes quarantined"
        ),
        notes=(
            "transport_free",
            "dry_run_true",
            "current_masking_context_freshness_gate",
            "source_identity_downstream_trust",
            "linked_mirror_eq_pair",
            "no_live_writes",
        ),
    ),
    ProcessingMethod(
        id="frequency_separation_proposal",
        label="Frequency Separation Proposal",
        domain="eq",
        status="proposal_only",
        summary=(
            "Transport-free source/stem frequency-lane analyzer that emits "
            "freshness-gated split proposals with contribution and rollback evidence."
        ),
        module="backend/frequency_separation_proposal.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("lab_dsp", "offline_mix", "main_live_assist"),
        blocked_reason=(
            "frequency_split_live_writes_need_fresh_source_stem_spectrum_readback_"
            "sum_bus_check_rollback_and_hardware_proof"
        ),
        promotion_next=(
            "route fresh-context frequency-lane proposals to operator review; "
            "keep derived EQ or fader moves quarantined until readback and rollback proof"
        ),
        notes=(
            "transport_free",
            "proposal_only",
            "source_stem_contribution",
            "current_frequency_context_freshness_gate",
            "no_live_writes",
        ),
    ),
    ProcessingMethod(
        id="auto_compressor",
        label="Auto Compressor",
        domain="dynamics",
        status="proposal_only",
        summary="Dynamics analysis may recommend compressor settings; direct compressor enable/apply paths are fail-closed blocked evidence.",
        module="backend/auto_compressor.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("lab_dsp",),
        blocked_reason="compressor_live_writes_need_fresh_dynamics_readback_makeup_headroom_rollback_and_hardware_proof",
        promotion_next="route blocked compressor evidence to operator review; require dynamics readback, makeup/headroom proof and rollback before live apply",
        notes=("proposal_only", "direct_compressor_apply_quarantined", "blocked_apply_evidence", "bounded_makeup", "no_live_writes"),
    ),
    ProcessingMethod(
        id="compressor_dynamics_proposal",
        label="Compressor Dynamics Proposal Layer",
        domain="dynamics",
        status="proposal_only",
        summary="Transport-free dynamics analyzer that emits compressor proposals without live writes and gates stale cached dynamics state.",
        module="backend/compressor_dynamics_proposal.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason="compressor_live_writes_need_dynamics_readback_rollback_and_hardware_proof",
        promotion_next="connect compressor proposals to fresh dynamics readback and operator review; keep dynamics writes out of the live executor",
        notes=("transport_free", "proposal_only", "bounded_makeup", "freshness_gated", "no_live_writes"),
    ),
    ProcessingMethod(
        id="auto_compressor_cf",
        label="Crest-Factor Compressor",
        domain="dynamics",
        status="proposal_only",
        summary="Experimental crest-factor/LUFS compressor variant; direct apply path is fail-closed blocked evidence.",
        module="backend/auto_compressor_cf.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("lab_dsp",),
        blocked_reason="cf_compressor_live_writes_need_fresh_dynamics_readback_makeup_headroom_rollback_and_hardware_proof",
        promotion_next="use CF/LUFS parameters as proposal evidence only until hardware proof and supervised dynamics apply exist",
        notes=("proposal_only", "direct_cf_compressor_apply_quarantined", "blocked_apply_evidence", "no_live_writes"),
    ),
    ProcessingMethod(
        id="deesser_vocal_sibilance",
        label="Vocal De-esser Proposal Layer",
        domain="dynamics",
        status="proposal_only",
        summary="Transport-free vocal sibilance detector that emits de-esser proposals without live writes and gates stale dynamics state.",
        module="backend/deesser_vocal_sibilance.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason="deesser_live_writes_need_dynamics_or_dynamic_eq_hardware_proof",
        promotion_next="connect proposals to fresh dynamics or dynamic-EQ readback; keep de-esser writes out of the live executor",
        notes=("transport_free", "proposal_only", "freshness_gated", "no_live_writes"),
    ),
    ProcessingMethod(
        id="auto_gate_caig",
        label="CAIG Auto Gate",
        domain="gate",
        status="proposal_only",
        summary="Gate analysis and recommendation source; direct _send_osc is fail-closed and not a live executor.",
        module="backend/auto_gate_caig.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("lab_dsp",),
        blocked_reason="gate_live_writes_need_false_decision_replay_physical_readback_and_rollback_proof",
        promotion_next="keep direct CAIG gate sends quarantined; promote only through a proposal-only contract with physical proof",
        notes=("proposal_only", "direct_gate_send_quarantined", "dry_run_default", "no_live_writes"),
    ),
    ProcessingMethod(
        id="auto_panner",
        label="Auto Panner",
        domain="pan",
        status="proposal_only",
        summary="Panning automation candidate for offline/proposal review; direct apply_to_mixer and adaptive _send_osc are fail-closed and do not mutate mixer state.",
        module="backend/auto_panner.py + backend/auto_panner_adaptive.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("lab_dsp", "offline_mix"),
        blocked_reason="pan_live_writes_need_current_pan_readback_mono_check_and_rollback_proof",
        promotion_next="keep AutoPanner, adaptive panner, and pan-separation apply calls as blocked evidence; promote only through proposal review plus physical pan readback proof",
        notes=(
            "proposal_only",
            "direct_pan_apply_quarantined",
            "adaptive_pan_send_quarantined",
            "blocked_apply_evidence",
            "no_live_writes",
        ),
    ),
    ProcessingMethod(
        id="pan_spatial_proposal",
        label="Spatial Pan Proposal Layer",
        domain="pan",
        status="proposal_only",
        summary="Transport-free spatial analyzer that emits pan proposals from role, stereo-pair, stage-position, mono-compatibility, and fresh current-pan evidence.",
        module="backend/pan_spatial_proposal.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason="pan_live_writes_need_fresh_pan_readback_mono_check_and_hardware_proof",
        promotion_next="connect pan proposals to operator review only after stale current-pan cache states are blocked; keep pan writes out of the live executor",
        notes=("transport_free", "proposal_only", "current_pan_freshness_gate", "mono_compatibility_guard", "no_live_writes"),
    ),
    ProcessingMethod(
        id="phase_alignment",
        label="Phase / Delay Alignment",
        domain="phase",
        status="proposal_only",
        summary="Phase and delay analysis can produce recommendations; direct delay/polarity apply and reset are fail-closed blocked evidence.",
        module="backend/phase_alignment.py + backend/auto_phase_gcc_phat.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("lab_dsp", "offline_mix"),
        blocked_reason="phase_delay_polarity_live_writes_need_fresh_phase_context_readback_rollback_and_hardware_proof",
        promotion_next="connect blocked phase/delay evidence to operator review only after fresh phase context plus delay/polarity readback and rollback proof exist",
        notes=("proposal_only", "phase_delay_apply_quarantined", "phase_delay_reset_quarantined", "blocked_apply_evidence", "no_live_writes"),
    ),
    ProcessingMethod(
        id="phase_aware_eq_guard",
        label="Phase-Aware EQ Guard",
        domain="phase",
        status="proposal_only",
        summary="Pi-inspired original group phase guard that blocks risky EQ boosts and emits delay/polarity proposals only from fresh phase context without live writes.",
        module="backend/phase_aware_eq_guard.py",
        can_analyze=True,
        can_propose=True,
        can_render_offline=True,
        pipeline_profiles=("shadow_quality", "lab_dsp", "offline_mix"),
        blocked_reason="phase_delay_polarity_live_writes_need_fresh_phase_context_and_separate_supervised_hardware_proof",
        promotion_next="connect only fresh-context proposal records to operator review; keep WING phase/delay writes quarantined",
        notes=("transport_free", "dry_run_true", "current_phase_context_freshness_gate", "no_pace_or_proprietary_pi_logic", "phase_before_eq_guard"),
    ),
    ProcessingMethod(
        id="auto_fx",
        label="FX / Reverb / Sends",
        domain="fx",
        status="proposal_only",
        summary="FX, reverb and send planners emit reviewable plans; direct apply_to_mixer paths are fail-closed and do not mutate mixer state.",
        module="backend/auto_fx.py + backend/auto_reverb.py + backend/auto_effects.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("lab_dsp", "offline_mix"),
        blocked_reason="fx_rack_send_live_writes_need_send_return_readback_feedback_check_and_rollback_proof",
        promotion_next="keep AutoFX and AutoReverb apply calls as blocked evidence; promote only through proposal review plus physical FX readback proof",
        notes=(
            "proposal_only",
            "direct_fx_apply_quarantined",
            "direct_reverb_apply_quarantined",
            "blocked_apply_evidence",
            "no_live_writes",
        ),
    ),
    ProcessingMethod(
        id="fx_send_proposal",
        label="FX Send Proposal Layer",
        domain="fx",
        status="proposal_only",
        summary="Transport-free FX send analyzer that emits send/return proposals from fresh current-FX state and scoped Product Layer FX snapshot schema hints without live writes.",
        module="backend/fx_send_proposal.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason="fx_send_live_writes_need_fresh_send_return_readback_schema_hints_feedback_check_and_hardware_proof",
        promotion_next="connect FX send proposals to operator review only after stale send/return cache states and unknown FX parameter schemas are blocked; keep FX writes out of the live executor",
        notes=("transport_free", "proposal_only", "current_fx_freshness_gate", "fx_snapshot_freshness_gate", "fx_parameter_schema_gate", "fx_inserts_snapshot_schema_bridge", "batch_fx_snapshot_schema_bridge", "feedback_guard", "no_live_writes"),
    ),
    ProcessingMethod(
        id="master_bus_optimization_proposal",
        label="Master Bus Optimization Proposal Layer",
        domain="master_bus",
        status="proposal_only",
        summary="Transport-free master bus analyzer that emits loudness, true-peak, tonal, and glue-compression proposals from fresh current main-bus state without live writes.",
        module="backend/master_bus_optimization_proposal.py",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("main_live_assist", "shadow_quality", "lab_dsp"),
        blocked_reason="master_bus_live_writes_need_fresh_main_bus_readback_limiter_dynamics_rollback_and_hardware_proof",
        promotion_next="connect master-bus proposals to operator review only after stale main-bus processing cache states are blocked; keep main bus writes out of the live executor",
        notes=("transport_free", "proposal_only", "current_main_bus_freshness_gate", "true_peak_guard", "no_live_writes"),
    ),
    ProcessingMethod(
        id="mix_agent_conservative",
        label="Mix Agent Conservative Renderer",
        domain="offline_render",
        status="offline_only",
        summary="Offline conservative render/report path; unsupported actions remain placeholders.",
        module="mix_agent/",
        can_analyze=True,
        can_propose=True,
        can_render_offline=True,
        requires_supervision=False,
        pipeline_profiles=("offline_mix",),
    ),
    ProcessingMethod(
        id="offline_agent_mix",
        label="Offline Agent Mix",
        domain="offline_render",
        status="offline_only",
        summary="Extended offline renderer with references, FX and reporting; no live console imports by policy.",
        module="tools/offline_agent_mix.py",
        can_analyze=True,
        can_propose=True,
        can_render_offline=True,
        requires_supervision=False,
        pipeline_profiles=("offline_mix",),
    ),
    ProcessingMethod(
        id="muq_perceptual_shadow",
        label="MuQ / Perceptual Shadow Quality",
        domain="evaluation",
        status="shadow_only",
        summary="Quality scoring and warnings may prioritize proposals but never execute console writes.",
        module="backend/evaluation/muq_eval_service.py + backend/perceptual/",
        can_analyze=True,
        pipeline_profiles=("shadow_quality",),
    ),
    ProcessingMethod(
        id="ml_processing_graph",
        label="ML Processing Graph",
        domain="evaluation",
        status="lab",
        summary="Research graph/differentiable processors; not part of live executor.",
        module="backend/ml/",
        can_analyze=True,
        can_propose=True,
        pipeline_profiles=("shadow_quality", "lab_dsp"),
    ),
    ProcessingMethod(
        id="autoeq_direct_apply",
        label="Direct AutoEQ Apply",
        domain="eq",
        status="quarantined",
        summary="Lower-level EQ write path must not enter main live executor.",
        module="backend/auto_eq.py",
        blocked_reason="direct_set_eq_surface",
        pipeline_profiles=("quarantine",),
    ),
    ProcessingMethod(
        id="compressor_direct_apply",
        label="Direct Compressor Apply",
        domain="dynamics",
        status="quarantined",
        summary="Compressor writes stay blocked on WING deployment boundary.",
        module="backend/auto_compressor.py + backend/server.py",
        blocked_reason="direct_set_compressor_surface",
        pipeline_profiles=("quarantine",),
    ),
    ProcessingMethod(
        id="routing_snapshot_reset_writes",
        label="Routing / Snapshot / Reset Writes",
        domain="evaluation",
        status="quarantined",
        summary="Routing, snapshot, reset and input-routing writes are not processing pipeline executors.",
        module="backend/wing_client.py + backend/handlers/",
        blocked_reason="destructive_or_broad_mixer_state_surface",
        pipeline_profiles=("quarantine",),
    ),
    ProcessingMethod(
        id="fx_direct_apply",
        label="Direct FX / Insert Apply",
        domain="fx",
        status="quarantined",
        summary="FX rack, insert, send and FX parameter writes are blocked from the main live executor.",
        module="backend/auto_fx.py + backend/handlers/fx_handlers.py",
        blocked_reason="direct_fx_or_insert_write_surface",
        pipeline_profiles=("quarantine",),
    ),
    ProcessingMethod(
        id="mix_agent_unsupported_actions",
        label="Mix Agent Unsupported Actions",
        domain="dynamics",
        status="placeholder",
        summary="Compression, sends, stereo width, dynamic EQ and sidechain are advisory placeholders in mix_agent.",
        module="mix_agent/actions/",
        blocked_reason="placeholder_not_applied",
        pipeline_profiles=("offline_mix", "quarantine"),
    ),
    ProcessingMethod(
        id="automixer_production_mix_v1",
        label="Production Mix v1 Package",
        domain="offline_render",
        status="placeholder",
        summary="Documented offline package is not a current executable entrypoint in this checkout.",
        module="automixer/production_mix_v1/",
        blocked_reason="empty_or_missing_entrypoint",
        pipeline_profiles=("offline_mix", "quarantine"),
    ),
    ProcessingMethod(
        id="ai_mixing_pipeline_legacy",
        label="AI Mixing Pipeline Legacy Skeleton",
        domain="offline_render",
        status="deprecated",
        summary="Older empty/legacy pipeline skeleton; not selected as a current executor.",
        module="ai_mixing_pipeline/",
        blocked_reason="legacy_empty_skeleton",
        pipeline_profiles=("quarantine",),
    ),
)


PIPELINE_PROFILES: Tuple[PipelineProfile, ...] = (
    PipelineProfile(
        id="main_live_assist",
        label="Main Live Assist",
        status="canonical",
        summary="Live-safe operator pipeline: analyze, propose, safety-check and apply only supervised channel fader/gain.",
        method_ids=(
            "product_operator_state",
            "auto_soundcheck_engine",
            "source_identity_confidence_proposal",
            "source_identity_replay_calibration",
            "source_identity_calibration_acceptance_ledger",
            "source_identity_calibration_persistence_plan",
            "source_identity_calibration_memory_draft",
            "source_identity_calibration_memory_writer",
            "source_identity_calibration_memory_index",
            "operator_proposal_apply_sequence_policy_memory_index",
            "operator_proposal_apply_sequence_policy_memory_trust_review",
            "operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_plan",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_contract",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_executor_dry_run",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_executor",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_apply_audit",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_rollback_verification",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_sequence_policy_gate",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_policy_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering_preview",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_ordering_plan",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_importer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_plan",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_sequence_plan_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_dry_run_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_gate_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_audit_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_audit_policy_gate_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_draft_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_dry_run_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_apply_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_index_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_review_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_acceptance_ledger_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_plan_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_contract_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_acceptance_ledger_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_plan_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_dry_run_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_apply_audit_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_rollback_verification_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_sequence_policy_gate_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_policy_consumer_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_ordering_preview_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_ordering_plan_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_preflight_plan_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_batch_preflight_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_sequence_plan_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_dry_run_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_gate_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_audit_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_audit_policy_gate_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_draft_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_writer_dry_run_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_writer_apply_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_index_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_trust_review_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_trust_acceptance_ledger_result_consumer",
            "source_identity_calibration_memory_trust_review",
            "source_identity_calibration_memory_trust_acceptance_ledger",
            "source_identity_calibration_memory_trust_promotion_plan",
            "source_identity_calibration_memory_trust_promotion_contract",
            "source_identity_calibration_memory_trust_promotion_acceptance_ledger",
            "source_identity_calibration_memory_trust_promotion_execution_plan",
            "source_identity_calibration_memory_trust_promotion_executor_dry_run",
            "source_identity_calibration_memory_trust_promotion_executor",
            "source_identity_calibration_memory_trust_promotion_apply_audit",
            "source_identity_calibration_memory_trust_promotion_rollback_verification",
            "source_identity_calibration_memory_trust_promotion_priority_gate",
            "source_identity_proposal_priority_policy",
            "source_identity_proposal_ordering_preview",
            "source_identity_proposal_queue_ordering_plan",
            "source_identity_proposal_queue_importer",
            "auto_fader_v1",
            "lufs_gain_staging",
            "live_shared_mix",
            "bus_group_balance_proposal",
            "mute_management_proposal",
            "routing_management_proposal",
            "scene_preset_management_proposal",
            "mix_variant_profile_proposal",
            "performance_adaptation_proposal",
            "arrangement_role_proposal",
            "decision_evidence_packet_proposal",
            "long_term_memory_learning_proposal",
            "agentic_method_foundation",
            "context_aware_gate_agent",
            "compressor_dynamics_proposal",
            "deesser_vocal_sibilance",
            "pan_spatial_proposal",
            "fx_send_proposal",
            "master_bus_optimization_proposal",
            "supervised_manual_fader_gain",
        ),
        proposal_method_ids=(
            "auto_soundcheck_engine",
            "source_identity_confidence_proposal",
            "source_identity_replay_calibration",
            "auto_fader_v1",
            "lufs_gain_staging",
            "live_shared_mix",
            "bus_group_balance_proposal",
            "mute_management_proposal",
            "routing_management_proposal",
            "scene_preset_management_proposal",
            "mix_variant_profile_proposal",
            "performance_adaptation_proposal",
            "arrangement_role_proposal",
            "decision_evidence_packet_proposal",
            "long_term_memory_learning_proposal",
            "agentic_method_foundation",
            "context_aware_gate_agent",
            "compressor_dynamics_proposal",
            "deesser_vocal_sibilance",
            "pan_spatial_proposal",
            "fx_send_proposal",
            "master_bus_optimization_proposal",
        ),
        executor_method_ids=("supervised_manual_fader_gain",),
        can_write_live=True,
        requires_supervision=True,
        live_write_kinds=LIVE_ALLOWED_WRITE_KINDS,
        mode="operator_supervised",
        notes=("not_autonomous_live_mixing", "no_eq_or_compressor_apply_in_v1"),
    ),
    PipelineProfile(
        id="offline_mix",
        label="Offline Mix",
        status="offline_only",
        summary="Offline render and comparison pipeline; produces WAV/report/JSON traces without OSC.",
        method_ids=(
            "mix_agent_conservative",
            "offline_agent_mix",
            "lufs_gain_staging",
            "cross_adaptive_eq",
            "auto_panner",
            "phase_alignment",
            "phase_aware_eq_guard",
            "auto_fx",
            "mix_variant_profile_proposal",
        ),
        blocked_method_ids=("mix_agent_unsupported_actions", "automixer_production_mix_v1"),
        can_write_live=False,
        requires_supervision=False,
        mode="offline_render",
    ),
    PipelineProfile(
        id="shadow_quality",
        label="Shadow Quality",
        status="shadow_only",
        summary="Quality/evaluation pipeline; returns scores, warnings and confidence only.",
        method_ids=(
            "agentic_method_foundation",
            "source_identity_confidence_proposal",
            "source_identity_replay_calibration",
            "source_identity_calibration_acceptance_ledger",
            "source_identity_calibration_persistence_plan",
            "source_identity_calibration_memory_draft",
            "source_identity_calibration_memory_writer",
            "source_identity_calibration_memory_index",
            "operator_proposal_apply_sequence_policy_memory_index",
            "operator_proposal_apply_sequence_policy_memory_trust_review",
            "operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_plan",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_contract",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_executor_dry_run",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_executor",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_apply_audit",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_rollback_verification",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_sequence_policy_gate",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_policy_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering_preview",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_ordering_plan",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_importer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_plan",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_sequence_plan_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_dry_run_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_gate_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_audit_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_audit_policy_gate_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_draft_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_dry_run_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_apply_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_index_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_review_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_acceptance_ledger_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_plan_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_contract_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_acceptance_ledger_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_plan_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_dry_run_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_apply_audit_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_rollback_verification_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_sequence_policy_gate_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_policy_consumer_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_ordering_preview_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_ordering_plan_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_preflight_plan_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_batch_preflight_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_sequence_plan_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_dry_run_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_gate_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_audit_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_audit_policy_gate_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_draft_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_writer_dry_run_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_writer_apply_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_index_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_trust_review_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_trust_acceptance_ledger_result_consumer",
            "source_identity_calibration_memory_trust_review",
            "source_identity_calibration_memory_trust_acceptance_ledger",
            "source_identity_calibration_memory_trust_promotion_plan",
            "source_identity_calibration_memory_trust_promotion_contract",
            "source_identity_calibration_memory_trust_promotion_acceptance_ledger",
            "source_identity_calibration_memory_trust_promotion_execution_plan",
            "source_identity_calibration_memory_trust_promotion_executor_dry_run",
            "source_identity_calibration_memory_trust_promotion_executor",
            "source_identity_calibration_memory_trust_promotion_apply_audit",
            "source_identity_calibration_memory_trust_promotion_rollback_verification",
            "source_identity_calibration_memory_trust_promotion_priority_gate",
            "source_identity_proposal_priority_policy",
            "source_identity_proposal_ordering_preview",
            "source_identity_proposal_queue_ordering_plan",
            "source_identity_proposal_queue_importer",
            "bus_group_balance_proposal",
            "mute_management_proposal",
            "routing_management_proposal",
            "scene_preset_management_proposal",
            "mix_variant_profile_proposal",
            "performance_adaptation_proposal",
            "arrangement_role_proposal",
            "decision_evidence_packet_proposal",
            "long_term_memory_learning_proposal",
            "context_aware_gate_agent",
            "supervised_proof_artifact_evaluator",
            "supervised_promotion_evidence_bridge",
            "supervised_wing_proof_plan",
            "wing_readonly_preflight_evidence",
            "supervised_proof_remediation_plan",
            "autonomy_evidence_overlay",
            "autonomy_mode_transition_gate",
            "read_surface_remediation_plan",
            "autoeq_audit_validation",
            "diff_deq_playbook_bridge",
            "compressor_dynamics_proposal",
            "deesser_vocal_sibilance",
            "pan_spatial_proposal",
            "fx_send_proposal",
            "master_bus_optimization_proposal",
            "phase_aware_eq_guard",
            "muq_perceptual_shadow",
            "ml_processing_graph",
        ),
        can_write_live=False,
        requires_supervision=False,
        mode="shadow_evaluation",
    ),
    PipelineProfile(
        id="lab_dsp",
        label="Lab DSP",
        status="lab",
        summary="DSP research/proposal pool kept outside live runtime defaults.",
        method_ids=(
            "auto_eq",
            "source_identity_confidence_proposal",
            "source_identity_replay_calibration",
            "source_identity_calibration_acceptance_ledger",
            "source_identity_calibration_persistence_plan",
            "source_identity_calibration_memory_draft",
            "source_identity_calibration_memory_writer",
            "source_identity_calibration_memory_index",
            "operator_proposal_apply_sequence_policy_memory_index",
            "operator_proposal_apply_sequence_policy_memory_trust_review",
            "operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_plan",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_contract",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_executor_dry_run",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_executor",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_apply_audit",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_rollback_verification",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_sequence_policy_gate",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_policy_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering_preview",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_ordering_plan",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_importer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_plan",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_sequence_plan_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_dry_run_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_gate_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_audit_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_audit_policy_gate_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_draft_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_dry_run_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_apply_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_index_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_review_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_acceptance_ledger_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_plan_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_contract_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_acceptance_ledger_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_plan_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_dry_run_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_apply_audit_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_rollback_verification_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_sequence_policy_gate_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_policy_consumer_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_ordering_preview_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_ordering_plan_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_preflight_plan_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_batch_preflight_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_sequence_plan_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_dry_run_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_gate_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_audit_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_audit_policy_gate_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_draft_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_writer_dry_run_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_writer_apply_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_index_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_trust_review_result_consumer",
            "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_trust_acceptance_ledger_result_consumer",
            "source_identity_calibration_memory_trust_review",
            "source_identity_calibration_memory_trust_acceptance_ledger",
            "source_identity_calibration_memory_trust_promotion_plan",
            "source_identity_calibration_memory_trust_promotion_contract",
            "source_identity_calibration_memory_trust_promotion_acceptance_ledger",
            "source_identity_calibration_memory_trust_promotion_execution_plan",
            "source_identity_calibration_memory_trust_promotion_executor_dry_run",
            "source_identity_calibration_memory_trust_promotion_executor",
            "source_identity_calibration_memory_trust_promotion_apply_audit",
            "source_identity_calibration_memory_trust_promotion_rollback_verification",
            "source_identity_calibration_memory_trust_promotion_priority_gate",
            "source_identity_proposal_priority_policy",
            "source_identity_proposal_ordering_preview",
            "source_identity_proposal_queue_ordering_plan",
            "source_identity_proposal_queue_importer",
            "cross_adaptive_eq",
            "autoeq_audit_validation",
            "diff_deq_playbook_bridge",
            "auto_compressor",
            "compressor_dynamics_proposal",
            "auto_compressor_cf",
            "deesser_vocal_sibilance",
            "context_aware_gate_agent",
            "auto_gate_caig",
            "auto_panner",
            "pan_spatial_proposal",
            "phase_alignment",
            "phase_aware_eq_guard",
            "auto_fx",
            "fx_send_proposal",
            "master_bus_optimization_proposal",
            "bus_group_balance_proposal",
            "mute_management_proposal",
            "routing_management_proposal",
            "scene_preset_management_proposal",
            "mix_variant_profile_proposal",
            "performance_adaptation_proposal",
            "arrangement_role_proposal",
            "decision_evidence_packet_proposal",
            "long_term_memory_learning_proposal",
            "agentic_method_foundation",
            "supervised_proof_artifact_evaluator",
            "supervised_promotion_evidence_bridge",
            "supervised_wing_proof_plan",
            "wing_readonly_preflight_evidence",
            "ml_processing_graph",
        ),
        can_write_live=False,
        requires_supervision=False,
        mode="proposal_or_lab_only",
    ),
    PipelineProfile(
        id="quarantine",
        label="Quarantine",
        status="quarantined",
        summary="Known direct-write, placeholder, empty or deprecated paths blocked from canonical execution.",
        method_ids=(
            "autoeq_direct_apply",
            "compressor_direct_apply",
            "routing_snapshot_reset_writes",
            "fx_direct_apply",
            "mix_agent_unsupported_actions",
            "automixer_production_mix_v1",
            "ai_mixing_pipeline_legacy",
        ),
        can_write_live=False,
        requires_supervision=True,
        mode="blocked",
        notes=("requires_adr_and_tests_before_promotion",),
    ),
)


PROMOTION_LADDER = (
    {
        "from": "lab",
        "to": "proposal_only",
        "requirements": [
            "adapter emits dry-run trace",
            "tests prove no direct mixer writes",
            "operator-facing reason and confidence are present",
        ],
    },
    {
        "from": "proposal_only",
        "to": "supervised_live_candidate",
        "requirements": [
            "fake mixer tests cover bounds and blocked cases",
            "readback and rollback schema exists",
            "cooldown and emergency-stop behavior are preserved",
        ],
    },
    {
        "from": "supervised_live_candidate",
        "to": "canonical_live",
        "requirements": [
            "physical WING proof with readback",
            "rollback succeeds",
            "cooldown and emergency stop are verified",
            "operator approval is explicit",
        ],
    },
)

MODULE_STATUS_OVERRIDES: Dict[str, Dict[str, str]] = {
    "product_operator_state": {
        "entrypoint": "get_dashboard_snapshot / get_decision_queue / get_safety_monitor",
        "ui_surface": "Control Center dashboard, Task Queue, Safety Monitor",
        "next_step": "Keep as the canonical read model and route new automation through proposals.",
    },
    "auto_soundcheck_engine": {
        "entrypoint": "run_operator_analysis / import_soundcheck_recommendations",
        "ui_surface": "Dashboard analysis and Task Queue proposals",
        "next_step": "Keep live behavior guarded; emit fader/gain proposals before any apply.",
    },
    "source_identity_confidence_proposal": {
        "entrypoint": "source_identity_confidence_proposal.analyze_source_identity_confidence / build_source_identity_confidence_proposals",
        "ui_surface": "Channel Intelligence, source identity review, downstream proposal priority evidence",
        "next_step": "Keep source identity review proposal-only; replay accepted/rejected labels before allowing downstream proposals to trust low-confidence classifications.",
    },
    "source_identity_replay_calibration": {
        "entrypoint": "source_identity_replay_calibration.analyze_source_identity_replay_calibration / build_source_identity_replay_calibration_proposals",
        "ui_surface": "Channel Intelligence, replay calibration evidence, downstream proposal priority review",
        "next_step": "Use accepted/rejected replay outcomes only as operator-reviewed calibration evidence before changing source-specific proposal priority.",
    },
    "source_identity_calibration_acceptance_ledger": {
        "entrypoint": "build_source_identity_calibration_acceptance_ledger",
        "ui_surface": "Channel Intelligence, source identity calibration acceptance ledger",
        "next_step": "Use only to record operator review decisions; do not write models, memory, downstream priority, queues, or live console state.",
    },
    "source_identity_calibration_persistence_plan": {
        "entrypoint": "build_source_identity_calibration_persistence_plan",
        "ui_surface": "Channel Intelligence, manual source identity calibration persistence plan",
        "next_step": "Use only to plan separate persistence work; do not write memory, models, downstream priority, queues, or live console state.",
    },
    "source_identity_calibration_memory_draft": {
        "entrypoint": "build_source_identity_calibration_memory_draft",
        "ui_surface": "Channel Intelligence, append-only source identity memory draft",
        "next_step": "Use only to draft append-only memory updates; do not write memory, models, downstream priority, queues, or live console state.",
    },
    "source_identity_calibration_memory_writer": {
        "entrypoint": "build_source_identity_calibration_memory_writer",
        "ui_surface": "Channel Intelligence, append-only source identity memory writer dry-run",
        "next_step": "Use preview as dry-run only; apply requires explicit operator acknowledgement and must not change runtime trust or live console state.",
    },
    "source_identity_calibration_memory_index": {
        "entrypoint": "build_source_identity_calibration_memory_index",
        "ui_surface": "Channel Intelligence, append-only source identity memory index",
        "next_step": "Use indexed memory as review evidence only; separate replay/trust promotion is required before downstream priority changes.",
    },
    "operator_proposal_apply_sequence_policy_memory_index": {
        "entrypoint": "build_operator_proposal_apply_sequence_policy_memory_index",
        "ui_surface": "Decision Queue, append-only sequence policy memory index",
        "next_step": "Use indexed sequence memory as review evidence only; separate policy/trust promotion is required before any execution influence.",
    },
    "operator_proposal_apply_sequence_policy_memory_trust_review": {
        "entrypoint": "build_operator_proposal_apply_sequence_policy_memory_trust_review",
        "ui_surface": "Decision Queue, sequence policy memory trust review",
        "next_step": "Use only as operator-review evidence with fresh sequence context; execution influence still requires a separate promotion contract.",
    },
    "operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger": {
        "entrypoint": "build_operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger",
        "ui_surface": "Decision Queue, sequence policy memory trust acceptance ledger",
        "next_step": "Record accepted/rejected sequence-memory trust decisions as evidence only; execution influence still requires a separate promotion contract.",
    },
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_plan": {
        "entrypoint": "build_operator_proposal_apply_sequence_policy_memory_trust_promotion_plan",
        "ui_surface": "Decision Queue, manual sequence policy memory trust promotion plan",
        "next_step": "Use accepted sequence-memory trust records only as manual promotion candidates; policy and execution influence still require a separate promotion contract.",
    },
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_contract": {
        "entrypoint": "build_operator_proposal_apply_sequence_policy_memory_trust_promotion_contract",
        "ui_surface": "Decision Queue, sequence policy memory trust promotion contract",
        "next_step": "Use contract records only for operator review; policy and execution influence still require a separate executor and rollback path.",
    },
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger": {
        "entrypoint": "build_operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger",
        "ui_surface": "Decision Queue, sequence policy memory trust promotion acceptance ledger",
        "next_step": "Record accepted/rejected promotion contract records as evidence only; policy and execution influence still require a separate executor plan.",
    },
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan": {
        "entrypoint": "build_operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan",
        "ui_surface": "Decision Queue, sequence policy memory trust promotion execution plan",
        "next_step": "Surface accepted promotion records only as non-executing candidates; policy and execution influence still require a separate acknowledged executor.",
    },
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_executor_dry_run": {
        "entrypoint": "build_operator_proposal_apply_sequence_policy_memory_trust_promotion_executor_dry_run",
        "ui_surface": "Decision Queue, sequence policy memory trust promotion executor dry-run",
        "next_step": "Use dry-run specs as review evidence only; sequence policy promotion still requires a separate acknowledged executor.",
    },
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_executor": {
        "entrypoint": "build_operator_proposal_apply_sequence_policy_memory_trust_promotion_executor",
        "ui_surface": "Decision Queue, sequence policy memory trust promotion executor",
        "next_step": "Apply only to an explicit policy store with operator acknowledgement; audit and rollback verification are still required before sequence policy consumption.",
    },
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_apply_audit": {
        "entrypoint": "build_operator_proposal_apply_sequence_policy_memory_trust_promotion_apply_audit",
        "ui_surface": "Decision Queue, sequence policy memory trust promotion apply audit",
        "next_step": "Use audit records as rollback evidence only; rollback verification is still required before sequence policy consumption.",
    },
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_rollback_verification": {
        "entrypoint": "build_operator_proposal_apply_sequence_policy_memory_trust_promotion_rollback_verification",
        "ui_surface": "Decision Queue, sequence policy memory trust promotion rollback verification",
        "next_step": "Use verified rollback records only as evidence for a separate sequence-policy gate before policy consumption.",
    },
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_sequence_policy_gate": {
        "entrypoint": "build_operator_proposal_apply_sequence_policy_memory_trust_promotion_sequence_policy_gate",
        "ui_surface": "Decision Queue, sequence policy memory trust promotion policy gate",
        "next_step": "Use eligible records only as evidence for a separate sequence-policy consumer before ordering, queue, or execution changes.",
    },
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_policy_consumer": {
        "entrypoint": "build_operator_proposal_apply_sequence_policy_memory_trust_promotion_policy_consumer",
        "ui_surface": "Decision Queue, sequence policy memory trust promotion policy consumer",
        "next_step": "Use consumer records only as read-only decision evidence before any separate ordering, queue, or execution consumer is reviewed.",
    },
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering_preview": {
        "entrypoint": "build_operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering_preview",
        "ui_surface": "Decision Queue, sequence policy memory trust promotion ordering preview",
        "next_step": "Use ordering-preview rows only as read-only decision evidence before any separate queue or sequence consumer is reviewed.",
    },
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_ordering_plan": {
        "entrypoint": "build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_ordering_plan",
        "ui_surface": "Decision Queue, sequence policy memory trust promotion queue-ordering plan",
        "next_step": "Use queue-ordering plan records only as read-only evidence before any separate acknowledged queue or sequence consumer is reviewed.",
    },
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_importer": {
        "entrypoint": "build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_importer",
        "ui_surface": "Decision Queue, sequence policy memory trust promotion queue importer",
        "next_step": "Preview queue imports as dry-run evidence; explicit operator acknowledgement and a target queue are required before queue mutation, and direct sequence execution still requires a separate consumer.",
    },
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_plan": {
        "entrypoint": "build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_plan",
        "ui_surface": "Decision Queue, sequence policy memory trust promotion queue import preflight plan",
        "next_step": "Use the emitted batch-preflight request template only with explicit operator approval; sequence planning, dry-run, gate, and execution remain separate steps.",
    },
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_result_consumer": {
        "entrypoint": "build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_result_consumer",
        "ui_surface": "Decision Queue, sequence policy memory trust promotion queue import preflight result consumer",
        "next_step": "Use verified sequence-plan request templates only after approved batch preflight results match the queue-import preflight plan; dry-run, gate, and executor remain separate steps.",
    },
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_sequence_plan_result_consumer": {
        "entrypoint": "build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_sequence_plan_result_consumer",
        "ui_surface": "Decision Queue, sequence policy memory trust promotion queue import sequence plan result consumer",
        "next_step": "Use verified dry-run request templates only after generic sequence plans match the sequence-policy queue-import provenance; gate and executor remain separate steps.",
    },
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_dry_run_result_consumer": {
        "entrypoint": "build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_dry_run_result_consumer",
        "ui_surface": "Decision Queue, sequence policy memory trust promotion queue import dry-run result consumer",
        "next_step": "Use verified execution-gate request templates only after generic dry-run results match the sequence-policy queue-import provenance; executor and live apply remain separate steps.",
    },
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_gate_result_consumer": {
        "entrypoint": "build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_gate_result_consumer",
        "ui_surface": "Decision Queue, sequence policy memory trust promotion queue import execution-gate result consumer",
        "next_step": "Use verified sequence-executor request templates only after generic execution-gate results match the sequence-policy queue-import provenance; execute, operator ack, and approval remain explicit.",
    },
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_result_consumer": {
        "entrypoint": "build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_result_consumer",
        "ui_surface": "Decision Queue, sequence policy memory trust promotion queue import execution result consumer",
        "next_step": "Use verified execution-audit request templates only after generic executor results match the sequence-policy queue-import provenance and delegated apply proof.",
    },
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_audit_result_consumer": {
        "entrypoint": "build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_audit_result_consumer",
        "ui_surface": "Decision Queue, sequence policy memory trust promotion queue import execution audit result consumer",
        "next_step": "Use verified audit-policy-gate request templates only after generic execution audits match the sequence-policy queue-import provenance.",
    },
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_audit_policy_gate_result_consumer": {
        "entrypoint": "build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_audit_policy_gate_result_consumer",
        "ui_surface": "Decision Queue, sequence policy memory trust promotion queue import audit-policy-gate result consumer",
        "next_step": "Use verified policy-memory draft request templates only after generic audit-policy gates match the sequence-policy queue-import provenance.",
    },
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_draft_result_consumer": {
        "entrypoint": "build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_draft_result_consumer",
        "ui_surface": "Decision Queue, sequence policy memory trust promotion queue import memory-draft result consumer",
        "next_step": "Use verified memory-writer request templates only after generic policy-memory drafts match the sequence-policy queue-import provenance.",
    },
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_dry_run_result_consumer": {
        "entrypoint": "build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_dry_run_result_consumer",
        "ui_surface": "Decision Queue, sequence policy memory trust promotion queue import memory-writer dry-run result consumer",
        "next_step": "Use verified memory-writer apply request templates only after generic memory-writer dry runs match the sequence-policy queue-import provenance.",
    },
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_apply_result_consumer": {
        "entrypoint": "build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_apply_result_consumer",
        "ui_surface": "Decision Queue, sequence policy memory trust promotion queue import memory-writer apply result consumer",
        "next_step": "Use verified memory-index request templates only after generic append-only memory writer results match the sequence-policy queue-import provenance.",
    },
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_index_result_consumer": {
        "entrypoint": "build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_index_result_consumer",
        "ui_surface": "Decision Queue, sequence policy memory trust promotion queue import memory-index result consumer",
        "next_step": "Use verified memory-trust-review request templates only after sequence memory index results and fresh review context match queue-import writer-apply provenance.",
    },
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_review_result_consumer": {
        "entrypoint": "build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_review_result_consumer",
        "ui_surface": "Decision Queue, sequence policy memory trust promotion queue import memory-trust-review result consumer",
        "next_step": "Use verified acceptance-ledger request templates only after trust-review candidates match queue-import memory-index provenance and the operator supplies accept/reject decisions.",
    },
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_acceptance_ledger_result_consumer": {
        "entrypoint": "build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_acceptance_ledger_result_consumer",
        "ui_surface": "Decision Queue, sequence policy memory trust promotion queue import memory-trust-acceptance-ledger result consumer",
        "next_step": "Use verified promotion-plan request templates only after operator-supplied accepted/rejected memory trust ledger records match queue-import trust-review provenance.",
    },
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_plan_result_consumer": {
        "entrypoint": "build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_plan_result_consumer",
        "ui_surface": "Decision Queue, sequence policy memory trust promotion queue import promotion-plan result consumer",
        "next_step": "Use verified promotion-contract request templates only after promotion-plan candidates match queue-import acceptance-ledger provenance and fresh context plus rollback evidence are supplied.",
    },
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_contract_result_consumer": {
        "entrypoint": "build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_contract_result_consumer",
        "ui_surface": "Decision Queue, sequence policy memory trust promotion queue import promotion-contract result consumer",
        "next_step": "Use verified promotion-acceptance-ledger request templates only after ready promotion-contract records match queue-import promotion-plan provenance and the operator supplies accept/reject decisions.",
    },
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_acceptance_ledger_result_consumer": {
        "entrypoint": "build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_acceptance_ledger_result_consumer",
        "ui_surface": "Decision Queue, sequence policy memory trust promotion queue import promotion-acceptance-ledger result consumer",
        "next_step": "Use verified promotion execution-plan request templates only after explicit operator acceptance matches queue-import promotion-contract provenance.",
    },
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_plan_result_consumer": {
        "entrypoint": "build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_plan_result_consumer",
        "ui_surface": "Decision Queue, sequence policy memory trust promotion queue import promotion-execution-plan result consumer",
        "next_step": "Use verified promotion executor dry-run request templates only after execution-plan candidates match accepted queue-import promotion records.",
    },
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_dry_run_result_consumer": {
        "entrypoint": "build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_dry_run_result_consumer",
        "ui_surface": "Decision Queue, sequence policy memory trust promotion queue import promotion-executor-dry-run result consumer",
        "next_step": "Use verified promotion executor request templates only after dry-run specs match queue-import promotion execution-plan provenance.",
    },
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_result_consumer": {
        "entrypoint": "build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_result_consumer",
        "ui_surface": "Decision Queue, sequence policy memory trust promotion queue import promotion-executor result consumer",
        "next_step": "Use verified promotion apply-audit request templates only after applied executor results match queue-import executor dry-run provenance.",
    },
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_apply_audit_result_consumer": {
        "entrypoint": "build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_apply_audit_result_consumer",
        "ui_surface": "Decision Queue, sequence policy memory trust promotion queue import promotion-apply-audit result consumer",
        "next_step": "Use verified promotion rollback-verification request templates only after apply-audit records match queue-import executor-result provenance and operator verification records are supplied.",
    },
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_rollback_verification_result_consumer": {
        "entrypoint": "build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_rollback_verification_result_consumer",
        "ui_surface": "Decision Queue, sequence policy memory trust promotion queue import promotion-rollback-verification result consumer",
        "next_step": "Use verified promotion sequence-policy gate request templates only after rollback-verification records match queue-import apply-audit provenance.",
    },
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_sequence_policy_gate_result_consumer": {
        "entrypoint": "build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_sequence_policy_gate_result_consumer",
        "ui_surface": "Decision Queue, sequence policy memory trust promotion queue import promotion-sequence-policy gate result consumer",
        "next_step": "Use verified promotion policy-consumer request templates only after sequence-policy gate records match queue-import rollback-verification provenance and operator proposals are supplied.",
    },
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_policy_consumer_result_consumer": {
        "entrypoint": "build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_policy_consumer_result_consumer",
        "ui_surface": "Decision Queue, sequence policy memory trust promotion queue import promotion-policy-consumer result consumer",
        "next_step": "Use verified promotion ordering-preview request templates only after policy-consumer records match queue-import sequence-policy gate provenance and operator proposals are supplied.",
    },
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_ordering_preview_result_consumer": {
        "entrypoint": "build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_ordering_preview_result_consumer",
        "ui_surface": "Decision Queue, sequence policy memory trust promotion queue import promotion-ordering-preview result consumer",
        "next_step": "Use verified promotion queue-ordering-plan request templates only after ordering-preview rows match queue-import policy-consumer provenance.",
    },
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_ordering_plan_result_consumer": {
        "entrypoint": "build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_ordering_plan_result_consumer",
        "ui_surface": "Decision Queue, sequence policy memory trust promotion queue import promotion-queue-ordering-plan result consumer",
        "next_step": "Use verified promotion queue-importer request templates only after queue-ordering-plan records match queue-import ordering-preview provenance and operator proposal payloads are supplied.",
    },
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer": {
        "entrypoint": "build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer",
        "ui_surface": "Decision Queue, sequence policy memory trust promotion queue import promotion-queue-importer result consumer",
        "next_step": "Use verified promotion queue-import preflight-plan request templates only after applied queue-importer results match queue-import queue-ordering-plan provenance.",
    },
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_preflight_plan_result_consumer": {
        "entrypoint": "build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_preflight_plan_result_consumer",
        "ui_surface": "Decision Queue, sequence policy memory trust promotion queue import promotion-preflight-plan result consumer",
        "next_step": "Use verified promotion batch-preflight request templates only after queue-import preflight plans match applied queue-importer provenance and operator approval is supplied separately.",
    },
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_batch_preflight_result_consumer": {
        "entrypoint": "build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_batch_preflight_result_consumer",
        "ui_surface": "Decision Queue, sequence policy memory trust promotion queue import promotion-batch-preflight result consumer",
        "next_step": "Use verified promotion sequence-plan request templates only after approved batch-preflight results match promotion preflight-plan provenance.",
    },
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_sequence_plan_result_consumer": {
        "entrypoint": "build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_sequence_plan_result_consumer",
        "ui_surface": "Decision Queue, sequence policy memory trust promotion queue import promotion-sequence-plan result consumer",
        "next_step": "Use verified promotion dry-run request templates only after generic sequence plans match promotion batch-preflight provenance.",
    },
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_dry_run_result_consumer": {
        "entrypoint": "build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_dry_run_result_consumer",
        "ui_surface": "Decision Queue, sequence policy memory trust promotion queue import promotion-dry-run result consumer",
        "next_step": "Use verified promotion execution-gate request templates only after generic dry-run results match promotion sequence-plan provenance and operator acknowledgement is supplied separately.",
    },
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_gate_result_consumer": {
        "entrypoint": "build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_gate_result_consumer",
        "ui_surface": "Decision Queue, sequence policy memory trust promotion queue import promotion-execution-gate result consumer",
        "next_step": "Use verified promotion executor request templates only after generic execution-gate results match promotion dry-run provenance and execute, operator acknowledgement, and approval are supplied separately.",
    },
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_result_consumer": {
        "entrypoint": "build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_result_consumer",
        "ui_surface": "Decision Queue, sequence policy memory trust promotion queue import promotion-execution result consumer",
        "next_step": "Use verified promotion execution-audit request templates only after generic sequence-executor results match promotion execution-gate provenance.",
    },
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_audit_result_consumer": {
        "entrypoint": "build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_audit_result_consumer",
        "ui_surface": "Decision Queue, sequence policy memory trust promotion queue import promotion-execution-audit result consumer",
        "next_step": "Use verified promotion audit-policy-gate request templates only after generic execution audits match promotion execution-result provenance.",
    },
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_audit_policy_gate_result_consumer": {
        "entrypoint": "build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_audit_policy_gate_result_consumer",
        "ui_surface": "Decision Queue, sequence policy memory trust promotion queue import promotion-audit-policy-gate result consumer",
        "next_step": "Use verified promotion memory-draft request templates only after generic audit-policy gates match promotion execution-audit provenance.",
    },
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_draft_result_consumer": {
        "entrypoint": "build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_draft_result_consumer",
        "ui_surface": "Decision Queue, sequence policy memory trust promotion queue import promotion-memory-draft result consumer",
        "next_step": "Use verified promotion memory-writer request templates only after generic policy-memory drafts match promotion audit-policy-gate provenance.",
    },
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_writer_dry_run_result_consumer": {
        "entrypoint": "build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_writer_dry_run_result_consumer",
        "ui_surface": "Decision Queue, sequence policy memory trust promotion queue import promotion-memory-writer dry-run result consumer",
        "next_step": "Use verified promotion memory-writer apply request templates only after generic memory-writer dry runs match promotion memory-draft provenance.",
    },
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_writer_apply_result_consumer": {
        "entrypoint": "build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_writer_apply_result_consumer",
        "ui_surface": "Decision Queue, sequence policy memory trust promotion queue import promotion-memory-writer apply result consumer",
        "next_step": "Use verified promotion memory-index request templates only after generic append-only memory writer results match promotion memory-writer dry-run provenance.",
    },
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_index_result_consumer": {
        "entrypoint": "build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_index_result_consumer",
        "ui_surface": "Decision Queue, sequence policy memory trust promotion queue import promotion-memory-index result consumer",
        "next_step": "Use verified promotion memory-trust-review request templates only after read-only memory index results match promotion writer-apply provenance and fresh review context.",
    },
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_trust_review_result_consumer": {
        "entrypoint": "build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_trust_review_result_consumer",
        "ui_surface": "Decision Queue, sequence policy memory trust promotion queue import promotion-memory-trust-review result consumer",
        "next_step": "Use verified promotion acceptance-ledger request templates only after memory trust-review candidates match promotion memory-index provenance.",
    },
    "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_trust_acceptance_ledger_result_consumer": {
        "entrypoint": "build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_trust_acceptance_ledger_result_consumer",
        "ui_surface": "Decision Queue, sequence policy memory trust promotion queue import promotion-memory-trust-acceptance-ledger result consumer",
        "next_step": "Use verified promotion-plan request templates only after operator acceptance-ledger records match promotion memory-trust-review provenance.",
    },
    "source_identity_calibration_memory_trust_review": {
        "entrypoint": "build_source_identity_calibration_memory_trust_review",
        "ui_surface": "Channel Intelligence, source identity memory trust review",
        "next_step": "Use only as operator-review evidence with fresh replay context; downstream priority changes still require a separate promotion contract.",
    },
    "source_identity_calibration_memory_trust_acceptance_ledger": {
        "entrypoint": "build_source_identity_calibration_memory_trust_acceptance_ledger",
        "ui_surface": "Channel Intelligence, source identity memory trust acceptance ledger",
        "next_step": "Record operator accepted/rejected memory-trust decisions as evidence only; downstream priority changes still require a separate promotion contract.",
    },
    "source_identity_calibration_memory_trust_promotion_plan": {
        "entrypoint": "build_source_identity_calibration_memory_trust_promotion_plan",
        "ui_surface": "Channel Intelligence, manual source identity memory trust promotion plan",
        "next_step": "Use accepted memory-trust records only as manual promotion candidates; downstream priority changes still require a separate promotion contract.",
    },
    "source_identity_calibration_memory_trust_promotion_contract": {
        "entrypoint": (
            "build_source_identity_calibration_memory_trust_promotion_contract"
        ),
        "ui_surface": "Channel Intelligence, source identity memory trust promotion contract",
        "next_step": "Use contract-ready records only for operator review; downstream priority changes still require a separate executor and rollback path.",
    },
    "source_identity_calibration_memory_trust_promotion_acceptance_ledger": {
        "entrypoint": (
            "build_source_identity_calibration_memory_trust_promotion_acceptance_ledger"
        ),
        "ui_surface": "Channel Intelligence, source identity memory trust promotion acceptance ledger",
        "next_step": "Record accepted/rejected promotion contract records as evidence only; downstream priority changes still require a separate executor.",
    },
    "source_identity_calibration_memory_trust_promotion_execution_plan": {
        "entrypoint": (
            "build_source_identity_calibration_memory_trust_promotion_execution_plan"
        ),
        "ui_surface": "Channel Intelligence, source identity memory trust promotion execution plan",
        "next_step": "Use execution candidates as review evidence only; downstream priority changes still require a separate executor implementation and safety contract.",
    },
    "source_identity_calibration_memory_trust_promotion_executor_dry_run": {
        "entrypoint": (
            "build_source_identity_calibration_memory_trust_promotion_executor_dry_run"
        ),
        "ui_surface": "Channel Intelligence, source identity memory trust promotion executor dry-run",
        "next_step": "Use dry-run apply specs as executor evidence only; applying trust promotion still requires a separate operator-gated executor contract.",
    },
    "source_identity_calibration_memory_trust_promotion_executor": {
        "entrypoint": (
            "build_source_identity_calibration_memory_trust_promotion_executor"
        ),
        "ui_surface": "Channel Intelligence, source identity memory trust promotion executor",
        "next_step": "Use only with explicit operator acknowledgement and an explicit trust store; Product Layer preview remains non-mutating.",
    },
    "source_identity_calibration_memory_trust_promotion_apply_audit": {
        "entrypoint": (
            "build_source_identity_calibration_memory_trust_promotion_apply_audit"
        ),
        "ui_surface": "Channel Intelligence, source identity memory trust promotion apply audit",
        "next_step": "Use rollback audit records as review evidence only; downstream priority still requires a separate reviewed gate.",
    },
    "source_identity_calibration_memory_trust_promotion_rollback_verification": {
        "entrypoint": (
            "build_source_identity_calibration_memory_trust_promotion_rollback_verification"
        ),
        "ui_surface": "Channel Intelligence, source identity memory trust promotion rollback verification",
        "next_step": "Use verified rollback records as review evidence only; downstream priority still requires a separate reviewed gate.",
    },
    "source_identity_calibration_memory_trust_promotion_priority_gate": {
        "entrypoint": (
            "build_source_identity_calibration_memory_trust_promotion_priority_gate"
        ),
        "ui_surface": "Channel Intelligence, source identity memory trust promotion priority gate",
        "next_step": "Use eligible records only as proposal-priority policy evidence; direct priority, queue, model, memory, and live mutations remain blocked.",
    },
    "source_identity_proposal_priority_policy": {
        "entrypoint": "build_source_identity_proposal_priority_policy",
        "ui_surface": "Channel Intelligence, source identity proposal priority policy projection",
        "next_step": "Use ranked records as read-only policy evidence only; proposal order, queue import, and live authority still require a separate consumer.",
    },
    "source_identity_proposal_ordering_preview": {
        "entrypoint": "build_source_identity_proposal_ordering_preview",
        "ui_surface": "Channel Intelligence, source identity proposal ordering preview",
        "next_step": "Use the recommended order view as read-only operator evidence only; queue import and live authority still require a separate consumer.",
    },
    "source_identity_proposal_queue_ordering_plan": {
        "entrypoint": "build_source_identity_proposal_queue_ordering_plan",
        "ui_surface": "Channel Intelligence, source identity proposal queue ordering plan",
        "next_step": "Use queue-ordering plan records as read-only import candidates only; an acknowledged queue import consumer is still required before queue mutation.",
    },
    "source_identity_proposal_queue_importer": {
        "entrypoint": "build_source_identity_proposal_queue_importer",
        "ui_surface": "Channel Intelligence, source identity proposal queue importer dry run",
        "next_step": "Use Product Layer dry-run specs for operator review; queue mutation requires apply=true, operator acknowledgement, and an explicit target queue.",
    },
    "supervised_manual_fader_gain": {
        "entrypoint": "apply_operator_proposal -> supervised manual gate",
        "ui_surface": "Task Queue supervised apply",
        "next_step": "Complete separate physical gain proof before widening readiness claims.",
    },
    "supervised_proof_artifact_evaluator": {
        "entrypoint": "evaluate_supervised_proof_artifact / tools/evaluate_supervised_proof_artifacts.py",
        "ui_surface": "Processing registry, safety reports, proof artifact review",
        "next_step": "Evaluate dry-run and physical proof artifacts before changing safety coverage or promotion readiness rows.",
    },
    "supervised_promotion_evidence_bridge": {
        "entrypoint": "build_supervised_promotion_evidence_snapshot",
        "ui_surface": "Safety reports, proof artifact report, promotion readiness review",
        "next_step": "Use report evidence to justify manual safety-row review; never auto-promote live authority.",
    },
    "supervised_wing_proof_plan": {
        "entrypoint": "build_supervised_wing_proof_plan",
        "ui_surface": "Proof artifact report, next WING test runbook",
        "next_step": "Use only as an operator checklist; do not execute physical proof without explicit approval.",
    },
    "wing_readonly_preflight_evidence": {
        "entrypoint": "build_wing_readonly_preflight_evidence",
        "ui_surface": "Processing registry, WING read-only preflight readiness matrix",
        "next_step": "Use only to surface operator-captured read-only preflight candidates; manual review still gates readiness updates.",
    },
    "supervised_proof_remediation_plan": {
        "entrypoint": "build_supervised_proof_remediation_plan",
        "ui_surface": "Proof artifact report, blocked proof remediation checklist",
        "next_step": "Use only to identify rerun/fix actions for blocked proof artifacts; do not execute physical reruns without explicit approval.",
    },
    "autonomy_evidence_overlay": {
        "entrypoint": "build_autonomy_evidence_overlay",
        "ui_surface": "Proof artifact report, autonomy promotion evidence review",
        "next_step": "Use only to connect proof artifacts to manual safety review; do not auto-promote any live authority.",
    },
    "autonomy_mode_transition_gate": {
        "entrypoint": "build_autonomy_mode_transition_gate",
        "ui_surface": "Proof artifact report, operating mode readiness gate",
        "next_step": "Use only to explain mode availability and blockers; do not switch runtime mode automatically.",
    },
    "read_surface_remediation_plan": {
        "entrypoint": "build_read_surface_remediation_plan",
        "ui_surface": "Processing registry, console read coverage remediation",
        "next_step": "Use only to plan read DTO and fixture work; do not query or mutate a live console from this plan.",
    },
    "read_surface_physical_readback_capture_plan": {
        "entrypoint": "build_read_surface_physical_readback_capture_plan",
        "ui_surface": "Processing registry, operator readback capture checklist",
        "next_step": "Use only as a non-executing checklist; do not capture or query hardware without explicit operator action.",
    },
    "raw_mixer_state_physical_readback_contract": {
        "entrypoint": (
            "build_raw_mixer_state_physical_readback_contract / "
            "validate_raw_mixer_state_physical_readback_artifact"
        ),
        "ui_surface": "Processing registry, read-surface remediation artifact contract",
        "next_step": "Use only to validate a separately captured physical readback artifact; do not infer live authority.",
    },
    "raw_mixer_state_physical_readback_artifact_intake": {
        "entrypoint": "build_raw_mixer_state_physical_readback_artifact_intake",
        "ui_surface": "Processing registry, raw mixer-state physical artifact intake",
        "next_step": "Use only to surface operator-captured artifact candidates; manual review still gates fixture closure.",
    },
    "channel_basic_controls_physical_readback_contract": {
        "entrypoint": (
            "build_channel_basic_controls_physical_readback_contract / "
            "validate_channel_basic_controls_physical_readback_artifact"
        ),
        "ui_surface": "Processing registry, read-surface remediation artifact contract",
        "next_step": "Use only to validate separately captured channel identity/control readback; do not infer live authority.",
    },
    "channel_basic_controls_physical_readback_artifact_intake": {
        "entrypoint": "build_channel_basic_controls_physical_readback_artifact_intake",
        "ui_surface": "Processing registry, channel basic-controls physical artifact intake",
        "next_step": "Use only to surface operator-captured channel identity/control candidates; manual review still gates fixture closure.",
    },
    "channel_processing_physical_readback_contract": {
        "entrypoint": (
            "build_channel_processing_physical_readback_contract / "
            "validate_channel_processing_physical_readback_artifact"
        ),
        "ui_surface": "Processing registry, read-surface remediation artifact contract",
        "next_step": "Use only to validate separately captured HPF/EQ/dynamics/gate readback; do not infer live authority.",
    },
    "channel_processing_physical_readback_artifact_intake": {
        "entrypoint": "build_channel_processing_physical_readback_artifact_intake",
        "ui_surface": "Processing registry, channel processing physical artifact intake",
        "next_step": "Use only to surface operator-captured HPF/EQ/dynamics/gate candidates; manual review still gates fixture closure.",
    },
    "bus_group_main_dca_physical_readback_contract": {
        "entrypoint": (
            "build_bus_group_main_dca_physical_readback_contract / "
            "validate_bus_group_main_dca_physical_readback_artifact"
        ),
        "ui_surface": "Processing registry, read-surface remediation artifact contract",
        "next_step": "Use only to validate separately captured bus/group/main/DCA readback; do not infer live authority.",
    },
    "bus_group_main_dca_physical_readback_artifact_intake": {
        "entrypoint": "build_bus_group_main_dca_physical_readback_artifact_intake",
        "ui_surface": "Processing registry, bus/group/main/DCA physical artifact intake",
        "next_step": "Use only to surface operator-captured bus/group/main/DCA candidates; manual review still gates fixture closure.",
    },
    "matrix_state_physical_readback_contract": {
        "entrypoint": (
            "build_matrix_state_physical_readback_contract / "
            "validate_matrix_state_physical_readback_artifact"
        ),
        "ui_surface": "Processing registry, read-surface remediation artifact contract",
        "next_step": "Use only to validate separately captured matrix-state readback; do not infer live authority.",
    },
    "matrix_state_physical_readback_artifact_intake": {
        "entrypoint": "build_matrix_state_physical_readback_artifact_intake",
        "ui_surface": "Processing registry, matrix-state physical artifact intake",
        "next_step": "Use only to surface operator-captured matrix candidates; manual review still gates fixture closure.",
    },
    "fx_slots_inserts_physical_readback_contract": {
        "entrypoint": (
            "build_fx_slots_inserts_physical_readback_contract / "
            "validate_fx_slots_inserts_physical_readback_artifact"
        ),
        "ui_surface": "Processing registry, read-surface remediation artifact contract",
        "next_step": "Use only to validate separately captured FX slots/inserts readback; do not infer live authority.",
    },
    "fx_slots_inserts_physical_readback_artifact_intake": {
        "entrypoint": "build_fx_slots_inserts_physical_readback_artifact_intake",
        "ui_surface": "Processing registry, FX slots/inserts physical artifact intake",
        "next_step": "Use only to surface operator-captured FX slots/inserts candidates; manual review still gates fixture closure.",
    },
    "routing_sends_physical_readback_contract": {
        "entrypoint": (
            "build_routing_sends_physical_readback_contract / "
            "validate_routing_sends_physical_readback_artifact"
        ),
        "ui_surface": "Processing registry, read-surface remediation artifact contract",
        "next_step": "Use only to validate separately captured routing/sends readback; do not infer live authority.",
    },
    "routing_sends_physical_readback_artifact_intake": {
        "entrypoint": "build_routing_sends_physical_readback_artifact_intake",
        "ui_surface": "Processing registry, routing/sends physical artifact intake",
        "next_step": "Use only to surface operator-captured routing/sends candidates; manual review still gates fixture closure.",
    },
    "snapshots_scenes_physical_readback_contract": {
        "entrypoint": (
            "build_snapshots_scenes_physical_readback_contract / "
            "validate_snapshots_scenes_physical_readback_artifact"
        ),
        "ui_surface": "Processing registry, read-surface remediation artifact contract",
        "next_step": "Use only to validate separately captured scene/snapshot readback; do not infer live authority.",
    },
    "snapshots_scenes_physical_readback_artifact_intake": {
        "entrypoint": "build_snapshots_scenes_physical_readback_artifact_intake",
        "ui_surface": "Processing registry, snapshots/scenes physical artifact intake",
        "next_step": "Use only to surface operator-captured snapshots/scenes candidates; manual review still gates fixture closure.",
    },
    "user_presets_physical_readback_contract": {
        "entrypoint": (
            "build_user_presets_physical_readback_contract / "
            "validate_user_presets_physical_readback_artifact"
        ),
        "ui_surface": "Processing registry, read-surface remediation artifact contract",
        "next_step": "Use only to validate separately captured user-preset readback; do not infer live authority.",
    },
    "user_presets_physical_readback_artifact_intake": {
        "entrypoint": "build_user_presets_physical_readback_artifact_intake",
        "ui_surface": "Processing registry, user-presets physical artifact intake",
        "next_step": "Use only to surface operator-captured user-preset candidates; manual review still gates fixture closure.",
    },
    "dlive_state_physical_readback_contract": {
        "entrypoint": (
            "build_dlive_state_physical_readback_contract / "
            "validate_dlive_state_physical_readback_artifact"
        ),
        "ui_surface": "Processing registry, read-surface remediation artifact contract",
        "next_step": "Use only to validate separately captured dLive state/parity readback; do not infer live authority.",
    },
    "dlive_state_physical_readback_artifact_intake": {
        "entrypoint": "build_dlive_state_physical_readback_artifact_intake",
        "ui_surface": "Processing registry, dLive-state physical artifact intake",
        "next_step": "Use only to surface operator-captured dLive-state candidates; manual review still gates fixture closure.",
    },
    "read_surface_physical_readback_artifact_review_queue": {
        "entrypoint": "build_read_surface_physical_readback_artifact_review_queue",
        "ui_surface": "Processing registry, physical artifact review queue",
        "next_step": "Use only to aggregate artifact-intake review work; do not close fixtures or grant live authority.",
    },
    "read_surface_physical_readback_fixture_acceptance_ledger": {
        "entrypoint": "build_read_surface_physical_readback_fixture_acceptance_ledger",
        "ui_surface": "Processing registry, physical artifact acceptance ledger",
        "next_step": "Use only to record operator review decisions; do not close fixtures or grant live authority.",
    },
    "read_surface_physical_readback_fixture_closure_plan": {
        "entrypoint": "build_read_surface_physical_readback_fixture_closure_plan",
        "ui_surface": "Processing registry, manual physical fixture closure plan",
        "next_step": "Use only to plan separate manual fixture baseline updates; do not update fixtures or grant live authority.",
    },
    "auto_fader_v1": {
        "entrypoint": "main_pipeline_preview proposal adapter",
        "ui_surface": "Task Queue preview/proposal rows",
        "next_step": "Expand proposal evidence without adding direct fader writes.",
    },
    "lufs_gain_staging": {
        "entrypoint": "safe_gain_suggestions / input-trim proposal adapter / tools/offline_input_trim_autogain_mix.py",
        "ui_surface": "Gain staging, Task Queue proposals, offline stem render",
        "next_step": "Keep live trim moves proposal-first with rollback/readback metadata.",
    },
    "live_shared_mix": {
        "entrypoint": "main_pipeline_preview proposal adapter",
        "ui_surface": "Task Queue linked fader proposals",
        "next_step": "Keep grouped balance as linked fader/gain proposals only.",
    },
    "bus_group_balance_proposal": {
        "entrypoint": "bus_group_balance_proposal.analyze_bus_group_balance_need / build_bus_group_balance_proposals",
        "ui_surface": "Group/DCA/matrix balance evidence, future Bus Balance panel",
        "next_step": "Keep bus/group balance proposal-only until fresh assignment/fader readback, true-peak proof, and rollback are proven.",
    },
    "mute_management_proposal": {
        "entrypoint": "mute_management_proposal.analyze_mute_management_need / build_mute_management_proposals",
        "ui_surface": "Channel Intelligence, Safety Monitor, future Mute panel",
        "next_step": "Keep mute-state output proposal-only until fresh intent/readback, rollback, and hardware proof are complete.",
    },
    "routing_management_proposal": {
        "entrypoint": "routing_management_proposal.analyze_routing_management_need / build_routing_management_proposals",
        "ui_surface": "Routing evidence, Safety Monitor, future Routing panel",
        "next_step": "Keep routing output proposal-only until fresh full route readback, rollback, scope limits, and hardware proof are complete.",
    },
    "scene_preset_management_proposal": {
        "entrypoint": "scene_preset_management_proposal.analyze_scene_preset_management_need / build_scene_preset_management_proposals",
        "ui_surface": "Scene/preset evidence, Safety Monitor, future Scene panel",
        "next_step": "Keep scene/preset output proposal-only; use Product Layer snapshot freshness, read-only scoped diff evidence, and read-only rollback evidence until hardware proof is complete.",
    },
    "mix_variant_profile_proposal": {
        "entrypoint": "mix_variant_profile_proposal.analyze_mix_variant_profile_need / build_mix_variant_profile_proposals",
        "ui_surface": "Mix variant evidence, reference comparison, future Variant panel",
        "next_step": "Keep mix variant output proposal-only until fresh references, arrangement context, and follow-up proposal fan-out are proven.",
    },
    "performance_adaptation_proposal": {
        "entrypoint": "performance_adaptation_proposal.analyze_performance_adaptation_need / build_performance_adaptation_proposals",
        "ui_surface": "Live adaptation evidence, Safety Monitor, future Ride panel",
        "next_step": "Keep performance adaptation output proposal-only until fresh trend/readback, replay match, cooldown, rollback, and hardware proof are complete.",
    },
    "arrangement_role_proposal": {
        "entrypoint": "arrangement_role_proposal.analyze_arrangement_role_need / build_arrangement_role_proposals",
        "ui_surface": "Arrangement role evidence, Channel Intelligence, future Role panel",
        "next_step": "Keep arrangement role output proposal-only until fresh context, accepted/rejected operator decisions, and replay correlation are proven.",
    },
    "decision_evidence_packet_proposal": {
        "entrypoint": "decision_evidence_packet_proposal.analyze_decision_evidence_packet_need / build_decision_evidence_packet_proposals",
        "ui_surface": "Decision evidence, Safety Monitor, Task Queue review metadata",
        "next_step": "Attach fresh-context evidence packets to proposal and rejected shadow decisions; keep them non-queueable until replay learning proves value.",
    },
    "long_term_memory_learning_proposal": {
        "entrypoint": "long_term_memory_learning_proposal.analyze_long_term_memory_learning_need / build_long_term_memory_learning_proposals",
        "ui_surface": "Memory learning evidence, replay reports, future Learning Review panel",
        "next_step": "Keep memory learning proposal-only until fresh accepted/rejected outcomes and replay correlation prove useful future decisions.",
    },
    "phase_aware_eq_guard": {
        "entrypoint": "phase_aware_eq_guard.analyze_phase_aware_eq_guard / PhaseAwareEQGuardPlugin.process",
        "ui_surface": "Future AutoEQ guard, Channel Intelligence, phase/EQ proposal traces",
        "next_step": "Use as fresh-context proposal-only phase-before-EQ evidence; do not promote phase/delay/polarity writes without separate hardware proof.",
    },
    "agentic_method_foundation": {
        "entrypoint": "channel_intelligence foundation DTO / BaselineAgenticRecommender",
        "ui_surface": "Channel Intelligence, Safety Monitor evidence, future agent panels",
        "next_step": "Promote only bounded fader/gain outputs through existing proposal adapters; keep EQ/dynamics/gate/pan/FX proposal-only.",
    },
    "context_aware_gate_agent": {
        "entrypoint": "context_aware_gate.extract_bleed_telemetry / ContextAwareGateAgent.recommend",
        "ui_surface": "Channel Intelligence, Decision Log evidence, future Gate/Bleed panel",
        "next_step": "Feed real telemetry/offline bleed simulations; keep all gate/expander outputs dry-run/proposal-only.",
    },
    "compressor_dynamics_proposal": {
        "entrypoint": "compressor_dynamics_proposal.analyze_compressor_need / build_compressor_proposals",
        "ui_surface": "Channel Intelligence, vocal/instrument dynamics proposals, future Compressor panel",
        "next_step": "Keep compressor output proposal-only until fresh dynamics readback, bounded makeup, and rollback are proven.",
    },
    "deesser_vocal_sibilance": {
        "entrypoint": "deesser_vocal_sibilance.analyze_deesser_need / build_deesser_proposals",
        "ui_surface": "Channel Intelligence, vocal processing proposals, future De-esser panel",
        "next_step": "Keep de-esser output proposal-only until fresh dynamics/dynamic-EQ readback and rollback are proven.",
    },
    "pan_spatial_proposal": {
        "entrypoint": "pan_spatial_proposal.analyze_pan_need / build_pan_proposals",
        "ui_surface": "Channel Intelligence, stereo-pair/stage pan proposals, future Panning panel",
        "next_step": "Keep pan output proposal-only until fresh current-pan readback, mono checks, and rollback are proven.",
    },
    "fx_send_proposal": {
        "entrypoint": "fx_send_proposal.analyze_fx_send_need / build_fx_send_proposals",
        "ui_surface": "Channel Intelligence, FX send proposals, future FX panel",
        "next_step": "Keep FX sends proposal-only until fresh send/return readback, feedback checks, and rollback are proven.",
    },
    "master_bus_optimization_proposal": {
        "entrypoint": "master_bus_optimization_proposal.analyze_master_bus_need / build_master_bus_proposals",
        "ui_surface": "Master bus evidence, Safety Monitor, future Master panel",
        "next_step": "Keep master-bus output proposal-only until fresh main-bus readback, limiter/dynamics proof, and rollback are proven.",
    },
    "mix_agent_conservative": {
        "entrypoint": "python -m mix_agent analyze|suggest|apply",
        "ui_surface": "Offline reports; no live cockpit apply",
        "next_step": "Import useful findings as proposals/reports, never direct WING writes.",
    },
    "offline_agent_mix": {
        "entrypoint": "tools/offline_agent_mix.py",
        "ui_surface": "Offline reports; no live cockpit apply",
        "next_step": "Keep as offline/lab renderer until a narrow adapter exists.",
    },
    "automixer_production_mix_v1": {
        "entrypoint": "none in this checkout",
        "ui_surface": "Hidden from live cockpit",
        "next_step": "Restore as an explicit offline entrypoint or retire as legacy.",
    },
    "ai_mixing_pipeline_legacy": {
        "entrypoint": "none in this checkout",
        "ui_surface": "Hidden from live cockpit",
        "next_step": "Keep deprecated until a replacement decision is made.",
    },
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _normalize_goal_capability_readiness_rows(raw_rows: Any) -> Dict[str, Any]:
    return normalize_goal_capability_readiness_rows(raw_rows)


def _normalize_goal_capability_readiness_hotspots(raw_rows: Any) -> list[Dict[str, Any]]:
    return normalize_goal_capability_readiness_hotspots(raw_rows)


def iter_processing_methods() -> Iterable[ProcessingMethod]:
    return PROCESSING_METHODS


def iter_pipeline_profiles() -> Iterable[PipelineProfile]:
    return PIPELINE_PROFILES


def get_processing_method(method_id: str) -> Optional[ProcessingMethod]:
    return next((method for method in PROCESSING_METHODS if method.id == method_id), None)


def get_pipeline_profile(profile_id: str) -> Optional[PipelineProfile]:
    return next((profile for profile in PIPELINE_PROFILES if profile.id == profile_id), None)


def build_module_status_map() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for method in PROCESSING_METHODS:
        override = MODULE_STATUS_OVERRIDES.get(method.id, {})
        next_step = (
            override.get("next_step")
            or method.promotion_next
            or (
                "Requires separate ADR, tests, physical readback, rollback, cooldown, and emergency-stop proof."
                if method.status in {"proposal_only", "lab"}
                else "Keep isolated from the main live pipeline."
            )
        )
        rows.append(
            {
                "module_id": method.id,
                "label": method.label,
                "domain": method.domain,
                "status": method.status,
                "entrypoint": override.get("entrypoint", method.module),
                "module": method.module,
                "ui_surface": override.get("ui_surface", "Processing registry / status-only surfaces"),
                "can_write_live": method.can_write_live,
                "live_write_kinds": list(method.live_write_kinds),
                "pipeline_profiles": list(method.pipeline_profiles),
                "blocked_reason": method.blocked_reason,
                "next_step": next_step,
            }
        )
    return rows


def build_processing_registry_snapshot(
    *,
    include_methods: bool = True,
    connection_mode: str = "wing",
    wing_boundary_active: bool = True,
    auto_mode_unlocked: bool = False,
) -> Dict[str, Any]:
    methods = [method.to_dict() for method in PROCESSING_METHODS]
    profiles = [profile.to_dict() for profile in PIPELINE_PROFILES]
    autonomy_promotion_readiness = build_autonomy_promotion_readiness_snapshot()
    objective_alignment_snapshot = build_project_goal_objective_alignment_snapshot()
    autonomy_mode_transition_gate = build_autonomy_mode_transition_gate(
        {
            "autonomy_promotion_readiness": autonomy_promotion_readiness,
            "auto_mode_unlocked": bool(auto_mode_unlocked),
        },
        connection_mode=connection_mode,
        wing_boundary_active=wing_boundary_active,
    )
    objective_alignment_summary = objective_alignment_snapshot.get("summary", {})
    project_goal_capability_progress_snapshot = build_project_goal_capability_progress_snapshot()
    progress_summary = project_goal_capability_progress_snapshot.get("summary", {})
    status_counts: Dict[str, int] = {status: 0 for status in sorted(PROCESSING_STATUSES)}
    domain_counts: Dict[str, int] = {domain: 0 for domain in sorted(PROCESSING_DOMAINS)}
    for method in PROCESSING_METHODS:
        status_counts[method.status] += 1
        domain_counts[method.domain] += 1

    live_executor_ids = [
        method.id
        for method in PROCESSING_METHODS
        if method.can_write_live
    ]
    return {
        "type": "processing_pipeline_snapshot",
        "generated_at": _utc_now(),
        "status": "ok",
        "summary": {
            "method_count": len(PROCESSING_METHODS),
            "pipeline_count": len(PIPELINE_PROFILES),
            "status_counts": status_counts,
            "domain_counts": domain_counts,
            "project_goal_capability_objective_required_row_count": int(
                objective_alignment_summary.get("required_row_count", 0) or 0
            ),
            "project_goal_capability_objective_present_row_count": int(
                objective_alignment_summary.get("present_row_count", 0) or 0
            ),
            "project_goal_capability_objective_missing_row_count": int(
                objective_alignment_summary.get("missing_row_count", 0) or 0
            ),
            "project_goal_capability_objective_readiness_score": int(
                objective_alignment_summary.get("autonomy_readiness_score", 0) or 0
            ),
            "project_goal_capability_objective_readiness_ready_row_count": int(
                objective_alignment_summary.get("autonomy_readiness_ready_row_count", 0) or 0
            ),
            "project_goal_capability_objective_readiness_gap_row_count": int(
                objective_alignment_summary.get("autonomy_readiness_gap_row_count", 0) or 0
            ),
            "project_goal_capability_objective_readiness_goal_area_priorities": list(
                objective_alignment_summary.get(
                    "autonomy_readiness_goal_area_priorities", []
                )
                if isinstance(
                    objective_alignment_summary.get("autonomy_readiness_goal_area_priorities"),
                    list,
                )
                else []
            ),
            "project_goal_capability_objective_readiness_by_goal_area": _normalize_goal_capability_readiness_rows(
                objective_alignment_summary.get(
                    "autonomy_readiness_by_goal_area", {}
                )
            ),
            "project_goal_capability_objective_readiness_hotspots": _normalize_goal_capability_readiness_hotspots(
                progress_summary.get(
                    "project_goal_objective_autonomy_readiness_hotspots", []
                )
                or []
            ),
            "project_goal_capability_objective_blocked_live_row_count": len(
                objective_alignment_summary.get("blocked_live_rows", ())
            ),
            "project_goal_capability_objective_currently_live_writable_row_count": int(
                objective_alignment_summary.get("currently_live_writable_row_count", 0)
                or 0
            ),
            "project_goal_capability_objective_full_autonomous_status": objective_alignment_summary.get(
                "full_autonomous_status"
            ),
            "live_executor_method_ids": live_executor_ids,
            "live_allowed_write_kinds": list(LIVE_ALLOWED_WRITE_KINDS),
            "main_pipeline_id": "main_live_assist",
            "live_apply_policy": "supervised_fader_gain_only",
        },
        "pipeline_profiles": profiles,
        "module_status_map": build_module_status_map(),
        "project_goal_capability_matrix": build_project_goal_capability_matrix_snapshot(),
        "project_goal_capability_progress_snapshot": project_goal_capability_progress_snapshot,
        "project_goal_capability_objective_alignment_snapshot": (
            objective_alignment_snapshot
        ),
        "autonomy_promotion_readiness": autonomy_promotion_readiness,
        "autonomy_mode_transition_gate": autonomy_mode_transition_gate,
        "read_surface_inventory": build_read_surface_inventory_snapshot(),
        "read_surface_remediation_plan": build_read_surface_remediation_plan(),
        "read_surface_physical_readback_capture_plan": (
            build_read_surface_physical_readback_capture_plan()
        ),
        "read_surface_fixture_evidence": build_read_surface_fixture_evidence_snapshot(),
        "wing_readonly_preflight_evidence": build_wing_readonly_preflight_evidence(),
        "raw_mixer_state_physical_readback_contract": build_raw_mixer_state_physical_readback_contract(),
        "raw_mixer_state_physical_readback_artifact_intake": (
            build_raw_mixer_state_physical_readback_artifact_intake()
        ),
        "channel_basic_controls_physical_readback_contract": (
            build_channel_basic_controls_physical_readback_contract()
        ),
        "channel_basic_controls_physical_readback_artifact_intake": (
            build_channel_basic_controls_physical_readback_artifact_intake()
        ),
        "channel_processing_physical_readback_contract": (
            build_channel_processing_physical_readback_contract()
        ),
        "channel_processing_physical_readback_artifact_intake": (
            build_channel_processing_physical_readback_artifact_intake()
        ),
        "bus_group_main_dca_physical_readback_contract": (
            build_bus_group_main_dca_physical_readback_contract()
        ),
        "bus_group_main_dca_physical_readback_artifact_intake": (
            build_bus_group_main_dca_physical_readback_artifact_intake()
        ),
        "matrix_state_physical_readback_contract": (
            build_matrix_state_physical_readback_contract()
        ),
        "matrix_state_physical_readback_artifact_intake": (
            build_matrix_state_physical_readback_artifact_intake()
        ),
        "fx_slots_inserts_physical_readback_contract": (
            build_fx_slots_inserts_physical_readback_contract()
        ),
        "fx_slots_inserts_physical_readback_artifact_intake": (
            build_fx_slots_inserts_physical_readback_artifact_intake()
        ),
        "routing_sends_physical_readback_contract": (
            build_routing_sends_physical_readback_contract()
        ),
        "routing_sends_physical_readback_artifact_intake": (
            build_routing_sends_physical_readback_artifact_intake()
        ),
        "snapshots_scenes_physical_readback_contract": (
            build_snapshots_scenes_physical_readback_contract()
        ),
        "snapshots_scenes_physical_readback_artifact_intake": (
            build_snapshots_scenes_physical_readback_artifact_intake()
        ),
        "user_presets_physical_readback_contract": (
            build_user_presets_physical_readback_contract()
        ),
        "user_presets_physical_readback_artifact_intake": (
            build_user_presets_physical_readback_artifact_intake()
        ),
        "dlive_state_physical_readback_contract": (
            build_dlive_state_physical_readback_contract()
        ),
        "dlive_state_physical_readback_artifact_intake": (
            build_dlive_state_physical_readback_artifact_intake()
        ),
        "read_surface_physical_readback_artifact_review_queue": (
            build_read_surface_physical_readback_artifact_review_queue()
        ),
        "read_surface_physical_readback_fixture_acceptance_ledger": (
            build_read_surface_physical_readback_fixture_acceptance_ledger()
        ),
        "read_surface_physical_readback_fixture_closure_plan": (
            build_read_surface_physical_readback_fixture_closure_plan()
        ),
        "source_identity_calibration_acceptance_ledger": (
            build_source_identity_calibration_acceptance_ledger()
        ),
        "source_identity_calibration_persistence_plan": (
            build_source_identity_calibration_persistence_plan()
        ),
        "source_identity_calibration_memory_draft": (
            build_source_identity_calibration_memory_draft()
        ),
        "source_identity_calibration_memory_writer": (
            build_source_identity_calibration_memory_writer()
        ),
        "source_identity_calibration_memory_index": (
            build_source_identity_calibration_memory_index()
        ),
        "operator_proposal_apply_sequence_policy_memory_index": (
            build_operator_proposal_apply_sequence_policy_memory_index()
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_review": (
            build_operator_proposal_apply_sequence_policy_memory_trust_review()
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger": (
            build_operator_proposal_apply_sequence_policy_memory_trust_acceptance_ledger()
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_plan": (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_plan()
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_contract": (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_contract()
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger": (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_acceptance_ledger()
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan": (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_execution_plan()
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_executor_dry_run": (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_executor_dry_run()
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_executor": (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_executor()
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_apply_audit": (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_apply_audit()
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_rollback_verification": (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_rollback_verification()
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_sequence_policy_gate": (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_sequence_policy_gate()
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_policy_consumer": (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_policy_consumer()
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering_preview": (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_ordering_preview()
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_ordering_plan": (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_ordering_plan()
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_importer": (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_importer()
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_plan": (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_plan()
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_result_consumer": (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_preflight_result_consumer()
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_sequence_plan_result_consumer": (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_sequence_plan_result_consumer()
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_dry_run_result_consumer": (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_dry_run_result_consumer()
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_gate_result_consumer": (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_gate_result_consumer()
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_result_consumer": (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_result_consumer()
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_audit_result_consumer": (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_execution_audit_result_consumer()
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_audit_policy_gate_result_consumer": (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_audit_policy_gate_result_consumer()
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_draft_result_consumer": (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_draft_result_consumer()
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_dry_run_result_consumer": (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_dry_run_result_consumer()
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_apply_result_consumer": (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_writer_apply_result_consumer()
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_index_result_consumer": (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_index_result_consumer()
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_review_result_consumer": (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_review_result_consumer()
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_acceptance_ledger_result_consumer": (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_memory_trust_acceptance_ledger_result_consumer()
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_plan_result_consumer": (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_plan_result_consumer()
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_contract_result_consumer": (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_contract_result_consumer()
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_acceptance_ledger_result_consumer": (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_acceptance_ledger_result_consumer()
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_plan_result_consumer": (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_plan_result_consumer()
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_dry_run_result_consumer": (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_dry_run_result_consumer()
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_result_consumer": (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_executor_result_consumer()
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_apply_audit_result_consumer": (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_apply_audit_result_consumer()
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_rollback_verification_result_consumer": (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_rollback_verification_result_consumer()
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_sequence_policy_gate_result_consumer": (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_sequence_policy_gate_result_consumer()
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_policy_consumer_result_consumer": (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_policy_consumer_result_consumer()
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_ordering_preview_result_consumer": (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_ordering_preview_result_consumer()
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_ordering_plan_result_consumer": (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_ordering_plan_result_consumer()
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer": (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_queue_importer_result_consumer()
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_preflight_plan_result_consumer": (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_preflight_plan_result_consumer()
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_batch_preflight_result_consumer": (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_batch_preflight_result_consumer()
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_sequence_plan_result_consumer": (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_sequence_plan_result_consumer()
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_dry_run_result_consumer": (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_dry_run_result_consumer()
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_gate_result_consumer": (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_gate_result_consumer()
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_result_consumer": (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_result_consumer()
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_audit_result_consumer": (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_execution_audit_result_consumer()
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_audit_policy_gate_result_consumer": (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_audit_policy_gate_result_consumer()
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_draft_result_consumer": (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_draft_result_consumer()
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_writer_dry_run_result_consumer": (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_writer_dry_run_result_consumer()
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_writer_apply_result_consumer": (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_writer_apply_result_consumer()
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_index_result_consumer": (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_index_result_consumer()
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_trust_review_result_consumer": (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_trust_review_result_consumer()
        ),
        "operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_trust_acceptance_ledger_result_consumer": (
            build_operator_proposal_apply_sequence_policy_memory_trust_promotion_queue_import_promotion_memory_trust_acceptance_ledger_result_consumer()
        ),
        "source_identity_calibration_memory_trust_review": (
            build_source_identity_calibration_memory_trust_review()
        ),
        "source_identity_calibration_memory_trust_acceptance_ledger": (
            build_source_identity_calibration_memory_trust_acceptance_ledger()
        ),
        "source_identity_calibration_memory_trust_promotion_plan": (
            build_source_identity_calibration_memory_trust_promotion_plan()
        ),
        "source_identity_calibration_memory_trust_promotion_contract": (
            build_source_identity_calibration_memory_trust_promotion_contract()
        ),
        "source_identity_calibration_memory_trust_promotion_acceptance_ledger": (
            build_source_identity_calibration_memory_trust_promotion_acceptance_ledger()
        ),
        "source_identity_calibration_memory_trust_promotion_execution_plan": (
            build_source_identity_calibration_memory_trust_promotion_execution_plan()
        ),
        "source_identity_calibration_memory_trust_promotion_executor_dry_run": (
            build_source_identity_calibration_memory_trust_promotion_executor_dry_run()
        ),
        "source_identity_calibration_memory_trust_promotion_executor": (
            build_source_identity_calibration_memory_trust_promotion_executor()
        ),
        "source_identity_calibration_memory_trust_promotion_apply_audit": (
            build_source_identity_calibration_memory_trust_promotion_apply_audit()
        ),
        "source_identity_calibration_memory_trust_promotion_rollback_verification": (
            build_source_identity_calibration_memory_trust_promotion_rollback_verification()
        ),
        "source_identity_calibration_memory_trust_promotion_priority_gate": (
            build_source_identity_calibration_memory_trust_promotion_priority_gate()
        ),
        "source_identity_proposal_priority_policy": (
            build_source_identity_proposal_priority_policy()
        ),
        "source_identity_proposal_ordering_preview": (
            build_source_identity_proposal_ordering_preview()
        ),
        "source_identity_proposal_queue_ordering_plan": (
            build_source_identity_proposal_queue_ordering_plan()
        ),
        "source_identity_proposal_queue_importer": (
            build_source_identity_proposal_queue_importer()
        ),
        "safety_coverage_matrix": build_safety_coverage_matrix_snapshot(),
        "write_surface_inventory": build_write_surface_inventory_snapshot(),
        "methods": methods if include_methods else [],
        "promotion_ladder": list(PROMOTION_LADDER),
    }


def build_pipeline_runtime_summary(operator_mode: Dict[str, Any], manual_write_supervision: Dict[str, Any]) -> Dict[str, Any]:
    manual = manual_write_supervision if isinstance(manual_write_supervision, dict) else {}
    approved_live_write_kinds = manual.get("approved_live_write_kinds")
    if not approved_live_write_kinds:
        allowed = list(
            operator_mode.get("capabilities", {}).get("allowed_live_write_kinds") or []
        )
        if (
            operator_mode.get("mode") == "supervised"
            and "mute" not in allowed
        ):
            allowed.append("mute")
        approved_live_write_kinds = allowed
    else:
        approved_live_write_kinds = normalize_manual_live_write_kinds(
            approved_live_write_kinds
        )
    main = get_pipeline_profile("main_live_assist")
    return {
        "main_pipeline_id": "main_live_assist",
        "main_pipeline_status": main.status if main else "missing",
        "live_apply_policy": "supervised_fader_gain_only",
        "allowed_live_write_kinds": list(approved_live_write_kinds),
        "operator_mode": operator_mode.get("mode"),
        "can_apply_to_console": bool(operator_mode.get("capabilities", {}).get("can_apply_to_console")),
        "manual_write_supervision": manual_write_supervision,
        "secondary_pipeline_ids": ["offline_mix", "shadow_quality", "lab_dsp", "quarantine"],
    }
