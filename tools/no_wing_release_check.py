#!/usr/bin/env python3
"""Run the no-WING Automixer release readiness checks.

The runner is intentionally local/offline. It executes tests, builds, and the
offline operator validation harness, then writes a small artifact bundle with
logs and a machine-readable summary. It must never connect to WING or send live
write commands.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import re
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.operator_mode_policy import (
    CURRENT_PROVEN_LIVE_WRITE_KINDS,
    MODE_SUPERVISED,
    build_operator_mode_status,
)  # noqa: E402
from backend.bus_group_main_dca_physical_readback_contract import (  # noqa: E402
    BUS_GROUP_MAIN_DCA_FORBIDDEN_MUTATION_EVIDENCE_FIELDS,
    BUS_GROUP_MAIN_DCA_PHYSICAL_READBACK_ARTIFACT_TYPE,
    BUS_GROUP_MAIN_DCA_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
    BUS_GROUP_MAIN_DCA_PHYSICAL_READBACK_SCHEMA_ID,
    BUS_GROUP_MAIN_DCA_PHYSICAL_READBACK_SOURCES,
    BUS_GROUP_MAIN_DCA_REMEDIATION_ACTION_ID,
    BUS_GROUP_MAIN_DCA_REQUIRED_BUS_ASSIGNMENT_FIELDS,
    BUS_GROUP_MAIN_DCA_REQUIRED_BUS_CONTROL_FIELDS,
    BUS_GROUP_MAIN_DCA_REQUIRED_BUS_COVERAGE_FLAGS,
    BUS_GROUP_MAIN_DCA_REQUIRED_BUS_FIELDS,
    BUS_GROUP_MAIN_DCA_REQUIRED_DCA_ASSIGNMENT_FIELDS,
    BUS_GROUP_MAIN_DCA_REQUIRED_DCA_CONTROL_FIELDS,
    BUS_GROUP_MAIN_DCA_REQUIRED_DCA_COVERAGE_FLAGS,
    BUS_GROUP_MAIN_DCA_REQUIRED_DCA_FIELDS,
    BUS_GROUP_MAIN_DCA_REQUIRED_FALSE_SAFETY_FLAGS,
    BUS_GROUP_MAIN_DCA_REQUIRED_MAIN_CONTROL_FIELDS,
    BUS_GROUP_MAIN_DCA_REQUIRED_MAIN_COVERAGE_FLAGS,
    BUS_GROUP_MAIN_DCA_REQUIRED_MAIN_FIELDS,
    BUS_GROUP_MAIN_DCA_REQUIRED_METERING_FIELDS,
    BUS_GROUP_MAIN_DCA_REQUIRED_TOP_LEVEL_FIELDS,
    BUS_GROUP_MAIN_DCA_SURFACE_ID,
    BUS_GROUP_MAIN_DCA_VERIFIED_PHYSICAL_STATUSES,
    build_bus_group_main_dca_physical_readback_contract,
)
from backend.bus_group_main_dca_physical_readback_artifact_intake import (  # noqa: E402
    BUS_GROUP_MAIN_DCA_INTAKE_STATUSES,
    BUS_GROUP_MAIN_DCA_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID,
    build_bus_group_main_dca_physical_readback_artifact_intake,
)
from backend.channel_basic_controls_physical_readback_contract import (  # noqa: E402
    CHANNEL_BASIC_CONTROLS_FORBIDDEN_MUTATION_EVIDENCE_FIELDS,
    CHANNEL_BASIC_CONTROLS_PHYSICAL_READBACK_ARTIFACT_TYPE,
    CHANNEL_BASIC_CONTROLS_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
    CHANNEL_BASIC_CONTROLS_PHYSICAL_READBACK_SCHEMA_ID,
    CHANNEL_BASIC_CONTROLS_PHYSICAL_READBACK_SOURCES,
    CHANNEL_BASIC_CONTROLS_REMEDIATION_ACTION_ID,
    CHANNEL_BASIC_CONTROLS_REQUIRED_CHANNEL_FIELDS,
    CHANNEL_BASIC_CONTROLS_REQUIRED_CONTROL_FIELDS,
    CHANNEL_BASIC_CONTROLS_REQUIRED_COVERAGE_FLAGS,
    CHANNEL_BASIC_CONTROLS_REQUIRED_FALSE_SAFETY_FLAGS,
    CHANNEL_BASIC_CONTROLS_REQUIRED_IDENTITY_FIELDS,
    CHANNEL_BASIC_CONTROLS_REQUIRED_TOP_LEVEL_FIELDS,
    CHANNEL_BASIC_CONTROLS_SURFACE_ID,
    CHANNEL_BASIC_CONTROLS_VERIFIED_PHYSICAL_STATUSES,
    build_channel_basic_controls_physical_readback_contract,
)
from backend.channel_basic_controls_physical_readback_artifact_intake import (  # noqa: E402
    CHANNEL_BASIC_CONTROLS_INTAKE_STATUSES,
    CHANNEL_BASIC_CONTROLS_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID,
    build_channel_basic_controls_physical_readback_artifact_intake,
)
from backend.channel_processing_physical_readback_contract import (  # noqa: E402
    CHANNEL_PROCESSING_FORBIDDEN_MUTATION_EVIDENCE_FIELDS,
    CHANNEL_PROCESSING_PHYSICAL_READBACK_ARTIFACT_TYPE,
    CHANNEL_PROCESSING_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
    CHANNEL_PROCESSING_PHYSICAL_READBACK_SCHEMA_ID,
    CHANNEL_PROCESSING_PHYSICAL_READBACK_SOURCES,
    CHANNEL_PROCESSING_REMEDIATION_ACTION_ID,
    CHANNEL_PROCESSING_REQUIRED_CHANNEL_FIELDS,
    CHANNEL_PROCESSING_REQUIRED_COMPRESSOR_FIELDS,
    CHANNEL_PROCESSING_REQUIRED_COVERAGE_FLAGS,
    CHANNEL_PROCESSING_REQUIRED_EQ_BAND_FIELDS,
    CHANNEL_PROCESSING_REQUIRED_EQ_FIELDS,
    CHANNEL_PROCESSING_REQUIRED_FALSE_SAFETY_FLAGS,
    CHANNEL_PROCESSING_REQUIRED_GATE_FIELDS,
    CHANNEL_PROCESSING_REQUIRED_HPF_FIELDS,
    CHANNEL_PROCESSING_REQUIRED_TOP_LEVEL_FIELDS,
    CHANNEL_PROCESSING_SURFACE_ID,
    CHANNEL_PROCESSING_VERIFIED_PHYSICAL_STATUSES,
    MINIMUM_EQ_BAND_COUNT,
    build_channel_processing_physical_readback_contract,
)
from backend.channel_processing_physical_readback_artifact_intake import (  # noqa: E402
    CHANNEL_PROCESSING_INTAKE_STATUSES,
    CHANNEL_PROCESSING_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID,
    build_channel_processing_physical_readback_artifact_intake,
)
from backend.dlive_state_physical_readback_contract import (  # noqa: E402
    DLIVE_STATE_FORBIDDEN_MUTATION_EVIDENCE_FIELDS,
    DLIVE_STATE_PHYSICAL_READBACK_ARTIFACT_TYPE,
    DLIVE_STATE_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
    DLIVE_STATE_PHYSICAL_READBACK_SCHEMA_ID,
    DLIVE_STATE_PHYSICAL_READBACK_SOURCES,
    DLIVE_STATE_REMEDIATION_ACTION_ID,
    DLIVE_STATE_REQUIRED_CHANNEL_FIELDS,
    DLIVE_STATE_REQUIRED_COVERAGE_FLAGS,
    DLIVE_STATE_REQUIRED_FALSE_SAFETY_FLAGS,
    DLIVE_STATE_REQUIRED_MIDI_STATE_FIELDS,
    DLIVE_STATE_REQUIRED_PARITY_FIELDS,
    DLIVE_STATE_REQUIRED_TOP_LEVEL_FIELDS,
    DLIVE_STATE_SURFACE_ID,
    DLIVE_STATE_VERIFIED_PHYSICAL_STATUSES,
    MINIMUM_DLIVE_STATE_CHANNEL_COUNT,
    build_dlive_state_physical_readback_contract,
)
from backend.dlive_state_physical_readback_artifact_intake import (  # noqa: E402
    DLIVE_STATE_INTAKE_STATUSES,
    DLIVE_STATE_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID,
    build_dlive_state_physical_readback_artifact_intake,
)
from backend.fx_slots_inserts_physical_readback_contract import (  # noqa: E402
    FX_INSERT_POSITIONS,
    FX_INSERT_REQUIRED_COVERAGE_FLAGS,
    FX_INSERT_REQUIRED_FIELDS,
    FX_INSERT_TARGET_TYPES,
    FX_PARAMETER_REQUIRED_FIELDS,
    FX_PARAMETER_SAFE_RANGE_FIELDS,
    FX_PARAMETER_SCHEMA_REQUIRED_FIELDS,
    FX_SLOT_REQUIRED_COVERAGE_FLAGS,
    FX_SLOT_REQUIRED_FIELDS,
    FX_SLOTS_INSERTS_FORBIDDEN_MUTATION_EVIDENCE_FIELDS,
    FX_SLOTS_INSERTS_PHYSICAL_READBACK_ARTIFACT_TYPE,
    FX_SLOTS_INSERTS_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
    FX_SLOTS_INSERTS_PHYSICAL_READBACK_SCHEMA_ID,
    FX_SLOTS_INSERTS_PHYSICAL_READBACK_SOURCES,
    FX_SLOTS_INSERTS_REMEDIATION_ACTION_ID,
    FX_SLOTS_INSERTS_REQUIRED_FALSE_SAFETY_FLAGS,
    FX_SLOTS_INSERTS_REQUIRED_TOP_LEVEL_FIELDS,
    FX_SLOTS_INSERTS_SURFACE_ID,
    FX_SLOTS_INSERTS_VERIFIED_PHYSICAL_STATUSES,
    MINIMUM_FX_INSERT_COUNT,
    MINIMUM_FX_PARAMETER_COUNT,
    MINIMUM_FX_SLOT_COUNT,
    build_fx_slots_inserts_physical_readback_contract,
)
from backend.fx_slots_inserts_physical_readback_artifact_intake import (  # noqa: E402
    FX_SLOTS_INSERTS_INTAKE_STATUSES,
    FX_SLOTS_INSERTS_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID,
    build_fx_slots_inserts_physical_readback_artifact_intake,
)
from backend.matrix_state_physical_readback_contract import (  # noqa: E402
    MATRIX_STATE_FORBIDDEN_MUTATION_EVIDENCE_FIELDS,
    MATRIX_STATE_PHYSICAL_READBACK_ARTIFACT_TYPE,
    MATRIX_STATE_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
    MATRIX_STATE_PHYSICAL_READBACK_SCHEMA_ID,
    MATRIX_STATE_PHYSICAL_READBACK_SOURCES,
    MATRIX_STATE_REMEDIATION_ACTION_ID,
    MATRIX_STATE_REQUIRED_CONTROL_FIELDS,
    MATRIX_STATE_REQUIRED_COVERAGE_FLAGS,
    MATRIX_STATE_REQUIRED_DYNAMICS_FIELDS,
    MATRIX_STATE_REQUIRED_EQ_BAND_FIELDS,
    MATRIX_STATE_REQUIRED_EQ_FIELDS,
    MATRIX_STATE_REQUIRED_FALSE_SAFETY_FLAGS,
    MATRIX_STATE_REQUIRED_INSERT_FIELDS,
    MATRIX_STATE_REQUIRED_MATRIX_FIELDS,
    MATRIX_STATE_REQUIRED_METERING_FIELDS,
    MATRIX_STATE_REQUIRED_TOP_LEVEL_FIELDS,
    MATRIX_STATE_SURFACE_ID,
    MATRIX_STATE_VERIFIED_PHYSICAL_STATUSES,
    MINIMUM_MATRIX_EQ_BAND_COUNT,
    build_matrix_state_physical_readback_contract,
)
from backend.matrix_state_physical_readback_artifact_intake import (  # noqa: E402
    MATRIX_STATE_INTAKE_STATUSES,
    MATRIX_STATE_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID,
    build_matrix_state_physical_readback_artifact_intake,
)
from backend.read_surface_fixture_evidence import (  # noqa: E402
    FIXTURE_EVIDENCE_STATUSES,
    build_read_surface_fixture_evidence_snapshot,
)
from backend.raw_mixer_state_physical_readback_contract import (  # noqa: E402
    FORBIDDEN_MUTATION_EVIDENCE_FIELDS,
    PHYSICAL_READBACK_SOURCES,
    RAW_MIXER_STATE_PHYSICAL_READBACK_ARTIFACT_TYPE,
    RAW_MIXER_STATE_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
    RAW_MIXER_STATE_PHYSICAL_READBACK_SCHEMA_ID,
    RAW_MIXER_STATE_REMEDIATION_ACTION_ID,
    RAW_MIXER_STATE_SURFACE_ID,
    REQUIRED_FALSE_SAFETY_FLAGS,
    REQUIRED_ROW_FIELDS,
    REQUIRED_TOP_LEVEL_FIELDS,
    VERIFIED_PHYSICAL_ROW_STATUSES,
    build_raw_mixer_state_physical_readback_contract,
)
from backend.raw_mixer_state_physical_readback_artifact_intake import (  # noqa: E402
    INTAKE_STATUSES,
    RAW_MIXER_STATE_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID,
    build_raw_mixer_state_physical_readback_artifact_intake,
)
from backend.routing_sends_physical_readback_contract import (  # noqa: E402
    MINIMUM_CHANNEL_BUS_SEND_COUNT,
    MINIMUM_CHANNEL_MAIN_SEND_COUNT,
    MINIMUM_CHANNEL_MATRIX_SEND_COUNT,
    MINIMUM_OUTPUT_ROUTE_COUNT,
    MINIMUM_STATIC_DANTE_ROUTE_COUNT,
    ROUTING_SENDS_FORBIDDEN_MUTATION_EVIDENCE_FIELDS,
    ROUTING_SENDS_PHYSICAL_READBACK_ARTIFACT_TYPE,
    ROUTING_SENDS_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
    ROUTING_SENDS_PHYSICAL_READBACK_SCHEMA_ID,
    ROUTING_SENDS_PHYSICAL_READBACK_SOURCES,
    ROUTING_SENDS_REMEDIATION_ACTION_ID,
    ROUTING_SENDS_REQUIRED_BUS_SEND_FIELDS,
    ROUTING_SENDS_REQUIRED_FALSE_SAFETY_FLAGS,
    ROUTING_SENDS_REQUIRED_MAIN_SEND_FIELDS,
    ROUTING_SENDS_REQUIRED_MATRIX_SEND_FIELDS,
    ROUTING_SENDS_REQUIRED_OUTPUT_ROUTE_COVERAGE_FLAGS,
    ROUTING_SENDS_REQUIRED_OUTPUT_ROUTE_FIELDS,
    ROUTING_SENDS_REQUIRED_SEND_COVERAGE_FLAGS,
    ROUTING_SENDS_REQUIRED_STATIC_ROUTE_COVERAGE_FLAGS,
    ROUTING_SENDS_REQUIRED_STATIC_ROUTE_FIELDS,
    ROUTING_SENDS_REQUIRED_TOP_LEVEL_FIELDS,
    ROUTING_SENDS_SURFACE_ID,
    ROUTING_SENDS_VERIFIED_PHYSICAL_STATUSES,
    build_routing_sends_physical_readback_contract,
)
from backend.routing_sends_physical_readback_artifact_intake import (  # noqa: E402
    ROUTING_SENDS_INTAKE_STATUSES,
    ROUTING_SENDS_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID,
    build_routing_sends_physical_readback_artifact_intake,
)
from backend.snapshots_scenes_physical_readback_contract import (  # noqa: E402
    MINIMUM_SNAPSHOT_SCENE_COUNT,
    SNAPSHOTS_SCENES_FORBIDDEN_MUTATION_EVIDENCE_FIELDS,
    SNAPSHOTS_SCENES_PHYSICAL_READBACK_ARTIFACT_TYPE,
    SNAPSHOTS_SCENES_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
    SNAPSHOTS_SCENES_PHYSICAL_READBACK_SCHEMA_ID,
    SNAPSHOTS_SCENES_PHYSICAL_READBACK_SOURCES,
    SNAPSHOTS_SCENES_REMEDIATION_ACTION_ID,
    SNAPSHOTS_SCENES_REQUIRED_ACTIVE_SCENE_FIELDS,
    SNAPSHOTS_SCENES_REQUIRED_COVERAGE_FLAGS,
    SNAPSHOTS_SCENES_REQUIRED_FALSE_SAFETY_FLAGS,
    SNAPSHOTS_SCENES_REQUIRED_SNAPSHOT_FIELDS,
    SNAPSHOTS_SCENES_REQUIRED_TOP_LEVEL_FIELDS,
    SNAPSHOTS_SCENES_SURFACE_ID,
    SNAPSHOTS_SCENES_VERIFIED_PHYSICAL_STATUSES,
    build_snapshots_scenes_physical_readback_contract,
)
from backend.snapshots_scenes_physical_readback_artifact_intake import (  # noqa: E402
    SNAPSHOTS_SCENES_INTAKE_STATUSES,
    SNAPSHOTS_SCENES_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID,
    build_snapshots_scenes_physical_readback_artifact_intake,
)
from backend.user_presets_physical_readback_contract import (  # noqa: E402
    MINIMUM_USER_PRESET_COUNT,
    USER_PRESETS_FORBIDDEN_MUTATION_EVIDENCE_FIELDS,
    USER_PRESETS_PHYSICAL_READBACK_ARTIFACT_TYPE,
    USER_PRESETS_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
    USER_PRESETS_PHYSICAL_READBACK_SCHEMA_ID,
    USER_PRESETS_PHYSICAL_READBACK_SOURCES,
    USER_PRESETS_REMEDIATION_ACTION_ID,
    USER_PRESETS_REQUIRED_COVERAGE_FLAGS,
    USER_PRESETS_REQUIRED_FALSE_SAFETY_FLAGS,
    USER_PRESETS_REQUIRED_PRESET_FIELDS,
    USER_PRESETS_REQUIRED_TOP_LEVEL_FIELDS,
    USER_PRESETS_SURFACE_ID,
    USER_PRESETS_VERIFIED_PHYSICAL_STATUSES,
    build_user_presets_physical_readback_contract,
)
from backend.user_presets_physical_readback_artifact_intake import (  # noqa: E402
    USER_PRESETS_INTAKE_STATUSES,
    USER_PRESETS_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID,
    build_user_presets_physical_readback_artifact_intake,
)
from backend.wing_readonly_preflight_evidence import (  # noqa: E402
    EVIDENCE_STATUSES as WING_READONLY_PREFLIGHT_EVIDENCE_STATUSES,
    WING_READONLY_PREFLIGHT_EVIDENCE_METHOD_ID,
    build_wing_readonly_preflight_evidence,
)
from backend.safety_coverage_matrix import get_safety_coverage_row  # noqa: E402
from backend import wing_readonly_preflight_contract as preflight_contract  # noqa: E402
from backend.wing_readonly_preflight_review_decision import (  # noqa: E402
    WING_READONLY_PREFLIGHT_REVIEW_DECISION_LEDGER_METHOD_ID,
    WING_READONLY_PREFLIGHT_REVIEW_DECISION_LEDGER_STATUSES,
    build_wing_readonly_preflight_review_decision_ledger,
)
from backend.read_surface_physical_readback_capture_plan import (  # noqa: E402
    READ_SURFACE_PHYSICAL_READBACK_CAPTURE_PLAN_METHOD_ID,
    build_read_surface_physical_readback_capture_plan,
)
from backend.read_surface_physical_readback_artifact_review_queue import (  # noqa: E402
    INTAKE_SPECS,
    READ_SURFACE_PHYSICAL_READBACK_ARTIFACT_REVIEW_QUEUE_METHOD_ID,
    READ_SURFACE_PHYSICAL_READBACK_ARTIFACT_REVIEW_QUEUE_STATUSES,
    build_read_surface_physical_readback_artifact_review_queue,
)
from backend.read_surface_physical_readback_fixture_acceptance_ledger import (  # noqa: E402
    READ_SURFACE_PHYSICAL_READBACK_FIXTURE_ACCEPTANCE_LEDGER_METHOD_ID,
    READ_SURFACE_PHYSICAL_READBACK_FIXTURE_ACCEPTANCE_LEDGER_STATUSES,
    build_read_surface_physical_readback_fixture_acceptance_ledger,
)
from backend.read_surface_physical_readback_fixture_closure_plan import (  # noqa: E402
    READ_SURFACE_PHYSICAL_READBACK_FIXTURE_CLOSURE_PLAN_METHOD_ID,
    READ_SURFACE_PHYSICAL_READBACK_FIXTURE_CLOSURE_PLAN_STATUSES,
    build_read_surface_physical_readback_fixture_closure_plan,
)
from backend.read_surface_remediation_plan import build_read_surface_remediation_plan  # noqa: E402

DEFAULT_ARTIFACT_ROOT = REPO_ROOT / "artifacts" / "no_wing_release_check"
DEFAULT_BACKEND_PORT = 8765
DEFAULT_COMMAND_TIMEOUT_SEC = 900
AUTONOMY_GAP_ROLLUP_SOURCE = "autonomy_mode_transition_gate.rows[mode=auto]"


def _required_gain_evidence_components() -> list[str]:
    row = get_safety_coverage_row("gain_supervised")
    return list(row.required_evidence_components) if row else []


DEFAULT_HOMEBREW_GIT = Path("/opt/homebrew/bin/git")
GAIN_DRY_RUN_REASON = "no_wing_release_gain_dry_run"
REPO_LOCAL_COMMAND_OUTPUT_FLAGS = {"--artifact-root", "--output", "--plan-output"}
WING_READONLY_PREFLIGHT_RUNNER_TOOL = "tools/wing_readonly_preflight_runner.py"
WING_READONLY_PREFLIGHT_RUNNER_URL = "ws://localhost:8765"
WING_READONLY_PREFLIGHT_RUNNER_CHANNELS = "3"
WING_READONLY_PREFLIGHT_RUNNER_VALUE_TYPES = "fader,gain,pan"
WING_READONLY_PREFLIGHT_CAPTURE_ARTIFACT_ROOT = "artifacts/wing_readonly_preflight"
WING_READONLY_PREFLIGHT_PLAN_OUTPUT = (
    "artifacts/wing_readonly_preflight_plan/latest_plan.json"
)
WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND = list(
    preflight_contract.WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND
)
WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS = list(
    getattr(
        preflight_contract,
        "WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS",
    )
)
WING_READONLY_PREFLIGHT_PROOF_ARTIFACT_SCHEMA = (
    preflight_contract.WING_READONLY_PREFLIGHT_PROOF_ARTIFACT_SCHEMA
)
WING_READONLY_PREFLIGHT_SUMMARY_ARTIFACT_NAME = (
    preflight_contract.WING_READONLY_PREFLIGHT_SUMMARY_ARTIFACT_NAME
)
WING_READONLY_PREFLIGHT_EVENTS_ARTIFACT_NAME = (
    preflight_contract.WING_READONLY_PREFLIGHT_EVENTS_ARTIFACT_NAME
)
WING_READONLY_PREFLIGHT_NEXT_MANUAL_ARTIFACT_REQUIRED = (
    preflight_contract.WING_READONLY_PREFLIGHT_NEXT_MANUAL_ARTIFACT_REQUIRED
)
BACKEND_PORT_LISTENER_CONFIRMATION_CHECKLIST = [
    "confirm whether the listener is the intended Automixer backend for this WING preflight",
    "confirm the listener command line and PID match the operator-started backend session",
    "do not stop or reconfigure the listener from this no-WING gate",
    "rerun the no-WING gate after manual confirmation or after clearing an unexpected listener",
]
BACKEND_PORT_LISTENER_CONFIRMATION_ARTIFACT_TYPE = (
    "wing_backend_port_listener_confirmation.v1"
)
BACKEND_PORT_LISTENER_CONFIRMATION_ACCEPTED_STATUS = (
    "operator_confirmed_intended_backend"
)
WING_READONLY_PREFLIGHT_REVIEW_CANDIDATE_PRESTART_BLOCKER = (
    preflight_contract.WING_READONLY_PREFLIGHT_REVIEW_CANDIDATE_PRESTART_BLOCKER
)
WING_READONLY_PREFLIGHT_HANDOFF_REQUIRED_SUPERVISED_PROOF_GATE_MARKERS = (
    "accepted_decision_ready",
    "valid accepted-candidate prestart trace",
    "WING_SUPERVISED_FADER_GAIN_PROOF",
)
WING_READONLY_PREFLIGHT_REVIEW_TOOL = "tools/review_wing_readonly_preflight_artifacts.py"
WING_READONLY_PREFLIGHT_REVIEW_ARTIFACT_ROOT = WING_READONLY_PREFLIGHT_CAPTURE_ARTIFACT_ROOT
WING_READONLY_PREFLIGHT_REVIEW_OUTPUT = "Docs/reports/WING_READONLY_PREFLIGHT_REVIEW_LATEST.md"
WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_OUTPUT = (
    "artifacts/wing_readonly_preflight_review_decisions/pending_review_decision_template.json"
)
WING_READONLY_PREFLIGHT_REVIEW_DECISION_TOOL = (
    "tools/review_wing_readonly_preflight_decisions.py"
)
WING_READONLY_PREFLIGHT_REVIEW_DECISION_ROOT = (
    "artifacts/wing_readonly_preflight_review_decisions"
)
WING_READONLY_PREFLIGHT_REVIEW_DECISION_OUTPUT = (
    "Docs/reports/WING_READONLY_PREFLIGHT_DECISION_REVIEW_LATEST.md"
)
WING_READONLY_PREFLIGHT_HANDOFF_PACKET_TOOL = (
    "tools/build_wing_readonly_preflight_handoff_packet.py"
)
WING_READONLY_PREFLIGHT_HANDOFF_PACKET_OUTPUT = (
    "Docs/reports/WING_READONLY_PREFLIGHT_HANDOFF_PACKET_LATEST.md"
)
READ_SURFACE_PHYSICAL_READBACK_CAPTURE_PACKET_TOOL = (
    "tools/build_read_surface_physical_readback_capture_packet.py"
)
READ_SURFACE_PHYSICAL_READBACK_CAPTURE_TEMPLATE_TOOL = (
    "tools/build_read_surface_physical_readback_capture_template.py"
)
READ_SURFACE_PHYSICAL_READBACK_ARTIFACT_REVIEW_QUEUE_TOOL = (
    "tools/build_read_surface_physical_readback_artifact_review_queue.py"
)
READ_SURFACE_PHYSICAL_READBACK_ACCEPTANCE_TEMPLATE_TOOL = (
    "tools/build_read_surface_physical_readback_acceptance_template.py"
)
READ_SURFACE_PHYSICAL_READBACK_FIXTURE_CLOSURE_PLAN_TOOL = (
    "tools/build_read_surface_physical_readback_fixture_closure_plan.py"
)
WING_READONLY_PREFLIGHT_REVIEW_EXIT_CODES = {
    "candidate_ready": 0,
    "artifact_missing": 2,
    "invalid_artifact": 3,
}
WING_READONLY_PREFLIGHT_REVIEW_DECISION_EXIT_CODES = {
    "accepted_decision_ready": 0,
    "preflight_candidate_missing": 2,
    "decision_missing_or_invalid": 3,
    "candidate_rejected": 4,
}
WING_READONLY_PREFLIGHT_REVIEW_MANUAL_CHECKLIST_IDS = (
    "confirm_artifact_paths",
    "confirm_operator_channel_scope",
    "confirm_operator_session_ready",
    "confirm_physical_readbacks",
    "confirm_event_log_readonly_sequence",
    "confirm_no_live_authority",
)
WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_REQUIRED_FIELDS = (
    "review_id",
    "artifact_source_id",
    "run_id",
    "artifact_captured_at",
    "summary_path",
    "summary_sha256",
    "events_path",
    "events_sha256",
    "decision",
    "reviewed_by",
    "reviewed_at",
    "checklist_ids",
)
WING_READONLY_PREFLIGHT_HANDOFF_REQUIRED_FORBIDDEN_REQUEST_TYPES = (
    "create_operator_proposal",
    "apply_operator_proposal",
    "rollback_last_manual_write",
    "emergency_stop",
    "set_operator_mode",
    "set_fader",
    "set_gain",
    "set_trim",
    "set_input_trim",
    "set_channel_trim",
    "set_preamp_trim",
    "set_preamp_gain",
    "set_eq",
    "set_compressor",
    "route_output",
    "load_snap",
    "save_snap",
    "reset",
    "mute",
    "unmute",
    "set_insert",
    "set_fx_model",
    "set_fx_on",
    "set_fx_mix",
    "set_fx_parameter",
    "connect_wing",
    "disconnect_wing",
    "connect_mixer",
    "disconnect_mixer",
)
WING_READONLY_PREFLIGHT_HANDOFF_REQUIRED_NOT_CLOSED_BLOCKERS = (
    "wing_readonly_connection_proof",
    "physical_console_readback_proof",
    "supervised_fader_write_proof",
    "supervised_gain_down_trim_proof",
    "readiness_matrix_update_based_on_real_proof",
)
WING_MVP_BLOCKERS_TABLE = ".ai/project-layer/wing_mvp_blockers.md"
WING_READONLY_PREFLIGHT_PLAN_REQUIRED_FAIL_CLOSED_CONDITIONS = (
    "backend_not_in_wing_mode",
    "event_log_wing_mode_not_proven",
    "transport_timeout_or_connection_loss",
    "ws_handshake_rejected",
    "operator_channel_confirmation_mismatch",
    "operator_session_confirmation_missing",
    "physical_readback_missing",
    "readback_identity_mismatch",
    "readback_request_id_mismatch",
    "expected_plan_missing_or_mismatch",
    "stale_backend_port_prestart_detail_contract",
    "existing_backend_connection_boundary_missing",
    "forbidden_request_type_sent",
    "unexpected_response_type",
    "event_log_missing_or_invalid",
    "manual_review_decision_missing_or_invalid",
)


def _markdown_table_cells(line: str) -> list[str]:
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return []
    return [cell.strip() for cell in stripped.strip("|").split("|")]


def _extract_backtick_paths(text: str) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r"`([^`]+)`", text):
        path = match.group(1)
        if "/" not in path or path in seen:
            continue
        seen.add(path)
        paths.append(path)
    return paths


def wing_mvp_blocker_anchor_sha256(anchor: Mapping[str, object]) -> str:
    canonical = json.dumps(anchor, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def wing_mvp_first_open_blocker_anchor(
    blockers_path: Path | None = None,
) -> dict[str, object]:
    path = blockers_path or (REPO_ROOT / WING_MVP_BLOCKERS_TABLE)
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        cells = _markdown_table_cells(line)
        if len(cells) != 4:
            continue
        if cells[0] in {"Blocker", "---"}:
            continue
        rows.append(
            {
                "blocker": cells[0],
                "status": cells[1],
                "evidence": cells[2],
                "next_action": cells[3],
            }
        )
    first_open = next((row for row in rows if row["status"] != "closed"), None)
    if first_open is None:
        return {
            "blocker_table": WING_MVP_BLOCKERS_TABLE,
            "selected_blocker": "none",
            "selected_blocker_status": "closed",
            "selected_action": "All MVP blockers are closed.",
            "selected_evidence_file_count": 0,
            "selected_evidence_files": [],
            "packet_does_not_close_blocker": True,
        }
    evidence_files = _extract_backtick_paths(first_open["evidence"])
    return {
        "blocker_table": WING_MVP_BLOCKERS_TABLE,
        "selected_blocker": first_open["blocker"],
        "selected_blocker_status": first_open["status"],
        "selected_action": first_open["next_action"],
        "selected_evidence_file_count": len(evidence_files),
        "selected_evidence_files": evidence_files,
        "packet_does_not_close_blocker": True,
    }


def wing_readonly_preflight_capture_command_template() -> list[str]:
    return [
        "python3",
        WING_READONLY_PREFLIGHT_RUNNER_TOOL,
        "--url",
        WING_READONLY_PREFLIGHT_RUNNER_URL,
        "--channels",
        WING_READONLY_PREFLIGHT_RUNNER_CHANNELS,
        "--operator-confirm-channels",
        WING_READONLY_PREFLIGHT_RUNNER_CHANNELS,
        "--operator-confirm-session-ready",
        "--value-types",
        WING_READONLY_PREFLIGHT_RUNNER_VALUE_TYPES,
        "--require-physical-readback",
        "--artifact-root",
        WING_READONLY_PREFLIGHT_CAPTURE_ARTIFACT_ROOT,
        "--expected-plan",
        WING_READONLY_PREFLIGHT_PLAN_OUTPUT,
    ]


def wing_readonly_preflight_plan_command() -> list[str]:
    return [
        "python3",
        WING_READONLY_PREFLIGHT_RUNNER_TOOL,
        "--url",
        WING_READONLY_PREFLIGHT_RUNNER_URL,
        "--channels",
        WING_READONLY_PREFLIGHT_RUNNER_CHANNELS,
        "--operator-confirm-channels",
        WING_READONLY_PREFLIGHT_RUNNER_CHANNELS,
        "--operator-confirm-session-ready",
        "--value-types",
        WING_READONLY_PREFLIGHT_RUNNER_VALUE_TYPES,
        "--require-physical-readback",
        "--artifact-root",
        WING_READONLY_PREFLIGHT_CAPTURE_ARTIFACT_ROOT,
        "--plan-only",
        "--plan-output",
        WING_READONLY_PREFLIGHT_PLAN_OUTPUT,
    ]


def command_value_after_flag(command: Sequence[str], flag: str) -> str | None:
    try:
        index = list(command).index(flag)
    except ValueError:
        return None
    if index + 1 >= len(command):
        return None
    return str(command[index + 1])


def wing_readonly_preflight_review_command() -> list[str]:
    return [
        "python3",
        WING_READONLY_PREFLIGHT_REVIEW_TOOL,
        "--artifact-root",
        WING_READONLY_PREFLIGHT_REVIEW_ARTIFACT_ROOT,
        "--output",
        WING_READONLY_PREFLIGHT_REVIEW_OUTPUT,
        "--decision-template-output",
        WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_OUTPUT,
    ]


def wing_readonly_preflight_review_decision_command() -> list[str]:
    return [
        "python3",
        WING_READONLY_PREFLIGHT_REVIEW_DECISION_TOOL,
        "--artifact-root",
        WING_READONLY_PREFLIGHT_REVIEW_ARTIFACT_ROOT,
        "--decision-root",
        WING_READONLY_PREFLIGHT_REVIEW_DECISION_ROOT,
        "--output",
        WING_READONLY_PREFLIGHT_REVIEW_DECISION_OUTPUT,
    ]


def _command_string(command: Sequence[str]) -> str:
    return " ".join(str(part) for part in command)


def wing_readonly_preflight_handoff_packet_command() -> list[str]:
    return [
        "python3",
        WING_READONLY_PREFLIGHT_HANDOFF_PACKET_TOOL,
        "--url",
        WING_READONLY_PREFLIGHT_RUNNER_URL,
        "--channels",
        WING_READONLY_PREFLIGHT_RUNNER_CHANNELS,
        "--value-types",
        WING_READONLY_PREFLIGHT_RUNNER_VALUE_TYPES,
        "--artifact-root",
        WING_READONLY_PREFLIGHT_CAPTURE_ARTIFACT_ROOT,
        "--plan-output",
        WING_READONLY_PREFLIGHT_PLAN_OUTPUT,
        "--output",
        WING_READONLY_PREFLIGHT_HANDOFF_PACKET_OUTPUT,
    ]


def wing_readonly_preflight_review_manual_checklist_ids() -> list[str]:
    review_tool = load_wing_readonly_preflight_review_tool()
    if review_tool is None:
        return []
    try:
        checklist_factory = getattr(review_tool, "manual_review_checklist", None)
        if not callable(checklist_factory):
            return []
        checklist = checklist_factory()
    except Exception:
        return []
    if not isinstance(checklist, list):
        return []
    ids: list[str] = []
    for item in checklist:
        if not isinstance(item, Mapping):
            continue
        item_id = item.get("id")
        if isinstance(item_id, str) and item_id:
            ids.append(item_id)
    return ids


def load_wing_readonly_preflight_review_tool() -> object | None:
    review_tool_path = REPO_ROOT / WING_READONLY_PREFLIGHT_REVIEW_TOOL
    if not review_tool_path.is_file():
        return None
    try:
        spec = importlib.util.spec_from_file_location(
            "_automixer_wing_readonly_preflight_review_tool",
            review_tool_path,
        )
        if spec is None or spec.loader is None:
            return None
        review_tool = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(review_tool)
    except Exception:
        return None
    return review_tool


def wing_readonly_preflight_review_decision_template_required_fields() -> list[str]:
    review_tool = load_wing_readonly_preflight_review_tool()
    if review_tool is None:
        return []
    try:
        fields_factory = getattr(review_tool, "review_decision_template_required_fields", None)
        if not callable(fields_factory):
            return []
        fields = fields_factory()
    except Exception:
        return []
    return [str(item) for item in fields if isinstance(item, str) and item] if isinstance(fields, list) else []


FORBIDDEN_COMMAND_TOKENS = (
    "wing_operator_proposal_validation_runner.py",
    "wing_supervised_smoke_runner.py",
    "apply_operator_proposal",
    "set_fader",
    "set_channel_fader",
    "set_main_fader",
    "set_bus_fader",
    "set_dca_fader",
    "set_dca_level",
    "set_group_fader",
    "set_matrix_fader",
    "set_gain",
    "set_channel_gain",
    "set_trim",
    "set_input_trim",
    "set_channel_trim",
    "set_preamp_trim",
    "set_preamp_gain",
    "set_eq",
    "set_eq_band",
    "set_bus_eq_band",
    "set_bus_eq_on",
    "set_main_eq_band",
    "set_main_eq_on",
    "set_matrix_eq_band",
    "set_matrix_eq_on",
    "set_compressor",
    "set_compressor_gain",
    "set_bus_compressor",
    "set_bus_compressor_gain",
    "set_gate",
    "set_gate_on",
    "set_hpf",
    "set_bus_pan",
    "set_main_pan",
    "set_delay",
    "set_polarity",
    "set_mute",
    "set_channel_mute",
    "set_bus_mute",
    "set_dca_mute",
    "set_group_mute",
    "set_main_mute",
    "set_matrix_mute",
    "route_output",
    "route_multiple_outputs",
    "load_snap",
    "save_snap",
    "recall_snap",
    "store_snap",
    "restore_snapshot",
    "load_scene",
    "save_scene",
    "recall_scene",
    "restore_scene",
    "reset",
    "reset_channel_eq",
    "reset_channel_hpf",
    "reset_channel_dynamics",
    "reset_channel_processing",
    "reset_phase_delay",
    "mute_channel",
    "unmute_channel",
    "unmute",
    "mute",
    "set_channel_input",
    "set_channel_alt_input",
    "set_insert",
    "set_fx_model",
    "set_fx_on",
    "set_fx_mix",
    "set_fx_parameter",
    "set_fx_return_level",
    "set_send_level",
    "set_channel_send",
    "set_channel_main_send",
    "set_reverb_send",
    "connect_wing",
    "mixer_discovery",
    "WingClient",
)

FORBIDDEN_COMMAND_TOKEN_BOUNDARY = re.compile(r"[A-Za-z0-9]")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def compact_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def git_executable() -> str:
    configured = os.environ.get("AUTOMIXER_GIT_BIN")
    if configured:
        return configured
    if DEFAULT_HOMEBREW_GIT.exists():
        return str(DEFAULT_HOMEBREW_GIT)
    return "git"


def current_supervised_live_write_kinds() -> tuple[str, ...]:
    status = build_operator_mode_status(
        mode=MODE_SUPERVISED,
        connection_mode="wing",
        wing_boundary_active=True,
        auto_mode_unlocked=False,
    )
    capabilities = status.get("capabilities") if isinstance(status.get("capabilities"), Mapping) else {}
    return tuple(str(kind) for kind in capabilities.get("allowed_live_write_kinds", ()) if kind)


def current_supervised_live_write_surfaces() -> tuple[str, ...]:
    status = build_operator_mode_status(
        mode=MODE_SUPERVISED,
        connection_mode="wing",
        wing_boundary_active=True,
        auto_mode_unlocked=False,
    )
    capabilities = status.get("capabilities") if isinstance(status.get("capabilities"), Mapping) else {}
    return tuple(
        str(surface)
        for surface in capabilities.get("allowed_live_write_surfaces", ())
        if surface
    )


@dataclass(frozen=True)
class CommandCheck:
    name: str
    command: tuple[str, ...]
    cwd: str
    env: dict[str, str] | None = None
    timeout_sec: float | None = None

    def shell_display(self) -> str:
        return " ".join(self.command)


@dataclass
class CheckResult:
    name: str
    status: str
    started_at: str
    finished_at: str
    duration_sec: float
    command: str | None = None
    cwd: str | None = None
    returncode: int | None = None
    stdout_log: str | None = None
    stderr_log: str | None = None
    detail: dict[str, object] | None = None


def build_checks(
    *,
    repo_root: Path = REPO_ROOT,
    run_dir: Path | None = None,
    include_frontend: bool = True,
    include_package: bool = False,
    quick: bool = False,
    command_timeout_sec: float | None = DEFAULT_COMMAND_TIMEOUT_SEC,
) -> list[CommandCheck]:
    offline_artifact_root = (run_dir or DEFAULT_ARTIFACT_ROOT / "pending") / "offline_validation"
    wing_readonly_plan_output = wing_readonly_preflight_plan_path_for_run(run_dir)
    read_surface_capture_packet_output = (
        read_surface_physical_readback_capture_packet_path_for_run(run_dir)
    )
    read_surface_capture_template_output = (
        read_surface_physical_readback_capture_template_path_for_run(run_dir)
    )
    read_surface_artifact_review_queue_output = (
        read_surface_physical_readback_artifact_review_queue_path_for_run(run_dir)
    )
    read_surface_acceptance_template_output = (
        read_surface_physical_readback_acceptance_template_path_for_run(run_dir)
    )
    read_surface_fixture_closure_plan_output = (
        read_surface_physical_readback_fixture_closure_plan_path_for_run(run_dir)
    )
    handoff_packet_output = wing_readonly_preflight_handoff_packet_path_for_run(run_dir)
    gain_proof_root = gain_proof_artifact_root_for_run(run_dir)
    proof_report_path = proof_report_path_for_run(run_dir)
    checks = [
        CommandCheck(
            name="write_surface_static_scan",
            command=(
                sys.executable,
                "-m",
                "pytest",
                "tests/test_write_surface_inventory_static_scan.py",
                "-q",
                "--tb=short",
            ),
            cwd=str(repo_root),
            env={"PYTHONPATH": "backend"},
        ),
        CommandCheck(
            name="read_surface_fixture_evidence",
            command=(
                sys.executable,
                "-m",
                "pytest",
                "tests/test_read_surface_fixture_evidence.py",
                "tests/test_read_surface_physical_readback_capture_packet.py",
                "tests/test_read_surface_physical_readback_capture_template.py",
                "tests/test_read_surface_physical_readback_artifact_review_queue.py",
                "tests/test_read_surface_physical_readback_fixture_acceptance_template.py",
                "tests/test_read_surface_physical_readback_fixture_acceptance_ledger.py",
                "tests/test_read_surface_physical_readback_fixture_closure_plan.py",
                "tests/test_raw_mixer_state_physical_readback_contract.py",
                "tests/test_raw_mixer_state_physical_readback_artifact_intake.py",
                "tests/test_channel_basic_controls_physical_readback_contract.py",
                "tests/test_channel_basic_controls_physical_readback_artifact_intake.py",
                "tests/test_channel_processing_physical_readback_contract.py",
                "tests/test_channel_processing_physical_readback_artifact_intake.py",
                "tests/test_bus_group_main_dca_physical_readback_contract.py",
                "tests/test_bus_group_main_dca_physical_readback_artifact_intake.py",
                "tests/test_matrix_state_physical_readback_contract.py",
                "tests/test_matrix_state_physical_readback_artifact_intake.py",
                "tests/test_fx_slots_inserts_physical_readback_contract.py",
                "tests/test_fx_slots_inserts_physical_readback_artifact_intake.py",
                "tests/test_routing_sends_physical_readback_contract.py",
                "tests/test_routing_sends_physical_readback_artifact_intake.py",
                "tests/test_snapshots_scenes_physical_readback_contract.py",
                "tests/test_snapshots_scenes_physical_readback_artifact_intake.py",
                "tests/test_user_presets_physical_readback_contract.py",
                "tests/test_user_presets_physical_readback_artifact_intake.py",
                "tests/test_dlive_state_physical_readback_contract.py",
                "tests/test_dlive_state_physical_readback_artifact_intake.py",
                "tests/test_wing_readonly_preflight_evidence.py",
                "tests/test_wing_readonly_preflight_runner.py",
                "tests/test_wing_readonly_preflight_review_tool.py",
                "tests/test_wing_readonly_preflight_review_decision.py",
                "tests/test_wing_readonly_preflight_review_decision_tool.py",
                "tests/test_wing_readonly_preflight_handoff_packet.py",
                "-q",
                "--tb=short",
            ),
            cwd=str(repo_root),
            env={"PYTHONPATH": "backend"},
        ),
        CommandCheck(
            name="backend_pytest",
            command=(
                sys.executable,
                "-m",
                "pytest",
                "tests/test_offline_validation_harness.py" if quick else "tests/",
                "-q" if quick else "-x",
                "--tb=short",
                "-q",
            ),
            cwd=str(repo_root),
            env={"PYTHONPATH": "backend"},
        ),
        CommandCheck(
            name="offline_operator_validation_harness",
            command=(
                sys.executable,
                "tools/offline_operator_validation_harness.py",
                "--artifact-root",
                str(offline_artifact_root),
            ),
            cwd=str(repo_root),
        ),
        CommandCheck(
            name="wing_readonly_preflight_plan",
            command=(
                sys.executable,
                WING_READONLY_PREFLIGHT_RUNNER_TOOL,
                "--url",
                WING_READONLY_PREFLIGHT_RUNNER_URL,
                "--channels",
                WING_READONLY_PREFLIGHT_RUNNER_CHANNELS,
                "--operator-confirm-channels",
                WING_READONLY_PREFLIGHT_RUNNER_CHANNELS,
                "--operator-confirm-session-ready",
                "--value-types",
                WING_READONLY_PREFLIGHT_RUNNER_VALUE_TYPES,
                "--require-physical-readback",
                "--artifact-root",
                WING_READONLY_PREFLIGHT_CAPTURE_ARTIFACT_ROOT,
                "--plan-only",
                "--plan-output",
                str(wing_readonly_plan_output),
            ),
            cwd=str(repo_root),
        ),
        CommandCheck(
            name="read_surface_physical_readback_capture_packet",
            command=(
                sys.executable,
                READ_SURFACE_PHYSICAL_READBACK_CAPTURE_PACKET_TOOL,
                "--format",
                "json",
                "--output",
                str(read_surface_capture_packet_output),
                "--quiet",
            ),
            cwd=str(repo_root),
        ),
        CommandCheck(
            name="read_surface_physical_readback_capture_template",
            command=(
                sys.executable,
                READ_SURFACE_PHYSICAL_READBACK_CAPTURE_TEMPLATE_TOOL,
                "--format",
                "json",
                "--output",
                str(read_surface_capture_template_output),
                "--quiet",
            ),
            cwd=str(repo_root),
        ),
        CommandCheck(
            name="read_surface_physical_readback_artifact_review_queue",
            command=(
                sys.executable,
                READ_SURFACE_PHYSICAL_READBACK_ARTIFACT_REVIEW_QUEUE_TOOL,
                "--format",
                "json",
                "--output",
                str(read_surface_artifact_review_queue_output),
                "--quiet",
            ),
            cwd=str(repo_root),
        ),
        CommandCheck(
            name="read_surface_physical_readback_acceptance_template",
            command=(
                sys.executable,
                READ_SURFACE_PHYSICAL_READBACK_ACCEPTANCE_TEMPLATE_TOOL,
                "--output",
                str(read_surface_acceptance_template_output),
                "--quiet",
            ),
            cwd=str(repo_root),
        ),
        CommandCheck(
            name="read_surface_physical_readback_fixture_closure_plan",
            command=(
                sys.executable,
                READ_SURFACE_PHYSICAL_READBACK_FIXTURE_CLOSURE_PLAN_TOOL,
                "--format",
                "json",
                "--output",
                str(read_surface_fixture_closure_plan_output),
                "--quiet",
            ),
            cwd=str(repo_root),
        ),
        CommandCheck(
            name="wing_readonly_preflight_handoff_packet",
            command=(
                sys.executable,
                WING_READONLY_PREFLIGHT_HANDOFF_PACKET_TOOL,
                "--url",
                WING_READONLY_PREFLIGHT_RUNNER_URL,
                "--channels",
                WING_READONLY_PREFLIGHT_RUNNER_CHANNELS,
                "--value-types",
                WING_READONLY_PREFLIGHT_RUNNER_VALUE_TYPES,
                "--artifact-root",
                WING_READONLY_PREFLIGHT_CAPTURE_ARTIFACT_ROOT,
                "--plan-output",
                str(wing_readonly_plan_output),
                "--format",
                "json",
                "--output",
                str(handoff_packet_output),
                "--quiet",
            ),
            cwd=str(repo_root),
        ),
        CommandCheck(
            name="gain_dry_run_proof",
            command=(
                sys.executable,
                "tools/gain_proof_dry_run_runner.py",
                "--artifact-root",
                str(gain_proof_root),
                "--channel",
                "1",
                "--initial-gain-db",
                "1.0",
                "--delta-db",
                "-0.25",
                "--cooldown-sec",
                "2.0",
                "--reason",
                GAIN_DRY_RUN_REASON,
            ),
            cwd=str(repo_root),
            env={"PYTHONPATH": "backend"},
        ),
        CommandCheck(
            name="supervised_proof_artifact_report",
            command=(
                sys.executable,
                "tools/evaluate_supervised_proof_artifacts.py",
                "--artifact-root",
                str(offline_artifact_root),
                "--artifact-root",
                str(gain_proof_root),
                "--require-physical",
                "--output",
                str(proof_report_path),
                "--quiet",
            ),
            cwd=str(repo_root),
            env={"PYTHONPATH": "backend"},
        ),
        CommandCheck(
            name="memory_lint",
            command=(sys.executable, "tools/memory/memory_lint.py"),
            cwd=str(repo_root),
        ),
    ]

    if include_frontend:
        checks.extend(
            [
                CommandCheck(
                    name="frontend_typescript",
                    command=("npx", "tsc", "--noEmit"),
                    cwd=str(repo_root / "frontend"),
                ),
                CommandCheck(
                    name="frontend_unit",
                    command=("npm", "test", "--", "--watch=false"),
                    cwd=str(repo_root / "frontend"),
                ),
                CommandCheck(
                    name="frontend_e2e",
                    command=("npm", "run", "test:e2e"),
                    cwd=str(repo_root / "frontend"),
                ),
                CommandCheck(
                    name="frontend_build",
                    command=("npm", "run", "build"),
                    cwd=str(repo_root / "frontend"),
                ),
            ]
        )
        if include_package:
            checks.append(
                CommandCheck(
                    name="frontend_dist_package",
                    command=("npm", "run", "dist"),
                    cwd=str(repo_root / "frontend"),
                )
            )

    checks.append(
        CommandCheck(
            name="diff_check",
            command=(git_executable(), "diff", "--check"),
            cwd=str(repo_root),
        )
    )
    if command_timeout_sec is not None:
        checks = [replace(check, timeout_sec=command_timeout_sec) for check in checks]
    return checks


def proof_report_path_for_run(run_dir: Path | None) -> Path:
    return (run_dir or DEFAULT_ARTIFACT_ROOT / "pending") / "supervised_proof_artifact_report.json"


def wing_readonly_preflight_plan_path_for_run(run_dir: Path | None) -> Path:
    return (run_dir or DEFAULT_ARTIFACT_ROOT / "pending") / "wing_readonly_preflight_plan.json"


def read_surface_physical_readback_capture_packet_path_for_run(run_dir: Path | None) -> Path:
    return (
        (run_dir or DEFAULT_ARTIFACT_ROOT / "pending")
        / "read_surface_physical_readback_capture_packet.json"
    )


def read_surface_physical_readback_capture_template_path_for_run(
    run_dir: Path | None,
) -> Path:
    return (
        (run_dir or DEFAULT_ARTIFACT_ROOT / "pending")
        / "read_surface_physical_readback_capture_template.json"
    )


def read_surface_physical_readback_artifact_review_queue_path_for_run(
    run_dir: Path | None,
) -> Path:
    return (
        (run_dir or DEFAULT_ARTIFACT_ROOT / "pending")
        / "read_surface_physical_readback_artifact_review_queue.json"
    )


def read_surface_physical_readback_acceptance_template_path_for_run(
    run_dir: Path | None,
) -> Path:
    return (
        (run_dir or DEFAULT_ARTIFACT_ROOT / "pending")
        / "read_surface_physical_readback_acceptance_template.json"
    )


def read_surface_physical_readback_fixture_closure_plan_path_for_run(
    run_dir: Path | None,
) -> Path:
    return (
        (run_dir or DEFAULT_ARTIFACT_ROOT / "pending")
        / "read_surface_physical_readback_fixture_closure_plan.json"
    )


def gain_proof_artifact_root_for_run(run_dir: Path | None) -> Path:
    return (run_dir or DEFAULT_ARTIFACT_ROOT / "pending") / "gain_proof_dry_run"


def wing_readonly_preflight_handoff_packet_path_for_run(run_dir: Path | None) -> Path:
    return (
        (run_dir or DEFAULT_ARTIFACT_ROOT / "pending")
        / "wing_readonly_preflight_handoff_packet.json"
    )


def validate_no_live_commands(checks: Iterable[CommandCheck]) -> None:
    violations: list[str] = []
    for check in checks:
        command_text = " ".join((check.name, check.shell_display())).lower()
        for token in FORBIDDEN_COMMAND_TOKENS:
            if _contains_forbidden_command_token(command_text, token):
                violations.append(f"{check.name}:{token}")
    if violations:
        joined = ", ".join(sorted(violations))
        raise ValueError(f"no-WING release checks include forbidden live command tokens: {joined}")


def _contains_forbidden_command_token(command_text: str, token: str) -> bool:
    normalized_token = token.lower()
    start = 0
    while True:
        index = command_text.find(normalized_token, start)
        if index == -1:
            return False

        before = command_text[index - 1] if index > 0 else ""
        after_index = index + len(normalized_token)
        after = command_text[after_index] if after_index < len(command_text) else ""
        before_is_word = bool(before and FORBIDDEN_COMMAND_TOKEN_BOUNDARY.match(before))
        after_is_word = bool(after and FORBIDDEN_COMMAND_TOKEN_BOUNDARY.match(after))

        if not before_is_word and not after_is_word:
            return True

        start = index + 1


def validate_repo_local_commands(
    checks: Iterable[CommandCheck],
    *,
    repo_root: Path = REPO_ROOT,
) -> None:
    """Fail the no-WING plan if local command paths leave this checkout."""

    try:
        resolved_repo_root = repo_root.resolve()
    except OSError as exc:
        raise ValueError(f"no-WING release check repo root is not resolvable: {repo_root}") from exc

    violations: list[str] = []
    for check in checks:
        cwd_path = _resolve_repo_scoped_path(check.cwd, resolved_repo_root)
        if not _is_path_inside(cwd_path, resolved_repo_root):
            violations.append(f"{check.name}:cwd:{check.cwd}")

        skip_next_path_token = False
        for token in check.command[1:]:
            if skip_next_path_token:
                skip_next_path_token = False
                continue
            if token in REPO_LOCAL_COMMAND_OUTPUT_FLAGS:
                skip_next_path_token = True
                continue
            if not _looks_like_local_path_token(token):
                continue
            token_path = _resolve_repo_scoped_path(token, resolved_repo_root)
            if not _is_path_inside(token_path, resolved_repo_root):
                violations.append(f"{check.name}:command_path:{token}")

        for key, value in (check.env or {}).items():
            if not _looks_like_local_path_token(value):
                continue
            value_path = _resolve_repo_scoped_path(value, resolved_repo_root)
            if not _is_path_inside(value_path, resolved_repo_root):
                violations.append(f"{check.name}:env_path:{key}")

    if violations:
        joined = ", ".join(sorted(violations))
        raise ValueError(f"no-WING release checks leave the Automixer repo boundary: {joined}")


def _resolve_repo_scoped_path(value: str | os.PathLike[str], repo_root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    try:
        return path.resolve()
    except OSError:
        return path.absolute()


def _is_path_inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _looks_like_local_path_token(value: object) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text:
        return False
    if text in {".", ".."} or text.startswith(("~", "./", "../", "/")):
        return True
    return "/" in text or "\\" in text


def _process_command_line(pid: int | str) -> tuple[str | None, str | None]:
    try:
        completed = subprocess.run(
            ("ps", "-p", str(pid), "-o", "command="),
            capture_output=True,
            check=False,
            text=True,
            timeout=1.0,
        )
    except FileNotFoundError:
        return None, "ps_not_available"
    except subprocess.TimeoutExpired:
        return None, "ps_timeout"
    except OSError as exc:
        return None, f"ps_error:{exc.__class__.__name__}"
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        if stderr:
            return None, f"ps_failed:{stderr}"
        return None, f"ps_failed:returncode_{completed.returncode}"
    command_line = completed.stdout.strip()
    return (command_line or None), None


def _listener_processes_for_port(port: int) -> tuple[list[dict[str, object]], str | None]:
    command = (
        "lsof",
        "-nP",
        f"-iTCP:{port}",
        "-sTCP:LISTEN",
        "-F",
        "pcn",
    )
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
            timeout=1.0,
        )
    except FileNotFoundError:
        return [], "lsof_not_available"
    except subprocess.TimeoutExpired:
        return [], "lsof_timeout"
    except OSError as exc:
        return [], f"lsof_error:{exc.__class__.__name__}"

    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        if stderr:
            return [], f"lsof_failed:{stderr}"
        return [], f"lsof_failed:returncode_{completed.returncode}"

    processes: list[dict[str, object]] = []
    current: dict[str, object] = {}
    for raw_line in completed.stdout.splitlines():
        if not raw_line:
            continue
        field = raw_line[0]
        value = raw_line[1:]
        if field == "p":
            if current:
                processes.append(current)
            current = {"pid": int(value) if value.isdigit() else value}
        elif field == "c":
            current["command"] = value
        elif field == "n":
            current["name"] = value
    if current:
        processes.append(current)
    for process in processes:
        pid = process.get("pid")
        if pid is None:
            continue
        command_line, command_line_error = _process_command_line(pid)
        if command_line:
            process["command_line"] = command_line
        if command_line_error:
            process["command_line_error"] = command_line_error
    return processes, None


def _backend_port_markdown_details(result: CheckResult) -> list[str]:
    if result.name != "backend_port_free":
        return []
    detail = result.detail or {}
    if not detail.get("listener_detected"):
        return []
    lines = []
    listener_processes = detail.get("listener_processes")
    if isinstance(listener_processes, list) and listener_processes:
        lines.append(
            "  - listener_processes: "
            f"`{json.dumps(listener_processes, ensure_ascii=False, sort_keys=True)}`"
        )
    listener_process_error = detail.get("listener_process_error")
    if listener_process_error:
        lines.append(f"  - listener_process_error: `{listener_process_error}`")
    listener_confirmation_checklist = detail.get("listener_confirmation_checklist")
    if isinstance(listener_confirmation_checklist, list) and listener_confirmation_checklist:
        lines.append(
            "  - listener_confirmation_checklist: "
            f"`{json.dumps(listener_confirmation_checklist, ensure_ascii=False)}`"
        )
    for field in (
        "listener_confirmation_path",
        "listener_confirmation_status",
        "intended_backend_confirmed",
        "matched_listener_pid",
    ):
        if field in detail:
            lines.append(f"  - {field}: `{detail[field]}`")
    listener_confirmation_violations = detail.get("listener_confirmation_violations")
    if (
        isinstance(listener_confirmation_violations, list)
        and listener_confirmation_violations
    ):
        lines.append(
            "  - listener_confirmation_violations: "
            f"`{json.dumps(listener_confirmation_violations, ensure_ascii=False)}`"
        )
    for field in (
        "preflight_blocked_until_listener_confirmed",
        "intended_backend_confirmation_required",
        "safe_to_stop_listener_by_gate",
        "live_authority_granted",
        "fail_closed_reason",
    ):
        if field in detail:
            lines.append(f"  - {field}: `{detail[field]}`")
    operator_next_step = detail.get("operator_next_step")
    if operator_next_step:
        lines.append(f"  - operator_next_step: `{operator_next_step}`")
    return lines


def _normalize_listener_pid(value: object) -> int | str | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    if isinstance(value, str) and value:
        return value
    return None


def _backend_listener_confirmation_detail(
    *,
    path: Path,
    host: str,
    port: int,
    listener_processes: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    payload, error = _json_object_from_path(path)
    detail: dict[str, object] = {
        "listener_confirmation_path": str(path),
        "listener_confirmation_status": "invalid",
        "listener_confirmation_violations": [],
        "safe_to_stop_listener_by_gate": False,
        "live_authority_granted": False,
    }
    violations: list[str] = []
    if error:
        violations.append(f"listener_confirmation_{error}")
        detail["listener_confirmation_violations"] = violations
        return detail
    assert payload is not None

    if payload.get("type") != BACKEND_PORT_LISTENER_CONFIRMATION_ARTIFACT_TYPE:
        violations.append("listener_confirmation_type_mismatch")
    if payload.get("status") != BACKEND_PORT_LISTENER_CONFIRMATION_ACCEPTED_STATUS:
        violations.append("listener_confirmation_status_not_accepted")
    if payload.get("host") != host:
        violations.append("listener_confirmation_host_mismatch")
    if payload.get("port") != port:
        violations.append("listener_confirmation_port_mismatch")
    if payload.get("confirms_intended_backend") is not True:
        violations.append("listener_confirmation_intended_backend_not_true")
    if payload.get("safe_to_stop_listener_by_gate") is not False:
        violations.append("listener_confirmation_safe_to_stop_not_false")
    if payload.get("live_authority_granted") is not False:
        violations.append("listener_confirmation_live_authority_not_false")
    for field in ("operator_reviewed_at", "operator_name"):
        if not str(payload.get(field) or "").strip():
            violations.append(f"listener_confirmation_{field}_missing")

    reviewed_pid = _normalize_listener_pid(payload.get("reviewed_listener_pid"))
    reviewed_command_line = str(payload.get("reviewed_listener_command_line") or "")
    if reviewed_pid is None:
        violations.append("listener_confirmation_reviewed_listener_pid_missing")
    if not reviewed_command_line:
        violations.append("listener_confirmation_reviewed_listener_command_line_missing")

    matching_process = None
    for process in listener_processes:
        if _normalize_listener_pid(process.get("pid")) == reviewed_pid:
            matching_process = process
            break
    if matching_process is None:
        violations.append("listener_confirmation_pid_not_current_listener")
    else:
        current_command_line = str(matching_process.get("command_line") or "")
        if not current_command_line:
            violations.append("listener_confirmation_current_command_line_missing")
        elif current_command_line != reviewed_command_line:
            violations.append("listener_confirmation_reviewed_command_line_mismatch")
        detail["matched_listener_pid"] = reviewed_pid

    if not violations:
        detail["listener_confirmation_status"] = "accepted"
        detail["intended_backend_confirmed"] = True
    detail["listener_confirmation_violations"] = violations
    return detail


def check_port_free(
    port: int = DEFAULT_BACKEND_PORT,
    host: str = "127.0.0.1",
    listener_confirmation_path: Path | None = None,
) -> CheckResult:
    started = datetime.now(timezone.utc)
    detail: dict[str, object] = {"host": host, "port": port, "policy": "port_must_not_have_unexpected_listener"}
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        connected = sock.connect_ex((host, port)) == 0
    finished = datetime.now(timezone.utc)
    detail["listener_detected"] = connected
    if connected:
        listener_processes, listener_process_error = _listener_processes_for_port(port)
        detail["listener_processes"] = listener_processes
        detail["listener_confirmation_checklist"] = list(
            BACKEND_PORT_LISTENER_CONFIRMATION_CHECKLIST
        )
        detail["preflight_blocked_until_listener_confirmed"] = True
        detail["intended_backend_confirmation_required"] = True
        detail["safe_to_stop_listener_by_gate"] = False
        detail["fail_closed_reason"] = "backend_port_listener_requires_operator_confirmation"
        detail["operator_next_step"] = (
            "Stop or reconfigure the unexpected local listener only after "
            "confirming it is not the intended Automixer backend, then rerun "
            "the no-WING gate and WING read-only preflight."
        )
        if listener_process_error:
            detail["listener_process_error"] = listener_process_error
        if listener_confirmation_path is not None:
            confirmation_detail = _backend_listener_confirmation_detail(
                path=listener_confirmation_path,
                host=host,
                port=port,
                listener_processes=listener_processes,
            )
            detail.update(confirmation_detail)
            if confirmation_detail.get("listener_confirmation_status") == "accepted":
                detail["policy"] = (
                    "port_must_be_free_or_operator_confirmed_intended_backend"
                )
                detail["preflight_blocked_until_listener_confirmed"] = False
                detail["intended_backend_confirmation_required"] = False
                detail.pop("fail_closed_reason", None)
                detail["operator_next_step"] = (
                    "Proceed only with the operator-confirmed intended backend "
                    "session and WING read-only preflight; this gate still grants "
                    "no live authority."
                )
            else:
                detail["operator_next_step"] = (
                    "Fix or replace the backend listener confirmation artifact, "
                    "or clear the unexpected listener after operator review, then "
                    "rerun the no-WING gate."
                )
    elif listener_confirmation_path is not None:
        detail["listener_confirmation_path"] = str(listener_confirmation_path)
        detail["listener_confirmation_status"] = "not_used_no_listener_detected"
        detail["safe_to_stop_listener_by_gate"] = False
        detail["live_authority_granted"] = False
    status = "failed" if connected else "passed"
    if (
        connected
        and listener_confirmation_path is not None
        and detail.get("listener_confirmation_status") == "accepted"
    ):
        status = "passed"
    return CheckResult(
        name="backend_port_free",
        status=status,
        started_at=started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        finished_at=finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration_sec=round((finished - started).total_seconds(), 3),
        detail=detail,
    )


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _coerce_timeout_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return value


def run_command_check(check: CommandCheck, log_dir: Path) -> CheckResult:
    started = datetime.now(timezone.utc)
    env = os.environ.copy()
    if check.env:
        env.update(check.env)
    detail: dict[str, object] | None = None
    try:
        completed = subprocess.run(
            check.command,
            cwd=check.cwd,
            env=env,
            text=True,
            capture_output=True,
            check=False,
            timeout=check.timeout_sec,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        status = "passed" if completed.returncode == 0 else "failed"
        returncode = completed.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = _coerce_timeout_output(exc.stdout)
        stderr = _coerce_timeout_output(exc.stderr)
        timeout_note = f"Command timed out after {check.timeout_sec} seconds."
        stderr = f"{stderr.rstrip()}\n{timeout_note}\n" if stderr else f"{timeout_note}\n"
        status = "timeout"
        returncode = None
        detail = {
            "timeout_sec": check.timeout_sec,
            "reason": "command_timeout",
        }
    finished = datetime.now(timezone.utc)

    stdout_log = log_dir / f"{check.name}.stdout.log"
    stderr_log = log_dir / f"{check.name}.stderr.log"
    _write_text(stdout_log, stdout)
    _write_text(stderr_log, stderr)

    return CheckResult(
        name=check.name,
        status=status,
        started_at=started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        finished_at=finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration_sec=round((finished - started).total_seconds(), 3),
        command=check.shell_display(),
        cwd=check.cwd,
        returncode=returncode,
        stdout_log=str(stdout_log),
        stderr_log=str(stderr_log),
        detail=detail,
    )


def _json_object_from_path(path: Path) -> tuple[Mapping[str, Any] | None, str | None]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, "missing_report"
    except json.JSONDecodeError:
        return None, "invalid_json"
    if not isinstance(payload, Mapping):
        return None, "json_payload_not_object"
    return payload, None


def _declared_pytest_functions(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name.startswith("test_")
    }


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in value if item] if isinstance(value, list) else []


def _string_list_map(value: Any) -> tuple[dict[str, list[str]], bool]:
    if not isinstance(value, Mapping):
        return {}, False
    normalized: dict[str, list[str]] = {}
    for key, raw_items in value.items():
        if not isinstance(key, str) or not key:
            return {}, False
        if not isinstance(raw_items, list):
            return {}, False
        items = [str(item) for item in raw_items if isinstance(item, str) and item]
        if len(items) != len(raw_items):
            return {}, False
        if items:
            normalized[key] = items
    return normalized, True


def validate_supervised_proof_report_gap_rollup(report_path: Path) -> CheckResult:
    started = datetime.now(timezone.utc)
    payload, load_error = _json_object_from_path(report_path)
    detail: dict[str, object] = {
        "report_path": str(report_path),
        "policy": "supervised_proof_report_must_expose_read_only_autonomy_gap_rollup",
    }
    violations: list[str] = []

    if load_error is not None:
        violations.append(load_error)
    else:
        assert payload is not None
        summary = payload.get("summary") if isinstance(payload.get("summary"), Mapping) else {}
        rollup = summary.get("autonomy_gap_rollup") if isinstance(summary.get("autonomy_gap_rollup"), Mapping) else {}
        mode_gate = (
            payload.get("autonomy_mode_transition_gate")
            if isinstance(payload.get("autonomy_mode_transition_gate"), Mapping)
            else {}
        )
        gate_summary = mode_gate.get("summary") if isinstance(mode_gate.get("summary"), Mapping) else {}

        expected_false = ("live_write", "live_widening_allowed", "full_autonomous_ready")
        for key in expected_false:
            if rollup.get(key) is not False:
                violations.append(f"autonomy_gap_rollup_{key}_not_false")
        if rollup.get("read_only") is not True:
            violations.append("autonomy_gap_rollup_read_only_not_true")
        if rollup.get("full_autonomous_blocked") is not True:
            violations.append("autonomy_gap_rollup_full_autonomous_blocked_not_true")
        if rollup.get("source") != AUTONOMY_GAP_ROLLUP_SOURCE:
            violations.append("autonomy_gap_rollup_source_not_auto_mode_gate_row")
        claim = rollup.get("claim")
        if not isinstance(claim, str) or "Full Autonomous remains blocked" not in claim:
            violations.append("autonomy_gap_rollup_claim_does_not_preserve_auto_block")
        if not isinstance(claim, str) or "does not widen" not in claim:
            violations.append("autonomy_gap_rollup_claim_does_not_preserve_live_boundary")
        next_required_evidence = rollup.get("next_required_evidence")
        if not isinstance(next_required_evidence, list) or not next_required_evidence:
            violations.append("autonomy_gap_rollup_next_required_evidence_missing")
        rollup_counts = rollup.get("full_auto_gap_bucket_counts")
        if not isinstance(rollup_counts, Mapping):
            violations.append("autonomy_gap_rollup_bucket_counts_missing")
            rollup_counts = {}
        evidence_by_family, evidence_by_family_valid = _string_list_map(
            rollup.get("evidence_requirement_ids_by_action_family")
        )
        next_by_family, next_by_family_valid = _string_list_map(
            rollup.get("next_required_evidence_by_action_family")
        )
        if not evidence_by_family_valid:
            violations.append("autonomy_gap_rollup_evidence_by_family_missing")
        if not next_by_family_valid:
            violations.append("autonomy_gap_rollup_next_by_family_missing")
        if int(rollup_counts.get("evidence_requirement") or 0) > 0 and not evidence_by_family:
            violations.append("autonomy_gap_rollup_evidence_by_family_empty")
        if int(rollup_counts.get("evidence_requirement") or 0) > 0 and not next_by_family:
            violations.append("autonomy_gap_rollup_next_by_family_empty")
        required_gain_evidence_components = _required_gain_evidence_components()
        gain_evidence_components = evidence_by_family.get("gain", [])
        if gain_evidence_components != required_gain_evidence_components:
            violations.append("autonomy_gap_rollup_gain_evidence_components_mismatch")
        if gate_summary.get("autonomous_apply_allowed_modes") != []:
            violations.append("mode_gate_autonomous_apply_modes_not_empty")
        blocked_modes = gate_summary.get("blocked_modes")
        if not isinstance(blocked_modes, list):
            blocked_modes = []
        if "auto" not in blocked_modes:
            violations.append("mode_gate_auto_not_blocked")
        supervised_write_kinds = rollup.get("supervised_current_live_write_kinds")
        if not isinstance(supervised_write_kinds, list):
            supervised_write_kinds = []
        supervised_write_surfaces = rollup.get("supervised_current_live_write_surfaces")
        if not isinstance(supervised_write_surfaces, list):
            supervised_write_surfaces = []
        unexpected_write_kinds = [
            str(kind)
            for kind in supervised_write_kinds
            if str(kind) not in current_supervised_live_write_kinds()
        ]
        unexpected_write_surfaces = [
            str(surface)
            for surface in supervised_write_surfaces
            if str(surface) not in current_supervised_live_write_surfaces()
        ]
        if unexpected_write_kinds:
            violations.append("autonomy_gap_rollup_unexpected_supervised_live_write_kinds")
        if unexpected_write_surfaces:
            violations.append("autonomy_gap_rollup_unexpected_supervised_live_write_surfaces")

        detail["autonomy_gap_rollup_present"] = bool(rollup)
        detail["full_auto_gap_bucket_counts"] = dict(rollup_counts)
        detail["evidence_requirement_ids_by_action_family"] = evidence_by_family
        detail["required_gain_evidence_components"] = required_gain_evidence_components
        detail["next_required_evidence_by_action_family"] = next_by_family
        detail["supervised_current_live_write_kinds"] = list(supervised_write_kinds)
        detail["policy_supervised_live_write_kinds"] = list(current_supervised_live_write_kinds())
        detail["supervised_current_live_write_surfaces"] = list(supervised_write_surfaces)
        detail["policy_supervised_live_write_surfaces"] = list(
            current_supervised_live_write_surfaces()
        )
        detail["unexpected_supervised_live_write_surfaces"] = unexpected_write_surfaces
        detail["unexpected_supervised_live_write_kinds"] = unexpected_write_kinds

    detail["violations"] = violations
    finished = datetime.now(timezone.utc)
    return CheckResult(
        name="supervised_proof_report_gap_rollup_schema",
        status="passed" if not violations else "failed",
        started_at=started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        finished_at=finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration_sec=round((finished - started).total_seconds(), 3),
        detail=detail,
    )


def validate_read_surface_fixture_evidence_snapshot() -> CheckResult:
    started = datetime.now(timezone.utc)
    detail: dict[str, object] = {
        "policy": "read_surface_fixture_evidence_must_remain_read_only_and_non_autonomous",
    }
    violations: list[str] = []

    try:
        snapshot = build_read_surface_fixture_evidence_snapshot()
    except Exception as exc:  # pragma: no cover - defensive release-check guard
        snapshot = {}
        violations.append(f"snapshot_build_error:{type(exc).__name__}")

    safety = snapshot.get("safety") if isinstance(snapshot.get("safety"), Mapping) else {}
    summary = snapshot.get("summary") if isinstance(snapshot.get("summary"), Mapping) else {}
    status_counts = summary.get("status_counts") if isinstance(summary.get("status_counts"), Mapping) else {}
    surfaces = snapshot.get("surfaces") if isinstance(snapshot.get("surfaces"), list) else []
    if not isinstance(snapshot.get("surfaces"), list):
        violations.append("read_surface_fixture_surfaces_missing")

    def summary_int(key: str) -> int:
        try:
            return int(summary.get(key) or 0)
        except (TypeError, ValueError):
            violations.append(f"read_surface_fixture_{key}_not_integer")
            return -1

    expected_status_counts = {status: 0 for status in sorted(FIXTURE_EVIDENCE_STATUSES)}
    expected_physical_missing: list[str] = []
    expected_surface_with_fixture_count = 0
    declared_tests_by_path: dict[str, set[str]] = {}
    declared_fixture_path_count = 0
    declared_test_name_count = 0
    for index, row in enumerate(surfaces):
        if not isinstance(row, Mapping):
            violations.append(f"read_surface_fixture_row_{index}_not_object")
            continue
        surface_id = str(row.get("surface_id") or f"index_{index}")
        if not isinstance(row.get("surface_id"), str) or not row.get("surface_id"):
            violations.append(f"read_surface_fixture_row_{surface_id}_surface_id_missing")
        if not isinstance(row.get("target_console_surface"), str) or not row.get("target_console_surface"):
            violations.append(f"read_surface_fixture_row_{surface_id}_target_console_surface_missing")
        status = str(row.get("status") or "")
        if not status:
            violations.append(f"read_surface_fixture_row_{surface_id}_status_missing")
        else:
            if status not in FIXTURE_EVIDENCE_STATUSES:
                violations.append(f"read_surface_fixture_row_{surface_id}_status_unknown")
            expected_status_counts[status] = expected_status_counts.get(status, 0) + 1
            if status != "missing_fixture_evidence":
                expected_surface_with_fixture_count += 1
        if row.get("full_autonomous_ready") is not False:
            violations.append(f"read_surface_fixture_row_{surface_id}_full_autonomous_ready_not_false")
        remaining_gaps = row.get("remaining_gaps")
        if not isinstance(remaining_gaps, list):
            remaining_gaps = []
            violations.append(f"read_surface_fixture_row_{surface_id}_remaining_gaps_missing")
        if "physical_readback_fixture_missing" in remaining_gaps:
            expected_physical_missing.append(surface_id)
        covered_contracts_value = row.get("covered_contracts")
        if not isinstance(covered_contracts_value, list):
            covered_contracts: list[object] = []
            violations.append(f"read_surface_fixture_row_{surface_id}_covered_contracts_not_list")
        else:
            covered_contracts = list(covered_contracts_value)
        invalid_contracts = [
            value
            for value in covered_contracts
            if not isinstance(value, str) or not value.strip()
        ]
        if invalid_contracts:
            violations.append(f"read_surface_fixture_row_{surface_id}_covered_contracts_invalid")
        invalid_gaps = [
            value
            for value in remaining_gaps
            if not isinstance(value, str) or not value.strip()
        ]
        if invalid_gaps:
            violations.append(f"read_surface_fixture_row_{surface_id}_remaining_gaps_invalid")
        fixture_paths_value = row.get("fixture_paths")
        if not isinstance(fixture_paths_value, list):
            fixture_paths: list[object] = []
            violations.append(f"read_surface_fixture_row_{surface_id}_fixture_paths_not_list")
        else:
            fixture_paths = list(fixture_paths_value)
        test_names_value = row.get("test_names")
        if not isinstance(test_names_value, list):
            test_names: list[object] = []
            violations.append(f"read_surface_fixture_row_{surface_id}_test_names_not_list")
        else:
            test_names = list(test_names_value)
        declared_fixture_path_count += len(fixture_paths)
        declared_test_name_count += len(test_names)
        if status == "missing_fixture_evidence":
            if covered_contracts:
                violations.append(f"read_surface_fixture_row_{surface_id}_missing_status_has_covered_contracts")
            if fixture_paths:
                violations.append(f"read_surface_fixture_row_{surface_id}_missing_status_has_fixture_paths")
            if test_names:
                violations.append(f"read_surface_fixture_row_{surface_id}_missing_status_has_test_names")
            continue
        if not covered_contracts:
            violations.append(f"read_surface_fixture_row_{surface_id}_covered_contracts_missing")
        has_physical_contract = "physical_readback_fixture" in covered_contracts
        has_physical_gap = "physical_readback_fixture_missing" in remaining_gaps
        has_hardware_gap = "hardware_parity_unproven" in remaining_gaps
        if status == "partial_fixture_evidence":
            if has_physical_contract:
                violations.append(f"read_surface_fixture_row_{surface_id}_partial_status_has_physical_contract")
            if not has_physical_gap:
                violations.append(f"read_surface_fixture_row_{surface_id}_partial_status_missing_physical_gap")
            if not has_hardware_gap:
                violations.append(f"read_surface_fixture_row_{surface_id}_partial_status_missing_hardware_gap")
        if status == "fixture_evidence_present":
            if not has_physical_contract:
                violations.append(f"read_surface_fixture_row_{surface_id}_present_status_missing_physical_contract")
            if has_physical_gap:
                violations.append(f"read_surface_fixture_row_{surface_id}_present_status_has_physical_gap")
            if has_hardware_gap:
                violations.append(f"read_surface_fixture_row_{surface_id}_present_status_has_hardware_gap")
        if not fixture_paths:
            violations.append(f"read_surface_fixture_row_{surface_id}_fixture_paths_missing")
        if not test_names:
            violations.append(f"read_surface_fixture_row_{surface_id}_test_names_missing")
        available_tests: set[str] = set()
        for value in fixture_paths:
            if not isinstance(value, str) or not value.strip():
                violations.append(f"read_surface_fixture_row_{surface_id}_fixture_path_invalid")
                continue
            relative_path = value.strip()
            path_parts = Path(relative_path).parts
            if Path(relative_path).is_absolute() or ".." in path_parts or not path_parts or path_parts[0] != "tests":
                violations.append(f"read_surface_fixture_row_{surface_id}_fixture_path_not_tests")
                continue
            fixture_path = REPO_ROOT / relative_path
            try:
                resolved_fixture_path = fixture_path.resolve()
                resolved_repo_root = REPO_ROOT.resolve()
                if resolved_fixture_path != resolved_repo_root and resolved_repo_root not in resolved_fixture_path.parents:
                    violations.append(f"read_surface_fixture_row_{surface_id}_fixture_path_escapes_repo")
                    continue
            except OSError:
                violations.append(f"read_surface_fixture_row_{surface_id}_fixture_path_unresolvable")
                continue
            if not fixture_path.exists():
                violations.append(f"read_surface_fixture_row_{surface_id}_fixture_path_missing")
                continue
            if not fixture_path.is_file():
                violations.append(f"read_surface_fixture_row_{surface_id}_fixture_path_not_file")
                continue
            if relative_path not in declared_tests_by_path:
                try:
                    declared_tests_by_path[relative_path] = _declared_pytest_functions(fixture_path)
                except (OSError, SyntaxError, UnicodeDecodeError):
                    declared_tests_by_path[relative_path] = set()
                    violations.append(f"read_surface_fixture_row_{surface_id}_fixture_path_unreadable")
            available_tests.update(declared_tests_by_path[relative_path])
        for value in test_names:
            if not isinstance(value, str) or not value.startswith("test_"):
                violations.append(f"read_surface_fixture_row_{surface_id}_test_name_invalid")
                continue
            if value not in available_tests:
                violations.append(f"read_surface_fixture_row_{surface_id}_test_name_missing")

    normalized_status_counts = {status: 0 for status in sorted(FIXTURE_EVIDENCE_STATUSES)}
    for key, value in status_counts.items():
        status_key = str(key)
        try:
            normalized_status_counts[status_key] = int(value)
        except (TypeError, ValueError):
            normalized_status_counts[status_key] = -1
            violations.append(f"read_surface_fixture_status_count_{status_key}_not_integer")
    physical_missing = summary.get("physical_fixture_missing_surface_ids")
    if not isinstance(physical_missing, list):
        physical_missing = []
        violations.append("physical_fixture_missing_surface_ids_missing")
    expected_next_physical_readback_surface_id = (
        expected_physical_missing[0] if expected_physical_missing else None
    )
    next_physical_readback_surface_id = summary.get("next_physical_readback_surface_id")
    if next_physical_readback_surface_id != expected_next_physical_readback_surface_id:
        violations.append("read_surface_fixture_next_physical_readback_surface_id_mismatch")

    expected_true = ("read_only",)
    for key in expected_true:
        if safety.get(key) is not True:
            violations.append(f"read_surface_fixture_{key}_not_true")
    expected_false = (
        "live_write",
        "mutates_console_state",
        "queries_console",
        "executes_tests",
        "full_autonomous_ready",
    )
    for key in expected_false:
        if safety.get(key) is not False:
            violations.append(f"read_surface_fixture_{key}_not_false")
    if snapshot.get("type") != "read_surface_fixture_evidence":
        violations.append("read_surface_fixture_type_mismatch")
    if snapshot.get("status") == "fixture_evidence_complete" and physical_missing:
        violations.append("fixture_status_complete_but_physical_gaps_remain")
    if not physical_missing:
        violations.append("physical_readback_gaps_not_reported")
    summary_surface_count = summary_int("surface_count")
    summary_surface_with_fixture_count = summary_int("surface_with_fixture_count")
    if summary_surface_count != len(surfaces):
        violations.append("read_surface_fixture_surface_count_mismatch")
    if summary_surface_with_fixture_count != expected_surface_with_fixture_count:
        violations.append("read_surface_fixture_surface_with_fixture_count_mismatch")
    if normalized_status_counts != expected_status_counts:
        violations.append("read_surface_fixture_status_counts_mismatch")
    if list(physical_missing) != expected_physical_missing:
        violations.append("read_surface_fixture_physical_missing_surface_ids_mismatch")

    detail["snapshot_status"] = snapshot.get("status")
    detail["surface_count"] = summary_surface_count
    detail["surface_with_fixture_count"] = summary_surface_with_fixture_count
    detail["status_counts"] = dict(status_counts)
    detail["physical_fixture_missing_surface_ids"] = list(physical_missing)
    detail["next_physical_readback_surface_id"] = next_physical_readback_surface_id
    detail["declared_fixture_path_count"] = declared_fixture_path_count
    detail["declared_test_name_count"] = declared_test_name_count
    detail["violations"] = violations
    finished = datetime.now(timezone.utc)
    return CheckResult(
        name="read_surface_fixture_evidence_schema",
        status="passed" if not violations else "failed",
        started_at=started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        finished_at=finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration_sec=round((finished - started).total_seconds(), 3),
        detail=detail,
    )


def validate_read_surface_remediation_plan_snapshot() -> CheckResult:
    started = datetime.now(timezone.utc)
    detail: dict[str, object] = {
        "policy": "read_surface_remediation_plan_must_track_fixture_next_target_without_live_authority",
    }
    violations: list[str] = []

    try:
        fixture_snapshot = build_read_surface_fixture_evidence_snapshot()
        plan = build_read_surface_remediation_plan(fixture_evidence_snapshot=fixture_snapshot)
    except Exception as exc:  # pragma: no cover - defensive release-check guard
        fixture_snapshot = {}
        plan = {}
        violations.append(f"read_surface_remediation_plan_build_error:{type(exc).__name__}")

    fixture_summary = (
        fixture_snapshot.get("summary")
        if isinstance(fixture_snapshot.get("summary"), Mapping)
        else {}
    )
    plan_summary = plan.get("summary") if isinstance(plan.get("summary"), Mapping) else {}
    safety = plan.get("safety") if isinstance(plan.get("safety"), Mapping) else {}
    actions = plan.get("actions") if isinstance(plan.get("actions"), list) else []
    if not isinstance(plan.get("actions"), list):
        violations.append("read_surface_remediation_actions_missing")

    expected_physical_ids = fixture_summary.get("physical_fixture_missing_surface_ids")
    if not isinstance(expected_physical_ids, list):
        expected_physical_ids = []
        violations.append("read_surface_remediation_fixture_physical_ids_missing")
    expected_physical_ids = [item for item in expected_physical_ids if isinstance(item, str) and item]
    expected_next_target = fixture_summary.get("next_physical_readback_surface_id")
    if not isinstance(expected_next_target, str):
        expected_next_target = expected_physical_ids[0] if expected_physical_ids else None
    expected_next_action_id = f"remediate_{expected_next_target}" if expected_next_target else None

    physical_ids = plan_summary.get("physical_fixture_missing_surface_ids")
    if not isinstance(physical_ids, list):
        physical_ids = []
        violations.append("read_surface_remediation_physical_ids_missing")
    physical_ids = [item for item in physical_ids if isinstance(item, str) and item]
    if physical_ids != expected_physical_ids:
        violations.append("read_surface_remediation_physical_ids_mismatch")

    next_target = plan_summary.get("next_physical_readback_surface_id")
    if next_target != expected_next_target:
        violations.append("read_surface_remediation_next_target_mismatch")
    next_action_id = plan_summary.get("next_physical_readback_action_id")
    if next_action_id != expected_next_action_id:
        violations.append("read_surface_remediation_next_action_id_mismatch")

    action_by_id = {
        row.get("id"): row
        for row in actions
        if isinstance(row, Mapping) and isinstance(row.get("id"), str)
    }
    next_action = action_by_id.get(next_action_id) if isinstance(next_action_id, str) else None
    target_action_ids = [
        row.get("id")
        for row in actions
        if isinstance(row, Mapping)
        and isinstance(row.get("id"), str)
        and row.get("next_physical_readback_target") is True
    ]
    expected_target_action_ids = [expected_next_action_id] if expected_next_action_id else []
    if target_action_ids != expected_target_action_ids:
        violations.append("read_surface_remediation_next_target_action_flags_mismatch")
    if expected_next_action_id and not isinstance(next_action, Mapping):
        violations.append("read_surface_remediation_next_action_missing")
    elif isinstance(next_action, Mapping):
        if next_action.get("surface_id") != expected_next_target:
            violations.append("read_surface_remediation_next_action_surface_mismatch")
        if next_action.get("physical_readback_gap") is not True:
            violations.append("read_surface_remediation_next_action_not_physical_gap")
        if next_action.get("next_physical_readback_target") is not True:
            violations.append("read_surface_remediation_next_action_target_flag_missing")
        missing_contracts = next_action.get("missing_contracts")
        if not isinstance(missing_contracts, list) or "physical_readback_fixture" not in missing_contracts:
            violations.append("read_surface_remediation_next_action_missing_physical_contract")

    expected_true = ("read_only",)
    for key in expected_true:
        if safety.get(key) is not True:
            violations.append(f"read_surface_remediation_{key}_not_true")
    expected_false = (
        "live_write",
        "mutates_console_state",
        "live_widening_allowed",
        "full_autonomous_ready",
    )
    for key in expected_false:
        if safety.get(key) is not False:
            violations.append(f"read_surface_remediation_{key}_not_false")
    if plan.get("type") != "read_surface_remediation_plan":
        violations.append("read_surface_remediation_type_mismatch")
    if plan.get("status") == "read_coverage_complete" and physical_ids:
        violations.append("read_surface_remediation_complete_but_physical_gaps_remain")
    if expected_physical_ids and not actions:
        violations.append("read_surface_remediation_actions_empty_with_physical_gaps")

    detail["plan_status"] = plan.get("status")
    detail["action_count"] = int(plan_summary.get("action_count") or len(actions))
    detail["incomplete_surface_count"] = int(plan_summary.get("incomplete_surface_count") or 0)
    detail["physical_fixture_missing_surface_ids"] = list(physical_ids)
    detail["next_physical_readback_surface_id"] = next_target
    detail["next_physical_readback_action_id"] = next_action_id
    detail["next_physical_readback_target_action_ids"] = list(target_action_ids)
    detail["expected_next_physical_readback_surface_id"] = expected_next_target
    detail["expected_next_physical_readback_action_id"] = expected_next_action_id
    detail["violations"] = violations
    finished = datetime.now(timezone.utc)
    return CheckResult(
        name="read_surface_remediation_plan_schema",
        status="passed" if not violations else "failed",
        started_at=started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        finished_at=finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration_sec=round((finished - started).total_seconds(), 3),
        detail=detail,
    )


def validate_read_surface_physical_readback_capture_plan_snapshot() -> CheckResult:
    started = datetime.now(timezone.utc)
    detail: dict[str, object] = {
        "policy": "read_surface_physical_readback_capture_plan_must_be_operator_required_and_non_executing",
    }
    violations: list[str] = []

    try:
        fixture_snapshot = build_read_surface_fixture_evidence_snapshot()
        remediation_plan = build_read_surface_remediation_plan(
            fixture_evidence_snapshot=fixture_snapshot
        )
        capture_plan = build_read_surface_physical_readback_capture_plan(
            fixture_evidence_snapshot=fixture_snapshot,
            remediation_plan=remediation_plan,
        )
    except Exception as exc:  # pragma: no cover - defensive release-check guard
        fixture_snapshot = {}
        remediation_plan = {}
        capture_plan = {}
        violations.append(f"read_surface_capture_plan_build_error:{type(exc).__name__}")

    fixture_summary = (
        fixture_snapshot.get("summary")
        if isinstance(fixture_snapshot.get("summary"), Mapping)
        else {}
    )
    remediation_summary = (
        remediation_plan.get("summary")
        if isinstance(remediation_plan.get("summary"), Mapping)
        else {}
    )
    capture_summary = (
        capture_plan.get("summary")
        if isinstance(capture_plan.get("summary"), Mapping)
        else {}
    )
    safety = (
        capture_plan.get("safety")
        if isinstance(capture_plan.get("safety"), Mapping)
        else {}
    )
    steps = capture_plan.get("steps") if isinstance(capture_plan.get("steps"), list) else []
    if not isinstance(capture_plan.get("steps"), list):
        violations.append("read_surface_capture_plan_steps_missing")

    expected_physical_ids = fixture_summary.get("physical_fixture_missing_surface_ids")
    if not isinstance(expected_physical_ids, list):
        expected_physical_ids = []
        violations.append("read_surface_capture_plan_fixture_physical_ids_missing")
    expected_physical_ids = [
        item for item in expected_physical_ids if isinstance(item, str) and item
    ]
    expected_next_target = fixture_summary.get("next_physical_readback_surface_id")
    if not isinstance(expected_next_target, str):
        expected_next_target = expected_physical_ids[0] if expected_physical_ids else None
    expected_next_action_id = f"remediate_{expected_next_target}" if expected_next_target else None

    next_target = capture_summary.get("next_physical_readback_surface_id")
    if next_target != expected_next_target:
        violations.append("read_surface_capture_plan_next_target_mismatch")
    next_action_id = capture_summary.get("next_physical_readback_action_id")
    if next_action_id != expected_next_action_id:
        violations.append("read_surface_capture_plan_next_action_id_mismatch")
    if capture_summary.get("target_action_consistency") is not bool(expected_next_target):
        violations.append("read_surface_capture_plan_target_action_consistency_mismatch")
    if remediation_summary.get("next_physical_readback_surface_id") != expected_next_target:
        violations.append("read_surface_capture_plan_remediation_target_mismatch")
    if remediation_summary.get("next_physical_readback_action_id") != expected_next_action_id:
        violations.append("read_surface_capture_plan_remediation_action_mismatch")

    expected_true = ("read_only", "operator_required")
    for key in expected_true:
        if safety.get(key) is not True:
            violations.append(f"read_surface_capture_plan_{key}_not_true")
    expected_false = (
        "live_write",
        "mutates_console_state",
        "queries_console",
        "executes_tests",
        "execution_allowed_by_plan",
        "live_widening_allowed",
        "live_authority_granted",
        "full_autonomous_ready",
    )
    for key in expected_false:
        if safety.get(key) is not False:
            violations.append(f"read_surface_capture_plan_{key}_not_false")

    if capture_plan.get("type") != "read_surface_physical_readback_capture_plan":
        violations.append("read_surface_capture_plan_type_mismatch")
    if (
        capture_plan.get("method_id")
        != READ_SURFACE_PHYSICAL_READBACK_CAPTURE_PLAN_METHOD_ID
    ):
        violations.append("read_surface_capture_plan_method_id_mismatch")
    expected_status = (
        "operator_capture_required"
        if expected_next_target
        else "read_surface_physical_readback_complete"
    )
    if capture_plan.get("status") != expected_status:
        violations.append("read_surface_capture_plan_status_mismatch")
    if expected_next_target and len(steps) != 1:
        violations.append("read_surface_capture_plan_step_count_mismatch")
    if not expected_next_target and steps:
        violations.append("read_surface_capture_plan_complete_but_steps_present")

    required_artifact_contracts = capture_summary.get("required_artifact_contracts")
    if not isinstance(required_artifact_contracts, list):
        required_artifact_contracts = []
        violations.append("read_surface_capture_plan_required_contracts_missing")
    elif expected_next_target and "physical_readback_fixture" not in required_artifact_contracts:
        violations.append("read_surface_capture_plan_missing_physical_fixture_contract")

    step = steps[0] if steps and isinstance(steps[0], Mapping) else {}
    if expected_next_target:
        if not step:
            violations.append("read_surface_capture_plan_next_step_missing")
        elif step.get("surface_id") != expected_next_target:
            violations.append("read_surface_capture_plan_step_surface_mismatch")
        elif step.get("remediation_action_id") != expected_next_action_id:
            violations.append("read_surface_capture_plan_step_action_mismatch")
        for key in ("operator_required",):
            if step and step.get(key) is not True:
                violations.append(f"read_surface_capture_plan_step_{key}_not_true")
        for key in (
            "execution_allowed_by_plan",
            "live_write",
            "mutates_console_state",
            "queries_console",
            "executes_tests",
        ):
            if step and step.get(key) is not False:
                violations.append(f"read_surface_capture_plan_step_{key}_not_false")

    detail["plan_status"] = capture_plan.get("status")
    detail["capture_step_count"] = int(capture_summary.get("capture_step_count") or len(steps))
    detail["physical_fixture_missing_surface_ids"] = list(expected_physical_ids)
    detail["next_physical_readback_surface_id"] = next_target
    detail["next_physical_readback_action_id"] = next_action_id
    detail["expected_next_physical_readback_surface_id"] = expected_next_target
    detail["expected_next_physical_readback_action_id"] = expected_next_action_id
    detail["target_action_consistency"] = (
        capture_summary.get("target_action_consistency") is True
    )
    detail["required_artifact_contracts"] = (
        list(required_artifact_contracts)
        if isinstance(required_artifact_contracts, list)
        else []
    )
    detail["blocked_by"] = (
        list(capture_summary.get("blocked_by"))
        if isinstance(capture_summary.get("blocked_by"), list)
        else []
    )
    detail["step_surface_id"] = step.get("surface_id") if step else None
    detail["step_remediation_action_id"] = step.get("remediation_action_id") if step else None
    detail["step_forbidden_actions"] = (
        list(step.get("forbidden_actions"))
        if step and isinstance(step.get("forbidden_actions"), list)
        else []
    )
    detail["violations"] = violations
    finished = datetime.now(timezone.utc)
    return CheckResult(
        name="read_surface_physical_readback_capture_plan_schema",
        status="passed" if not violations else "failed",
        started_at=started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        finished_at=finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration_sec=round((finished - started).total_seconds(), 3),
        detail=detail,
    )


def validate_read_surface_physical_readback_capture_packet_artifact_snapshot(
    packet_path: Path,
) -> CheckResult:
    started = datetime.now(timezone.utc)
    detail: dict[str, object] = {
        "policy": "read_surface_physical_readback_capture_packet_artifact_is_readonly_operator_gate",
        "packet_path": str(packet_path),
    }
    violations: list[str] = []

    payload, load_error = _json_object_from_path(packet_path)
    packet = payload or {}
    step = packet.get("step") if isinstance(packet.get("step"), Mapping) else {}
    commands = packet.get("commands") if isinstance(packet.get("commands"), Mapping) else {}
    physical_ids = (
        packet.get("physical_fixture_missing_surface_ids")
        if isinstance(packet.get("physical_fixture_missing_surface_ids"), list)
        else []
    )
    required_contracts = (
        packet.get("required_artifact_contracts")
        if isinstance(packet.get("required_artifact_contracts"), list)
        else []
    )
    blocked_by = (
        packet.get("blocked_by") if isinstance(packet.get("blocked_by"), list) else []
    )
    remaining_targets = (
        packet.get("remaining_capture_targets")
        if isinstance(packet.get("remaining_capture_targets"), list)
        else []
    )
    operator_sequence = (
        packet.get("operator_sequence")
        if isinstance(packet.get("operator_sequence"), list)
        else []
    )
    expected_artifacts = (
        step.get("expected_artifacts")
        if isinstance(step.get("expected_artifacts"), list)
        else []
    )
    blockers_to_clear = (
        step.get("blockers_to_clear")
        if isinstance(step.get("blockers_to_clear"), list)
        else []
    )
    forbidden_actions = (
        step.get("forbidden_actions")
        if isinstance(step.get("forbidden_actions"), list)
        else []
    )

    if load_error:
        violations.append(
            f"read_surface_physical_readback_capture_packet_artifact_{load_error}"
        )
    if packet.get("type") != "read_surface_physical_readback_capture_packet":
        violations.append("read_surface_capture_packet_artifact_type_mismatch")
    if packet.get("source_method_id") != READ_SURFACE_PHYSICAL_READBACK_CAPTURE_PLAN_METHOD_ID:
        violations.append("read_surface_capture_packet_artifact_source_method_mismatch")
    if packet.get("status") not in (
        "operator_capture_required",
        "read_surface_physical_readback_complete",
    ):
        violations.append("read_surface_capture_packet_artifact_status_unknown")
    if packet.get("source_plan_status") != packet.get("status"):
        violations.append("read_surface_capture_packet_artifact_source_status_mismatch")

    for key in ("read_only", "operator_required"):
        if packet.get(key) is not True:
            violations.append(f"read_surface_capture_packet_artifact_{key}_not_true")
    for key in (
        "live_write",
        "mutates_console_state",
        "queries_console",
        "executes_tests",
        "execution_allowed_by_packet",
        "live_widening_allowed",
        "live_authority_granted",
        "full_autonomous_ready",
    ):
        if packet.get(key) is not False:
            violations.append(f"read_surface_capture_packet_artifact_{key}_not_false")

    if not all(isinstance(item, str) and item for item in physical_ids):
        violations.append("read_surface_capture_packet_artifact_physical_ids_invalid")
    physical_gap_surface_count = packet.get("physical_gap_surface_count")
    if (
        not isinstance(physical_gap_surface_count, int)
        or physical_gap_surface_count != len(physical_ids)
    ):
        violations.append("read_surface_capture_packet_artifact_gap_count_mismatch")
        physical_gap_surface_count = len(physical_ids)
    capture_step_count = packet.get("capture_step_count")
    if not isinstance(capture_step_count, int):
        violations.append("read_surface_capture_packet_artifact_step_count_missing")
        capture_step_count = 0
    remaining_target_count = packet.get("remaining_capture_target_count")
    if (
        not isinstance(remaining_target_count, int)
        or remaining_target_count != len(remaining_targets)
    ):
        violations.append(
            "read_surface_capture_packet_artifact_remaining_target_count_mismatch"
        )
        remaining_target_count = len(remaining_targets)

    next_surface_id = packet.get("next_physical_readback_surface_id")
    next_action_id = packet.get("next_physical_readback_action_id")
    capture_required = packet.get("status") == "operator_capture_required"
    target_surface_ids: list[str] = []
    target_action_ids: list[str] = []
    target_contract_ids: list[str] = []
    target_forbidden_flags: list[str] = []
    for index, target in enumerate(remaining_targets):
        if not isinstance(target, Mapping):
            violations.append(
                f"read_surface_capture_packet_artifact_target_{index}_not_object"
            )
            continue
        surface_id = target.get("surface_id")
        action_id = target.get("remediation_action_id")
        contract_id = target.get("contract_method_id")
        if not isinstance(surface_id, str) or not surface_id:
            violations.append(
                f"read_surface_capture_packet_artifact_target_{index}_surface_missing"
            )
        else:
            target_surface_ids.append(surface_id)
        if not isinstance(action_id, str) or not action_id:
            violations.append(
                f"read_surface_capture_packet_artifact_target_{index}_action_missing"
            )
        else:
            target_action_ids.append(action_id)
        if not isinstance(contract_id, str) or not contract_id:
            violations.append(
                f"read_surface_capture_packet_artifact_target_{index}_contract_missing"
            )
        else:
            target_contract_ids.append(contract_id)
        for key in ("target_console_surface", "read_mode"):
            if not isinstance(target.get(key), str) or not target.get(key):
                violations.append(
                    f"read_surface_capture_packet_artifact_target_{index}_{key}_missing"
                )
        target_contracts = (
            target.get("required_artifact_contracts")
            if isinstance(target.get("required_artifact_contracts"), list)
            else []
        )
        if "physical_readback_fixture" not in target_contracts:
            violations.append(
                "read_surface_capture_packet_artifact_target_"
                f"{index}_physical_fixture_contract_missing"
            )
        if isinstance(contract_id, str) and contract_id and contract_id not in target_contracts:
            violations.append(
                f"read_surface_capture_packet_artifact_target_{index}_contract_not_required"
            )
        for key in (
            "execution_allowed_by_target",
            "live_write",
            "mutates_console_state",
            "queries_console",
        ):
            if target.get(key) is not False:
                target_forbidden_flags.append(f"{index}:{key}")
                violations.append(
                    f"read_surface_capture_packet_artifact_target_{index}_{key}_not_false"
                )
        for key in ("expected_artifacts", "blockers_to_clear"):
            if not isinstance(target.get(key), list) or not target.get(key):
                violations.append(
                    f"read_surface_capture_packet_artifact_target_{index}_{key}_missing"
                )
    if capture_required:
        if not isinstance(next_surface_id, str) or not next_surface_id:
            violations.append("read_surface_capture_packet_artifact_next_surface_missing")
        if not isinstance(next_action_id, str) or not next_action_id:
            violations.append("read_surface_capture_packet_artifact_next_action_missing")
        elif next_action_id != f"remediate_{next_surface_id}":
            violations.append("read_surface_capture_packet_artifact_next_action_mismatch")
        if packet.get("target_action_consistency") is not True:
            violations.append(
                "read_surface_capture_packet_artifact_target_action_consistency_not_true"
            )
        if not remaining_targets:
            violations.append("read_surface_capture_packet_artifact_targets_missing")
        if target_surface_ids != physical_ids:
            violations.append(
                "read_surface_capture_packet_artifact_target_surface_sequence_mismatch"
            )
        if target_surface_ids and target_surface_ids[0] != next_surface_id:
            violations.append(
                "read_surface_capture_packet_artifact_first_target_surface_mismatch"
            )
        if target_action_ids and target_action_ids[0] != next_action_id:
            violations.append(
                "read_surface_capture_packet_artifact_first_target_action_mismatch"
            )
        if target_surface_ids and len(set(target_surface_ids)) != len(target_surface_ids):
            violations.append(
                "read_surface_capture_packet_artifact_duplicate_target_surface"
            )
        if capture_step_count < 1:
            violations.append("read_surface_capture_packet_artifact_step_count_too_low")
        if not required_contracts:
            violations.append("read_surface_capture_packet_artifact_contracts_missing")
        elif "physical_readback_fixture" not in required_contracts:
            violations.append(
                "read_surface_capture_packet_artifact_physical_fixture_contract_missing"
            )
        if not blocked_by:
            violations.append("read_surface_capture_packet_artifact_blocked_by_missing")
        if not step:
            violations.append("read_surface_capture_packet_artifact_step_missing")
        else:
            if step.get("surface_id") != next_surface_id:
                violations.append("read_surface_capture_packet_artifact_step_surface_mismatch")
            if step.get("remediation_action_id") != next_action_id:
                violations.append("read_surface_capture_packet_artifact_step_action_mismatch")
            for key in ("target_console_surface", "read_mode", "contract_method_id"):
                if not isinstance(step.get(key), str) or not step.get(key):
                    violations.append(
                        f"read_surface_capture_packet_artifact_step_{key}_missing"
                    )
            if not expected_artifacts:
                violations.append(
                    "read_surface_capture_packet_artifact_step_expected_artifacts_missing"
                )
            if not blockers_to_clear:
                violations.append(
                    "read_surface_capture_packet_artifact_step_blockers_missing"
                )
            if not forbidden_actions:
                violations.append(
                    "read_surface_capture_packet_artifact_step_forbidden_actions_missing"
                )
    else:
        if next_surface_id is not None or next_action_id is not None:
            violations.append("read_surface_capture_packet_artifact_complete_target_present")
        if capture_step_count != 0:
            violations.append("read_surface_capture_packet_artifact_complete_step_count_nonzero")
        if step:
            violations.append("read_surface_capture_packet_artifact_complete_step_present")
        if packet.get("target_action_consistency") is not False:
            violations.append(
                "read_surface_capture_packet_artifact_complete_target_consistency_not_false"
            )
        if remaining_target_count != 0 or remaining_targets:
            violations.append("read_surface_capture_packet_artifact_complete_targets_present")

    if len(operator_sequence) < 5:
        violations.append("read_surface_capture_packet_artifact_operator_sequence_short")
    focused_validation = commands.get("focused_validation")
    if not isinstance(focused_validation, list):
        violations.append("read_surface_capture_packet_artifact_focused_validation_missing")
        focused_validation = []
    else:
        for test_path in (
            "tests/test_read_surface_physical_readback_capture_plan.py",
            "tests/test_no_wing_release_check.py",
        ):
            if test_path not in focused_validation:
                violations.append(
                    "read_surface_capture_packet_artifact_focused_validation_missing:"
                    f"{test_path}"
                )
    forbidden_text = "\n".join(str(item).lower() for item in forbidden_actions)
    for phrase in (
        "do not apply",
        "do not transmit osc",
        "do not change safety rows",
    ):
        if capture_required and phrase not in forbidden_text:
            violations.append(
                "read_surface_capture_packet_artifact_forbidden_action_missing:"
                f"{phrase}"
            )

    detail["packet_type"] = packet.get("type")
    detail["packet_status"] = packet.get("status")
    detail["source_plan_status"] = packet.get("source_plan_status")
    detail["source_method_id"] = packet.get("source_method_id")
    detail["read_only"] = packet.get("read_only")
    detail["live_write"] = packet.get("live_write")
    detail["mutates_console_state"] = packet.get("mutates_console_state")
    detail["queries_console"] = packet.get("queries_console")
    detail["executes_tests"] = packet.get("executes_tests")
    detail["operator_required"] = packet.get("operator_required")
    detail["execution_allowed_by_packet"] = packet.get("execution_allowed_by_packet")
    detail["live_widening_allowed"] = packet.get("live_widening_allowed")
    detail["live_authority_granted"] = packet.get("live_authority_granted")
    detail["full_autonomous_ready"] = packet.get("full_autonomous_ready")
    detail["physical_fixture_missing_surface_ids"] = list(physical_ids)
    detail["physical_gap_surface_count"] = physical_gap_surface_count
    detail["remaining_capture_target_count"] = remaining_target_count
    detail["remaining_capture_target_surface_ids"] = list(target_surface_ids)
    detail["remaining_capture_target_action_ids"] = list(target_action_ids)
    detail["remaining_capture_target_contract_ids"] = list(target_contract_ids)
    detail["remaining_capture_target_forbidden_flags"] = list(target_forbidden_flags)
    detail["next_physical_readback_surface_id"] = next_surface_id
    detail["next_physical_readback_action_id"] = next_action_id
    detail["target_action_consistency"] = packet.get("target_action_consistency")
    detail["capture_step_count"] = capture_step_count
    detail["required_artifact_contracts"] = list(required_contracts)
    detail["blocked_by"] = list(blocked_by)
    detail["step_surface_id"] = step.get("surface_id") if step else None
    detail["step_remediation_action_id"] = (
        step.get("remediation_action_id") if step else None
    )
    detail["step_target_console_surface"] = (
        step.get("target_console_surface") if step else None
    )
    detail["step_read_mode"] = step.get("read_mode") if step else None
    detail["step_contract_method_id"] = step.get("contract_method_id") if step else None
    detail["step_expected_artifact_count"] = len(expected_artifacts)
    detail["step_blocker_count"] = len(blockers_to_clear)
    detail["step_forbidden_actions"] = list(forbidden_actions)
    detail["operator_sequence_count"] = len(operator_sequence)
    detail["focused_validation"] = list(focused_validation)
    detail["violations"] = violations
    finished = datetime.now(timezone.utc)
    return CheckResult(
        name="read_surface_physical_readback_capture_packet_artifact_schema",
        status="passed" if not violations else "failed",
        started_at=started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        finished_at=finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration_sec=round((finished - started).total_seconds(), 3),
        detail=detail,
    )


def validate_read_surface_physical_readback_capture_template_artifact_snapshot(
    template_path: Path,
) -> CheckResult:
    started = datetime.now(timezone.utc)
    detail: dict[str, object] = {
        "policy": "read_surface_physical_readback_capture_template_artifact_is_pending_readonly_operator_gate",
        "template_path": str(template_path),
    }
    violations: list[str] = []

    payload, load_error = _json_object_from_path(template_path)
    template = payload or {}
    records = (
        template.get("capture_templates")
        if isinstance(template.get("capture_templates"), list)
        else []
    )
    required_edits = (
        template.get("required_operator_edits")
        if isinstance(template.get("required_operator_edits"), list)
        else []
    )
    blocked_by = (
        template.get("blocked_by")
        if isinstance(template.get("blocked_by"), list)
        else []
    )
    operator_sequence = (
        template.get("operator_sequence")
        if isinstance(template.get("operator_sequence"), list)
        else []
    )
    commands = (
        template.get("commands")
        if isinstance(template.get("commands"), Mapping)
        else {}
    )

    if load_error:
        violations.append(
            f"read_surface_physical_readback_capture_template_artifact_{load_error}"
        )
    if template.get("type") != "read_surface_physical_readback_artifact_capture_template":
        violations.append("capture_template_artifact_type_mismatch")
    if template.get("status") not in (
        "operator_capture_template_ready",
        "read_surface_physical_readback_complete",
        "operator_capture_target_missing",
    ):
        violations.append("capture_template_artifact_status_unknown")
    if template.get("source_packet_type") != "read_surface_physical_readback_capture_packet":
        violations.append("capture_template_artifact_source_packet_type_mismatch")
    if template.get("source_method_id") != READ_SURFACE_PHYSICAL_READBACK_CAPTURE_PLAN_METHOD_ID:
        violations.append("capture_template_artifact_source_method_mismatch")

    for key in ("read_only", "operator_required"):
        if template.get(key) is not True:
            violations.append(f"capture_template_artifact_{key}_not_true")
    for key in (
        "live_write",
        "mutates_console_state",
        "queries_console",
        "executes_tests",
        "execution_allowed_by_template",
        "accepted_as_physical_fixture_candidate",
        "fixture_closure_allowed_by_template",
        "live_widening_allowed",
        "live_authority_granted",
        "full_autonomous_ready",
    ):
        if template.get(key) is not False:
            violations.append(f"capture_template_artifact_{key}_not_false")

    all_remaining_target_count = template.get("all_remaining_target_count")
    if not isinstance(all_remaining_target_count, int):
        violations.append("capture_template_artifact_remaining_target_count_missing")
        all_remaining_target_count = 0
    capture_template_count = template.get("capture_template_count")
    if not isinstance(capture_template_count, int) or capture_template_count != len(records):
        violations.append("capture_template_artifact_record_count_mismatch")
        capture_template_count = len(records)
    if template.get("status") == "operator_capture_template_ready" and not records:
        violations.append("capture_template_artifact_ready_status_has_no_records")
    if (
        template.get("status") == "read_surface_physical_readback_complete"
        and records
    ):
        violations.append("capture_template_artifact_complete_status_has_records")
    if template.get("status") == "operator_capture_target_missing" and records:
        violations.append("capture_template_artifact_missing_status_has_records")

    for required_edit in (
        "artifact_source_id",
        "captured_at",
        "captured_by",
        "source",
        "console_model",
        "capture_scope",
        "physical_readback_payload",
    ):
        if required_edit not in required_edits:
            violations.append(
                f"capture_template_artifact_required_edit_{required_edit}_missing"
            )
    if len(operator_sequence) < 6:
        violations.append("capture_template_artifact_operator_sequence_short")
    focused_validation = commands.get("focused_validation")
    if not isinstance(focused_validation, list):
        violations.append("capture_template_artifact_focused_validation_missing")
        focused_validation = []
    else:
        for test_path in (
            "tests/test_read_surface_physical_readback_capture_template.py",
            "tests/test_no_wing_release_check.py",
        ):
            if test_path not in focused_validation:
                violations.append(
                    "capture_template_artifact_focused_validation_missing:"
                    f"{test_path}"
                )
    if template.get("status") == "operator_capture_template_ready" and not blocked_by:
        violations.append("capture_template_artifact_blocked_by_missing")

    record_unsafe_flags: list[str] = []
    record_surface_ids: list[str] = []
    record_contract_ids: list[str] = []
    payload_template_contract_ids: list[str] = []
    payload_template_artifact_types: list[str] = []
    payload_template_schema_ids: list[str] = []
    payload_template_field_counts: list[int] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            violations.append(f"capture_template_artifact_record_{index}_not_object")
            continue
        for key in (
            "template_id",
            "surface_id",
            "remediation_action_id",
            "target_console_surface",
            "read_mode",
            "required_contract_method_id",
        ):
            if not isinstance(record.get(key), str) or not record.get(key):
                violations.append(
                    f"capture_template_artifact_record_{index}_{key}_missing"
                )
        surface_id = record.get("surface_id")
        if isinstance(surface_id, str) and surface_id:
            record_surface_ids.append(surface_id)
        contract_id = record.get("required_contract_method_id")
        if isinstance(contract_id, str) and contract_id:
            record_contract_ids.append(contract_id)
        required_contracts = (
            record.get("required_artifact_contracts")
            if isinstance(record.get("required_artifact_contracts"), list)
            else []
        )
        if "physical_readback_fixture" not in required_contracts:
            violations.append(
                f"capture_template_artifact_record_{index}_physical_fixture_contract_missing"
            )
        if (
            isinstance(contract_id, str)
            and contract_id
            and contract_id not in required_contracts
        ):
            violations.append(
                f"capture_template_artifact_record_{index}_contract_not_required"
            )
        if not isinstance(record.get("expected_artifacts"), list) or not record.get(
            "expected_artifacts"
        ):
            violations.append(
                f"capture_template_artifact_record_{index}_expected_artifacts_missing"
            )
        if not isinstance(record.get("blockers_to_clear"), list) or not record.get(
            "blockers_to_clear"
        ):
            violations.append(
                f"capture_template_artifact_record_{index}_blockers_missing"
            )
        pending_fields = {
            "artifact_source_id": "<path-to-operator-captured-json>",
            "captured_by": "<operator-name>",
            "console_model": "<console model and firmware>",
            "capture_scope": "<manual capture scope and show-state notes>",
        }
        for key, expected_value in pending_fields.items():
            if record.get(key) != expected_value:
                violations.append(
                    f"capture_template_artifact_record_{index}_{key}_not_pending"
                )
        captured_at = record.get("captured_at")
        if not isinstance(captured_at, str) or not captured_at.startswith(
            "<UTC ISO timestamp"
        ):
            violations.append(
                f"capture_template_artifact_record_{index}_captured_at_not_pending"
            )
        source = record.get("source")
        if not isinstance(source, str) or not source.startswith("<physical_readback"):
            violations.append(
                f"capture_template_artifact_record_{index}_source_not_pending"
            )
        payload_placeholder = record.get("physical_readback_payload")
        if not isinstance(payload_placeholder, str) or not payload_placeholder.startswith(
            "<replace with JSON payload"
        ):
            violations.append(
                f"capture_template_artifact_record_{index}_payload_not_pending"
            )
        payload_template = (
            record.get("physical_readback_payload_template")
            if isinstance(record.get("physical_readback_payload_template"), Mapping)
            else {}
        )
        if not payload_template:
            violations.append(
                f"capture_template_artifact_record_{index}_payload_template_missing"
            )
        else:
            if (
                payload_template.get("template_status")
                != "operator_must_fill_from_physical_readback"
            ):
                violations.append(
                    f"capture_template_artifact_record_{index}_payload_template_status_mismatch"
                )
            if payload_template.get("required_contract_method_id") != contract_id:
                violations.append(
                    f"capture_template_artifact_record_{index}_payload_template_contract_mismatch"
                )
            payload_contract_id = payload_template.get("required_contract_method_id")
            if isinstance(payload_contract_id, str) and payload_contract_id:
                payload_template_contract_ids.append(payload_contract_id)
            artifact_type = payload_template.get("artifact_type")
            schema_id = payload_template.get("schema_id")
            if not isinstance(artifact_type, str) or not artifact_type:
                violations.append(
                    f"capture_template_artifact_record_{index}_payload_template_type_missing"
                )
            else:
                payload_template_artifact_types.append(artifact_type)
            if not isinstance(schema_id, str) or not schema_id:
                violations.append(
                    f"capture_template_artifact_record_{index}_payload_template_schema_missing"
                )
            else:
                payload_template_schema_ids.append(schema_id)
            required_top = (
                payload_template.get("required_top_level_fields")
                if isinstance(payload_template.get("required_top_level_fields"), list)
                else []
            )
            if not required_top:
                violations.append(
                    f"capture_template_artifact_record_{index}_payload_template_required_fields_missing"
                )
            for required_field in (
                "type",
                "schema_id",
                "surface_id",
                "remediation_action_id",
                "captured_at",
                "source",
                "console_model",
                "safety",
            ):
                if required_field not in required_top:
                    violations.append(
                        "capture_template_artifact_record_"
                        f"{index}_payload_template_required_field_missing:{required_field}"
                    )
            required_sets = payload_template.get("required_field_sets")
            if not isinstance(required_sets, Mapping):
                violations.append(
                    f"capture_template_artifact_record_{index}_payload_template_field_sets_missing"
                )
            payload = (
                payload_template.get("payload")
                if isinstance(payload_template.get("payload"), Mapping)
                else {}
            )
            if not payload:
                violations.append(
                    f"capture_template_artifact_record_{index}_payload_template_payload_missing"
                )
            else:
                payload_template_field_counts.append(len(payload))
                for field in required_top:
                    if isinstance(field, str) and field and field not in payload:
                        violations.append(
                            "capture_template_artifact_record_"
                            f"{index}_payload_template_payload_missing_field:{field}"
                        )
                if payload.get("type") != artifact_type:
                    violations.append(
                        f"capture_template_artifact_record_{index}_payload_template_payload_type_mismatch"
                    )
                if payload.get("schema_id") != schema_id:
                    violations.append(
                        f"capture_template_artifact_record_{index}_payload_template_payload_schema_mismatch"
                    )
                if payload.get("surface_id") != surface_id:
                    violations.append(
                        f"capture_template_artifact_record_{index}_payload_template_payload_surface_mismatch"
                    )
                if payload.get("remediation_action_id") != record.get(
                    "remediation_action_id"
                ):
                    violations.append(
                        f"capture_template_artifact_record_{index}_payload_template_payload_action_mismatch"
                    )
                if not isinstance(payload.get("captured_at"), str) or not payload[
                    "captured_at"
                ].startswith("<UTC ISO timestamp"):
                    violations.append(
                        f"capture_template_artifact_record_{index}_payload_template_captured_at_not_pending"
                    )
                if not isinstance(payload.get("source"), str) or not payload[
                    "source"
                ].startswith("<physical_readback"):
                    violations.append(
                        f"capture_template_artifact_record_{index}_payload_template_source_not_pending"
                    )
                if payload.get("console_model") != "<console model and firmware>":
                    violations.append(
                        f"capture_template_artifact_record_{index}_payload_template_console_model_not_pending"
                    )
                payload_safety = (
                    payload.get("safety")
                    if isinstance(payload.get("safety"), Mapping)
                    else {}
                )
                if payload_safety.get("read_only") is not True:
                    violations.append(
                        f"capture_template_artifact_record_{index}_payload_template_safety_read_only_not_true"
                    )
                for key in (
                    "live_write",
                    "mutates_console_state",
                    "live_widening_allowed",
                    "live_authority_granted",
                    "full_autonomous_ready",
                    "accepted_as_physical_fixture_candidate",
                ):
                    if payload_safety.get(key) is not False:
                        flag = f"record_{index}_payload_template_{key}"
                        record_unsafe_flags.append(flag)
                        violations.append(
                            f"capture_template_artifact_{flag}_not_false"
                        )
        safety = record.get("safety") if isinstance(record.get("safety"), Mapping) else {}
        if safety.get("read_only") is not True or safety.get("operator_required") is not True:
            violations.append(
                f"capture_template_artifact_record_{index}_safety_true_flags_missing"
            )
        for key in (
            "live_write",
            "mutates_console_state",
            "queries_console",
            "executes_tests",
            "execution_allowed_by_template",
            "accepted_as_physical_fixture_candidate",
            "fixture_closure_allowed_by_template",
            "live_widening_allowed",
            "live_authority_granted",
            "full_autonomous_ready",
        ):
            if safety.get(key) is not False:
                flag = f"record_{index}_{key}"
                record_unsafe_flags.append(flag)
                violations.append(
                    f"capture_template_artifact_{flag}_not_false"
                )

    detail["template_type"] = template.get("type")
    detail["template_status"] = template.get("status")
    detail["source_packet_type"] = template.get("source_packet_type")
    detail["source_packet_status"] = template.get("source_packet_status")
    detail["source_method_id"] = template.get("source_method_id")
    detail["selected_surface_id"] = template.get("selected_surface_id")
    detail["include_all_targets"] = template.get("include_all_targets")
    detail["read_only"] = template.get("read_only")
    detail["live_write"] = template.get("live_write")
    detail["mutates_console_state"] = template.get("mutates_console_state")
    detail["queries_console"] = template.get("queries_console")
    detail["executes_tests"] = template.get("executes_tests")
    detail["operator_required"] = template.get("operator_required")
    detail["execution_allowed_by_template"] = template.get(
        "execution_allowed_by_template"
    )
    detail["accepted_as_physical_fixture_candidate"] = template.get(
        "accepted_as_physical_fixture_candidate"
    )
    detail["fixture_closure_allowed_by_template"] = template.get(
        "fixture_closure_allowed_by_template"
    )
    detail["live_widening_allowed"] = template.get("live_widening_allowed")
    detail["live_authority_granted"] = template.get("live_authority_granted")
    detail["full_autonomous_ready"] = template.get("full_autonomous_ready")
    detail["all_remaining_target_count"] = all_remaining_target_count
    detail["capture_template_count"] = capture_template_count
    detail["capture_template_surface_ids"] = record_surface_ids
    detail["capture_template_contract_ids"] = record_contract_ids
    detail["payload_template_contract_ids"] = payload_template_contract_ids
    detail["payload_template_artifact_types"] = payload_template_artifact_types
    detail["payload_template_schema_ids"] = payload_template_schema_ids
    detail["payload_template_field_counts"] = payload_template_field_counts
    detail["required_operator_edits"] = (
        list(required_edits) if isinstance(required_edits, list) else []
    )
    detail["operator_sequence_count"] = len(operator_sequence)
    detail["blocked_by"] = list(blocked_by) if isinstance(blocked_by, list) else []
    detail["record_unsafe_flags"] = record_unsafe_flags
    detail["focused_validation"] = list(focused_validation)
    detail["violations"] = violations
    finished = datetime.now(timezone.utc)
    return CheckResult(
        name="read_surface_physical_readback_capture_template_artifact_schema",
        status="passed" if not violations else "failed",
        started_at=started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        finished_at=finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration_sec=round((finished - started).total_seconds(), 3),
        detail=detail,
    )


def validate_wing_readonly_preflight_evidence_snapshot() -> CheckResult:
    started = datetime.now(timezone.utc)
    detail: dict[str, object] = {
        "policy": "wing_readonly_preflight_evidence_is_readonly_candidate_matrix",
    }
    violations: list[str] = []

    try:
        evidence = build_wing_readonly_preflight_evidence()
    except Exception as exc:  # pragma: no cover - defensive release-check guard
        evidence = {}
        violations.append(f"wing_readonly_preflight_evidence_build_error:{type(exc).__name__}")

    summary = evidence.get("summary") if isinstance(evidence.get("summary"), Mapping) else {}
    safety = evidence.get("safety") if isinstance(evidence.get("safety"), Mapping) else {}
    matrix = (
        evidence.get("mvp_blocker_matrix")
        if isinstance(evidence.get("mvp_blocker_matrix"), list)
        else []
    )
    artifacts = evidence.get("artifacts") if isinstance(evidence.get("artifacts"), list) else []
    artifact_operator_next_steps: list[str] = []
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            continue
        operator_next_step = str(artifact.get("operator_next_step") or "")
        if operator_next_step and operator_next_step not in artifact_operator_next_steps:
            artifact_operator_next_steps.append(operator_next_step)
    capture_tool_available = (REPO_ROOT / WING_READONLY_PREFLIGHT_RUNNER_TOOL).is_file()
    review_tool_available = (REPO_ROOT / WING_READONLY_PREFLIGHT_REVIEW_TOOL).is_file()
    handoff_tool_available = (REPO_ROOT / WING_READONLY_PREFLIGHT_HANDOFF_PACKET_TOOL).is_file()
    plan_command = wing_readonly_preflight_plan_command()
    capture_command = wing_readonly_preflight_capture_command_template()
    review_command = wing_readonly_preflight_review_command()
    handoff_command = wing_readonly_preflight_handoff_packet_command()
    prestart_backend_port_check_command = (
        WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND
    )
    review_manual_checklist_ids = wing_readonly_preflight_review_manual_checklist_ids()
    review_decision_template_required_fields = (
        wing_readonly_preflight_review_decision_template_required_fields()
    )
    review_manual_checklist_complete = tuple(review_manual_checklist_ids) == (
        WING_READONLY_PREFLIGHT_REVIEW_MANUAL_CHECKLIST_IDS
    )
    review_decision_template_contract_complete = tuple(
        review_decision_template_required_fields
    ) == WING_READONLY_PREFLIGHT_REVIEW_DECISION_TEMPLATE_REQUIRED_FIELDS
    capture_channels = command_value_after_flag(capture_command, "--channels")
    capture_confirmed_channels = command_value_after_flag(
        capture_command,
        "--operator-confirm-channels",
    )
    capture_confirmation_matches_channels = (
        bool(capture_channels)
        and bool(capture_confirmed_channels)
        and capture_channels == capture_confirmed_channels
    )
    capture_confirms_session_ready = (
        "--operator-confirm-session-ready" in capture_command
    )
    plan_channels = command_value_after_flag(plan_command, "--channels")
    plan_confirmed_channels = command_value_after_flag(
        plan_command,
        "--operator-confirm-channels",
    )
    plan_confirms_session_ready = "--operator-confirm-session-ready" in plan_command
    plan_value_types = command_value_after_flag(plan_command, "--value-types")
    plan_artifact_root = command_value_after_flag(plan_command, "--artifact-root")
    plan_output = command_value_after_flag(plan_command, "--plan-output")
    capture_value_types = command_value_after_flag(capture_command, "--value-types")
    capture_artifact_root = command_value_after_flag(capture_command, "--artifact-root")
    capture_expected_plan = command_value_after_flag(capture_command, "--expected-plan")
    plan_matches_capture = (
        bool(plan_channels)
        and bool(plan_confirmed_channels)
        and plan_channels == capture_channels
        and plan_confirmed_channels == capture_confirmed_channels
        and plan_value_types == capture_value_types
        and plan_artifact_root == capture_artifact_root
    )
    handoff_channels = command_value_after_flag(handoff_command, "--channels")
    handoff_value_types = command_value_after_flag(handoff_command, "--value-types")
    handoff_url = command_value_after_flag(handoff_command, "--url")
    handoff_artifact_root = command_value_after_flag(handoff_command, "--artifact-root")
    handoff_plan_output = command_value_after_flag(handoff_command, "--plan-output")
    handoff_channel_matches_capture = bool(handoff_channels) and handoff_channels == capture_channels
    handoff_value_types_match_capture = (
        bool(handoff_value_types) and handoff_value_types == capture_value_types
    )
    handoff_url_matches_capture = bool(handoff_url) and handoff_url == WING_READONLY_PREFLIGHT_RUNNER_URL
    handoff_artifact_root_matches_capture = (
        bool(handoff_artifact_root) and handoff_artifact_root == capture_artifact_root
    )

    if not capture_tool_available:
        violations.append("wing_readonly_preflight_runner_tool_missing")
    if not review_tool_available:
        violations.append("wing_readonly_preflight_review_tool_missing")
    if not handoff_tool_available:
        violations.append("wing_readonly_preflight_handoff_packet_tool_missing")
    if "--plan-only" not in plan_command:
        violations.append("wing_readonly_preflight_plan_only_missing")
    if "--plan-output" not in plan_command or not plan_output:
        violations.append("wing_readonly_preflight_plan_output_missing")
    if not plan_matches_capture:
        violations.append("wing_readonly_preflight_plan_command_mismatch")
    if "--channels" not in handoff_command:
        violations.append("wing_readonly_preflight_handoff_packet_channels_missing")
    if "--output" not in handoff_command:
        violations.append("wing_readonly_preflight_handoff_packet_output_missing")
    if "--url" not in handoff_command or not handoff_url:
        violations.append("wing_readonly_preflight_handoff_packet_url_missing")
    if "--artifact-root" not in handoff_command or not handoff_artifact_root:
        violations.append("wing_readonly_preflight_handoff_packet_artifact_root_missing")
    if "--plan-output" not in handoff_command or not handoff_plan_output:
        violations.append("wing_readonly_preflight_handoff_packet_plan_output_missing")
    if not handoff_url_matches_capture:
        violations.append("wing_readonly_preflight_handoff_packet_url_mismatch")
    if not handoff_artifact_root_matches_capture:
        violations.append("wing_readonly_preflight_handoff_packet_artifact_root_mismatch")
    if not handoff_channel_matches_capture:
        violations.append("wing_readonly_preflight_handoff_packet_channel_mismatch")
    if not handoff_value_types_match_capture:
        violations.append("wing_readonly_preflight_handoff_packet_value_types_mismatch")
    if not review_manual_checklist_complete:
        violations.append("wing_readonly_preflight_review_manual_checklist_incomplete")
    if not review_decision_template_contract_complete:
        violations.append("wing_readonly_preflight_review_decision_template_contract_incomplete")
    if "--decision-template-output" not in review_command:
        violations.append("wing_readonly_preflight_review_decision_template_output_missing")
    if "--operator-confirm-channels" not in capture_command:
        violations.append("wing_readonly_preflight_capture_operator_confirmation_missing")
    if "--operator-confirm-session-ready" not in capture_command:
        violations.append("wing_readonly_preflight_capture_session_confirmation_missing")
    if "--operator-confirm-session-ready" not in plan_command:
        violations.append("wing_readonly_preflight_plan_session_confirmation_missing")
    if "--channels" not in capture_command:
        violations.append("wing_readonly_preflight_capture_channels_missing")
    if (
        "--operator-confirm-channels" in capture_command
        and "--channels" in capture_command
        and not capture_confirmation_matches_channels
    ):
        violations.append("wing_readonly_preflight_capture_operator_confirmation_mismatch")
    if "--require-physical-readback" not in capture_command:
        violations.append("wing_readonly_preflight_capture_physical_readback_missing")
    if "--expected-plan" not in capture_command or not capture_expected_plan:
        violations.append("wing_readonly_preflight_capture_expected_plan_missing")
    if capture_expected_plan != plan_output:
        violations.append("wing_readonly_preflight_capture_expected_plan_mismatch")
    if (
        prestart_backend_port_check_command
        != WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND
    ):
        violations.append(
            "wing_readonly_preflight_prestart_backend_port_check_command_mismatch"
        )

    if evidence.get("type") != "wing_readonly_preflight_evidence":
        violations.append("wing_readonly_preflight_evidence_type_mismatch")
    if evidence.get("method_id") != WING_READONLY_PREFLIGHT_EVIDENCE_METHOD_ID:
        violations.append("wing_readonly_preflight_evidence_method_id_mismatch")
    if evidence.get("status") not in WING_READONLY_PREFLIGHT_EVIDENCE_STATUSES:
        violations.append("wing_readonly_preflight_evidence_status_mismatch")
    if len(matrix) != 4:
        violations.append("wing_readonly_preflight_evidence_matrix_count_mismatch")

    expected_true = ("read_only", "operator_required")
    for key in expected_true:
        if safety.get(key) is not True:
            violations.append(f"wing_readonly_preflight_evidence_{key}_not_true")
    expected_false = (
        "live_write",
        "mutates_console_state",
        "queries_console",
        "executes_tests",
        "execution_allowed_by_evidence",
        "readiness_closure_allowed_by_evidence",
        "live_widening_allowed",
        "live_authority_granted",
        "full_autonomous_ready",
    )
    for key in expected_false:
        if safety.get(key) is not False:
            violations.append(f"wing_readonly_preflight_evidence_{key}_not_false")

    valid_candidate_count = int(summary.get("valid_candidate_count") or 0)
    invalid_candidate_count = int(summary.get("invalid_candidate_count") or 0)
    accepted = summary.get("accepted_as_readonly_preflight_candidate") is True
    update_allowed = summary.get("readiness_matrix_update_allowed") is True
    candidate_expected = bool(valid_candidate_count) and invalid_candidate_count == 0
    closure_allowed = summary.get("readiness_closure_allowed_by_evidence") is True
    live_authority = summary.get("live_authority_granted") is True
    full_auto = summary.get("full_autonomous_ready") is True

    if invalid_candidate_count:
        violations.append("wing_readonly_preflight_invalid_artifacts_present")
    if accepted != candidate_expected:
        violations.append("wing_readonly_preflight_candidate_summary_mismatch")
    if update_allowed != candidate_expected:
        violations.append("wing_readonly_preflight_readiness_update_allowed_mismatch")
    if closure_allowed:
        violations.append("wing_readonly_preflight_readiness_closure_allowed")
    if live_authority:
        violations.append("wing_readonly_preflight_live_authority_granted")
    if full_auto:
        violations.append("wing_readonly_preflight_full_autonomous_ready")

    matrix_ids = [row.get("id") for row in matrix if isinstance(row, Mapping)]
    expected_matrix_ids = [
        "wing_readonly_connection_proof",
        "physical_console_readback_proof",
        "operator_visible_evidence_report",
        "readiness_matrix_update_based_on_real_proof",
    ]
    if matrix_ids != expected_matrix_ids:
        violations.append("wing_readonly_preflight_matrix_ids_mismatch")
    for row in matrix:
        row_map = row if isinstance(row, Mapping) else {}
        if row_map.get("closed") is not False:
            violations.append("wing_readonly_preflight_matrix_row_closed_not_false")
        if row_map.get("manual_review_required") is not True:
            violations.append("wing_readonly_preflight_matrix_manual_review_not_true")
        if row_map.get("automatic_live_authority_allowed") is not False:
            violations.append("wing_readonly_preflight_matrix_live_authority_not_false")

    detail["evidence_status"] = evidence.get("status")
    detail["artifact_count"] = int(summary.get("artifact_count") or len(artifacts))
    detail["valid_candidate_count"] = valid_candidate_count
    detail["invalid_candidate_count"] = invalid_candidate_count
    detail["accepted_as_readonly_preflight_candidate"] = accepted
    detail["readiness_matrix_update_allowed"] = update_allowed
    detail["readiness_closure_allowed_by_evidence"] = closure_allowed
    detail["live_authority_granted"] = live_authority
    detail["full_autonomous_ready"] = full_auto
    detail["mvp_blocker_candidate_count"] = int(summary.get("mvp_blocker_candidate_count") or 0)
    detail["mvp_blockers_closed_by_evidence"] = (
        list(summary.get("mvp_blockers_closed_by_evidence"))
        if isinstance(summary.get("mvp_blockers_closed_by_evidence"), list)
        else []
    )
    detail["mvp_blocker_matrix_ids"] = matrix_ids
    detail["latest_valid_candidate_run_id"] = summary.get("latest_valid_candidate_run_id")
    detail["artifact_operator_next_steps"] = artifact_operator_next_steps
    detail["capture_tool_path"] = WING_READONLY_PREFLIGHT_RUNNER_TOOL
    detail["capture_tool_available"] = capture_tool_available
    detail["plan_command"] = plan_command
    detail["plan_command_uses_plan_only"] = "--plan-only" in plan_command
    detail["plan_command_output"] = plan_output
    detail["plan_command_channels"] = plan_channels
    detail["plan_command_operator_confirmed_channels"] = plan_confirmed_channels
    detail["plan_command_confirms_session_ready"] = plan_confirms_session_ready
    detail["plan_command_value_types"] = plan_value_types
    detail["plan_command_artifact_root"] = plan_artifact_root
    detail["plan_command_matches_capture"] = plan_matches_capture
    detail["prestart_backend_port_check_required"] = True
    detail["prestart_backend_port_check_must_pass_before_starting_backend"] = True
    detail["prestart_backend_port_check_command"] = list(
        prestart_backend_port_check_command
    )
    detail["prestart_backend_port_check_failure_detail_fields"] = list(
        WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS
    )
    detail["capture_command_template"] = capture_command
    detail["capture_command_channels"] = capture_channels
    detail["capture_command_operator_confirmed_channels"] = capture_confirmed_channels
    detail["capture_command_requires_operator_channel_confirmation"] = (
        "--operator-confirm-channels" in capture_command
    )
    detail["capture_command_operator_confirmation_matches_channels"] = (
        capture_confirmation_matches_channels
    )
    detail["capture_command_confirms_session_ready"] = capture_confirms_session_ready
    detail["capture_command_requires_physical_readback"] = (
        "--require-physical-readback" in capture_command
    )
    detail["capture_command_expected_plan"] = capture_expected_plan
    detail["capture_command_expected_plan_matches_plan_output"] = (
        bool(capture_expected_plan) and capture_expected_plan == plan_output
    )
    detail["review_tool_path"] = WING_READONLY_PREFLIGHT_REVIEW_TOOL
    detail["review_tool_available"] = review_tool_available
    detail["review_command"] = review_command
    detail["review_command_writes_decision_template"] = (
        "--decision-template-output" in review_command
    )
    detail["review_exit_codes"] = dict(WING_READONLY_PREFLIGHT_REVIEW_EXIT_CODES)
    detail["review_manual_checklist_ids"] = list(review_manual_checklist_ids)
    detail["review_manual_checklist_complete"] = review_manual_checklist_complete
    detail["review_decision_template_required_fields"] = list(
        review_decision_template_required_fields
    )
    detail["review_decision_template_contract_complete"] = (
        review_decision_template_contract_complete
    )
    detail["handoff_packet_tool_path"] = WING_READONLY_PREFLIGHT_HANDOFF_PACKET_TOOL
    detail["handoff_packet_tool_available"] = handoff_tool_available
    detail["handoff_packet_command"] = handoff_command
    detail["handoff_packet_output"] = WING_READONLY_PREFLIGHT_HANDOFF_PACKET_OUTPUT
    detail["handoff_packet_plan_output"] = handoff_plan_output
    detail["handoff_packet_url"] = handoff_url
    detail["handoff_packet_artifact_root"] = handoff_artifact_root
    detail["handoff_packet_channels"] = handoff_channels
    detail["handoff_packet_value_types"] = handoff_value_types
    detail["handoff_packet_url_matches_capture"] = handoff_url_matches_capture
    detail["handoff_packet_artifact_root_matches_capture"] = handoff_artifact_root_matches_capture
    detail["handoff_packet_channel_matches_capture"] = handoff_channel_matches_capture
    detail["handoff_packet_value_types_match_capture"] = handoff_value_types_match_capture
    detail["blocked_by"] = (
        list(summary.get("blocked_by"))
        if isinstance(summary.get("blocked_by"), list)
        else []
    )
    detail["violations"] = violations
    finished = datetime.now(timezone.utc)
    return CheckResult(
        name="wing_readonly_preflight_evidence_schema",
        status="passed" if not violations else "failed",
        started_at=started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        finished_at=finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration_sec=round((finished - started).total_seconds(), 3),
        detail=detail,
    )


def _csv_strings(value: str | None) -> list[str]:
    if not isinstance(value, str):
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _csv_ints(value: str | None) -> list[int]:
    parsed: list[int] = []
    for item in _csv_strings(value):
        try:
            parsed.append(int(item))
        except ValueError:
            return []
    return parsed


def _expected_wing_read_request(channel: int, value_type: str) -> dict[str, object]:
    if value_type == "fader":
        address = f"/ch/{channel}/fdr"
    elif value_type == "gain":
        address = f"/ch/{channel}/in/set/trim"
    elif value_type == "pan":
        address = f"/ch/{channel}/pan"
    else:
        address = ""
    return {
        "type": "query_mixer_parameter",
        "request_id": f"wing-readonly-preflight:ch{channel}:{value_type}",
        "channel": channel,
        "value_type": value_type,
        "address": address,
    }


def validate_wing_readonly_preflight_plan_artifact_snapshot(
    plan_path: Path,
) -> CheckResult:
    started = datetime.now(timezone.utc)
    detail: dict[str, object] = {
        "policy": "wing_readonly_preflight_plan_artifact_is_offline_readonly_preview",
        "plan_path": str(plan_path),
    }
    violations: list[str] = []

    payload, load_error = _json_object_from_path(plan_path)
    plan = payload or {}
    plan_command = wing_readonly_preflight_plan_command()
    expected_channels = _csv_ints(command_value_after_flag(plan_command, "--channels"))
    expected_confirmed_channels = _csv_ints(
        command_value_after_flag(plan_command, "--operator-confirm-channels")
    )
    expected_value_types = _csv_strings(command_value_after_flag(plan_command, "--value-types"))
    expected_artifact_root = command_value_after_flag(plan_command, "--artifact-root")
    expected_plan_output = command_value_after_flag(plan_command, "--plan-output")
    expected_requests = [
        {"type": "get_operator_mode"},
        {"type": "get_state"},
        *[
            _expected_wing_read_request(channel, value_type)
            for channel in expected_channels
            for value_type in expected_value_types
        ],
    ]

    if load_error:
        violations.append(f"wing_readonly_preflight_plan_artifact_{load_error}")

    if plan.get("type") != "wing_readonly_preflight_offline_plan":
        violations.append("wing_readonly_preflight_plan_type_mismatch")
    if plan.get("status") != "ready_for_operator_review":
        violations.append("wing_readonly_preflight_plan_status_mismatch")
    if (
        plan.get("proof_artifact_schema")
        != WING_READONLY_PREFLIGHT_PROOF_ARTIFACT_SCHEMA
    ):
        violations.append("wing_readonly_preflight_plan_schema_mismatch")
    if plan.get("proof_runner") != "wing_readonly_preflight_runner":
        violations.append("wing_readonly_preflight_plan_runner_mismatch")
    if plan.get("artifact_root") != expected_artifact_root:
        violations.append("wing_readonly_preflight_plan_artifact_root_mismatch")
    if plan.get("plan_output") != str(plan_path):
        violations.append("wing_readonly_preflight_plan_output_path_mismatch")
    if expected_plan_output != WING_READONLY_PREFLIGHT_PLAN_OUTPUT:
        violations.append("wing_readonly_preflight_plan_default_output_mismatch")
    if plan.get("channels") != expected_channels:
        violations.append("wing_readonly_preflight_plan_channels_mismatch")
    if plan.get("operator_confirmed_channels") != expected_confirmed_channels:
        violations.append("wing_readonly_preflight_plan_operator_confirmation_mismatch")
    if plan.get("value_types") != expected_value_types:
        violations.append("wing_readonly_preflight_plan_value_types_mismatch")
    if plan.get("operator_channel_confirmation_required") is not True:
        violations.append("wing_readonly_preflight_plan_operator_confirmation_not_required")
    if plan.get("operator_session_confirmation_required") is not True:
        violations.append("wing_readonly_preflight_plan_session_confirmation_not_required")
    if plan.get("operator_confirmed_session_ready") is not True:
        violations.append("wing_readonly_preflight_plan_session_not_confirmed_ready")
    if plan.get("require_physical_readback") is not True:
        violations.append("wing_readonly_preflight_plan_physical_readback_not_required")
    if plan.get("selected_mvp_blocker") != "WING read-only connection proof":
        violations.append("wing_readonly_preflight_plan_selected_mvp_blocker_mismatch")
    if plan.get("selected_mvp_blocker_status") != "tooling_ready_waits_real_wing":
        violations.append("wing_readonly_preflight_plan_selected_mvp_blocker_status_mismatch")
    if plan.get("operator_required") is not True:
        violations.append("wing_readonly_preflight_plan_operator_required_not_true")
    if plan.get("expected_plan_required_for_real_run") is not True:
        violations.append(
            "wing_readonly_preflight_plan_expected_plan_required_not_true"
        )
    if plan.get("physical_preflight_requires_existing_backend_connection") is not True:
        violations.append(
            "wing_readonly_preflight_plan_existing_backend_connection_boundary_not_true"
        )
    if plan.get("backend_port_prestart_check_required") is not True:
        violations.append(
            "wing_readonly_preflight_plan_backend_port_prestart_check_not_required"
        )
    if plan.get("backend_port_prestart_check_must_pass_before_starting_backend") is not True:
        violations.append(
            "wing_readonly_preflight_plan_backend_port_prestart_check_boundary_not_true"
        )
    if (
        plan.get("backend_port_prestart_check_command")
        != WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND
    ):
        violations.append(
            "wing_readonly_preflight_plan_backend_port_prestart_check_command_mismatch"
        )
    if plan.get("backend_port_prestart_check_failure_detail_fields") != list(
        WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS
    ):
        violations.append(
            "wing_readonly_preflight_plan_backend_port_prestart_check_detail_fields_mismatch"
        )
    if plan.get("manual_review_required_before_readiness_update") is not True:
        violations.append(
            "wing_readonly_preflight_plan_manual_review_boundary_not_true"
        )
    if plan.get("request_id_policy") != "required_for_every_query_mixer_parameter_response":
        violations.append("wing_readonly_preflight_plan_request_id_policy_missing")

    for key in (
        "read_only",
    ):
        if plan.get(key) is not True:
            violations.append(f"wing_readonly_preflight_plan_{key}_not_true")
    for key in (
        "live_write",
        "contacts_backend",
        "queries_console",
        "execution_allowed_by_plan",
        "live_authority_granted",
    ):
        if plan.get(key) is not False:
            violations.append(f"wing_readonly_preflight_plan_{key}_not_false")

    allowed_request_types = plan.get("allowed_request_types")
    if allowed_request_types != [
        "get_operator_mode",
        "get_state",
        "query_mixer_parameter",
    ]:
        violations.append("wing_readonly_preflight_plan_allowed_request_types_mismatch")
        if not isinstance(allowed_request_types, list):
            allowed_request_types = []
    forbidden_request_types = plan.get("forbidden_request_types")
    if not isinstance(forbidden_request_types, list):
        forbidden_request_types = []
        violations.append("wing_readonly_preflight_plan_forbidden_request_types_missing")
    for required_forbidden in (
        "apply_operator_proposal",
        "set_gain",
        "set_fader",
        "set_eq",
        "set_compressor",
    ):
        if required_forbidden not in forbidden_request_types:
            violations.append(
                f"wing_readonly_preflight_plan_forbidden_{required_forbidden}_missing"
            )

    request_sequence = plan.get("request_sequence")
    if not isinstance(request_sequence, list):
        request_sequence = []
        violations.append("wing_readonly_preflight_plan_request_sequence_missing")
    if request_sequence != expected_requests:
        violations.append("wing_readonly_preflight_plan_request_sequence_mismatch")
    unsafe_requests = [
        item.get("type")
        for item in request_sequence
        if isinstance(item, Mapping) and item.get("type") not in allowed_request_types
    ]
    if unsafe_requests:
        violations.append("wing_readonly_preflight_plan_unsafe_request_type_present")

    expected_artifacts = plan.get("expected_artifacts")
    expected_artifact_paths = [
        f"{expected_artifact_root}/<run_id>/{WING_READONLY_PREFLIGHT_SUMMARY_ARTIFACT_NAME}",
        f"{expected_artifact_root}/<run_id>/{WING_READONLY_PREFLIGHT_EVENTS_ARTIFACT_NAME}",
    ]
    if expected_artifacts != expected_artifact_paths:
        violations.append("wing_readonly_preflight_plan_expected_artifacts_mismatch")

    fail_closed_conditions = plan.get("fail_closed_conditions")
    if not isinstance(fail_closed_conditions, list):
        fail_closed_conditions = []
        violations.append("wing_readonly_preflight_plan_fail_closed_conditions_missing")
    required_fail_closed = list(WING_READONLY_PREFLIGHT_PLAN_REQUIRED_FAIL_CLOSED_CONDITIONS)
    for condition in WING_READONLY_PREFLIGHT_PLAN_REQUIRED_FAIL_CLOSED_CONDITIONS:
        if condition not in fail_closed_conditions:
            violations.append(
                f"wing_readonly_preflight_plan_fail_closed_condition_{condition}_missing"
            )
    if fail_closed_conditions != required_fail_closed:
        violations.append("wing_readonly_preflight_plan_fail_closed_conditions_mismatch")

    post_run_review_commands = plan.get("post_run_review_commands")
    expected_post_run_review_commands = [
        _command_string(wing_readonly_preflight_review_command()),
        _command_string(wing_readonly_preflight_review_decision_command()),
    ]
    if not isinstance(post_run_review_commands, list):
        post_run_review_commands = []
        violations.append("wing_readonly_preflight_plan_post_run_review_commands_missing")
    if post_run_review_commands != expected_post_run_review_commands:
        violations.append("wing_readonly_preflight_plan_post_run_review_commands_mismatch")

    detail["plan_type"] = plan.get("type")
    detail["plan_status"] = plan.get("status")
    detail["plan_output"] = plan.get("plan_output")
    detail["expected_plan_output"] = str(plan_path)
    detail["channels"] = plan.get("channels") if isinstance(plan.get("channels"), list) else []
    detail["operator_confirmed_channels"] = (
        plan.get("operator_confirmed_channels")
        if isinstance(plan.get("operator_confirmed_channels"), list)
        else []
    )
    detail["operator_session_confirmation_required"] = plan.get(
        "operator_session_confirmation_required"
    )
    detail["operator_confirmed_session_ready"] = plan.get(
        "operator_confirmed_session_ready"
    )
    detail["value_types"] = (
        plan.get("value_types") if isinstance(plan.get("value_types"), list) else []
    )
    detail["request_count"] = len(request_sequence)
    detail["expected_request_count"] = len(expected_requests)
    detail["request_sequence_matches_expected"] = request_sequence == expected_requests
    detail["request_id_policy"] = plan.get("request_id_policy")
    detail["read_only"] = plan.get("read_only")
    detail["live_write"] = plan.get("live_write")
    detail["contacts_backend"] = plan.get("contacts_backend")
    detail["queries_console"] = plan.get("queries_console")
    detail["execution_allowed_by_plan"] = plan.get("execution_allowed_by_plan")
    detail["live_authority_granted"] = plan.get("live_authority_granted")
    detail["forbidden_request_types"] = list(forbidden_request_types)
    detail["selected_mvp_blocker"] = plan.get("selected_mvp_blocker")
    detail["selected_mvp_blocker_status"] = plan.get("selected_mvp_blocker_status")
    detail["operator_required"] = plan.get("operator_required")
    detail["expected_plan_required_for_real_run"] = plan.get(
        "expected_plan_required_for_real_run"
    )
    detail["physical_preflight_requires_existing_backend_connection"] = plan.get(
        "physical_preflight_requires_existing_backend_connection"
    )
    detail["backend_port_prestart_check_required"] = plan.get(
        "backend_port_prestart_check_required"
    )
    detail["backend_port_prestart_check_must_pass_before_starting_backend"] = plan.get(
        "backend_port_prestart_check_must_pass_before_starting_backend"
    )
    detail["backend_port_prestart_check_command"] = plan.get(
        "backend_port_prestart_check_command"
    )
    detail["backend_port_prestart_check_failure_detail_fields"] = plan.get(
        "backend_port_prestart_check_failure_detail_fields"
    )
    detail["manual_review_required_before_readiness_update"] = plan.get(
        "manual_review_required_before_readiness_update"
    )
    detail["fail_closed_conditions"] = list(fail_closed_conditions)
    detail["required_fail_closed_conditions"] = list(
        WING_READONLY_PREFLIGHT_PLAN_REQUIRED_FAIL_CLOSED_CONDITIONS
    )
    detail["post_run_review_commands"] = list(post_run_review_commands)
    detail["violations"] = violations
    finished = datetime.now(timezone.utc)
    return CheckResult(
        name="wing_readonly_preflight_plan_artifact_schema",
        status="passed" if not violations else "failed",
        started_at=started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        finished_at=finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration_sec=round((finished - started).total_seconds(), 3),
        detail=detail,
    )


def validate_wing_readonly_preflight_handoff_packet_artifact_snapshot(
    packet_path: Path,
    *,
    plan_path: Path | None = None,
) -> CheckResult:
    started = datetime.now(timezone.utc)
    expected_plan_path = plan_path or wing_readonly_preflight_plan_path_for_run(
        packet_path.parent
    )
    detail: dict[str, object] = {
        "policy": "wing_readonly_preflight_handoff_packet_is_readonly_operator_gate",
        "packet_path": str(packet_path),
        "expected_plan_output": str(expected_plan_path),
    }
    violations: list[str] = []

    payload, load_error = _json_object_from_path(packet_path)
    packet = payload or {}
    plan_artifact_result = validate_wing_readonly_preflight_plan_artifact_snapshot(
        expected_plan_path
    )
    plan_artifact_detail = (
        plan_artifact_result.detail
        if isinstance(plan_artifact_result.detail, Mapping)
        else {}
    )
    plan_artifact_violations = (
        plan_artifact_detail.get("violations")
        if isinstance(plan_artifact_detail.get("violations"), list)
        else []
    )
    commands = packet.get("commands") if isinstance(packet.get("commands"), Mapping) else {}
    operator_sequence = (
        packet.get("operator_sequence")
        if isinstance(packet.get("operator_sequence"), list)
        else []
    )
    supervised_prestart_gate = (
        packet.get("supervised_prestart_gate")
        if isinstance(packet.get("supervised_prestart_gate"), Mapping)
        else {}
    )
    backend_port_prestart_gate = (
        packet.get("backend_port_prestart_gate")
        if isinstance(packet.get("backend_port_prestart_gate"), Mapping)
        else {}
    )
    declared_expected_backend_port_prestart_gate = (
        packet.get("expected_backend_port_prestart_gate")
        if isinstance(packet.get("expected_backend_port_prestart_gate"), Mapping)
        else {}
    )
    declared_expected_supervised_prestart_gate = (
        packet.get("expected_supervised_prestart_gate")
        if isinstance(packet.get("expected_supervised_prestart_gate"), Mapping)
        else {}
    )
    mvp_blocker_anchor = (
        packet.get("mvp_blocker_anchor")
        if isinstance(packet.get("mvp_blocker_anchor"), Mapping)
        else {}
    )
    expected_mvp_blocker_anchor = wing_mvp_first_open_blocker_anchor()
    mvp_blocker_anchor_sha256 = (
        str(packet.get("mvp_blocker_anchor_sha256"))
        if isinstance(packet.get("mvp_blocker_anchor_sha256"), str)
        else ""
    )
    actual_mvp_blocker_anchor_sha256 = (
        wing_mvp_blocker_anchor_sha256(mvp_blocker_anchor)
        if mvp_blocker_anchor
        else ""
    )
    expected_mvp_blocker_anchor_sha256 = wing_mvp_blocker_anchor_sha256(
        expected_mvp_blocker_anchor
    )
    required_checklist = (
        packet.get("required_review_checklist_ids")
        if isinstance(packet.get("required_review_checklist_ids"), list)
        else []
    )
    blockers = (
        packet.get("blockers_not_closed_by_packet")
        if isinstance(packet.get("blockers_not_closed_by_packet"), list)
        else []
    )
    forbidden_request_types = (
        packet.get("forbidden_live_request_types")
        if isinstance(packet.get("forbidden_live_request_types"), list)
        else []
    )
    expected_preview_command = [
        "python3",
        WING_READONLY_PREFLIGHT_RUNNER_TOOL,
        "--url",
        WING_READONLY_PREFLIGHT_RUNNER_URL,
        "--channels",
        WING_READONLY_PREFLIGHT_RUNNER_CHANNELS,
        "--operator-confirm-channels",
        WING_READONLY_PREFLIGHT_RUNNER_CHANNELS,
        "--operator-confirm-session-ready",
        "--value-types",
        WING_READONLY_PREFLIGHT_RUNNER_VALUE_TYPES,
        "--require-physical-readback",
        "--artifact-root",
        WING_READONLY_PREFLIGHT_CAPTURE_ARTIFACT_ROOT,
        "--plan-only",
        "--plan-output",
        str(expected_plan_path),
    ]
    expected_capture_command = [
        "python3",
        WING_READONLY_PREFLIGHT_RUNNER_TOOL,
        "--url",
        WING_READONLY_PREFLIGHT_RUNNER_URL,
        "--channels",
        WING_READONLY_PREFLIGHT_RUNNER_CHANNELS,
        "--operator-confirm-channels",
        WING_READONLY_PREFLIGHT_RUNNER_CHANNELS,
        "--operator-confirm-session-ready",
        "--value-types",
        WING_READONLY_PREFLIGHT_RUNNER_VALUE_TYPES,
        "--require-physical-readback",
        "--artifact-root",
        WING_READONLY_PREFLIGHT_CAPTURE_ARTIFACT_ROOT,
        "--expected-plan",
        str(expected_plan_path),
    ]
    expected_commands = {
        "prestart_backend_port_check": (
            WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND
        ),
        "preview_readonly_request_plan": expected_preview_command,
        "capture_readonly_preflight": expected_capture_command,
        "review_artifact_and_write_pending_decision_template": (
            wing_readonly_preflight_review_command()
        ),
        "review_edited_decision_json": wing_readonly_preflight_review_decision_command(),
    }

    if load_error:
        violations.append(f"wing_readonly_preflight_handoff_packet_{load_error}")
    if plan_artifact_result.status != "passed":
        violations.append("wing_readonly_preflight_handoff_packet_plan_artifact_invalid")

    if packet.get("type") != "wing_readonly_preflight_handoff_packet":
        violations.append("wing_readonly_preflight_handoff_packet_type_mismatch")
    if packet.get("status") != "operator_hardware_step_required":
        violations.append("wing_readonly_preflight_handoff_packet_status_mismatch")
    if packet.get("url") != WING_READONLY_PREFLIGHT_RUNNER_URL:
        violations.append("wing_readonly_preflight_handoff_packet_url_mismatch")
    if packet.get("plan_output") != str(expected_plan_path):
        violations.append("wing_readonly_preflight_handoff_packet_plan_output_mismatch")
    if packet.get("artifact_root") != WING_READONLY_PREFLIGHT_CAPTURE_ARTIFACT_ROOT:
        violations.append("wing_readonly_preflight_handoff_packet_artifact_root_mismatch")
    if packet.get("decision_root") != WING_READONLY_PREFLIGHT_REVIEW_DECISION_ROOT:
        violations.append("wing_readonly_preflight_handoff_packet_decision_root_mismatch")
    if packet.get("channels") != _csv_ints(WING_READONLY_PREFLIGHT_RUNNER_CHANNELS):
        violations.append("wing_readonly_preflight_handoff_packet_channels_mismatch")
    if packet.get("value_types") != _csv_strings(WING_READONLY_PREFLIGHT_RUNNER_VALUE_TYPES):
        violations.append("wing_readonly_preflight_handoff_packet_value_types_mismatch")
    if packet.get("expected_run_artifact_glob") != (
        f"{WING_READONLY_PREFLIGHT_CAPTURE_ARTIFACT_ROOT}/<run_id>/"
        f"{WING_READONLY_PREFLIGHT_SUMMARY_ARTIFACT_NAME}"
    ):
        violations.append("wing_readonly_preflight_handoff_packet_summary_glob_mismatch")
    if packet.get("expected_event_log_glob") != (
        f"{WING_READONLY_PREFLIGHT_CAPTURE_ARTIFACT_ROOT}/<run_id>/"
        f"{WING_READONLY_PREFLIGHT_EVENTS_ARTIFACT_NAME}"
    ):
        violations.append("wing_readonly_preflight_handoff_packet_events_glob_mismatch")

    for key in (
        "read_only",
        "operator_required",
        "expected_plan_required_for_real_run",
        "physical_preflight_requires_existing_backend_connection",
        "backend_port_prestart_check_required",
        "backend_port_prestart_check_must_pass_before_starting_backend",
        "backend_port_prestart_check_must_be_first_operator_step",
        "manual_review_required_before_readiness_update",
    ):
        if packet.get(key) is not True:
            violations.append(f"wing_readonly_preflight_handoff_packet_{key}_not_true")
    backend_port_prestart_step_index = packet.get(
        "backend_port_prestart_check_operator_sequence_step_index"
    )
    if backend_port_prestart_step_index != 0:
        violations.append(
            "wing_readonly_preflight_handoff_packet_backend_port_prestart_step_index_mismatch"
        )
    if packet.get("backend_port_prestart_check_failure_detail_fields") != list(
        WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS
    ):
        violations.append(
            "wing_readonly_preflight_handoff_packet_backend_port_prestart_check_detail_fields_mismatch"
        )
    if packet.get("backend_port_prestart_check_command") != (
        WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND
    ):
        violations.append(
            "wing_readonly_preflight_handoff_packet_backend_port_prestart_check_command_mismatch"
        )
    for key in (
        "live_write",
        "mutates_console_state",
        "queries_console",
        "executes_tests",
        "execution_allowed_by_packet",
        "readiness_matrix_update_allowed_by_packet",
        "readiness_closure_allowed_by_packet",
        "live_authority_granted",
        "full_autonomous_ready",
    ):
        if packet.get(key) is not False:
            violations.append(f"wing_readonly_preflight_handoff_packet_{key}_not_false")

    for command_name, expected_command in expected_commands.items():
        if commands.get(command_name) != expected_command:
            violations.append(
                f"wing_readonly_preflight_handoff_packet_{command_name}_mismatch"
            )
    focused_validation = commands.get("focused_validation")
    if not isinstance(focused_validation, list) or "tests/test_no_wing_release_check.py" not in focused_validation:
        violations.append("wing_readonly_preflight_handoff_packet_focused_validation_missing")
    if "--plan-only" not in commands.get("preview_readonly_request_plan", []):
        violations.append("wing_readonly_preflight_handoff_packet_preview_plan_only_missing")
    if commands.get("prestart_backend_port_check") != (
        WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND
    ):
        violations.append(
            "wing_readonly_preflight_handoff_packet_prestart_backend_port_check_mismatch"
        )
    if "--operator-confirm-session-ready" not in commands.get(
        "preview_readonly_request_plan", []
    ):
        violations.append(
            "wing_readonly_preflight_handoff_packet_preview_session_confirmation_missing"
        )
    if "--require-physical-readback" not in commands.get("capture_readonly_preflight", []):
        violations.append(
            "wing_readonly_preflight_handoff_packet_capture_physical_readback_missing"
        )
    if "--operator-confirm-session-ready" not in commands.get(
        "capture_readonly_preflight", []
    ):
        violations.append(
            "wing_readonly_preflight_handoff_packet_capture_session_confirmation_missing"
        )
    if "--decision-template-output" not in commands.get(
        "review_artifact_and_write_pending_decision_template", []
    ):
        violations.append(
            "wing_readonly_preflight_handoff_packet_decision_template_output_missing"
        )

    if required_checklist != list(WING_READONLY_PREFLIGHT_REVIEW_MANUAL_CHECKLIST_IDS):
        violations.append("wing_readonly_preflight_handoff_packet_checklist_mismatch")
    required_blockers = list(
        WING_READONLY_PREFLIGHT_HANDOFF_REQUIRED_NOT_CLOSED_BLOCKERS
    )
    if blockers != required_blockers:
        violations.append("wing_readonly_preflight_handoff_packet_blockers_mismatch")
    for required_blocker in required_blockers:
        if required_blocker not in blockers:
            violations.append(
                f"wing_readonly_preflight_handoff_packet_blocker_{required_blocker}_missing"
            )
    for required_forbidden in WING_READONLY_PREFLIGHT_HANDOFF_REQUIRED_FORBIDDEN_REQUEST_TYPES:
        if required_forbidden not in forbidden_request_types:
            violations.append(
                "wing_readonly_preflight_handoff_packet_forbidden_"
                f"{required_forbidden}_missing"
            )
    if len(operator_sequence) < 8:
        violations.append("wing_readonly_preflight_handoff_packet_operator_sequence_short")
    backend_port_prestart_step = str(operator_sequence[0]) if operator_sequence else ""
    backend_port_prestart_step_is_first = (
        bool(backend_port_prestart_step)
        and "Before starting the intended backend/WING session" in backend_port_prestart_step
        and "prestart_backend_port_check" in backend_port_prestart_step
    )
    if not backend_port_prestart_step_is_first:
        violations.append(
            "wing_readonly_preflight_handoff_packet_backend_port_prestart_step_not_first"
        )
    expected_backend_port_prestart_gate = {
        "required": True,
        "operator_sequence_step_index": 0,
        "operator_sequence_step_must_be_first": True,
        "operator_sequence_step": backend_port_prestart_step,
        "required_markers": [
            "Before starting the intended backend/WING session",
            "prestart_backend_port_check",
            "unexpected listeners",
        ],
        "runs_command": "prestart_backend_port_check",
        "must_pass_before_backend_wing_session": True,
        "grants_live_authority": False,
    }
    if backend_port_prestart_gate != expected_backend_port_prestart_gate:
        violations.append(
            "wing_readonly_preflight_handoff_packet_backend_port_prestart_gate_contract_mismatch"
        )
    if declared_expected_backend_port_prestart_gate != expected_backend_port_prestart_gate:
        violations.append(
            "wing_readonly_preflight_handoff_packet_expected_backend_port_prestart_gate_contract_mismatch"
        )
    operator_sequence_supervised_prestart_gate_step_index = next(
        (
            index
            for index, item in enumerate(operator_sequence)
            if all(
                marker in str(item)
                for marker in WING_READONLY_PREFLIGHT_HANDOFF_REQUIRED_SUPERVISED_PROOF_GATE_MARKERS
            )
        ),
        None,
    )
    operator_sequence_supervised_prestart_gate_step = (
        str(operator_sequence[operator_sequence_supervised_prestart_gate_step_index])
        if operator_sequence_supervised_prestart_gate_step_index is not None
        else ""
    )
    operator_sequence_has_supervised_prestart_gate = bool(
        operator_sequence_supervised_prestart_gate_step
    )
    operator_sequence_supervised_prestart_gate_is_final = (
        operator_sequence_supervised_prestart_gate_step_index == len(operator_sequence) - 1
        if operator_sequence_has_supervised_prestart_gate
        else False
    )
    if not operator_sequence_has_supervised_prestart_gate:
        violations.append(
            "wing_readonly_preflight_handoff_packet_supervised_prestart_gate_missing"
        )
    elif not operator_sequence_supervised_prestart_gate_is_final:
        violations.append(
            "wing_readonly_preflight_handoff_packet_supervised_prestart_gate_not_final"
        )
    operator_sequence_mvp_blocker_anchor_digest_review_step_index = next(
        (
            index
            for index, item in enumerate(operator_sequence)
            if "mvp_blocker_anchor" in str(item)
            and "mvp_blocker_anchor_sha256" in str(item)
            and "current first open blocker" in str(item)
        ),
        None,
    )
    operator_sequence_mvp_blocker_anchor_digest_review_step = (
        str(operator_sequence[operator_sequence_mvp_blocker_anchor_digest_review_step_index])
        if operator_sequence_mvp_blocker_anchor_digest_review_step_index is not None
        else ""
    )
    operator_sequence_has_mvp_blocker_anchor_digest_review = bool(
        operator_sequence_mvp_blocker_anchor_digest_review_step
    )
    if not operator_sequence_has_mvp_blocker_anchor_digest_review:
        violations.append(
            "wing_readonly_preflight_handoff_packet_mvp_blocker_anchor_digest_review_step_missing"
        )
    expected_supervised_prestart_gate = {
        "required": True,
        "operator_sequence_step_index": operator_sequence_supervised_prestart_gate_step_index,
        "operator_sequence_step_must_be_final": True,
        "operator_sequence_step": operator_sequence_supervised_prestart_gate_step,
        "required_markers": list(
            WING_READONLY_PREFLIGHT_HANDOFF_REQUIRED_SUPERVISED_PROOF_GATE_MARKERS
        ),
        "unlocks_runbook": "Docs/runbooks/WING_SUPERVISED_FADER_GAIN_PROOF.md",
        "grants_live_authority": False,
        "requires_operator_decision_status": "accepted_decision_ready",
        "requires_accepted_candidate_prestart_trace": True,
    }
    if supervised_prestart_gate != expected_supervised_prestart_gate:
        violations.append(
            "wing_readonly_preflight_handoff_packet_supervised_prestart_gate_contract_mismatch"
        )
    if declared_expected_supervised_prestart_gate != expected_supervised_prestart_gate:
        violations.append(
            "wing_readonly_preflight_handoff_packet_expected_supervised_prestart_gate_contract_mismatch"
        )
    if (
        packet.get("next_manual_artifact_required")
        != WING_READONLY_PREFLIGHT_NEXT_MANUAL_ARTIFACT_REQUIRED
    ):
        violations.append(
            "wing_readonly_preflight_handoff_packet_next_manual_artifact_mismatch"
        )
    if not mvp_blocker_anchor:
        violations.append("wing_readonly_preflight_handoff_packet_mvp_blocker_anchor_missing")
    elif dict(mvp_blocker_anchor) != expected_mvp_blocker_anchor:
        violations.append("wing_readonly_preflight_handoff_packet_mvp_blocker_anchor_mismatch")
    if mvp_blocker_anchor.get("packet_does_not_close_blocker") is not True:
        violations.append(
            "wing_readonly_preflight_handoff_packet_mvp_blocker_anchor_closes_blocker"
        )
    if not mvp_blocker_anchor_sha256:
        violations.append("wing_readonly_preflight_handoff_packet_mvp_blocker_anchor_sha256_missing")
    elif mvp_blocker_anchor_sha256 != actual_mvp_blocker_anchor_sha256:
        violations.append("wing_readonly_preflight_handoff_packet_mvp_blocker_anchor_sha256_mismatch")
    if mvp_blocker_anchor_sha256 != expected_mvp_blocker_anchor_sha256:
        violations.append(
            "wing_readonly_preflight_handoff_packet_expected_mvp_blocker_anchor_sha256_mismatch"
        )

    detail["packet_type"] = packet.get("type")
    detail["packet_status"] = packet.get("status")
    detail["plan_artifact_status"] = plan_artifact_result.status
    detail["plan_artifact_violations"] = list(plan_artifact_violations)
    detail["read_only"] = packet.get("read_only")
    detail["live_write"] = packet.get("live_write")
    detail["mutates_console_state"] = packet.get("mutates_console_state")
    detail["queries_console"] = packet.get("queries_console")
    detail["executes_tests"] = packet.get("executes_tests")
    detail["operator_required"] = packet.get("operator_required")
    detail["expected_plan_required_for_real_run"] = packet.get(
        "expected_plan_required_for_real_run"
    )
    detail["physical_preflight_requires_existing_backend_connection"] = packet.get(
        "physical_preflight_requires_existing_backend_connection"
    )
    detail["backend_port_prestart_check_required"] = packet.get(
        "backend_port_prestart_check_required"
    )
    detail["backend_port_prestart_check_must_pass_before_starting_backend"] = packet.get(
        "backend_port_prestart_check_must_pass_before_starting_backend"
    )
    detail["backend_port_prestart_check_operator_sequence_step_index"] = packet.get(
        "backend_port_prestart_check_operator_sequence_step_index"
    )
    detail["backend_port_prestart_check_must_be_first_operator_step"] = packet.get(
        "backend_port_prestart_check_must_be_first_operator_step"
    )
    detail["backend_port_prestart_check_operator_sequence_step"] = (
        backend_port_prestart_step
    )
    detail["backend_port_prestart_check_operator_sequence_step_is_first"] = (
        backend_port_prestart_step_is_first
    )
    detail["backend_port_prestart_gate"] = dict(backend_port_prestart_gate)
    detail["expected_backend_port_prestart_gate"] = dict(
        expected_backend_port_prestart_gate
    )
    detail["declared_expected_backend_port_prestart_gate"] = dict(
        declared_expected_backend_port_prestart_gate
    )
    detail["backend_port_prestart_check_command"] = packet.get(
        "backend_port_prestart_check_command"
    )
    detail["backend_port_prestart_check_failure_detail_fields"] = packet.get(
        "backend_port_prestart_check_failure_detail_fields"
    )
    detail["manual_review_required_before_readiness_update"] = packet.get(
        "manual_review_required_before_readiness_update"
    )
    detail["execution_allowed_by_packet"] = packet.get("execution_allowed_by_packet")
    detail["readiness_matrix_update_allowed_by_packet"] = packet.get(
        "readiness_matrix_update_allowed_by_packet"
    )
    detail["readiness_closure_allowed_by_packet"] = packet.get(
        "readiness_closure_allowed_by_packet"
    )
    detail["live_authority_granted"] = packet.get("live_authority_granted")
    detail["full_autonomous_ready"] = packet.get("full_autonomous_ready")
    detail["next_manual_artifact_required"] = packet.get(
        "next_manual_artifact_required"
    )
    detail["url"] = packet.get("url")
    detail["plan_output"] = packet.get("plan_output")
    detail["artifact_root"] = packet.get("artifact_root")
    detail["channels"] = packet.get("channels") if isinstance(packet.get("channels"), list) else []
    detail["value_types"] = (
        packet.get("value_types") if isinstance(packet.get("value_types"), list) else []
    )
    detail["command_names"] = sorted(commands)
    detail["preview_command_uses_plan_only"] = (
        "--plan-only" in commands.get("preview_readonly_request_plan", [])
    )
    detail["capture_command_requires_physical_readback"] = (
        "--require-physical-readback" in commands.get("capture_readonly_preflight", [])
    )
    detail["prestart_backend_port_check_command"] = commands.get(
        "prestart_backend_port_check"
    )
    detail["review_command_writes_decision_template"] = (
        "--decision-template-output"
        in commands.get("review_artifact_and_write_pending_decision_template", [])
    )
    detail["required_review_checklist_ids"] = list(required_checklist)
    detail["forbidden_live_request_types"] = list(forbidden_request_types)
    detail["required_forbidden_live_request_types"] = list(
        WING_READONLY_PREFLIGHT_HANDOFF_REQUIRED_FORBIDDEN_REQUEST_TYPES
    )
    detail["blockers_not_closed_by_packet"] = list(blockers)
    detail["required_not_closed_blockers"] = list(
        WING_READONLY_PREFLIGHT_HANDOFF_REQUIRED_NOT_CLOSED_BLOCKERS
    )
    detail["operator_sequence_count"] = len(operator_sequence)
    detail["operator_sequence_has_mvp_blocker_anchor_digest_review"] = (
        operator_sequence_has_mvp_blocker_anchor_digest_review
    )
    detail["operator_sequence_mvp_blocker_anchor_digest_review_step"] = (
        operator_sequence_mvp_blocker_anchor_digest_review_step
    )
    detail["operator_sequence_mvp_blocker_anchor_digest_review_step_index"] = (
        operator_sequence_mvp_blocker_anchor_digest_review_step_index
    )
    detail["operator_sequence_has_supervised_prestart_gate"] = (
        operator_sequence_has_supervised_prestart_gate
    )
    detail["operator_sequence_supervised_prestart_gate_step"] = (
        operator_sequence_supervised_prestart_gate_step
    )
    detail["operator_sequence_supervised_prestart_gate_step_index"] = (
        operator_sequence_supervised_prestart_gate_step_index
    )
    detail["operator_sequence_supervised_prestart_gate_is_final"] = (
        operator_sequence_supervised_prestart_gate_is_final
    )
    detail["supervised_prestart_gate"] = dict(supervised_prestart_gate)
    detail["expected_supervised_prestart_gate"] = dict(expected_supervised_prestart_gate)
    detail["declared_expected_supervised_prestart_gate"] = dict(
        declared_expected_supervised_prestart_gate
    )
    detail["required_supervised_prestart_gate_markers"] = list(
        WING_READONLY_PREFLIGHT_HANDOFF_REQUIRED_SUPERVISED_PROOF_GATE_MARKERS
    )
    detail["mvp_blocker_anchor"] = dict(mvp_blocker_anchor)
    detail["expected_mvp_blocker_anchor"] = dict(expected_mvp_blocker_anchor)
    detail["mvp_blocker_anchor_matches_current_first_open"] = (
        dict(mvp_blocker_anchor) == expected_mvp_blocker_anchor
    )
    detail["mvp_blocker_anchor_sha256"] = mvp_blocker_anchor_sha256
    detail["actual_mvp_blocker_anchor_sha256"] = actual_mvp_blocker_anchor_sha256
    detail["expected_mvp_blocker_anchor_sha256"] = expected_mvp_blocker_anchor_sha256
    detail["mvp_blocker_anchor_sha256_matches_anchor"] = (
        mvp_blocker_anchor_sha256 == actual_mvp_blocker_anchor_sha256
    )
    detail["mvp_blocker_anchor_sha256_matches_current_first_open"] = (
        mvp_blocker_anchor_sha256 == expected_mvp_blocker_anchor_sha256
    )
    detail["violations"] = violations
    finished = datetime.now(timezone.utc)
    return CheckResult(
        name="wing_readonly_preflight_handoff_packet_artifact_schema",
        status="passed" if not violations else "failed",
        started_at=started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        finished_at=finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration_sec=round((finished - started).total_seconds(), 3),
        detail=detail,
    )


def validate_read_surface_physical_readback_acceptance_template_artifact_snapshot(
    template_path: Path,
) -> CheckResult:
    started = datetime.now(timezone.utc)
    detail: dict[str, object] = {
        "policy": "read_surface_physical_readback_acceptance_template_artifact_is_readonly_operator_gate",
        "template_path": str(template_path),
    }
    violations: list[str] = []

    payload, load_error = _json_object_from_path(template_path)
    template = payload or {}
    records = (
        template.get("template_records")
        if isinstance(template.get("template_records"), list)
        else []
    )
    required_edits = (
        template.get("required_operator_edits")
        if isinstance(template.get("required_operator_edits"), list)
        else []
    )
    blocked_by = (
        template.get("blocked_by")
        if isinstance(template.get("blocked_by"), list)
        else []
    )
    operator_sequence = (
        template.get("operator_sequence")
        if isinstance(template.get("operator_sequence"), list)
        else []
    )
    commands = (
        template.get("commands") if isinstance(template.get("commands"), Mapping) else {}
    )

    if load_error:
        violations.append(
            f"read_surface_physical_readback_acceptance_template_artifact_{load_error}"
        )
    if template.get("type") != "read_surface_physical_readback_fixture_acceptance_template":
        violations.append("acceptance_template_artifact_type_mismatch")
    if template.get("status") not in (
        "operator_candidate_missing",
        "operator_acceptance_template_ready",
    ):
        violations.append("acceptance_template_artifact_status_unknown")
    if not isinstance(template.get("source_queue_status"), str):
        violations.append("acceptance_template_artifact_source_queue_status_missing")

    for key in ("read_only", "operator_required"):
        if template.get(key) is not True:
            violations.append(f"acceptance_template_artifact_{key}_not_true")
    for key in (
        "live_write",
        "mutates_console_state",
        "queries_console",
        "executes_tests",
        "execution_allowed_by_template",
        "fixture_closure_allowed_by_template",
        "live_widening_allowed",
        "live_authority_granted",
        "full_autonomous_ready",
    ):
        if template.get(key) is not False:
            violations.append(f"acceptance_template_artifact_{key}_not_false")

    review_candidate_count = template.get("review_candidate_count")
    template_record_count = template.get("template_record_count")
    if not isinstance(review_candidate_count, int):
        violations.append("acceptance_template_artifact_review_candidate_count_missing")
        review_candidate_count = 0
    if not isinstance(template_record_count, int) or template_record_count != len(records):
        violations.append("acceptance_template_artifact_record_count_mismatch")
        template_record_count = len(records)
    if template.get("status") == "operator_candidate_missing" and template_record_count != 0:
        violations.append("acceptance_template_artifact_missing_status_has_records")
    if template.get("status") == "operator_acceptance_template_ready" and template_record_count < 1:
        violations.append("acceptance_template_artifact_ready_status_has_no_records")
    for required_edit in ("decision", "reviewed_by", "reviewed_at", "review_notes"):
        if required_edit not in required_edits:
            violations.append(
                f"acceptance_template_artifact_required_edit_{required_edit}_missing"
            )
    if len(operator_sequence) < 5:
        violations.append("acceptance_template_artifact_operator_sequence_short")
    focused_validation = commands.get("focused_validation")
    if not isinstance(focused_validation, list) or (
        "tests/test_read_surface_physical_readback_fixture_acceptance_ledger.py"
        not in focused_validation
    ):
        violations.append("acceptance_template_artifact_focused_validation_missing")

    candidate_trace_count = 0
    record_unsafe_flags: list[str] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            violations.append(f"acceptance_template_artifact_record_{index}_not_object")
            continue
        candidate_trace = (
            record.get("candidate_trace")
            if isinstance(record.get("candidate_trace"), Mapping)
            else None
        )
        if candidate_trace is None:
            violations.append(
                f"acceptance_template_artifact_record_{index}_candidate_trace_missing"
            )
        else:
            candidate_trace_count += 1
            for key in (
                "source_queue_method_id",
                "source_queue_status",
                "candidate_status",
                "intake_status",
            ):
                if not isinstance(candidate_trace.get(key), str) or not candidate_trace.get(
                    key
                ):
                    violations.append(
                        "acceptance_template_artifact_record_"
                        f"{index}_candidate_trace_{key}_missing"
                    )
            artifact_source_id = record.get("artifact_source_id")
            trace_artifact_sources = candidate_trace.get("artifact_source_ids")
            if candidate_trace.get("source_queue_method_id") != template.get(
                "source_queue_method_id"
            ):
                violations.append(
                    "acceptance_template_artifact_record_"
                    f"{index}_candidate_trace_source_queue_method_id_mismatch"
                )
            if candidate_trace.get("source_queue_status") != template.get(
                "source_queue_status"
            ):
                violations.append(
                    "acceptance_template_artifact_record_"
                    f"{index}_candidate_trace_source_queue_status_mismatch"
                )
            if not isinstance(trace_artifact_sources, list):
                violations.append(
                    "acceptance_template_artifact_record_"
                    f"{index}_candidate_trace_artifact_sources_missing"
                )
            elif artifact_source_id not in trace_artifact_sources:
                violations.append(
                    "acceptance_template_artifact_record_"
                    f"{index}_candidate_trace_artifact_source_missing"
                )
            valid_candidate_count = candidate_trace.get("valid_candidate_count")
            if (
                not isinstance(valid_candidate_count, int)
                or isinstance(valid_candidate_count, bool)
                or valid_candidate_count < 1
            ):
                violations.append(
                    "acceptance_template_artifact_record_"
                    f"{index}_candidate_trace_valid_candidate_count_missing"
                )
            invalid_candidate_count = candidate_trace.get("invalid_candidate_count")
            if (
                not isinstance(invalid_candidate_count, int)
                or isinstance(invalid_candidate_count, bool)
                or invalid_candidate_count != 0
            ):
                violations.append(
                    "acceptance_template_artifact_record_"
                    f"{index}_candidate_trace_invalid_candidate_count_not_zero"
                )
            for key in ("target_consistency", "accepted_as_physical_fixture_candidate"):
                if candidate_trace.get(key) is not True:
                    violations.append(
                        "acceptance_template_artifact_record_"
                        f"{index}_candidate_trace_{key}_not_true"
                    )
            if candidate_trace.get("candidate_source_id_missing") is not False:
                violations.append(
                    "acceptance_template_artifact_record_"
                    f"{index}_candidate_trace_candidate_source_id_missing_not_false"
                )
            for key in (
                "fixture_closure_allowed_by_queue",
                "live_authority_granted",
                "full_autonomous_ready",
            ):
                if candidate_trace.get(key) is not False:
                    violations.append(
                        "acceptance_template_artifact_record_"
                        f"{index}_candidate_trace_{key}_not_false"
                    )
        for key in (
            "review_id",
            "surface_id",
            "artifact_source_id",
            "required_contract_method_id",
        ):
            if not isinstance(record.get(key), str) or not record.get(key):
                violations.append(
                    f"acceptance_template_artifact_record_{index}_{key}_missing"
                )
        if record.get("decision") != "<accepted|rejected>":
            violations.append(
                f"acceptance_template_artifact_record_{index}_decision_not_pending"
            )
        if record.get("reviewed_by") != "<operator-name>":
            violations.append(
                f"acceptance_template_artifact_record_{index}_reviewed_by_not_pending"
            )
        reviewed_at = record.get("reviewed_at")
        if not isinstance(reviewed_at, str) or not reviewed_at.startswith(
            "<UTC ISO timestamp"
        ):
            violations.append(
                f"acceptance_template_artifact_record_{index}_reviewed_at_not_pending"
            )
        for key in (
            "live_write",
            "mutates_console_state",
            "queries_console",
            "executes_tests",
            "execution_allowed_by_acceptance",
            "fixture_closure_allowed_by_acceptance",
            "fixture_closure_allowed",
            "live_widening_allowed",
            "live_authority_granted",
            "full_autonomous_ready",
        ):
            if record.get(key) is not False:
                flag = f"record_{index}_{key}"
                record_unsafe_flags.append(flag)
                violations.append(
                    f"acceptance_template_artifact_{flag}_not_false"
                )

    if not blocked_by:
        violations.append("acceptance_template_artifact_blocked_by_missing")

    detail["template_type"] = template.get("type")
    detail["template_status"] = template.get("status")
    detail["source_queue_status"] = template.get("source_queue_status")
    detail["selected_surface_id"] = template.get("selected_surface_id")
    detail["read_only"] = template.get("read_only")
    detail["live_write"] = template.get("live_write")
    detail["mutates_console_state"] = template.get("mutates_console_state")
    detail["queries_console"] = template.get("queries_console")
    detail["executes_tests"] = template.get("executes_tests")
    detail["operator_required"] = template.get("operator_required")
    detail["execution_allowed_by_template"] = template.get(
        "execution_allowed_by_template"
    )
    detail["fixture_closure_allowed_by_template"] = template.get(
        "fixture_closure_allowed_by_template"
    )
    detail["live_widening_allowed"] = template.get("live_widening_allowed")
    detail["live_authority_granted"] = template.get("live_authority_granted")
    detail["full_autonomous_ready"] = template.get("full_autonomous_ready")
    detail["review_candidate_count"] = review_candidate_count
    detail["template_record_count"] = template_record_count
    detail["candidate_trace_count"] = candidate_trace_count
    detail["required_operator_edits"] = list(required_edits)
    detail["operator_sequence_count"] = len(operator_sequence)
    detail["blocked_by"] = list(blocked_by)
    detail["record_unsafe_flags"] = record_unsafe_flags
    detail["violations"] = violations
    finished = datetime.now(timezone.utc)
    return CheckResult(
        name="read_surface_physical_readback_acceptance_template_artifact_schema",
        status="passed" if not violations else "failed",
        started_at=started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        finished_at=finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration_sec=round((finished - started).total_seconds(), 3),
        detail=detail,
    )


def validate_read_surface_physical_readback_artifact_review_queue_artifact_snapshot(
    queue_path: Path,
) -> CheckResult:
    started = datetime.now(timezone.utc)
    detail: dict[str, object] = {
        "policy": "physical_readback_artifact_review_queue_artifact_is_read_only_operator_gate",
        "queue_path": str(queue_path),
    }
    violations: list[str] = []

    payload, load_error = _json_object_from_path(queue_path)
    queue = payload or {}
    safety = queue.get("safety") if isinstance(queue.get("safety"), Mapping) else {}
    summary = queue.get("summary") if isinstance(queue.get("summary"), Mapping) else {}
    items = queue.get("queue_items") if isinstance(queue.get("queue_items"), list) else []
    operator_sequence = (
        queue.get("operator_sequence")
        if isinstance(queue.get("operator_sequence"), list)
        else []
    )
    commands = queue.get("commands") if isinstance(queue.get("commands"), Mapping) else {}

    if load_error:
        violations.append(
            f"read_surface_physical_readback_artifact_review_queue_artifact_{load_error}"
        )
    if queue.get("type") != "read_surface_physical_readback_artifact_review_queue":
        violations.append("artifact_review_queue_artifact_type_mismatch")
    if queue.get("method_id") != READ_SURFACE_PHYSICAL_READBACK_ARTIFACT_REVIEW_QUEUE_METHOD_ID:
        violations.append("artifact_review_queue_artifact_method_id_mismatch")
    if queue.get("status") not in READ_SURFACE_PHYSICAL_READBACK_ARTIFACT_REVIEW_QUEUE_STATUSES:
        violations.append("artifact_review_queue_artifact_status_unknown")

    for key in ("read_only", "operator_required"):
        if safety.get(key) is not True:
            violations.append(f"artifact_review_queue_artifact_{key}_not_true")
    for key in (
        "live_write",
        "mutates_console_state",
        "queries_console",
        "executes_tests",
        "execution_allowed_by_queue",
        "fixture_closure_allowed_by_queue",
        "live_widening_allowed",
        "live_authority_granted",
        "full_autonomous_ready",
    ):
        if safety.get(key) is not False:
            violations.append(f"artifact_review_queue_artifact_{key}_not_false")

    intake_surface_count = summary.get("intake_surface_count")
    available_intake_count = summary.get("available_intake_count")
    artifact_count = summary.get("artifact_count")
    valid_candidate_count = summary.get("valid_candidate_count")
    invalid_candidate_count = summary.get("invalid_candidate_count")
    review_candidate_count = summary.get("review_candidate_count")
    unsafe_authority_claim_count = summary.get("unsafe_authority_claim_count")
    candidate_source_id_missing_count = summary.get("candidate_source_id_missing_count")
    target_inconsistent_surface_count = summary.get("target_inconsistent_surface_count")
    missing_artifact_surface_count = summary.get("missing_artifact_surface_count")
    expected_intake_count = len(INTAKE_SPECS)
    if not isinstance(intake_surface_count, int) or intake_surface_count != len(items):
        violations.append("artifact_review_queue_artifact_intake_count_mismatch")
        intake_surface_count = len(items)
    if intake_surface_count != expected_intake_count:
        violations.append("artifact_review_queue_artifact_expected_intake_count_mismatch")
    if not isinstance(available_intake_count, int):
        violations.append("artifact_review_queue_artifact_available_count_missing")
        available_intake_count = 0
    if not isinstance(artifact_count, int):
        violations.append("artifact_review_queue_artifact_artifact_count_missing")
        artifact_count = 0
    if not isinstance(valid_candidate_count, int):
        violations.append("artifact_review_queue_artifact_valid_count_missing")
        valid_candidate_count = 0
    if not isinstance(invalid_candidate_count, int):
        violations.append("artifact_review_queue_artifact_invalid_count_missing")
        invalid_candidate_count = 0
    if not isinstance(review_candidate_count, int):
        violations.append("artifact_review_queue_artifact_review_count_missing")
        review_candidate_count = 0
    if not isinstance(unsafe_authority_claim_count, int):
        violations.append("artifact_review_queue_artifact_unsafe_count_missing")
        unsafe_authority_claim_count = 0
    if not isinstance(candidate_source_id_missing_count, int):
        violations.append("artifact_review_queue_artifact_candidate_source_missing_count_missing")
        candidate_source_id_missing_count = 0
    if not isinstance(target_inconsistent_surface_count, int):
        violations.append("artifact_review_queue_artifact_target_inconsistent_count_missing")
        target_inconsistent_surface_count = 0
    if not isinstance(missing_artifact_surface_count, int):
        violations.append("artifact_review_queue_artifact_missing_count_missing")
        missing_artifact_surface_count = 0

    item_artifact_count = 0
    item_valid_count = 0
    item_invalid_count = 0
    item_review_count = 0
    item_unsafe_count = 0
    item_candidate_source_missing_count = 0
    item_target_inconsistent_count = 0
    item_missing_count = 0
    item_method_ids: list[str] = []
    item_unsafe_flags: list[str] = []
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            violations.append(f"artifact_review_queue_artifact_item_{index}_not_object")
            continue
        method_id = item.get("method_id")
        if not isinstance(method_id, str) or not method_id:
            violations.append(f"artifact_review_queue_artifact_item_{index}_method_id_missing")
        else:
            item_method_ids.append(method_id)
        for key in (
            "surface_id",
            "remediation_action_id",
            "required_contract_method_id",
            "status",
        ):
            if not isinstance(item.get(key), str) or not item.get(key):
                violations.append(
                    f"artifact_review_queue_artifact_item_{index}_{key}_missing"
                )
        for key in (
            "fixture_closure_allowed_by_queue",
            "fixture_closure_allowed_by_intake",
            "live_authority_granted",
            "full_autonomous_ready",
        ):
            if item.get(key) is not False:
                violations.append(
                    f"artifact_review_queue_artifact_item_{index}_{key}_not_false"
                )
        if item.get("operator_review_required") is not True:
            violations.append(
                f"artifact_review_queue_artifact_item_{index}_operator_review_not_true"
            )
        if item.get("physical_fixture_gap_remains") is not True:
            violations.append(
                f"artifact_review_queue_artifact_item_{index}_physical_gap_not_true"
            )
        item_artifact_count += int(item.get("artifact_count") or 0)
        item_valid_count += int(item.get("valid_candidate_count") or 0)
        item_invalid_count += int(item.get("invalid_candidate_count") or 0)
        if item.get("accepted_as_physical_fixture_candidate") is True:
            item_review_count += 1
            artifact_source_ids = item.get("artifact_source_ids")
            if not isinstance(artifact_source_ids, list) or not artifact_source_ids:
                violations.append(
                    "artifact_review_queue_artifact_item_"
                    f"{index}_review_candidate_source_ids_missing"
                )
        if item.get("candidate_source_id_missing") is True:
            item_candidate_source_missing_count += 1
        unsafe_flags = item.get("unsafe_authority_flags")
        if isinstance(unsafe_flags, list) and unsafe_flags:
            item_unsafe_count += 1
            item_unsafe_flags.extend(str(flag) for flag in unsafe_flags)
        if item.get("target_consistency") is not True:
            item_target_inconsistent_count += 1
        if item.get("status") == "operator_artifact_missing":
            item_missing_count += 1

    if item_method_ids != [str(spec["method_id"]) for spec in INTAKE_SPECS]:
        violations.append("artifact_review_queue_artifact_method_sequence_mismatch")
    if item_artifact_count != artifact_count:
        violations.append("artifact_review_queue_artifact_item_artifact_count_mismatch")
    if item_valid_count != valid_candidate_count:
        violations.append("artifact_review_queue_artifact_item_valid_count_mismatch")
    if item_invalid_count != invalid_candidate_count:
        violations.append("artifact_review_queue_artifact_item_invalid_count_mismatch")
    if item_review_count != review_candidate_count:
        violations.append("artifact_review_queue_artifact_item_review_count_mismatch")
    if item_unsafe_count != unsafe_authority_claim_count:
        violations.append("artifact_review_queue_artifact_item_unsafe_count_mismatch")
    if item_candidate_source_missing_count != candidate_source_id_missing_count:
        violations.append(
            "artifact_review_queue_artifact_item_candidate_source_missing_count_mismatch"
        )
    if item_target_inconsistent_count != target_inconsistent_surface_count:
        violations.append(
            "artifact_review_queue_artifact_item_target_inconsistent_count_mismatch"
        )
    if item_missing_count != missing_artifact_surface_count:
        violations.append("artifact_review_queue_artifact_item_missing_count_mismatch")

    for key in (
        "operator_review_required",
        "physical_fixture_gap_remains",
    ):
        if summary.get(key) is not True:
            violations.append(f"artifact_review_queue_artifact_{key}_not_true")
    for key in (
        "execution_allowed_by_queue",
        "fixture_closure_allowed_by_queue",
        "live_widening_allowed",
        "live_authority_granted",
        "full_autonomous_ready",
    ):
        if summary.get(key) is not False:
            violations.append(f"artifact_review_queue_artifact_summary_{key}_not_false")
    blocked_by = summary.get("blocked_by")
    if not isinstance(blocked_by, list) or not blocked_by:
        violations.append("artifact_review_queue_artifact_blocked_by_missing")
        blocked_by = []
    if unsafe_authority_claim_count:
        violations.append("artifact_review_queue_artifact_unsafe_authority_claim_present")
    if candidate_source_id_missing_count:
        violations.append("artifact_review_queue_artifact_candidate_source_id_missing_present")
    if target_inconsistent_surface_count:
        violations.append("artifact_review_queue_artifact_target_inconsistent_surface_present")
    if invalid_candidate_count:
        violations.append("artifact_review_queue_artifact_invalid_candidate_present")

    if len(operator_sequence) < 5:
        violations.append("artifact_review_queue_artifact_operator_sequence_short")
    focused_validation = commands.get("focused_validation")
    if not isinstance(focused_validation, list):
        violations.append("artifact_review_queue_artifact_focused_validation_missing")
        focused_validation = []
    else:
        for test_path in (
            "tests/test_read_surface_physical_readback_artifact_review_queue.py",
            "tests/test_no_wing_release_check.py",
        ):
            if test_path not in focused_validation:
                violations.append(
                    "artifact_review_queue_artifact_focused_validation_missing:"
                    f"{test_path}"
                )

    review_candidate_surface_ids = summary.get("review_candidate_surface_ids")
    invalid_surface_ids = summary.get("invalid_surface_ids")
    missing_surface_ids = summary.get("missing_artifact_surface_ids")
    unsafe_surface_ids = summary.get("unsafe_authority_surface_ids")
    candidate_source_missing_surface_ids = summary.get(
        "candidate_source_id_missing_surface_ids"
    )
    target_inconsistent_surface_ids = summary.get("target_inconsistent_surface_ids")
    queue_item_method_ids = summary.get("queue_item_method_ids")
    promotion_families = summary.get("promotion_families_requiring_surface")
    detail["queue_type"] = queue.get("type")
    detail["queue_status"] = queue.get("status")
    detail["method_id"] = queue.get("method_id")
    detail["read_only"] = safety.get("read_only")
    detail["live_write"] = safety.get("live_write")
    detail["mutates_console_state"] = safety.get("mutates_console_state")
    detail["queries_console"] = safety.get("queries_console")
    detail["executes_tests"] = safety.get("executes_tests")
    detail["operator_required"] = safety.get("operator_required")
    detail["execution_allowed_by_queue"] = safety.get("execution_allowed_by_queue")
    detail["fixture_closure_allowed_by_queue"] = safety.get(
        "fixture_closure_allowed_by_queue"
    )
    detail["live_widening_allowed"] = safety.get("live_widening_allowed")
    detail["live_authority_granted"] = safety.get("live_authority_granted")
    detail["full_autonomous_ready"] = safety.get("full_autonomous_ready")
    detail["intake_surface_count"] = intake_surface_count
    detail["available_intake_count"] = available_intake_count
    detail["artifact_count"] = artifact_count
    detail["valid_candidate_count"] = valid_candidate_count
    detail["invalid_candidate_count"] = invalid_candidate_count
    detail["review_candidate_count"] = review_candidate_count
    detail["missing_artifact_surface_count"] = missing_artifact_surface_count
    detail["unsafe_authority_claim_count"] = unsafe_authority_claim_count
    detail["candidate_source_id_missing_count"] = candidate_source_id_missing_count
    detail["target_inconsistent_surface_count"] = target_inconsistent_surface_count
    detail["review_candidate_surface_ids"] = (
        list(review_candidate_surface_ids)
        if isinstance(review_candidate_surface_ids, list)
        else []
    )
    detail["invalid_surface_ids"] = (
        list(invalid_surface_ids) if isinstance(invalid_surface_ids, list) else []
    )
    detail["missing_artifact_surface_ids"] = (
        list(missing_surface_ids) if isinstance(missing_surface_ids, list) else []
    )
    detail["unsafe_authority_surface_ids"] = (
        list(unsafe_surface_ids) if isinstance(unsafe_surface_ids, list) else []
    )
    detail["candidate_source_id_missing_surface_ids"] = (
        list(candidate_source_missing_surface_ids)
        if isinstance(candidate_source_missing_surface_ids, list)
        else []
    )
    detail["target_inconsistent_surface_ids"] = (
        list(target_inconsistent_surface_ids)
        if isinstance(target_inconsistent_surface_ids, list)
        else []
    )
    detail["queue_item_method_ids"] = (
        list(queue_item_method_ids) if isinstance(queue_item_method_ids, list) else []
    )
    detail["promotion_families_requiring_surface"] = (
        list(promotion_families) if isinstance(promotion_families, list) else []
    )
    detail["operator_review_required"] = summary.get("operator_review_required")
    detail["physical_fixture_gap_remains"] = summary.get("physical_fixture_gap_remains")
    detail["blocked_by"] = list(blocked_by)
    detail["item_unsafe_flags"] = item_unsafe_flags
    detail["operator_sequence_count"] = len(operator_sequence)
    detail["focused_validation"] = list(focused_validation)
    detail["violations"] = violations
    finished = datetime.now(timezone.utc)
    return CheckResult(
        name="read_surface_physical_readback_artifact_review_queue_artifact_schema",
        status="passed" if not violations else "failed",
        started_at=started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        finished_at=finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration_sec=round((finished - started).total_seconds(), 3),
        detail=detail,
    )


def validate_wing_readonly_preflight_review_decision_ledger_snapshot() -> CheckResult:
    started = datetime.now(timezone.utc)
    detail: dict[str, object] = {
        "policy": "wing_readonly_preflight_review_decision_is_readonly_operator_gate",
    }
    violations: list[str] = []

    try:
        ledger = build_wing_readonly_preflight_review_decision_ledger()
    except Exception as exc:  # pragma: no cover - defensive release-check guard
        ledger = {}
        violations.append(f"wing_readonly_preflight_review_decision_build_error:{type(exc).__name__}")

    summary = ledger.get("summary") if isinstance(ledger.get("summary"), Mapping) else {}
    safety = ledger.get("safety") if isinstance(ledger.get("safety"), Mapping) else {}
    records = ledger.get("records") if isinstance(ledger.get("records"), list) else []
    review_decision_tool_available = (
        REPO_ROOT / WING_READONLY_PREFLIGHT_REVIEW_DECISION_TOOL
    ).is_file()
    review_decision_command = wing_readonly_preflight_review_decision_command()

    if ledger.get("type") != "wing_readonly_preflight_review_decision_ledger":
        violations.append("wing_readonly_preflight_review_decision_type_mismatch")
    if ledger.get("method_id") != WING_READONLY_PREFLIGHT_REVIEW_DECISION_LEDGER_METHOD_ID:
        violations.append("wing_readonly_preflight_review_decision_method_id_mismatch")
    if ledger.get("status") not in WING_READONLY_PREFLIGHT_REVIEW_DECISION_LEDGER_STATUSES:
        violations.append("wing_readonly_preflight_review_decision_status_mismatch")
    if not review_decision_tool_available:
        violations.append("wing_readonly_preflight_review_decision_tool_missing")
    if "--artifact-root" not in review_decision_command:
        violations.append("wing_readonly_preflight_review_decision_artifact_root_missing")
    if "--decision-root" not in review_decision_command:
        violations.append("wing_readonly_preflight_review_decision_root_missing")
    if "--output" not in review_decision_command:
        violations.append("wing_readonly_preflight_review_decision_output_missing")

    expected_true = ("read_only", "operator_required")
    for key in expected_true:
        if safety.get(key) is not True:
            violations.append(f"wing_readonly_preflight_review_decision_{key}_not_true")
    expected_false = (
        "live_write",
        "mutates_console_state",
        "queries_console",
        "executes_tests",
        "execution_allowed_by_decision",
        "readiness_matrix_update_allowed_by_decision",
        "readiness_closure_allowed_by_decision",
        "live_widening_allowed",
        "live_authority_granted",
        "full_autonomous_ready",
    )
    for key in expected_false:
        if safety.get(key) is not False:
            violations.append(f"wing_readonly_preflight_review_decision_{key}_not_false")

    review_candidate_count = int(summary.get("review_candidate_count") or 0)
    decision_record_count = int(summary.get("decision_record_count") or 0)
    accepted_record_count = int(summary.get("accepted_record_count") or 0)
    rejected_record_count = int(summary.get("rejected_record_count") or 0)
    invalid_record_count = int(summary.get("invalid_record_count") or 0)
    unreviewed_candidate_count = int(summary.get("unreviewed_candidate_count") or 0)
    accepted_source_ids = (
        list(summary.get("accepted_source_ids"))
        if isinstance(summary.get("accepted_source_ids"), list)
        else []
    )
    accepted_candidate_prestart_source_ids = (
        list(summary.get("accepted_candidate_prestart_source_ids"))
        if isinstance(summary.get("accepted_candidate_prestart_source_ids"), list)
        else []
    )
    accepted_candidate_prestart_contract_valid = (
        summary.get("accepted_candidate_prestart_contract_valid") is True
    )
    blocked_by = (
        list(summary.get("blocked_by"))
        if isinstance(summary.get("blocked_by"), list)
        else []
    )
    if decision_record_count != len(records):
        violations.append("wing_readonly_preflight_review_decision_record_count_mismatch")
    if invalid_record_count:
        violations.append("wing_readonly_preflight_review_decision_invalid_records_present")
    if summary.get("operator_review_required") is not True:
        violations.append("wing_readonly_preflight_review_decision_operator_review_not_true")
    if summary.get("readiness_matrix_update_allowed_by_decision") is not False:
        violations.append("wing_readonly_preflight_review_decision_readiness_update_not_false")
    if summary.get("readiness_closure_allowed_by_decision") is not False:
        violations.append("wing_readonly_preflight_review_decision_readiness_closure_not_false")
    if summary.get("live_authority_granted") is not False:
        violations.append("wing_readonly_preflight_review_decision_live_authority_not_false")
    if summary.get("full_autonomous_ready") is not False:
        violations.append("wing_readonly_preflight_review_decision_full_auto_not_false")
    if accepted_candidate_prestart_contract_valid and (
        accepted_candidate_prestart_source_ids != accepted_source_ids
    ):
        violations.append(
            "wing_readonly_preflight_review_decision_prestart_sources_mismatch"
        )
    if accepted_source_ids and not accepted_candidate_prestart_contract_valid:
        if WING_READONLY_PREFLIGHT_REVIEW_CANDIDATE_PRESTART_BLOCKER not in blocked_by:
            violations.append(
                "wing_readonly_preflight_review_decision_prestart_blocker_missing"
            )
        if summary.get("manual_readiness_update_candidate_available") is True:
            violations.append(
                "wing_readonly_preflight_review_decision_manual_candidate_not_blocked_by_prestart"
            )

    detail["ledger_status"] = ledger.get("status")
    detail["review_tool_path"] = WING_READONLY_PREFLIGHT_REVIEW_DECISION_TOOL
    detail["review_tool_available"] = review_decision_tool_available
    detail["review_command"] = review_decision_command
    detail["review_exit_codes"] = dict(WING_READONLY_PREFLIGHT_REVIEW_DECISION_EXIT_CODES)
    detail["preflight_evidence_status"] = summary.get("preflight_evidence_status")
    detail["decision_search_roots"] = (
        list(summary.get("decision_search_roots"))
        if isinstance(summary.get("decision_search_roots"), list)
        else []
    )
    detail["review_candidate_count"] = review_candidate_count
    detail["decision_record_count"] = decision_record_count
    detail["accepted_record_count"] = accepted_record_count
    detail["rejected_record_count"] = rejected_record_count
    detail["invalid_record_count"] = invalid_record_count
    detail["unreviewed_candidate_count"] = unreviewed_candidate_count
    detail["manual_readiness_update_candidate_available"] = (
        summary.get("manual_readiness_update_candidate_available") is True
    )
    detail["operator_review_required"] = summary.get("operator_review_required")
    detail["readiness_matrix_update_allowed_by_decision"] = summary.get(
        "readiness_matrix_update_allowed_by_decision"
    )
    detail["readiness_closure_allowed_by_decision"] = summary.get(
        "readiness_closure_allowed_by_decision"
    )
    detail["live_authority_granted"] = summary.get("live_authority_granted")
    detail["full_autonomous_ready"] = summary.get("full_autonomous_ready")
    detail["accepted_source_ids"] = accepted_source_ids
    detail["accepted_candidate_prestart_contract_valid"] = (
        accepted_candidate_prestart_contract_valid
    )
    detail["accepted_candidate_prestart_source_ids"] = (
        accepted_candidate_prestart_source_ids
    )
    detail["rejected_source_ids"] = (
        list(summary.get("rejected_source_ids"))
        if isinstance(summary.get("rejected_source_ids"), list)
        else []
    )
    detail["invalid_record_ids"] = (
        list(summary.get("invalid_record_ids"))
        if isinstance(summary.get("invalid_record_ids"), list)
        else []
    )
    detail["duplicate_source_ids"] = (
        list(summary.get("duplicate_source_ids"))
        if isinstance(summary.get("duplicate_source_ids"), list)
        else []
    )
    detail["conflicting_source_ids"] = (
        list(summary.get("conflicting_source_ids"))
        if isinstance(summary.get("conflicting_source_ids"), list)
        else []
    )
    detail["unreviewed_candidate_source_ids"] = (
        list(summary.get("unreviewed_candidate_source_ids"))
        if isinstance(summary.get("unreviewed_candidate_source_ids"), list)
        else []
    )
    detail["candidate_reviews"] = (
        list(summary.get("candidate_reviews"))
        if isinstance(summary.get("candidate_reviews"), list)
        else []
    )
    detail["blocked_by"] = blocked_by
    detail["violations"] = violations
    finished = datetime.now(timezone.utc)
    return CheckResult(
        name="wing_readonly_preflight_review_decision_ledger_schema",
        status="passed" if not violations else "failed",
        started_at=started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        finished_at=finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration_sec=round((finished - started).total_seconds(), 3),
        detail=detail,
    )


def validate_raw_mixer_state_physical_readback_contract_snapshot() -> CheckResult:
    started = datetime.now(timezone.utc)
    detail: dict[str, object] = {
        "policy": "raw_mixer_state_readback_contract_is_schema_only_no_live_authority",
    }
    violations: list[str] = []

    try:
        fixture_snapshot = build_read_surface_fixture_evidence_snapshot()
        remediation_plan = build_read_surface_remediation_plan(
            fixture_evidence_snapshot=fixture_snapshot
        )
        contract = build_raw_mixer_state_physical_readback_contract()
    except Exception as exc:  # pragma: no cover - defensive release-check guard
        fixture_snapshot = {}
        remediation_plan = {}
        contract = {}
        violations.append(f"raw_mixer_readback_contract_build_error:{type(exc).__name__}")

    fixture_summary = (
        fixture_snapshot.get("summary")
        if isinstance(fixture_snapshot.get("summary"), Mapping)
        else {}
    )
    remediation_summary = (
        remediation_plan.get("summary")
        if isinstance(remediation_plan.get("summary"), Mapping)
        else {}
    )
    safety = contract.get("safety") if isinstance(contract.get("safety"), Mapping) else {}
    contract_summary = (
        contract.get("summary") if isinstance(contract.get("summary"), Mapping) else {}
    )
    artifact_contract = (
        contract.get("artifact_contract")
        if isinstance(contract.get("artifact_contract"), Mapping)
        else {}
    )

    if contract.get("type") != "raw_mixer_state_physical_readback_contract":
        violations.append("raw_mixer_readback_contract_type_mismatch")
    if contract.get("method_id") != RAW_MIXER_STATE_PHYSICAL_READBACK_CONTRACT_METHOD_ID:
        violations.append("raw_mixer_readback_contract_method_id_mismatch")
    if contract.get("schema_id") != RAW_MIXER_STATE_PHYSICAL_READBACK_SCHEMA_ID:
        violations.append("raw_mixer_readback_contract_schema_id_mismatch")
    if contract.get("status") != "contract_ready_physical_fixture_missing":
        violations.append("raw_mixer_readback_contract_status_mismatch")
    if contract.get("surface_id") != RAW_MIXER_STATE_SURFACE_ID:
        violations.append("raw_mixer_readback_contract_surface_id_mismatch")
    if contract.get("remediation_action_id") != RAW_MIXER_STATE_REMEDIATION_ACTION_ID:
        violations.append("raw_mixer_readback_contract_action_id_mismatch")

    expected_true = ("read_only",)
    for key in expected_true:
        if safety.get(key) is not True:
            violations.append(f"raw_mixer_readback_contract_{key}_not_true")
    expected_false = (
        "live_write",
        "mutates_console_state",
        "queries_console",
        "executes_tests",
        "live_widening_allowed",
        "full_autonomous_ready",
    )
    for key in expected_false:
        if safety.get(key) is not False:
            violations.append(f"raw_mixer_readback_contract_{key}_not_false")

    if artifact_contract.get("type") != RAW_MIXER_STATE_PHYSICAL_READBACK_ARTIFACT_TYPE:
        violations.append("raw_mixer_readback_artifact_type_mismatch")
    if artifact_contract.get("schema_id") != RAW_MIXER_STATE_PHYSICAL_READBACK_SCHEMA_ID:
        violations.append("raw_mixer_readback_artifact_schema_id_mismatch")
    if artifact_contract.get("minimum_row_count") != 1:
        violations.append("raw_mixer_readback_artifact_minimum_row_count_mismatch")

    required_top_level = artifact_contract.get("required_top_level_fields")
    if not isinstance(required_top_level, list) or required_top_level != list(REQUIRED_TOP_LEVEL_FIELDS):
        violations.append("raw_mixer_readback_required_top_level_fields_mismatch")
        required_top_level = []
    required_row_fields = artifact_contract.get("required_row_fields")
    if not isinstance(required_row_fields, list) or required_row_fields != list(REQUIRED_ROW_FIELDS):
        violations.append("raw_mixer_readback_required_row_fields_mismatch")
        required_row_fields = []
    required_false_flags = artifact_contract.get("required_false_safety_flags")
    if not isinstance(required_false_flags, list) or required_false_flags != list(REQUIRED_FALSE_SAFETY_FLAGS):
        violations.append("raw_mixer_readback_required_false_safety_flags_mismatch")
        required_false_flags = []

    allowed_sources = artifact_contract.get("allowed_sources")
    if not isinstance(allowed_sources, list):
        allowed_sources = []
        violations.append("raw_mixer_readback_allowed_sources_missing")
    elif list(allowed_sources) != list(PHYSICAL_READBACK_SOURCES):
        violations.append("raw_mixer_readback_allowed_sources_mismatch")
    if "mixer_state_cache" in allowed_sources or "cache" in allowed_sources:
        violations.append("raw_mixer_readback_allows_cached_source")

    accepted_statuses = artifact_contract.get("accepted_row_statuses")
    if not isinstance(accepted_statuses, list) or accepted_statuses != list(VERIFIED_PHYSICAL_ROW_STATUSES):
        violations.append("raw_mixer_readback_accepted_statuses_mismatch")
        accepted_statuses = []
    forbidden_mutation_fields = artifact_contract.get("forbidden_mutation_evidence_fields")
    if (
        not isinstance(forbidden_mutation_fields, list)
        or forbidden_mutation_fields != list(FORBIDDEN_MUTATION_EVIDENCE_FIELDS)
    ):
        violations.append("raw_mixer_readback_forbidden_mutation_fields_mismatch")
        forbidden_mutation_fields = []

    fixture_next_target = fixture_summary.get("next_physical_readback_surface_id")
    remediation_next_target = remediation_summary.get("next_physical_readback_surface_id")
    remediation_next_action = remediation_summary.get("next_physical_readback_action_id")
    contract_next_target = contract_summary.get("next_physical_readback_surface_id")
    contract_next_action = contract_summary.get("next_physical_readback_action_id")
    if fixture_next_target != RAW_MIXER_STATE_SURFACE_ID:
        violations.append("raw_mixer_readback_fixture_next_target_mismatch")
    if remediation_next_target != RAW_MIXER_STATE_SURFACE_ID:
        violations.append("raw_mixer_readback_remediation_next_target_mismatch")
    if remediation_next_action != RAW_MIXER_STATE_REMEDIATION_ACTION_ID:
        violations.append("raw_mixer_readback_remediation_next_action_mismatch")
    if contract_next_target != RAW_MIXER_STATE_SURFACE_ID:
        violations.append("raw_mixer_readback_contract_next_target_mismatch")
    if contract_next_action != RAW_MIXER_STATE_REMEDIATION_ACTION_ID:
        violations.append("raw_mixer_readback_contract_next_action_mismatch")
    if contract_summary.get("artifact_required") is not True:
        violations.append("raw_mixer_readback_contract_artifact_required_not_true")
    if contract_summary.get("physical_fixture_gap_remains") is not True:
        violations.append("raw_mixer_readback_contract_physical_gap_not_true")
    if contract_summary.get("accepted_as_physical_fixture_candidate") is not False:
        violations.append("raw_mixer_readback_contract_candidate_not_false")
    if contract_summary.get("live_authority_granted") is not False:
        violations.append("raw_mixer_readback_contract_live_authority_not_false")

    detail["contract_status"] = contract.get("status")
    detail["schema_id"] = contract.get("schema_id")
    detail["surface_id"] = contract.get("surface_id")
    detail["remediation_action_id"] = contract.get("remediation_action_id")
    detail["next_physical_readback_surface_id"] = contract_next_target
    detail["next_physical_readback_action_id"] = contract_next_action
    detail["fixture_next_physical_readback_surface_id"] = fixture_next_target
    detail["remediation_next_physical_readback_surface_id"] = remediation_next_target
    detail["remediation_next_physical_readback_action_id"] = remediation_next_action
    detail["artifact_required"] = contract_summary.get("artifact_required")
    detail["physical_fixture_gap_remains"] = contract_summary.get("physical_fixture_gap_remains")
    detail["accepted_as_physical_fixture_candidate"] = contract_summary.get(
        "accepted_as_physical_fixture_candidate"
    )
    detail["live_authority_granted"] = contract_summary.get("live_authority_granted")
    detail["required_top_level_field_count"] = len(required_top_level)
    detail["required_row_field_count"] = len(required_row_fields)
    detail["required_false_safety_flags"] = list(required_false_flags)
    detail["allowed_sources"] = list(allowed_sources)
    detail["accepted_row_statuses"] = list(accepted_statuses)
    detail["forbidden_mutation_evidence_fields"] = list(forbidden_mutation_fields)
    detail["violations"] = violations
    finished = datetime.now(timezone.utc)
    return CheckResult(
        name="raw_mixer_state_physical_readback_contract_schema",
        status="passed" if not violations else "failed",
        started_at=started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        finished_at=finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration_sec=round((finished - started).total_seconds(), 3),
        detail=detail,
    )


def validate_raw_mixer_state_physical_readback_artifact_intake_snapshot() -> CheckResult:
    started = datetime.now(timezone.utc)
    detail: dict[str, object] = {
        "policy": "raw_mixer_state_artifact_intake_is_read_only_operator_review_gate",
    }
    violations: list[str] = []

    try:
        intake = build_raw_mixer_state_physical_readback_artifact_intake()
    except Exception as exc:  # pragma: no cover - defensive release-check guard
        intake = {}
        violations.append(f"raw_mixer_artifact_intake_build_error:{type(exc).__name__}")

    safety = intake.get("safety") if isinstance(intake.get("safety"), Mapping) else {}
    summary = intake.get("summary") if isinstance(intake.get("summary"), Mapping) else {}
    artifacts = intake.get("artifacts") if isinstance(intake.get("artifacts"), list) else []

    if intake.get("type") != "raw_mixer_state_physical_readback_artifact_intake":
        violations.append("raw_mixer_artifact_intake_type_mismatch")
    if intake.get("method_id") != RAW_MIXER_STATE_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID:
        violations.append("raw_mixer_artifact_intake_method_id_mismatch")
    if intake.get("required_contract_method_id") != RAW_MIXER_STATE_PHYSICAL_READBACK_CONTRACT_METHOD_ID:
        violations.append("raw_mixer_artifact_intake_required_contract_mismatch")
    if intake.get("schema_id") != RAW_MIXER_STATE_PHYSICAL_READBACK_SCHEMA_ID:
        violations.append("raw_mixer_artifact_intake_schema_id_mismatch")
    if intake.get("status") not in INTAKE_STATUSES:
        violations.append("raw_mixer_artifact_intake_status_unknown")
    if intake.get("surface_id") != RAW_MIXER_STATE_SURFACE_ID:
        violations.append("raw_mixer_artifact_intake_surface_id_mismatch")
    if intake.get("remediation_action_id") != RAW_MIXER_STATE_REMEDIATION_ACTION_ID:
        violations.append("raw_mixer_artifact_intake_action_id_mismatch")

    expected_true = ("read_only", "operator_required")
    for key in expected_true:
        if safety.get(key) is not True:
            violations.append(f"raw_mixer_artifact_intake_{key}_not_true")
    expected_false = (
        "live_write",
        "mutates_console_state",
        "queries_console",
        "executes_tests",
        "execution_allowed_by_intake",
        "fixture_closure_allowed_by_intake",
        "live_widening_allowed",
        "full_autonomous_ready",
    )
    for key in expected_false:
        if safety.get(key) is not False:
            violations.append(f"raw_mixer_artifact_intake_{key}_not_false")

    artifact_count = summary.get("artifact_count")
    valid_count = summary.get("valid_candidate_count")
    invalid_count = summary.get("invalid_candidate_count")
    if not isinstance(artifact_count, int) or artifact_count != len(artifacts):
        violations.append("raw_mixer_artifact_intake_artifact_count_mismatch")
        artifact_count = len(artifacts)
    if not isinstance(valid_count, int):
        violations.append("raw_mixer_artifact_intake_valid_count_missing")
        valid_count = 0
    if not isinstance(invalid_count, int):
        violations.append("raw_mixer_artifact_intake_invalid_count_missing")
        invalid_count = 0
    if valid_count + invalid_count != artifact_count:
        violations.append("raw_mixer_artifact_intake_candidate_count_mismatch")
    if invalid_count:
        violations.append("raw_mixer_artifact_intake_invalid_candidate_present")

    if summary.get("operator_review_required") is not True:
        violations.append("raw_mixer_artifact_intake_operator_review_not_true")
    if summary.get("physical_fixture_gap_remains") is not True:
        violations.append("raw_mixer_artifact_intake_physical_gap_not_true")
    if summary.get("fixture_closure_allowed_by_intake") is not False:
        violations.append("raw_mixer_artifact_intake_fixture_closure_not_false")
    if summary.get("live_authority_granted") is not False:
        violations.append("raw_mixer_artifact_intake_live_authority_not_false")
    if summary.get("full_autonomous_ready") is not False:
        violations.append("raw_mixer_artifact_intake_full_autonomous_not_false")
    if summary.get("accepted_as_physical_fixture_candidate") is not bool(valid_count):
        violations.append("raw_mixer_artifact_intake_candidate_flag_mismatch")

    intake_violations = (
        list(summary.get("violations")) if isinstance(summary.get("violations"), list) else []
    )
    artifact_source_ids = [
        row.get("source_id")
        for row in artifacts
        if isinstance(row, Mapping) and isinstance(row.get("source_id"), str)
    ]
    detail["intake_status"] = intake.get("status")
    detail["schema_id"] = intake.get("schema_id")
    detail["surface_id"] = intake.get("surface_id")
    detail["remediation_action_id"] = intake.get("remediation_action_id")
    detail["required_contract_method_id"] = intake.get("required_contract_method_id")
    detail["artifact_search_roots"] = (
        list(summary.get("artifact_search_roots"))
        if isinstance(summary.get("artifact_search_roots"), list)
        else []
    )
    detail["artifact_count"] = artifact_count
    detail["valid_candidate_count"] = valid_count
    detail["invalid_candidate_count"] = invalid_count
    detail["accepted_as_physical_fixture_candidate"] = summary.get(
        "accepted_as_physical_fixture_candidate"
    )
    detail["operator_review_required"] = summary.get("operator_review_required")
    detail["physical_fixture_gap_remains"] = summary.get("physical_fixture_gap_remains")
    detail["fixture_closure_allowed_by_intake"] = summary.get(
        "fixture_closure_allowed_by_intake"
    )
    detail["live_authority_granted"] = summary.get("live_authority_granted")
    detail["full_autonomous_ready"] = summary.get("full_autonomous_ready")
    detail["latest_valid_candidate_source_id"] = summary.get("latest_valid_candidate_source_id")
    detail["latest_valid_candidate_captured_at"] = summary.get(
        "latest_valid_candidate_captured_at"
    )
    detail["blocked_by"] = (
        list(summary.get("blocked_by")) if isinstance(summary.get("blocked_by"), list) else []
    )
    detail["artifact_source_ids"] = artifact_source_ids
    detail["intake_violations"] = intake_violations
    detail["violations"] = violations
    finished = datetime.now(timezone.utc)
    return CheckResult(
        name="raw_mixer_state_physical_readback_artifact_intake_schema",
        status="passed" if not violations else "failed",
        started_at=started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        finished_at=finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration_sec=round((finished - started).total_seconds(), 3),
        detail=detail,
    )


def validate_channel_basic_controls_physical_readback_contract_snapshot() -> CheckResult:
    started = datetime.now(timezone.utc)
    detail: dict[str, object] = {
        "policy": "channel_basic_controls_readback_contract_is_schema_only_no_live_authority",
    }
    violations: list[str] = []

    try:
        fixture_snapshot = build_read_surface_fixture_evidence_snapshot()
        remediation_plan = build_read_surface_remediation_plan(
            fixture_evidence_snapshot=fixture_snapshot
        )
        contract = build_channel_basic_controls_physical_readback_contract()
    except Exception as exc:  # pragma: no cover - defensive release-check guard
        fixture_snapshot = {}
        remediation_plan = {}
        contract = {}
        violations.append(f"channel_basic_controls_readback_contract_build_error:{type(exc).__name__}")

    fixture_summary = (
        fixture_snapshot.get("summary")
        if isinstance(fixture_snapshot.get("summary"), Mapping)
        else {}
    )
    remediation_actions = (
        remediation_plan.get("actions")
        if isinstance(remediation_plan.get("actions"), list)
        else []
    )
    safety = contract.get("safety") if isinstance(contract.get("safety"), Mapping) else {}
    contract_summary = (
        contract.get("summary") if isinstance(contract.get("summary"), Mapping) else {}
    )
    artifact_contract = (
        contract.get("artifact_contract")
        if isinstance(contract.get("artifact_contract"), Mapping)
        else {}
    )

    if contract.get("type") != "channel_basic_controls_physical_readback_contract":
        violations.append("channel_basic_controls_readback_contract_type_mismatch")
    if contract.get("method_id") != CHANNEL_BASIC_CONTROLS_PHYSICAL_READBACK_CONTRACT_METHOD_ID:
        violations.append("channel_basic_controls_readback_contract_method_id_mismatch")
    if contract.get("schema_id") != CHANNEL_BASIC_CONTROLS_PHYSICAL_READBACK_SCHEMA_ID:
        violations.append("channel_basic_controls_readback_contract_schema_id_mismatch")
    if contract.get("status") != "contract_ready_physical_fixture_missing":
        violations.append("channel_basic_controls_readback_contract_status_mismatch")
    if contract.get("surface_id") != CHANNEL_BASIC_CONTROLS_SURFACE_ID:
        violations.append("channel_basic_controls_readback_contract_surface_id_mismatch")
    if contract.get("remediation_action_id") != CHANNEL_BASIC_CONTROLS_REMEDIATION_ACTION_ID:
        violations.append("channel_basic_controls_readback_contract_action_id_mismatch")

    if safety.get("read_only") is not True:
        violations.append("channel_basic_controls_readback_contract_read_only_not_true")
    for key in (
        "live_write",
        "mutates_console_state",
        "queries_console",
        "executes_tests",
        "live_widening_allowed",
        "full_autonomous_ready",
    ):
        if safety.get(key) is not False:
            violations.append(f"channel_basic_controls_readback_contract_{key}_not_false")

    if artifact_contract.get("type") != CHANNEL_BASIC_CONTROLS_PHYSICAL_READBACK_ARTIFACT_TYPE:
        violations.append("channel_basic_controls_readback_artifact_type_mismatch")
    if artifact_contract.get("schema_id") != CHANNEL_BASIC_CONTROLS_PHYSICAL_READBACK_SCHEMA_ID:
        violations.append("channel_basic_controls_readback_artifact_schema_id_mismatch")
    if artifact_contract.get("minimum_channel_count") != 1:
        violations.append("channel_basic_controls_readback_artifact_minimum_channel_count_mismatch")

    required_top_level = artifact_contract.get("required_top_level_fields")
    if (
        not isinstance(required_top_level, list)
        or required_top_level != list(CHANNEL_BASIC_CONTROLS_REQUIRED_TOP_LEVEL_FIELDS)
    ):
        violations.append("channel_basic_controls_readback_required_top_level_fields_mismatch")
        required_top_level = []
    required_channel_fields = artifact_contract.get("required_channel_fields")
    if (
        not isinstance(required_channel_fields, list)
        or required_channel_fields != list(CHANNEL_BASIC_CONTROLS_REQUIRED_CHANNEL_FIELDS)
    ):
        violations.append("channel_basic_controls_readback_required_channel_fields_mismatch")
        required_channel_fields = []
    required_identity_fields = artifact_contract.get("required_identity_fields")
    if (
        not isinstance(required_identity_fields, list)
        or required_identity_fields != list(CHANNEL_BASIC_CONTROLS_REQUIRED_IDENTITY_FIELDS)
    ):
        violations.append("channel_basic_controls_readback_required_identity_fields_mismatch")
        required_identity_fields = []
    required_control_fields = artifact_contract.get("required_control_fields")
    if (
        not isinstance(required_control_fields, list)
        or required_control_fields != list(CHANNEL_BASIC_CONTROLS_REQUIRED_CONTROL_FIELDS)
    ):
        violations.append("channel_basic_controls_readback_required_control_fields_mismatch")
        required_control_fields = []
    required_coverage_flags = artifact_contract.get("required_coverage_flags")
    if (
        not isinstance(required_coverage_flags, list)
        or required_coverage_flags != list(CHANNEL_BASIC_CONTROLS_REQUIRED_COVERAGE_FLAGS)
    ):
        violations.append("channel_basic_controls_readback_required_coverage_flags_mismatch")
        required_coverage_flags = []
    required_false_flags = artifact_contract.get("required_false_safety_flags")
    if (
        not isinstance(required_false_flags, list)
        or required_false_flags != list(CHANNEL_BASIC_CONTROLS_REQUIRED_FALSE_SAFETY_FLAGS)
    ):
        violations.append("channel_basic_controls_readback_required_false_safety_flags_mismatch")
        required_false_flags = []

    allowed_sources = artifact_contract.get("allowed_sources")
    if not isinstance(allowed_sources, list):
        allowed_sources = []
        violations.append("channel_basic_controls_readback_allowed_sources_missing")
    elif list(allowed_sources) != list(CHANNEL_BASIC_CONTROLS_PHYSICAL_READBACK_SOURCES):
        violations.append("channel_basic_controls_readback_allowed_sources_mismatch")
    if "mixer_state_cache" in allowed_sources or "cache" in allowed_sources:
        violations.append("channel_basic_controls_readback_allows_cached_source")

    accepted_statuses = artifact_contract.get("accepted_channel_statuses")
    if (
        not isinstance(accepted_statuses, list)
        or accepted_statuses != list(CHANNEL_BASIC_CONTROLS_VERIFIED_PHYSICAL_STATUSES)
    ):
        violations.append("channel_basic_controls_readback_accepted_statuses_mismatch")
        accepted_statuses = []
    forbidden_mutation_fields = artifact_contract.get("forbidden_mutation_evidence_fields")
    if (
        not isinstance(forbidden_mutation_fields, list)
        or forbidden_mutation_fields
        != list(CHANNEL_BASIC_CONTROLS_FORBIDDEN_MUTATION_EVIDENCE_FIELDS)
    ):
        violations.append("channel_basic_controls_readback_forbidden_mutation_fields_mismatch")
        forbidden_mutation_fields = []

    physical_gap_ids = fixture_summary.get("physical_fixture_missing_surface_ids")
    if not isinstance(physical_gap_ids, list):
        physical_gap_ids = []
        violations.append("channel_basic_controls_fixture_physical_gap_ids_missing")
    if CHANNEL_BASIC_CONTROLS_SURFACE_ID not in physical_gap_ids:
        violations.append("channel_basic_controls_fixture_gap_not_declared")

    remediation_action = {}
    for action in remediation_actions:
        if (
            isinstance(action, Mapping)
            and action.get("surface_id") == CHANNEL_BASIC_CONTROLS_SURFACE_ID
        ):
            remediation_action = dict(action)
            break
    if not remediation_action:
        violations.append("channel_basic_controls_remediation_action_missing")
    elif remediation_action.get("id") != CHANNEL_BASIC_CONTROLS_REMEDIATION_ACTION_ID:
        violations.append("channel_basic_controls_remediation_action_id_mismatch")
    if remediation_action and remediation_action.get("physical_readback_gap") is not True:
        violations.append("channel_basic_controls_remediation_physical_gap_not_true")

    if contract_summary.get("physical_readback_surface_id") != CHANNEL_BASIC_CONTROLS_SURFACE_ID:
        violations.append("channel_basic_controls_contract_surface_summary_mismatch")
    if contract_summary.get("physical_readback_action_id") != CHANNEL_BASIC_CONTROLS_REMEDIATION_ACTION_ID:
        violations.append("channel_basic_controls_contract_action_summary_mismatch")
    if contract_summary.get("artifact_required") is not True:
        violations.append("channel_basic_controls_contract_artifact_required_not_true")
    if contract_summary.get("physical_fixture_gap_remains") is not True:
        violations.append("channel_basic_controls_contract_physical_gap_not_true")
    if contract_summary.get("accepted_as_physical_fixture_candidate") is not False:
        violations.append("channel_basic_controls_contract_candidate_not_false")
    if contract_summary.get("live_authority_granted") is not False:
        violations.append("channel_basic_controls_contract_live_authority_not_false")
    promotion_families = contract_summary.get("promotion_families_requiring_surface")
    if not isinstance(promotion_families, list):
        promotion_families = []
        violations.append("channel_basic_controls_promotion_families_missing")
    for family in ("fader", "gain", "mute", "pan_phase_delay"):
        if family not in promotion_families:
            violations.append(f"channel_basic_controls_promotion_family_missing:{family}")

    detail["contract_status"] = contract.get("status")
    detail["schema_id"] = contract.get("schema_id")
    detail["surface_id"] = contract.get("surface_id")
    detail["remediation_action_id"] = contract.get("remediation_action_id")
    detail["physical_readback_surface_id"] = contract_summary.get("physical_readback_surface_id")
    detail["physical_readback_action_id"] = contract_summary.get("physical_readback_action_id")
    detail["fixture_physical_gap_declared"] = CHANNEL_BASIC_CONTROLS_SURFACE_ID in physical_gap_ids
    detail["remediation_physical_gap_declared"] = bool(
        remediation_action and remediation_action.get("physical_readback_gap") is True
    )
    detail["artifact_required"] = contract_summary.get("artifact_required")
    detail["physical_fixture_gap_remains"] = contract_summary.get("physical_fixture_gap_remains")
    detail["accepted_as_physical_fixture_candidate"] = contract_summary.get(
        "accepted_as_physical_fixture_candidate"
    )
    detail["live_authority_granted"] = contract_summary.get("live_authority_granted")
    detail["promotion_families_requiring_surface"] = list(promotion_families)
    detail["required_top_level_field_count"] = len(required_top_level)
    detail["required_channel_field_count"] = len(required_channel_fields)
    detail["required_identity_fields"] = list(required_identity_fields)
    detail["required_control_fields"] = list(required_control_fields)
    detail["required_coverage_flags"] = list(required_coverage_flags)
    detail["required_false_safety_flags"] = list(required_false_flags)
    detail["allowed_sources"] = list(allowed_sources)
    detail["accepted_channel_statuses"] = list(accepted_statuses)
    detail["forbidden_mutation_evidence_fields"] = list(forbidden_mutation_fields)
    detail["violations"] = violations
    finished = datetime.now(timezone.utc)
    return CheckResult(
        name="channel_basic_controls_physical_readback_contract_schema",
        status="passed" if not violations else "failed",
        started_at=started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        finished_at=finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration_sec=round((finished - started).total_seconds(), 3),
        detail=detail,
    )


def validate_channel_basic_controls_physical_readback_artifact_intake_snapshot() -> CheckResult:
    started = datetime.now(timezone.utc)
    detail: dict[str, object] = {
        "policy": "channel_basic_controls_artifact_intake_is_read_only_operator_review_gate",
    }
    violations: list[str] = []

    try:
        intake = build_channel_basic_controls_physical_readback_artifact_intake()
    except Exception as exc:  # pragma: no cover - defensive release-check guard
        intake = {}
        violations.append(f"channel_basic_artifact_intake_build_error:{type(exc).__name__}")

    safety = intake.get("safety") if isinstance(intake.get("safety"), Mapping) else {}
    summary = intake.get("summary") if isinstance(intake.get("summary"), Mapping) else {}
    artifacts = intake.get("artifacts") if isinstance(intake.get("artifacts"), list) else []

    if intake.get("type") != "channel_basic_controls_physical_readback_artifact_intake":
        violations.append("channel_basic_artifact_intake_type_mismatch")
    if intake.get("method_id") != CHANNEL_BASIC_CONTROLS_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID:
        violations.append("channel_basic_artifact_intake_method_id_mismatch")
    if intake.get("required_contract_method_id") != CHANNEL_BASIC_CONTROLS_PHYSICAL_READBACK_CONTRACT_METHOD_ID:
        violations.append("channel_basic_artifact_intake_required_contract_mismatch")
    if intake.get("schema_id") != CHANNEL_BASIC_CONTROLS_PHYSICAL_READBACK_SCHEMA_ID:
        violations.append("channel_basic_artifact_intake_schema_id_mismatch")
    if intake.get("status") not in CHANNEL_BASIC_CONTROLS_INTAKE_STATUSES:
        violations.append("channel_basic_artifact_intake_status_unknown")
    if intake.get("surface_id") != CHANNEL_BASIC_CONTROLS_SURFACE_ID:
        violations.append("channel_basic_artifact_intake_surface_id_mismatch")
    if intake.get("remediation_action_id") != CHANNEL_BASIC_CONTROLS_REMEDIATION_ACTION_ID:
        violations.append("channel_basic_artifact_intake_action_id_mismatch")

    for key in ("read_only", "operator_required"):
        if safety.get(key) is not True:
            violations.append(f"channel_basic_artifact_intake_{key}_not_true")
    for key in (
        "live_write",
        "mutates_console_state",
        "queries_console",
        "executes_tests",
        "execution_allowed_by_intake",
        "fixture_closure_allowed_by_intake",
        "live_widening_allowed",
        "full_autonomous_ready",
    ):
        if safety.get(key) is not False:
            violations.append(f"channel_basic_artifact_intake_{key}_not_false")

    artifact_count = summary.get("artifact_count")
    valid_count = summary.get("valid_candidate_count")
    invalid_count = summary.get("invalid_candidate_count")
    valid_channel_count = summary.get("valid_channel_count")
    if not isinstance(artifact_count, int) or artifact_count != len(artifacts):
        violations.append("channel_basic_artifact_intake_artifact_count_mismatch")
        artifact_count = len(artifacts)
    if not isinstance(valid_count, int):
        violations.append("channel_basic_artifact_intake_valid_count_missing")
        valid_count = 0
    if not isinstance(invalid_count, int):
        violations.append("channel_basic_artifact_intake_invalid_count_missing")
        invalid_count = 0
    if not isinstance(valid_channel_count, int):
        violations.append("channel_basic_artifact_intake_valid_channel_count_missing")
        valid_channel_count = 0
    if valid_count + invalid_count != artifact_count:
        violations.append("channel_basic_artifact_intake_candidate_count_mismatch")
    if invalid_count:
        violations.append("channel_basic_artifact_intake_invalid_candidate_present")

    if summary.get("operator_review_required") is not True:
        violations.append("channel_basic_artifact_intake_operator_review_not_true")
    if summary.get("physical_fixture_gap_remains") is not True:
        violations.append("channel_basic_artifact_intake_physical_gap_not_true")
    if summary.get("fixture_closure_allowed_by_intake") is not False:
        violations.append("channel_basic_artifact_intake_fixture_closure_not_false")
    if summary.get("live_authority_granted") is not False:
        violations.append("channel_basic_artifact_intake_live_authority_not_false")
    if summary.get("full_autonomous_ready") is not False:
        violations.append("channel_basic_artifact_intake_full_autonomous_not_false")
    if summary.get("accepted_as_physical_fixture_candidate") is not bool(valid_count):
        violations.append("channel_basic_artifact_intake_candidate_flag_mismatch")

    promotion_families = summary.get("promotion_families_requiring_surface")
    if not isinstance(promotion_families, list):
        promotion_families = []
        violations.append("channel_basic_artifact_intake_promotion_families_missing")
    for family in ("fader", "gain", "mute", "pan_phase_delay"):
        if family not in promotion_families:
            violations.append(f"channel_basic_artifact_intake_promotion_family_missing:{family}")

    intake_violations = (
        list(summary.get("violations")) if isinstance(summary.get("violations"), list) else []
    )
    artifact_source_ids = [
        row.get("source_id")
        for row in artifacts
        if isinstance(row, Mapping) and isinstance(row.get("source_id"), str)
    ]
    detail["intake_status"] = intake.get("status")
    detail["schema_id"] = intake.get("schema_id")
    detail["surface_id"] = intake.get("surface_id")
    detail["remediation_action_id"] = intake.get("remediation_action_id")
    detail["required_contract_method_id"] = intake.get("required_contract_method_id")
    detail["artifact_search_roots"] = (
        list(summary.get("artifact_search_roots"))
        if isinstance(summary.get("artifact_search_roots"), list)
        else []
    )
    detail["artifact_count"] = artifact_count
    detail["valid_candidate_count"] = valid_count
    detail["invalid_candidate_count"] = invalid_count
    detail["valid_channel_count"] = valid_channel_count
    detail["accepted_as_physical_fixture_candidate"] = summary.get(
        "accepted_as_physical_fixture_candidate"
    )
    detail["operator_review_required"] = summary.get("operator_review_required")
    detail["physical_fixture_gap_remains"] = summary.get("physical_fixture_gap_remains")
    detail["fixture_closure_allowed_by_intake"] = summary.get(
        "fixture_closure_allowed_by_intake"
    )
    detail["live_authority_granted"] = summary.get("live_authority_granted")
    detail["full_autonomous_ready"] = summary.get("full_autonomous_ready")
    detail["promotion_families_requiring_surface"] = list(promotion_families)
    detail["latest_valid_candidate_source_id"] = summary.get("latest_valid_candidate_source_id")
    detail["latest_valid_candidate_captured_at"] = summary.get(
        "latest_valid_candidate_captured_at"
    )
    detail["blocked_by"] = (
        list(summary.get("blocked_by")) if isinstance(summary.get("blocked_by"), list) else []
    )
    detail["artifact_source_ids"] = artifact_source_ids
    detail["intake_violations"] = intake_violations
    detail["violations"] = violations
    finished = datetime.now(timezone.utc)
    return CheckResult(
        name="channel_basic_controls_physical_readback_artifact_intake_schema",
        status="passed" if not violations else "failed",
        started_at=started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        finished_at=finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration_sec=round((finished - started).total_seconds(), 3),
        detail=detail,
    )


def validate_channel_processing_physical_readback_contract_snapshot() -> CheckResult:
    started = datetime.now(timezone.utc)
    detail: dict[str, object] = {
        "policy": "channel_processing_readback_contract_is_schema_only_no_live_authority",
    }
    violations: list[str] = []

    try:
        fixture_snapshot = build_read_surface_fixture_evidence_snapshot()
        remediation_plan = build_read_surface_remediation_plan(
            fixture_evidence_snapshot=fixture_snapshot
        )
        contract = build_channel_processing_physical_readback_contract()
    except Exception as exc:  # pragma: no cover - defensive release-check guard
        fixture_snapshot = {}
        remediation_plan = {}
        contract = {}
        violations.append(f"channel_processing_readback_contract_build_error:{type(exc).__name__}")

    fixture_summary = (
        fixture_snapshot.get("summary")
        if isinstance(fixture_snapshot.get("summary"), Mapping)
        else {}
    )
    remediation_actions = (
        remediation_plan.get("actions")
        if isinstance(remediation_plan.get("actions"), list)
        else []
    )
    safety = contract.get("safety") if isinstance(contract.get("safety"), Mapping) else {}
    contract_summary = (
        contract.get("summary") if isinstance(contract.get("summary"), Mapping) else {}
    )
    artifact_contract = (
        contract.get("artifact_contract")
        if isinstance(contract.get("artifact_contract"), Mapping)
        else {}
    )

    if contract.get("type") != "channel_processing_physical_readback_contract":
        violations.append("channel_processing_readback_contract_type_mismatch")
    if contract.get("method_id") != CHANNEL_PROCESSING_PHYSICAL_READBACK_CONTRACT_METHOD_ID:
        violations.append("channel_processing_readback_contract_method_id_mismatch")
    if contract.get("schema_id") != CHANNEL_PROCESSING_PHYSICAL_READBACK_SCHEMA_ID:
        violations.append("channel_processing_readback_contract_schema_id_mismatch")
    if contract.get("status") != "contract_ready_physical_fixture_missing":
        violations.append("channel_processing_readback_contract_status_mismatch")
    if contract.get("surface_id") != CHANNEL_PROCESSING_SURFACE_ID:
        violations.append("channel_processing_readback_contract_surface_id_mismatch")
    if contract.get("remediation_action_id") != CHANNEL_PROCESSING_REMEDIATION_ACTION_ID:
        violations.append("channel_processing_readback_contract_action_id_mismatch")

    if safety.get("read_only") is not True:
        violations.append("channel_processing_readback_contract_read_only_not_true")
    for key in (
        "live_write",
        "mutates_console_state",
        "queries_console",
        "executes_tests",
        "live_widening_allowed",
        "full_autonomous_ready",
    ):
        if safety.get(key) is not False:
            violations.append(f"channel_processing_readback_contract_{key}_not_false")

    if artifact_contract.get("type") != CHANNEL_PROCESSING_PHYSICAL_READBACK_ARTIFACT_TYPE:
        violations.append("channel_processing_readback_artifact_type_mismatch")
    if artifact_contract.get("schema_id") != CHANNEL_PROCESSING_PHYSICAL_READBACK_SCHEMA_ID:
        violations.append("channel_processing_readback_artifact_schema_id_mismatch")
    if artifact_contract.get("minimum_channel_count") != 1:
        violations.append("channel_processing_readback_artifact_minimum_channel_count_mismatch")
    if artifact_contract.get("minimum_eq_band_count") != MINIMUM_EQ_BAND_COUNT:
        violations.append("channel_processing_readback_artifact_minimum_eq_band_count_mismatch")

    required_top_level = artifact_contract.get("required_top_level_fields")
    if (
        not isinstance(required_top_level, list)
        or required_top_level != list(CHANNEL_PROCESSING_REQUIRED_TOP_LEVEL_FIELDS)
    ):
        violations.append("channel_processing_readback_required_top_level_fields_mismatch")
        required_top_level = []
    required_channel_fields = artifact_contract.get("required_channel_fields")
    if (
        not isinstance(required_channel_fields, list)
        or required_channel_fields != list(CHANNEL_PROCESSING_REQUIRED_CHANNEL_FIELDS)
    ):
        violations.append("channel_processing_readback_required_channel_fields_mismatch")
        required_channel_fields = []
    required_hpf_fields = artifact_contract.get("required_hpf_fields")
    if (
        not isinstance(required_hpf_fields, list)
        or required_hpf_fields != list(CHANNEL_PROCESSING_REQUIRED_HPF_FIELDS)
    ):
        violations.append("channel_processing_readback_required_hpf_fields_mismatch")
        required_hpf_fields = []
    required_eq_fields = artifact_contract.get("required_eq_fields")
    if (
        not isinstance(required_eq_fields, list)
        or required_eq_fields != list(CHANNEL_PROCESSING_REQUIRED_EQ_FIELDS)
    ):
        violations.append("channel_processing_readback_required_eq_fields_mismatch")
        required_eq_fields = []
    required_eq_band_fields = artifact_contract.get("required_eq_band_fields")
    if (
        not isinstance(required_eq_band_fields, list)
        or required_eq_band_fields != list(CHANNEL_PROCESSING_REQUIRED_EQ_BAND_FIELDS)
    ):
        violations.append("channel_processing_readback_required_eq_band_fields_mismatch")
        required_eq_band_fields = []
    required_compressor_fields = artifact_contract.get("required_compressor_fields")
    if (
        not isinstance(required_compressor_fields, list)
        or required_compressor_fields != list(CHANNEL_PROCESSING_REQUIRED_COMPRESSOR_FIELDS)
    ):
        violations.append("channel_processing_readback_required_compressor_fields_mismatch")
        required_compressor_fields = []
    required_gate_fields = artifact_contract.get("required_gate_fields")
    if (
        not isinstance(required_gate_fields, list)
        or required_gate_fields != list(CHANNEL_PROCESSING_REQUIRED_GATE_FIELDS)
    ):
        violations.append("channel_processing_readback_required_gate_fields_mismatch")
        required_gate_fields = []
    required_coverage_flags = artifact_contract.get("required_coverage_flags")
    if (
        not isinstance(required_coverage_flags, list)
        or required_coverage_flags != list(CHANNEL_PROCESSING_REQUIRED_COVERAGE_FLAGS)
    ):
        violations.append("channel_processing_readback_required_coverage_flags_mismatch")
        required_coverage_flags = []
    required_false_flags = artifact_contract.get("required_false_safety_flags")
    if (
        not isinstance(required_false_flags, list)
        or required_false_flags != list(CHANNEL_PROCESSING_REQUIRED_FALSE_SAFETY_FLAGS)
    ):
        violations.append("channel_processing_readback_required_false_safety_flags_mismatch")
        required_false_flags = []

    allowed_sources = artifact_contract.get("allowed_sources")
    if not isinstance(allowed_sources, list):
        allowed_sources = []
        violations.append("channel_processing_readback_allowed_sources_missing")
    elif list(allowed_sources) != list(CHANNEL_PROCESSING_PHYSICAL_READBACK_SOURCES):
        violations.append("channel_processing_readback_allowed_sources_mismatch")
    if "mixer_state_cache" in allowed_sources or "cache" in allowed_sources:
        violations.append("channel_processing_readback_allows_cached_source")

    accepted_statuses = artifact_contract.get("accepted_channel_statuses")
    if (
        not isinstance(accepted_statuses, list)
        or accepted_statuses != list(CHANNEL_PROCESSING_VERIFIED_PHYSICAL_STATUSES)
    ):
        violations.append("channel_processing_readback_accepted_statuses_mismatch")
        accepted_statuses = []
    forbidden_mutation_fields = artifact_contract.get("forbidden_mutation_evidence_fields")
    if (
        not isinstance(forbidden_mutation_fields, list)
        or forbidden_mutation_fields != list(CHANNEL_PROCESSING_FORBIDDEN_MUTATION_EVIDENCE_FIELDS)
    ):
        violations.append("channel_processing_readback_forbidden_mutation_fields_mismatch")
        forbidden_mutation_fields = []

    physical_gap_ids = fixture_summary.get("physical_fixture_missing_surface_ids")
    if not isinstance(physical_gap_ids, list):
        physical_gap_ids = []
        violations.append("channel_processing_fixture_physical_gap_ids_missing")
    if CHANNEL_PROCESSING_SURFACE_ID not in physical_gap_ids:
        violations.append("channel_processing_fixture_gap_not_declared")

    remediation_action = {}
    for action in remediation_actions:
        if isinstance(action, Mapping) and action.get("surface_id") == CHANNEL_PROCESSING_SURFACE_ID:
            remediation_action = dict(action)
            break
    if not remediation_action:
        violations.append("channel_processing_remediation_action_missing")
    elif remediation_action.get("id") != CHANNEL_PROCESSING_REMEDIATION_ACTION_ID:
        violations.append("channel_processing_remediation_action_id_mismatch")
    if remediation_action and remediation_action.get("physical_readback_gap") is not True:
        violations.append("channel_processing_remediation_physical_gap_not_true")

    if contract_summary.get("physical_readback_surface_id") != CHANNEL_PROCESSING_SURFACE_ID:
        violations.append("channel_processing_contract_surface_summary_mismatch")
    if contract_summary.get("physical_readback_action_id") != CHANNEL_PROCESSING_REMEDIATION_ACTION_ID:
        violations.append("channel_processing_contract_action_summary_mismatch")
    if contract_summary.get("artifact_required") is not True:
        violations.append("channel_processing_contract_artifact_required_not_true")
    if contract_summary.get("physical_fixture_gap_remains") is not True:
        violations.append("channel_processing_contract_physical_gap_not_true")
    if contract_summary.get("accepted_as_physical_fixture_candidate") is not False:
        violations.append("channel_processing_contract_candidate_not_false")
    if contract_summary.get("live_authority_granted") is not False:
        violations.append("channel_processing_contract_live_authority_not_false")
    promotion_families = contract_summary.get("promotion_families_requiring_surface")
    if not isinstance(promotion_families, list):
        promotion_families = []
        violations.append("channel_processing_promotion_families_missing")
    for family in ("eq", "compressor", "gate", "deessing"):
        if family not in promotion_families:
            violations.append(f"channel_processing_promotion_family_missing:{family}")

    detail["contract_status"] = contract.get("status")
    detail["schema_id"] = contract.get("schema_id")
    detail["surface_id"] = contract.get("surface_id")
    detail["remediation_action_id"] = contract.get("remediation_action_id")
    detail["physical_readback_surface_id"] = contract_summary.get("physical_readback_surface_id")
    detail["physical_readback_action_id"] = contract_summary.get("physical_readback_action_id")
    detail["fixture_physical_gap_declared"] = CHANNEL_PROCESSING_SURFACE_ID in physical_gap_ids
    detail["remediation_physical_gap_declared"] = bool(
        remediation_action and remediation_action.get("physical_readback_gap") is True
    )
    detail["artifact_required"] = contract_summary.get("artifact_required")
    detail["physical_fixture_gap_remains"] = contract_summary.get("physical_fixture_gap_remains")
    detail["accepted_as_physical_fixture_candidate"] = contract_summary.get(
        "accepted_as_physical_fixture_candidate"
    )
    detail["live_authority_granted"] = contract_summary.get("live_authority_granted")
    detail["promotion_families_requiring_surface"] = list(promotion_families)
    detail["required_top_level_field_count"] = len(required_top_level)
    detail["required_channel_field_count"] = len(required_channel_fields)
    detail["required_hpf_fields"] = list(required_hpf_fields)
    detail["required_eq_fields"] = list(required_eq_fields)
    detail["required_eq_band_fields"] = list(required_eq_band_fields)
    detail["required_compressor_fields"] = list(required_compressor_fields)
    detail["required_gate_fields"] = list(required_gate_fields)
    detail["required_coverage_flags"] = list(required_coverage_flags)
    detail["required_false_safety_flags"] = list(required_false_flags)
    detail["minimum_eq_band_count"] = int(artifact_contract.get("minimum_eq_band_count") or 0)
    detail["allowed_sources"] = list(allowed_sources)
    detail["accepted_channel_statuses"] = list(accepted_statuses)
    detail["forbidden_mutation_evidence_fields"] = list(forbidden_mutation_fields)
    detail["violations"] = violations
    finished = datetime.now(timezone.utc)
    return CheckResult(
        name="channel_processing_physical_readback_contract_schema",
        status="passed" if not violations else "failed",
        started_at=started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        finished_at=finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration_sec=round((finished - started).total_seconds(), 3),
        detail=detail,
    )


def validate_channel_processing_physical_readback_artifact_intake_snapshot() -> CheckResult:
    started = datetime.now(timezone.utc)
    detail: dict[str, object] = {
        "policy": "channel_processing_artifact_intake_is_read_only_operator_review_gate",
    }
    violations: list[str] = []

    try:
        intake = build_channel_processing_physical_readback_artifact_intake()
    except Exception as exc:  # pragma: no cover - defensive release-check guard
        intake = {}
        violations.append(f"channel_processing_artifact_intake_build_error:{type(exc).__name__}")

    safety = intake.get("safety") if isinstance(intake.get("safety"), Mapping) else {}
    summary = intake.get("summary") if isinstance(intake.get("summary"), Mapping) else {}
    artifacts = intake.get("artifacts") if isinstance(intake.get("artifacts"), list) else []

    if intake.get("type") != "channel_processing_physical_readback_artifact_intake":
        violations.append("channel_processing_artifact_intake_type_mismatch")
    if intake.get("method_id") != CHANNEL_PROCESSING_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID:
        violations.append("channel_processing_artifact_intake_method_id_mismatch")
    if intake.get("required_contract_method_id") != CHANNEL_PROCESSING_PHYSICAL_READBACK_CONTRACT_METHOD_ID:
        violations.append("channel_processing_artifact_intake_required_contract_mismatch")
    if intake.get("schema_id") != CHANNEL_PROCESSING_PHYSICAL_READBACK_SCHEMA_ID:
        violations.append("channel_processing_artifact_intake_schema_id_mismatch")
    if intake.get("status") not in CHANNEL_PROCESSING_INTAKE_STATUSES:
        violations.append("channel_processing_artifact_intake_status_unknown")
    if intake.get("surface_id") != CHANNEL_PROCESSING_SURFACE_ID:
        violations.append("channel_processing_artifact_intake_surface_id_mismatch")
    if intake.get("remediation_action_id") != CHANNEL_PROCESSING_REMEDIATION_ACTION_ID:
        violations.append("channel_processing_artifact_intake_action_id_mismatch")

    for key in ("read_only", "operator_required"):
        if safety.get(key) is not True:
            violations.append(f"channel_processing_artifact_intake_{key}_not_true")
    for key in (
        "live_write",
        "mutates_console_state",
        "queries_console",
        "executes_tests",
        "execution_allowed_by_intake",
        "fixture_closure_allowed_by_intake",
        "live_widening_allowed",
        "full_autonomous_ready",
    ):
        if safety.get(key) is not False:
            violations.append(f"channel_processing_artifact_intake_{key}_not_false")

    artifact_count = summary.get("artifact_count")
    valid_count = summary.get("valid_candidate_count")
    invalid_count = summary.get("invalid_candidate_count")
    valid_channel_count = summary.get("valid_channel_count")
    if not isinstance(artifact_count, int) or artifact_count != len(artifacts):
        violations.append("channel_processing_artifact_intake_artifact_count_mismatch")
        artifact_count = len(artifacts)
    if not isinstance(valid_count, int):
        violations.append("channel_processing_artifact_intake_valid_count_missing")
        valid_count = 0
    if not isinstance(invalid_count, int):
        violations.append("channel_processing_artifact_intake_invalid_count_missing")
        invalid_count = 0
    if not isinstance(valid_channel_count, int):
        violations.append("channel_processing_artifact_intake_valid_channel_count_missing")
        valid_channel_count = 0
    if valid_count + invalid_count != artifact_count:
        violations.append("channel_processing_artifact_intake_candidate_count_mismatch")
    if invalid_count:
        violations.append("channel_processing_artifact_intake_invalid_candidate_present")

    if summary.get("operator_review_required") is not True:
        violations.append("channel_processing_artifact_intake_operator_review_not_true")
    if summary.get("physical_fixture_gap_remains") is not True:
        violations.append("channel_processing_artifact_intake_physical_gap_not_true")
    if summary.get("fixture_closure_allowed_by_intake") is not False:
        violations.append("channel_processing_artifact_intake_fixture_closure_not_false")
    if summary.get("live_authority_granted") is not False:
        violations.append("channel_processing_artifact_intake_live_authority_not_false")
    if summary.get("full_autonomous_ready") is not False:
        violations.append("channel_processing_artifact_intake_full_autonomous_not_false")
    if summary.get("accepted_as_physical_fixture_candidate") is not bool(valid_count):
        violations.append("channel_processing_artifact_intake_candidate_flag_mismatch")

    promotion_families = summary.get("promotion_families_requiring_surface")
    if not isinstance(promotion_families, list):
        promotion_families = []
        violations.append("channel_processing_artifact_intake_promotion_families_missing")
    for family in (
        "eq",
        "compressor",
        "gate",
        "deessing",
        "frequency_separation",
        "cross_channel_unmasking",
    ):
        if family not in promotion_families:
            violations.append(f"channel_processing_artifact_intake_promotion_family_missing:{family}")

    intake_violations = (
        list(summary.get("violations")) if isinstance(summary.get("violations"), list) else []
    )
    artifact_source_ids = [
        row.get("source_id")
        for row in artifacts
        if isinstance(row, Mapping) and isinstance(row.get("source_id"), str)
    ]
    detail["intake_status"] = intake.get("status")
    detail["schema_id"] = intake.get("schema_id")
    detail["surface_id"] = intake.get("surface_id")
    detail["remediation_action_id"] = intake.get("remediation_action_id")
    detail["required_contract_method_id"] = intake.get("required_contract_method_id")
    detail["artifact_search_roots"] = (
        list(summary.get("artifact_search_roots"))
        if isinstance(summary.get("artifact_search_roots"), list)
        else []
    )
    detail["artifact_count"] = artifact_count
    detail["valid_candidate_count"] = valid_count
    detail["invalid_candidate_count"] = invalid_count
    detail["valid_channel_count"] = valid_channel_count
    detail["accepted_as_physical_fixture_candidate"] = summary.get(
        "accepted_as_physical_fixture_candidate"
    )
    detail["operator_review_required"] = summary.get("operator_review_required")
    detail["physical_fixture_gap_remains"] = summary.get("physical_fixture_gap_remains")
    detail["fixture_closure_allowed_by_intake"] = summary.get(
        "fixture_closure_allowed_by_intake"
    )
    detail["live_authority_granted"] = summary.get("live_authority_granted")
    detail["full_autonomous_ready"] = summary.get("full_autonomous_ready")
    detail["promotion_families_requiring_surface"] = list(promotion_families)
    detail["latest_valid_candidate_source_id"] = summary.get("latest_valid_candidate_source_id")
    detail["latest_valid_candidate_captured_at"] = summary.get(
        "latest_valid_candidate_captured_at"
    )
    detail["blocked_by"] = (
        list(summary.get("blocked_by")) if isinstance(summary.get("blocked_by"), list) else []
    )
    detail["artifact_source_ids"] = artifact_source_ids
    detail["intake_violations"] = intake_violations
    detail["violations"] = violations
    finished = datetime.now(timezone.utc)
    return CheckResult(
        name="channel_processing_physical_readback_artifact_intake_schema",
        status="passed" if not violations else "failed",
        started_at=started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        finished_at=finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration_sec=round((finished - started).total_seconds(), 3),
        detail=detail,
    )


def validate_bus_group_main_dca_physical_readback_contract_snapshot() -> CheckResult:
    started = datetime.now(timezone.utc)
    detail: dict[str, object] = {
        "policy": "bus_group_main_dca_readback_contract_is_schema_only_no_live_authority",
    }
    violations: list[str] = []

    try:
        fixture_snapshot = build_read_surface_fixture_evidence_snapshot()
        remediation_plan = build_read_surface_remediation_plan(
            fixture_evidence_snapshot=fixture_snapshot
        )
        contract = build_bus_group_main_dca_physical_readback_contract()
    except Exception as exc:  # pragma: no cover - defensive release-check guard
        fixture_snapshot = {}
        remediation_plan = {}
        contract = {}
        violations.append(f"bus_group_main_dca_readback_contract_build_error:{type(exc).__name__}")

    fixture_summary = (
        fixture_snapshot.get("summary")
        if isinstance(fixture_snapshot.get("summary"), Mapping)
        else {}
    )
    remediation_actions = (
        remediation_plan.get("actions")
        if isinstance(remediation_plan.get("actions"), list)
        else []
    )
    safety = contract.get("safety") if isinstance(contract.get("safety"), Mapping) else {}
    contract_summary = (
        contract.get("summary") if isinstance(contract.get("summary"), Mapping) else {}
    )
    artifact_contract = (
        contract.get("artifact_contract")
        if isinstance(contract.get("artifact_contract"), Mapping)
        else {}
    )

    if contract.get("type") != "bus_group_main_dca_physical_readback_contract":
        violations.append("bus_group_main_dca_readback_contract_type_mismatch")
    if contract.get("method_id") != BUS_GROUP_MAIN_DCA_PHYSICAL_READBACK_CONTRACT_METHOD_ID:
        violations.append("bus_group_main_dca_readback_contract_method_id_mismatch")
    if contract.get("schema_id") != BUS_GROUP_MAIN_DCA_PHYSICAL_READBACK_SCHEMA_ID:
        violations.append("bus_group_main_dca_readback_contract_schema_id_mismatch")
    if contract.get("status") != "contract_ready_physical_fixture_missing":
        violations.append("bus_group_main_dca_readback_contract_status_mismatch")
    if contract.get("surface_id") != BUS_GROUP_MAIN_DCA_SURFACE_ID:
        violations.append("bus_group_main_dca_readback_contract_surface_id_mismatch")
    if contract.get("remediation_action_id") != BUS_GROUP_MAIN_DCA_REMEDIATION_ACTION_ID:
        violations.append("bus_group_main_dca_readback_contract_action_id_mismatch")

    if safety.get("read_only") is not True:
        violations.append("bus_group_main_dca_readback_contract_read_only_not_true")
    for key in (
        "live_write",
        "mutates_console_state",
        "queries_console",
        "executes_tests",
        "live_widening_allowed",
        "full_autonomous_ready",
    ):
        if safety.get(key) is not False:
            violations.append(f"bus_group_main_dca_readback_contract_{key}_not_false")

    if artifact_contract.get("type") != BUS_GROUP_MAIN_DCA_PHYSICAL_READBACK_ARTIFACT_TYPE:
        violations.append("bus_group_main_dca_readback_artifact_type_mismatch")
    if artifact_contract.get("schema_id") != BUS_GROUP_MAIN_DCA_PHYSICAL_READBACK_SCHEMA_ID:
        violations.append("bus_group_main_dca_readback_artifact_schema_id_mismatch")
    if artifact_contract.get("minimum_bus_count") != 1:
        violations.append("bus_group_main_dca_readback_artifact_minimum_bus_count_mismatch")
    if artifact_contract.get("minimum_main_count") != 1:
        violations.append("bus_group_main_dca_readback_artifact_minimum_main_count_mismatch")
    if artifact_contract.get("minimum_dca_count") != 1:
        violations.append("bus_group_main_dca_readback_artifact_minimum_dca_count_mismatch")

    required_top_level = artifact_contract.get("required_top_level_fields")
    if (
        not isinstance(required_top_level, list)
        or required_top_level != list(BUS_GROUP_MAIN_DCA_REQUIRED_TOP_LEVEL_FIELDS)
    ):
        violations.append("bus_group_main_dca_readback_required_top_level_fields_mismatch")
        required_top_level = []
    required_bus_fields = artifact_contract.get("required_bus_fields")
    if (
        not isinstance(required_bus_fields, list)
        or required_bus_fields != list(BUS_GROUP_MAIN_DCA_REQUIRED_BUS_FIELDS)
    ):
        violations.append("bus_group_main_dca_readback_required_bus_fields_mismatch")
        required_bus_fields = []
    required_main_fields = artifact_contract.get("required_main_fields")
    if (
        not isinstance(required_main_fields, list)
        or required_main_fields != list(BUS_GROUP_MAIN_DCA_REQUIRED_MAIN_FIELDS)
    ):
        violations.append("bus_group_main_dca_readback_required_main_fields_mismatch")
        required_main_fields = []
    required_dca_fields = artifact_contract.get("required_dca_fields")
    if (
        not isinstance(required_dca_fields, list)
        or required_dca_fields != list(BUS_GROUP_MAIN_DCA_REQUIRED_DCA_FIELDS)
    ):
        violations.append("bus_group_main_dca_readback_required_dca_fields_mismatch")
        required_dca_fields = []
    required_bus_control_fields = artifact_contract.get("required_bus_control_fields")
    if (
        not isinstance(required_bus_control_fields, list)
        or required_bus_control_fields != list(BUS_GROUP_MAIN_DCA_REQUIRED_BUS_CONTROL_FIELDS)
    ):
        violations.append("bus_group_main_dca_readback_required_bus_control_fields_mismatch")
        required_bus_control_fields = []
    required_main_control_fields = artifact_contract.get("required_main_control_fields")
    if (
        not isinstance(required_main_control_fields, list)
        or required_main_control_fields != list(BUS_GROUP_MAIN_DCA_REQUIRED_MAIN_CONTROL_FIELDS)
    ):
        violations.append("bus_group_main_dca_readback_required_main_control_fields_mismatch")
        required_main_control_fields = []
    required_dca_control_fields = artifact_contract.get("required_dca_control_fields")
    if (
        not isinstance(required_dca_control_fields, list)
        or required_dca_control_fields != list(BUS_GROUP_MAIN_DCA_REQUIRED_DCA_CONTROL_FIELDS)
    ):
        violations.append("bus_group_main_dca_readback_required_dca_control_fields_mismatch")
        required_dca_control_fields = []
    required_bus_assignment_fields = artifact_contract.get("required_bus_assignment_fields")
    if (
        not isinstance(required_bus_assignment_fields, list)
        or required_bus_assignment_fields != list(BUS_GROUP_MAIN_DCA_REQUIRED_BUS_ASSIGNMENT_FIELDS)
    ):
        violations.append("bus_group_main_dca_readback_required_bus_assignment_fields_mismatch")
        required_bus_assignment_fields = []
    required_dca_assignment_fields = artifact_contract.get("required_dca_assignment_fields")
    if (
        not isinstance(required_dca_assignment_fields, list)
        or required_dca_assignment_fields != list(BUS_GROUP_MAIN_DCA_REQUIRED_DCA_ASSIGNMENT_FIELDS)
    ):
        violations.append("bus_group_main_dca_readback_required_dca_assignment_fields_mismatch")
        required_dca_assignment_fields = []
    required_metering_fields = artifact_contract.get("required_metering_fields")
    if (
        not isinstance(required_metering_fields, list)
        or required_metering_fields != list(BUS_GROUP_MAIN_DCA_REQUIRED_METERING_FIELDS)
    ):
        violations.append("bus_group_main_dca_readback_required_metering_fields_mismatch")
        required_metering_fields = []
    required_bus_coverage_flags = artifact_contract.get("required_bus_coverage_flags")
    if (
        not isinstance(required_bus_coverage_flags, list)
        or required_bus_coverage_flags != list(BUS_GROUP_MAIN_DCA_REQUIRED_BUS_COVERAGE_FLAGS)
    ):
        violations.append("bus_group_main_dca_readback_required_bus_coverage_flags_mismatch")
        required_bus_coverage_flags = []
    required_main_coverage_flags = artifact_contract.get("required_main_coverage_flags")
    if (
        not isinstance(required_main_coverage_flags, list)
        or required_main_coverage_flags != list(BUS_GROUP_MAIN_DCA_REQUIRED_MAIN_COVERAGE_FLAGS)
    ):
        violations.append("bus_group_main_dca_readback_required_main_coverage_flags_mismatch")
        required_main_coverage_flags = []
    required_dca_coverage_flags = artifact_contract.get("required_dca_coverage_flags")
    if (
        not isinstance(required_dca_coverage_flags, list)
        or required_dca_coverage_flags != list(BUS_GROUP_MAIN_DCA_REQUIRED_DCA_COVERAGE_FLAGS)
    ):
        violations.append("bus_group_main_dca_readback_required_dca_coverage_flags_mismatch")
        required_dca_coverage_flags = []
    required_false_flags = artifact_contract.get("required_false_safety_flags")
    if (
        not isinstance(required_false_flags, list)
        or required_false_flags != list(BUS_GROUP_MAIN_DCA_REQUIRED_FALSE_SAFETY_FLAGS)
    ):
        violations.append("bus_group_main_dca_readback_required_false_safety_flags_mismatch")
        required_false_flags = []

    allowed_sources = artifact_contract.get("allowed_sources")
    if not isinstance(allowed_sources, list):
        allowed_sources = []
        violations.append("bus_group_main_dca_readback_allowed_sources_missing")
    elif list(allowed_sources) != list(BUS_GROUP_MAIN_DCA_PHYSICAL_READBACK_SOURCES):
        violations.append("bus_group_main_dca_readback_allowed_sources_mismatch")
    if "mixer_state_cache" in allowed_sources or "cache" in allowed_sources:
        violations.append("bus_group_main_dca_readback_allows_cached_source")

    accepted_statuses = artifact_contract.get("accepted_surface_statuses")
    if (
        not isinstance(accepted_statuses, list)
        or accepted_statuses != list(BUS_GROUP_MAIN_DCA_VERIFIED_PHYSICAL_STATUSES)
    ):
        violations.append("bus_group_main_dca_readback_accepted_statuses_mismatch")
        accepted_statuses = []
    forbidden_mutation_fields = artifact_contract.get("forbidden_mutation_evidence_fields")
    if (
        not isinstance(forbidden_mutation_fields, list)
        or forbidden_mutation_fields
        != list(BUS_GROUP_MAIN_DCA_FORBIDDEN_MUTATION_EVIDENCE_FIELDS)
    ):
        violations.append("bus_group_main_dca_readback_forbidden_mutation_fields_mismatch")
        forbidden_mutation_fields = []

    physical_gap_ids = fixture_summary.get("physical_fixture_missing_surface_ids")
    if not isinstance(physical_gap_ids, list):
        physical_gap_ids = []
        violations.append("bus_group_main_dca_fixture_physical_gap_ids_missing")
    if BUS_GROUP_MAIN_DCA_SURFACE_ID not in physical_gap_ids:
        violations.append("bus_group_main_dca_fixture_gap_not_declared")

    remediation_action = {}
    for action in remediation_actions:
        if isinstance(action, Mapping) and action.get("surface_id") == BUS_GROUP_MAIN_DCA_SURFACE_ID:
            remediation_action = dict(action)
            break
    if not remediation_action:
        violations.append("bus_group_main_dca_remediation_action_missing")
    elif remediation_action.get("id") != BUS_GROUP_MAIN_DCA_REMEDIATION_ACTION_ID:
        violations.append("bus_group_main_dca_remediation_action_id_mismatch")
    if remediation_action and remediation_action.get("physical_readback_gap") is not True:
        violations.append("bus_group_main_dca_remediation_physical_gap_not_true")

    if contract_summary.get("physical_readback_surface_id") != BUS_GROUP_MAIN_DCA_SURFACE_ID:
        violations.append("bus_group_main_dca_contract_surface_summary_mismatch")
    if contract_summary.get("physical_readback_action_id") != BUS_GROUP_MAIN_DCA_REMEDIATION_ACTION_ID:
        violations.append("bus_group_main_dca_contract_action_summary_mismatch")
    if contract_summary.get("artifact_required") is not True:
        violations.append("bus_group_main_dca_contract_artifact_required_not_true")
    if contract_summary.get("physical_fixture_gap_remains") is not True:
        violations.append("bus_group_main_dca_contract_physical_gap_not_true")
    if contract_summary.get("accepted_as_physical_fixture_candidate") is not False:
        violations.append("bus_group_main_dca_contract_candidate_not_false")
    if contract_summary.get("live_authority_granted") is not False:
        violations.append("bus_group_main_dca_contract_live_authority_not_false")
    promotion_families = contract_summary.get("promotion_families_requiring_surface")
    if not isinstance(promotion_families, list):
        promotion_families = []
        violations.append("bus_group_main_dca_promotion_families_missing")
    for family in ("bus_group_matrix", "dca", "master_bus"):
        if family not in promotion_families:
            violations.append(f"bus_group_main_dca_promotion_family_missing:{family}")

    detail["contract_status"] = contract.get("status")
    detail["schema_id"] = contract.get("schema_id")
    detail["surface_id"] = contract.get("surface_id")
    detail["remediation_action_id"] = contract.get("remediation_action_id")
    detail["physical_readback_surface_id"] = contract_summary.get("physical_readback_surface_id")
    detail["physical_readback_action_id"] = contract_summary.get("physical_readback_action_id")
    detail["fixture_physical_gap_declared"] = BUS_GROUP_MAIN_DCA_SURFACE_ID in physical_gap_ids
    detail["remediation_physical_gap_declared"] = bool(
        remediation_action and remediation_action.get("physical_readback_gap") is True
    )
    detail["artifact_required"] = contract_summary.get("artifact_required")
    detail["physical_fixture_gap_remains"] = contract_summary.get("physical_fixture_gap_remains")
    detail["accepted_as_physical_fixture_candidate"] = contract_summary.get(
        "accepted_as_physical_fixture_candidate"
    )
    detail["live_authority_granted"] = contract_summary.get("live_authority_granted")
    detail["promotion_families_requiring_surface"] = list(promotion_families)
    detail["required_top_level_field_count"] = len(required_top_level)
    detail["required_bus_field_count"] = len(required_bus_fields)
    detail["required_main_field_count"] = len(required_main_fields)
    detail["required_dca_field_count"] = len(required_dca_fields)
    detail["required_bus_control_fields"] = list(required_bus_control_fields)
    detail["required_main_control_fields"] = list(required_main_control_fields)
    detail["required_dca_control_fields"] = list(required_dca_control_fields)
    detail["required_bus_assignment_fields"] = list(required_bus_assignment_fields)
    detail["required_dca_assignment_fields"] = list(required_dca_assignment_fields)
    detail["required_metering_fields"] = list(required_metering_fields)
    detail["required_bus_coverage_flags"] = list(required_bus_coverage_flags)
    detail["required_main_coverage_flags"] = list(required_main_coverage_flags)
    detail["required_dca_coverage_flags"] = list(required_dca_coverage_flags)
    detail["required_false_safety_flags"] = list(required_false_flags)
    detail["minimum_bus_count"] = int(artifact_contract.get("minimum_bus_count") or 0)
    detail["minimum_main_count"] = int(artifact_contract.get("minimum_main_count") or 0)
    detail["minimum_dca_count"] = int(artifact_contract.get("minimum_dca_count") or 0)
    detail["allowed_sources"] = list(allowed_sources)
    detail["accepted_surface_statuses"] = list(accepted_statuses)
    detail["forbidden_mutation_evidence_fields"] = list(forbidden_mutation_fields)
    detail["violations"] = violations
    finished = datetime.now(timezone.utc)
    return CheckResult(
        name="bus_group_main_dca_physical_readback_contract_schema",
        status="passed" if not violations else "failed",
        started_at=started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        finished_at=finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration_sec=round((finished - started).total_seconds(), 3),
        detail=detail,
    )


def validate_bus_group_main_dca_physical_readback_artifact_intake_snapshot() -> CheckResult:
    started = datetime.now(timezone.utc)
    detail: dict[str, object] = {
        "policy": "bus_group_main_dca_artifact_intake_is_read_only_operator_review_gate",
    }
    violations: list[str] = []

    try:
        intake = build_bus_group_main_dca_physical_readback_artifact_intake()
    except Exception as exc:  # pragma: no cover - defensive release-check guard
        intake = {}
        violations.append(f"bus_group_main_dca_artifact_intake_build_error:{type(exc).__name__}")

    safety = intake.get("safety") if isinstance(intake.get("safety"), Mapping) else {}
    summary = intake.get("summary") if isinstance(intake.get("summary"), Mapping) else {}
    artifacts = intake.get("artifacts") if isinstance(intake.get("artifacts"), list) else []

    if intake.get("type") != "bus_group_main_dca_physical_readback_artifact_intake":
        violations.append("bus_group_main_dca_artifact_intake_type_mismatch")
    if intake.get("method_id") != BUS_GROUP_MAIN_DCA_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID:
        violations.append("bus_group_main_dca_artifact_intake_method_id_mismatch")
    if intake.get("required_contract_method_id") != BUS_GROUP_MAIN_DCA_PHYSICAL_READBACK_CONTRACT_METHOD_ID:
        violations.append("bus_group_main_dca_artifact_intake_required_contract_mismatch")
    if intake.get("schema_id") != BUS_GROUP_MAIN_DCA_PHYSICAL_READBACK_SCHEMA_ID:
        violations.append("bus_group_main_dca_artifact_intake_schema_id_mismatch")
    if intake.get("status") not in BUS_GROUP_MAIN_DCA_INTAKE_STATUSES:
        violations.append("bus_group_main_dca_artifact_intake_status_unknown")
    if intake.get("surface_id") != BUS_GROUP_MAIN_DCA_SURFACE_ID:
        violations.append("bus_group_main_dca_artifact_intake_surface_id_mismatch")
    if intake.get("remediation_action_id") != BUS_GROUP_MAIN_DCA_REMEDIATION_ACTION_ID:
        violations.append("bus_group_main_dca_artifact_intake_action_id_mismatch")

    for key in ("read_only", "operator_required"):
        if safety.get(key) is not True:
            violations.append(f"bus_group_main_dca_artifact_intake_{key}_not_true")
    for key in (
        "live_write",
        "mutates_console_state",
        "queries_console",
        "executes_tests",
        "execution_allowed_by_intake",
        "fixture_closure_allowed_by_intake",
        "live_widening_allowed",
        "full_autonomous_ready",
    ):
        if safety.get(key) is not False:
            violations.append(f"bus_group_main_dca_artifact_intake_{key}_not_false")

    artifact_count = summary.get("artifact_count")
    valid_count = summary.get("valid_candidate_count")
    invalid_count = summary.get("invalid_candidate_count")
    valid_bus_count = summary.get("valid_bus_count")
    valid_main_count = summary.get("valid_main_count")
    valid_dca_count = summary.get("valid_dca_count")
    if not isinstance(artifact_count, int) or artifact_count != len(artifacts):
        violations.append("bus_group_main_dca_artifact_intake_artifact_count_mismatch")
        artifact_count = len(artifacts)
    if not isinstance(valid_count, int):
        violations.append("bus_group_main_dca_artifact_intake_valid_count_missing")
        valid_count = 0
    if not isinstance(invalid_count, int):
        violations.append("bus_group_main_dca_artifact_intake_invalid_count_missing")
        invalid_count = 0
    if not isinstance(valid_bus_count, int):
        violations.append("bus_group_main_dca_artifact_intake_valid_bus_count_missing")
        valid_bus_count = 0
    if not isinstance(valid_main_count, int):
        violations.append("bus_group_main_dca_artifact_intake_valid_main_count_missing")
        valid_main_count = 0
    if not isinstance(valid_dca_count, int):
        violations.append("bus_group_main_dca_artifact_intake_valid_dca_count_missing")
        valid_dca_count = 0
    if valid_count + invalid_count != artifact_count:
        violations.append("bus_group_main_dca_artifact_intake_candidate_count_mismatch")
    if invalid_count:
        violations.append("bus_group_main_dca_artifact_intake_invalid_candidate_present")

    if summary.get("operator_review_required") is not True:
        violations.append("bus_group_main_dca_artifact_intake_operator_review_not_true")
    if summary.get("physical_fixture_gap_remains") is not True:
        violations.append("bus_group_main_dca_artifact_intake_physical_gap_not_true")
    if summary.get("fixture_closure_allowed_by_intake") is not False:
        violations.append("bus_group_main_dca_artifact_intake_fixture_closure_not_false")
    if summary.get("live_authority_granted") is not False:
        violations.append("bus_group_main_dca_artifact_intake_live_authority_not_false")
    if summary.get("full_autonomous_ready") is not False:
        violations.append("bus_group_main_dca_artifact_intake_full_autonomous_not_false")
    if summary.get("accepted_as_physical_fixture_candidate") is not bool(valid_count):
        violations.append("bus_group_main_dca_artifact_intake_candidate_flag_mismatch")

    promotion_families = summary.get("promotion_families_requiring_surface")
    if not isinstance(promotion_families, list):
        promotion_families = []
        violations.append("bus_group_main_dca_artifact_intake_promotion_families_missing")
    for family in (
        "bus_group_matrix",
        "dca",
        "master_bus",
        "mix_balance",
        "performance_adaptation",
    ):
        if family not in promotion_families:
            violations.append(f"bus_group_main_dca_artifact_intake_promotion_family_missing:{family}")

    intake_violations = (
        list(summary.get("violations")) if isinstance(summary.get("violations"), list) else []
    )
    artifact_source_ids = [
        row.get("source_id")
        for row in artifacts
        if isinstance(row, Mapping) and isinstance(row.get("source_id"), str)
    ]
    detail["intake_status"] = intake.get("status")
    detail["schema_id"] = intake.get("schema_id")
    detail["surface_id"] = intake.get("surface_id")
    detail["remediation_action_id"] = intake.get("remediation_action_id")
    detail["required_contract_method_id"] = intake.get("required_contract_method_id")
    detail["artifact_search_roots"] = (
        list(summary.get("artifact_search_roots"))
        if isinstance(summary.get("artifact_search_roots"), list)
        else []
    )
    detail["artifact_count"] = artifact_count
    detail["valid_candidate_count"] = valid_count
    detail["invalid_candidate_count"] = invalid_count
    detail["valid_bus_count"] = valid_bus_count
    detail["valid_main_count"] = valid_main_count
    detail["valid_dca_count"] = valid_dca_count
    detail["accepted_as_physical_fixture_candidate"] = summary.get(
        "accepted_as_physical_fixture_candidate"
    )
    detail["operator_review_required"] = summary.get("operator_review_required")
    detail["physical_fixture_gap_remains"] = summary.get("physical_fixture_gap_remains")
    detail["fixture_closure_allowed_by_intake"] = summary.get(
        "fixture_closure_allowed_by_intake"
    )
    detail["live_authority_granted"] = summary.get("live_authority_granted")
    detail["full_autonomous_ready"] = summary.get("full_autonomous_ready")
    detail["promotion_families_requiring_surface"] = list(promotion_families)
    detail["latest_valid_candidate_source_id"] = summary.get("latest_valid_candidate_source_id")
    detail["latest_valid_candidate_captured_at"] = summary.get(
        "latest_valid_candidate_captured_at"
    )
    detail["blocked_by"] = (
        list(summary.get("blocked_by")) if isinstance(summary.get("blocked_by"), list) else []
    )
    detail["artifact_source_ids"] = artifact_source_ids
    detail["intake_violations"] = intake_violations
    detail["violations"] = violations
    finished = datetime.now(timezone.utc)
    return CheckResult(
        name="bus_group_main_dca_physical_readback_artifact_intake_schema",
        status="passed" if not violations else "failed",
        started_at=started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        finished_at=finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration_sec=round((finished - started).total_seconds(), 3),
        detail=detail,
    )


def validate_matrix_state_physical_readback_contract_snapshot() -> CheckResult:
    started = datetime.now(timezone.utc)
    detail: dict[str, object] = {
        "policy": "matrix_state_readback_contract_is_schema_only_no_live_authority",
    }
    violations: list[str] = []

    try:
        fixture_snapshot = build_read_surface_fixture_evidence_snapshot()
        remediation_plan = build_read_surface_remediation_plan(
            fixture_evidence_snapshot=fixture_snapshot
        )
        contract = build_matrix_state_physical_readback_contract()
    except Exception as exc:  # pragma: no cover - defensive release-check guard
        fixture_snapshot = {}
        remediation_plan = {}
        contract = {}
        violations.append(f"matrix_state_readback_contract_build_error:{type(exc).__name__}")

    fixture_summary = (
        fixture_snapshot.get("summary")
        if isinstance(fixture_snapshot.get("summary"), Mapping)
        else {}
    )
    remediation_actions = (
        remediation_plan.get("actions")
        if isinstance(remediation_plan.get("actions"), list)
        else []
    )
    safety = contract.get("safety") if isinstance(contract.get("safety"), Mapping) else {}
    contract_summary = (
        contract.get("summary") if isinstance(contract.get("summary"), Mapping) else {}
    )
    artifact_contract = (
        contract.get("artifact_contract")
        if isinstance(contract.get("artifact_contract"), Mapping)
        else {}
    )

    if contract.get("type") != "matrix_state_physical_readback_contract":
        violations.append("matrix_state_readback_contract_type_mismatch")
    if contract.get("method_id") != MATRIX_STATE_PHYSICAL_READBACK_CONTRACT_METHOD_ID:
        violations.append("matrix_state_readback_contract_method_id_mismatch")
    if contract.get("schema_id") != MATRIX_STATE_PHYSICAL_READBACK_SCHEMA_ID:
        violations.append("matrix_state_readback_contract_schema_id_mismatch")
    if contract.get("status") != "contract_ready_physical_fixture_missing":
        violations.append("matrix_state_readback_contract_status_mismatch")
    if contract.get("surface_id") != MATRIX_STATE_SURFACE_ID:
        violations.append("matrix_state_readback_contract_surface_id_mismatch")
    if contract.get("remediation_action_id") != MATRIX_STATE_REMEDIATION_ACTION_ID:
        violations.append("matrix_state_readback_contract_action_id_mismatch")

    if safety.get("read_only") is not True:
        violations.append("matrix_state_readback_contract_read_only_not_true")
    for key in (
        "live_write",
        "mutates_console_state",
        "queries_console",
        "executes_tests",
        "live_widening_allowed",
        "full_autonomous_ready",
    ):
        if safety.get(key) is not False:
            violations.append(f"matrix_state_readback_contract_{key}_not_false")

    if artifact_contract.get("type") != MATRIX_STATE_PHYSICAL_READBACK_ARTIFACT_TYPE:
        violations.append("matrix_state_readback_artifact_type_mismatch")
    if artifact_contract.get("schema_id") != MATRIX_STATE_PHYSICAL_READBACK_SCHEMA_ID:
        violations.append("matrix_state_readback_artifact_schema_id_mismatch")
    if artifact_contract.get("minimum_matrix_count") != 1:
        violations.append("matrix_state_readback_artifact_minimum_matrix_count_mismatch")
    if artifact_contract.get("minimum_eq_band_count") != MINIMUM_MATRIX_EQ_BAND_COUNT:
        violations.append("matrix_state_readback_artifact_minimum_eq_band_count_mismatch")

    expected_lists = {
        "required_top_level_fields": MATRIX_STATE_REQUIRED_TOP_LEVEL_FIELDS,
        "required_matrix_fields": MATRIX_STATE_REQUIRED_MATRIX_FIELDS,
        "required_control_fields": MATRIX_STATE_REQUIRED_CONTROL_FIELDS,
        "required_eq_fields": MATRIX_STATE_REQUIRED_EQ_FIELDS,
        "required_eq_band_fields": MATRIX_STATE_REQUIRED_EQ_BAND_FIELDS,
        "required_dynamics_fields": MATRIX_STATE_REQUIRED_DYNAMICS_FIELDS,
        "required_insert_fields": MATRIX_STATE_REQUIRED_INSERT_FIELDS,
        "required_metering_fields": MATRIX_STATE_REQUIRED_METERING_FIELDS,
        "required_coverage_flags": MATRIX_STATE_REQUIRED_COVERAGE_FLAGS,
        "required_false_safety_flags": MATRIX_STATE_REQUIRED_FALSE_SAFETY_FLAGS,
    }
    list_details: dict[str, list[object]] = {}
    for key, expected in expected_lists.items():
        value = artifact_contract.get(key)
        if not isinstance(value, list) or value != list(expected):
            violations.append(f"matrix_state_readback_{key}_mismatch")
            list_details[key] = []
        else:
            list_details[key] = list(value)

    allowed_sources = artifact_contract.get("allowed_sources")
    if not isinstance(allowed_sources, list):
        allowed_sources = []
        violations.append("matrix_state_readback_allowed_sources_missing")
    elif list(allowed_sources) != list(MATRIX_STATE_PHYSICAL_READBACK_SOURCES):
        violations.append("matrix_state_readback_allowed_sources_mismatch")
    if "mixer_state_cache" in allowed_sources or "cache" in allowed_sources:
        violations.append("matrix_state_readback_allows_cached_source")

    accepted_statuses = artifact_contract.get("accepted_matrix_statuses")
    if (
        not isinstance(accepted_statuses, list)
        or accepted_statuses != list(MATRIX_STATE_VERIFIED_PHYSICAL_STATUSES)
    ):
        violations.append("matrix_state_readback_accepted_statuses_mismatch")
        accepted_statuses = []
    forbidden_mutation_fields = artifact_contract.get("forbidden_mutation_evidence_fields")
    if (
        not isinstance(forbidden_mutation_fields, list)
        or forbidden_mutation_fields != list(MATRIX_STATE_FORBIDDEN_MUTATION_EVIDENCE_FIELDS)
    ):
        violations.append("matrix_state_readback_forbidden_mutation_fields_mismatch")
        forbidden_mutation_fields = []

    physical_gap_ids = fixture_summary.get("physical_fixture_missing_surface_ids")
    if not isinstance(physical_gap_ids, list):
        physical_gap_ids = []
        violations.append("matrix_state_fixture_physical_gap_ids_missing")
    if MATRIX_STATE_SURFACE_ID not in physical_gap_ids:
        violations.append("matrix_state_fixture_gap_not_declared")

    remediation_action = {}
    for action in remediation_actions:
        if isinstance(action, Mapping) and action.get("surface_id") == MATRIX_STATE_SURFACE_ID:
            remediation_action = dict(action)
            break
    if not remediation_action:
        violations.append("matrix_state_remediation_action_missing")
    elif remediation_action.get("id") != MATRIX_STATE_REMEDIATION_ACTION_ID:
        violations.append("matrix_state_remediation_action_id_mismatch")
    if remediation_action and remediation_action.get("physical_readback_gap") is not True:
        violations.append("matrix_state_remediation_physical_gap_not_true")

    if contract_summary.get("physical_readback_surface_id") != MATRIX_STATE_SURFACE_ID:
        violations.append("matrix_state_contract_surface_summary_mismatch")
    if contract_summary.get("physical_readback_action_id") != MATRIX_STATE_REMEDIATION_ACTION_ID:
        violations.append("matrix_state_contract_action_summary_mismatch")
    if contract_summary.get("artifact_required") is not True:
        violations.append("matrix_state_contract_artifact_required_not_true")
    if contract_summary.get("physical_fixture_gap_remains") is not True:
        violations.append("matrix_state_contract_physical_gap_not_true")
    if contract_summary.get("accepted_as_physical_fixture_candidate") is not False:
        violations.append("matrix_state_contract_candidate_not_false")
    if contract_summary.get("live_authority_granted") is not False:
        violations.append("matrix_state_contract_live_authority_not_false")
    promotion_families = contract_summary.get("promotion_families_requiring_surface")
    if not isinstance(promotion_families, list):
        promotion_families = []
        violations.append("matrix_state_promotion_families_missing")
    for family in ("bus_group_matrix", "fx_send_insert", "routing"):
        if family not in promotion_families:
            violations.append(f"matrix_state_promotion_family_missing:{family}")

    detail["contract_status"] = contract.get("status")
    detail["schema_id"] = contract.get("schema_id")
    detail["surface_id"] = contract.get("surface_id")
    detail["remediation_action_id"] = contract.get("remediation_action_id")
    detail["physical_readback_surface_id"] = contract_summary.get("physical_readback_surface_id")
    detail["physical_readback_action_id"] = contract_summary.get("physical_readback_action_id")
    detail["fixture_physical_gap_declared"] = MATRIX_STATE_SURFACE_ID in physical_gap_ids
    detail["remediation_physical_gap_declared"] = bool(
        remediation_action and remediation_action.get("physical_readback_gap") is True
    )
    detail["artifact_required"] = contract_summary.get("artifact_required")
    detail["physical_fixture_gap_remains"] = contract_summary.get("physical_fixture_gap_remains")
    detail["accepted_as_physical_fixture_candidate"] = contract_summary.get(
        "accepted_as_physical_fixture_candidate"
    )
    detail["live_authority_granted"] = contract_summary.get("live_authority_granted")
    detail["promotion_families_requiring_surface"] = list(promotion_families)
    detail["required_top_level_field_count"] = len(
        list_details.get("required_top_level_fields", [])
    )
    detail["required_matrix_field_count"] = len(
        list_details.get("required_matrix_fields", [])
    )
    detail["required_control_fields"] = list(list_details.get("required_control_fields", []))
    detail["required_eq_fields"] = list(list_details.get("required_eq_fields", []))
    detail["required_eq_band_fields"] = list(
        list_details.get("required_eq_band_fields", [])
    )
    detail["required_dynamics_fields"] = list(
        list_details.get("required_dynamics_fields", [])
    )
    detail["required_insert_fields"] = list(list_details.get("required_insert_fields", []))
    detail["required_metering_fields"] = list(
        list_details.get("required_metering_fields", [])
    )
    detail["required_coverage_flags"] = list(
        list_details.get("required_coverage_flags", [])
    )
    detail["required_false_safety_flags"] = list(
        list_details.get("required_false_safety_flags", [])
    )
    detail["minimum_matrix_count"] = int(artifact_contract.get("minimum_matrix_count") or 0)
    detail["minimum_eq_band_count"] = int(artifact_contract.get("minimum_eq_band_count") or 0)
    detail["allowed_sources"] = list(allowed_sources)
    detail["accepted_matrix_statuses"] = list(accepted_statuses)
    detail["forbidden_mutation_evidence_fields"] = list(forbidden_mutation_fields)
    detail["violations"] = violations
    finished = datetime.now(timezone.utc)
    return CheckResult(
        name="matrix_state_physical_readback_contract_schema",
        status="passed" if not violations else "failed",
        started_at=started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        finished_at=finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration_sec=round((finished - started).total_seconds(), 3),
        detail=detail,
    )


def validate_matrix_state_physical_readback_artifact_intake_snapshot() -> CheckResult:
    started = datetime.now(timezone.utc)
    detail: dict[str, object] = {
        "policy": "matrix_state_artifact_intake_is_read_only_operator_review_gate",
    }
    violations: list[str] = []

    try:
        intake = build_matrix_state_physical_readback_artifact_intake()
    except Exception as exc:  # pragma: no cover - defensive release-check guard
        intake = {}
        violations.append(f"matrix_state_artifact_intake_build_error:{type(exc).__name__}")

    safety = intake.get("safety") if isinstance(intake.get("safety"), Mapping) else {}
    summary = intake.get("summary") if isinstance(intake.get("summary"), Mapping) else {}
    artifacts = intake.get("artifacts") if isinstance(intake.get("artifacts"), list) else []

    if intake.get("type") != "matrix_state_physical_readback_artifact_intake":
        violations.append("matrix_state_artifact_intake_type_mismatch")
    if intake.get("method_id") != MATRIX_STATE_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID:
        violations.append("matrix_state_artifact_intake_method_id_mismatch")
    if intake.get("required_contract_method_id") != MATRIX_STATE_PHYSICAL_READBACK_CONTRACT_METHOD_ID:
        violations.append("matrix_state_artifact_intake_required_contract_mismatch")
    if intake.get("schema_id") != MATRIX_STATE_PHYSICAL_READBACK_SCHEMA_ID:
        violations.append("matrix_state_artifact_intake_schema_id_mismatch")
    if intake.get("status") not in MATRIX_STATE_INTAKE_STATUSES:
        violations.append("matrix_state_artifact_intake_status_unknown")
    if intake.get("surface_id") != MATRIX_STATE_SURFACE_ID:
        violations.append("matrix_state_artifact_intake_surface_id_mismatch")
    if intake.get("remediation_action_id") != MATRIX_STATE_REMEDIATION_ACTION_ID:
        violations.append("matrix_state_artifact_intake_action_id_mismatch")

    for key in ("read_only", "operator_required"):
        if safety.get(key) is not True:
            violations.append(f"matrix_state_artifact_intake_{key}_not_true")
    for key in (
        "live_write",
        "mutates_console_state",
        "queries_console",
        "executes_tests",
        "execution_allowed_by_intake",
        "fixture_closure_allowed_by_intake",
        "live_widening_allowed",
        "full_autonomous_ready",
    ):
        if safety.get(key) is not False:
            violations.append(f"matrix_state_artifact_intake_{key}_not_false")

    artifact_count = summary.get("artifact_count")
    valid_count = summary.get("valid_candidate_count")
    invalid_count = summary.get("invalid_candidate_count")
    valid_matrix_count = summary.get("valid_matrix_count")
    if not isinstance(artifact_count, int) or artifact_count != len(artifacts):
        violations.append("matrix_state_artifact_intake_artifact_count_mismatch")
        artifact_count = len(artifacts)
    if not isinstance(valid_count, int):
        violations.append("matrix_state_artifact_intake_valid_count_missing")
        valid_count = 0
    if not isinstance(invalid_count, int):
        violations.append("matrix_state_artifact_intake_invalid_count_missing")
        invalid_count = 0
    if not isinstance(valid_matrix_count, int):
        violations.append("matrix_state_artifact_intake_valid_matrix_count_missing")
        valid_matrix_count = 0
    if valid_count + invalid_count != artifact_count:
        violations.append("matrix_state_artifact_intake_candidate_count_mismatch")
    if invalid_count:
        violations.append("matrix_state_artifact_intake_invalid_candidate_present")

    if summary.get("operator_review_required") is not True:
        violations.append("matrix_state_artifact_intake_operator_review_not_true")
    if summary.get("physical_fixture_gap_remains") is not True:
        violations.append("matrix_state_artifact_intake_physical_gap_not_true")
    if summary.get("fixture_closure_allowed_by_intake") is not False:
        violations.append("matrix_state_artifact_intake_fixture_closure_not_false")
    if summary.get("live_authority_granted") is not False:
        violations.append("matrix_state_artifact_intake_live_authority_not_false")
    if summary.get("full_autonomous_ready") is not False:
        violations.append("matrix_state_artifact_intake_full_autonomous_not_false")
    if summary.get("accepted_as_physical_fixture_candidate") is not bool(valid_count):
        violations.append("matrix_state_artifact_intake_candidate_flag_mismatch")

    promotion_families = summary.get("promotion_families_requiring_surface")
    if not isinstance(promotion_families, list):
        promotion_families = []
        violations.append("matrix_state_artifact_intake_promotion_families_missing")
    for family in (
        "bus_group_matrix",
        "fx_send_insert",
        "master_bus",
        "mix_balance",
        "routing",
        "performance_adaptation",
    ):
        if family not in promotion_families:
            violations.append(f"matrix_state_artifact_intake_promotion_family_missing:{family}")

    intake_violations = (
        list(summary.get("violations")) if isinstance(summary.get("violations"), list) else []
    )
    artifact_source_ids = [
        row.get("source_id")
        for row in artifacts
        if isinstance(row, Mapping) and isinstance(row.get("source_id"), str)
    ]
    detail["intake_status"] = intake.get("status")
    detail["schema_id"] = intake.get("schema_id")
    detail["surface_id"] = intake.get("surface_id")
    detail["remediation_action_id"] = intake.get("remediation_action_id")
    detail["required_contract_method_id"] = intake.get("required_contract_method_id")
    detail["artifact_search_roots"] = (
        list(summary.get("artifact_search_roots"))
        if isinstance(summary.get("artifact_search_roots"), list)
        else []
    )
    detail["artifact_count"] = artifact_count
    detail["valid_candidate_count"] = valid_count
    detail["invalid_candidate_count"] = invalid_count
    detail["valid_matrix_count"] = valid_matrix_count
    detail["accepted_as_physical_fixture_candidate"] = summary.get(
        "accepted_as_physical_fixture_candidate"
    )
    detail["operator_review_required"] = summary.get("operator_review_required")
    detail["physical_fixture_gap_remains"] = summary.get("physical_fixture_gap_remains")
    detail["fixture_closure_allowed_by_intake"] = summary.get(
        "fixture_closure_allowed_by_intake"
    )
    detail["live_authority_granted"] = summary.get("live_authority_granted")
    detail["full_autonomous_ready"] = summary.get("full_autonomous_ready")
    detail["promotion_families_requiring_surface"] = list(promotion_families)
    detail["latest_valid_candidate_source_id"] = summary.get("latest_valid_candidate_source_id")
    detail["latest_valid_candidate_captured_at"] = summary.get(
        "latest_valid_candidate_captured_at"
    )
    detail["blocked_by"] = (
        list(summary.get("blocked_by")) if isinstance(summary.get("blocked_by"), list) else []
    )
    detail["artifact_source_ids"] = artifact_source_ids
    detail["intake_violations"] = intake_violations
    detail["violations"] = violations
    finished = datetime.now(timezone.utc)
    return CheckResult(
        name="matrix_state_physical_readback_artifact_intake_schema",
        status="passed" if not violations else "failed",
        started_at=started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        finished_at=finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration_sec=round((finished - started).total_seconds(), 3),
        detail=detail,
    )


def validate_fx_slots_inserts_physical_readback_contract_snapshot() -> CheckResult:
    started = datetime.now(timezone.utc)
    detail: dict[str, object] = {
        "policy": "fx_slots_inserts_readback_contract_is_schema_only_no_live_authority",
    }
    violations: list[str] = []

    try:
        fixture_snapshot = build_read_surface_fixture_evidence_snapshot()
        remediation_plan = build_read_surface_remediation_plan(
            fixture_evidence_snapshot=fixture_snapshot
        )
        contract = build_fx_slots_inserts_physical_readback_contract()
    except Exception as exc:  # pragma: no cover - defensive release-check guard
        fixture_snapshot = {}
        remediation_plan = {}
        contract = {}
        violations.append(f"fx_slots_inserts_readback_contract_build_error:{type(exc).__name__}")

    fixture_summary = (
        fixture_snapshot.get("summary")
        if isinstance(fixture_snapshot.get("summary"), Mapping)
        else {}
    )
    remediation_actions = (
        remediation_plan.get("actions")
        if isinstance(remediation_plan.get("actions"), list)
        else []
    )
    safety = contract.get("safety") if isinstance(contract.get("safety"), Mapping) else {}
    contract_summary = (
        contract.get("summary") if isinstance(contract.get("summary"), Mapping) else {}
    )
    artifact_contract = (
        contract.get("artifact_contract")
        if isinstance(contract.get("artifact_contract"), Mapping)
        else {}
    )

    if contract.get("type") != "fx_slots_inserts_physical_readback_contract":
        violations.append("fx_slots_inserts_readback_contract_type_mismatch")
    if contract.get("method_id") != FX_SLOTS_INSERTS_PHYSICAL_READBACK_CONTRACT_METHOD_ID:
        violations.append("fx_slots_inserts_readback_contract_method_id_mismatch")
    if contract.get("schema_id") != FX_SLOTS_INSERTS_PHYSICAL_READBACK_SCHEMA_ID:
        violations.append("fx_slots_inserts_readback_contract_schema_id_mismatch")
    if contract.get("status") != "contract_ready_physical_fixture_missing":
        violations.append("fx_slots_inserts_readback_contract_status_mismatch")
    if contract.get("surface_id") != FX_SLOTS_INSERTS_SURFACE_ID:
        violations.append("fx_slots_inserts_readback_contract_surface_id_mismatch")
    if contract.get("remediation_action_id") != FX_SLOTS_INSERTS_REMEDIATION_ACTION_ID:
        violations.append("fx_slots_inserts_readback_contract_action_id_mismatch")

    if safety.get("read_only") is not True:
        violations.append("fx_slots_inserts_readback_contract_read_only_not_true")
    for key in (
        "live_write",
        "mutates_console_state",
        "queries_console",
        "executes_tests",
        "live_widening_allowed",
        "full_autonomous_ready",
    ):
        if safety.get(key) is not False:
            violations.append(f"fx_slots_inserts_readback_contract_{key}_not_false")

    if artifact_contract.get("type") != FX_SLOTS_INSERTS_PHYSICAL_READBACK_ARTIFACT_TYPE:
        violations.append("fx_slots_inserts_readback_artifact_type_mismatch")
    if artifact_contract.get("schema_id") != FX_SLOTS_INSERTS_PHYSICAL_READBACK_SCHEMA_ID:
        violations.append("fx_slots_inserts_readback_artifact_schema_id_mismatch")
    if artifact_contract.get("minimum_fx_slot_count") != MINIMUM_FX_SLOT_COUNT:
        violations.append("fx_slots_inserts_readback_artifact_minimum_fx_slot_count_mismatch")
    if artifact_contract.get("minimum_fx_parameter_count") != MINIMUM_FX_PARAMETER_COUNT:
        violations.append("fx_slots_inserts_readback_artifact_minimum_fx_parameter_count_mismatch")
    if artifact_contract.get("minimum_insert_count") != MINIMUM_FX_INSERT_COUNT:
        violations.append("fx_slots_inserts_readback_artifact_minimum_insert_count_mismatch")

    expected_lists = {
        "required_top_level_fields": FX_SLOTS_INSERTS_REQUIRED_TOP_LEVEL_FIELDS,
        "required_fx_slot_fields": FX_SLOT_REQUIRED_FIELDS,
        "required_parameter_fields": FX_PARAMETER_REQUIRED_FIELDS,
        "required_parameter_safe_range_fields": FX_PARAMETER_SAFE_RANGE_FIELDS,
        "required_parameter_schema_fields": FX_PARAMETER_SCHEMA_REQUIRED_FIELDS,
        "required_fx_slot_coverage_flags": FX_SLOT_REQUIRED_COVERAGE_FLAGS,
        "required_insert_fields": FX_INSERT_REQUIRED_FIELDS,
        "required_insert_coverage_flags": FX_INSERT_REQUIRED_COVERAGE_FLAGS,
        "required_false_safety_flags": FX_SLOTS_INSERTS_REQUIRED_FALSE_SAFETY_FLAGS,
    }
    list_details: dict[str, list[object]] = {}
    for key, expected in expected_lists.items():
        value = artifact_contract.get(key)
        if not isinstance(value, list) or value != list(expected):
            violations.append(f"fx_slots_inserts_readback_{key}_mismatch")
            list_details[key] = []
        else:
            list_details[key] = list(value)

    allowed_sources = artifact_contract.get("allowed_sources")
    if not isinstance(allowed_sources, list):
        allowed_sources = []
        violations.append("fx_slots_inserts_readback_allowed_sources_missing")
    elif list(allowed_sources) != list(FX_SLOTS_INSERTS_PHYSICAL_READBACK_SOURCES):
        violations.append("fx_slots_inserts_readback_allowed_sources_mismatch")
    if "mixer_state_cache" in allowed_sources or "cache" in allowed_sources:
        violations.append("fx_slots_inserts_readback_allows_cached_source")

    accepted_statuses = artifact_contract.get("accepted_row_statuses")
    if (
        not isinstance(accepted_statuses, list)
        or accepted_statuses != list(FX_SLOTS_INSERTS_VERIFIED_PHYSICAL_STATUSES)
    ):
        violations.append("fx_slots_inserts_readback_accepted_statuses_mismatch")
        accepted_statuses = []
    target_types = artifact_contract.get("allowed_insert_target_types")
    if not isinstance(target_types, list) or target_types != list(FX_INSERT_TARGET_TYPES):
        violations.append("fx_slots_inserts_readback_insert_target_types_mismatch")
        target_types = []
    positions = artifact_contract.get("allowed_insert_positions")
    if not isinstance(positions, list) or positions != list(FX_INSERT_POSITIONS):
        violations.append("fx_slots_inserts_readback_insert_positions_mismatch")
        positions = []
    forbidden_mutation_fields = artifact_contract.get("forbidden_mutation_evidence_fields")
    if (
        not isinstance(forbidden_mutation_fields, list)
        or forbidden_mutation_fields
        != list(FX_SLOTS_INSERTS_FORBIDDEN_MUTATION_EVIDENCE_FIELDS)
    ):
        violations.append("fx_slots_inserts_readback_forbidden_mutation_fields_mismatch")
        forbidden_mutation_fields = []

    physical_gap_ids = fixture_summary.get("physical_fixture_missing_surface_ids")
    if not isinstance(physical_gap_ids, list):
        physical_gap_ids = []
        violations.append("fx_slots_inserts_fixture_physical_gap_ids_missing")
    if FX_SLOTS_INSERTS_SURFACE_ID not in physical_gap_ids:
        violations.append("fx_slots_inserts_fixture_gap_not_declared")

    remediation_action = {}
    for action in remediation_actions:
        if isinstance(action, Mapping) and action.get("surface_id") == FX_SLOTS_INSERTS_SURFACE_ID:
            remediation_action = dict(action)
            break
    if not remediation_action:
        violations.append("fx_slots_inserts_remediation_action_missing")
    elif remediation_action.get("id") != FX_SLOTS_INSERTS_REMEDIATION_ACTION_ID:
        violations.append("fx_slots_inserts_remediation_action_id_mismatch")
    if remediation_action and remediation_action.get("physical_readback_gap") is not True:
        violations.append("fx_slots_inserts_remediation_physical_gap_not_true")

    if contract_summary.get("physical_readback_surface_id") != FX_SLOTS_INSERTS_SURFACE_ID:
        violations.append("fx_slots_inserts_contract_surface_summary_mismatch")
    if contract_summary.get("physical_readback_action_id") != FX_SLOTS_INSERTS_REMEDIATION_ACTION_ID:
        violations.append("fx_slots_inserts_contract_action_summary_mismatch")
    if contract_summary.get("artifact_required") is not True:
        violations.append("fx_slots_inserts_contract_artifact_required_not_true")
    if contract_summary.get("physical_fixture_gap_remains") is not True:
        violations.append("fx_slots_inserts_contract_physical_gap_not_true")
    if contract_summary.get("accepted_as_physical_fixture_candidate") is not False:
        violations.append("fx_slots_inserts_contract_candidate_not_false")
    if contract_summary.get("live_authority_granted") is not False:
        violations.append("fx_slots_inserts_contract_live_authority_not_false")
    promotion_families = contract_summary.get("promotion_families_requiring_surface")
    if not isinstance(promotion_families, list):
        promotion_families = []
        violations.append("fx_slots_inserts_promotion_families_missing")
    for family in ("fx_send_insert", "routing", "mix_balance"):
        if family not in promotion_families:
            violations.append(f"fx_slots_inserts_promotion_family_missing:{family}")

    detail["contract_status"] = contract.get("status")
    detail["schema_id"] = contract.get("schema_id")
    detail["surface_id"] = contract.get("surface_id")
    detail["remediation_action_id"] = contract.get("remediation_action_id")
    detail["physical_readback_surface_id"] = contract_summary.get("physical_readback_surface_id")
    detail["physical_readback_action_id"] = contract_summary.get("physical_readback_action_id")
    detail["fixture_physical_gap_declared"] = FX_SLOTS_INSERTS_SURFACE_ID in physical_gap_ids
    detail["remediation_physical_gap_declared"] = bool(
        remediation_action and remediation_action.get("physical_readback_gap") is True
    )
    detail["artifact_required"] = contract_summary.get("artifact_required")
    detail["physical_fixture_gap_remains"] = contract_summary.get("physical_fixture_gap_remains")
    detail["accepted_as_physical_fixture_candidate"] = contract_summary.get(
        "accepted_as_physical_fixture_candidate"
    )
    detail["live_authority_granted"] = contract_summary.get("live_authority_granted")
    detail["promotion_families_requiring_surface"] = list(promotion_families)
    detail["required_top_level_field_count"] = len(
        list_details.get("required_top_level_fields", [])
    )
    detail["required_fx_slot_field_count"] = len(
        list_details.get("required_fx_slot_fields", [])
    )
    detail["required_fx_slot_fields"] = list(list_details.get("required_fx_slot_fields", []))
    detail["required_parameter_fields"] = list(
        list_details.get("required_parameter_fields", [])
    )
    detail["required_parameter_safe_range_fields"] = list(
        list_details.get("required_parameter_safe_range_fields", [])
    )
    detail["required_parameter_schema_fields"] = list(
        list_details.get("required_parameter_schema_fields", [])
    )
    detail["required_fx_slot_coverage_flags"] = list(
        list_details.get("required_fx_slot_coverage_flags", [])
    )
    detail["required_insert_fields"] = list(list_details.get("required_insert_fields", []))
    detail["required_insert_coverage_flags"] = list(
        list_details.get("required_insert_coverage_flags", [])
    )
    detail["required_false_safety_flags"] = list(
        list_details.get("required_false_safety_flags", [])
    )
    detail["minimum_fx_slot_count"] = int(
        artifact_contract.get("minimum_fx_slot_count") or 0
    )
    detail["minimum_fx_parameter_count"] = int(
        artifact_contract.get("minimum_fx_parameter_count") or 0
    )
    detail["minimum_insert_count"] = int(artifact_contract.get("minimum_insert_count") or 0)
    detail["allowed_sources"] = list(allowed_sources)
    detail["accepted_row_statuses"] = list(accepted_statuses)
    detail["allowed_insert_target_types"] = list(target_types)
    detail["allowed_insert_positions"] = list(positions)
    detail["forbidden_mutation_evidence_fields"] = list(forbidden_mutation_fields)
    detail["violations"] = violations
    finished = datetime.now(timezone.utc)
    return CheckResult(
        name="fx_slots_inserts_physical_readback_contract_schema",
        status="passed" if not violations else "failed",
        started_at=started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        finished_at=finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration_sec=round((finished - started).total_seconds(), 3),
        detail=detail,
    )


def validate_fx_slots_inserts_physical_readback_artifact_intake_snapshot() -> CheckResult:
    started = datetime.now(timezone.utc)
    detail: dict[str, object] = {
        "policy": "fx_slots_inserts_artifact_intake_is_read_only_operator_review_gate",
    }
    violations: list[str] = []

    try:
        intake = build_fx_slots_inserts_physical_readback_artifact_intake()
    except Exception as exc:  # pragma: no cover - defensive release-check guard
        intake = {}
        violations.append(f"fx_slots_inserts_artifact_intake_build_error:{type(exc).__name__}")

    safety = intake.get("safety") if isinstance(intake.get("safety"), Mapping) else {}
    summary = intake.get("summary") if isinstance(intake.get("summary"), Mapping) else {}
    artifacts = intake.get("artifacts") if isinstance(intake.get("artifacts"), list) else []

    if intake.get("type") != "fx_slots_inserts_physical_readback_artifact_intake":
        violations.append("fx_slots_inserts_artifact_intake_type_mismatch")
    if intake.get("method_id") != FX_SLOTS_INSERTS_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID:
        violations.append("fx_slots_inserts_artifact_intake_method_id_mismatch")
    if (
        intake.get("required_contract_method_id")
        != FX_SLOTS_INSERTS_PHYSICAL_READBACK_CONTRACT_METHOD_ID
    ):
        violations.append("fx_slots_inserts_artifact_intake_required_contract_mismatch")
    if intake.get("schema_id") != FX_SLOTS_INSERTS_PHYSICAL_READBACK_SCHEMA_ID:
        violations.append("fx_slots_inserts_artifact_intake_schema_id_mismatch")
    if intake.get("status") not in FX_SLOTS_INSERTS_INTAKE_STATUSES:
        violations.append("fx_slots_inserts_artifact_intake_status_unknown")
    if intake.get("surface_id") != FX_SLOTS_INSERTS_SURFACE_ID:
        violations.append("fx_slots_inserts_artifact_intake_surface_id_mismatch")
    if intake.get("remediation_action_id") != FX_SLOTS_INSERTS_REMEDIATION_ACTION_ID:
        violations.append("fx_slots_inserts_artifact_intake_action_id_mismatch")

    for key in ("read_only", "operator_required"):
        if safety.get(key) is not True:
            violations.append(f"fx_slots_inserts_artifact_intake_{key}_not_true")
    for key in (
        "live_write",
        "mutates_console_state",
        "queries_console",
        "executes_tests",
        "execution_allowed_by_intake",
        "fixture_closure_allowed_by_intake",
        "live_widening_allowed",
        "full_autonomous_ready",
    ):
        if safety.get(key) is not False:
            violations.append(f"fx_slots_inserts_artifact_intake_{key}_not_false")

    artifact_count = summary.get("artifact_count")
    valid_count = summary.get("valid_candidate_count")
    invalid_count = summary.get("invalid_candidate_count")
    valid_fx_slot_count = summary.get("valid_fx_slot_count")
    valid_insert_count = summary.get("valid_insert_count")
    if not isinstance(artifact_count, int) or artifact_count != len(artifacts):
        violations.append("fx_slots_inserts_artifact_intake_artifact_count_mismatch")
        artifact_count = len(artifacts)
    if not isinstance(valid_count, int):
        violations.append("fx_slots_inserts_artifact_intake_valid_count_missing")
        valid_count = 0
    if not isinstance(invalid_count, int):
        violations.append("fx_slots_inserts_artifact_intake_invalid_count_missing")
        invalid_count = 0
    if not isinstance(valid_fx_slot_count, int):
        violations.append("fx_slots_inserts_artifact_intake_valid_fx_slot_count_missing")
        valid_fx_slot_count = 0
    if not isinstance(valid_insert_count, int):
        violations.append("fx_slots_inserts_artifact_intake_valid_insert_count_missing")
        valid_insert_count = 0
    if valid_count + invalid_count != artifact_count:
        violations.append("fx_slots_inserts_artifact_intake_candidate_count_mismatch")
    if invalid_count:
        violations.append("fx_slots_inserts_artifact_intake_invalid_candidate_present")

    if summary.get("operator_review_required") is not True:
        violations.append("fx_slots_inserts_artifact_intake_operator_review_not_true")
    if summary.get("physical_fixture_gap_remains") is not True:
        violations.append("fx_slots_inserts_artifact_intake_physical_gap_not_true")
    if summary.get("fixture_closure_allowed_by_intake") is not False:
        violations.append("fx_slots_inserts_artifact_intake_fixture_closure_not_false")
    if summary.get("live_authority_granted") is not False:
        violations.append("fx_slots_inserts_artifact_intake_live_authority_not_false")
    if summary.get("full_autonomous_ready") is not False:
        violations.append("fx_slots_inserts_artifact_intake_full_autonomous_not_false")
    expected_candidate_accepted = bool(valid_count) and invalid_count == 0
    if summary.get("accepted_as_physical_fixture_candidate") is not expected_candidate_accepted:
        violations.append("fx_slots_inserts_artifact_intake_candidate_flag_mismatch")

    promotion_families = summary.get("promotion_families_requiring_surface")
    if not isinstance(promotion_families, list):
        promotion_families = []
        violations.append("fx_slots_inserts_artifact_intake_promotion_families_missing")
    for family in ("fx_send_insert", "routing", "mix_balance", "performance_adaptation"):
        if family not in promotion_families:
            violations.append(f"fx_slots_inserts_artifact_intake_promotion_family_missing:{family}")

    intake_violations = (
        list(summary.get("violations")) if isinstance(summary.get("violations"), list) else []
    )
    artifact_source_ids = [
        row.get("source_id")
        for row in artifacts
        if isinstance(row, Mapping) and isinstance(row.get("source_id"), str)
    ]
    detail["intake_status"] = intake.get("status")
    detail["schema_id"] = intake.get("schema_id")
    detail["surface_id"] = intake.get("surface_id")
    detail["remediation_action_id"] = intake.get("remediation_action_id")
    detail["required_contract_method_id"] = intake.get("required_contract_method_id")
    detail["artifact_search_roots"] = (
        list(summary.get("artifact_search_roots"))
        if isinstance(summary.get("artifact_search_roots"), list)
        else []
    )
    detail["artifact_count"] = artifact_count
    detail["valid_candidate_count"] = valid_count
    detail["invalid_candidate_count"] = invalid_count
    detail["valid_fx_slot_count"] = valid_fx_slot_count
    detail["valid_insert_count"] = valid_insert_count
    detail["accepted_as_physical_fixture_candidate"] = summary.get(
        "accepted_as_physical_fixture_candidate"
    )
    detail["operator_review_required"] = summary.get("operator_review_required")
    detail["physical_fixture_gap_remains"] = summary.get("physical_fixture_gap_remains")
    detail["fixture_closure_allowed_by_intake"] = summary.get(
        "fixture_closure_allowed_by_intake"
    )
    detail["live_authority_granted"] = summary.get("live_authority_granted")
    detail["full_autonomous_ready"] = summary.get("full_autonomous_ready")
    detail["promotion_families_requiring_surface"] = list(promotion_families)
    detail["latest_valid_candidate_source_id"] = summary.get("latest_valid_candidate_source_id")
    detail["latest_valid_candidate_captured_at"] = summary.get(
        "latest_valid_candidate_captured_at"
    )
    detail["blocked_by"] = (
        list(summary.get("blocked_by")) if isinstance(summary.get("blocked_by"), list) else []
    )
    detail["artifact_source_ids"] = artifact_source_ids
    detail["intake_violations"] = intake_violations
    detail["violations"] = violations
    finished = datetime.now(timezone.utc)
    return CheckResult(
        name="fx_slots_inserts_physical_readback_artifact_intake_schema",
        status="passed" if not violations else "failed",
        started_at=started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        finished_at=finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration_sec=round((finished - started).total_seconds(), 3),
        detail=detail,
    )


def validate_routing_sends_physical_readback_contract_snapshot() -> CheckResult:
    started = datetime.now(timezone.utc)
    detail: dict[str, object] = {
        "policy": "routing_sends_readback_contract_is_schema_only_no_live_authority",
    }
    violations: list[str] = []

    try:
        fixture_snapshot = build_read_surface_fixture_evidence_snapshot()
        remediation_plan = build_read_surface_remediation_plan(
            fixture_evidence_snapshot=fixture_snapshot
        )
        contract = build_routing_sends_physical_readback_contract()
    except Exception as exc:  # pragma: no cover - defensive release-check guard
        fixture_snapshot = {}
        remediation_plan = {}
        contract = {}
        violations.append(f"routing_sends_readback_contract_build_error:{type(exc).__name__}")

    fixture_summary = (
        fixture_snapshot.get("summary")
        if isinstance(fixture_snapshot.get("summary"), Mapping)
        else {}
    )
    remediation_actions = (
        remediation_plan.get("actions")
        if isinstance(remediation_plan.get("actions"), list)
        else []
    )
    safety = contract.get("safety") if isinstance(contract.get("safety"), Mapping) else {}
    contract_summary = (
        contract.get("summary") if isinstance(contract.get("summary"), Mapping) else {}
    )
    artifact_contract = (
        contract.get("artifact_contract")
        if isinstance(contract.get("artifact_contract"), Mapping)
        else {}
    )

    if contract.get("type") != "routing_sends_physical_readback_contract":
        violations.append("routing_sends_readback_contract_type_mismatch")
    if contract.get("method_id") != ROUTING_SENDS_PHYSICAL_READBACK_CONTRACT_METHOD_ID:
        violations.append("routing_sends_readback_contract_method_id_mismatch")
    if contract.get("schema_id") != ROUTING_SENDS_PHYSICAL_READBACK_SCHEMA_ID:
        violations.append("routing_sends_readback_contract_schema_id_mismatch")
    if contract.get("status") != "contract_ready_physical_fixture_missing":
        violations.append("routing_sends_readback_contract_status_mismatch")
    if contract.get("surface_id") != ROUTING_SENDS_SURFACE_ID:
        violations.append("routing_sends_readback_contract_surface_id_mismatch")
    if contract.get("remediation_action_id") != ROUTING_SENDS_REMEDIATION_ACTION_ID:
        violations.append("routing_sends_readback_contract_action_id_mismatch")

    if safety.get("read_only") is not True:
        violations.append("routing_sends_readback_contract_read_only_not_true")
    for key in (
        "live_write",
        "mutates_console_state",
        "queries_console",
        "executes_tests",
        "live_widening_allowed",
        "full_autonomous_ready",
    ):
        if safety.get(key) is not False:
            violations.append(f"routing_sends_readback_contract_{key}_not_false")

    if artifact_contract.get("type") != ROUTING_SENDS_PHYSICAL_READBACK_ARTIFACT_TYPE:
        violations.append("routing_sends_readback_artifact_type_mismatch")
    if artifact_contract.get("schema_id") != ROUTING_SENDS_PHYSICAL_READBACK_SCHEMA_ID:
        violations.append("routing_sends_readback_artifact_schema_id_mismatch")

    minimums = {
        "minimum_static_dante_route_count": MINIMUM_STATIC_DANTE_ROUTE_COUNT,
        "minimum_channel_bus_send_count": MINIMUM_CHANNEL_BUS_SEND_COUNT,
        "minimum_channel_main_send_count": MINIMUM_CHANNEL_MAIN_SEND_COUNT,
        "minimum_channel_matrix_send_count": MINIMUM_CHANNEL_MATRIX_SEND_COUNT,
        "minimum_output_route_count": MINIMUM_OUTPUT_ROUTE_COUNT,
    }
    for key, expected in minimums.items():
        if artifact_contract.get(key) != expected:
            violations.append(f"routing_sends_readback_artifact_{key}_mismatch")

    expected_lists = {
        "required_top_level_fields": ROUTING_SENDS_REQUIRED_TOP_LEVEL_FIELDS,
        "required_static_route_fields": ROUTING_SENDS_REQUIRED_STATIC_ROUTE_FIELDS,
        "required_bus_send_fields": ROUTING_SENDS_REQUIRED_BUS_SEND_FIELDS,
        "required_main_send_fields": ROUTING_SENDS_REQUIRED_MAIN_SEND_FIELDS,
        "required_matrix_send_fields": ROUTING_SENDS_REQUIRED_MATRIX_SEND_FIELDS,
        "required_output_route_fields": ROUTING_SENDS_REQUIRED_OUTPUT_ROUTE_FIELDS,
        "required_static_route_coverage_flags": (
            ROUTING_SENDS_REQUIRED_STATIC_ROUTE_COVERAGE_FLAGS
        ),
        "required_send_coverage_flags": ROUTING_SENDS_REQUIRED_SEND_COVERAGE_FLAGS,
        "required_output_route_coverage_flags": (
            ROUTING_SENDS_REQUIRED_OUTPUT_ROUTE_COVERAGE_FLAGS
        ),
        "required_false_safety_flags": ROUTING_SENDS_REQUIRED_FALSE_SAFETY_FLAGS,
    }
    list_details: dict[str, list[object]] = {}
    for key, expected in expected_lists.items():
        value = artifact_contract.get(key)
        if not isinstance(value, list) or value != list(expected):
            violations.append(f"routing_sends_readback_{key}_mismatch")
            list_details[key] = []
        else:
            list_details[key] = list(value)

    allowed_sources = artifact_contract.get("allowed_sources")
    if not isinstance(allowed_sources, list):
        allowed_sources = []
        violations.append("routing_sends_readback_allowed_sources_missing")
    elif list(allowed_sources) != list(ROUTING_SENDS_PHYSICAL_READBACK_SOURCES):
        violations.append("routing_sends_readback_allowed_sources_mismatch")
    if "mixer_state_cache" in allowed_sources or "cache" in allowed_sources:
        violations.append("routing_sends_readback_allows_cached_source")

    accepted_statuses = artifact_contract.get("accepted_row_statuses")
    if (
        not isinstance(accepted_statuses, list)
        or accepted_statuses != list(ROUTING_SENDS_VERIFIED_PHYSICAL_STATUSES)
    ):
        violations.append("routing_sends_readback_accepted_statuses_mismatch")
        accepted_statuses = []
    forbidden_mutation_fields = artifact_contract.get("forbidden_mutation_evidence_fields")
    if (
        not isinstance(forbidden_mutation_fields, list)
        or forbidden_mutation_fields
        != list(ROUTING_SENDS_FORBIDDEN_MUTATION_EVIDENCE_FIELDS)
    ):
        violations.append("routing_sends_readback_forbidden_mutation_fields_mismatch")
        forbidden_mutation_fields = []

    physical_gap_ids = fixture_summary.get("physical_fixture_missing_surface_ids")
    if not isinstance(physical_gap_ids, list):
        physical_gap_ids = []
        violations.append("routing_sends_fixture_physical_gap_ids_missing")
    if ROUTING_SENDS_SURFACE_ID not in physical_gap_ids:
        violations.append("routing_sends_fixture_gap_not_declared")

    remediation_action = {}
    for action in remediation_actions:
        if isinstance(action, Mapping) and action.get("surface_id") == ROUTING_SENDS_SURFACE_ID:
            remediation_action = dict(action)
            break
    if not remediation_action:
        violations.append("routing_sends_remediation_action_missing")
    elif remediation_action.get("id") != ROUTING_SENDS_REMEDIATION_ACTION_ID:
        violations.append("routing_sends_remediation_action_id_mismatch")
    if remediation_action and remediation_action.get("physical_readback_gap") is not True:
        violations.append("routing_sends_remediation_physical_gap_not_true")

    if contract_summary.get("physical_readback_surface_id") != ROUTING_SENDS_SURFACE_ID:
        violations.append("routing_sends_contract_surface_summary_mismatch")
    if contract_summary.get("physical_readback_action_id") != ROUTING_SENDS_REMEDIATION_ACTION_ID:
        violations.append("routing_sends_contract_action_summary_mismatch")
    if contract_summary.get("artifact_required") is not True:
        violations.append("routing_sends_contract_artifact_required_not_true")
    if contract_summary.get("physical_fixture_gap_remains") is not True:
        violations.append("routing_sends_contract_physical_gap_not_true")
    if contract_summary.get("accepted_as_physical_fixture_candidate") is not False:
        violations.append("routing_sends_contract_candidate_not_false")
    if contract_summary.get("live_authority_granted") is not False:
        violations.append("routing_sends_contract_live_authority_not_false")
    promotion_families = contract_summary.get("promotion_families_requiring_surface")
    if not isinstance(promotion_families, list):
        promotion_families = []
        violations.append("routing_sends_promotion_families_missing")
    for family in ("routing", "fx_send_insert", "scene_preset"):
        if family not in promotion_families:
            violations.append(f"routing_sends_promotion_family_missing:{family}")

    detail["contract_status"] = contract.get("status")
    detail["schema_id"] = contract.get("schema_id")
    detail["surface_id"] = contract.get("surface_id")
    detail["remediation_action_id"] = contract.get("remediation_action_id")
    detail["physical_readback_surface_id"] = contract_summary.get("physical_readback_surface_id")
    detail["physical_readback_action_id"] = contract_summary.get("physical_readback_action_id")
    detail["fixture_physical_gap_declared"] = ROUTING_SENDS_SURFACE_ID in physical_gap_ids
    detail["remediation_physical_gap_declared"] = bool(
        remediation_action and remediation_action.get("physical_readback_gap") is True
    )
    detail["artifact_required"] = contract_summary.get("artifact_required")
    detail["physical_fixture_gap_remains"] = contract_summary.get("physical_fixture_gap_remains")
    detail["accepted_as_physical_fixture_candidate"] = contract_summary.get(
        "accepted_as_physical_fixture_candidate"
    )
    detail["live_authority_granted"] = contract_summary.get("live_authority_granted")
    detail["promotion_families_requiring_surface"] = list(promotion_families)
    detail["required_top_level_field_count"] = len(
        list_details.get("required_top_level_fields", [])
    )
    detail["required_static_route_field_count"] = len(
        list_details.get("required_static_route_fields", [])
    )
    detail["required_static_route_fields"] = list(
        list_details.get("required_static_route_fields", [])
    )
    detail["required_bus_send_fields"] = list(
        list_details.get("required_bus_send_fields", [])
    )
    detail["required_main_send_fields"] = list(
        list_details.get("required_main_send_fields", [])
    )
    detail["required_matrix_send_fields"] = list(
        list_details.get("required_matrix_send_fields", [])
    )
    detail["required_output_route_fields"] = list(
        list_details.get("required_output_route_fields", [])
    )
    detail["required_static_route_coverage_flags"] = list(
        list_details.get("required_static_route_coverage_flags", [])
    )
    detail["required_send_coverage_flags"] = list(
        list_details.get("required_send_coverage_flags", [])
    )
    detail["required_output_route_coverage_flags"] = list(
        list_details.get("required_output_route_coverage_flags", [])
    )
    detail["required_false_safety_flags"] = list(
        list_details.get("required_false_safety_flags", [])
    )
    for key in minimums:
        detail[key] = int(artifact_contract.get(key) or 0)
    detail["allowed_sources"] = list(allowed_sources)
    detail["accepted_row_statuses"] = list(accepted_statuses)
    detail["forbidden_mutation_evidence_fields"] = list(forbidden_mutation_fields)
    detail["violations"] = violations
    finished = datetime.now(timezone.utc)
    return CheckResult(
        name="routing_sends_physical_readback_contract_schema",
        status="passed" if not violations else "failed",
        started_at=started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        finished_at=finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration_sec=round((finished - started).total_seconds(), 3),
        detail=detail,
    )


def validate_routing_sends_physical_readback_artifact_intake_snapshot() -> CheckResult:
    started = datetime.now(timezone.utc)
    detail: dict[str, object] = {
        "policy": "routing_sends_artifact_intake_is_read_only_operator_review_gate",
    }
    violations: list[str] = []

    try:
        intake = build_routing_sends_physical_readback_artifact_intake()
    except Exception as exc:  # pragma: no cover - defensive release-check guard
        intake = {}
        violations.append(f"routing_sends_artifact_intake_build_error:{type(exc).__name__}")

    safety = intake.get("safety") if isinstance(intake.get("safety"), Mapping) else {}
    summary = intake.get("summary") if isinstance(intake.get("summary"), Mapping) else {}
    artifacts = intake.get("artifacts") if isinstance(intake.get("artifacts"), list) else []

    if intake.get("type") != "routing_sends_physical_readback_artifact_intake":
        violations.append("routing_sends_artifact_intake_type_mismatch")
    if intake.get("method_id") != ROUTING_SENDS_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID:
        violations.append("routing_sends_artifact_intake_method_id_mismatch")
    if (
        intake.get("required_contract_method_id")
        != ROUTING_SENDS_PHYSICAL_READBACK_CONTRACT_METHOD_ID
    ):
        violations.append("routing_sends_artifact_intake_required_contract_mismatch")
    if intake.get("schema_id") != ROUTING_SENDS_PHYSICAL_READBACK_SCHEMA_ID:
        violations.append("routing_sends_artifact_intake_schema_id_mismatch")
    if intake.get("status") not in ROUTING_SENDS_INTAKE_STATUSES:
        violations.append("routing_sends_artifact_intake_status_unknown")
    if intake.get("surface_id") != ROUTING_SENDS_SURFACE_ID:
        violations.append("routing_sends_artifact_intake_surface_id_mismatch")
    if intake.get("remediation_action_id") != ROUTING_SENDS_REMEDIATION_ACTION_ID:
        violations.append("routing_sends_artifact_intake_action_id_mismatch")

    for key in ("read_only", "operator_required"):
        if safety.get(key) is not True:
            violations.append(f"routing_sends_artifact_intake_{key}_not_true")
    for key in (
        "live_write",
        "mutates_console_state",
        "queries_console",
        "executes_tests",
        "execution_allowed_by_intake",
        "fixture_closure_allowed_by_intake",
        "live_widening_allowed",
        "full_autonomous_ready",
    ):
        if safety.get(key) is not False:
            violations.append(f"routing_sends_artifact_intake_{key}_not_false")

    artifact_count = summary.get("artifact_count")
    valid_count = summary.get("valid_candidate_count")
    invalid_count = summary.get("invalid_candidate_count")
    valid_static_count = summary.get("valid_static_dante_route_count")
    valid_bus_count = summary.get("valid_channel_bus_send_count")
    valid_main_count = summary.get("valid_channel_main_send_count")
    valid_matrix_count = summary.get("valid_channel_matrix_send_count")
    valid_output_count = summary.get("valid_output_route_count")
    if not isinstance(artifact_count, int) or artifact_count != len(artifacts):
        violations.append("routing_sends_artifact_intake_artifact_count_mismatch")
        artifact_count = len(artifacts)
    if not isinstance(valid_count, int):
        violations.append("routing_sends_artifact_intake_valid_count_missing")
        valid_count = 0
    if not isinstance(invalid_count, int):
        violations.append("routing_sends_artifact_intake_invalid_count_missing")
        invalid_count = 0
    count_fields = {
        "valid_static_dante_route_count": valid_static_count,
        "valid_channel_bus_send_count": valid_bus_count,
        "valid_channel_main_send_count": valid_main_count,
        "valid_channel_matrix_send_count": valid_matrix_count,
        "valid_output_route_count": valid_output_count,
    }
    for key, value in list(count_fields.items()):
        if not isinstance(value, int):
            violations.append(f"routing_sends_artifact_intake_{key}_missing")
            count_fields[key] = 0
    if valid_count + invalid_count != artifact_count:
        violations.append("routing_sends_artifact_intake_candidate_count_mismatch")
    if invalid_count:
        violations.append("routing_sends_artifact_intake_invalid_candidate_present")

    if summary.get("operator_review_required") is not True:
        violations.append("routing_sends_artifact_intake_operator_review_not_true")
    if summary.get("physical_fixture_gap_remains") is not True:
        violations.append("routing_sends_artifact_intake_physical_gap_not_true")
    if summary.get("fixture_closure_allowed_by_intake") is not False:
        violations.append("routing_sends_artifact_intake_fixture_closure_not_false")
    if summary.get("live_authority_granted") is not False:
        violations.append("routing_sends_artifact_intake_live_authority_not_false")
    if summary.get("full_autonomous_ready") is not False:
        violations.append("routing_sends_artifact_intake_full_autonomous_not_false")
    expected_candidate_accepted = bool(valid_count) and invalid_count == 0
    if summary.get("accepted_as_physical_fixture_candidate") is not expected_candidate_accepted:
        violations.append("routing_sends_artifact_intake_candidate_flag_mismatch")

    promotion_families = summary.get("promotion_families_requiring_surface")
    if not isinstance(promotion_families, list):
        promotion_families = []
        violations.append("routing_sends_artifact_intake_promotion_families_missing")
    for family in (
        "routing",
        "fx_send_insert",
        "bus_group_matrix",
        "scene_preset",
        "mix_balance",
        "performance_adaptation",
    ):
        if family not in promotion_families:
            violations.append(f"routing_sends_artifact_intake_promotion_family_missing:{family}")

    intake_violations = (
        list(summary.get("violations")) if isinstance(summary.get("violations"), list) else []
    )
    artifact_source_ids = [
        row.get("source_id")
        for row in artifacts
        if isinstance(row, Mapping) and isinstance(row.get("source_id"), str)
    ]
    detail["intake_status"] = intake.get("status")
    detail["schema_id"] = intake.get("schema_id")
    detail["surface_id"] = intake.get("surface_id")
    detail["remediation_action_id"] = intake.get("remediation_action_id")
    detail["required_contract_method_id"] = intake.get("required_contract_method_id")
    detail["artifact_search_roots"] = (
        list(summary.get("artifact_search_roots"))
        if isinstance(summary.get("artifact_search_roots"), list)
        else []
    )
    detail["artifact_count"] = artifact_count
    detail["valid_candidate_count"] = valid_count
    detail["invalid_candidate_count"] = invalid_count
    detail["valid_static_dante_route_count"] = count_fields[
        "valid_static_dante_route_count"
    ]
    detail["valid_channel_bus_send_count"] = count_fields["valid_channel_bus_send_count"]
    detail["valid_channel_main_send_count"] = count_fields["valid_channel_main_send_count"]
    detail["valid_channel_matrix_send_count"] = count_fields[
        "valid_channel_matrix_send_count"
    ]
    detail["valid_output_route_count"] = count_fields["valid_output_route_count"]
    detail["accepted_as_physical_fixture_candidate"] = summary.get(
        "accepted_as_physical_fixture_candidate"
    )
    detail["operator_review_required"] = summary.get("operator_review_required")
    detail["physical_fixture_gap_remains"] = summary.get("physical_fixture_gap_remains")
    detail["fixture_closure_allowed_by_intake"] = summary.get(
        "fixture_closure_allowed_by_intake"
    )
    detail["live_authority_granted"] = summary.get("live_authority_granted")
    detail["full_autonomous_ready"] = summary.get("full_autonomous_ready")
    detail["promotion_families_requiring_surface"] = list(promotion_families)
    detail["latest_valid_candidate_source_id"] = summary.get("latest_valid_candidate_source_id")
    detail["latest_valid_candidate_captured_at"] = summary.get(
        "latest_valid_candidate_captured_at"
    )
    detail["blocked_by"] = (
        list(summary.get("blocked_by")) if isinstance(summary.get("blocked_by"), list) else []
    )
    detail["artifact_source_ids"] = artifact_source_ids
    detail["intake_violations"] = intake_violations
    detail["violations"] = violations
    finished = datetime.now(timezone.utc)
    return CheckResult(
        name="routing_sends_physical_readback_artifact_intake_schema",
        status="passed" if not violations else "failed",
        started_at=started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        finished_at=finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration_sec=round((finished - started).total_seconds(), 3),
        detail=detail,
    )


def validate_snapshots_scenes_physical_readback_contract_snapshot() -> CheckResult:
    started = datetime.now(timezone.utc)
    detail: dict[str, object] = {
        "policy": "snapshots_scenes_readback_contract_is_schema_only_no_live_authority",
    }
    violations: list[str] = []

    try:
        fixture_snapshot = build_read_surface_fixture_evidence_snapshot()
        remediation_plan = build_read_surface_remediation_plan(
            fixture_evidence_snapshot=fixture_snapshot
        )
        contract = build_snapshots_scenes_physical_readback_contract()
    except Exception as exc:  # pragma: no cover - defensive release-check guard
        fixture_snapshot = {}
        remediation_plan = {}
        contract = {}
        violations.append(f"snapshots_scenes_readback_contract_build_error:{type(exc).__name__}")

    fixture_summary = (
        fixture_snapshot.get("summary")
        if isinstance(fixture_snapshot.get("summary"), Mapping)
        else {}
    )
    remediation_actions = (
        remediation_plan.get("actions")
        if isinstance(remediation_plan.get("actions"), list)
        else []
    )
    safety = contract.get("safety") if isinstance(contract.get("safety"), Mapping) else {}
    contract_summary = (
        contract.get("summary") if isinstance(contract.get("summary"), Mapping) else {}
    )
    artifact_contract = (
        contract.get("artifact_contract")
        if isinstance(contract.get("artifact_contract"), Mapping)
        else {}
    )

    if contract.get("type") != "snapshots_scenes_physical_readback_contract":
        violations.append("snapshots_scenes_readback_contract_type_mismatch")
    if contract.get("method_id") != SNAPSHOTS_SCENES_PHYSICAL_READBACK_CONTRACT_METHOD_ID:
        violations.append("snapshots_scenes_readback_contract_method_id_mismatch")
    if contract.get("schema_id") != SNAPSHOTS_SCENES_PHYSICAL_READBACK_SCHEMA_ID:
        violations.append("snapshots_scenes_readback_contract_schema_id_mismatch")
    if contract.get("status") != "contract_ready_physical_fixture_missing":
        violations.append("snapshots_scenes_readback_contract_status_mismatch")
    if contract.get("surface_id") != SNAPSHOTS_SCENES_SURFACE_ID:
        violations.append("snapshots_scenes_readback_contract_surface_id_mismatch")
    if contract.get("remediation_action_id") != SNAPSHOTS_SCENES_REMEDIATION_ACTION_ID:
        violations.append("snapshots_scenes_readback_contract_action_id_mismatch")

    if safety.get("read_only") is not True:
        violations.append("snapshots_scenes_readback_contract_read_only_not_true")
    for key in (
        "live_write",
        "mutates_console_state",
        "queries_console",
        "executes_tests",
        "live_widening_allowed",
        "full_autonomous_ready",
        "scene_load_allowed",
        "scene_save_allowed",
    ):
        if safety.get(key) is not False:
            violations.append(f"snapshots_scenes_readback_contract_{key}_not_false")

    if artifact_contract.get("type") != SNAPSHOTS_SCENES_PHYSICAL_READBACK_ARTIFACT_TYPE:
        violations.append("snapshots_scenes_readback_artifact_type_mismatch")
    if artifact_contract.get("schema_id") != SNAPSHOTS_SCENES_PHYSICAL_READBACK_SCHEMA_ID:
        violations.append("snapshots_scenes_readback_artifact_schema_id_mismatch")
    if artifact_contract.get("minimum_snapshot_count") != MINIMUM_SNAPSHOT_SCENE_COUNT:
        violations.append("snapshots_scenes_readback_artifact_minimum_snapshot_count_mismatch")

    expected_lists = {
        "required_top_level_fields": SNAPSHOTS_SCENES_REQUIRED_TOP_LEVEL_FIELDS,
        "required_active_scene_fields": SNAPSHOTS_SCENES_REQUIRED_ACTIVE_SCENE_FIELDS,
        "required_snapshot_fields": SNAPSHOTS_SCENES_REQUIRED_SNAPSHOT_FIELDS,
        "required_coverage_flags": SNAPSHOTS_SCENES_REQUIRED_COVERAGE_FLAGS,
        "required_false_safety_flags": SNAPSHOTS_SCENES_REQUIRED_FALSE_SAFETY_FLAGS,
    }
    list_details: dict[str, list[object]] = {}
    for key, expected in expected_lists.items():
        value = artifact_contract.get(key)
        if not isinstance(value, list) or value != list(expected):
            violations.append(f"snapshots_scenes_readback_{key}_mismatch")
            list_details[key] = []
        else:
            list_details[key] = list(value)

    allowed_sources = artifact_contract.get("allowed_sources")
    if not isinstance(allowed_sources, list):
        allowed_sources = []
        violations.append("snapshots_scenes_readback_allowed_sources_missing")
    elif list(allowed_sources) != list(SNAPSHOTS_SCENES_PHYSICAL_READBACK_SOURCES):
        violations.append("snapshots_scenes_readback_allowed_sources_mismatch")
    if "mixer_state_cache" in allowed_sources or "cache" in allowed_sources:
        violations.append("snapshots_scenes_readback_allows_cached_source")

    accepted_statuses = artifact_contract.get("accepted_row_statuses")
    if (
        not isinstance(accepted_statuses, list)
        or accepted_statuses != list(SNAPSHOTS_SCENES_VERIFIED_PHYSICAL_STATUSES)
    ):
        violations.append("snapshots_scenes_readback_accepted_statuses_mismatch")
        accepted_statuses = []
    forbidden_mutation_fields = artifact_contract.get("forbidden_mutation_evidence_fields")
    if (
        not isinstance(forbidden_mutation_fields, list)
        or forbidden_mutation_fields
        != list(SNAPSHOTS_SCENES_FORBIDDEN_MUTATION_EVIDENCE_FIELDS)
    ):
        violations.append("snapshots_scenes_readback_forbidden_mutation_fields_mismatch")
        forbidden_mutation_fields = []

    physical_gap_ids = fixture_summary.get("physical_fixture_missing_surface_ids")
    if not isinstance(physical_gap_ids, list):
        physical_gap_ids = []
        violations.append("snapshots_scenes_fixture_physical_gap_ids_missing")
    if SNAPSHOTS_SCENES_SURFACE_ID not in physical_gap_ids:
        violations.append("snapshots_scenes_fixture_gap_not_declared")

    remediation_action = {}
    for action in remediation_actions:
        if isinstance(action, Mapping) and action.get("surface_id") == SNAPSHOTS_SCENES_SURFACE_ID:
            remediation_action = dict(action)
            break
    if not remediation_action:
        violations.append("snapshots_scenes_remediation_action_missing")
    elif remediation_action.get("id") != SNAPSHOTS_SCENES_REMEDIATION_ACTION_ID:
        violations.append("snapshots_scenes_remediation_action_id_mismatch")
    if remediation_action and remediation_action.get("physical_readback_gap") is not True:
        violations.append("snapshots_scenes_remediation_physical_gap_not_true")

    if contract_summary.get("physical_readback_surface_id") != SNAPSHOTS_SCENES_SURFACE_ID:
        violations.append("snapshots_scenes_contract_surface_summary_mismatch")
    if contract_summary.get("physical_readback_action_id") != SNAPSHOTS_SCENES_REMEDIATION_ACTION_ID:
        violations.append("snapshots_scenes_contract_action_summary_mismatch")
    if contract_summary.get("artifact_required") is not True:
        violations.append("snapshots_scenes_contract_artifact_required_not_true")
    if contract_summary.get("physical_fixture_gap_remains") is not True:
        violations.append("snapshots_scenes_contract_physical_gap_not_true")
    if contract_summary.get("accepted_as_physical_fixture_candidate") is not False:
        violations.append("snapshots_scenes_contract_candidate_not_false")
    if contract_summary.get("live_authority_granted") is not False:
        violations.append("snapshots_scenes_contract_live_authority_not_false")
    promotion_families = contract_summary.get("promotion_families_requiring_surface")
    if not isinstance(promotion_families, list):
        promotion_families = []
        violations.append("snapshots_scenes_promotion_families_missing")
    for family in ("scene_preset", "routing"):
        if family not in promotion_families:
            violations.append(f"snapshots_scenes_promotion_family_missing:{family}")

    detail["contract_status"] = contract.get("status")
    detail["schema_id"] = contract.get("schema_id")
    detail["surface_id"] = contract.get("surface_id")
    detail["remediation_action_id"] = contract.get("remediation_action_id")
    detail["physical_readback_surface_id"] = contract_summary.get("physical_readback_surface_id")
    detail["physical_readback_action_id"] = contract_summary.get("physical_readback_action_id")
    detail["fixture_physical_gap_declared"] = SNAPSHOTS_SCENES_SURFACE_ID in physical_gap_ids
    detail["remediation_physical_gap_declared"] = bool(
        remediation_action and remediation_action.get("physical_readback_gap") is True
    )
    detail["artifact_required"] = contract_summary.get("artifact_required")
    detail["physical_fixture_gap_remains"] = contract_summary.get("physical_fixture_gap_remains")
    detail["accepted_as_physical_fixture_candidate"] = contract_summary.get(
        "accepted_as_physical_fixture_candidate"
    )
    detail["live_authority_granted"] = contract_summary.get("live_authority_granted")
    detail["promotion_families_requiring_surface"] = list(promotion_families)
    detail["required_top_level_field_count"] = len(
        list_details.get("required_top_level_fields", [])
    )
    detail["required_active_scene_fields"] = list(
        list_details.get("required_active_scene_fields", [])
    )
    detail["required_snapshot_fields"] = list(
        list_details.get("required_snapshot_fields", [])
    )
    detail["required_coverage_flags"] = list(
        list_details.get("required_coverage_flags", [])
    )
    detail["required_false_safety_flags"] = list(
        list_details.get("required_false_safety_flags", [])
    )
    detail["minimum_snapshot_count"] = int(
        artifact_contract.get("minimum_snapshot_count") or 0
    )
    detail["allowed_sources"] = list(allowed_sources)
    detail["accepted_row_statuses"] = list(accepted_statuses)
    detail["forbidden_mutation_evidence_fields"] = list(forbidden_mutation_fields)
    detail["violations"] = violations
    finished = datetime.now(timezone.utc)
    return CheckResult(
        name="snapshots_scenes_physical_readback_contract_schema",
        status="passed" if not violations else "failed",
        started_at=started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        finished_at=finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration_sec=round((finished - started).total_seconds(), 3),
        detail=detail,
    )


def validate_snapshots_scenes_physical_readback_artifact_intake_snapshot() -> CheckResult:
    started = datetime.now(timezone.utc)
    detail: dict[str, object] = {
        "policy": "snapshots_scenes_artifact_intake_is_read_only_operator_review_gate",
    }
    violations: list[str] = []

    try:
        intake = build_snapshots_scenes_physical_readback_artifact_intake()
    except Exception as exc:  # pragma: no cover - defensive release-check guard
        intake = {}
        violations.append(
            f"snapshots_scenes_artifact_intake_build_error:{type(exc).__name__}"
        )

    safety = intake.get("safety") if isinstance(intake.get("safety"), Mapping) else {}
    summary = intake.get("summary") if isinstance(intake.get("summary"), Mapping) else {}
    artifacts = intake.get("artifacts") if isinstance(intake.get("artifacts"), list) else []

    if intake.get("type") != "snapshots_scenes_physical_readback_artifact_intake":
        violations.append("snapshots_scenes_artifact_intake_type_mismatch")
    if (
        intake.get("method_id")
        != SNAPSHOTS_SCENES_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID
    ):
        violations.append("snapshots_scenes_artifact_intake_method_id_mismatch")
    if (
        intake.get("required_contract_method_id")
        != SNAPSHOTS_SCENES_PHYSICAL_READBACK_CONTRACT_METHOD_ID
    ):
        violations.append("snapshots_scenes_artifact_intake_required_contract_mismatch")
    if intake.get("schema_id") != SNAPSHOTS_SCENES_PHYSICAL_READBACK_SCHEMA_ID:
        violations.append("snapshots_scenes_artifact_intake_schema_id_mismatch")
    if intake.get("status") not in SNAPSHOTS_SCENES_INTAKE_STATUSES:
        violations.append("snapshots_scenes_artifact_intake_status_unknown")
    if intake.get("surface_id") != SNAPSHOTS_SCENES_SURFACE_ID:
        violations.append("snapshots_scenes_artifact_intake_surface_id_mismatch")
    if intake.get("remediation_action_id") != SNAPSHOTS_SCENES_REMEDIATION_ACTION_ID:
        violations.append("snapshots_scenes_artifact_intake_action_id_mismatch")

    for key in ("read_only", "operator_required"):
        if safety.get(key) is not True:
            violations.append(f"snapshots_scenes_artifact_intake_{key}_not_true")
    for key in (
        "live_write",
        "mutates_console_state",
        "queries_console",
        "executes_tests",
        "execution_allowed_by_intake",
        "fixture_closure_allowed_by_intake",
        "live_widening_allowed",
        "full_autonomous_ready",
        "scene_load_allowed",
        "scene_save_allowed",
    ):
        if safety.get(key) is not False:
            violations.append(f"snapshots_scenes_artifact_intake_{key}_not_false")

    artifact_count = summary.get("artifact_count")
    valid_count = summary.get("valid_candidate_count")
    invalid_count = summary.get("invalid_candidate_count")
    valid_snapshot_count = summary.get("valid_snapshot_count")
    if not isinstance(artifact_count, int) or artifact_count != len(artifacts):
        violations.append("snapshots_scenes_artifact_intake_artifact_count_mismatch")
        artifact_count = len(artifacts)
    if not isinstance(valid_count, int):
        violations.append("snapshots_scenes_artifact_intake_valid_count_missing")
        valid_count = 0
    if not isinstance(invalid_count, int):
        violations.append("snapshots_scenes_artifact_intake_invalid_count_missing")
        invalid_count = 0
    if not isinstance(valid_snapshot_count, int):
        violations.append("snapshots_scenes_artifact_intake_valid_snapshot_count_missing")
        valid_snapshot_count = 0
    if valid_count + invalid_count != artifact_count:
        violations.append("snapshots_scenes_artifact_intake_candidate_count_mismatch")
    if invalid_count:
        violations.append("snapshots_scenes_artifact_intake_invalid_candidate_present")

    if summary.get("operator_review_required") is not True:
        violations.append("snapshots_scenes_artifact_intake_operator_review_not_true")
    if summary.get("physical_fixture_gap_remains") is not True:
        violations.append("snapshots_scenes_artifact_intake_physical_gap_not_true")
    if summary.get("fixture_closure_allowed_by_intake") is not False:
        violations.append("snapshots_scenes_artifact_intake_fixture_closure_not_false")
    if summary.get("scene_load_allowed") is not False:
        violations.append("snapshots_scenes_artifact_intake_scene_load_not_false")
    if summary.get("scene_save_allowed") is not False:
        violations.append("snapshots_scenes_artifact_intake_scene_save_not_false")
    if summary.get("live_authority_granted") is not False:
        violations.append("snapshots_scenes_artifact_intake_live_authority_not_false")
    if summary.get("full_autonomous_ready") is not False:
        violations.append("snapshots_scenes_artifact_intake_full_autonomous_not_false")
    expected_candidate_accepted = bool(valid_count) and invalid_count == 0
    if summary.get("accepted_as_physical_fixture_candidate") is not expected_candidate_accepted:
        violations.append("snapshots_scenes_artifact_intake_candidate_flag_mismatch")

    promotion_families = summary.get("promotion_families_requiring_surface")
    if not isinstance(promotion_families, list):
        promotion_families = []
        violations.append("snapshots_scenes_artifact_intake_promotion_families_missing")
    for family in ("scene_preset", "routing", "mix_balance", "performance_adaptation"):
        if family not in promotion_families:
            violations.append(
                f"snapshots_scenes_artifact_intake_promotion_family_missing:{family}"
            )

    intake_violations = (
        list(summary.get("violations")) if isinstance(summary.get("violations"), list) else []
    )
    artifact_source_ids = [
        row.get("source_id")
        for row in artifacts
        if isinstance(row, Mapping) and isinstance(row.get("source_id"), str)
    ]
    detail["intake_status"] = intake.get("status")
    detail["schema_id"] = intake.get("schema_id")
    detail["surface_id"] = intake.get("surface_id")
    detail["remediation_action_id"] = intake.get("remediation_action_id")
    detail["required_contract_method_id"] = intake.get("required_contract_method_id")
    detail["artifact_search_roots"] = (
        list(summary.get("artifact_search_roots"))
        if isinstance(summary.get("artifact_search_roots"), list)
        else []
    )
    detail["artifact_count"] = artifact_count
    detail["valid_candidate_count"] = valid_count
    detail["invalid_candidate_count"] = invalid_count
    detail["valid_snapshot_count"] = valid_snapshot_count
    detail["accepted_as_physical_fixture_candidate"] = summary.get(
        "accepted_as_physical_fixture_candidate"
    )
    detail["operator_review_required"] = summary.get("operator_review_required")
    detail["physical_fixture_gap_remains"] = summary.get("physical_fixture_gap_remains")
    detail["fixture_closure_allowed_by_intake"] = summary.get(
        "fixture_closure_allowed_by_intake"
    )
    detail["scene_load_allowed"] = summary.get("scene_load_allowed")
    detail["scene_save_allowed"] = summary.get("scene_save_allowed")
    detail["live_authority_granted"] = summary.get("live_authority_granted")
    detail["full_autonomous_ready"] = summary.get("full_autonomous_ready")
    detail["promotion_families_requiring_surface"] = list(promotion_families)
    detail["latest_valid_candidate_source_id"] = summary.get("latest_valid_candidate_source_id")
    detail["latest_valid_candidate_captured_at"] = summary.get(
        "latest_valid_candidate_captured_at"
    )
    detail["blocked_by"] = (
        list(summary.get("blocked_by")) if isinstance(summary.get("blocked_by"), list) else []
    )
    detail["artifact_source_ids"] = artifact_source_ids
    detail["intake_violations"] = intake_violations
    detail["violations"] = violations
    finished = datetime.now(timezone.utc)
    return CheckResult(
        name="snapshots_scenes_physical_readback_artifact_intake_schema",
        status="passed" if not violations else "failed",
        started_at=started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        finished_at=finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration_sec=round((finished - started).total_seconds(), 3),
        detail=detail,
    )


def validate_user_presets_physical_readback_contract_snapshot() -> CheckResult:
    started = datetime.now(timezone.utc)
    detail: dict[str, object] = {
        "policy": "user_presets_readback_contract_is_schema_only_no_live_authority",
    }
    violations: list[str] = []

    try:
        fixture_snapshot = build_read_surface_fixture_evidence_snapshot()
        remediation_plan = build_read_surface_remediation_plan(
            fixture_evidence_snapshot=fixture_snapshot
        )
        contract = build_user_presets_physical_readback_contract()
    except Exception as exc:  # pragma: no cover - defensive release-check guard
        fixture_snapshot = {}
        remediation_plan = {}
        contract = {}
        violations.append(f"user_presets_readback_contract_build_error:{type(exc).__name__}")

    fixture_summary = (
        fixture_snapshot.get("summary")
        if isinstance(fixture_snapshot.get("summary"), Mapping)
        else {}
    )
    remediation_actions = (
        remediation_plan.get("actions")
        if isinstance(remediation_plan.get("actions"), list)
        else []
    )
    safety = contract.get("safety") if isinstance(contract.get("safety"), Mapping) else {}
    contract_summary = (
        contract.get("summary") if isinstance(contract.get("summary"), Mapping) else {}
    )
    artifact_contract = (
        contract.get("artifact_contract")
        if isinstance(contract.get("artifact_contract"), Mapping)
        else {}
    )

    if contract.get("type") != "user_presets_physical_readback_contract":
        violations.append("user_presets_readback_contract_type_mismatch")
    if contract.get("method_id") != USER_PRESETS_PHYSICAL_READBACK_CONTRACT_METHOD_ID:
        violations.append("user_presets_readback_contract_method_id_mismatch")
    if contract.get("schema_id") != USER_PRESETS_PHYSICAL_READBACK_SCHEMA_ID:
        violations.append("user_presets_readback_contract_schema_id_mismatch")
    if contract.get("status") != "contract_ready_physical_fixture_missing":
        violations.append("user_presets_readback_contract_status_mismatch")
    if contract.get("surface_id") != USER_PRESETS_SURFACE_ID:
        violations.append("user_presets_readback_contract_surface_id_mismatch")
    if contract.get("remediation_action_id") != USER_PRESETS_REMEDIATION_ACTION_ID:
        violations.append("user_presets_readback_contract_action_id_mismatch")

    if safety.get("read_only") is not True:
        violations.append("user_presets_readback_contract_read_only_not_true")
    for key in (
        "live_write",
        "mutates_console_state",
        "queries_console",
        "executes_tests",
        "live_widening_allowed",
        "full_autonomous_ready",
        "preset_recall_allowed",
        "preset_store_allowed",
    ):
        if safety.get(key) is not False:
            violations.append(f"user_presets_readback_contract_{key}_not_false")

    if artifact_contract.get("type") != USER_PRESETS_PHYSICAL_READBACK_ARTIFACT_TYPE:
        violations.append("user_presets_readback_artifact_type_mismatch")
    if artifact_contract.get("schema_id") != USER_PRESETS_PHYSICAL_READBACK_SCHEMA_ID:
        violations.append("user_presets_readback_artifact_schema_id_mismatch")
    if artifact_contract.get("minimum_preset_count") != MINIMUM_USER_PRESET_COUNT:
        violations.append("user_presets_readback_artifact_minimum_preset_count_mismatch")

    expected_lists = {
        "required_top_level_fields": USER_PRESETS_REQUIRED_TOP_LEVEL_FIELDS,
        "required_preset_fields": USER_PRESETS_REQUIRED_PRESET_FIELDS,
        "required_coverage_flags": USER_PRESETS_REQUIRED_COVERAGE_FLAGS,
        "required_false_safety_flags": USER_PRESETS_REQUIRED_FALSE_SAFETY_FLAGS,
    }
    list_details: dict[str, list[object]] = {}
    for key, expected in expected_lists.items():
        value = artifact_contract.get(key)
        if not isinstance(value, list) or value != list(expected):
            violations.append(f"user_presets_readback_{key}_mismatch")
            list_details[key] = []
        else:
            list_details[key] = list(value)

    allowed_sources = artifact_contract.get("allowed_sources")
    if not isinstance(allowed_sources, list):
        allowed_sources = []
        violations.append("user_presets_readback_allowed_sources_missing")
    elif list(allowed_sources) != list(USER_PRESETS_PHYSICAL_READBACK_SOURCES):
        violations.append("user_presets_readback_allowed_sources_mismatch")
    if "mixer_state_cache" in allowed_sources or "cache" in allowed_sources:
        violations.append("user_presets_readback_allows_cached_source")

    accepted_statuses = artifact_contract.get("accepted_row_statuses")
    if (
        not isinstance(accepted_statuses, list)
        or accepted_statuses != list(USER_PRESETS_VERIFIED_PHYSICAL_STATUSES)
    ):
        violations.append("user_presets_readback_accepted_statuses_mismatch")
        accepted_statuses = []
    forbidden_mutation_fields = artifact_contract.get("forbidden_mutation_evidence_fields")
    if (
        not isinstance(forbidden_mutation_fields, list)
        or forbidden_mutation_fields
        != list(USER_PRESETS_FORBIDDEN_MUTATION_EVIDENCE_FIELDS)
    ):
        violations.append("user_presets_readback_forbidden_mutation_fields_mismatch")
        forbidden_mutation_fields = []

    physical_gap_ids = fixture_summary.get("physical_fixture_missing_surface_ids")
    if not isinstance(physical_gap_ids, list):
        physical_gap_ids = []
        violations.append("user_presets_fixture_physical_gap_ids_missing")
    if USER_PRESETS_SURFACE_ID not in physical_gap_ids:
        violations.append("user_presets_fixture_gap_not_declared")

    remediation_action = {}
    for action in remediation_actions:
        if isinstance(action, Mapping) and action.get("surface_id") == USER_PRESETS_SURFACE_ID:
            remediation_action = dict(action)
            break
    if not remediation_action:
        violations.append("user_presets_remediation_action_missing")
    elif remediation_action.get("id") != USER_PRESETS_REMEDIATION_ACTION_ID:
        violations.append("user_presets_remediation_action_id_mismatch")
    if remediation_action and remediation_action.get("physical_readback_gap") is not True:
        violations.append("user_presets_remediation_physical_gap_not_true")

    if contract_summary.get("physical_readback_surface_id") != USER_PRESETS_SURFACE_ID:
        violations.append("user_presets_contract_surface_summary_mismatch")
    if contract_summary.get("physical_readback_action_id") != USER_PRESETS_REMEDIATION_ACTION_ID:
        violations.append("user_presets_contract_action_summary_mismatch")
    if contract_summary.get("artifact_required") is not True:
        violations.append("user_presets_contract_artifact_required_not_true")
    if contract_summary.get("physical_fixture_gap_remains") is not True:
        violations.append("user_presets_contract_physical_gap_not_true")
    if contract_summary.get("accepted_as_physical_fixture_candidate") is not False:
        violations.append("user_presets_contract_candidate_not_false")
    if contract_summary.get("live_authority_granted") is not False:
        violations.append("user_presets_contract_live_authority_not_false")
    promotion_families = contract_summary.get("promotion_families_requiring_surface")
    if not isinstance(promotion_families, list):
        promotion_families = []
        violations.append("user_presets_promotion_families_missing")
    for family in ("scene_preset", "routing"):
        if family not in promotion_families:
            violations.append(f"user_presets_promotion_family_missing:{family}")

    detail["contract_status"] = contract.get("status")
    detail["schema_id"] = contract.get("schema_id")
    detail["surface_id"] = contract.get("surface_id")
    detail["remediation_action_id"] = contract.get("remediation_action_id")
    detail["physical_readback_surface_id"] = contract_summary.get("physical_readback_surface_id")
    detail["physical_readback_action_id"] = contract_summary.get("physical_readback_action_id")
    detail["fixture_physical_gap_declared"] = USER_PRESETS_SURFACE_ID in physical_gap_ids
    detail["remediation_physical_gap_declared"] = bool(
        remediation_action and remediation_action.get("physical_readback_gap") is True
    )
    detail["artifact_required"] = contract_summary.get("artifact_required")
    detail["physical_fixture_gap_remains"] = contract_summary.get("physical_fixture_gap_remains")
    detail["accepted_as_physical_fixture_candidate"] = contract_summary.get(
        "accepted_as_physical_fixture_candidate"
    )
    detail["live_authority_granted"] = contract_summary.get("live_authority_granted")
    detail["promotion_families_requiring_surface"] = list(promotion_families)
    detail["required_top_level_field_count"] = len(
        list_details.get("required_top_level_fields", [])
    )
    detail["required_preset_fields"] = list(
        list_details.get("required_preset_fields", [])
    )
    detail["required_coverage_flags"] = list(
        list_details.get("required_coverage_flags", [])
    )
    detail["required_false_safety_flags"] = list(
        list_details.get("required_false_safety_flags", [])
    )
    detail["minimum_preset_count"] = int(
        artifact_contract.get("minimum_preset_count") or 0
    )
    detail["allowed_sources"] = list(allowed_sources)
    detail["accepted_row_statuses"] = list(accepted_statuses)
    detail["forbidden_mutation_evidence_fields"] = list(forbidden_mutation_fields)
    detail["violations"] = violations
    finished = datetime.now(timezone.utc)
    return CheckResult(
        name="user_presets_physical_readback_contract_schema",
        status="passed" if not violations else "failed",
        started_at=started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        finished_at=finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration_sec=round((finished - started).total_seconds(), 3),
        detail=detail,
    )


def validate_user_presets_physical_readback_artifact_intake_snapshot() -> CheckResult:
    started = datetime.now(timezone.utc)
    detail: dict[str, object] = {
        "policy": "user_presets_artifact_intake_is_read_only_operator_review_gate",
    }
    violations: list[str] = []

    try:
        intake = build_user_presets_physical_readback_artifact_intake()
    except Exception as exc:  # pragma: no cover - defensive release-check guard
        intake = {}
        violations.append(f"user_presets_artifact_intake_build_error:{type(exc).__name__}")

    safety = intake.get("safety") if isinstance(intake.get("safety"), Mapping) else {}
    summary = intake.get("summary") if isinstance(intake.get("summary"), Mapping) else {}
    artifacts = intake.get("artifacts") if isinstance(intake.get("artifacts"), list) else []

    if intake.get("type") != "user_presets_physical_readback_artifact_intake":
        violations.append("user_presets_artifact_intake_type_mismatch")
    if intake.get("method_id") != USER_PRESETS_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID:
        violations.append("user_presets_artifact_intake_method_id_mismatch")
    if (
        intake.get("required_contract_method_id")
        != USER_PRESETS_PHYSICAL_READBACK_CONTRACT_METHOD_ID
    ):
        violations.append("user_presets_artifact_intake_required_contract_mismatch")
    if intake.get("schema_id") != USER_PRESETS_PHYSICAL_READBACK_SCHEMA_ID:
        violations.append("user_presets_artifact_intake_schema_id_mismatch")
    if intake.get("status") not in USER_PRESETS_INTAKE_STATUSES:
        violations.append("user_presets_artifact_intake_status_unknown")
    if intake.get("surface_id") != USER_PRESETS_SURFACE_ID:
        violations.append("user_presets_artifact_intake_surface_id_mismatch")
    if intake.get("remediation_action_id") != USER_PRESETS_REMEDIATION_ACTION_ID:
        violations.append("user_presets_artifact_intake_action_id_mismatch")

    for key in ("read_only", "operator_required"):
        if safety.get(key) is not True:
            violations.append(f"user_presets_artifact_intake_{key}_not_true")
    for key in (
        "live_write",
        "mutates_console_state",
        "queries_console",
        "executes_tests",
        "execution_allowed_by_intake",
        "fixture_closure_allowed_by_intake",
        "live_widening_allowed",
        "full_autonomous_ready",
        "preset_recall_allowed",
        "preset_store_allowed",
    ):
        if safety.get(key) is not False:
            violations.append(f"user_presets_artifact_intake_{key}_not_false")

    artifact_count = summary.get("artifact_count")
    valid_count = summary.get("valid_candidate_count")
    invalid_count = summary.get("invalid_candidate_count")
    valid_preset_count = summary.get("valid_preset_count")
    if not isinstance(artifact_count, int) or artifact_count != len(artifacts):
        violations.append("user_presets_artifact_intake_artifact_count_mismatch")
        artifact_count = len(artifacts)
    if not isinstance(valid_count, int):
        violations.append("user_presets_artifact_intake_valid_count_missing")
        valid_count = 0
    if not isinstance(invalid_count, int):
        violations.append("user_presets_artifact_intake_invalid_count_missing")
        invalid_count = 0
    if not isinstance(valid_preset_count, int):
        violations.append("user_presets_artifact_intake_valid_preset_count_missing")
        valid_preset_count = 0
    if valid_count + invalid_count != artifact_count:
        violations.append("user_presets_artifact_intake_candidate_count_mismatch")
    if invalid_count:
        violations.append("user_presets_artifact_intake_invalid_candidate_present")

    if summary.get("operator_review_required") is not True:
        violations.append("user_presets_artifact_intake_operator_review_not_true")
    if summary.get("physical_fixture_gap_remains") is not True:
        violations.append("user_presets_artifact_intake_physical_gap_not_true")
    if summary.get("fixture_closure_allowed_by_intake") is not False:
        violations.append("user_presets_artifact_intake_fixture_closure_not_false")
    if summary.get("preset_recall_allowed") is not False:
        violations.append("user_presets_artifact_intake_preset_recall_not_false")
    if summary.get("preset_store_allowed") is not False:
        violations.append("user_presets_artifact_intake_preset_store_not_false")
    if summary.get("live_authority_granted") is not False:
        violations.append("user_presets_artifact_intake_live_authority_not_false")
    if summary.get("full_autonomous_ready") is not False:
        violations.append("user_presets_artifact_intake_full_autonomous_not_false")
    expected_candidate_accepted = bool(valid_count) and invalid_count == 0
    if summary.get("accepted_as_physical_fixture_candidate") is not expected_candidate_accepted:
        violations.append("user_presets_artifact_intake_candidate_flag_mismatch")

    promotion_families = summary.get("promotion_families_requiring_surface")
    if not isinstance(promotion_families, list):
        promotion_families = []
        violations.append("user_presets_artifact_intake_promotion_families_missing")
    for family in ("scene_preset", "routing", "mix_balance", "performance_adaptation"):
        if family not in promotion_families:
            violations.append(f"user_presets_artifact_intake_promotion_family_missing:{family}")

    intake_violations = (
        list(summary.get("violations")) if isinstance(summary.get("violations"), list) else []
    )
    artifact_source_ids = [
        row.get("source_id")
        for row in artifacts
        if isinstance(row, Mapping) and isinstance(row.get("source_id"), str)
    ]
    detail["intake_status"] = intake.get("status")
    detail["schema_id"] = intake.get("schema_id")
    detail["surface_id"] = intake.get("surface_id")
    detail["remediation_action_id"] = intake.get("remediation_action_id")
    detail["required_contract_method_id"] = intake.get("required_contract_method_id")
    detail["artifact_search_roots"] = (
        list(summary.get("artifact_search_roots"))
        if isinstance(summary.get("artifact_search_roots"), list)
        else []
    )
    detail["artifact_count"] = artifact_count
    detail["valid_candidate_count"] = valid_count
    detail["invalid_candidate_count"] = invalid_count
    detail["valid_preset_count"] = valid_preset_count
    detail["accepted_as_physical_fixture_candidate"] = summary.get(
        "accepted_as_physical_fixture_candidate"
    )
    detail["operator_review_required"] = summary.get("operator_review_required")
    detail["physical_fixture_gap_remains"] = summary.get("physical_fixture_gap_remains")
    detail["fixture_closure_allowed_by_intake"] = summary.get(
        "fixture_closure_allowed_by_intake"
    )
    detail["preset_recall_allowed"] = summary.get("preset_recall_allowed")
    detail["preset_store_allowed"] = summary.get("preset_store_allowed")
    detail["live_authority_granted"] = summary.get("live_authority_granted")
    detail["full_autonomous_ready"] = summary.get("full_autonomous_ready")
    detail["promotion_families_requiring_surface"] = list(promotion_families)
    detail["latest_valid_candidate_source_id"] = summary.get("latest_valid_candidate_source_id")
    detail["latest_valid_candidate_captured_at"] = summary.get(
        "latest_valid_candidate_captured_at"
    )
    detail["blocked_by"] = (
        list(summary.get("blocked_by")) if isinstance(summary.get("blocked_by"), list) else []
    )
    detail["artifact_source_ids"] = artifact_source_ids
    detail["intake_violations"] = intake_violations
    detail["violations"] = violations
    finished = datetime.now(timezone.utc)
    return CheckResult(
        name="user_presets_physical_readback_artifact_intake_schema",
        status="passed" if not violations else "failed",
        started_at=started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        finished_at=finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration_sec=round((finished - started).total_seconds(), 3),
        detail=detail,
    )


def validate_dlive_state_physical_readback_contract_snapshot() -> CheckResult:
    started = datetime.now(timezone.utc)
    detail: dict[str, object] = {
        "policy": "dlive_state_readback_contract_is_schema_only_no_midi_tcp_or_dlive_write_authority",
    }
    violations: list[str] = []

    try:
        fixture_snapshot = build_read_surface_fixture_evidence_snapshot()
        remediation_plan = build_read_surface_remediation_plan(
            fixture_evidence_snapshot=fixture_snapshot
        )
        contract = build_dlive_state_physical_readback_contract()
    except Exception as exc:  # pragma: no cover - defensive release-check guard
        fixture_snapshot = {}
        remediation_plan = {}
        contract = {}
        violations.append(f"dlive_state_readback_contract_build_error:{type(exc).__name__}")

    fixture_summary = (
        fixture_snapshot.get("summary")
        if isinstance(fixture_snapshot.get("summary"), Mapping)
        else {}
    )
    remediation_actions = (
        remediation_plan.get("actions")
        if isinstance(remediation_plan.get("actions"), list)
        else []
    )
    safety = contract.get("safety") if isinstance(contract.get("safety"), Mapping) else {}
    contract_summary = (
        contract.get("summary") if isinstance(contract.get("summary"), Mapping) else {}
    )
    artifact_contract = (
        contract.get("artifact_contract")
        if isinstance(contract.get("artifact_contract"), Mapping)
        else {}
    )

    if contract.get("type") != "dlive_state_physical_readback_contract":
        violations.append("dlive_state_readback_contract_type_mismatch")
    if contract.get("method_id") != DLIVE_STATE_PHYSICAL_READBACK_CONTRACT_METHOD_ID:
        violations.append("dlive_state_readback_contract_method_id_mismatch")
    if contract.get("schema_id") != DLIVE_STATE_PHYSICAL_READBACK_SCHEMA_ID:
        violations.append("dlive_state_readback_contract_schema_id_mismatch")
    if contract.get("status") != "contract_ready_physical_fixture_missing":
        violations.append("dlive_state_readback_contract_status_mismatch")
    if contract.get("surface_id") != DLIVE_STATE_SURFACE_ID:
        violations.append("dlive_state_readback_contract_surface_id_mismatch")
    if contract.get("remediation_action_id") != DLIVE_STATE_REMEDIATION_ACTION_ID:
        violations.append("dlive_state_readback_contract_action_id_mismatch")

    if safety.get("read_only") is not True:
        violations.append("dlive_state_readback_contract_read_only_not_true")
    for key in (
        "live_write",
        "mutates_console_state",
        "queries_console",
        "executes_tests",
        "live_widening_allowed",
        "full_autonomous_ready",
        "midi_send_allowed",
        "tcp_send_allowed",
        "dlive_write_allowed",
        "dlive_apply_allowed",
    ):
        if safety.get(key) is not False:
            violations.append(f"dlive_state_readback_contract_{key}_not_false")

    if artifact_contract.get("type") != DLIVE_STATE_PHYSICAL_READBACK_ARTIFACT_TYPE:
        violations.append("dlive_state_readback_artifact_type_mismatch")
    if artifact_contract.get("schema_id") != DLIVE_STATE_PHYSICAL_READBACK_SCHEMA_ID:
        violations.append("dlive_state_readback_artifact_schema_id_mismatch")
    if artifact_contract.get("minimum_channel_count") != MINIMUM_DLIVE_STATE_CHANNEL_COUNT:
        violations.append("dlive_state_readback_artifact_minimum_channel_count_mismatch")

    expected_lists = {
        "required_top_level_fields": DLIVE_STATE_REQUIRED_TOP_LEVEL_FIELDS,
        "required_channel_fields": DLIVE_STATE_REQUIRED_CHANNEL_FIELDS,
        "required_midi_state_fields": DLIVE_STATE_REQUIRED_MIDI_STATE_FIELDS,
        "required_parity_fields": DLIVE_STATE_REQUIRED_PARITY_FIELDS,
        "required_coverage_flags": DLIVE_STATE_REQUIRED_COVERAGE_FLAGS,
        "required_false_safety_flags": DLIVE_STATE_REQUIRED_FALSE_SAFETY_FLAGS,
    }
    list_details: dict[str, list[object]] = {}
    for key, expected in expected_lists.items():
        value = artifact_contract.get(key)
        if not isinstance(value, list) or value != list(expected):
            violations.append(f"dlive_state_readback_{key}_mismatch")
            list_details[key] = []
        else:
            list_details[key] = list(value)

    allowed_sources = artifact_contract.get("allowed_sources")
    if not isinstance(allowed_sources, list):
        allowed_sources = []
        violations.append("dlive_state_readback_allowed_sources_missing")
    elif list(allowed_sources) != list(DLIVE_STATE_PHYSICAL_READBACK_SOURCES):
        violations.append("dlive_state_readback_allowed_sources_mismatch")
    if "mixer_state_cache" in allowed_sources or "cache" in allowed_sources:
        violations.append("dlive_state_readback_allows_cached_source")

    accepted_statuses = artifact_contract.get("accepted_row_statuses")
    if (
        not isinstance(accepted_statuses, list)
        or accepted_statuses != list(DLIVE_STATE_VERIFIED_PHYSICAL_STATUSES)
    ):
        violations.append("dlive_state_readback_accepted_statuses_mismatch")
        accepted_statuses = []
    forbidden_mutation_fields = artifact_contract.get("forbidden_mutation_evidence_fields")
    if (
        not isinstance(forbidden_mutation_fields, list)
        or forbidden_mutation_fields
        != list(DLIVE_STATE_FORBIDDEN_MUTATION_EVIDENCE_FIELDS)
    ):
        violations.append("dlive_state_readback_forbidden_mutation_fields_mismatch")
        forbidden_mutation_fields = []

    physical_gap_ids = fixture_summary.get("physical_fixture_missing_surface_ids")
    if not isinstance(physical_gap_ids, list):
        physical_gap_ids = []
        violations.append("dlive_state_fixture_physical_gap_ids_missing")
    if DLIVE_STATE_SURFACE_ID not in physical_gap_ids:
        violations.append("dlive_state_fixture_gap_not_declared")

    remediation_action = {}
    for action in remediation_actions:
        if isinstance(action, Mapping) and action.get("surface_id") == DLIVE_STATE_SURFACE_ID:
            remediation_action = dict(action)
            break
    if not remediation_action:
        violations.append("dlive_state_remediation_action_missing")
    elif remediation_action.get("id") != DLIVE_STATE_REMEDIATION_ACTION_ID:
        violations.append("dlive_state_remediation_action_id_mismatch")
    if remediation_action and remediation_action.get("physical_readback_gap") is not True:
        violations.append("dlive_state_remediation_physical_gap_not_true")

    if contract_summary.get("physical_readback_surface_id") != DLIVE_STATE_SURFACE_ID:
        violations.append("dlive_state_contract_surface_summary_mismatch")
    if contract_summary.get("physical_readback_action_id") != DLIVE_STATE_REMEDIATION_ACTION_ID:
        violations.append("dlive_state_contract_action_summary_mismatch")
    if contract_summary.get("artifact_required") is not True:
        violations.append("dlive_state_contract_artifact_required_not_true")
    if contract_summary.get("physical_fixture_gap_remains") is not True:
        violations.append("dlive_state_contract_physical_gap_not_true")
    if contract_summary.get("accepted_as_physical_fixture_candidate") is not False:
        violations.append("dlive_state_contract_candidate_not_false")
    if contract_summary.get("live_authority_granted") is not False:
        violations.append("dlive_state_contract_live_authority_not_false")
    promotion_families = contract_summary.get("promotion_families_requiring_surface")
    if not isinstance(promotion_families, list):
        promotion_families = []
        violations.append("dlive_state_promotion_families_missing")
    for family in ("fader", "gain", "mute", "routing", "performance_adaptation"):
        if family not in promotion_families:
            violations.append(f"dlive_state_promotion_family_missing:{family}")

    detail["contract_status"] = contract.get("status")
    detail["schema_id"] = contract.get("schema_id")
    detail["surface_id"] = contract.get("surface_id")
    detail["remediation_action_id"] = contract.get("remediation_action_id")
    detail["physical_readback_surface_id"] = contract_summary.get("physical_readback_surface_id")
    detail["physical_readback_action_id"] = contract_summary.get("physical_readback_action_id")
    detail["fixture_physical_gap_declared"] = DLIVE_STATE_SURFACE_ID in physical_gap_ids
    detail["remediation_physical_gap_declared"] = bool(
        remediation_action and remediation_action.get("physical_readback_gap") is True
    )
    detail["artifact_required"] = contract_summary.get("artifact_required")
    detail["physical_fixture_gap_remains"] = contract_summary.get("physical_fixture_gap_remains")
    detail["accepted_as_physical_fixture_candidate"] = contract_summary.get(
        "accepted_as_physical_fixture_candidate"
    )
    detail["live_authority_granted"] = contract_summary.get("live_authority_granted")
    detail["promotion_families_requiring_surface"] = list(promotion_families)
    detail["required_top_level_field_count"] = len(
        list_details.get("required_top_level_fields", [])
    )
    detail["required_channel_fields"] = list(
        list_details.get("required_channel_fields", [])
    )
    detail["required_midi_state_fields"] = list(
        list_details.get("required_midi_state_fields", [])
    )
    detail["required_parity_fields"] = list(
        list_details.get("required_parity_fields", [])
    )
    detail["required_coverage_flags"] = list(
        list_details.get("required_coverage_flags", [])
    )
    detail["required_false_safety_flags"] = list(
        list_details.get("required_false_safety_flags", [])
    )
    detail["minimum_channel_count"] = int(
        artifact_contract.get("minimum_channel_count") or 0
    )
    detail["allowed_sources"] = list(allowed_sources)
    detail["accepted_row_statuses"] = list(accepted_statuses)
    detail["forbidden_mutation_evidence_fields"] = list(forbidden_mutation_fields)
    detail["violations"] = violations
    finished = datetime.now(timezone.utc)
    return CheckResult(
        name="dlive_state_physical_readback_contract_schema",
        status="passed" if not violations else "failed",
        started_at=started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        finished_at=finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration_sec=round((finished - started).total_seconds(), 3),
        detail=detail,
    )


def validate_dlive_state_physical_readback_artifact_intake_snapshot() -> CheckResult:
    started = datetime.now(timezone.utc)
    detail: dict[str, object] = {
        "policy": "dlive_state_artifact_intake_is_read_only_operator_review_gate",
    }
    violations: list[str] = []

    try:
        intake = build_dlive_state_physical_readback_artifact_intake()
    except Exception as exc:  # pragma: no cover - defensive release-check guard
        intake = {}
        violations.append(f"dlive_state_artifact_intake_build_error:{type(exc).__name__}")

    safety = intake.get("safety") if isinstance(intake.get("safety"), Mapping) else {}
    summary = intake.get("summary") if isinstance(intake.get("summary"), Mapping) else {}
    artifacts = intake.get("artifacts") if isinstance(intake.get("artifacts"), list) else []

    if intake.get("type") != "dlive_state_physical_readback_artifact_intake":
        violations.append("dlive_state_artifact_intake_type_mismatch")
    if intake.get("method_id") != DLIVE_STATE_PHYSICAL_READBACK_ARTIFACT_INTAKE_METHOD_ID:
        violations.append("dlive_state_artifact_intake_method_id_mismatch")
    if (
        intake.get("required_contract_method_id")
        != DLIVE_STATE_PHYSICAL_READBACK_CONTRACT_METHOD_ID
    ):
        violations.append("dlive_state_artifact_intake_required_contract_mismatch")
    if intake.get("schema_id") != DLIVE_STATE_PHYSICAL_READBACK_SCHEMA_ID:
        violations.append("dlive_state_artifact_intake_schema_id_mismatch")
    if intake.get("status") not in DLIVE_STATE_INTAKE_STATUSES:
        violations.append("dlive_state_artifact_intake_status_unknown")
    if intake.get("surface_id") != DLIVE_STATE_SURFACE_ID:
        violations.append("dlive_state_artifact_intake_surface_id_mismatch")
    if intake.get("remediation_action_id") != DLIVE_STATE_REMEDIATION_ACTION_ID:
        violations.append("dlive_state_artifact_intake_action_id_mismatch")

    for key in ("read_only", "operator_required"):
        if safety.get(key) is not True:
            violations.append(f"dlive_state_artifact_intake_{key}_not_true")
    for key in (
        "live_write",
        "mutates_console_state",
        "queries_console",
        "executes_tests",
        "execution_allowed_by_intake",
        "fixture_closure_allowed_by_intake",
        "live_widening_allowed",
        "full_autonomous_ready",
        "midi_send_allowed",
        "tcp_send_allowed",
        "dlive_write_allowed",
        "dlive_apply_allowed",
    ):
        if safety.get(key) is not False:
            violations.append(f"dlive_state_artifact_intake_{key}_not_false")

    artifact_count = summary.get("artifact_count")
    valid_count = summary.get("valid_candidate_count")
    invalid_count = summary.get("invalid_candidate_count")
    valid_channel_count = summary.get("valid_channel_count")
    if not isinstance(artifact_count, int) or artifact_count != len(artifacts):
        violations.append("dlive_state_artifact_intake_artifact_count_mismatch")
        artifact_count = len(artifacts)
    if not isinstance(valid_count, int):
        violations.append("dlive_state_artifact_intake_valid_count_missing")
        valid_count = 0
    if not isinstance(invalid_count, int):
        violations.append("dlive_state_artifact_intake_invalid_count_missing")
        invalid_count = 0
    if not isinstance(valid_channel_count, int):
        violations.append("dlive_state_artifact_intake_valid_channel_count_missing")
        valid_channel_count = 0
    if valid_count + invalid_count != artifact_count:
        violations.append("dlive_state_artifact_intake_candidate_count_mismatch")
    if invalid_count:
        violations.append("dlive_state_artifact_intake_invalid_candidate_present")

    if summary.get("operator_review_required") is not True:
        violations.append("dlive_state_artifact_intake_operator_review_not_true")
    if summary.get("physical_fixture_gap_remains") is not True:
        violations.append("dlive_state_artifact_intake_physical_gap_not_true")
    if summary.get("fixture_closure_allowed_by_intake") is not False:
        violations.append("dlive_state_artifact_intake_fixture_closure_not_false")
    if summary.get("midi_send_allowed") is not False:
        violations.append("dlive_state_artifact_intake_midi_send_not_false")
    if summary.get("tcp_send_allowed") is not False:
        violations.append("dlive_state_artifact_intake_tcp_send_not_false")
    if summary.get("dlive_write_allowed") is not False:
        violations.append("dlive_state_artifact_intake_dlive_write_not_false")
    if summary.get("dlive_apply_allowed") is not False:
        violations.append("dlive_state_artifact_intake_dlive_apply_not_false")
    if summary.get("live_authority_granted") is not False:
        violations.append("dlive_state_artifact_intake_live_authority_not_false")
    if summary.get("full_autonomous_ready") is not False:
        violations.append("dlive_state_artifact_intake_full_autonomous_not_false")
    expected_candidate_accepted = bool(valid_count) and invalid_count == 0
    if summary.get("accepted_as_physical_fixture_candidate") is not expected_candidate_accepted:
        violations.append("dlive_state_artifact_intake_candidate_flag_mismatch")

    promotion_families = summary.get("promotion_families_requiring_surface")
    if not isinstance(promotion_families, list):
        promotion_families = []
        violations.append("dlive_state_artifact_intake_promotion_families_missing")
    for family in ("fader", "gain", "mute", "routing", "performance_adaptation"):
        if family not in promotion_families:
            violations.append(f"dlive_state_artifact_intake_promotion_family_missing:{family}")

    intake_violations = (
        list(summary.get("violations")) if isinstance(summary.get("violations"), list) else []
    )
    artifact_source_ids = [
        row.get("source_id")
        for row in artifacts
        if isinstance(row, Mapping) and isinstance(row.get("source_id"), str)
    ]
    detail["intake_status"] = intake.get("status")
    detail["schema_id"] = intake.get("schema_id")
    detail["surface_id"] = intake.get("surface_id")
    detail["remediation_action_id"] = intake.get("remediation_action_id")
    detail["required_contract_method_id"] = intake.get("required_contract_method_id")
    detail["artifact_search_roots"] = (
        list(summary.get("artifact_search_roots"))
        if isinstance(summary.get("artifact_search_roots"), list)
        else []
    )
    detail["artifact_count"] = artifact_count
    detail["valid_candidate_count"] = valid_count
    detail["invalid_candidate_count"] = invalid_count
    detail["valid_channel_count"] = valid_channel_count
    detail["accepted_as_physical_fixture_candidate"] = summary.get(
        "accepted_as_physical_fixture_candidate"
    )
    detail["operator_review_required"] = summary.get("operator_review_required")
    detail["physical_fixture_gap_remains"] = summary.get("physical_fixture_gap_remains")
    detail["fixture_closure_allowed_by_intake"] = summary.get(
        "fixture_closure_allowed_by_intake"
    )
    detail["midi_send_allowed"] = summary.get("midi_send_allowed")
    detail["tcp_send_allowed"] = summary.get("tcp_send_allowed")
    detail["dlive_write_allowed"] = summary.get("dlive_write_allowed")
    detail["dlive_apply_allowed"] = summary.get("dlive_apply_allowed")
    detail["live_authority_granted"] = summary.get("live_authority_granted")
    detail["full_autonomous_ready"] = summary.get("full_autonomous_ready")
    detail["promotion_families_requiring_surface"] = list(promotion_families)
    detail["latest_valid_candidate_source_id"] = summary.get("latest_valid_candidate_source_id")
    detail["latest_valid_candidate_captured_at"] = summary.get(
        "latest_valid_candidate_captured_at"
    )
    detail["blocked_by"] = (
        list(summary.get("blocked_by")) if isinstance(summary.get("blocked_by"), list) else []
    )
    detail["artifact_source_ids"] = artifact_source_ids
    detail["intake_violations"] = intake_violations
    detail["violations"] = violations
    finished = datetime.now(timezone.utc)
    return CheckResult(
        name="dlive_state_physical_readback_artifact_intake_schema",
        status="passed" if not violations else "failed",
        started_at=started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        finished_at=finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration_sec=round((finished - started).total_seconds(), 3),
        detail=detail,
    )


def validate_read_surface_physical_readback_artifact_review_queue_snapshot() -> CheckResult:
    started = datetime.now(timezone.utc)
    detail: dict[str, object] = {
        "policy": "physical_readback_artifact_review_queue_is_read_only_operator_gate",
    }
    violations: list[str] = []

    try:
        queue = build_read_surface_physical_readback_artifact_review_queue()
    except Exception as exc:  # pragma: no cover - defensive release-check guard
        queue = {}
        violations.append(f"artifact_review_queue_build_error:{type(exc).__name__}")

    safety = queue.get("safety") if isinstance(queue.get("safety"), Mapping) else {}
    summary = queue.get("summary") if isinstance(queue.get("summary"), Mapping) else {}
    queue_items = queue.get("queue_items") if isinstance(queue.get("queue_items"), list) else []

    if queue.get("type") != "read_surface_physical_readback_artifact_review_queue":
        violations.append("artifact_review_queue_type_mismatch")
    if queue.get("method_id") != READ_SURFACE_PHYSICAL_READBACK_ARTIFACT_REVIEW_QUEUE_METHOD_ID:
        violations.append("artifact_review_queue_method_id_mismatch")
    if queue.get("status") not in READ_SURFACE_PHYSICAL_READBACK_ARTIFACT_REVIEW_QUEUE_STATUSES:
        violations.append("artifact_review_queue_status_unknown")

    for key in ("read_only", "operator_required"):
        if safety.get(key) is not True:
            violations.append(f"artifact_review_queue_{key}_not_true")
    for key in (
        "live_write",
        "mutates_console_state",
        "queries_console",
        "executes_tests",
        "execution_allowed_by_queue",
        "fixture_closure_allowed_by_queue",
        "live_widening_allowed",
        "live_authority_granted",
        "full_autonomous_ready",
    ):
        if safety.get(key) is not False:
            violations.append(f"artifact_review_queue_{key}_not_false")

    intake_surface_count = summary.get("intake_surface_count")
    available_intake_count = summary.get("available_intake_count")
    artifact_count = summary.get("artifact_count")
    valid_candidate_count = summary.get("valid_candidate_count")
    invalid_candidate_count = summary.get("invalid_candidate_count")
    review_candidate_count = summary.get("review_candidate_count")
    unsafe_claim_count = summary.get("unsafe_authority_claim_count")
    candidate_source_missing_count = summary.get("candidate_source_id_missing_count")
    target_inconsistent_count = summary.get("target_inconsistent_surface_count")
    missing_surface_count = summary.get("missing_artifact_surface_count")
    if not isinstance(intake_surface_count, int) or intake_surface_count != len(queue_items):
        violations.append("artifact_review_queue_intake_surface_count_mismatch")
        intake_surface_count = len(queue_items)
    if not isinstance(available_intake_count, int):
        violations.append("artifact_review_queue_available_intake_count_missing")
        available_intake_count = 0
    if not isinstance(artifact_count, int):
        violations.append("artifact_review_queue_artifact_count_missing")
        artifact_count = 0
    if not isinstance(valid_candidate_count, int):
        violations.append("artifact_review_queue_valid_count_missing")
        valid_candidate_count = 0
    if not isinstance(invalid_candidate_count, int):
        violations.append("artifact_review_queue_invalid_count_missing")
        invalid_candidate_count = 0
    if not isinstance(review_candidate_count, int):
        violations.append("artifact_review_queue_review_candidate_count_missing")
        review_candidate_count = 0
    if not isinstance(unsafe_claim_count, int):
        violations.append("artifact_review_queue_unsafe_claim_count_missing")
        unsafe_claim_count = 0
    if not isinstance(candidate_source_missing_count, int):
        violations.append("artifact_review_queue_candidate_source_missing_count_missing")
        candidate_source_missing_count = 0
    if not isinstance(target_inconsistent_count, int):
        violations.append("artifact_review_queue_target_inconsistent_count_missing")
        target_inconsistent_count = 0
    if not isinstance(missing_surface_count, int):
        violations.append("artifact_review_queue_missing_surface_count_missing")
        missing_surface_count = 0
    if invalid_candidate_count:
        violations.append("artifact_review_queue_invalid_candidate_present")
    if unsafe_claim_count:
        violations.append("artifact_review_queue_unsafe_authority_claim_present")
    if candidate_source_missing_count:
        violations.append("artifact_review_queue_candidate_source_id_missing_present")
    if target_inconsistent_count:
        violations.append("artifact_review_queue_target_inconsistent_surface_present")

    if summary.get("operator_review_required") is not True:
        violations.append("artifact_review_queue_operator_review_not_true")
    if summary.get("physical_fixture_gap_remains") is not True:
        violations.append("artifact_review_queue_physical_gap_not_true")
    if summary.get("fixture_closure_allowed_by_queue") is not False:
        violations.append("artifact_review_queue_fixture_closure_not_false")
    if summary.get("live_authority_granted") is not False:
        violations.append("artifact_review_queue_live_authority_not_false")
    if summary.get("full_autonomous_ready") is not False:
        violations.append("artifact_review_queue_full_autonomous_not_false")

    review_candidate_surface_ids = summary.get("review_candidate_surface_ids")
    invalid_surface_ids = summary.get("invalid_surface_ids")
    missing_surface_ids = summary.get("missing_artifact_surface_ids")
    unsafe_surface_ids = summary.get("unsafe_authority_surface_ids")
    candidate_source_missing_surface_ids = summary.get(
        "candidate_source_id_missing_surface_ids"
    )
    target_inconsistent_surface_ids = summary.get("target_inconsistent_surface_ids")
    blocked_by = summary.get("blocked_by")
    promotion_families = summary.get("promotion_families_requiring_surface")
    detail["queue_status"] = queue.get("status")
    detail["intake_surface_count"] = intake_surface_count
    detail["available_intake_count"] = available_intake_count
    detail["artifact_count"] = artifact_count
    detail["valid_candidate_count"] = valid_candidate_count
    detail["invalid_candidate_count"] = invalid_candidate_count
    detail["review_candidate_count"] = review_candidate_count
    detail["missing_artifact_surface_count"] = missing_surface_count
    detail["unsafe_authority_claim_count"] = unsafe_claim_count
    detail["candidate_source_id_missing_count"] = candidate_source_missing_count
    detail["target_inconsistent_surface_count"] = target_inconsistent_count
    detail["review_candidate_surface_ids"] = (
        list(review_candidate_surface_ids)
        if isinstance(review_candidate_surface_ids, list)
        else []
    )
    detail["invalid_surface_ids"] = (
        list(invalid_surface_ids) if isinstance(invalid_surface_ids, list) else []
    )
    detail["missing_artifact_surface_ids"] = (
        list(missing_surface_ids) if isinstance(missing_surface_ids, list) else []
    )
    detail["unsafe_authority_surface_ids"] = (
        list(unsafe_surface_ids) if isinstance(unsafe_surface_ids, list) else []
    )
    detail["candidate_source_id_missing_surface_ids"] = (
        list(candidate_source_missing_surface_ids)
        if isinstance(candidate_source_missing_surface_ids, list)
        else []
    )
    detail["target_inconsistent_surface_ids"] = (
        list(target_inconsistent_surface_ids)
        if isinstance(target_inconsistent_surface_ids, list)
        else []
    )
    detail["operator_review_required"] = summary.get("operator_review_required")
    detail["physical_fixture_gap_remains"] = summary.get("physical_fixture_gap_remains")
    detail["fixture_closure_allowed_by_queue"] = summary.get(
        "fixture_closure_allowed_by_queue"
    )
    detail["live_authority_granted"] = summary.get("live_authority_granted")
    detail["full_autonomous_ready"] = summary.get("full_autonomous_ready")
    detail["promotion_families_requiring_surface"] = (
        list(promotion_families) if isinstance(promotion_families, list) else []
    )
    detail["blocked_by"] = list(blocked_by) if isinstance(blocked_by, list) else []
    detail["queue_item_method_ids"] = [
        item.get("method_id")
        for item in queue_items
        if isinstance(item, Mapping) and isinstance(item.get("method_id"), str)
    ]
    detail["violations"] = violations
    finished = datetime.now(timezone.utc)
    return CheckResult(
        name="read_surface_physical_readback_artifact_review_queue_schema",
        status="passed" if not violations else "failed",
        started_at=started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        finished_at=finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration_sec=round((finished - started).total_seconds(), 3),
        detail=detail,
    )


def validate_read_surface_physical_readback_fixture_acceptance_ledger_snapshot() -> CheckResult:
    started = datetime.now(timezone.utc)
    detail: dict[str, object] = {
        "policy": "physical_readback_fixture_acceptance_ledger_is_read_only_operator_gate",
    }
    violations: list[str] = []

    try:
        ledger = build_read_surface_physical_readback_fixture_acceptance_ledger()
    except Exception as exc:  # pragma: no cover - defensive release-check guard
        ledger = {}
        violations.append(f"fixture_acceptance_ledger_build_error:{type(exc).__name__}")

    safety = ledger.get("safety") if isinstance(ledger.get("safety"), Mapping) else {}
    summary = ledger.get("summary") if isinstance(ledger.get("summary"), Mapping) else {}
    records = ledger.get("records") if isinstance(ledger.get("records"), list) else []
    acceptance_template_tool_available = (
        REPO_ROOT / READ_SURFACE_PHYSICAL_READBACK_ACCEPTANCE_TEMPLATE_TOOL
    ).is_file()

    if ledger.get("type") != "read_surface_physical_readback_fixture_acceptance_ledger":
        violations.append("fixture_acceptance_ledger_type_mismatch")
    if (
        ledger.get("method_id")
        != READ_SURFACE_PHYSICAL_READBACK_FIXTURE_ACCEPTANCE_LEDGER_METHOD_ID
    ):
        violations.append("fixture_acceptance_ledger_method_id_mismatch")
    if ledger.get("status") not in READ_SURFACE_PHYSICAL_READBACK_FIXTURE_ACCEPTANCE_LEDGER_STATUSES:
        violations.append("fixture_acceptance_ledger_status_unknown")
    if not acceptance_template_tool_available:
        violations.append("fixture_acceptance_template_tool_missing")

    for key in ("read_only", "operator_required"):
        if safety.get(key) is not True:
            violations.append(f"fixture_acceptance_ledger_{key}_not_true")
    for key in (
        "live_write",
        "mutates_console_state",
        "queries_console",
        "executes_tests",
        "execution_allowed_by_acceptance",
        "fixture_closure_allowed_by_acceptance",
        "live_widening_allowed",
        "live_authority_granted",
        "full_autonomous_ready",
    ):
        if safety.get(key) is not False:
            violations.append(f"fixture_acceptance_ledger_{key}_not_false")

    review_candidate_count = summary.get("review_candidate_count")
    acceptance_record_count = summary.get("acceptance_record_count")
    accepted_record_count = summary.get("accepted_record_count")
    rejected_record_count = summary.get("rejected_record_count")
    invalid_record_count = summary.get("invalid_record_count")
    candidate_trace_missing_count = summary.get("candidate_trace_missing_count")
    candidate_trace_mismatch_count = summary.get("candidate_trace_mismatch_count")
    unreviewed_candidate_count = summary.get("unreviewed_candidate_count")
    if not isinstance(review_candidate_count, int):
        violations.append("fixture_acceptance_ledger_review_candidate_count_missing")
        review_candidate_count = 0
    if not isinstance(acceptance_record_count, int) or acceptance_record_count != len(records):
        violations.append("fixture_acceptance_ledger_record_count_mismatch")
        acceptance_record_count = len(records)
    if not isinstance(accepted_record_count, int):
        violations.append("fixture_acceptance_ledger_accepted_count_missing")
        accepted_record_count = 0
    if not isinstance(rejected_record_count, int):
        violations.append("fixture_acceptance_ledger_rejected_count_missing")
        rejected_record_count = 0
    if not isinstance(invalid_record_count, int):
        violations.append("fixture_acceptance_ledger_invalid_count_missing")
        invalid_record_count = 0
    if not isinstance(candidate_trace_missing_count, int):
        violations.append("fixture_acceptance_ledger_candidate_trace_missing_count_missing")
        candidate_trace_missing_count = 0
    if not isinstance(candidate_trace_mismatch_count, int):
        violations.append("fixture_acceptance_ledger_candidate_trace_mismatch_count_missing")
        candidate_trace_mismatch_count = 0
    if not isinstance(unreviewed_candidate_count, int):
        violations.append("fixture_acceptance_ledger_unreviewed_count_missing")
        unreviewed_candidate_count = 0
    if invalid_record_count:
        violations.append("fixture_acceptance_ledger_invalid_record_present")
    if candidate_trace_missing_count:
        violations.append("fixture_acceptance_ledger_candidate_trace_missing_present")
    if candidate_trace_mismatch_count:
        violations.append("fixture_acceptance_ledger_candidate_trace_mismatch_present")

    if summary.get("operator_review_required") is not True:
        violations.append("fixture_acceptance_ledger_operator_review_not_true")
    if summary.get("physical_fixture_gap_remains") is not True:
        violations.append("fixture_acceptance_ledger_physical_gap_not_true")
    if summary.get("fixture_closure_allowed_by_acceptance") is not False:
        violations.append("fixture_acceptance_ledger_fixture_closure_not_false")
    if summary.get("live_authority_granted") is not False:
        violations.append("fixture_acceptance_ledger_live_authority_not_false")
    if summary.get("full_autonomous_ready") is not False:
        violations.append("fixture_acceptance_ledger_full_autonomous_not_false")

    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            violations.append(f"fixture_acceptance_ledger_record_{index}_not_object")
            continue
        if record.get("fixture_closure_allowed_by_acceptance") is not False:
            violations.append(f"fixture_acceptance_ledger_record_{index}_closure_not_false")
        if record.get("live_authority_granted") is not False:
            violations.append(f"fixture_acceptance_ledger_record_{index}_live_authority_not_false")
        if record.get("full_autonomous_ready") is not False:
            violations.append(f"fixture_acceptance_ledger_record_{index}_full_auto_not_false")

    accepted_surface_ids = summary.get("accepted_surface_ids")
    rejected_surface_ids = summary.get("rejected_surface_ids")
    invalid_record_ids = summary.get("invalid_record_ids")
    unreviewed_surface_ids = summary.get("unreviewed_candidate_surface_ids")
    blocked_by = summary.get("blocked_by")
    detail["ledger_status"] = ledger.get("status")
    detail["acceptance_template_tool_path"] = (
        READ_SURFACE_PHYSICAL_READBACK_ACCEPTANCE_TEMPLATE_TOOL
    )
    detail["acceptance_template_tool_available"] = acceptance_template_tool_available
    detail["queue_status"] = summary.get("queue_status")
    detail["review_candidate_count"] = review_candidate_count
    detail["acceptance_record_count"] = acceptance_record_count
    detail["accepted_record_count"] = accepted_record_count
    detail["rejected_record_count"] = rejected_record_count
    detail["invalid_record_count"] = invalid_record_count
    detail["candidate_trace_missing_count"] = candidate_trace_missing_count
    detail["candidate_trace_mismatch_count"] = candidate_trace_mismatch_count
    detail["unreviewed_candidate_count"] = unreviewed_candidate_count
    detail["accepted_surface_ids"] = (
        list(accepted_surface_ids) if isinstance(accepted_surface_ids, list) else []
    )
    detail["rejected_surface_ids"] = (
        list(rejected_surface_ids) if isinstance(rejected_surface_ids, list) else []
    )
    detail["invalid_record_ids"] = (
        list(invalid_record_ids) if isinstance(invalid_record_ids, list) else []
    )
    detail["unreviewed_candidate_surface_ids"] = (
        list(unreviewed_surface_ids) if isinstance(unreviewed_surface_ids, list) else []
    )
    detail["operator_acceptance_required"] = summary.get("operator_acceptance_required")
    detail["operator_review_required"] = summary.get("operator_review_required")
    detail["physical_fixture_gap_remains"] = summary.get("physical_fixture_gap_remains")
    detail["fixture_closure_allowed_by_acceptance"] = summary.get(
        "fixture_closure_allowed_by_acceptance"
    )
    detail["live_authority_granted"] = summary.get("live_authority_granted")
    detail["full_autonomous_ready"] = summary.get("full_autonomous_ready")
    detail["blocked_by"] = list(blocked_by) if isinstance(blocked_by, list) else []
    detail["violations"] = violations
    finished = datetime.now(timezone.utc)
    return CheckResult(
        name="read_surface_physical_readback_fixture_acceptance_ledger_schema",
        status="passed" if not violations else "failed",
        started_at=started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        finished_at=finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration_sec=round((finished - started).total_seconds(), 3),
        detail=detail,
    )


def validate_read_surface_physical_readback_fixture_closure_plan_snapshot() -> CheckResult:
    started = datetime.now(timezone.utc)
    detail: dict[str, object] = {
        "policy": "physical_readback_fixture_closure_plan_is_manual_read_only_gate",
    }
    violations: list[str] = []

    try:
        plan = build_read_surface_physical_readback_fixture_closure_plan()
    except Exception as exc:  # pragma: no cover - defensive release-check guard
        plan = {}
        violations.append(f"fixture_closure_plan_build_error:{type(exc).__name__}")

    safety = plan.get("safety") if isinstance(plan.get("safety"), Mapping) else {}
    summary = plan.get("summary") if isinstance(plan.get("summary"), Mapping) else {}
    candidates = (
        plan.get("closure_candidates")
        if isinstance(plan.get("closure_candidates"), list)
        else []
    )

    if plan.get("type") != "read_surface_physical_readback_fixture_closure_plan":
        violations.append("fixture_closure_plan_type_mismatch")
    if plan.get("method_id") != READ_SURFACE_PHYSICAL_READBACK_FIXTURE_CLOSURE_PLAN_METHOD_ID:
        violations.append("fixture_closure_plan_method_id_mismatch")
    if plan.get("status") not in READ_SURFACE_PHYSICAL_READBACK_FIXTURE_CLOSURE_PLAN_STATUSES:
        violations.append("fixture_closure_plan_status_unknown")

    for key in ("read_only", "operator_required"):
        if safety.get(key) is not True:
            violations.append(f"fixture_closure_plan_{key}_not_true")
    for key in (
        "live_write",
        "mutates_console_state",
        "queries_console",
        "executes_tests",
        "execution_allowed_by_plan",
        "fixture_update_allowed_by_plan",
        "fixture_closure_allowed_by_plan",
        "live_widening_allowed",
        "live_authority_granted",
        "full_autonomous_ready",
    ):
        if safety.get(key) is not False:
            violations.append(f"fixture_closure_plan_{key}_not_false")

    closure_candidate_count = summary.get("closure_candidate_count")
    accepted_record_count = summary.get("accepted_record_count")
    invalid_record_count = summary.get("invalid_record_count")
    unreviewed_candidate_count = summary.get("unreviewed_candidate_count")
    if not isinstance(closure_candidate_count, int) or closure_candidate_count != len(candidates):
        violations.append("fixture_closure_plan_candidate_count_mismatch")
        closure_candidate_count = len(candidates)
    if not isinstance(accepted_record_count, int):
        violations.append("fixture_closure_plan_accepted_count_missing")
        accepted_record_count = 0
    if not isinstance(invalid_record_count, int):
        violations.append("fixture_closure_plan_invalid_count_missing")
        invalid_record_count = 0
    if not isinstance(unreviewed_candidate_count, int):
        violations.append("fixture_closure_plan_unreviewed_count_missing")
        unreviewed_candidate_count = 0
    if invalid_record_count:
        violations.append("fixture_closure_plan_invalid_record_present")

    if summary.get("manual_fixture_closure_required") is not True:
        violations.append("fixture_closure_plan_manual_closure_required_not_true")
    if summary.get("operator_review_required") is not True:
        violations.append("fixture_closure_plan_operator_review_not_true")
    if summary.get("physical_fixture_gap_remains") is not True:
        violations.append("fixture_closure_plan_physical_gap_not_true")
    if summary.get("fixture_update_allowed_by_plan") is not False:
        violations.append("fixture_closure_plan_fixture_update_not_false")
    if summary.get("fixture_closure_allowed_by_plan") is not False:
        violations.append("fixture_closure_plan_fixture_closure_not_false")
    if summary.get("live_authority_granted") is not False:
        violations.append("fixture_closure_plan_live_authority_not_false")
    if summary.get("full_autonomous_ready") is not False:
        violations.append("fixture_closure_plan_full_autonomous_not_false")

    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            violations.append(f"fixture_closure_plan_candidate_{index}_not_object")
            continue
        if candidate.get("manual_fixture_closure_required") is not True:
            violations.append(f"fixture_closure_plan_candidate_{index}_manual_required_not_true")
        if candidate.get("fixture_update_allowed_by_plan") is not False:
            violations.append(f"fixture_closure_plan_candidate_{index}_update_not_false")
        if candidate.get("fixture_closure_allowed_by_plan") is not False:
            violations.append(f"fixture_closure_plan_candidate_{index}_closure_not_false")
        if candidate.get("live_authority_granted") is not False:
            violations.append(f"fixture_closure_plan_candidate_{index}_live_authority_not_false")
        if candidate.get("full_autonomous_ready") is not False:
            violations.append(f"fixture_closure_plan_candidate_{index}_full_auto_not_false")

    closure_candidate_surface_ids = summary.get("closure_candidate_surface_ids")
    rejected_surface_ids = summary.get("rejected_surface_ids")
    invalid_record_ids = summary.get("invalid_record_ids")
    unreviewed_surface_ids = summary.get("unreviewed_candidate_surface_ids")
    blocked_by = summary.get("blocked_by")
    detail["plan_status"] = plan.get("status")
    detail["ledger_status"] = summary.get("ledger_status")
    detail["review_candidate_count"] = int(summary.get("review_candidate_count") or 0)
    detail["acceptance_record_count"] = int(summary.get("acceptance_record_count") or 0)
    detail["accepted_record_count"] = accepted_record_count
    detail["rejected_record_count"] = int(summary.get("rejected_record_count") or 0)
    detail["invalid_record_count"] = invalid_record_count
    detail["unreviewed_candidate_count"] = unreviewed_candidate_count
    detail["closure_candidate_count"] = closure_candidate_count
    detail["closure_candidate_surface_ids"] = (
        list(closure_candidate_surface_ids)
        if isinstance(closure_candidate_surface_ids, list)
        else []
    )
    detail["rejected_surface_ids"] = (
        list(rejected_surface_ids) if isinstance(rejected_surface_ids, list) else []
    )
    detail["invalid_record_ids"] = (
        list(invalid_record_ids) if isinstance(invalid_record_ids, list) else []
    )
    detail["unreviewed_candidate_surface_ids"] = (
        list(unreviewed_surface_ids) if isinstance(unreviewed_surface_ids, list) else []
    )
    detail["manual_fixture_closure_candidate_available"] = summary.get(
        "manual_fixture_closure_candidate_available"
    )
    detail["manual_fixture_closure_required"] = summary.get(
        "manual_fixture_closure_required"
    )
    detail["operator_review_required"] = summary.get("operator_review_required")
    detail["physical_fixture_gap_remains"] = summary.get("physical_fixture_gap_remains")
    detail["fixture_update_allowed_by_plan"] = summary.get(
        "fixture_update_allowed_by_plan"
    )
    detail["fixture_closure_allowed_by_plan"] = summary.get(
        "fixture_closure_allowed_by_plan"
    )
    detail["live_authority_granted"] = summary.get("live_authority_granted")
    detail["full_autonomous_ready"] = summary.get("full_autonomous_ready")
    detail["blocked_by"] = list(blocked_by) if isinstance(blocked_by, list) else []
    detail["violations"] = violations
    finished = datetime.now(timezone.utc)
    return CheckResult(
        name="read_surface_physical_readback_fixture_closure_plan_schema",
        status="passed" if not violations else "failed",
        started_at=started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        finished_at=finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration_sec=round((finished - started).total_seconds(), 3),
        detail=detail,
    )


def validate_read_surface_physical_readback_fixture_closure_plan_artifact_snapshot(
    plan_path: Path,
) -> CheckResult:
    started = datetime.now(timezone.utc)
    detail: dict[str, object] = {
        "policy": "physical_readback_fixture_closure_plan_artifact_is_manual_read_only_gate",
        "plan_path": str(plan_path),
    }
    violations: list[str] = []

    payload, load_error = _json_object_from_path(plan_path)
    plan = payload or {}
    safety = plan.get("safety") if isinstance(plan.get("safety"), Mapping) else {}
    summary = plan.get("summary") if isinstance(plan.get("summary"), Mapping) else {}
    candidates = (
        plan.get("closure_candidates")
        if isinstance(plan.get("closure_candidates"), list)
        else []
    )
    operator_sequence = (
        plan.get("operator_sequence")
        if isinstance(plan.get("operator_sequence"), list)
        else []
    )
    commands = plan.get("commands") if isinstance(plan.get("commands"), Mapping) else {}

    if load_error:
        violations.append(
            f"read_surface_physical_readback_fixture_closure_plan_artifact_{load_error}"
        )
    if plan.get("type") != "read_surface_physical_readback_fixture_closure_plan":
        violations.append("fixture_closure_plan_artifact_type_mismatch")
    if plan.get("method_id") != READ_SURFACE_PHYSICAL_READBACK_FIXTURE_CLOSURE_PLAN_METHOD_ID:
        violations.append("fixture_closure_plan_artifact_method_id_mismatch")
    if (
        plan.get("source_method_id")
        != READ_SURFACE_PHYSICAL_READBACK_FIXTURE_ACCEPTANCE_LEDGER_METHOD_ID
    ):
        violations.append("fixture_closure_plan_artifact_source_method_mismatch")
    if plan.get("status") not in READ_SURFACE_PHYSICAL_READBACK_FIXTURE_CLOSURE_PLAN_STATUSES:
        violations.append("fixture_closure_plan_artifact_status_unknown")

    for key in ("read_only", "operator_required"):
        if safety.get(key) is not True:
            violations.append(f"fixture_closure_plan_artifact_{key}_not_true")
    for key in (
        "live_write",
        "mutates_console_state",
        "queries_console",
        "executes_tests",
        "execution_allowed_by_plan",
        "fixture_update_allowed_by_plan",
        "fixture_closure_allowed_by_plan",
        "live_widening_allowed",
        "live_authority_granted",
        "full_autonomous_ready",
    ):
        if safety.get(key) is not False:
            violations.append(f"fixture_closure_plan_artifact_{key}_not_false")

    closure_candidate_count = summary.get("closure_candidate_count")
    accepted_record_count = summary.get("accepted_record_count")
    invalid_record_count = summary.get("invalid_record_count")
    unreviewed_candidate_count = summary.get("unreviewed_candidate_count")
    if not isinstance(closure_candidate_count, int) or closure_candidate_count != len(candidates):
        violations.append("fixture_closure_plan_artifact_candidate_count_mismatch")
        closure_candidate_count = len(candidates)
    if not isinstance(accepted_record_count, int):
        violations.append("fixture_closure_plan_artifact_accepted_count_missing")
        accepted_record_count = 0
    if not isinstance(invalid_record_count, int):
        violations.append("fixture_closure_plan_artifact_invalid_count_missing")
        invalid_record_count = 0
    if not isinstance(unreviewed_candidate_count, int):
        violations.append("fixture_closure_plan_artifact_unreviewed_count_missing")
        unreviewed_candidate_count = 0
    if invalid_record_count:
        violations.append("fixture_closure_plan_artifact_invalid_record_present")

    for key in ("manual_fixture_closure_required", "operator_review_required", "physical_fixture_gap_remains"):
        if summary.get(key) is not True:
            violations.append(f"fixture_closure_plan_artifact_{key}_not_true")
    for key in (
        "fixture_update_allowed_by_plan",
        "fixture_closure_allowed_by_plan",
        "live_authority_granted",
        "full_autonomous_ready",
    ):
        if summary.get(key) is not False:
            violations.append(f"fixture_closure_plan_artifact_summary_{key}_not_false")
    blocked_by = summary.get("blocked_by")
    if not isinstance(blocked_by, list) or not blocked_by:
        violations.append("fixture_closure_plan_artifact_blocked_by_missing")
        blocked_by = []
    if not isinstance(summary.get("next_global_step"), str) or not summary.get(
        "next_global_step"
    ):
        violations.append("fixture_closure_plan_artifact_next_global_step_missing")

    if summary.get("manual_fixture_closure_candidate_available") is not bool(candidates):
        violations.append("fixture_closure_plan_artifact_manual_candidate_flag_mismatch")
    if plan.get("status") == "manual_fixture_closure_candidate_available" and not candidates:
        violations.append("fixture_closure_plan_artifact_candidate_status_without_candidates")
    if plan.get("status") != "manual_fixture_closure_candidate_available" and candidates:
        violations.append("fixture_closure_plan_artifact_candidates_without_candidate_status")

    closure_candidate_surface_ids = (
        summary.get("closure_candidate_surface_ids")
        if isinstance(summary.get("closure_candidate_surface_ids"), list)
        else []
    )
    if len(closure_candidate_surface_ids) != len(candidates):
        violations.append("fixture_closure_plan_artifact_candidate_surface_count_mismatch")
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, Mapping):
            violations.append(f"fixture_closure_plan_artifact_candidate_{index}_not_object")
            continue
        for field in (
            "review_id",
            "surface_id",
            "artifact_source_id",
            "required_contract_method_id",
            "reviewed_by",
            "reviewed_at",
        ):
            if not isinstance(candidate.get(field), str) or not candidate.get(field):
                violations.append(
                    f"fixture_closure_plan_artifact_candidate_{index}_{field}_missing"
                )
        if candidate.get("manual_fixture_closure_required") is not True:
            violations.append(
                f"fixture_closure_plan_artifact_candidate_{index}_manual_required_not_true"
            )
        for key in (
            "fixture_update_allowed_by_plan",
            "fixture_closure_allowed_by_plan",
            "live_authority_granted",
            "full_autonomous_ready",
        ):
            if candidate.get(key) is not False:
                violations.append(
                    f"fixture_closure_plan_artifact_candidate_{index}_{key}_not_false"
                )

    if len(operator_sequence) < 5:
        violations.append("fixture_closure_plan_artifact_operator_sequence_short")
    focused_validation = commands.get("focused_validation")
    if not isinstance(focused_validation, list):
        violations.append("fixture_closure_plan_artifact_focused_validation_missing")
        focused_validation = []
    else:
        for test_path in (
            "tests/test_read_surface_physical_readback_fixture_closure_plan.py",
            "tests/test_no_wing_release_check.py",
        ):
            if test_path not in focused_validation:
                violations.append(
                    "fixture_closure_plan_artifact_focused_validation_missing:"
                    f"{test_path}"
                )

    rejected_surface_ids = summary.get("rejected_surface_ids")
    invalid_record_ids = summary.get("invalid_record_ids")
    unreviewed_surface_ids = summary.get("unreviewed_candidate_surface_ids")
    detail["plan_type"] = plan.get("type")
    detail["plan_status"] = plan.get("status")
    detail["method_id"] = plan.get("method_id")
    detail["source_method_id"] = plan.get("source_method_id")
    detail["read_only"] = safety.get("read_only")
    detail["live_write"] = safety.get("live_write")
    detail["mutates_console_state"] = safety.get("mutates_console_state")
    detail["queries_console"] = safety.get("queries_console")
    detail["executes_tests"] = safety.get("executes_tests")
    detail["operator_required"] = safety.get("operator_required")
    detail["execution_allowed_by_plan"] = safety.get("execution_allowed_by_plan")
    detail["fixture_update_allowed_by_plan"] = safety.get(
        "fixture_update_allowed_by_plan"
    )
    detail["fixture_closure_allowed_by_plan"] = safety.get(
        "fixture_closure_allowed_by_plan"
    )
    detail["live_widening_allowed"] = safety.get("live_widening_allowed")
    detail["live_authority_granted"] = safety.get("live_authority_granted")
    detail["full_autonomous_ready"] = safety.get("full_autonomous_ready")
    detail["ledger_status"] = summary.get("ledger_status")
    detail["review_candidate_count"] = int(summary.get("review_candidate_count") or 0)
    detail["acceptance_record_count"] = int(summary.get("acceptance_record_count") or 0)
    detail["accepted_record_count"] = accepted_record_count
    detail["rejected_record_count"] = int(summary.get("rejected_record_count") or 0)
    detail["invalid_record_count"] = invalid_record_count
    detail["unreviewed_candidate_count"] = unreviewed_candidate_count
    detail["closure_candidate_count"] = closure_candidate_count
    detail["closure_candidate_surface_ids"] = list(closure_candidate_surface_ids)
    detail["rejected_surface_ids"] = (
        list(rejected_surface_ids) if isinstance(rejected_surface_ids, list) else []
    )
    detail["invalid_record_ids"] = (
        list(invalid_record_ids) if isinstance(invalid_record_ids, list) else []
    )
    detail["unreviewed_candidate_surface_ids"] = (
        list(unreviewed_surface_ids) if isinstance(unreviewed_surface_ids, list) else []
    )
    detail["manual_fixture_closure_candidate_available"] = summary.get(
        "manual_fixture_closure_candidate_available"
    )
    detail["manual_fixture_closure_required"] = summary.get(
        "manual_fixture_closure_required"
    )
    detail["operator_review_required"] = summary.get("operator_review_required")
    detail["physical_fixture_gap_remains"] = summary.get("physical_fixture_gap_remains")
    detail["summary_fixture_update_allowed_by_plan"] = summary.get(
        "fixture_update_allowed_by_plan"
    )
    detail["summary_fixture_closure_allowed_by_plan"] = summary.get(
        "fixture_closure_allowed_by_plan"
    )
    detail["summary_live_authority_granted"] = summary.get("live_authority_granted")
    detail["summary_full_autonomous_ready"] = summary.get("full_autonomous_ready")
    detail["blocked_by"] = list(blocked_by)
    detail["next_global_step"] = summary.get("next_global_step")
    detail["operator_sequence_count"] = len(operator_sequence)
    detail["focused_validation"] = list(focused_validation)
    detail["violations"] = violations
    finished = datetime.now(timezone.utc)
    return CheckResult(
        name="read_surface_physical_readback_fixture_closure_plan_artifact_schema",
        status="passed" if not violations else "failed",
        started_at=started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        finished_at=finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration_sec=round((finished - started).total_seconds(), 3),
        detail=detail,
    )


def validate_supervised_proof_report_gain_dry_run(report_path: Path) -> CheckResult:
    started = datetime.now(timezone.utc)
    payload, load_error = _json_object_from_path(report_path)
    detail: dict[str, object] = {
        "report_path": str(report_path),
        "policy": "supervised_proof_report_must_include_read_only_gain_dry_run_evidence",
    }
    violations: list[str] = []
    gain_rows: list[Mapping[str, Any]] = []

    if load_error is not None:
        violations.append(load_error)
    else:
        assert payload is not None
        safety = payload.get("safety") if isinstance(payload.get("safety"), Mapping) else {}
        summary = payload.get("summary") if isinstance(payload.get("summary"), Mapping) else {}
        rows = payload.get("rows") if isinstance(payload.get("rows"), list) else []

        if safety.get("read_only") is not True:
            violations.append("proof_report_read_only_not_true")
        for key in ("live_write", "live_widening_allowed", "full_autonomous_ready"):
            if safety.get(key) is not False:
                violations.append(f"proof_report_{key}_not_false")

        gain_rows = [
            row
            for row in rows
            if isinstance(row, Mapping)
            and row.get("value_type") == "gain"
            and row.get("proof_mode") == "dry_run"
            and row.get("status") == "dry_run_ready"
        ]
        if not gain_rows:
            violations.append("gain_dry_run_ready_row_missing")

        value_type_counts = summary.get("value_type_counts")
        if not isinstance(value_type_counts, Mapping) or int(value_type_counts.get("gain") or 0) < 1:
            violations.append("summary_gain_value_type_count_missing")
        dry_run_ids = summary.get("dry_run_ready_proof_ids")
        if not isinstance(dry_run_ids, list) or not dry_run_ids:
            violations.append("summary_dry_run_ready_proof_ids_missing")

        for row in gain_rows:
            if row.get("promotion_allowed") is not False:
                violations.append("gain_dry_run_promotion_allowed_not_false")
            if row.get("autonomous_ready") is not False:
                violations.append("gain_dry_run_autonomous_ready_not_false")
            blockers = row.get("blockers") if isinstance(row.get("blockers"), list) else []
            if "physical_proof_required_for_promotion" not in blockers:
                violations.append("gain_dry_run_physical_proof_blocker_missing")
            evidence = row.get("evidence") if isinstance(row.get("evidence"), Mapping) else {}
            expected_sources = {
                "pre_read_source": "fake_mixer",
                "readback_source": "fake_mixer",
                "rollback_readback_source": "fake_mixer",
            }
            for key, expected in expected_sources.items():
                if evidence.get(key) != expected:
                    violations.append(f"gain_dry_run_{key}_not_fake_mixer")
            if evidence.get("operator_approval_recorded") is not True:
                violations.append("gain_dry_run_operator_approval_missing")
            if evidence.get("idempotency_key_present") is not True:
                violations.append("gain_dry_run_idempotency_key_missing")

        detail["gain_dry_run_row_count"] = len(gain_rows)
        detail["dry_run_ready_proof_ids"] = list(dry_run_ids) if isinstance(dry_run_ids, list) else []
        detail["value_type_counts"] = dict(value_type_counts) if isinstance(value_type_counts, Mapping) else {}

    detail["violations"] = violations
    finished = datetime.now(timezone.utc)
    return CheckResult(
        name="supervised_proof_report_gain_dry_run_schema",
        status="passed" if not violations else "failed",
        started_at=started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        finished_at=finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration_sec=round((finished - started).total_seconds(), 3),
        detail=detail,
    )


def validate_supervised_proof_report_next_wing_plan(report_path: Path) -> CheckResult:
    started = datetime.now(timezone.utc)
    payload, load_error = _json_object_from_path(report_path)
    detail: dict[str, object] = {
        "report_path": str(report_path),
        "policy": "next_wing_proof_plan_must_be_operator_required_and_non_executing",
    }
    violations: list[str] = []
    planned_value_types: list[str] = []
    planned_channels: list[int] = []
    global_blockers: list[str] = []
    readonly_preflight_required = False
    readonly_preflight_candidate_available = False
    readonly_preflight_evidence_status: object = None
    readonly_preflight_blocked_by: list[str] = []
    readonly_preflight_prestart_check_required = False
    readonly_preflight_prestart_check_must_pass_before_starting_backend = False
    readonly_preflight_prestart_check_command: list[str] = []
    readonly_preflight_prestart_check_failure_detail_fields: list[str] = []
    readonly_preflight_step_count = 0
    readonly_preflight_review_decision_required = False
    readonly_preflight_review_decision_accepted = False
    readonly_preflight_review_decision_status: object = None
    readonly_preflight_review_decision_blocked_by: list[str] = []
    readonly_preflight_review_decision_accepted_record_count = 0
    readonly_preflight_review_decision_accepted_source_ids: list[str] = []
    readonly_preflight_review_decision_candidate_prestart_contract_valid = False
    readonly_preflight_review_decision_candidate_prestart_source_ids: list[str] = []
    readonly_preflight_review_decision_rejected_source_ids: list[str] = []
    readonly_preflight_review_decision_invalid_record_count = 0
    readonly_preflight_review_decision_invalid_record_ids: list[str] = []
    readonly_preflight_review_decision_duplicate_source_ids: list[str] = []
    readonly_preflight_review_decision_conflicting_source_ids: list[str] = []
    readonly_preflight_review_decision_unreviewed_candidate_count = 0
    step_count = 0
    plan_status: object = None

    if load_error is not None:
        violations.append(load_error)
    else:
        assert payload is not None
        plan = payload.get("next_wing_proof_plan") if isinstance(payload.get("next_wing_proof_plan"), Mapping) else {}
        readiness = (
            payload.get("autonomy_promotion_readiness")
            if isinstance(payload.get("autonomy_promotion_readiness"), Mapping)
            else {}
        )
        readiness_summary = (
            readiness.get("summary") if isinstance(readiness.get("summary"), Mapping) else {}
        )
        preflight = (
            readiness_summary.get("wing_readonly_preflight_evidence")
            if isinstance(readiness_summary.get("wing_readonly_preflight_evidence"), Mapping)
            else {}
        )
        review_decision = (
            payload.get("wing_readonly_preflight_review_decision_ledger")
            if isinstance(payload.get("wing_readonly_preflight_review_decision_ledger"), Mapping)
            else readiness_summary.get("wing_readonly_preflight_review_decision_ledger")
            if isinstance(readiness_summary.get("wing_readonly_preflight_review_decision_ledger"), Mapping)
            else {}
        )
        review_summary = (
            review_decision.get("summary")
            if isinstance(review_decision.get("summary"), Mapping)
            else review_decision
        )
        safety = plan.get("safety") if isinstance(plan.get("safety"), Mapping) else {}
        summary = plan.get("summary") if isinstance(plan.get("summary"), Mapping) else {}
        preflight_steps = (
            plan.get("readonly_preflight_steps")
            if isinstance(plan.get("readonly_preflight_steps"), list)
            else []
        )
        steps = plan.get("steps") if isinstance(plan.get("steps"), list) else []
        plan_status = plan.get("status")

        if plan.get("type") != "supervised_wing_proof_plan":
            violations.append("next_wing_proof_plan_type_mismatch")
        if readiness.get("type") != "autonomy_promotion_readiness":
            violations.append("next_wing_proof_plan_readiness_snapshot_missing")
        if not preflight:
            violations.append("next_wing_proof_plan_preflight_overlay_missing")
        if not review_decision:
            violations.append("next_wing_proof_plan_review_decision_overlay_missing")
        if plan_status not in {
            "operator_proof_required",
            "manual_review_ready",
            "blocked_by_invalid_artifacts",
            "blocked_by_readonly_preflight",
        }:
            violations.append("next_wing_proof_plan_status_unknown")

        expected_true = ("read_only", "operator_required")
        for key in expected_true:
            if safety.get(key) is not True:
                violations.append(f"next_wing_proof_plan_{key}_not_true")
        expected_false = (
            "live_write",
            "execution_allowed_by_plan",
            "live_widening_allowed",
            "automatic_safety_update_allowed",
            "full_autonomous_ready",
        )
        for key in expected_false:
            if safety.get(key) is not False:
                violations.append(f"next_wing_proof_plan_{key}_not_false")

        step_count = int(summary.get("step_count") or 0)
        if step_count != len(steps):
            violations.append("next_wing_proof_plan_step_count_mismatch")
        planned_value_types = [str(item) for item in summary.get("planned_value_types", []) if item] if isinstance(summary.get("planned_value_types"), list) else []
        planned_channels = [
            int(item)
            for item in summary.get("planned_channels", [])
            if isinstance(item, (int, float, str)) and str(item).strip().lstrip("-").isdigit()
        ] if isinstance(summary.get("planned_channels"), list) else []
        global_blockers = [str(item) for item in summary.get("global_blockers", []) if item] if isinstance(summary.get("global_blockers"), list) else []
        readonly_preflight_required = summary.get("readonly_preflight_required") is True
        readonly_preflight_candidate_available = (
            summary.get("readonly_preflight_candidate_available") is True
        )
        readonly_preflight_evidence_status = summary.get("readonly_preflight_evidence_status")
        readonly_preflight_blocked_by = (
            [str(item) for item in summary.get("readonly_preflight_blocked_by", []) if item]
            if isinstance(summary.get("readonly_preflight_blocked_by"), list)
            else []
        )
        readonly_preflight_step_count = int(summary.get("readonly_preflight_step_count") or 0)
        readonly_preflight_review_decision_required = (
            summary.get("readonly_preflight_review_decision_required") is True
        )
        readonly_preflight_review_decision_accepted = (
            summary.get("readonly_preflight_review_decision_accepted") is True
        )
        readonly_preflight_review_decision_status = summary.get(
            "readonly_preflight_review_decision_status"
        )
        readonly_preflight_review_decision_blocked_by = (
            [
                str(item)
                for item in summary.get("readonly_preflight_review_decision_blocked_by", [])
                if item
            ]
            if isinstance(summary.get("readonly_preflight_review_decision_blocked_by"), list)
            else []
        )
        readonly_preflight_review_decision_accepted_record_count = int(
            summary.get("readonly_preflight_review_decision_accepted_record_count") or 0
        )
        readonly_preflight_review_decision_accepted_source_ids = (
            [
                str(item)
                for item in summary.get("readonly_preflight_review_decision_accepted_source_ids", [])
                if item
            ]
            if isinstance(summary.get("readonly_preflight_review_decision_accepted_source_ids"), list)
            else []
        )
        readonly_preflight_review_decision_candidate_prestart_contract_valid = (
            summary.get(
                "readonly_preflight_review_decision_candidate_prestart_contract_valid"
            )
            is True
        )
        readonly_preflight_review_decision_candidate_prestart_source_ids = (
            [
                str(item)
                for item in summary.get(
                    "readonly_preflight_review_decision_candidate_prestart_source_ids",
                    [],
                )
                if item
            ]
            if isinstance(
                summary.get("readonly_preflight_review_decision_candidate_prestart_source_ids"),
                list,
            )
            else []
        )
        readonly_preflight_review_decision_rejected_source_ids = (
            [
                str(item)
                for item in summary.get("readonly_preflight_review_decision_rejected_source_ids", [])
                if item
            ]
            if isinstance(summary.get("readonly_preflight_review_decision_rejected_source_ids"), list)
            else []
        )
        readonly_preflight_review_decision_invalid_record_count = int(
            summary.get("readonly_preflight_review_decision_invalid_record_count") or 0
        )
        readonly_preflight_review_decision_invalid_record_ids = (
            [
                str(item)
                for item in summary.get("readonly_preflight_review_decision_invalid_record_ids", [])
                if item
            ]
            if isinstance(summary.get("readonly_preflight_review_decision_invalid_record_ids"), list)
            else []
        )
        readonly_preflight_review_decision_duplicate_source_ids = (
            [
                str(item)
                for item in summary.get("readonly_preflight_review_decision_duplicate_source_ids", [])
                if item
            ]
            if isinstance(summary.get("readonly_preflight_review_decision_duplicate_source_ids"), list)
            else []
        )
        readonly_preflight_review_decision_conflicting_source_ids = (
            [
                str(item)
                for item in summary.get("readonly_preflight_review_decision_conflicting_source_ids", [])
                if item
            ]
            if isinstance(summary.get("readonly_preflight_review_decision_conflicting_source_ids"), list)
            else []
        )
        readonly_preflight_review_decision_unreviewed_candidate_count = int(
            summary.get("readonly_preflight_review_decision_unreviewed_candidate_count") or 0
        )
        if readonly_preflight_step_count != len(preflight_steps):
            violations.append("next_wing_proof_plan_preflight_step_count_mismatch")
        if not readonly_preflight_required:
            violations.append("next_wing_proof_plan_readonly_preflight_required_not_true")
        if not readonly_preflight_review_decision_required:
            violations.append("next_wing_proof_plan_review_decision_required_not_true")
        if preflight:
            if readonly_preflight_evidence_status != preflight.get("evidence_status"):
                violations.append("next_wing_proof_plan_preflight_status_mismatch")
            if readonly_preflight_blocked_by != _string_list(preflight.get("blocked_by")):
                violations.append("next_wing_proof_plan_preflight_blockers_mismatch")
            readonly_preflight_prestart_check_required = (
                preflight.get("prestart_backend_port_check_required") is True
            )
            readonly_preflight_prestart_check_must_pass_before_starting_backend = (
                preflight.get(
                    "prestart_backend_port_check_must_pass_before_starting_backend"
                )
                is True
            )
            readonly_preflight_prestart_check_command = (
                [str(item) for item in preflight.get("prestart_backend_port_check_command", []) if item]
                if isinstance(preflight.get("prestart_backend_port_check_command"), list)
                else []
            )
            readonly_preflight_prestart_check_failure_detail_fields = (
                [
                    str(item)
                    for item in preflight.get(
                        "prestart_backend_port_check_failure_detail_fields", []
                    )
                    if item
                ]
                if isinstance(
                    preflight.get("prestart_backend_port_check_failure_detail_fields"),
                    list,
                )
                else []
            )
            if not readonly_preflight_prestart_check_required:
                violations.append(
                    "next_wing_proof_plan_preflight_prestart_check_not_required"
                )
            if not readonly_preflight_prestart_check_must_pass_before_starting_backend:
                violations.append(
                    "next_wing_proof_plan_preflight_prestart_check_boundary_missing"
                )
            if (
                readonly_preflight_prestart_check_command
                != WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND
            ):
                violations.append(
                    "next_wing_proof_plan_preflight_prestart_check_command_mismatch"
                )
            if readonly_preflight_prestart_check_failure_detail_fields != list(
                WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS
            ):
                violations.append(
                    "next_wing_proof_plan_preflight_prestart_check_detail_fields_mismatch"
                )
            expected_candidate = (
                preflight.get("readiness_update_candidate_available") is True
            )
            if readonly_preflight_candidate_available != expected_candidate:
                violations.append("next_wing_proof_plan_preflight_candidate_mismatch")
        if review_decision:
            expected_decision_status = review_decision.get("status") or review_summary.get("ledger_status")
            if readonly_preflight_review_decision_status != expected_decision_status:
                violations.append("next_wing_proof_plan_review_decision_status_mismatch")
            if readonly_preflight_review_decision_blocked_by != _string_list(review_summary.get("blocked_by")):
                violations.append("next_wing_proof_plan_review_decision_blockers_mismatch")
            candidate_reviews = (
                review_summary.get("candidate_reviews")
                if isinstance(review_summary.get("candidate_reviews"), list)
                else []
            )
            candidate_rows_by_source = {
                str(row.get("source_id")): row
                for row in candidate_reviews
                if isinstance(row, Mapping) and row.get("source_id")
            }
            expected_candidate_prestart_source_ids = []
            for source_id in _string_list(review_summary.get("accepted_source_ids")):
                row = candidate_rows_by_source.get(source_id)
                if not isinstance(row, Mapping):
                    continue
                if row.get("backend_port_prestart_check_required") is not True:
                    continue
                if (
                    row.get(
                        "backend_port_prestart_check_must_pass_before_starting_backend"
                    )
                    is not True
                ):
                    continue
                if _string_list(row.get("backend_port_prestart_check_command")) != list(
                    WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_COMMAND
                ):
                    continue
                if _string_list(
                    row.get("backend_port_prestart_check_failure_detail_fields")
                ) != list(
                    WING_READONLY_PREFLIGHT_BACKEND_PORT_PRESTART_CHECK_FAILURE_DETAIL_FIELDS
                ):
                    continue
                expected_candidate_prestart_source_ids.append(source_id)
            expected_candidate_prestart_contract_valid = (
                bool(_string_list(review_summary.get("accepted_source_ids")))
                and expected_candidate_prestart_source_ids
                == _string_list(review_summary.get("accepted_source_ids"))
            )
            if (
                readonly_preflight_review_decision_candidate_prestart_contract_valid
                != expected_candidate_prestart_contract_valid
            ):
                violations.append(
                    "next_wing_proof_plan_review_decision_candidate_prestart_contract_mismatch"
                )
            if (
                readonly_preflight_review_decision_candidate_prestart_source_ids
                != expected_candidate_prestart_source_ids
            ):
                violations.append(
                    "next_wing_proof_plan_review_decision_candidate_prestart_sources_mismatch"
                )
            if (
                _string_list(review_summary.get("accepted_source_ids"))
                and not expected_candidate_prestart_contract_valid
                and WING_READONLY_PREFLIGHT_REVIEW_CANDIDATE_PRESTART_BLOCKER
                not in global_blockers
            ):
                violations.append(
                    "next_wing_proof_plan_review_decision_candidate_prestart_blocker_missing"
                )
            expected_decision_accepted = (
                review_summary.get("manual_readiness_update_candidate_available") is True
                and int(review_summary.get("accepted_record_count") or 0) > 0
                and expected_candidate_prestart_contract_valid
                and int(review_summary.get("invalid_record_count") or 0) == 0
                and int(review_summary.get("unreviewed_candidate_count") or 0) == 0
                and review_summary.get("readiness_matrix_update_allowed_by_decision") is not True
                and review_summary.get("readiness_closure_allowed_by_decision") is not True
                and review_summary.get("live_authority_granted") is not True
                and review_summary.get("full_autonomous_ready") is not True
            )
            if readonly_preflight_review_decision_accepted != expected_decision_accepted:
                violations.append("next_wing_proof_plan_review_decision_accepted_mismatch")
            if readonly_preflight_review_decision_accepted_record_count != int(
                review_summary.get("accepted_record_count") or 0
            ):
                violations.append("next_wing_proof_plan_review_decision_accepted_count_mismatch")
            if readonly_preflight_review_decision_accepted_source_ids != _string_list(
                review_summary.get("accepted_source_ids")
            ):
                violations.append("next_wing_proof_plan_review_decision_accepted_sources_mismatch")
            if readonly_preflight_review_decision_rejected_source_ids != _string_list(
                review_summary.get("rejected_source_ids")
            ):
                violations.append("next_wing_proof_plan_review_decision_rejected_sources_mismatch")
            if readonly_preflight_review_decision_invalid_record_count != int(
                review_summary.get("invalid_record_count") or 0
            ):
                violations.append("next_wing_proof_plan_review_decision_invalid_count_mismatch")
            if readonly_preflight_review_decision_invalid_record_ids != _string_list(
                review_summary.get("invalid_record_ids")
            ):
                violations.append("next_wing_proof_plan_review_decision_invalid_ids_mismatch")
            if readonly_preflight_review_decision_duplicate_source_ids != _string_list(
                review_summary.get("duplicate_source_ids")
            ):
                violations.append("next_wing_proof_plan_review_decision_duplicate_sources_mismatch")
            if readonly_preflight_review_decision_conflicting_source_ids != _string_list(
                review_summary.get("conflicting_source_ids")
            ):
                violations.append("next_wing_proof_plan_review_decision_conflicting_sources_mismatch")
            if readonly_preflight_review_decision_unreviewed_candidate_count != int(
                review_summary.get("unreviewed_candidate_count") or 0
            ):
                violations.append("next_wing_proof_plan_review_decision_unreviewed_count_mismatch")
        if plan_status == "operator_proof_required" and step_count < 1:
            violations.append("next_wing_proof_plan_required_without_steps")
        if plan_status == "operator_proof_required" and "operator_must_approve_and_execute_manually" not in global_blockers:
            violations.append("next_wing_proof_plan_operator_blocker_missing")
        if plan_status == "operator_proof_required" and not readonly_preflight_candidate_available:
            violations.append("next_wing_proof_plan_preflight_candidate_missing_for_steps")
        if plan_status == "operator_proof_required" and not readonly_preflight_review_decision_accepted:
            violations.append("next_wing_proof_plan_operator_proof_without_review_decision")
        if step_count and not readonly_preflight_candidate_available:
            violations.append("next_wing_proof_plan_steps_without_readonly_preflight_candidate")
        if step_count and not readonly_preflight_review_decision_accepted:
            violations.append("next_wing_proof_plan_steps_without_accepted_review_decision")
        if plan_status == "blocked_by_readonly_preflight":
            if step_count:
                violations.append("next_wing_proof_plan_preflight_blocked_with_steps")
            if readonly_preflight_candidate_available:
                violations.append("next_wing_proof_plan_preflight_blocked_candidate_true")
            if readonly_preflight_step_count < 1:
                violations.append("next_wing_proof_plan_preflight_blocked_without_preflight_step")
            if "readonly_preflight_candidate_required_before_supervised_proof" not in global_blockers:
                violations.append("next_wing_proof_plan_preflight_blocker_missing")
        if readonly_preflight_candidate_available and readonly_preflight_step_count:
            violations.append("next_wing_proof_plan_preflight_step_present_after_candidate")
        if readonly_preflight_candidate_available and not readonly_preflight_review_decision_accepted:
            if "accepted_readonly_preflight_review_decision_required_before_supervised_proof" not in global_blockers:
                violations.append("next_wing_proof_plan_review_decision_blocker_missing")

        for index, step in enumerate(preflight_steps):
            if not isinstance(step, Mapping):
                violations.append(f"next_wing_proof_plan_preflight_step_{index}_not_object")
                continue
            if step.get("read_only") is not True:
                violations.append(f"next_wing_proof_plan_preflight_step_{index}_read_only_not_true")
            if step.get("live_execution_required") is not False:
                violations.append(f"next_wing_proof_plan_preflight_step_{index}_live_execution_not_false")
            if step.get("execution_allowed_by_plan") is not False:
                violations.append(f"next_wing_proof_plan_preflight_step_{index}_execution_allowed")
            if step.get("operator_approval_required") is not True:
                violations.append(f"next_wing_proof_plan_preflight_step_{index}_operator_approval_not_required")
            command = [str(item) for item in step.get("command", []) if item] if isinstance(step.get("command"), list) else []
            command_text = " ".join(command)
            if "tools/wing_readonly_preflight_runner.py" not in command:
                violations.append(f"next_wing_proof_plan_preflight_step_{index}_runner_missing")
            if "--require-physical-readback" not in command:
                violations.append(f"next_wing_proof_plan_preflight_step_{index}_physical_flag_missing")
            forbidden_tokens = (
                "--execute",
                "tools/wing_operator_proposal_validation_runner.py",
                "apply_operator_proposal",
                "create_operator_proposal",
                "set_operator_mode",
                "rollback_last_manual_write",
                "emergency_stop",
            )
            for token in forbidden_tokens:
                if token in command_text:
                    violations.append(
                        f"next_wing_proof_plan_preflight_step_{index}_forbidden_command:{token}"
                    )

        for index, step in enumerate(steps):
            if not isinstance(step, Mapping):
                violations.append(f"next_wing_proof_plan_step_{index}_not_object")
                continue
            try:
                channel = int(step.get("channel"))
            except (TypeError, ValueError):
                channel = None
                violations.append(f"next_wing_proof_plan_step_{index}_channel_invalid")
            value_type = str(step.get("value_type") or "")
            if step.get("execution_allowed_by_plan") is not False:
                violations.append(f"next_wing_proof_plan_step_{index}_execution_allowed")
            if step.get("operator_approval_required") is not True:
                violations.append(f"next_wing_proof_plan_step_{index}_operator_approval_not_required")
            if step.get("live_execution_required") is not True:
                violations.append(f"next_wing_proof_plan_step_{index}_live_execution_not_declared")
            try:
                delta_db = float(step.get("delta_db"))
            except (TypeError, ValueError):
                violations.append(f"next_wing_proof_plan_step_{index}_delta_invalid")
                continue
            if delta_db >= 0 or abs(delta_db) > 0.5:
                violations.append(f"next_wing_proof_plan_step_{index}_delta_not_safe_negative_proof")
            required_evidence_components = (
                [str(item) for item in step.get("required_evidence_components", []) if item]
                if isinstance(step.get("required_evidence_components"), list)
                else []
            )
            if value_type == "gain" and required_evidence_components != _required_gain_evidence_components():
                violations.append(
                    f"next_wing_proof_plan_step_{index}_gain_evidence_components_mismatch"
                )
            command = (
                [str(item) for item in step.get("command", []) if item]
                if isinstance(step.get("command"), list)
                else []
            )
            if "tools/wing_operator_proposal_validation_runner.py" not in command:
                violations.append(f"next_wing_proof_plan_step_{index}_runner_missing")

            def command_value(flag: str) -> str | None:
                try:
                    return command[command.index(flag) + 1]
                except (ValueError, IndexError):
                    return None

            confirmed_channel = command_value("--operator-confirm-channel")
            confirmed_value_type = command_value("--operator-confirm-value-type")
            confirmed_delta = command_value("--operator-confirm-delta-db")
            if confirmed_channel is None:
                violations.append(
                    f"next_wing_proof_plan_step_{index}_operator_confirm_channel_missing"
                )
            elif channel is not None and confirmed_channel != str(channel):
                violations.append(
                    f"next_wing_proof_plan_step_{index}_operator_confirm_channel_mismatch"
                )
            if confirmed_value_type is None:
                violations.append(
                    f"next_wing_proof_plan_step_{index}_operator_confirm_value_type_missing"
                )
            elif confirmed_value_type != value_type:
                violations.append(
                    f"next_wing_proof_plan_step_{index}_operator_confirm_value_type_mismatch"
                )
            if confirmed_delta is None:
                violations.append(
                    f"next_wing_proof_plan_step_{index}_operator_confirm_delta_missing"
                )
            else:
                try:
                    confirmed_delta_db = float(confirmed_delta)
                except ValueError:
                    violations.append(
                        f"next_wing_proof_plan_step_{index}_operator_confirm_delta_invalid"
                    )
                else:
                    if round(confirmed_delta_db, 3) != round(delta_db, 3):
                        violations.append(
                            f"next_wing_proof_plan_step_{index}_operator_confirm_delta_mismatch"
                        )

    detail["plan_status"] = plan_status
    detail["step_count"] = step_count
    detail["planned_value_types"] = planned_value_types
    detail["planned_channels"] = planned_channels
    detail["readonly_preflight_required"] = readonly_preflight_required
    detail["readonly_preflight_candidate_available"] = readonly_preflight_candidate_available
    detail["readonly_preflight_evidence_status"] = readonly_preflight_evidence_status
    detail["readonly_preflight_blocked_by"] = readonly_preflight_blocked_by
    detail["readonly_preflight_prestart_check_required"] = (
        readonly_preflight_prestart_check_required
    )
    detail["readonly_preflight_prestart_check_must_pass_before_starting_backend"] = (
        readonly_preflight_prestart_check_must_pass_before_starting_backend
    )
    detail["readonly_preflight_prestart_check_command"] = (
        readonly_preflight_prestart_check_command
    )
    detail["readonly_preflight_prestart_check_failure_detail_fields"] = (
        readonly_preflight_prestart_check_failure_detail_fields
    )
    detail["readonly_preflight_step_count"] = readonly_preflight_step_count
    detail["readonly_preflight_review_decision_required"] = (
        readonly_preflight_review_decision_required
    )
    detail["readonly_preflight_review_decision_accepted"] = (
        readonly_preflight_review_decision_accepted
    )
    detail["readonly_preflight_review_decision_status"] = (
        readonly_preflight_review_decision_status
    )
    detail["readonly_preflight_review_decision_blocked_by"] = (
        readonly_preflight_review_decision_blocked_by
    )
    detail["readonly_preflight_review_decision_accepted_record_count"] = (
        readonly_preflight_review_decision_accepted_record_count
    )
    detail["readonly_preflight_review_decision_accepted_source_ids"] = (
        readonly_preflight_review_decision_accepted_source_ids
    )
    detail["readonly_preflight_review_decision_candidate_prestart_contract_valid"] = (
        readonly_preflight_review_decision_candidate_prestart_contract_valid
    )
    detail["readonly_preflight_review_decision_candidate_prestart_source_ids"] = (
        readonly_preflight_review_decision_candidate_prestart_source_ids
    )
    detail["readonly_preflight_review_decision_rejected_source_ids"] = (
        readonly_preflight_review_decision_rejected_source_ids
    )
    detail["readonly_preflight_review_decision_invalid_record_count"] = (
        readonly_preflight_review_decision_invalid_record_count
    )
    detail["readonly_preflight_review_decision_invalid_record_ids"] = (
        readonly_preflight_review_decision_invalid_record_ids
    )
    detail["readonly_preflight_review_decision_duplicate_source_ids"] = (
        readonly_preflight_review_decision_duplicate_source_ids
    )
    detail["readonly_preflight_review_decision_conflicting_source_ids"] = (
        readonly_preflight_review_decision_conflicting_source_ids
    )
    detail["readonly_preflight_review_decision_unreviewed_candidate_count"] = (
        readonly_preflight_review_decision_unreviewed_candidate_count
    )
    detail["global_blockers"] = global_blockers
    detail["violations"] = violations
    finished = datetime.now(timezone.utc)
    return CheckResult(
        name="next_wing_proof_plan_schema",
        status="passed" if not violations else "failed",
        started_at=started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        finished_at=finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration_sec=round((finished - started).total_seconds(), 3),
        detail=detail,
    )


def validate_wing_operator_proposal_runner_physical_readback_policy(
    runner_path: Path | None = None,
) -> CheckResult:
    started = datetime.now(timezone.utc)
    path = runner_path or REPO_ROOT / "tools/wing_operator_proposal_validation_runner.py"
    detail: dict[str, object] = {
        "runner_path": str(path),
        "policy": "physical_wing_proof_must_require_physical_readback_not_cache",
    }
    violations: list[str] = []
    physical_statuses: list[str] = []

    try:
        source = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        source = ""
        violations.append("wing_operator_runner_missing")

    if source:
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError:
            tree = None
            violations.append("wing_operator_runner_syntax_error")
        if tree is not None:
            for node in ast.walk(tree):
                if not isinstance(node, ast.Assign):
                    continue
                if not any(
                    isinstance(target, ast.Name)
                    and target.id == "PHYSICAL_READBACK_STATUSES"
                    for target in node.targets
                ):
                    continue
                try:
                    value = ast.literal_eval(node.value)
                except (TypeError, ValueError):
                    violations.append("physical_readback_statuses_not_literal")
                    value = set()
                if isinstance(value, (set, tuple, list)):
                    physical_statuses = sorted(str(item) for item in value)
                else:
                    violations.append("physical_readback_statuses_not_sequence")
                break
            else:
                violations.append("physical_readback_statuses_constant_missing")

        if physical_statuses != ["verified_from_physical_readback"]:
            violations.append("physical_readback_statuses_include_non_physical_status")
        if "verified_from_cache" in physical_statuses:
            violations.append("physical_readback_statuses_allows_cache")
        required_markers = {
            "execute_requires_physical_pre_read": (
                "require_physical_readback=bool(args.execute)"
            ),
            "apply_requires_physical_status": (
                'result.get("readback_status") in PHYSICAL_READBACK_STATUSES'
            ),
            "rollback_requires_physical_status": (
                "rollback_readback_status not in PHYSICAL_READBACK_STATUSES"
            ),
            "summary_exposes_required_flag": '"physical_readback_required"',
            "cache_status_remains_dry_run_only": '"status": "verified_from_cache"',
        }
        for violation_id, marker in required_markers.items():
            if marker not in source:
                violations.append(violation_id)
        if "VERIFIED_READBACK_STATUSES" in source:
            violations.append("legacy_verified_readback_statuses_constant_present")

    detail["physical_readback_statuses"] = physical_statuses
    detail["violations"] = violations
    finished = datetime.now(timezone.utc)
    return CheckResult(
        name="wing_operator_proposal_runner_physical_readback_policy",
        status="passed" if not violations else "failed",
        started_at=started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        finished_at=finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration_sec=round((finished - started).total_seconds(), 3),
        detail=detail,
    )


def validate_supervised_proof_evaluator_physical_readback_policy(
    evaluator_path: Path | None = None,
) -> CheckResult:
    started = datetime.now(timezone.utc)
    path = evaluator_path or REPO_ROOT / "backend/supervised_proof_artifact_evaluator.py"
    detail: dict[str, object] = {
        "evaluator_path": str(path),
        "policy": "physical_proof_artifacts_must_require_physical_readback_status_not_cache",
    }
    violations: list[str] = []
    physical_statuses: list[str] = []

    try:
        source = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        source = ""
        violations.append("supervised_proof_evaluator_missing")

    if source:
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError:
            tree = None
            violations.append("supervised_proof_evaluator_syntax_error")
        if tree is not None:
            for node in ast.walk(tree):
                if not isinstance(node, ast.Assign):
                    continue
                if not any(
                    isinstance(target, ast.Name)
                    and target.id == "PHYSICAL_READBACK_STATUSES"
                    for target in node.targets
                ):
                    continue
                try:
                    value = ast.literal_eval(node.value)
                except (TypeError, ValueError):
                    violations.append("physical_readback_statuses_not_literal")
                    value = set()
                if isinstance(value, (set, tuple, list)):
                    physical_statuses = sorted(str(item) for item in value)
                else:
                    violations.append("physical_readback_statuses_not_sequence")
                break
            else:
                violations.append("physical_readback_statuses_constant_missing")

        if physical_statuses != ["verified_from_physical_readback"]:
            violations.append("physical_readback_statuses_include_non_physical_status")
        if "verified_from_cache" in physical_statuses:
            violations.append("physical_readback_statuses_allows_cache")
        required_markers = {
            "pre_read_status_requires_physical": (
                "artifact.pre_read_status not in PHYSICAL_READBACK_STATUSES"
            ),
            "apply_status_requires_physical": (
                "artifact.readback_status not in PHYSICAL_READBACK_STATUSES"
            ),
            "rollback_status_requires_physical": (
                "artifact.rollback_readback_status not in PHYSICAL_READBACK_STATUSES"
            ),
            "pre_read_status_blocker": '"physical_pre_read_status_required"',
            "apply_status_blocker": '"physical_apply_readback_status_required"',
            "rollback_status_blocker": '"physical_rollback_readback_status_required"',
        }
        for violation_id, marker in required_markers.items():
            if marker not in source:
                violations.append(violation_id)

    detail["physical_readback_statuses"] = physical_statuses
    detail["violations"] = violations
    finished = datetime.now(timezone.utc)
    return CheckResult(
        name="supervised_proof_evaluator_physical_readback_policy",
        status="passed" if not violations else "failed",
        started_at=started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        finished_at=finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration_sec=round((finished - started).total_seconds(), 3),
        detail=detail,
    )


def validate_supervised_proof_report_autonomy_overlay(report_path: Path) -> CheckResult:
    started = datetime.now(timezone.utc)
    payload, load_error = _json_object_from_path(report_path)
    detail: dict[str, object] = {
        "report_path": str(report_path),
        "policy": "autonomy_evidence_overlay_must_remain_read_only_and_non_promoting",
    }
    violations: list[str] = []
    overlay_status: object = None
    row_count = 0
    status_counts: Mapping[str, Any] = {}
    global_blockers: list[str] = []
    manual_review_ids: list[str] = []
    remediation_ids: list[str] = []
    dry_run_ids: list[str] = []
    no_evidence_ids: list[str] = []

    if load_error is not None:
        violations.append(load_error)
    else:
        assert payload is not None
        overlay = (
            payload.get("autonomy_evidence_overlay")
            if isinstance(payload.get("autonomy_evidence_overlay"), Mapping)
            else {}
        )
        safety = overlay.get("safety") if isinstance(overlay.get("safety"), Mapping) else {}
        summary = overlay.get("summary") if isinstance(overlay.get("summary"), Mapping) else {}
        rows = overlay.get("rows") if isinstance(overlay.get("rows"), list) else []
        overlay_status = overlay.get("status")

        if overlay.get("type") != "autonomy_evidence_overlay":
            violations.append("autonomy_overlay_type_mismatch")
        if overlay_status not in {
            "manual_remediation_required",
            "manual_review_ready",
            "dry_run_only",
            "no_current_evidence",
        }:
            violations.append("autonomy_overlay_status_unknown")

        expected_true = ("read_only", "operator_required")
        for key in expected_true:
            if safety.get(key) is not True:
                violations.append(f"autonomy_overlay_{key}_not_true")
        expected_false = (
            "live_write",
            "live_widening_allowed",
            "automatic_safety_update_allowed",
            "full_autonomous_ready",
        )
        for key in expected_false:
            if safety.get(key) is not False:
                violations.append(f"autonomy_overlay_{key}_not_false")

        status_counts = summary.get("status_counts") if isinstance(summary.get("status_counts"), Mapping) else {}
        global_blockers = [str(item) for item in summary.get("global_blockers", []) if item] if isinstance(summary.get("global_blockers"), list) else []
        manual_review_ids = [str(item) for item in summary.get("manual_safety_review_candidate_target_ids", []) if item] if isinstance(summary.get("manual_safety_review_candidate_target_ids"), list) else []
        remediation_ids = [str(item) for item in summary.get("remediation_required_target_ids", []) if item] if isinstance(summary.get("remediation_required_target_ids"), list) else []
        dry_run_ids = [str(item) for item in summary.get("dry_run_only_target_ids", []) if item] if isinstance(summary.get("dry_run_only_target_ids"), list) else []
        no_evidence_ids = [str(item) for item in summary.get("no_current_evidence_target_ids", []) if item] if isinstance(summary.get("no_current_evidence_target_ids"), list) else []

        if summary.get("autonomous_ready_target_ids") != []:
            violations.append("autonomy_overlay_autonomous_ready_targets_not_empty")
        for blocker in (
            "overlay_does_not_modify_safety_rows",
            "manual_safety_review_required_before_promotion",
            "full_autonomous_mode_remains_blocked",
        ):
            if blocker not in global_blockers:
                violations.append(f"autonomy_overlay_global_blocker_missing:{blocker}")

        row_count = len(rows)
        row_value_types: set[str] = set()
        expected_value_types = set(CURRENT_PROVEN_LIVE_WRITE_KINDS)
        expected_row_statuses = {
            "manual_safety_review_candidate",
            "remediation_required",
            "dry_run_only",
            "no_current_evidence",
        }
        actual_status_counts = {status: 0 for status in expected_row_statuses}
        actual_target_ids_by_status = {status: [] for status in expected_row_statuses}
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                violations.append(f"autonomy_overlay_row_{index}_not_object")
                continue
            value_type = str(row.get("value_type") or "")
            row_status = str(row.get("status") or "")
            target_id = str(row.get("target_id") or "")
            row_value_types.add(value_type)
            if value_type not in expected_value_types:
                violations.append(f"autonomy_overlay_row_{index}_unexpected_value_type")
            if row.get("autonomous_ready") is not False:
                violations.append(f"autonomy_overlay_row_{index}_autonomous_ready_not_false")
            if row_status not in expected_row_statuses:
                violations.append(f"autonomy_overlay_row_{index}_status_unknown")
            else:
                actual_status_counts[row_status] += 1
                if target_id:
                    actual_target_ids_by_status[row_status].append(target_id)
            if row_status == "manual_safety_review_candidate":
                blockers = row.get("blockers") if isinstance(row.get("blockers"), list) else []
                if "automatic_promotion_forbidden" not in blockers:
                    violations.append(f"autonomy_overlay_row_{index}_automatic_promotion_blocker_missing")
        if summary.get("target_count") != row_count:
            violations.append("autonomy_overlay_target_count_mismatch")
        normalized_status_counts: dict[str, int] = {}
        for status in expected_row_statuses:
            try:
                normalized_status_counts[status] = int(status_counts.get(status, -1))
            except (TypeError, ValueError):
                normalized_status_counts[status] = -1
        if normalized_status_counts != actual_status_counts:
            violations.append("autonomy_overlay_status_counts_mismatch")
        expected_summary_buckets = {
            "manual_safety_review_candidate": manual_review_ids,
            "remediation_required": remediation_ids,
            "dry_run_only": dry_run_ids,
            "no_current_evidence": no_evidence_ids,
        }
        for status, summary_ids in expected_summary_buckets.items():
            if summary_ids != actual_target_ids_by_status[status]:
                violations.append(f"autonomy_overlay_{status}_target_ids_mismatch")
        if row_value_types != expected_value_types:
            violations.append("autonomy_overlay_fader_gain_rows_missing")
        if not dry_run_ids:
            violations.append("autonomy_overlay_dry_run_target_ids_missing")

    detail["overlay_status"] = overlay_status
    detail["row_count"] = row_count
    detail["status_counts"] = dict(status_counts)
    detail["manual_safety_review_candidate_target_ids"] = manual_review_ids
    detail["remediation_required_target_ids"] = remediation_ids
    detail["dry_run_only_target_ids"] = dry_run_ids
    detail["no_current_evidence_target_ids"] = no_evidence_ids
    detail["global_blockers"] = global_blockers
    detail["violations"] = violations
    finished = datetime.now(timezone.utc)
    return CheckResult(
        name="autonomy_evidence_overlay_schema",
        status="passed" if not violations else "failed",
        started_at=started.isoformat(timespec="seconds").replace("+00:00", "Z"),
        finished_at=finished.isoformat(timespec="seconds").replace("+00:00", "Z"),
        duration_sec=round((finished - started).total_seconds(), 3),
        detail=detail,
    )


def autonomy_gap_rollup_summary(results: Sequence[CheckResult]) -> dict[str, object] | None:
    for result in results:
        if result.name != "supervised_proof_report_gap_rollup_schema":
            continue
        detail = result.detail if isinstance(result.detail, Mapping) else {}
        counts = detail.get("full_auto_gap_bucket_counts")
        evidence_by_family = detail.get("evidence_requirement_ids_by_action_family")
        next_by_family = detail.get("next_required_evidence_by_action_family")
        supervised_kinds = detail.get("supervised_current_live_write_kinds")
        policy_kinds = detail.get("policy_supervised_live_write_kinds")
        supervised_surfaces = detail.get("supervised_current_live_write_surfaces")
        policy_surfaces = detail.get("policy_supervised_live_write_surfaces")
        unexpected_kinds = detail.get("unexpected_supervised_live_write_kinds")
        unexpected_surfaces = detail.get("unexpected_supervised_live_write_surfaces")
        violations = detail.get("violations")
        return {
            "status": result.status,
            "read_only": True,
            "live_write": False,
            "full_autonomous_ready": False,
            "autonomy_gap_rollup_present": bool(detail.get("autonomy_gap_rollup_present")),
            "full_auto_gap_bucket_counts": dict(counts) if isinstance(counts, Mapping) else {},
            "evidence_requirement_ids_by_action_family": dict(evidence_by_family)
            if isinstance(evidence_by_family, Mapping)
            else {},
            "next_required_evidence_by_action_family": dict(next_by_family)
            if isinstance(next_by_family, Mapping)
            else {},
            "supervised_current_live_write_kinds": list(supervised_kinds) if isinstance(supervised_kinds, list) else [],
            "policy_supervised_live_write_kinds": list(policy_kinds) if isinstance(policy_kinds, list) else [],
            "supervised_current_live_write_surfaces": list(supervised_surfaces)
            if isinstance(supervised_surfaces, list)
            else [],
            "policy_supervised_live_write_surfaces": list(policy_surfaces)
            if isinstance(policy_surfaces, list)
            else [],
            "unexpected_supervised_live_write_surfaces": list(unexpected_surfaces)
            if isinstance(unexpected_surfaces, list)
            else [],
            "unexpected_supervised_live_write_kinds": list(unexpected_kinds)
            if isinstance(unexpected_kinds, list)
            else [],
            "violations": list(violations) if isinstance(violations, list) else [],
        }
    return None


def autonomy_evidence_overlay_summary(results: Sequence[CheckResult]) -> dict[str, object] | None:
    for result in results:
        if result.name != "autonomy_evidence_overlay_schema":
            continue
        detail = result.detail if isinstance(result.detail, Mapping) else {}
        status_counts = detail.get("status_counts")
        manual_review_ids = detail.get("manual_safety_review_candidate_target_ids")
        remediation_ids = detail.get("remediation_required_target_ids")
        dry_run_ids = detail.get("dry_run_only_target_ids")
        no_evidence_ids = detail.get("no_current_evidence_target_ids")
        global_blockers = detail.get("global_blockers")
        violations = detail.get("violations")
        return {
            "status": result.status,
            "read_only": True,
            "live_write": False,
            "live_widening_allowed": False,
            "automatic_safety_update_allowed": False,
            "full_autonomous_ready": False,
            "operator_required": True,
            "overlay_status": detail.get("overlay_status"),
            "row_count": int(detail.get("row_count") or 0),
            "status_counts": dict(status_counts) if isinstance(status_counts, Mapping) else {},
            "manual_safety_review_candidate_target_ids": list(manual_review_ids)
            if isinstance(manual_review_ids, list)
            else [],
            "remediation_required_target_ids": list(remediation_ids)
            if isinstance(remediation_ids, list)
            else [],
            "dry_run_only_target_ids": list(dry_run_ids) if isinstance(dry_run_ids, list) else [],
            "no_current_evidence_target_ids": list(no_evidence_ids)
            if isinstance(no_evidence_ids, list)
            else [],
            "global_blockers": list(global_blockers) if isinstance(global_blockers, list) else [],
            "violations": list(violations) if isinstance(violations, list) else [],
        }
    return None


def supervised_gain_dry_run_summary(results: Sequence[CheckResult]) -> dict[str, object] | None:
    for result in results:
        if result.name != "supervised_proof_report_gain_dry_run_schema":
            continue
        detail = result.detail if isinstance(result.detail, Mapping) else {}
        dry_run_ids = detail.get("dry_run_ready_proof_ids")
        value_type_counts = detail.get("value_type_counts")
        violations = detail.get("violations")
        return {
            "status": result.status,
            "read_only": True,
            "live_write": False,
            "promotion_allowed": False,
            "autonomous_ready": False,
            "gain_dry_run_row_count": int(detail.get("gain_dry_run_row_count") or 0),
            "dry_run_ready_proof_ids": list(dry_run_ids) if isinstance(dry_run_ids, list) else [],
            "value_type_counts": dict(value_type_counts) if isinstance(value_type_counts, Mapping) else {},
            "violations": list(violations) if isinstance(violations, list) else [],
        }
    return None


def read_surface_fixture_evidence_summary(results: Sequence[CheckResult]) -> dict[str, object] | None:
    for result in results:
        if result.name != "read_surface_fixture_evidence_schema":
            continue
        detail = result.detail if isinstance(result.detail, Mapping) else {}
        status_counts = detail.get("status_counts")
        physical_missing = detail.get("physical_fixture_missing_surface_ids")
        violations = detail.get("violations")
        physical_missing_ids = list(physical_missing) if isinstance(physical_missing, list) else []
        next_physical_readback_surface_id = detail.get("next_physical_readback_surface_id")
        if not isinstance(next_physical_readback_surface_id, str):
            next_physical_readback_surface_id = (
                physical_missing_ids[0] if physical_missing_ids else None
            )
        return {
            "status": result.status,
            "read_only": True,
            "live_write": False,
            "full_autonomous_ready": False,
            "snapshot_status": detail.get("snapshot_status"),
            "surface_count": int(detail.get("surface_count") or 0),
            "surface_with_fixture_count": int(detail.get("surface_with_fixture_count") or 0),
            "declared_fixture_path_count": int(detail.get("declared_fixture_path_count") or 0),
            "declared_test_name_count": int(detail.get("declared_test_name_count") or 0),
            "physical_gap_surface_count": len(physical_missing_ids),
            "summary_recomputed_from_rows": True,
            "fixture_declaration_validation": True,
            "contract_gap_consistency_enforced": True,
            "status_counts": dict(status_counts) if isinstance(status_counts, Mapping) else {},
            "physical_fixture_missing_surface_ids": physical_missing_ids,
            "next_physical_readback_surface_id": next_physical_readback_surface_id,
            "violations": list(violations) if isinstance(violations, list) else [],
        }
    return None


def read_surface_remediation_plan_summary(results: Sequence[CheckResult]) -> dict[str, object] | None:
    for result in results:
        if result.name != "read_surface_remediation_plan_schema":
            continue
        detail = result.detail if isinstance(result.detail, Mapping) else {}
        physical_missing = detail.get("physical_fixture_missing_surface_ids")
        violations = detail.get("violations")
        physical_missing_ids = list(physical_missing) if isinstance(physical_missing, list) else []
        return {
            "status": result.status,
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "live_widening_allowed": False,
            "full_autonomous_ready": False,
            "plan_status": detail.get("plan_status"),
            "action_count": int(detail.get("action_count") or 0),
            "incomplete_surface_count": int(detail.get("incomplete_surface_count") or 0),
            "physical_gap_surface_count": len(physical_missing_ids),
            "physical_fixture_missing_surface_ids": physical_missing_ids,
            "next_physical_readback_surface_id": detail.get("next_physical_readback_surface_id"),
            "next_physical_readback_action_id": detail.get("next_physical_readback_action_id"),
            "next_physical_readback_target_action_ids": (
                list(detail.get("next_physical_readback_target_action_ids"))
                if isinstance(detail.get("next_physical_readback_target_action_ids"), list)
                else []
            ),
            "target_action_consistency": (
                detail.get("next_physical_readback_surface_id")
                == detail.get("expected_next_physical_readback_surface_id")
                and detail.get("next_physical_readback_action_id")
                == detail.get("expected_next_physical_readback_action_id")
                and detail.get("next_physical_readback_target_action_ids")
                == [detail.get("expected_next_physical_readback_action_id")]
            ),
            "violations": list(violations) if isinstance(violations, list) else [],
        }
    return None


def read_surface_physical_readback_capture_plan_summary(
    results: Sequence[CheckResult],
) -> dict[str, object] | None:
    for result in results:
        if result.name != "read_surface_physical_readback_capture_plan_schema":
            continue
        detail = result.detail if isinstance(result.detail, Mapping) else {}
        physical_missing = detail.get("physical_fixture_missing_surface_ids")
        required_contracts = detail.get("required_artifact_contracts")
        blocked_by = detail.get("blocked_by")
        forbidden_actions = detail.get("step_forbidden_actions")
        violations = detail.get("violations")
        physical_missing_ids = list(physical_missing) if isinstance(physical_missing, list) else []
        return {
            "status": result.status,
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "operator_required": True,
            "execution_allowed_by_plan": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "plan_status": detail.get("plan_status"),
            "capture_step_count": int(detail.get("capture_step_count") or 0),
            "physical_gap_surface_count": len(physical_missing_ids),
            "physical_fixture_missing_surface_ids": physical_missing_ids,
            "next_physical_readback_surface_id": detail.get("next_physical_readback_surface_id"),
            "next_physical_readback_action_id": detail.get("next_physical_readback_action_id"),
            "expected_next_physical_readback_surface_id": detail.get(
                "expected_next_physical_readback_surface_id"
            ),
            "expected_next_physical_readback_action_id": detail.get(
                "expected_next_physical_readback_action_id"
            ),
            "target_action_consistency": detail.get("target_action_consistency") is True,
            "required_artifact_contracts": (
                list(required_contracts) if isinstance(required_contracts, list) else []
            ),
            "blocked_by": list(blocked_by) if isinstance(blocked_by, list) else [],
            "step_surface_id": detail.get("step_surface_id"),
            "step_remediation_action_id": detail.get("step_remediation_action_id"),
            "step_forbidden_actions": (
                list(forbidden_actions) if isinstance(forbidden_actions, list) else []
            ),
            "violations": list(violations) if isinstance(violations, list) else [],
        }
    return None


def read_surface_physical_readback_capture_packet_artifact_summary(
    results: Sequence[CheckResult],
) -> dict[str, object] | None:
    for result in results:
        if result.name != "read_surface_physical_readback_capture_packet_artifact_schema":
            continue
        detail = result.detail if isinstance(result.detail, Mapping) else {}
        physical_missing = detail.get("physical_fixture_missing_surface_ids")
        required_contracts = detail.get("required_artifact_contracts")
        blocked_by = detail.get("blocked_by")
        target_surface_ids = detail.get("remaining_capture_target_surface_ids")
        target_action_ids = detail.get("remaining_capture_target_action_ids")
        target_contract_ids = detail.get("remaining_capture_target_contract_ids")
        target_forbidden_flags = detail.get("remaining_capture_target_forbidden_flags")
        forbidden_actions = detail.get("step_forbidden_actions")
        focused_validation = detail.get("focused_validation")
        violations = detail.get("violations")
        return {
            "status": result.status,
            "read_only": detail.get("read_only") is True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "operator_required": detail.get("operator_required") is True,
            "execution_allowed_by_packet": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "packet_path": detail.get("packet_path"),
            "packet_type": detail.get("packet_type"),
            "packet_status": detail.get("packet_status"),
            "source_plan_status": detail.get("source_plan_status"),
            "source_method_id": detail.get("source_method_id"),
            "physical_gap_surface_count": int(
                detail.get("physical_gap_surface_count") or 0
            ),
            "physical_fixture_missing_surface_ids": (
                list(physical_missing) if isinstance(physical_missing, list) else []
            ),
            "remaining_capture_target_count": int(
                detail.get("remaining_capture_target_count") or 0
            ),
            "remaining_capture_target_surface_ids": (
                list(target_surface_ids)
                if isinstance(target_surface_ids, list)
                else []
            ),
            "remaining_capture_target_action_ids": (
                list(target_action_ids) if isinstance(target_action_ids, list) else []
            ),
            "remaining_capture_target_contract_ids": (
                list(target_contract_ids)
                if isinstance(target_contract_ids, list)
                else []
            ),
            "remaining_capture_target_forbidden_flags": (
                list(target_forbidden_flags)
                if isinstance(target_forbidden_flags, list)
                else []
            ),
            "next_physical_readback_surface_id": detail.get(
                "next_physical_readback_surface_id"
            ),
            "next_physical_readback_action_id": detail.get(
                "next_physical_readback_action_id"
            ),
            "target_action_consistency": (
                detail.get("target_action_consistency") is True
            ),
            "capture_step_count": int(detail.get("capture_step_count") or 0),
            "required_artifact_contracts": (
                list(required_contracts) if isinstance(required_contracts, list) else []
            ),
            "blocked_by": list(blocked_by) if isinstance(blocked_by, list) else [],
            "step_surface_id": detail.get("step_surface_id"),
            "step_remediation_action_id": detail.get("step_remediation_action_id"),
            "step_target_console_surface": detail.get("step_target_console_surface"),
            "step_read_mode": detail.get("step_read_mode"),
            "step_contract_method_id": detail.get("step_contract_method_id"),
            "step_expected_artifact_count": int(
                detail.get("step_expected_artifact_count") or 0
            ),
            "step_blocker_count": int(detail.get("step_blocker_count") or 0),
            "step_forbidden_actions": (
                list(forbidden_actions) if isinstance(forbidden_actions, list) else []
            ),
            "operator_sequence_count": int(detail.get("operator_sequence_count") or 0),
            "focused_validation": (
                list(focused_validation)
                if isinstance(focused_validation, list)
                else []
            ),
            "violations": list(violations) if isinstance(violations, list) else [],
        }
    return None


def read_surface_physical_readback_capture_template_artifact_summary(
    results: Sequence[CheckResult],
) -> dict[str, object] | None:
    for result in results:
        if (
            result.name
            != "read_surface_physical_readback_capture_template_artifact_schema"
        ):
            continue
        detail = result.detail if isinstance(result.detail, Mapping) else {}
        surface_ids = detail.get("capture_template_surface_ids")
        contract_ids = detail.get("capture_template_contract_ids")
        payload_contract_ids = detail.get("payload_template_contract_ids")
        payload_artifact_types = detail.get("payload_template_artifact_types")
        payload_schema_ids = detail.get("payload_template_schema_ids")
        payload_field_counts = detail.get("payload_template_field_counts")
        required_edits = detail.get("required_operator_edits")
        blocked_by = detail.get("blocked_by")
        unsafe_flags = detail.get("record_unsafe_flags")
        focused_validation = detail.get("focused_validation")
        violations = detail.get("violations")
        return {
            "status": result.status,
            "read_only": detail.get("read_only") is True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "operator_required": detail.get("operator_required") is True,
            "execution_allowed_by_template": False,
            "accepted_as_physical_fixture_candidate": False,
            "fixture_closure_allowed_by_template": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "template_path": detail.get("template_path"),
            "template_type": detail.get("template_type"),
            "template_status": detail.get("template_status"),
            "source_packet_type": detail.get("source_packet_type"),
            "source_packet_status": detail.get("source_packet_status"),
            "source_method_id": detail.get("source_method_id"),
            "selected_surface_id": detail.get("selected_surface_id"),
            "include_all_targets": detail.get("include_all_targets") is True,
            "all_remaining_target_count": int(
                detail.get("all_remaining_target_count") or 0
            ),
            "capture_template_count": int(detail.get("capture_template_count") or 0),
            "capture_template_surface_ids": (
                list(surface_ids) if isinstance(surface_ids, list) else []
            ),
            "capture_template_contract_ids": (
                list(contract_ids) if isinstance(contract_ids, list) else []
            ),
            "payload_template_contract_ids": (
                list(payload_contract_ids)
                if isinstance(payload_contract_ids, list)
                else []
            ),
            "payload_template_artifact_types": (
                list(payload_artifact_types)
                if isinstance(payload_artifact_types, list)
                else []
            ),
            "payload_template_schema_ids": (
                list(payload_schema_ids)
                if isinstance(payload_schema_ids, list)
                else []
            ),
            "payload_template_field_counts": (
                list(payload_field_counts)
                if isinstance(payload_field_counts, list)
                else []
            ),
            "required_operator_edits": (
                list(required_edits) if isinstance(required_edits, list) else []
            ),
            "operator_sequence_count": int(detail.get("operator_sequence_count") or 0),
            "blocked_by": list(blocked_by) if isinstance(blocked_by, list) else [],
            "record_unsafe_flags": (
                list(unsafe_flags) if isinstance(unsafe_flags, list) else []
            ),
            "focused_validation": (
                list(focused_validation)
                if isinstance(focused_validation, list)
                else []
            ),
            "violations": list(violations) if isinstance(violations, list) else [],
        }
    return None


def wing_readonly_preflight_evidence_summary(
    results: Sequence[CheckResult],
) -> dict[str, object] | None:
    for result in results:
        if result.name != "wing_readonly_preflight_evidence_schema":
            continue
        detail = result.detail if isinstance(result.detail, Mapping) else {}
        blocked_by = detail.get("blocked_by")
        matrix_ids = detail.get("mvp_blocker_matrix_ids")
        closed_ids = detail.get("mvp_blockers_closed_by_evidence")
        violations = detail.get("violations")
        artifact_operator_next_steps = detail.get("artifact_operator_next_steps")
        plan_command = detail.get("plan_command")
        capture_command = detail.get("capture_command_template")
        review_command = detail.get("review_command")
        review_exit_codes = detail.get("review_exit_codes")
        review_manual_checklist_ids = detail.get("review_manual_checklist_ids")
        review_command_writes_decision_template = (
            detail.get("review_command_writes_decision_template") is True
        )
        review_decision_template_required_fields = detail.get(
            "review_decision_template_required_fields"
        )
        handoff_packet_command = detail.get("handoff_packet_command")
        prestart_backend_port_check_command = detail.get(
            "prestart_backend_port_check_command"
        )
        valid_candidate_count = int(detail.get("valid_candidate_count") or 0)
        invalid_candidate_count = int(detail.get("invalid_candidate_count") or 0)
        readiness_update_candidate_available = (
            result.status == "passed"
            and valid_candidate_count > 0
            and invalid_candidate_count == 0
            and detail.get("readiness_matrix_update_allowed") is True
            and detail.get("readiness_closure_allowed_by_evidence") is not True
            and detail.get("live_authority_granted") is not True
        )
        return {
            "status": result.status,
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "operator_required": True,
            "execution_allowed_by_evidence": False,
            "readiness_closure_allowed_by_evidence": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "evidence_status": detail.get("evidence_status"),
            "artifact_count": int(detail.get("artifact_count") or 0),
            "valid_candidate_count": valid_candidate_count,
            "invalid_candidate_count": invalid_candidate_count,
            "accepted_as_readonly_preflight_candidate": (
                detail.get("accepted_as_readonly_preflight_candidate") is True
            ),
            "readiness_matrix_update_allowed": (
                detail.get("readiness_matrix_update_allowed") is True
            ),
            "readiness_update_candidate_available": readiness_update_candidate_available,
            "mvp_blocker_candidate_count": int(detail.get("mvp_blocker_candidate_count") or 0),
            "mvp_blockers_closed_by_evidence": (
                list(closed_ids) if isinstance(closed_ids, list) else []
            ),
            "mvp_blocker_matrix_ids": (
                list(matrix_ids) if isinstance(matrix_ids, list) else []
            ),
            "latest_valid_candidate_run_id": detail.get("latest_valid_candidate_run_id"),
            "artifact_operator_next_steps": (
                list(artifact_operator_next_steps)
                if isinstance(artifact_operator_next_steps, list)
                else []
            ),
            "capture_tool_path": detail.get("capture_tool_path"),
            "capture_tool_available": detail.get("capture_tool_available") is True,
            "plan_command": (
                list(plan_command) if isinstance(plan_command, list) else []
            ),
            "plan_command_uses_plan_only": (
                detail.get("plan_command_uses_plan_only") is True
            ),
            "plan_command_output": detail.get("plan_command_output"),
            "plan_command_channels": detail.get("plan_command_channels"),
            "plan_command_operator_confirmed_channels": detail.get(
                "plan_command_operator_confirmed_channels"
            ),
            "plan_command_confirms_session_ready": (
                detail.get("plan_command_confirms_session_ready") is True
            ),
            "plan_command_value_types": detail.get("plan_command_value_types"),
            "plan_command_artifact_root": detail.get("plan_command_artifact_root"),
            "plan_command_matches_capture": (
                detail.get("plan_command_matches_capture") is True
            ),
            "prestart_backend_port_check_required": (
                detail.get("prestart_backend_port_check_required") is True
            ),
            "prestart_backend_port_check_must_pass_before_starting_backend": (
                detail.get("prestart_backend_port_check_must_pass_before_starting_backend")
                is True
            ),
            "prestart_backend_port_check_command": (
                list(prestart_backend_port_check_command)
                if isinstance(prestart_backend_port_check_command, list)
                else []
            ),
            "prestart_backend_port_check_failure_detail_fields": (
                list(detail.get("prestart_backend_port_check_failure_detail_fields"))
                if isinstance(
                    detail.get("prestart_backend_port_check_failure_detail_fields"),
                    list,
                )
                else []
            ),
            "capture_command_template": (
                list(capture_command) if isinstance(capture_command, list) else []
            ),
            "capture_command_channels": detail.get("capture_command_channels"),
            "capture_command_operator_confirmed_channels": detail.get(
                "capture_command_operator_confirmed_channels"
            ),
            "capture_command_requires_operator_channel_confirmation": (
                detail.get("capture_command_requires_operator_channel_confirmation") is True
            ),
            "capture_command_operator_confirmation_matches_channels": (
                detail.get("capture_command_operator_confirmation_matches_channels") is True
            ),
            "capture_command_confirms_session_ready": (
                detail.get("capture_command_confirms_session_ready") is True
            ),
            "capture_command_requires_physical_readback": (
                detail.get("capture_command_requires_physical_readback") is True
            ),
            "capture_command_expected_plan": detail.get("capture_command_expected_plan"),
            "capture_command_expected_plan_matches_plan_output": (
                detail.get("capture_command_expected_plan_matches_plan_output") is True
            ),
            "review_tool_path": detail.get("review_tool_path"),
            "review_tool_available": detail.get("review_tool_available") is True,
            "review_command": (
                list(review_command) if isinstance(review_command, list) else []
            ),
            "review_command_writes_decision_template": (
                review_command_writes_decision_template
            ),
            "review_exit_codes": (
                dict(review_exit_codes) if isinstance(review_exit_codes, Mapping) else {}
            ),
            "review_manual_checklist_ids": (
                list(review_manual_checklist_ids)
                if isinstance(review_manual_checklist_ids, list)
                else []
            ),
            "review_manual_checklist_complete": (
                detail.get("review_manual_checklist_complete") is True
            ),
            "review_decision_template_required_fields": (
                list(review_decision_template_required_fields)
                if isinstance(review_decision_template_required_fields, list)
                else []
            ),
            "review_decision_template_contract_complete": (
                detail.get("review_decision_template_contract_complete") is True
            ),
            "handoff_packet_tool_path": detail.get("handoff_packet_tool_path"),
            "handoff_packet_tool_available": (
                detail.get("handoff_packet_tool_available") is True
            ),
            "handoff_packet_command": (
                list(handoff_packet_command)
                if isinstance(handoff_packet_command, list)
                else []
            ),
            "handoff_packet_output": detail.get("handoff_packet_output"),
            "handoff_packet_plan_output": detail.get("handoff_packet_plan_output"),
            "handoff_packet_url": detail.get("handoff_packet_url"),
            "handoff_packet_artifact_root": detail.get("handoff_packet_artifact_root"),
            "handoff_packet_channels": detail.get("handoff_packet_channels"),
            "handoff_packet_value_types": detail.get("handoff_packet_value_types"),
            "handoff_packet_url_matches_capture": (
                detail.get("handoff_packet_url_matches_capture") is True
            ),
            "handoff_packet_artifact_root_matches_capture": (
                detail.get("handoff_packet_artifact_root_matches_capture") is True
            ),
            "handoff_packet_channel_matches_capture": (
                detail.get("handoff_packet_channel_matches_capture") is True
            ),
            "handoff_packet_value_types_match_capture": (
                detail.get("handoff_packet_value_types_match_capture") is True
            ),
            "blocked_by": list(blocked_by) if isinstance(blocked_by, list) else [],
            "violations": list(violations) if isinstance(violations, list) else [],
        }
    return None


def wing_readonly_preflight_plan_artifact_summary(
    results: Sequence[CheckResult],
) -> dict[str, object] | None:
    for result in results:
        if result.name != "wing_readonly_preflight_plan_artifact_schema":
            continue
        detail = result.detail if isinstance(result.detail, Mapping) else {}
        violations = detail.get("violations")
        forbidden_request_types = detail.get("forbidden_request_types")
        fail_closed_conditions = detail.get("fail_closed_conditions")
        post_run_review_commands = detail.get("post_run_review_commands")
        return {
            "status": result.status,
            "read_only": detail.get("read_only") is True,
            "live_write": False,
            "contacts_backend": False,
            "queries_console": False,
            "operator_required": detail.get("operator_required") is True,
            "expected_plan_required_for_real_run": (
                detail.get("expected_plan_required_for_real_run") is True
            ),
            "physical_preflight_requires_existing_backend_connection": (
                detail.get("physical_preflight_requires_existing_backend_connection")
                is True
            ),
            "backend_port_prestart_check_required": (
                detail.get("backend_port_prestart_check_required") is True
            ),
            "backend_port_prestart_check_must_pass_before_starting_backend": (
                detail.get("backend_port_prestart_check_must_pass_before_starting_backend")
                is True
            ),
            "backend_port_prestart_check_command": (
                list(detail.get("backend_port_prestart_check_command"))
                if isinstance(detail.get("backend_port_prestart_check_command"), list)
                else []
            ),
            "backend_port_prestart_check_failure_detail_fields": (
                list(detail.get("backend_port_prestart_check_failure_detail_fields"))
                if isinstance(
                    detail.get("backend_port_prestart_check_failure_detail_fields"),
                    list,
                )
                else []
            ),
            "manual_review_required_before_readiness_update": (
                detail.get("manual_review_required_before_readiness_update") is True
            ),
            "execution_allowed_by_plan": False,
            "live_authority_granted": False,
            "plan_path": detail.get("plan_path"),
            "plan_type": detail.get("plan_type"),
            "plan_status": detail.get("plan_status"),
            "plan_output": detail.get("plan_output"),
            "expected_plan_output": detail.get("expected_plan_output"),
            "selected_mvp_blocker": detail.get("selected_mvp_blocker"),
            "selected_mvp_blocker_status": detail.get("selected_mvp_blocker_status"),
            "channels": (
                list(detail.get("channels"))
                if isinstance(detail.get("channels"), list)
                else []
            ),
            "operator_confirmed_channels": (
                list(detail.get("operator_confirmed_channels"))
                if isinstance(detail.get("operator_confirmed_channels"), list)
                else []
            ),
            "operator_session_confirmation_required": (
                detail.get("operator_session_confirmation_required") is True
            ),
            "operator_confirmed_session_ready": (
                detail.get("operator_confirmed_session_ready") is True
            ),
            "value_types": (
                list(detail.get("value_types"))
                if isinstance(detail.get("value_types"), list)
                else []
            ),
            "request_count": int(detail.get("request_count") or 0),
            "expected_request_count": int(detail.get("expected_request_count") or 0),
            "request_sequence_matches_expected": (
                detail.get("request_sequence_matches_expected") is True
            ),
            "forbidden_request_types": (
                list(forbidden_request_types)
                if isinstance(forbidden_request_types, list)
                else []
            ),
            "fail_closed_conditions": (
                list(fail_closed_conditions)
                if isinstance(fail_closed_conditions, list)
                else []
            ),
            "post_run_review_commands": (
                list(post_run_review_commands)
                if isinstance(post_run_review_commands, list)
                else []
            ),
            "violations": list(violations) if isinstance(violations, list) else [],
        }
    return None


def wing_readonly_preflight_handoff_packet_artifact_summary(
    results: Sequence[CheckResult],
) -> dict[str, object] | None:
    for result in results:
        if result.name != "wing_readonly_preflight_handoff_packet_artifact_schema":
            continue
        detail = result.detail if isinstance(result.detail, Mapping) else {}
        violations = detail.get("violations")
        command_names = detail.get("command_names")
        checklist_ids = detail.get("required_review_checklist_ids")
        forbidden_types = detail.get("forbidden_live_request_types")
        blockers = detail.get("blockers_not_closed_by_packet")
        backend_port_prestart_gate = detail.get("backend_port_prestart_gate")
        expected_backend_port_prestart_gate = detail.get(
            "expected_backend_port_prestart_gate"
        )
        declared_expected_backend_port_prestart_gate = detail.get(
            "declared_expected_backend_port_prestart_gate"
        )
        supervised_prestart_gate = detail.get("supervised_prestart_gate")
        expected_supervised_prestart_gate = detail.get("expected_supervised_prestart_gate")
        mvp_blocker_anchor = detail.get("mvp_blocker_anchor")
        expected_mvp_blocker_anchor = detail.get("expected_mvp_blocker_anchor")
        return {
            "status": result.status,
            "read_only": detail.get("read_only") is True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "operator_required": detail.get("operator_required") is True,
            "expected_plan_required_for_real_run": (
                detail.get("expected_plan_required_for_real_run") is True
            ),
            "physical_preflight_requires_existing_backend_connection": (
                detail.get("physical_preflight_requires_existing_backend_connection")
                is True
            ),
            "backend_port_prestart_check_required": (
                detail.get("backend_port_prestart_check_required") is True
            ),
            "backend_port_prestart_check_must_pass_before_starting_backend": (
                detail.get("backend_port_prestart_check_must_pass_before_starting_backend")
                is True
            ),
            "backend_port_prestart_check_operator_sequence_step_index": (
                detail.get("backend_port_prestart_check_operator_sequence_step_index")
            ),
            "backend_port_prestart_check_must_be_first_operator_step": (
                detail.get("backend_port_prestart_check_must_be_first_operator_step")
                is True
            ),
            "backend_port_prestart_check_operator_sequence_step": str(
                detail.get("backend_port_prestart_check_operator_sequence_step") or ""
            ),
            "backend_port_prestart_check_operator_sequence_step_is_first": (
                detail.get("backend_port_prestart_check_operator_sequence_step_is_first")
                is True
            ),
            "backend_port_prestart_gate": (
                dict(backend_port_prestart_gate)
                if isinstance(backend_port_prestart_gate, Mapping)
                else {}
            ),
            "expected_backend_port_prestart_gate": (
                dict(expected_backend_port_prestart_gate)
                if isinstance(expected_backend_port_prestart_gate, Mapping)
                else {}
            ),
            "declared_expected_backend_port_prestart_gate": (
                dict(declared_expected_backend_port_prestart_gate)
                if isinstance(declared_expected_backend_port_prestart_gate, Mapping)
                else {}
            ),
            "backend_port_prestart_check_failure_detail_fields": (
                list(detail.get("backend_port_prestart_check_failure_detail_fields"))
                if isinstance(
                    detail.get("backend_port_prestart_check_failure_detail_fields"),
                    list,
                )
                else []
            ),
            "manual_review_required_before_readiness_update": (
                detail.get("manual_review_required_before_readiness_update") is True
            ),
            "execution_allowed_by_packet": False,
            "readiness_matrix_update_allowed_by_packet": False,
            "readiness_closure_allowed_by_packet": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "next_manual_artifact_required": str(
                detail.get("next_manual_artifact_required") or ""
            ),
            "packet_path": detail.get("packet_path"),
            "packet_type": detail.get("packet_type"),
            "packet_status": detail.get("packet_status"),
            "plan_artifact_status": detail.get("plan_artifact_status"),
            "plan_artifact_violations": (
                list(detail.get("plan_artifact_violations"))
                if isinstance(detail.get("plan_artifact_violations"), list)
                else []
            ),
            "plan_output": detail.get("plan_output"),
            "expected_plan_output": detail.get("expected_plan_output"),
            "channels": (
                list(detail.get("channels"))
                if isinstance(detail.get("channels"), list)
                else []
            ),
            "value_types": (
                list(detail.get("value_types"))
                if isinstance(detail.get("value_types"), list)
                else []
            ),
            "command_names": list(command_names) if isinstance(command_names, list) else [],
            "preview_command_uses_plan_only": (
                detail.get("preview_command_uses_plan_only") is True
            ),
            "capture_command_requires_physical_readback": (
                detail.get("capture_command_requires_physical_readback") is True
            ),
            "prestart_backend_port_check_command": (
                list(detail.get("prestart_backend_port_check_command"))
                if isinstance(detail.get("prestart_backend_port_check_command"), list)
                else []
            ),
            "review_command_writes_decision_template": (
                detail.get("review_command_writes_decision_template") is True
            ),
            "required_review_checklist_ids": (
                list(checklist_ids) if isinstance(checklist_ids, list) else []
            ),
            "forbidden_live_request_types": (
                list(forbidden_types) if isinstance(forbidden_types, list) else []
            ),
            "blockers_not_closed_by_packet": (
                list(blockers) if isinstance(blockers, list) else []
            ),
            "mvp_blocker_anchor": (
                dict(mvp_blocker_anchor)
                if isinstance(mvp_blocker_anchor, Mapping)
                else {}
            ),
            "expected_mvp_blocker_anchor": (
                dict(expected_mvp_blocker_anchor)
                if isinstance(expected_mvp_blocker_anchor, Mapping)
                else {}
            ),
            "mvp_blocker_anchor_matches_current_first_open": (
                detail.get("mvp_blocker_anchor_matches_current_first_open") is True
            ),
            "mvp_blocker_anchor_sha256": str(
                detail.get("mvp_blocker_anchor_sha256") or ""
            ),
            "actual_mvp_blocker_anchor_sha256": str(
                detail.get("actual_mvp_blocker_anchor_sha256") or ""
            ),
            "expected_mvp_blocker_anchor_sha256": str(
                detail.get("expected_mvp_blocker_anchor_sha256") or ""
            ),
            "mvp_blocker_anchor_sha256_matches_anchor": (
                detail.get("mvp_blocker_anchor_sha256_matches_anchor") is True
            ),
            "mvp_blocker_anchor_sha256_matches_current_first_open": (
                detail.get("mvp_blocker_anchor_sha256_matches_current_first_open") is True
            ),
            "operator_sequence_count": int(detail.get("operator_sequence_count") or 0),
            "operator_sequence_has_mvp_blocker_anchor_digest_review": (
                detail.get("operator_sequence_has_mvp_blocker_anchor_digest_review")
                is True
            ),
            "operator_sequence_mvp_blocker_anchor_digest_review_step": (
                str(
                    detail.get(
                        "operator_sequence_mvp_blocker_anchor_digest_review_step"
                    )
                    or ""
                )
            ),
            "operator_sequence_mvp_blocker_anchor_digest_review_step_index": (
                detail.get("operator_sequence_mvp_blocker_anchor_digest_review_step_index")
            ),
            "operator_sequence_has_supervised_prestart_gate": (
                detail.get("operator_sequence_has_supervised_prestart_gate") is True
            ),
            "operator_sequence_supervised_prestart_gate_step": (
                str(detail.get("operator_sequence_supervised_prestart_gate_step") or "")
            ),
            "operator_sequence_supervised_prestart_gate_step_index": (
                detail.get("operator_sequence_supervised_prestart_gate_step_index")
            ),
            "operator_sequence_supervised_prestart_gate_is_final": (
                detail.get("operator_sequence_supervised_prestart_gate_is_final") is True
            ),
            "required_supervised_prestart_gate_markers": (
                list(detail.get("required_supervised_prestart_gate_markers"))
                if isinstance(
                    detail.get("required_supervised_prestart_gate_markers"),
                    list,
                )
                else []
            ),
            "supervised_prestart_gate": (
                dict(supervised_prestart_gate)
                if isinstance(supervised_prestart_gate, Mapping)
                else {}
            ),
            "expected_supervised_prestart_gate": (
                dict(expected_supervised_prestart_gate)
                if isinstance(expected_supervised_prestart_gate, Mapping)
                else {}
            ),
            "violations": list(violations) if isinstance(violations, list) else [],
        }
    return None


def wing_readonly_preflight_review_decision_summary(
    results: Sequence[CheckResult],
) -> dict[str, object] | None:
    for result in results:
        if result.name != "wing_readonly_preflight_review_decision_ledger_schema":
            continue
        detail = result.detail if isinstance(result.detail, Mapping) else {}
        blocked_by = detail.get("blocked_by")
        violations = detail.get("violations")
        accepted_source_ids = detail.get("accepted_source_ids")
        accepted_candidate_prestart_source_ids = detail.get(
            "accepted_candidate_prestart_source_ids"
        )
        rejected_source_ids = detail.get("rejected_source_ids")
        invalid_record_ids = detail.get("invalid_record_ids")
        duplicate_source_ids = detail.get("duplicate_source_ids")
        conflicting_source_ids = detail.get("conflicting_source_ids")
        unreviewed_candidate_source_ids = detail.get("unreviewed_candidate_source_ids")
        candidate_reviews = detail.get("candidate_reviews")
        decision_search_roots = detail.get("decision_search_roots")
        review_command = detail.get("review_command")
        review_exit_codes = detail.get("review_exit_codes")
        return {
            "status": result.status,
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "operator_required": True,
            "execution_allowed_by_decision": False,
            "readiness_matrix_update_allowed_by_decision": False,
            "readiness_closure_allowed_by_decision": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "ledger_status": detail.get("ledger_status"),
            "review_tool_path": detail.get("review_tool_path"),
            "review_tool_available": detail.get("review_tool_available") is True,
            "review_command": (
                list(review_command) if isinstance(review_command, list) else []
            ),
            "review_exit_codes": (
                dict(review_exit_codes) if isinstance(review_exit_codes, Mapping) else {}
            ),
            "preflight_evidence_status": detail.get("preflight_evidence_status"),
            "decision_search_roots": (
                list(decision_search_roots)
                if isinstance(decision_search_roots, list)
                else []
            ),
            "review_candidate_count": int(detail.get("review_candidate_count") or 0),
            "decision_record_count": int(detail.get("decision_record_count") or 0),
            "accepted_record_count": int(detail.get("accepted_record_count") or 0),
            "rejected_record_count": int(detail.get("rejected_record_count") or 0),
            "invalid_record_count": int(detail.get("invalid_record_count") or 0),
            "unreviewed_candidate_count": int(detail.get("unreviewed_candidate_count") or 0),
            "manual_readiness_update_candidate_available": (
                detail.get("manual_readiness_update_candidate_available") is True
            ),
            "operator_review_required": detail.get("operator_review_required") is True,
            "accepted_source_ids": (
                list(accepted_source_ids)
                if isinstance(accepted_source_ids, list)
                else []
            ),
            "accepted_candidate_prestart_contract_valid": (
                detail.get("accepted_candidate_prestart_contract_valid") is True
            ),
            "accepted_candidate_prestart_source_ids": (
                list(accepted_candidate_prestart_source_ids)
                if isinstance(accepted_candidate_prestart_source_ids, list)
                else []
            ),
            "rejected_source_ids": (
                list(rejected_source_ids)
                if isinstance(rejected_source_ids, list)
                else []
            ),
            "invalid_record_ids": (
                list(invalid_record_ids)
                if isinstance(invalid_record_ids, list)
                else []
            ),
            "duplicate_source_ids": (
                list(duplicate_source_ids)
                if isinstance(duplicate_source_ids, list)
                else []
            ),
            "conflicting_source_ids": (
                list(conflicting_source_ids)
                if isinstance(conflicting_source_ids, list)
                else []
            ),
            "unreviewed_candidate_source_ids": (
                list(unreviewed_candidate_source_ids)
                if isinstance(unreviewed_candidate_source_ids, list)
                else []
            ),
            "candidate_reviews": (
                list(candidate_reviews) if isinstance(candidate_reviews, list) else []
            ),
            "blocked_by": list(blocked_by) if isinstance(blocked_by, list) else [],
            "violations": list(violations) if isinstance(violations, list) else [],
        }
    return None


def raw_mixer_readback_contract_summary(results: Sequence[CheckResult]) -> dict[str, object] | None:
    for result in results:
        if result.name != "raw_mixer_state_physical_readback_contract_schema":
            continue
        detail = result.detail if isinstance(result.detail, Mapping) else {}
        allowed_sources = detail.get("allowed_sources")
        accepted_statuses = detail.get("accepted_row_statuses")
        forbidden_mutation_fields = detail.get("forbidden_mutation_evidence_fields")
        required_false_flags = detail.get("required_false_safety_flags")
        violations = detail.get("violations")
        return {
            "status": result.status,
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "live_widening_allowed": False,
            "full_autonomous_ready": False,
            "schema_id": detail.get("schema_id"),
            "contract_status": detail.get("contract_status"),
            "surface_id": detail.get("surface_id"),
            "remediation_action_id": detail.get("remediation_action_id"),
            "next_physical_readback_surface_id": detail.get("next_physical_readback_surface_id"),
            "next_physical_readback_action_id": detail.get("next_physical_readback_action_id"),
            "fixture_next_physical_readback_surface_id": detail.get(
                "fixture_next_physical_readback_surface_id"
            ),
            "remediation_next_physical_readback_surface_id": detail.get(
                "remediation_next_physical_readback_surface_id"
            ),
            "remediation_next_physical_readback_action_id": detail.get(
                "remediation_next_physical_readback_action_id"
            ),
            "artifact_required": detail.get("artifact_required") is True,
            "physical_fixture_gap_remains": detail.get("physical_fixture_gap_remains") is True,
            "accepted_as_physical_fixture_candidate": (
                detail.get("accepted_as_physical_fixture_candidate") is True
            ),
            "live_authority_granted": detail.get("live_authority_granted") is True,
            "required_top_level_field_count": int(detail.get("required_top_level_field_count") or 0),
            "required_row_field_count": int(detail.get("required_row_field_count") or 0),
            "required_false_safety_flags": (
                list(required_false_flags) if isinstance(required_false_flags, list) else []
            ),
            "allowed_sources": list(allowed_sources) if isinstance(allowed_sources, list) else [],
            "accepted_row_statuses": (
                list(accepted_statuses) if isinstance(accepted_statuses, list) else []
            ),
            "forbidden_mutation_evidence_fields": (
                list(forbidden_mutation_fields)
                if isinstance(forbidden_mutation_fields, list)
                else []
            ),
            "target_consistency": (
                detail.get("next_physical_readback_surface_id")
                == detail.get("fixture_next_physical_readback_surface_id")
                == detail.get("remediation_next_physical_readback_surface_id")
                and detail.get("next_physical_readback_action_id")
                == detail.get("remediation_next_physical_readback_action_id")
            ),
            "violations": list(violations) if isinstance(violations, list) else [],
        }
    return None


def raw_mixer_readback_artifact_intake_summary(
    results: Sequence[CheckResult],
) -> dict[str, object] | None:
    for result in results:
        if result.name != "raw_mixer_state_physical_readback_artifact_intake_schema":
            continue
        detail = result.detail if isinstance(result.detail, Mapping) else {}
        artifact_search_roots = detail.get("artifact_search_roots")
        blocked_by = detail.get("blocked_by")
        artifact_source_ids = detail.get("artifact_source_ids")
        intake_violations = detail.get("intake_violations")
        violations = detail.get("violations")
        return {
            "status": result.status,
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "operator_required": True,
            "execution_allowed_by_intake": False,
            "fixture_closure_allowed_by_intake": False,
            "live_widening_allowed": False,
            "full_autonomous_ready": False,
            "schema_id": detail.get("schema_id"),
            "intake_status": detail.get("intake_status"),
            "surface_id": detail.get("surface_id"),
            "remediation_action_id": detail.get("remediation_action_id"),
            "required_contract_method_id": detail.get("required_contract_method_id"),
            "artifact_search_roots": (
                list(artifact_search_roots) if isinstance(artifact_search_roots, list) else []
            ),
            "artifact_count": int(detail.get("artifact_count") or 0),
            "valid_candidate_count": int(detail.get("valid_candidate_count") or 0),
            "invalid_candidate_count": int(detail.get("invalid_candidate_count") or 0),
            "accepted_as_physical_fixture_candidate": (
                detail.get("accepted_as_physical_fixture_candidate") is True
            ),
            "operator_review_required": detail.get("operator_review_required") is True,
            "physical_fixture_gap_remains": detail.get("physical_fixture_gap_remains") is True,
            "live_authority_granted": detail.get("live_authority_granted") is True,
            "latest_valid_candidate_source_id": detail.get("latest_valid_candidate_source_id"),
            "latest_valid_candidate_captured_at": detail.get("latest_valid_candidate_captured_at"),
            "blocked_by": list(blocked_by) if isinstance(blocked_by, list) else [],
            "artifact_source_ids": (
                list(artifact_source_ids) if isinstance(artifact_source_ids, list) else []
            ),
            "intake_violations": (
                list(intake_violations) if isinstance(intake_violations, list) else []
            ),
            "violations": list(violations) if isinstance(violations, list) else [],
        }
    return None


def channel_basic_controls_readback_artifact_intake_summary(
    results: Sequence[CheckResult],
) -> dict[str, object] | None:
    for result in results:
        if result.name != "channel_basic_controls_physical_readback_artifact_intake_schema":
            continue
        detail = result.detail if isinstance(result.detail, Mapping) else {}
        artifact_search_roots = detail.get("artifact_search_roots")
        blocked_by = detail.get("blocked_by")
        artifact_source_ids = detail.get("artifact_source_ids")
        intake_violations = detail.get("intake_violations")
        promotion_families = detail.get("promotion_families_requiring_surface")
        violations = detail.get("violations")
        source_candidate_claimed = detail.get("accepted_as_physical_fixture_candidate") is True
        operator_review_required = detail.get("operator_review_required") is True
        physical_fixture_gap_remains = detail.get("physical_fixture_gap_remains") is True
        unsafe_claimed = (
            detail.get("fixture_closure_allowed_by_intake") is True
            or detail.get("live_authority_granted") is True
            or detail.get("full_autonomous_ready") is True
        )
        candidate_accepted = (
            result.status == "passed"
            and source_candidate_claimed
            and operator_review_required
            and physical_fixture_gap_remains
            and not unsafe_claimed
        )
        return {
            "status": result.status,
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "operator_required": True,
            "execution_allowed_by_intake": False,
            "fixture_closure_allowed_by_intake": False,
            "live_widening_allowed": False,
            "full_autonomous_ready": False,
            "schema_id": detail.get("schema_id"),
            "intake_status": detail.get("intake_status"),
            "surface_id": detail.get("surface_id"),
            "remediation_action_id": detail.get("remediation_action_id"),
            "required_contract_method_id": detail.get("required_contract_method_id"),
            "artifact_search_roots": (
                list(artifact_search_roots) if isinstance(artifact_search_roots, list) else []
            ),
            "artifact_count": int(detail.get("artifact_count") or 0),
            "valid_candidate_count": int(detail.get("valid_candidate_count") or 0),
            "invalid_candidate_count": int(detail.get("invalid_candidate_count") or 0),
            "valid_channel_count": int(detail.get("valid_channel_count") or 0),
            "accepted_as_physical_fixture_candidate": candidate_accepted,
            "operator_review_required": True,
            "physical_fixture_gap_remains": True,
            "live_authority_granted": False,
            "promotion_families_requiring_surface": (
                list(promotion_families) if isinstance(promotion_families, list) else []
            ),
            "latest_valid_candidate_source_id": detail.get("latest_valid_candidate_source_id"),
            "latest_valid_candidate_captured_at": detail.get("latest_valid_candidate_captured_at"),
            "blocked_by": list(blocked_by) if isinstance(blocked_by, list) else [],
            "artifact_source_ids": (
                list(artifact_source_ids) if isinstance(artifact_source_ids, list) else []
            ),
            "intake_violations": (
                list(intake_violations) if isinstance(intake_violations, list) else []
            ),
            "violations": list(violations) if isinstance(violations, list) else [],
        }
    return None


def channel_basic_controls_readback_contract_summary(
    results: Sequence[CheckResult],
) -> dict[str, object] | None:
    for result in results:
        if result.name != "channel_basic_controls_physical_readback_contract_schema":
            continue
        detail = result.detail if isinstance(result.detail, Mapping) else {}
        allowed_sources = detail.get("allowed_sources")
        accepted_statuses = detail.get("accepted_channel_statuses")
        forbidden_mutation_fields = detail.get("forbidden_mutation_evidence_fields")
        required_false_flags = detail.get("required_false_safety_flags")
        promotion_families = detail.get("promotion_families_requiring_surface")
        required_identity_fields = detail.get("required_identity_fields")
        required_control_fields = detail.get("required_control_fields")
        required_coverage_flags = detail.get("required_coverage_flags")
        violations = detail.get("violations")
        return {
            "status": result.status,
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "live_widening_allowed": False,
            "full_autonomous_ready": False,
            "schema_id": detail.get("schema_id"),
            "contract_status": detail.get("contract_status"),
            "surface_id": detail.get("surface_id"),
            "remediation_action_id": detail.get("remediation_action_id"),
            "physical_readback_surface_id": detail.get("physical_readback_surface_id"),
            "physical_readback_action_id": detail.get("physical_readback_action_id"),
            "fixture_physical_gap_declared": detail.get("fixture_physical_gap_declared") is True,
            "remediation_physical_gap_declared": (
                detail.get("remediation_physical_gap_declared") is True
            ),
            "artifact_required": detail.get("artifact_required") is True,
            "physical_fixture_gap_remains": detail.get("physical_fixture_gap_remains") is True,
            "accepted_as_physical_fixture_candidate": (
                detail.get("accepted_as_physical_fixture_candidate") is True
            ),
            "live_authority_granted": detail.get("live_authority_granted") is True,
            "promotion_families_requiring_surface": (
                list(promotion_families) if isinstance(promotion_families, list) else []
            ),
            "required_top_level_field_count": int(detail.get("required_top_level_field_count") or 0),
            "required_channel_field_count": int(detail.get("required_channel_field_count") or 0),
            "required_identity_fields": (
                list(required_identity_fields) if isinstance(required_identity_fields, list) else []
            ),
            "required_control_fields": (
                list(required_control_fields) if isinstance(required_control_fields, list) else []
            ),
            "required_coverage_flags": (
                list(required_coverage_flags) if isinstance(required_coverage_flags, list) else []
            ),
            "required_false_safety_flags": (
                list(required_false_flags) if isinstance(required_false_flags, list) else []
            ),
            "allowed_sources": list(allowed_sources) if isinstance(allowed_sources, list) else [],
            "accepted_channel_statuses": (
                list(accepted_statuses) if isinstance(accepted_statuses, list) else []
            ),
            "forbidden_mutation_evidence_fields": (
                list(forbidden_mutation_fields)
                if isinstance(forbidden_mutation_fields, list)
                else []
            ),
            "target_consistency": (
                detail.get("physical_readback_surface_id") == detail.get("surface_id")
                and detail.get("physical_readback_action_id")
                == detail.get("remediation_action_id")
                and detail.get("fixture_physical_gap_declared") is True
                and detail.get("remediation_physical_gap_declared") is True
            ),
            "violations": list(violations) if isinstance(violations, list) else [],
        }
    return None


def channel_processing_readback_contract_summary(
    results: Sequence[CheckResult],
) -> dict[str, object] | None:
    for result in results:
        if result.name != "channel_processing_physical_readback_contract_schema":
            continue
        detail = result.detail if isinstance(result.detail, Mapping) else {}
        allowed_sources = detail.get("allowed_sources")
        accepted_statuses = detail.get("accepted_channel_statuses")
        forbidden_mutation_fields = detail.get("forbidden_mutation_evidence_fields")
        required_false_flags = detail.get("required_false_safety_flags")
        promotion_families = detail.get("promotion_families_requiring_surface")
        required_hpf_fields = detail.get("required_hpf_fields")
        required_eq_fields = detail.get("required_eq_fields")
        required_eq_band_fields = detail.get("required_eq_band_fields")
        required_compressor_fields = detail.get("required_compressor_fields")
        required_gate_fields = detail.get("required_gate_fields")
        required_coverage_flags = detail.get("required_coverage_flags")
        violations = detail.get("violations")
        return {
            "status": result.status,
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "live_widening_allowed": False,
            "full_autonomous_ready": False,
            "schema_id": detail.get("schema_id"),
            "contract_status": detail.get("contract_status"),
            "surface_id": detail.get("surface_id"),
            "remediation_action_id": detail.get("remediation_action_id"),
            "physical_readback_surface_id": detail.get("physical_readback_surface_id"),
            "physical_readback_action_id": detail.get("physical_readback_action_id"),
            "fixture_physical_gap_declared": detail.get("fixture_physical_gap_declared") is True,
            "remediation_physical_gap_declared": (
                detail.get("remediation_physical_gap_declared") is True
            ),
            "artifact_required": detail.get("artifact_required") is True,
            "physical_fixture_gap_remains": detail.get("physical_fixture_gap_remains") is True,
            "accepted_as_physical_fixture_candidate": (
                detail.get("accepted_as_physical_fixture_candidate") is True
            ),
            "live_authority_granted": detail.get("live_authority_granted") is True,
            "promotion_families_requiring_surface": (
                list(promotion_families) if isinstance(promotion_families, list) else []
            ),
            "required_top_level_field_count": int(detail.get("required_top_level_field_count") or 0),
            "required_channel_field_count": int(detail.get("required_channel_field_count") or 0),
            "required_hpf_fields": (
                list(required_hpf_fields) if isinstance(required_hpf_fields, list) else []
            ),
            "required_eq_fields": (
                list(required_eq_fields) if isinstance(required_eq_fields, list) else []
            ),
            "required_eq_band_fields": (
                list(required_eq_band_fields) if isinstance(required_eq_band_fields, list) else []
            ),
            "required_compressor_fields": (
                list(required_compressor_fields)
                if isinstance(required_compressor_fields, list)
                else []
            ),
            "required_gate_fields": (
                list(required_gate_fields) if isinstance(required_gate_fields, list) else []
            ),
            "required_coverage_flags": (
                list(required_coverage_flags) if isinstance(required_coverage_flags, list) else []
            ),
            "required_false_safety_flags": (
                list(required_false_flags) if isinstance(required_false_flags, list) else []
            ),
            "minimum_eq_band_count": int(detail.get("minimum_eq_band_count") or 0),
            "allowed_sources": list(allowed_sources) if isinstance(allowed_sources, list) else [],
            "accepted_channel_statuses": (
                list(accepted_statuses) if isinstance(accepted_statuses, list) else []
            ),
            "forbidden_mutation_evidence_fields": (
                list(forbidden_mutation_fields)
                if isinstance(forbidden_mutation_fields, list)
                else []
            ),
            "target_consistency": (
                detail.get("physical_readback_surface_id") == detail.get("surface_id")
                and detail.get("physical_readback_action_id")
                == detail.get("remediation_action_id")
                and detail.get("fixture_physical_gap_declared") is True
                and detail.get("remediation_physical_gap_declared") is True
            ),
            "violations": list(violations) if isinstance(violations, list) else [],
        }
    return None


def channel_processing_readback_artifact_intake_summary(
    results: Sequence[CheckResult],
) -> dict[str, object] | None:
    for result in results:
        if result.name != "channel_processing_physical_readback_artifact_intake_schema":
            continue
        detail = result.detail if isinstance(result.detail, Mapping) else {}
        artifact_search_roots = detail.get("artifact_search_roots")
        blocked_by = detail.get("blocked_by")
        artifact_source_ids = detail.get("artifact_source_ids")
        intake_violations = detail.get("intake_violations")
        promotion_families = detail.get("promotion_families_requiring_surface")
        violations = detail.get("violations")
        source_candidate_claimed = detail.get("accepted_as_physical_fixture_candidate") is True
        operator_review_required = detail.get("operator_review_required") is True
        physical_fixture_gap_remains = detail.get("physical_fixture_gap_remains") is True
        unsafe_claimed = (
            detail.get("fixture_closure_allowed_by_intake") is True
            or detail.get("live_authority_granted") is True
            or detail.get("full_autonomous_ready") is True
        )
        candidate_accepted = (
            result.status == "passed"
            and source_candidate_claimed
            and operator_review_required
            and physical_fixture_gap_remains
            and not unsafe_claimed
        )
        return {
            "status": result.status,
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "operator_required": True,
            "execution_allowed_by_intake": False,
            "fixture_closure_allowed_by_intake": False,
            "live_widening_allowed": False,
            "full_autonomous_ready": False,
            "schema_id": detail.get("schema_id"),
            "intake_status": detail.get("intake_status"),
            "surface_id": detail.get("surface_id"),
            "remediation_action_id": detail.get("remediation_action_id"),
            "required_contract_method_id": detail.get("required_contract_method_id"),
            "artifact_search_roots": (
                list(artifact_search_roots) if isinstance(artifact_search_roots, list) else []
            ),
            "artifact_count": int(detail.get("artifact_count") or 0),
            "valid_candidate_count": int(detail.get("valid_candidate_count") or 0),
            "invalid_candidate_count": int(detail.get("invalid_candidate_count") or 0),
            "valid_channel_count": int(detail.get("valid_channel_count") or 0),
            "accepted_as_physical_fixture_candidate": candidate_accepted,
            "operator_review_required": True,
            "physical_fixture_gap_remains": True,
            "live_authority_granted": False,
            "promotion_families_requiring_surface": (
                list(promotion_families) if isinstance(promotion_families, list) else []
            ),
            "latest_valid_candidate_source_id": detail.get("latest_valid_candidate_source_id"),
            "latest_valid_candidate_captured_at": detail.get("latest_valid_candidate_captured_at"),
            "blocked_by": list(blocked_by) if isinstance(blocked_by, list) else [],
            "artifact_source_ids": (
                list(artifact_source_ids) if isinstance(artifact_source_ids, list) else []
            ),
            "intake_violations": (
                list(intake_violations) if isinstance(intake_violations, list) else []
            ),
            "violations": list(violations) if isinstance(violations, list) else [],
        }
    return None


def bus_group_main_dca_readback_contract_summary(
    results: Sequence[CheckResult],
) -> dict[str, object] | None:
    for result in results:
        if result.name != "bus_group_main_dca_physical_readback_contract_schema":
            continue
        detail = result.detail if isinstance(result.detail, Mapping) else {}
        allowed_sources = detail.get("allowed_sources")
        accepted_statuses = detail.get("accepted_surface_statuses")
        forbidden_mutation_fields = detail.get("forbidden_mutation_evidence_fields")
        required_false_flags = detail.get("required_false_safety_flags")
        promotion_families = detail.get("promotion_families_requiring_surface")
        required_bus_control_fields = detail.get("required_bus_control_fields")
        required_main_control_fields = detail.get("required_main_control_fields")
        required_dca_control_fields = detail.get("required_dca_control_fields")
        required_bus_assignment_fields = detail.get("required_bus_assignment_fields")
        required_dca_assignment_fields = detail.get("required_dca_assignment_fields")
        required_metering_fields = detail.get("required_metering_fields")
        required_bus_coverage_flags = detail.get("required_bus_coverage_flags")
        required_main_coverage_flags = detail.get("required_main_coverage_flags")
        required_dca_coverage_flags = detail.get("required_dca_coverage_flags")
        violations = detail.get("violations")
        return {
            "status": result.status,
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "live_widening_allowed": False,
            "full_autonomous_ready": False,
            "schema_id": detail.get("schema_id"),
            "contract_status": detail.get("contract_status"),
            "surface_id": detail.get("surface_id"),
            "remediation_action_id": detail.get("remediation_action_id"),
            "physical_readback_surface_id": detail.get("physical_readback_surface_id"),
            "physical_readback_action_id": detail.get("physical_readback_action_id"),
            "fixture_physical_gap_declared": detail.get("fixture_physical_gap_declared") is True,
            "remediation_physical_gap_declared": (
                detail.get("remediation_physical_gap_declared") is True
            ),
            "artifact_required": detail.get("artifact_required") is True,
            "physical_fixture_gap_remains": detail.get("physical_fixture_gap_remains") is True,
            "accepted_as_physical_fixture_candidate": (
                detail.get("accepted_as_physical_fixture_candidate") is True
            ),
            "live_authority_granted": detail.get("live_authority_granted") is True,
            "promotion_families_requiring_surface": (
                list(promotion_families) if isinstance(promotion_families, list) else []
            ),
            "required_top_level_field_count": int(detail.get("required_top_level_field_count") or 0),
            "required_bus_field_count": int(detail.get("required_bus_field_count") or 0),
            "required_main_field_count": int(detail.get("required_main_field_count") or 0),
            "required_dca_field_count": int(detail.get("required_dca_field_count") or 0),
            "required_bus_control_fields": (
                list(required_bus_control_fields)
                if isinstance(required_bus_control_fields, list)
                else []
            ),
            "required_main_control_fields": (
                list(required_main_control_fields)
                if isinstance(required_main_control_fields, list)
                else []
            ),
            "required_dca_control_fields": (
                list(required_dca_control_fields)
                if isinstance(required_dca_control_fields, list)
                else []
            ),
            "required_bus_assignment_fields": (
                list(required_bus_assignment_fields)
                if isinstance(required_bus_assignment_fields, list)
                else []
            ),
            "required_dca_assignment_fields": (
                list(required_dca_assignment_fields)
                if isinstance(required_dca_assignment_fields, list)
                else []
            ),
            "required_metering_fields": (
                list(required_metering_fields)
                if isinstance(required_metering_fields, list)
                else []
            ),
            "required_bus_coverage_flags": (
                list(required_bus_coverage_flags)
                if isinstance(required_bus_coverage_flags, list)
                else []
            ),
            "required_main_coverage_flags": (
                list(required_main_coverage_flags)
                if isinstance(required_main_coverage_flags, list)
                else []
            ),
            "required_dca_coverage_flags": (
                list(required_dca_coverage_flags)
                if isinstance(required_dca_coverage_flags, list)
                else []
            ),
            "required_false_safety_flags": (
                list(required_false_flags) if isinstance(required_false_flags, list) else []
            ),
            "minimum_bus_count": int(detail.get("minimum_bus_count") or 0),
            "minimum_main_count": int(detail.get("minimum_main_count") or 0),
            "minimum_dca_count": int(detail.get("minimum_dca_count") or 0),
            "allowed_sources": list(allowed_sources) if isinstance(allowed_sources, list) else [],
            "accepted_surface_statuses": (
                list(accepted_statuses) if isinstance(accepted_statuses, list) else []
            ),
            "forbidden_mutation_evidence_fields": (
                list(forbidden_mutation_fields)
                if isinstance(forbidden_mutation_fields, list)
                else []
            ),
            "target_consistency": (
                detail.get("physical_readback_surface_id") == detail.get("surface_id")
                and detail.get("physical_readback_action_id")
                == detail.get("remediation_action_id")
                and detail.get("fixture_physical_gap_declared") is True
                and detail.get("remediation_physical_gap_declared") is True
            ),
            "violations": list(violations) if isinstance(violations, list) else [],
        }
    return None


def bus_group_main_dca_readback_artifact_intake_summary(
    results: Sequence[CheckResult],
) -> dict[str, object] | None:
    for result in results:
        if result.name != "bus_group_main_dca_physical_readback_artifact_intake_schema":
            continue
        detail = result.detail if isinstance(result.detail, Mapping) else {}
        artifact_search_roots = detail.get("artifact_search_roots")
        blocked_by = detail.get("blocked_by")
        artifact_source_ids = detail.get("artifact_source_ids")
        intake_violations = detail.get("intake_violations")
        promotion_families = detail.get("promotion_families_requiring_surface")
        violations = detail.get("violations")
        source_candidate_claimed = detail.get("accepted_as_physical_fixture_candidate") is True
        operator_review_required = detail.get("operator_review_required") is True
        physical_fixture_gap_remains = detail.get("physical_fixture_gap_remains") is True
        unsafe_claimed = (
            detail.get("fixture_closure_allowed_by_intake") is True
            or detail.get("live_authority_granted") is True
            or detail.get("full_autonomous_ready") is True
        )
        candidate_accepted = (
            result.status == "passed"
            and source_candidate_claimed
            and operator_review_required
            and physical_fixture_gap_remains
            and not unsafe_claimed
        )
        return {
            "status": result.status,
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "operator_required": True,
            "execution_allowed_by_intake": False,
            "fixture_closure_allowed_by_intake": False,
            "live_widening_allowed": False,
            "full_autonomous_ready": False,
            "schema_id": detail.get("schema_id"),
            "intake_status": detail.get("intake_status"),
            "surface_id": detail.get("surface_id"),
            "remediation_action_id": detail.get("remediation_action_id"),
            "required_contract_method_id": detail.get("required_contract_method_id"),
            "artifact_search_roots": (
                list(artifact_search_roots) if isinstance(artifact_search_roots, list) else []
            ),
            "artifact_count": int(detail.get("artifact_count") or 0),
            "valid_candidate_count": int(detail.get("valid_candidate_count") or 0),
            "invalid_candidate_count": int(detail.get("invalid_candidate_count") or 0),
            "valid_bus_count": int(detail.get("valid_bus_count") or 0),
            "valid_main_count": int(detail.get("valid_main_count") or 0),
            "valid_dca_count": int(detail.get("valid_dca_count") or 0),
            "accepted_as_physical_fixture_candidate": candidate_accepted,
            "operator_review_required": True,
            "physical_fixture_gap_remains": True,
            "live_authority_granted": False,
            "promotion_families_requiring_surface": (
                list(promotion_families) if isinstance(promotion_families, list) else []
            ),
            "latest_valid_candidate_source_id": detail.get("latest_valid_candidate_source_id"),
            "latest_valid_candidate_captured_at": detail.get("latest_valid_candidate_captured_at"),
            "blocked_by": list(blocked_by) if isinstance(blocked_by, list) else [],
            "artifact_source_ids": (
                list(artifact_source_ids) if isinstance(artifact_source_ids, list) else []
            ),
            "intake_violations": (
                list(intake_violations) if isinstance(intake_violations, list) else []
            ),
            "violations": list(violations) if isinstance(violations, list) else [],
        }
    return None


def matrix_state_readback_contract_summary(
    results: Sequence[CheckResult],
) -> dict[str, object] | None:
    for result in results:
        if result.name != "matrix_state_physical_readback_contract_schema":
            continue
        detail = result.detail if isinstance(result.detail, Mapping) else {}
        allowed_sources = detail.get("allowed_sources")
        accepted_statuses = detail.get("accepted_matrix_statuses")
        forbidden_mutation_fields = detail.get("forbidden_mutation_evidence_fields")
        required_false_flags = detail.get("required_false_safety_flags")
        promotion_families = detail.get("promotion_families_requiring_surface")
        required_control_fields = detail.get("required_control_fields")
        required_eq_fields = detail.get("required_eq_fields")
        required_eq_band_fields = detail.get("required_eq_band_fields")
        required_dynamics_fields = detail.get("required_dynamics_fields")
        required_insert_fields = detail.get("required_insert_fields")
        required_metering_fields = detail.get("required_metering_fields")
        required_coverage_flags = detail.get("required_coverage_flags")
        violations = detail.get("violations")
        return {
            "status": result.status,
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "live_widening_allowed": False,
            "full_autonomous_ready": False,
            "schema_id": detail.get("schema_id"),
            "contract_status": detail.get("contract_status"),
            "surface_id": detail.get("surface_id"),
            "remediation_action_id": detail.get("remediation_action_id"),
            "physical_readback_surface_id": detail.get("physical_readback_surface_id"),
            "physical_readback_action_id": detail.get("physical_readback_action_id"),
            "fixture_physical_gap_declared": detail.get("fixture_physical_gap_declared") is True,
            "remediation_physical_gap_declared": (
                detail.get("remediation_physical_gap_declared") is True
            ),
            "artifact_required": detail.get("artifact_required") is True,
            "physical_fixture_gap_remains": detail.get("physical_fixture_gap_remains") is True,
            "accepted_as_physical_fixture_candidate": (
                detail.get("accepted_as_physical_fixture_candidate") is True
            ),
            "live_authority_granted": detail.get("live_authority_granted") is True,
            "promotion_families_requiring_surface": (
                list(promotion_families) if isinstance(promotion_families, list) else []
            ),
            "required_top_level_field_count": int(detail.get("required_top_level_field_count") or 0),
            "required_matrix_field_count": int(detail.get("required_matrix_field_count") or 0),
            "required_control_fields": (
                list(required_control_fields)
                if isinstance(required_control_fields, list)
                else []
            ),
            "required_eq_fields": (
                list(required_eq_fields) if isinstance(required_eq_fields, list) else []
            ),
            "required_eq_band_fields": (
                list(required_eq_band_fields)
                if isinstance(required_eq_band_fields, list)
                else []
            ),
            "required_dynamics_fields": (
                list(required_dynamics_fields)
                if isinstance(required_dynamics_fields, list)
                else []
            ),
            "required_insert_fields": (
                list(required_insert_fields)
                if isinstance(required_insert_fields, list)
                else []
            ),
            "required_metering_fields": (
                list(required_metering_fields)
                if isinstance(required_metering_fields, list)
                else []
            ),
            "required_coverage_flags": (
                list(required_coverage_flags)
                if isinstance(required_coverage_flags, list)
                else []
            ),
            "required_false_safety_flags": (
                list(required_false_flags) if isinstance(required_false_flags, list) else []
            ),
            "minimum_matrix_count": int(detail.get("minimum_matrix_count") or 0),
            "minimum_eq_band_count": int(detail.get("minimum_eq_band_count") or 0),
            "allowed_sources": list(allowed_sources) if isinstance(allowed_sources, list) else [],
            "accepted_matrix_statuses": (
                list(accepted_statuses) if isinstance(accepted_statuses, list) else []
            ),
            "forbidden_mutation_evidence_fields": (
                list(forbidden_mutation_fields)
                if isinstance(forbidden_mutation_fields, list)
                else []
            ),
            "target_consistency": (
                detail.get("physical_readback_surface_id") == detail.get("surface_id")
                and detail.get("physical_readback_action_id")
                == detail.get("remediation_action_id")
                and detail.get("fixture_physical_gap_declared") is True
                and detail.get("remediation_physical_gap_declared") is True
            ),
            "violations": list(violations) if isinstance(violations, list) else [],
        }
    return None


def matrix_state_readback_artifact_intake_summary(
    results: Sequence[CheckResult],
) -> dict[str, object] | None:
    for result in results:
        if result.name != "matrix_state_physical_readback_artifact_intake_schema":
            continue
        detail = result.detail if isinstance(result.detail, Mapping) else {}
        artifact_search_roots = detail.get("artifact_search_roots")
        blocked_by = detail.get("blocked_by")
        artifact_source_ids = detail.get("artifact_source_ids")
        intake_violations = detail.get("intake_violations")
        promotion_families = detail.get("promotion_families_requiring_surface")
        violations = detail.get("violations")
        source_candidate_claimed = detail.get("accepted_as_physical_fixture_candidate") is True
        operator_review_required = detail.get("operator_review_required") is True
        physical_fixture_gap_remains = detail.get("physical_fixture_gap_remains") is True
        unsafe_claimed = (
            detail.get("fixture_closure_allowed_by_intake") is True
            or detail.get("live_authority_granted") is True
            or detail.get("full_autonomous_ready") is True
        )
        candidate_accepted = (
            result.status == "passed"
            and source_candidate_claimed
            and operator_review_required
            and physical_fixture_gap_remains
            and not unsafe_claimed
        )
        return {
            "status": result.status,
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "operator_required": True,
            "execution_allowed_by_intake": False,
            "fixture_closure_allowed_by_intake": False,
            "live_widening_allowed": False,
            "full_autonomous_ready": False,
            "schema_id": detail.get("schema_id"),
            "intake_status": detail.get("intake_status"),
            "surface_id": detail.get("surface_id"),
            "remediation_action_id": detail.get("remediation_action_id"),
            "required_contract_method_id": detail.get("required_contract_method_id"),
            "artifact_search_roots": (
                list(artifact_search_roots) if isinstance(artifact_search_roots, list) else []
            ),
            "artifact_count": int(detail.get("artifact_count") or 0),
            "valid_candidate_count": int(detail.get("valid_candidate_count") or 0),
            "invalid_candidate_count": int(detail.get("invalid_candidate_count") or 0),
            "valid_matrix_count": int(detail.get("valid_matrix_count") or 0),
            "accepted_as_physical_fixture_candidate": candidate_accepted,
            "operator_review_required": True,
            "physical_fixture_gap_remains": True,
            "live_authority_granted": False,
            "promotion_families_requiring_surface": (
                list(promotion_families) if isinstance(promotion_families, list) else []
            ),
            "latest_valid_candidate_source_id": detail.get("latest_valid_candidate_source_id"),
            "latest_valid_candidate_captured_at": detail.get("latest_valid_candidate_captured_at"),
            "blocked_by": list(blocked_by) if isinstance(blocked_by, list) else [],
            "artifact_source_ids": (
                list(artifact_source_ids) if isinstance(artifact_source_ids, list) else []
            ),
            "intake_violations": (
                list(intake_violations) if isinstance(intake_violations, list) else []
            ),
            "violations": list(violations) if isinstance(violations, list) else [],
        }
    return None


def fx_slots_inserts_readback_contract_summary(
    results: Sequence[CheckResult],
) -> dict[str, object] | None:
    for result in results:
        if result.name != "fx_slots_inserts_physical_readback_contract_schema":
            continue
        detail = result.detail if isinstance(result.detail, Mapping) else {}
        allowed_sources = detail.get("allowed_sources")
        accepted_statuses = detail.get("accepted_row_statuses")
        target_types = detail.get("allowed_insert_target_types")
        positions = detail.get("allowed_insert_positions")
        forbidden_mutation_fields = detail.get("forbidden_mutation_evidence_fields")
        required_false_flags = detail.get("required_false_safety_flags")
        promotion_families = detail.get("promotion_families_requiring_surface")
        required_fx_slot_fields = detail.get("required_fx_slot_fields")
        required_parameter_fields = detail.get("required_parameter_fields")
        required_parameter_safe_range_fields = detail.get(
            "required_parameter_safe_range_fields"
        )
        required_parameter_schema_fields = detail.get("required_parameter_schema_fields")
        required_fx_slot_coverage_flags = detail.get("required_fx_slot_coverage_flags")
        required_insert_fields = detail.get("required_insert_fields")
        required_insert_coverage_flags = detail.get("required_insert_coverage_flags")
        violations = detail.get("violations")
        return {
            "status": result.status,
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "live_widening_allowed": False,
            "full_autonomous_ready": False,
            "schema_id": detail.get("schema_id"),
            "contract_status": detail.get("contract_status"),
            "surface_id": detail.get("surface_id"),
            "remediation_action_id": detail.get("remediation_action_id"),
            "physical_readback_surface_id": detail.get("physical_readback_surface_id"),
            "physical_readback_action_id": detail.get("physical_readback_action_id"),
            "fixture_physical_gap_declared": detail.get("fixture_physical_gap_declared") is True,
            "remediation_physical_gap_declared": (
                detail.get("remediation_physical_gap_declared") is True
            ),
            "artifact_required": detail.get("artifact_required") is True,
            "physical_fixture_gap_remains": detail.get("physical_fixture_gap_remains") is True,
            "accepted_as_physical_fixture_candidate": (
                detail.get("accepted_as_physical_fixture_candidate") is True
            ),
            "live_authority_granted": detail.get("live_authority_granted") is True,
            "promotion_families_requiring_surface": (
                list(promotion_families) if isinstance(promotion_families, list) else []
            ),
            "required_top_level_field_count": int(detail.get("required_top_level_field_count") or 0),
            "required_fx_slot_field_count": int(detail.get("required_fx_slot_field_count") or 0),
            "required_fx_slot_fields": (
                list(required_fx_slot_fields)
                if isinstance(required_fx_slot_fields, list)
                else []
            ),
            "required_parameter_fields": (
                list(required_parameter_fields)
                if isinstance(required_parameter_fields, list)
                else []
            ),
            "required_parameter_safe_range_fields": (
                list(required_parameter_safe_range_fields)
                if isinstance(required_parameter_safe_range_fields, list)
                else []
            ),
            "required_parameter_schema_fields": (
                list(required_parameter_schema_fields)
                if isinstance(required_parameter_schema_fields, list)
                else []
            ),
            "required_fx_slot_coverage_flags": (
                list(required_fx_slot_coverage_flags)
                if isinstance(required_fx_slot_coverage_flags, list)
                else []
            ),
            "required_insert_fields": (
                list(required_insert_fields)
                if isinstance(required_insert_fields, list)
                else []
            ),
            "required_insert_coverage_flags": (
                list(required_insert_coverage_flags)
                if isinstance(required_insert_coverage_flags, list)
                else []
            ),
            "required_false_safety_flags": (
                list(required_false_flags) if isinstance(required_false_flags, list) else []
            ),
            "minimum_fx_slot_count": int(detail.get("minimum_fx_slot_count") or 0),
            "minimum_fx_parameter_count": int(
                detail.get("minimum_fx_parameter_count") or 0
            ),
            "minimum_insert_count": int(detail.get("minimum_insert_count") or 0),
            "allowed_sources": list(allowed_sources) if isinstance(allowed_sources, list) else [],
            "accepted_row_statuses": (
                list(accepted_statuses) if isinstance(accepted_statuses, list) else []
            ),
            "allowed_insert_target_types": (
                list(target_types) if isinstance(target_types, list) else []
            ),
            "allowed_insert_positions": (
                list(positions) if isinstance(positions, list) else []
            ),
            "forbidden_mutation_evidence_fields": (
                list(forbidden_mutation_fields)
                if isinstance(forbidden_mutation_fields, list)
                else []
            ),
            "target_consistency": (
                detail.get("physical_readback_surface_id") == detail.get("surface_id")
                and detail.get("physical_readback_action_id")
                == detail.get("remediation_action_id")
                and detail.get("fixture_physical_gap_declared") is True
                and detail.get("remediation_physical_gap_declared") is True
            ),
            "violations": list(violations) if isinstance(violations, list) else [],
        }
    return None


def fx_slots_inserts_readback_artifact_intake_summary(
    results: Sequence[CheckResult],
) -> dict[str, object] | None:
    for result in results:
        if result.name != "fx_slots_inserts_physical_readback_artifact_intake_schema":
            continue
        detail = result.detail if isinstance(result.detail, Mapping) else {}
        artifact_search_roots = detail.get("artifact_search_roots")
        blocked_by = detail.get("blocked_by")
        artifact_source_ids = detail.get("artifact_source_ids")
        intake_violations = detail.get("intake_violations")
        promotion_families = detail.get("promotion_families_requiring_surface")
        violations = detail.get("violations")
        source_candidate_claimed = detail.get("accepted_as_physical_fixture_candidate") is True
        operator_review_required = detail.get("operator_review_required") is True
        physical_fixture_gap_remains = detail.get("physical_fixture_gap_remains") is True
        unsafe_claimed = (
            detail.get("fixture_closure_allowed_by_intake") is True
            or detail.get("live_authority_granted") is True
            or detail.get("full_autonomous_ready") is True
        )
        candidate_accepted = (
            result.status == "passed"
            and source_candidate_claimed
            and operator_review_required
            and physical_fixture_gap_remains
            and not unsafe_claimed
        )
        return {
            "status": result.status,
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "operator_required": True,
            "execution_allowed_by_intake": False,
            "fixture_closure_allowed_by_intake": False,
            "live_widening_allowed": False,
            "full_autonomous_ready": False,
            "schema_id": detail.get("schema_id"),
            "intake_status": detail.get("intake_status"),
            "surface_id": detail.get("surface_id"),
            "remediation_action_id": detail.get("remediation_action_id"),
            "required_contract_method_id": detail.get("required_contract_method_id"),
            "artifact_search_roots": (
                list(artifact_search_roots) if isinstance(artifact_search_roots, list) else []
            ),
            "artifact_count": int(detail.get("artifact_count") or 0),
            "valid_candidate_count": int(detail.get("valid_candidate_count") or 0),
            "invalid_candidate_count": int(detail.get("invalid_candidate_count") or 0),
            "valid_fx_slot_count": int(detail.get("valid_fx_slot_count") or 0),
            "valid_insert_count": int(detail.get("valid_insert_count") or 0),
            "accepted_as_physical_fixture_candidate": candidate_accepted,
            "operator_review_required": True,
            "physical_fixture_gap_remains": True,
            "live_authority_granted": False,
            "promotion_families_requiring_surface": (
                list(promotion_families) if isinstance(promotion_families, list) else []
            ),
            "latest_valid_candidate_source_id": detail.get("latest_valid_candidate_source_id"),
            "latest_valid_candidate_captured_at": detail.get("latest_valid_candidate_captured_at"),
            "blocked_by": list(blocked_by) if isinstance(blocked_by, list) else [],
            "artifact_source_ids": (
                list(artifact_source_ids) if isinstance(artifact_source_ids, list) else []
            ),
            "intake_violations": (
                list(intake_violations) if isinstance(intake_violations, list) else []
            ),
            "violations": list(violations) if isinstance(violations, list) else [],
        }
    return None


def routing_sends_readback_contract_summary(
    results: Sequence[CheckResult],
) -> dict[str, object] | None:
    for result in results:
        if result.name != "routing_sends_physical_readback_contract_schema":
            continue
        detail = result.detail if isinstance(result.detail, Mapping) else {}
        allowed_sources = detail.get("allowed_sources")
        accepted_statuses = detail.get("accepted_row_statuses")
        forbidden_mutation_fields = detail.get("forbidden_mutation_evidence_fields")
        required_false_flags = detail.get("required_false_safety_flags")
        promotion_families = detail.get("promotion_families_requiring_surface")
        required_static_route_fields = detail.get("required_static_route_fields")
        required_bus_send_fields = detail.get("required_bus_send_fields")
        required_main_send_fields = detail.get("required_main_send_fields")
        required_matrix_send_fields = detail.get("required_matrix_send_fields")
        required_output_route_fields = detail.get("required_output_route_fields")
        required_static_route_coverage_flags = detail.get(
            "required_static_route_coverage_flags"
        )
        required_send_coverage_flags = detail.get("required_send_coverage_flags")
        required_output_route_coverage_flags = detail.get(
            "required_output_route_coverage_flags"
        )
        violations = detail.get("violations")
        return {
            "status": result.status,
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "live_widening_allowed": False,
            "full_autonomous_ready": False,
            "schema_id": detail.get("schema_id"),
            "contract_status": detail.get("contract_status"),
            "surface_id": detail.get("surface_id"),
            "remediation_action_id": detail.get("remediation_action_id"),
            "physical_readback_surface_id": detail.get("physical_readback_surface_id"),
            "physical_readback_action_id": detail.get("physical_readback_action_id"),
            "fixture_physical_gap_declared": detail.get("fixture_physical_gap_declared") is True,
            "remediation_physical_gap_declared": (
                detail.get("remediation_physical_gap_declared") is True
            ),
            "artifact_required": detail.get("artifact_required") is True,
            "physical_fixture_gap_remains": detail.get("physical_fixture_gap_remains") is True,
            "accepted_as_physical_fixture_candidate": (
                detail.get("accepted_as_physical_fixture_candidate") is True
            ),
            "live_authority_granted": detail.get("live_authority_granted") is True,
            "promotion_families_requiring_surface": (
                list(promotion_families) if isinstance(promotion_families, list) else []
            ),
            "required_top_level_field_count": int(detail.get("required_top_level_field_count") or 0),
            "required_static_route_field_count": int(
                detail.get("required_static_route_field_count") or 0
            ),
            "required_static_route_fields": (
                list(required_static_route_fields)
                if isinstance(required_static_route_fields, list)
                else []
            ),
            "required_bus_send_fields": (
                list(required_bus_send_fields)
                if isinstance(required_bus_send_fields, list)
                else []
            ),
            "required_main_send_fields": (
                list(required_main_send_fields)
                if isinstance(required_main_send_fields, list)
                else []
            ),
            "required_matrix_send_fields": (
                list(required_matrix_send_fields)
                if isinstance(required_matrix_send_fields, list)
                else []
            ),
            "required_output_route_fields": (
                list(required_output_route_fields)
                if isinstance(required_output_route_fields, list)
                else []
            ),
            "required_static_route_coverage_flags": (
                list(required_static_route_coverage_flags)
                if isinstance(required_static_route_coverage_flags, list)
                else []
            ),
            "required_send_coverage_flags": (
                list(required_send_coverage_flags)
                if isinstance(required_send_coverage_flags, list)
                else []
            ),
            "required_output_route_coverage_flags": (
                list(required_output_route_coverage_flags)
                if isinstance(required_output_route_coverage_flags, list)
                else []
            ),
            "required_false_safety_flags": (
                list(required_false_flags) if isinstance(required_false_flags, list) else []
            ),
            "minimum_static_dante_route_count": int(
                detail.get("minimum_static_dante_route_count") or 0
            ),
            "minimum_channel_bus_send_count": int(
                detail.get("minimum_channel_bus_send_count") or 0
            ),
            "minimum_channel_main_send_count": int(
                detail.get("minimum_channel_main_send_count") or 0
            ),
            "minimum_channel_matrix_send_count": int(
                detail.get("minimum_channel_matrix_send_count") or 0
            ),
            "minimum_output_route_count": int(
                detail.get("minimum_output_route_count") or 0
            ),
            "allowed_sources": list(allowed_sources) if isinstance(allowed_sources, list) else [],
            "accepted_row_statuses": (
                list(accepted_statuses) if isinstance(accepted_statuses, list) else []
            ),
            "forbidden_mutation_evidence_fields": (
                list(forbidden_mutation_fields)
                if isinstance(forbidden_mutation_fields, list)
                else []
            ),
            "target_consistency": (
                detail.get("physical_readback_surface_id") == detail.get("surface_id")
                and detail.get("physical_readback_action_id")
                == detail.get("remediation_action_id")
                and detail.get("fixture_physical_gap_declared") is True
                and detail.get("remediation_physical_gap_declared") is True
            ),
            "violations": list(violations) if isinstance(violations, list) else [],
        }
    return None


def routing_sends_readback_artifact_intake_summary(
    results: Sequence[CheckResult],
) -> dict[str, object] | None:
    for result in results:
        if result.name != "routing_sends_physical_readback_artifact_intake_schema":
            continue
        detail = result.detail if isinstance(result.detail, Mapping) else {}
        artifact_search_roots = detail.get("artifact_search_roots")
        blocked_by = detail.get("blocked_by")
        artifact_source_ids = detail.get("artifact_source_ids")
        intake_violations = detail.get("intake_violations")
        promotion_families = detail.get("promotion_families_requiring_surface")
        violations = detail.get("violations")
        source_candidate_claimed = detail.get("accepted_as_physical_fixture_candidate") is True
        operator_review_required = detail.get("operator_review_required") is True
        physical_fixture_gap_remains = detail.get("physical_fixture_gap_remains") is True
        unsafe_claimed = (
            detail.get("fixture_closure_allowed_by_intake") is True
            or detail.get("live_authority_granted") is True
            or detail.get("full_autonomous_ready") is True
        )
        candidate_accepted = (
            result.status == "passed"
            and source_candidate_claimed
            and operator_review_required
            and physical_fixture_gap_remains
            and not unsafe_claimed
        )
        return {
            "status": result.status,
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "operator_required": True,
            "execution_allowed_by_intake": False,
            "fixture_closure_allowed_by_intake": False,
            "live_widening_allowed": False,
            "full_autonomous_ready": False,
            "schema_id": detail.get("schema_id"),
            "intake_status": detail.get("intake_status"),
            "surface_id": detail.get("surface_id"),
            "remediation_action_id": detail.get("remediation_action_id"),
            "required_contract_method_id": detail.get("required_contract_method_id"),
            "artifact_search_roots": (
                list(artifact_search_roots) if isinstance(artifact_search_roots, list) else []
            ),
            "artifact_count": int(detail.get("artifact_count") or 0),
            "valid_candidate_count": int(detail.get("valid_candidate_count") or 0),
            "invalid_candidate_count": int(detail.get("invalid_candidate_count") or 0),
            "valid_static_dante_route_count": int(
                detail.get("valid_static_dante_route_count") or 0
            ),
            "valid_channel_bus_send_count": int(
                detail.get("valid_channel_bus_send_count") or 0
            ),
            "valid_channel_main_send_count": int(
                detail.get("valid_channel_main_send_count") or 0
            ),
            "valid_channel_matrix_send_count": int(
                detail.get("valid_channel_matrix_send_count") or 0
            ),
            "valid_output_route_count": int(detail.get("valid_output_route_count") or 0),
            "accepted_as_physical_fixture_candidate": candidate_accepted,
            "operator_review_required": True,
            "physical_fixture_gap_remains": True,
            "live_authority_granted": False,
            "promotion_families_requiring_surface": (
                list(promotion_families) if isinstance(promotion_families, list) else []
            ),
            "latest_valid_candidate_source_id": detail.get("latest_valid_candidate_source_id"),
            "latest_valid_candidate_captured_at": detail.get("latest_valid_candidate_captured_at"),
            "blocked_by": list(blocked_by) if isinstance(blocked_by, list) else [],
            "artifact_source_ids": (
                list(artifact_source_ids) if isinstance(artifact_source_ids, list) else []
            ),
            "intake_violations": (
                list(intake_violations) if isinstance(intake_violations, list) else []
            ),
            "violations": list(violations) if isinstance(violations, list) else [],
        }
    return None


def snapshots_scenes_readback_contract_summary(
    results: Sequence[CheckResult],
) -> dict[str, object] | None:
    for result in results:
        if result.name != "snapshots_scenes_physical_readback_contract_schema":
            continue
        detail = result.detail if isinstance(result.detail, Mapping) else {}
        allowed_sources = detail.get("allowed_sources")
        accepted_statuses = detail.get("accepted_row_statuses")
        forbidden_mutation_fields = detail.get("forbidden_mutation_evidence_fields")
        required_false_flags = detail.get("required_false_safety_flags")
        promotion_families = detail.get("promotion_families_requiring_surface")
        required_active_scene_fields = detail.get("required_active_scene_fields")
        required_snapshot_fields = detail.get("required_snapshot_fields")
        required_coverage_flags = detail.get("required_coverage_flags")
        violations = detail.get("violations")
        return {
            "status": result.status,
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "live_widening_allowed": False,
            "full_autonomous_ready": False,
            "scene_load_allowed": False,
            "scene_save_allowed": False,
            "schema_id": detail.get("schema_id"),
            "contract_status": detail.get("contract_status"),
            "surface_id": detail.get("surface_id"),
            "remediation_action_id": detail.get("remediation_action_id"),
            "physical_readback_surface_id": detail.get("physical_readback_surface_id"),
            "physical_readback_action_id": detail.get("physical_readback_action_id"),
            "fixture_physical_gap_declared": detail.get("fixture_physical_gap_declared") is True,
            "remediation_physical_gap_declared": (
                detail.get("remediation_physical_gap_declared") is True
            ),
            "artifact_required": detail.get("artifact_required") is True,
            "physical_fixture_gap_remains": detail.get("physical_fixture_gap_remains") is True,
            "accepted_as_physical_fixture_candidate": (
                detail.get("accepted_as_physical_fixture_candidate") is True
            ),
            "live_authority_granted": detail.get("live_authority_granted") is True,
            "promotion_families_requiring_surface": (
                list(promotion_families) if isinstance(promotion_families, list) else []
            ),
            "required_top_level_field_count": int(detail.get("required_top_level_field_count") or 0),
            "required_active_scene_fields": (
                list(required_active_scene_fields)
                if isinstance(required_active_scene_fields, list)
                else []
            ),
            "required_snapshot_fields": (
                list(required_snapshot_fields)
                if isinstance(required_snapshot_fields, list)
                else []
            ),
            "required_coverage_flags": (
                list(required_coverage_flags) if isinstance(required_coverage_flags, list) else []
            ),
            "required_false_safety_flags": (
                list(required_false_flags) if isinstance(required_false_flags, list) else []
            ),
            "minimum_snapshot_count": int(detail.get("minimum_snapshot_count") or 0),
            "allowed_sources": list(allowed_sources) if isinstance(allowed_sources, list) else [],
            "accepted_row_statuses": (
                list(accepted_statuses) if isinstance(accepted_statuses, list) else []
            ),
            "forbidden_mutation_evidence_fields": (
                list(forbidden_mutation_fields)
                if isinstance(forbidden_mutation_fields, list)
                else []
            ),
            "target_consistency": (
                detail.get("physical_readback_surface_id") == detail.get("surface_id")
                and detail.get("physical_readback_action_id")
                == detail.get("remediation_action_id")
                and detail.get("fixture_physical_gap_declared") is True
                and detail.get("remediation_physical_gap_declared") is True
            ),
            "violations": list(violations) if isinstance(violations, list) else [],
        }
    return None


def snapshots_scenes_readback_artifact_intake_summary(
    results: Sequence[CheckResult],
) -> dict[str, object] | None:
    for result in results:
        if result.name != "snapshots_scenes_physical_readback_artifact_intake_schema":
            continue
        detail = result.detail if isinstance(result.detail, Mapping) else {}
        artifact_search_roots = detail.get("artifact_search_roots")
        blocked_by = detail.get("blocked_by")
        artifact_source_ids = detail.get("artifact_source_ids")
        intake_violations = detail.get("intake_violations")
        promotion_families = detail.get("promotion_families_requiring_surface")
        violations = detail.get("violations")
        source_candidate_claimed = detail.get("accepted_as_physical_fixture_candidate") is True
        operator_review_required = detail.get("operator_review_required") is True
        physical_fixture_gap_remains = detail.get("physical_fixture_gap_remains") is True
        unsafe_claimed = (
            detail.get("fixture_closure_allowed_by_intake") is True
            or detail.get("scene_load_allowed") is True
            or detail.get("scene_save_allowed") is True
            or detail.get("live_authority_granted") is True
            or detail.get("full_autonomous_ready") is True
        )
        candidate_accepted = (
            result.status == "passed"
            and source_candidate_claimed
            and operator_review_required
            and physical_fixture_gap_remains
            and not unsafe_claimed
        )
        return {
            "status": result.status,
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "operator_required": True,
            "execution_allowed_by_intake": False,
            "fixture_closure_allowed_by_intake": False,
            "scene_load_allowed": False,
            "scene_save_allowed": False,
            "live_widening_allowed": False,
            "full_autonomous_ready": False,
            "schema_id": detail.get("schema_id"),
            "intake_status": detail.get("intake_status"),
            "surface_id": detail.get("surface_id"),
            "remediation_action_id": detail.get("remediation_action_id"),
            "required_contract_method_id": detail.get("required_contract_method_id"),
            "artifact_search_roots": (
                list(artifact_search_roots) if isinstance(artifact_search_roots, list) else []
            ),
            "artifact_count": int(detail.get("artifact_count") or 0),
            "valid_candidate_count": int(detail.get("valid_candidate_count") or 0),
            "invalid_candidate_count": int(detail.get("invalid_candidate_count") or 0),
            "valid_snapshot_count": int(detail.get("valid_snapshot_count") or 0),
            "accepted_as_physical_fixture_candidate": candidate_accepted,
            "operator_review_required": True,
            "physical_fixture_gap_remains": True,
            "live_authority_granted": False,
            "promotion_families_requiring_surface": (
                list(promotion_families) if isinstance(promotion_families, list) else []
            ),
            "latest_valid_candidate_source_id": detail.get(
                "latest_valid_candidate_source_id"
            ),
            "latest_valid_candidate_captured_at": detail.get(
                "latest_valid_candidate_captured_at"
            ),
            "artifact_source_ids": (
                list(artifact_source_ids) if isinstance(artifact_source_ids, list) else []
            ),
            "blocked_by": list(blocked_by) if isinstance(blocked_by, list) else [],
            "intake_violations": (
                list(intake_violations) if isinstance(intake_violations, list) else []
            ),
            "target_consistency": (
                detail.get("surface_id") == SNAPSHOTS_SCENES_SURFACE_ID
                and detail.get("remediation_action_id")
                == SNAPSHOTS_SCENES_REMEDIATION_ACTION_ID
                and detail.get("required_contract_method_id")
                == SNAPSHOTS_SCENES_PHYSICAL_READBACK_CONTRACT_METHOD_ID
            ),
            "violations": list(violations) if isinstance(violations, list) else [],
        }
    return None


def user_presets_readback_contract_summary(
    results: Sequence[CheckResult],
) -> dict[str, object] | None:
    for result in results:
        if result.name != "user_presets_physical_readback_contract_schema":
            continue
        detail = result.detail if isinstance(result.detail, Mapping) else {}
        allowed_sources = detail.get("allowed_sources")
        accepted_statuses = detail.get("accepted_row_statuses")
        forbidden_mutation_fields = detail.get("forbidden_mutation_evidence_fields")
        required_false_flags = detail.get("required_false_safety_flags")
        promotion_families = detail.get("promotion_families_requiring_surface")
        required_preset_fields = detail.get("required_preset_fields")
        required_coverage_flags = detail.get("required_coverage_flags")
        violations = detail.get("violations")
        return {
            "status": result.status,
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "live_widening_allowed": False,
            "full_autonomous_ready": False,
            "preset_recall_allowed": False,
            "preset_store_allowed": False,
            "schema_id": detail.get("schema_id"),
            "contract_status": detail.get("contract_status"),
            "surface_id": detail.get("surface_id"),
            "remediation_action_id": detail.get("remediation_action_id"),
            "physical_readback_surface_id": detail.get("physical_readback_surface_id"),
            "physical_readback_action_id": detail.get("physical_readback_action_id"),
            "fixture_physical_gap_declared": detail.get("fixture_physical_gap_declared") is True,
            "remediation_physical_gap_declared": (
                detail.get("remediation_physical_gap_declared") is True
            ),
            "artifact_required": detail.get("artifact_required") is True,
            "physical_fixture_gap_remains": detail.get("physical_fixture_gap_remains") is True,
            "accepted_as_physical_fixture_candidate": (
                detail.get("accepted_as_physical_fixture_candidate") is True
            ),
            "live_authority_granted": detail.get("live_authority_granted") is True,
            "promotion_families_requiring_surface": (
                list(promotion_families) if isinstance(promotion_families, list) else []
            ),
            "required_top_level_field_count": int(detail.get("required_top_level_field_count") or 0),
            "required_preset_fields": (
                list(required_preset_fields)
                if isinstance(required_preset_fields, list)
                else []
            ),
            "required_coverage_flags": (
                list(required_coverage_flags)
                if isinstance(required_coverage_flags, list)
                else []
            ),
            "required_false_safety_flags": (
                list(required_false_flags) if isinstance(required_false_flags, list) else []
            ),
            "minimum_preset_count": int(detail.get("minimum_preset_count") or 0),
            "allowed_sources": list(allowed_sources) if isinstance(allowed_sources, list) else [],
            "accepted_row_statuses": (
                list(accepted_statuses) if isinstance(accepted_statuses, list) else []
            ),
            "forbidden_mutation_evidence_fields": (
                list(forbidden_mutation_fields)
                if isinstance(forbidden_mutation_fields, list)
                else []
            ),
            "target_consistency": (
                detail.get("physical_readback_surface_id") == detail.get("surface_id")
                and detail.get("physical_readback_action_id")
                == detail.get("remediation_action_id")
                and detail.get("fixture_physical_gap_declared") is True
                and detail.get("remediation_physical_gap_declared") is True
            ),
            "violations": list(violations) if isinstance(violations, list) else [],
        }
    return None


def user_presets_readback_artifact_intake_summary(
    results: Sequence[CheckResult],
) -> dict[str, object] | None:
    for result in results:
        if result.name != "user_presets_physical_readback_artifact_intake_schema":
            continue
        detail = result.detail if isinstance(result.detail, Mapping) else {}
        artifact_search_roots = detail.get("artifact_search_roots")
        blocked_by = detail.get("blocked_by")
        artifact_source_ids = detail.get("artifact_source_ids")
        intake_violations = detail.get("intake_violations")
        promotion_families = detail.get("promotion_families_requiring_surface")
        violations = detail.get("violations")
        source_candidate_claimed = detail.get("accepted_as_physical_fixture_candidate") is True
        operator_review_required = detail.get("operator_review_required") is True
        physical_fixture_gap_remains = detail.get("physical_fixture_gap_remains") is True
        unsafe_claimed = (
            detail.get("fixture_closure_allowed_by_intake") is True
            or detail.get("preset_recall_allowed") is True
            or detail.get("preset_store_allowed") is True
            or detail.get("live_authority_granted") is True
            or detail.get("full_autonomous_ready") is True
        )
        candidate_accepted = (
            result.status == "passed"
            and source_candidate_claimed
            and operator_review_required
            and physical_fixture_gap_remains
            and not unsafe_claimed
        )
        return {
            "status": result.status,
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "operator_required": True,
            "execution_allowed_by_intake": False,
            "fixture_closure_allowed_by_intake": False,
            "preset_recall_allowed": False,
            "preset_store_allowed": False,
            "live_widening_allowed": False,
            "full_autonomous_ready": False,
            "schema_id": detail.get("schema_id"),
            "intake_status": detail.get("intake_status"),
            "surface_id": detail.get("surface_id"),
            "remediation_action_id": detail.get("remediation_action_id"),
            "required_contract_method_id": detail.get("required_contract_method_id"),
            "artifact_search_roots": (
                list(artifact_search_roots) if isinstance(artifact_search_roots, list) else []
            ),
            "artifact_count": int(detail.get("artifact_count") or 0),
            "valid_candidate_count": int(detail.get("valid_candidate_count") or 0),
            "invalid_candidate_count": int(detail.get("invalid_candidate_count") or 0),
            "valid_preset_count": int(detail.get("valid_preset_count") or 0),
            "accepted_as_physical_fixture_candidate": candidate_accepted,
            "operator_review_required": True,
            "physical_fixture_gap_remains": True,
            "live_authority_granted": False,
            "promotion_families_requiring_surface": (
                list(promotion_families) if isinstance(promotion_families, list) else []
            ),
            "latest_valid_candidate_source_id": detail.get(
                "latest_valid_candidate_source_id"
            ),
            "latest_valid_candidate_captured_at": detail.get(
                "latest_valid_candidate_captured_at"
            ),
            "artifact_source_ids": (
                list(artifact_source_ids) if isinstance(artifact_source_ids, list) else []
            ),
            "blocked_by": list(blocked_by) if isinstance(blocked_by, list) else [],
            "intake_violations": (
                list(intake_violations) if isinstance(intake_violations, list) else []
            ),
            "target_consistency": (
                detail.get("surface_id") == USER_PRESETS_SURFACE_ID
                and detail.get("remediation_action_id")
                == USER_PRESETS_REMEDIATION_ACTION_ID
                and detail.get("required_contract_method_id")
                == USER_PRESETS_PHYSICAL_READBACK_CONTRACT_METHOD_ID
            ),
            "violations": list(violations) if isinstance(violations, list) else [],
        }
    return None


def dlive_state_readback_contract_summary(
    results: Sequence[CheckResult],
) -> dict[str, object] | None:
    for result in results:
        if result.name != "dlive_state_physical_readback_contract_schema":
            continue
        detail = result.detail if isinstance(result.detail, Mapping) else {}
        allowed_sources = detail.get("allowed_sources")
        accepted_statuses = detail.get("accepted_row_statuses")
        forbidden_mutation_fields = detail.get("forbidden_mutation_evidence_fields")
        required_false_flags = detail.get("required_false_safety_flags")
        promotion_families = detail.get("promotion_families_requiring_surface")
        required_channel_fields = detail.get("required_channel_fields")
        required_midi_state_fields = detail.get("required_midi_state_fields")
        required_parity_fields = detail.get("required_parity_fields")
        required_coverage_flags = detail.get("required_coverage_flags")
        violations = detail.get("violations")
        return {
            "status": result.status,
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "live_widening_allowed": False,
            "full_autonomous_ready": False,
            "midi_send_allowed": False,
            "tcp_send_allowed": False,
            "dlive_write_allowed": False,
            "dlive_apply_allowed": False,
            "schema_id": detail.get("schema_id"),
            "contract_status": detail.get("contract_status"),
            "surface_id": detail.get("surface_id"),
            "remediation_action_id": detail.get("remediation_action_id"),
            "physical_readback_surface_id": detail.get("physical_readback_surface_id"),
            "physical_readback_action_id": detail.get("physical_readback_action_id"),
            "fixture_physical_gap_declared": detail.get("fixture_physical_gap_declared") is True,
            "remediation_physical_gap_declared": (
                detail.get("remediation_physical_gap_declared") is True
            ),
            "artifact_required": detail.get("artifact_required") is True,
            "physical_fixture_gap_remains": detail.get("physical_fixture_gap_remains") is True,
            "accepted_as_physical_fixture_candidate": (
                detail.get("accepted_as_physical_fixture_candidate") is True
            ),
            "live_authority_granted": detail.get("live_authority_granted") is True,
            "promotion_families_requiring_surface": (
                list(promotion_families) if isinstance(promotion_families, list) else []
            ),
            "required_top_level_field_count": int(detail.get("required_top_level_field_count") or 0),
            "required_channel_fields": (
                list(required_channel_fields)
                if isinstance(required_channel_fields, list)
                else []
            ),
            "required_midi_state_fields": (
                list(required_midi_state_fields)
                if isinstance(required_midi_state_fields, list)
                else []
            ),
            "required_parity_fields": (
                list(required_parity_fields)
                if isinstance(required_parity_fields, list)
                else []
            ),
            "required_coverage_flags": (
                list(required_coverage_flags)
                if isinstance(required_coverage_flags, list)
                else []
            ),
            "required_false_safety_flags": (
                list(required_false_flags) if isinstance(required_false_flags, list) else []
            ),
            "minimum_channel_count": int(detail.get("minimum_channel_count") or 0),
            "allowed_sources": list(allowed_sources) if isinstance(allowed_sources, list) else [],
            "accepted_row_statuses": (
                list(accepted_statuses) if isinstance(accepted_statuses, list) else []
            ),
            "forbidden_mutation_evidence_fields": (
                list(forbidden_mutation_fields)
                if isinstance(forbidden_mutation_fields, list)
                else []
            ),
            "target_consistency": (
                detail.get("physical_readback_surface_id") == detail.get("surface_id")
                and detail.get("physical_readback_action_id")
                == detail.get("remediation_action_id")
                and detail.get("fixture_physical_gap_declared") is True
                and detail.get("remediation_physical_gap_declared") is True
            ),
            "violations": list(violations) if isinstance(violations, list) else [],
        }
    return None


def dlive_state_readback_artifact_intake_summary(
    results: Sequence[CheckResult],
) -> dict[str, object] | None:
    for result in results:
        if result.name != "dlive_state_physical_readback_artifact_intake_schema":
            continue
        detail = result.detail if isinstance(result.detail, Mapping) else {}
        artifact_search_roots = detail.get("artifact_search_roots")
        blocked_by = detail.get("blocked_by")
        artifact_source_ids = detail.get("artifact_source_ids")
        intake_violations = detail.get("intake_violations")
        promotion_families = detail.get("promotion_families_requiring_surface")
        violations = detail.get("violations")
        source_candidate_claimed = detail.get("accepted_as_physical_fixture_candidate") is True
        operator_review_required = detail.get("operator_review_required") is True
        physical_fixture_gap_remains = detail.get("physical_fixture_gap_remains") is True
        unsafe_claimed = (
            detail.get("fixture_closure_allowed_by_intake") is True
            or detail.get("midi_send_allowed") is True
            or detail.get("tcp_send_allowed") is True
            or detail.get("dlive_write_allowed") is True
            or detail.get("dlive_apply_allowed") is True
            or detail.get("live_authority_granted") is True
            or detail.get("full_autonomous_ready") is True
        )
        candidate_accepted = (
            result.status == "passed"
            and source_candidate_claimed
            and operator_review_required
            and physical_fixture_gap_remains
            and not unsafe_claimed
        )
        return {
            "status": result.status,
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "operator_required": True,
            "execution_allowed_by_intake": False,
            "fixture_closure_allowed_by_intake": False,
            "midi_send_allowed": False,
            "tcp_send_allowed": False,
            "dlive_write_allowed": False,
            "dlive_apply_allowed": False,
            "live_widening_allowed": False,
            "full_autonomous_ready": False,
            "schema_id": detail.get("schema_id"),
            "intake_status": detail.get("intake_status"),
            "surface_id": detail.get("surface_id"),
            "remediation_action_id": detail.get("remediation_action_id"),
            "required_contract_method_id": detail.get("required_contract_method_id"),
            "artifact_search_roots": (
                list(artifact_search_roots) if isinstance(artifact_search_roots, list) else []
            ),
            "artifact_count": int(detail.get("artifact_count") or 0),
            "valid_candidate_count": int(detail.get("valid_candidate_count") or 0),
            "invalid_candidate_count": int(detail.get("invalid_candidate_count") or 0),
            "valid_channel_count": int(detail.get("valid_channel_count") or 0),
            "accepted_as_physical_fixture_candidate": candidate_accepted,
            "operator_review_required": True,
            "physical_fixture_gap_remains": True,
            "live_authority_granted": False,
            "promotion_families_requiring_surface": (
                list(promotion_families) if isinstance(promotion_families, list) else []
            ),
            "latest_valid_candidate_source_id": detail.get(
                "latest_valid_candidate_source_id"
            ),
            "latest_valid_candidate_captured_at": detail.get(
                "latest_valid_candidate_captured_at"
            ),
            "artifact_source_ids": (
                list(artifact_source_ids) if isinstance(artifact_source_ids, list) else []
            ),
            "blocked_by": list(blocked_by) if isinstance(blocked_by, list) else [],
            "intake_violations": (
                list(intake_violations) if isinstance(intake_violations, list) else []
            ),
            "target_consistency": (
                detail.get("surface_id") == DLIVE_STATE_SURFACE_ID
                and detail.get("remediation_action_id")
                == DLIVE_STATE_REMEDIATION_ACTION_ID
                and detail.get("required_contract_method_id")
                == DLIVE_STATE_PHYSICAL_READBACK_CONTRACT_METHOD_ID
            ),
            "violations": list(violations) if isinstance(violations, list) else [],
        }
    return None


def read_surface_physical_readback_artifact_review_queue_summary(
    results: Sequence[CheckResult],
) -> dict[str, object] | None:
    for result in results:
        if result.name != "read_surface_physical_readback_artifact_review_queue_schema":
            continue
        detail = result.detail if isinstance(result.detail, Mapping) else {}
        blocked_by = detail.get("blocked_by")
        promotion_families = detail.get("promotion_families_requiring_surface")
        review_candidate_surface_ids = detail.get("review_candidate_surface_ids")
        invalid_surface_ids = detail.get("invalid_surface_ids")
        missing_surface_ids = detail.get("missing_artifact_surface_ids")
        unsafe_surface_ids = detail.get("unsafe_authority_surface_ids")
        candidate_source_missing_surface_ids = detail.get(
            "candidate_source_id_missing_surface_ids"
        )
        target_inconsistent_surface_ids = detail.get("target_inconsistent_surface_ids")
        queue_item_method_ids = detail.get("queue_item_method_ids")
        violations = detail.get("violations")
        unsafe_claimed = (
            detail.get("fixture_closure_allowed_by_queue") is True
            or detail.get("live_authority_granted") is True
            or detail.get("full_autonomous_ready") is True
        )
        review_candidate_count = int(detail.get("review_candidate_count") or 0)
        invalid_candidate_count = int(detail.get("invalid_candidate_count") or 0)
        unsafe_claim_count = int(detail.get("unsafe_authority_claim_count") or 0)
        candidate_source_missing_count = int(
            detail.get("candidate_source_id_missing_count") or 0
        )
        target_inconsistent_count = int(detail.get("target_inconsistent_surface_count") or 0)
        review_candidates_available = (
            result.status == "passed"
            and review_candidate_count > 0
            and invalid_candidate_count == 0
            and unsafe_claim_count == 0
            and candidate_source_missing_count == 0
            and target_inconsistent_count == 0
            and not unsafe_claimed
        )
        return {
            "status": result.status,
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "operator_required": True,
            "execution_allowed_by_queue": False,
            "fixture_closure_allowed_by_queue": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "queue_status": detail.get("queue_status"),
            "intake_surface_count": int(detail.get("intake_surface_count") or 0),
            "available_intake_count": int(detail.get("available_intake_count") or 0),
            "artifact_count": int(detail.get("artifact_count") or 0),
            "valid_candidate_count": int(detail.get("valid_candidate_count") or 0),
            "invalid_candidate_count": invalid_candidate_count,
            "review_candidate_count": review_candidate_count,
            "review_candidates_available": review_candidates_available,
            "missing_artifact_surface_count": int(
                detail.get("missing_artifact_surface_count") or 0
            ),
            "unsafe_authority_claim_count": unsafe_claim_count,
            "candidate_source_id_missing_count": candidate_source_missing_count,
            "target_inconsistent_surface_count": target_inconsistent_count,
            "review_candidate_surface_ids": (
                list(review_candidate_surface_ids)
                if isinstance(review_candidate_surface_ids, list)
                else []
            ),
            "invalid_surface_ids": (
                list(invalid_surface_ids) if isinstance(invalid_surface_ids, list) else []
            ),
            "missing_artifact_surface_ids": (
                list(missing_surface_ids) if isinstance(missing_surface_ids, list) else []
            ),
            "unsafe_authority_surface_ids": (
                list(unsafe_surface_ids) if isinstance(unsafe_surface_ids, list) else []
            ),
            "candidate_source_id_missing_surface_ids": (
                list(candidate_source_missing_surface_ids)
                if isinstance(candidate_source_missing_surface_ids, list)
                else []
            ),
            "target_inconsistent_surface_ids": (
                list(target_inconsistent_surface_ids)
                if isinstance(target_inconsistent_surface_ids, list)
                else []
            ),
            "promotion_families_requiring_surface": (
                list(promotion_families) if isinstance(promotion_families, list) else []
            ),
            "queue_item_method_ids": (
                list(queue_item_method_ids)
                if isinstance(queue_item_method_ids, list)
                else []
            ),
            "operator_review_required": True,
            "physical_fixture_gap_remains": True,
            "blocked_by": list(blocked_by) if isinstance(blocked_by, list) else [],
            "violations": list(violations) if isinstance(violations, list) else [],
        }
    return None


def read_surface_physical_readback_artifact_review_queue_artifact_summary(
    results: Sequence[CheckResult],
) -> dict[str, object] | None:
    for result in results:
        if (
            result.name
            != "read_surface_physical_readback_artifact_review_queue_artifact_schema"
        ):
            continue
        detail = result.detail if isinstance(result.detail, Mapping) else {}
        blocked_by = detail.get("blocked_by")
        promotion_families = detail.get("promotion_families_requiring_surface")
        review_candidate_surface_ids = detail.get("review_candidate_surface_ids")
        invalid_surface_ids = detail.get("invalid_surface_ids")
        missing_surface_ids = detail.get("missing_artifact_surface_ids")
        unsafe_surface_ids = detail.get("unsafe_authority_surface_ids")
        candidate_source_missing_surface_ids = detail.get(
            "candidate_source_id_missing_surface_ids"
        )
        target_inconsistent_surface_ids = detail.get("target_inconsistent_surface_ids")
        queue_item_method_ids = detail.get("queue_item_method_ids")
        focused_validation = detail.get("focused_validation")
        violations = detail.get("violations")
        unsafe_claimed = (
            detail.get("fixture_closure_allowed_by_queue") is True
            or detail.get("live_authority_granted") is True
            or detail.get("full_autonomous_ready") is True
        )
        review_candidate_count = int(detail.get("review_candidate_count") or 0)
        invalid_candidate_count = int(detail.get("invalid_candidate_count") or 0)
        unsafe_claim_count = int(detail.get("unsafe_authority_claim_count") or 0)
        candidate_source_missing_count = int(
            detail.get("candidate_source_id_missing_count") or 0
        )
        target_inconsistent_count = int(
            detail.get("target_inconsistent_surface_count") or 0
        )
        review_candidates_available = (
            result.status == "passed"
            and review_candidate_count > 0
            and invalid_candidate_count == 0
            and unsafe_claim_count == 0
            and candidate_source_missing_count == 0
            and target_inconsistent_count == 0
            and not unsafe_claimed
        )
        return {
            "status": result.status,
            "read_only": detail.get("read_only") is True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "operator_required": detail.get("operator_required") is True,
            "execution_allowed_by_queue": False,
            "fixture_closure_allowed_by_queue": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "queue_path": detail.get("queue_path"),
            "queue_type": detail.get("queue_type"),
            "queue_status": detail.get("queue_status"),
            "method_id": detail.get("method_id"),
            "intake_surface_count": int(detail.get("intake_surface_count") or 0),
            "available_intake_count": int(detail.get("available_intake_count") or 0),
            "artifact_count": int(detail.get("artifact_count") or 0),
            "valid_candidate_count": int(detail.get("valid_candidate_count") or 0),
            "invalid_candidate_count": invalid_candidate_count,
            "review_candidate_count": review_candidate_count,
            "review_candidates_available": review_candidates_available,
            "missing_artifact_surface_count": int(
                detail.get("missing_artifact_surface_count") or 0
            ),
            "unsafe_authority_claim_count": unsafe_claim_count,
            "candidate_source_id_missing_count": candidate_source_missing_count,
            "target_inconsistent_surface_count": target_inconsistent_count,
            "review_candidate_surface_ids": (
                list(review_candidate_surface_ids)
                if isinstance(review_candidate_surface_ids, list)
                else []
            ),
            "invalid_surface_ids": (
                list(invalid_surface_ids) if isinstance(invalid_surface_ids, list) else []
            ),
            "missing_artifact_surface_ids": (
                list(missing_surface_ids) if isinstance(missing_surface_ids, list) else []
            ),
            "unsafe_authority_surface_ids": (
                list(unsafe_surface_ids)
                if isinstance(unsafe_surface_ids, list)
                else []
            ),
            "candidate_source_id_missing_surface_ids": (
                list(candidate_source_missing_surface_ids)
                if isinstance(candidate_source_missing_surface_ids, list)
                else []
            ),
            "target_inconsistent_surface_ids": (
                list(target_inconsistent_surface_ids)
                if isinstance(target_inconsistent_surface_ids, list)
                else []
            ),
            "promotion_families_requiring_surface": (
                list(promotion_families) if isinstance(promotion_families, list) else []
            ),
            "queue_item_method_ids": (
                list(queue_item_method_ids)
                if isinstance(queue_item_method_ids, list)
                else []
            ),
            "operator_review_required": detail.get("operator_review_required") is True,
            "physical_fixture_gap_remains": (
                detail.get("physical_fixture_gap_remains") is True
            ),
            "blocked_by": list(blocked_by) if isinstance(blocked_by, list) else [],
            "operator_sequence_count": int(detail.get("operator_sequence_count") or 0),
            "focused_validation": (
                list(focused_validation)
                if isinstance(focused_validation, list)
                else []
            ),
            "violations": list(violations) if isinstance(violations, list) else [],
        }
    return None


def read_surface_physical_readback_acceptance_template_artifact_summary(
    results: Sequence[CheckResult],
) -> dict[str, object] | None:
    for result in results:
        if (
            result.name
            != "read_surface_physical_readback_acceptance_template_artifact_schema"
        ):
            continue
        detail = result.detail if isinstance(result.detail, Mapping) else {}
        required_edits = detail.get("required_operator_edits")
        blocked_by = detail.get("blocked_by")
        unsafe_flags = detail.get("record_unsafe_flags")
        violations = detail.get("violations")
        return {
            "status": result.status,
            "read_only": detail.get("read_only") is True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "operator_required": detail.get("operator_required") is True,
            "execution_allowed_by_template": False,
            "fixture_closure_allowed_by_template": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "template_path": detail.get("template_path"),
            "template_type": detail.get("template_type"),
            "template_status": detail.get("template_status"),
            "source_queue_status": detail.get("source_queue_status"),
            "selected_surface_id": detail.get("selected_surface_id"),
            "review_candidate_count": int(detail.get("review_candidate_count") or 0),
            "template_record_count": int(detail.get("template_record_count") or 0),
            "candidate_trace_count": int(detail.get("candidate_trace_count") or 0),
            "required_operator_edits": (
                list(required_edits) if isinstance(required_edits, list) else []
            ),
            "operator_sequence_count": int(detail.get("operator_sequence_count") or 0),
            "blocked_by": list(blocked_by) if isinstance(blocked_by, list) else [],
            "record_unsafe_flags": (
                list(unsafe_flags) if isinstance(unsafe_flags, list) else []
            ),
            "violations": list(violations) if isinstance(violations, list) else [],
        }
    return None


def read_surface_physical_readback_fixture_acceptance_ledger_summary(
    results: Sequence[CheckResult],
) -> dict[str, object] | None:
    for result in results:
        if result.name != "read_surface_physical_readback_fixture_acceptance_ledger_schema":
            continue
        detail = result.detail if isinstance(result.detail, Mapping) else {}
        blocked_by = detail.get("blocked_by")
        accepted_surface_ids = detail.get("accepted_surface_ids")
        rejected_surface_ids = detail.get("rejected_surface_ids")
        invalid_record_ids = detail.get("invalid_record_ids")
        unreviewed_surface_ids = detail.get("unreviewed_candidate_surface_ids")
        violations = detail.get("violations")
        return {
            "status": result.status,
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "operator_required": True,
            "execution_allowed_by_acceptance": False,
            "fixture_closure_allowed_by_acceptance": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "ledger_status": detail.get("ledger_status"),
            "queue_status": detail.get("queue_status"),
            "review_candidate_count": int(detail.get("review_candidate_count") or 0),
            "acceptance_record_count": int(detail.get("acceptance_record_count") or 0),
            "accepted_record_count": int(detail.get("accepted_record_count") or 0),
            "rejected_record_count": int(detail.get("rejected_record_count") or 0),
            "invalid_record_count": int(detail.get("invalid_record_count") or 0),
            "candidate_trace_missing_count": int(
                detail.get("candidate_trace_missing_count") or 0
            ),
            "candidate_trace_mismatch_count": int(
                detail.get("candidate_trace_mismatch_count") or 0
            ),
            "unreviewed_candidate_count": int(
                detail.get("unreviewed_candidate_count") or 0
            ),
            "accepted_surface_ids": (
                list(accepted_surface_ids)
                if isinstance(accepted_surface_ids, list)
                else []
            ),
            "rejected_surface_ids": (
                list(rejected_surface_ids)
                if isinstance(rejected_surface_ids, list)
                else []
            ),
            "invalid_record_ids": (
                list(invalid_record_ids) if isinstance(invalid_record_ids, list) else []
            ),
            "unreviewed_candidate_surface_ids": (
                list(unreviewed_surface_ids)
                if isinstance(unreviewed_surface_ids, list)
                else []
            ),
            "operator_acceptance_required": (
                detail.get("operator_acceptance_required") is True
            ),
            "operator_review_required": True,
            "physical_fixture_gap_remains": True,
            "blocked_by": list(blocked_by) if isinstance(blocked_by, list) else [],
            "violations": list(violations) if isinstance(violations, list) else [],
        }
    return None


def read_surface_physical_readback_fixture_closure_plan_summary(
    results: Sequence[CheckResult],
) -> dict[str, object] | None:
    for result in results:
        if result.name != "read_surface_physical_readback_fixture_closure_plan_schema":
            continue
        detail = result.detail if isinstance(result.detail, Mapping) else {}
        blocked_by = detail.get("blocked_by")
        closure_candidate_surface_ids = detail.get("closure_candidate_surface_ids")
        rejected_surface_ids = detail.get("rejected_surface_ids")
        invalid_record_ids = detail.get("invalid_record_ids")
        unreviewed_surface_ids = detail.get("unreviewed_candidate_surface_ids")
        violations = detail.get("violations")
        return {
            "status": result.status,
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "operator_required": True,
            "execution_allowed_by_plan": False,
            "fixture_update_allowed_by_plan": False,
            "fixture_closure_allowed_by_plan": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "plan_status": detail.get("plan_status"),
            "ledger_status": detail.get("ledger_status"),
            "review_candidate_count": int(detail.get("review_candidate_count") or 0),
            "acceptance_record_count": int(detail.get("acceptance_record_count") or 0),
            "accepted_record_count": int(detail.get("accepted_record_count") or 0),
            "rejected_record_count": int(detail.get("rejected_record_count") or 0),
            "invalid_record_count": int(detail.get("invalid_record_count") or 0),
            "unreviewed_candidate_count": int(
                detail.get("unreviewed_candidate_count") or 0
            ),
            "closure_candidate_count": int(detail.get("closure_candidate_count") or 0),
            "closure_candidate_surface_ids": (
                list(closure_candidate_surface_ids)
                if isinstance(closure_candidate_surface_ids, list)
                else []
            ),
            "rejected_surface_ids": (
                list(rejected_surface_ids)
                if isinstance(rejected_surface_ids, list)
                else []
            ),
            "invalid_record_ids": (
                list(invalid_record_ids) if isinstance(invalid_record_ids, list) else []
            ),
            "unreviewed_candidate_surface_ids": (
                list(unreviewed_surface_ids)
                if isinstance(unreviewed_surface_ids, list)
                else []
            ),
            "manual_fixture_closure_candidate_available": (
                detail.get("manual_fixture_closure_candidate_available") is True
            ),
            "manual_fixture_closure_required": True,
            "operator_review_required": True,
            "physical_fixture_gap_remains": True,
            "blocked_by": list(blocked_by) if isinstance(blocked_by, list) else [],
            "violations": list(violations) if isinstance(violations, list) else [],
        }
    return None


def read_surface_physical_readback_fixture_closure_plan_artifact_summary(
    results: Sequence[CheckResult],
) -> dict[str, object] | None:
    for result in results:
        if (
            result.name
            != "read_surface_physical_readback_fixture_closure_plan_artifact_schema"
        ):
            continue
        detail = result.detail if isinstance(result.detail, Mapping) else {}
        blocked_by = detail.get("blocked_by")
        closure_candidate_surface_ids = detail.get("closure_candidate_surface_ids")
        rejected_surface_ids = detail.get("rejected_surface_ids")
        invalid_record_ids = detail.get("invalid_record_ids")
        unreviewed_surface_ids = detail.get("unreviewed_candidate_surface_ids")
        focused_validation = detail.get("focused_validation")
        violations = detail.get("violations")
        return {
            "status": result.status,
            "read_only": detail.get("read_only") is True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "operator_required": detail.get("operator_required") is True,
            "execution_allowed_by_plan": False,
            "fixture_update_allowed_by_plan": False,
            "fixture_closure_allowed_by_plan": False,
            "live_widening_allowed": False,
            "live_authority_granted": False,
            "full_autonomous_ready": False,
            "plan_path": detail.get("plan_path"),
            "plan_type": detail.get("plan_type"),
            "plan_status": detail.get("plan_status"),
            "method_id": detail.get("method_id"),
            "source_method_id": detail.get("source_method_id"),
            "ledger_status": detail.get("ledger_status"),
            "review_candidate_count": int(detail.get("review_candidate_count") or 0),
            "acceptance_record_count": int(detail.get("acceptance_record_count") or 0),
            "accepted_record_count": int(detail.get("accepted_record_count") or 0),
            "rejected_record_count": int(detail.get("rejected_record_count") or 0),
            "invalid_record_count": int(detail.get("invalid_record_count") or 0),
            "unreviewed_candidate_count": int(
                detail.get("unreviewed_candidate_count") or 0
            ),
            "closure_candidate_count": int(detail.get("closure_candidate_count") or 0),
            "closure_candidate_surface_ids": (
                list(closure_candidate_surface_ids)
                if isinstance(closure_candidate_surface_ids, list)
                else []
            ),
            "rejected_surface_ids": (
                list(rejected_surface_ids)
                if isinstance(rejected_surface_ids, list)
                else []
            ),
            "invalid_record_ids": (
                list(invalid_record_ids) if isinstance(invalid_record_ids, list) else []
            ),
            "unreviewed_candidate_surface_ids": (
                list(unreviewed_surface_ids)
                if isinstance(unreviewed_surface_ids, list)
                else []
            ),
            "manual_fixture_closure_candidate_available": (
                detail.get("manual_fixture_closure_candidate_available") is True
            ),
            "manual_fixture_closure_required": (
                detail.get("manual_fixture_closure_required") is True
            ),
            "operator_review_required": detail.get("operator_review_required") is True,
            "physical_fixture_gap_remains": (
                detail.get("physical_fixture_gap_remains") is True
            ),
            "blocked_by": list(blocked_by) if isinstance(blocked_by, list) else [],
            "next_global_step": detail.get("next_global_step"),
            "operator_sequence_count": int(detail.get("operator_sequence_count") or 0),
            "focused_validation": (
                list(focused_validation)
                if isinstance(focused_validation, list)
                else []
            ),
            "violations": list(violations) if isinstance(violations, list) else [],
        }
    return None


def next_wing_proof_plan_summary(results: Sequence[CheckResult]) -> dict[str, object] | None:
    for result in results:
        if result.name != "next_wing_proof_plan_schema":
            continue
        detail = result.detail if isinstance(result.detail, Mapping) else {}
        planned_value_types = detail.get("planned_value_types")
        planned_channels = detail.get("planned_channels")
        readonly_preflight_blocked_by = detail.get("readonly_preflight_blocked_by")
        readonly_preflight_prestart_check_command = detail.get(
            "readonly_preflight_prestart_check_command"
        )
        readonly_preflight_prestart_check_failure_detail_fields = detail.get(
            "readonly_preflight_prestart_check_failure_detail_fields"
        )
        readonly_preflight_review_decision_blocked_by = detail.get(
            "readonly_preflight_review_decision_blocked_by"
        )
        readonly_preflight_review_decision_candidate_prestart_source_ids = detail.get(
            "readonly_preflight_review_decision_candidate_prestart_source_ids"
        )
        global_blockers = detail.get("global_blockers")
        violations = detail.get("violations")
        return {
            "status": result.status,
            "read_only": True,
            "live_write": False,
            "execution_allowed_by_plan": False,
            "operator_required": True,
            "full_autonomous_ready": False,
            "plan_status": detail.get("plan_status"),
            "step_count": int(detail.get("step_count") or 0),
            "planned_value_types": list(planned_value_types) if isinstance(planned_value_types, list) else [],
            "planned_channels": list(planned_channels) if isinstance(planned_channels, list) else [],
            "readonly_preflight_required": (
                detail.get("readonly_preflight_required") is True
            ),
            "readonly_preflight_candidate_available": (
                detail.get("readonly_preflight_candidate_available") is True
            ),
            "readonly_preflight_evidence_status": detail.get(
                "readonly_preflight_evidence_status"
            ),
            "readonly_preflight_blocked_by": (
                list(readonly_preflight_blocked_by)
                if isinstance(readonly_preflight_blocked_by, list)
                else []
            ),
            "readonly_preflight_prestart_check_required": (
                detail.get("readonly_preflight_prestart_check_required") is True
            ),
            "readonly_preflight_prestart_check_must_pass_before_starting_backend": (
                detail.get(
                    "readonly_preflight_prestart_check_must_pass_before_starting_backend"
                )
                is True
            ),
            "readonly_preflight_prestart_check_command": (
                list(readonly_preflight_prestart_check_command)
                if isinstance(readonly_preflight_prestart_check_command, list)
                else []
            ),
            "readonly_preflight_prestart_check_failure_detail_fields": (
                list(readonly_preflight_prestart_check_failure_detail_fields)
                if isinstance(
                    readonly_preflight_prestart_check_failure_detail_fields,
                    list,
                )
                else []
            ),
            "readonly_preflight_step_count": int(
                detail.get("readonly_preflight_step_count") or 0
            ),
            "readonly_preflight_review_decision_required": (
                detail.get("readonly_preflight_review_decision_required") is True
            ),
            "readonly_preflight_review_decision_accepted": (
                detail.get("readonly_preflight_review_decision_accepted") is True
            ),
            "readonly_preflight_review_decision_status": detail.get(
                "readonly_preflight_review_decision_status"
            ),
            "readonly_preflight_review_decision_blocked_by": (
                list(readonly_preflight_review_decision_blocked_by)
                if isinstance(readonly_preflight_review_decision_blocked_by, list)
                else []
            ),
            "readonly_preflight_review_decision_accepted_record_count": int(
                detail.get("readonly_preflight_review_decision_accepted_record_count") or 0
            ),
            "readonly_preflight_review_decision_accepted_source_ids": (
                list(detail.get("readonly_preflight_review_decision_accepted_source_ids"))
                if isinstance(
                    detail.get("readonly_preflight_review_decision_accepted_source_ids"),
                    list,
                )
                else []
            ),
            "readonly_preflight_review_decision_candidate_prestart_contract_valid": (
                detail.get(
                    "readonly_preflight_review_decision_candidate_prestart_contract_valid"
                )
                is True
            ),
            "readonly_preflight_review_decision_candidate_prestart_source_ids": (
                list(readonly_preflight_review_decision_candidate_prestart_source_ids)
                if isinstance(
                    readonly_preflight_review_decision_candidate_prestart_source_ids,
                    list,
                )
                else []
            ),
            "readonly_preflight_review_decision_rejected_source_ids": (
                list(detail.get("readonly_preflight_review_decision_rejected_source_ids"))
                if isinstance(
                    detail.get("readonly_preflight_review_decision_rejected_source_ids"),
                    list,
                )
                else []
            ),
            "readonly_preflight_review_decision_invalid_record_count": int(
                detail.get("readonly_preflight_review_decision_invalid_record_count") or 0
            ),
            "readonly_preflight_review_decision_invalid_record_ids": (
                list(detail.get("readonly_preflight_review_decision_invalid_record_ids"))
                if isinstance(
                    detail.get("readonly_preflight_review_decision_invalid_record_ids"),
                    list,
                )
                else []
            ),
            "readonly_preflight_review_decision_duplicate_source_ids": (
                list(detail.get("readonly_preflight_review_decision_duplicate_source_ids"))
                if isinstance(
                    detail.get("readonly_preflight_review_decision_duplicate_source_ids"),
                    list,
                )
                else []
            ),
            "readonly_preflight_review_decision_conflicting_source_ids": (
                list(detail.get("readonly_preflight_review_decision_conflicting_source_ids"))
                if isinstance(
                    detail.get("readonly_preflight_review_decision_conflicting_source_ids"),
                    list,
                )
                else []
            ),
            "readonly_preflight_review_decision_unreviewed_candidate_count": int(
                detail.get("readonly_preflight_review_decision_unreviewed_candidate_count") or 0
            ),
            "global_blockers": list(global_blockers) if isinstance(global_blockers, list) else [],
            "violations": list(violations) if isinstance(violations, list) else [],
        }
    return None


def write_summary(run_dir: Path, results: Sequence[CheckResult], *, generated_at: str) -> dict[str, object]:
    failed = [result.name for result in results if result.status != "passed"]
    gap_rollup = autonomy_gap_rollup_summary(results)
    autonomy_overlay = autonomy_evidence_overlay_summary(results)
    gain_dry_run = supervised_gain_dry_run_summary(results)
    read_fixture = read_surface_fixture_evidence_summary(results)
    read_remediation = read_surface_remediation_plan_summary(results)
    read_capture_plan = read_surface_physical_readback_capture_plan_summary(results)
    read_capture_packet_artifact = (
        read_surface_physical_readback_capture_packet_artifact_summary(results)
    )
    read_capture_template_artifact = (
        read_surface_physical_readback_capture_template_artifact_summary(results)
    )
    wing_readonly_preflight = wing_readonly_preflight_evidence_summary(results)
    wing_readonly_preflight_plan_artifact = (
        wing_readonly_preflight_plan_artifact_summary(results)
    )
    wing_readonly_preflight_handoff_packet_artifact = (
        wing_readonly_preflight_handoff_packet_artifact_summary(results)
    )
    wing_readonly_preflight_review_decision = (
        wing_readonly_preflight_review_decision_summary(results)
    )
    raw_mixer_contract = raw_mixer_readback_contract_summary(results)
    raw_mixer_intake = raw_mixer_readback_artifact_intake_summary(results)
    channel_basic_contract = channel_basic_controls_readback_contract_summary(results)
    channel_basic_intake = channel_basic_controls_readback_artifact_intake_summary(results)
    channel_processing_contract = channel_processing_readback_contract_summary(results)
    channel_processing_intake = channel_processing_readback_artifact_intake_summary(results)
    bus_group_main_dca_contract = bus_group_main_dca_readback_contract_summary(results)
    bus_group_main_dca_intake = bus_group_main_dca_readback_artifact_intake_summary(results)
    matrix_state_contract = matrix_state_readback_contract_summary(results)
    matrix_state_intake = matrix_state_readback_artifact_intake_summary(results)
    fx_slots_inserts_contract = fx_slots_inserts_readback_contract_summary(results)
    fx_slots_inserts_intake = fx_slots_inserts_readback_artifact_intake_summary(results)
    routing_sends_contract = routing_sends_readback_contract_summary(results)
    routing_sends_intake = routing_sends_readback_artifact_intake_summary(results)
    snapshots_scenes_contract = snapshots_scenes_readback_contract_summary(results)
    snapshots_scenes_intake = snapshots_scenes_readback_artifact_intake_summary(results)
    user_presets_contract = user_presets_readback_contract_summary(results)
    user_presets_intake = user_presets_readback_artifact_intake_summary(results)
    dlive_state_contract = dlive_state_readback_contract_summary(results)
    dlive_state_intake = dlive_state_readback_artifact_intake_summary(results)
    artifact_review_queue = read_surface_physical_readback_artifact_review_queue_summary(
        results
    )
    artifact_review_queue_artifact = (
        read_surface_physical_readback_artifact_review_queue_artifact_summary(results)
    )
    acceptance_template_artifact = (
        read_surface_physical_readback_acceptance_template_artifact_summary(results)
    )
    fixture_acceptance_ledger = (
        read_surface_physical_readback_fixture_acceptance_ledger_summary(results)
    )
    fixture_closure_plan = read_surface_physical_readback_fixture_closure_plan_summary(
        results
    )
    fixture_closure_plan_artifact = (
        read_surface_physical_readback_fixture_closure_plan_artifact_summary(results)
    )
    next_plan = next_wing_proof_plan_summary(results)
    summary = {
        "type": "no_wing_release_check",
        "generated_at": generated_at,
        "status": "passed" if not failed else "failed",
        "connection_policy": "no_hardware_no_network_no_live_writes",
        "live_write": False,
        "failed_checks": failed,
        "results": [asdict(result) for result in results],
    }
    if gap_rollup is not None:
        summary["autonomy_gap_rollup"] = gap_rollup
    if autonomy_overlay is not None:
        summary["autonomy_evidence_overlay"] = autonomy_overlay
    if gain_dry_run is not None:
        summary["supervised_gain_dry_run"] = gain_dry_run
    if read_fixture is not None:
        summary["read_surface_fixture_evidence"] = read_fixture
    if read_remediation is not None:
        summary["read_surface_remediation_plan"] = read_remediation
    if read_capture_plan is not None:
        summary["read_surface_physical_readback_capture_plan"] = read_capture_plan
    if read_capture_packet_artifact is not None:
        summary["read_surface_physical_readback_capture_packet_artifact"] = (
            read_capture_packet_artifact
        )
    if read_capture_template_artifact is not None:
        summary["read_surface_physical_readback_capture_template_artifact"] = (
            read_capture_template_artifact
        )
    if wing_readonly_preflight is not None:
        summary["wing_readonly_preflight_evidence"] = wing_readonly_preflight
    if wing_readonly_preflight_plan_artifact is not None:
        summary["wing_readonly_preflight_plan_artifact"] = (
            wing_readonly_preflight_plan_artifact
        )
    if wing_readonly_preflight_handoff_packet_artifact is not None:
        summary["wing_readonly_preflight_handoff_packet_artifact"] = (
            wing_readonly_preflight_handoff_packet_artifact
        )
    if wing_readonly_preflight_review_decision is not None:
        summary["wing_readonly_preflight_review_decision_ledger"] = (
            wing_readonly_preflight_review_decision
        )
    if raw_mixer_contract is not None:
        summary["raw_mixer_state_physical_readback_contract"] = raw_mixer_contract
    if raw_mixer_intake is not None:
        summary["raw_mixer_state_physical_readback_artifact_intake"] = raw_mixer_intake
    if channel_basic_contract is not None:
        summary["channel_basic_controls_physical_readback_contract"] = channel_basic_contract
    if channel_basic_intake is not None:
        summary["channel_basic_controls_physical_readback_artifact_intake"] = channel_basic_intake
    if channel_processing_contract is not None:
        summary["channel_processing_physical_readback_contract"] = channel_processing_contract
    if channel_processing_intake is not None:
        summary["channel_processing_physical_readback_artifact_intake"] = (
            channel_processing_intake
        )
    if bus_group_main_dca_contract is not None:
        summary["bus_group_main_dca_physical_readback_contract"] = bus_group_main_dca_contract
    if bus_group_main_dca_intake is not None:
        summary["bus_group_main_dca_physical_readback_artifact_intake"] = (
            bus_group_main_dca_intake
        )
    if matrix_state_contract is not None:
        summary["matrix_state_physical_readback_contract"] = matrix_state_contract
    if matrix_state_intake is not None:
        summary["matrix_state_physical_readback_artifact_intake"] = matrix_state_intake
    if fx_slots_inserts_contract is not None:
        summary["fx_slots_inserts_physical_readback_contract"] = fx_slots_inserts_contract
    if fx_slots_inserts_intake is not None:
        summary["fx_slots_inserts_physical_readback_artifact_intake"] = (
            fx_slots_inserts_intake
        )
    if routing_sends_contract is not None:
        summary["routing_sends_physical_readback_contract"] = routing_sends_contract
    if routing_sends_intake is not None:
        summary["routing_sends_physical_readback_artifact_intake"] = routing_sends_intake
    if snapshots_scenes_contract is not None:
        summary["snapshots_scenes_physical_readback_contract"] = snapshots_scenes_contract
    if snapshots_scenes_intake is not None:
        summary["snapshots_scenes_physical_readback_artifact_intake"] = (
            snapshots_scenes_intake
        )
    if user_presets_contract is not None:
        summary["user_presets_physical_readback_contract"] = user_presets_contract
    if user_presets_intake is not None:
        summary["user_presets_physical_readback_artifact_intake"] = user_presets_intake
    if dlive_state_contract is not None:
        summary["dlive_state_physical_readback_contract"] = dlive_state_contract
    if dlive_state_intake is not None:
        summary["dlive_state_physical_readback_artifact_intake"] = dlive_state_intake
    if artifact_review_queue is not None:
        summary["read_surface_physical_readback_artifact_review_queue"] = (
            artifact_review_queue
        )
    if artifact_review_queue_artifact is not None:
        summary["read_surface_physical_readback_artifact_review_queue_artifact"] = (
            artifact_review_queue_artifact
        )
    if acceptance_template_artifact is not None:
        summary["read_surface_physical_readback_acceptance_template_artifact"] = (
            acceptance_template_artifact
        )
    if fixture_acceptance_ledger is not None:
        summary["read_surface_physical_readback_fixture_acceptance_ledger"] = (
            fixture_acceptance_ledger
        )
    if fixture_closure_plan is not None:
        summary["read_surface_physical_readback_fixture_closure_plan"] = (
            fixture_closure_plan
        )
    if fixture_closure_plan_artifact is not None:
        summary["read_surface_physical_readback_fixture_closure_plan_artifact"] = (
            fixture_closure_plan_artifact
        )
    if next_plan is not None:
        summary["next_wing_proof_plan"] = next_plan
    summary_path = run_dir / WING_READONLY_PREFLIGHT_SUMMARY_ARTIFACT_NAME
    _write_text(summary_path, json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n")

    lines = [
        "# No-WING Release Check",
        "",
        f"- generated_at: `{generated_at}`",
        f"- status: `{summary['status']}`",
        "- connection_policy: `no_hardware_no_network_no_live_writes`",
        "- live_write: `false`",
        "",
        "## Checks",
        "",
    ]
    for result in results:
        lines.append(f"- `{result.name}`: `{result.status}`")
        lines.extend(_backend_port_markdown_details(result))
    if gap_rollup is not None:
        lines.extend(
            [
                "",
                "## Autonomy Gap Rollup",
                "",
                f"- schema_check: `{gap_rollup['status']}`",
                f"- read_only: `{str(gap_rollup['read_only']).lower()}`",
                f"- live_write: `{str(gap_rollup['live_write']).lower()}`",
                f"- full_autonomous_ready: `{str(gap_rollup['full_autonomous_ready']).lower()}`",
                "- full_auto_gap_bucket_counts: "
                f"`{json.dumps(gap_rollup['full_auto_gap_bucket_counts'], sort_keys=True)}`",
                "- evidence_requirement_ids_by_action_family: "
                f"`{json.dumps(gap_rollup['evidence_requirement_ids_by_action_family'], sort_keys=True)}`",
                "- next_required_evidence_by_action_family: "
                f"`{json.dumps(gap_rollup['next_required_evidence_by_action_family'], sort_keys=True)}`",
                "- supervised_current_live_write_kinds: "
                f"`{json.dumps(gap_rollup['supervised_current_live_write_kinds'])}`",
                "- supervised_current_live_write_surfaces: "
                f"`{json.dumps(gap_rollup['supervised_current_live_write_surfaces'])}`",
                "- policy_supervised_live_write_kinds: "
                f"`{json.dumps(gap_rollup['policy_supervised_live_write_kinds'])}`",
                "- policy_supervised_live_write_surfaces: "
                f"`{json.dumps(gap_rollup['policy_supervised_live_write_surfaces'])}`",
                "- unexpected_supervised_live_write_surfaces: "
                f"`{json.dumps(gap_rollup['unexpected_supervised_live_write_surfaces'])}`",
                f"- violations: `{json.dumps(gap_rollup['violations'])}`",
            ]
        )
    if autonomy_overlay is not None:
        lines.extend(
            [
                "",
                "## Autonomy Evidence Overlay",
                "",
                f"- schema_check: `{autonomy_overlay['status']}`",
                f"- read_only: `{str(autonomy_overlay['read_only']).lower()}`",
                f"- live_write: `{str(autonomy_overlay['live_write']).lower()}`",
                f"- live_widening_allowed: `{str(autonomy_overlay['live_widening_allowed']).lower()}`",
                f"- automatic_safety_update_allowed: `{str(autonomy_overlay['automatic_safety_update_allowed']).lower()}`",
                f"- full_autonomous_ready: `{str(autonomy_overlay['full_autonomous_ready']).lower()}`",
                f"- operator_required: `{str(autonomy_overlay['operator_required']).lower()}`",
                f"- overlay_status: `{autonomy_overlay['overlay_status']}`",
                f"- row_count: `{autonomy_overlay['row_count']}`",
                "- status_counts: "
                f"`{json.dumps(autonomy_overlay['status_counts'], sort_keys=True)}`",
                "- dry_run_only_target_ids: "
                f"`{json.dumps(autonomy_overlay['dry_run_only_target_ids'])}`",
                "- manual_safety_review_candidate_target_ids: "
                f"`{json.dumps(autonomy_overlay['manual_safety_review_candidate_target_ids'])}`",
                f"- violations: `{json.dumps(autonomy_overlay['violations'])}`",
            ]
        )
    if gain_dry_run is not None:
        lines.extend(
            [
                "",
                "## Supervised Gain Dry-Run Proof",
                "",
                f"- schema_check: `{gain_dry_run['status']}`",
                f"- read_only: `{str(gain_dry_run['read_only']).lower()}`",
                f"- live_write: `{str(gain_dry_run['live_write']).lower()}`",
                f"- promotion_allowed: `{str(gain_dry_run['promotion_allowed']).lower()}`",
                f"- autonomous_ready: `{str(gain_dry_run['autonomous_ready']).lower()}`",
                f"- gain_dry_run_row_count: `{gain_dry_run['gain_dry_run_row_count']}`",
                "- dry_run_ready_proof_ids: "
                f"`{json.dumps(gain_dry_run['dry_run_ready_proof_ids'])}`",
                f"- violations: `{json.dumps(gain_dry_run['violations'])}`",
            ]
        )
    if read_fixture is not None:
        lines.extend(
            [
                "",
                "## Read Surface Fixture Evidence",
                "",
                f"- schema_check: `{read_fixture['status']}`",
                f"- read_only: `{str(read_fixture['read_only']).lower()}`",
                f"- live_write: `{str(read_fixture['live_write']).lower()}`",
                f"- full_autonomous_ready: `{str(read_fixture['full_autonomous_ready']).lower()}`",
                f"- snapshot_status: `{read_fixture['snapshot_status']}`",
                f"- surface_count: `{read_fixture['surface_count']}`",
                f"- surface_with_fixture_count: `{read_fixture['surface_with_fixture_count']}`",
                f"- declared_fixture_path_count: `{read_fixture['declared_fixture_path_count']}`",
                f"- declared_test_name_count: `{read_fixture['declared_test_name_count']}`",
                f"- physical_gap_surface_count: `{read_fixture['physical_gap_surface_count']}`",
                f"- summary_recomputed_from_rows: `{str(read_fixture['summary_recomputed_from_rows']).lower()}`",
                f"- fixture_declaration_validation: `{str(read_fixture['fixture_declaration_validation']).lower()}`",
                f"- contract_gap_consistency_enforced: `{str(read_fixture['contract_gap_consistency_enforced']).lower()}`",
                "- status_counts: "
                f"`{json.dumps(read_fixture['status_counts'], sort_keys=True)}`",
                "- physical_fixture_missing_surface_ids: "
                f"`{json.dumps(read_fixture['physical_fixture_missing_surface_ids'])}`",
                "- next_physical_readback_surface_id: "
                f"`{read_fixture['next_physical_readback_surface_id']}`",
                f"- violations: `{json.dumps(read_fixture['violations'])}`",
            ]
        )
    if read_remediation is not None:
        lines.extend(
            [
                "",
                "## Read Surface Remediation Plan",
                "",
                f"- schema_check: `{read_remediation['status']}`",
                f"- read_only: `{str(read_remediation['read_only']).lower()}`",
                f"- live_write: `{str(read_remediation['live_write']).lower()}`",
                f"- mutates_console_state: `{str(read_remediation['mutates_console_state']).lower()}`",
                f"- live_widening_allowed: `{str(read_remediation['live_widening_allowed']).lower()}`",
                f"- full_autonomous_ready: `{str(read_remediation['full_autonomous_ready']).lower()}`",
                f"- plan_status: `{read_remediation['plan_status']}`",
                f"- action_count: `{read_remediation['action_count']}`",
                f"- incomplete_surface_count: `{read_remediation['incomplete_surface_count']}`",
                f"- physical_gap_surface_count: `{read_remediation['physical_gap_surface_count']}`",
                "- physical_fixture_missing_surface_ids: "
                f"`{json.dumps(read_remediation['physical_fixture_missing_surface_ids'])}`",
                "- next_physical_readback_surface_id: "
                f"`{read_remediation['next_physical_readback_surface_id']}`",
                "- next_physical_readback_action_id: "
                f"`{read_remediation['next_physical_readback_action_id']}`",
                "- next_physical_readback_target_action_ids: "
                f"`{json.dumps(read_remediation['next_physical_readback_target_action_ids'])}`",
                f"- target_action_consistency: `{str(read_remediation['target_action_consistency']).lower()}`",
                f"- violations: `{json.dumps(read_remediation['violations'])}`",
            ]
        )
    if read_capture_plan is not None:
        lines.extend(
            [
                "",
                "## Read Surface Physical Readback Capture Plan",
                "",
                f"- schema_check: `{read_capture_plan['status']}`",
                f"- read_only: `{str(read_capture_plan['read_only']).lower()}`",
                f"- live_write: `{str(read_capture_plan['live_write']).lower()}`",
                f"- mutates_console_state: `{str(read_capture_plan['mutates_console_state']).lower()}`",
                f"- queries_console: `{str(read_capture_plan['queries_console']).lower()}`",
                f"- executes_tests: `{str(read_capture_plan['executes_tests']).lower()}`",
                f"- operator_required: `{str(read_capture_plan['operator_required']).lower()}`",
                "- execution_allowed_by_plan: "
                f"`{str(read_capture_plan['execution_allowed_by_plan']).lower()}`",
                "- live_authority_granted: "
                f"`{str(read_capture_plan['live_authority_granted']).lower()}`",
                f"- full_autonomous_ready: `{str(read_capture_plan['full_autonomous_ready']).lower()}`",
                f"- plan_status: `{read_capture_plan['plan_status']}`",
                f"- capture_step_count: `{read_capture_plan['capture_step_count']}`",
                f"- physical_gap_surface_count: `{read_capture_plan['physical_gap_surface_count']}`",
                "- physical_fixture_missing_surface_ids: "
                f"`{json.dumps(read_capture_plan['physical_fixture_missing_surface_ids'])}`",
                "- next_physical_readback_surface_id: "
                f"`{read_capture_plan['next_physical_readback_surface_id']}`",
                "- next_physical_readback_action_id: "
                f"`{read_capture_plan['next_physical_readback_action_id']}`",
                f"- target_action_consistency: `{str(read_capture_plan['target_action_consistency']).lower()}`",
                "- required_artifact_contracts: "
                f"`{json.dumps(read_capture_plan['required_artifact_contracts'])}`",
                f"- blocked_by: `{json.dumps(read_capture_plan['blocked_by'])}`",
                f"- step_surface_id: `{read_capture_plan['step_surface_id']}`",
                f"- step_remediation_action_id: `{read_capture_plan['step_remediation_action_id']}`",
                f"- violations: `{json.dumps(read_capture_plan['violations'])}`",
            ]
        )
    if read_capture_packet_artifact is not None:
        lines.extend(
            [
                "",
                "## Read Surface Physical Readback Capture Packet Artifact",
                "",
                f"- schema_check: `{read_capture_packet_artifact['status']}`",
                f"- read_only: `{str(read_capture_packet_artifact['read_only']).lower()}`",
                f"- live_write: `{str(read_capture_packet_artifact['live_write']).lower()}`",
                f"- mutates_console_state: `{str(read_capture_packet_artifact['mutates_console_state']).lower()}`",
                f"- queries_console: `{str(read_capture_packet_artifact['queries_console']).lower()}`",
                f"- executes_tests: `{str(read_capture_packet_artifact['executes_tests']).lower()}`",
                f"- operator_required: `{str(read_capture_packet_artifact['operator_required']).lower()}`",
                "- execution_allowed_by_packet: "
                f"`{str(read_capture_packet_artifact['execution_allowed_by_packet']).lower()}`",
                "- live_authority_granted: "
                f"`{str(read_capture_packet_artifact['live_authority_granted']).lower()}`",
                f"- full_autonomous_ready: `{str(read_capture_packet_artifact['full_autonomous_ready']).lower()}`",
                f"- packet_path: `{read_capture_packet_artifact['packet_path']}`",
                f"- packet_status: `{read_capture_packet_artifact['packet_status']}`",
                f"- source_plan_status: `{read_capture_packet_artifact['source_plan_status']}`",
                f"- capture_step_count: `{read_capture_packet_artifact['capture_step_count']}`",
                f"- physical_gap_surface_count: `{read_capture_packet_artifact['physical_gap_surface_count']}`",
                "- physical_fixture_missing_surface_ids: "
                f"`{json.dumps(read_capture_packet_artifact['physical_fixture_missing_surface_ids'])}`",
                "- remaining_capture_target_count: "
                f"`{read_capture_packet_artifact['remaining_capture_target_count']}`",
                "- remaining_capture_target_surface_ids: "
                f"`{json.dumps(read_capture_packet_artifact['remaining_capture_target_surface_ids'])}`",
                "- remaining_capture_target_action_ids: "
                f"`{json.dumps(read_capture_packet_artifact['remaining_capture_target_action_ids'])}`",
                "- remaining_capture_target_contract_ids: "
                f"`{json.dumps(read_capture_packet_artifact['remaining_capture_target_contract_ids'])}`",
                "- remaining_capture_target_forbidden_flags: "
                f"`{json.dumps(read_capture_packet_artifact['remaining_capture_target_forbidden_flags'])}`",
                "- next_physical_readback_surface_id: "
                f"`{read_capture_packet_artifact['next_physical_readback_surface_id']}`",
                "- next_physical_readback_action_id: "
                f"`{read_capture_packet_artifact['next_physical_readback_action_id']}`",
                f"- target_action_consistency: `{str(read_capture_packet_artifact['target_action_consistency']).lower()}`",
                "- required_artifact_contracts: "
                f"`{json.dumps(read_capture_packet_artifact['required_artifact_contracts'])}`",
                f"- blocked_by: `{json.dumps(read_capture_packet_artifact['blocked_by'])}`",
                f"- step_surface_id: `{read_capture_packet_artifact['step_surface_id']}`",
                "- step_target_console_surface: "
                f"`{read_capture_packet_artifact['step_target_console_surface']}`",
                f"- step_read_mode: `{read_capture_packet_artifact['step_read_mode']}`",
                "- step_contract_method_id: "
                f"`{read_capture_packet_artifact['step_contract_method_id']}`",
                "- step_forbidden_actions: "
                f"`{json.dumps(read_capture_packet_artifact['step_forbidden_actions'])}`",
                "- focused_validation: "
                f"`{json.dumps(read_capture_packet_artifact['focused_validation'])}`",
                f"- violations: `{json.dumps(read_capture_packet_artifact['violations'])}`",
            ]
        )
    if read_capture_template_artifact is not None:
        lines.extend(
            [
                "",
                "## Read Surface Physical Readback Capture Template Artifact",
                "",
                f"- schema_check: `{read_capture_template_artifact['status']}`",
                f"- read_only: `{str(read_capture_template_artifact['read_only']).lower()}`",
                f"- live_write: `{str(read_capture_template_artifact['live_write']).lower()}`",
                f"- mutates_console_state: `{str(read_capture_template_artifact['mutates_console_state']).lower()}`",
                f"- queries_console: `{str(read_capture_template_artifact['queries_console']).lower()}`",
                f"- executes_tests: `{str(read_capture_template_artifact['executes_tests']).lower()}`",
                f"- operator_required: `{str(read_capture_template_artifact['operator_required']).lower()}`",
                "- execution_allowed_by_template: "
                f"`{str(read_capture_template_artifact['execution_allowed_by_template']).lower()}`",
                "- accepted_as_physical_fixture_candidate: "
                f"`{str(read_capture_template_artifact['accepted_as_physical_fixture_candidate']).lower()}`",
                "- fixture_closure_allowed_by_template: "
                f"`{str(read_capture_template_artifact['fixture_closure_allowed_by_template']).lower()}`",
                "- live_authority_granted: "
                f"`{str(read_capture_template_artifact['live_authority_granted']).lower()}`",
                f"- full_autonomous_ready: `{str(read_capture_template_artifact['full_autonomous_ready']).lower()}`",
                f"- template_path: `{read_capture_template_artifact['template_path']}`",
                f"- template_status: `{read_capture_template_artifact['template_status']}`",
                f"- source_packet_status: `{read_capture_template_artifact['source_packet_status']}`",
                f"- selected_surface_id: `{read_capture_template_artifact['selected_surface_id']}`",
                f"- include_all_targets: `{str(read_capture_template_artifact['include_all_targets']).lower()}`",
                "- all_remaining_target_count: "
                f"`{read_capture_template_artifact['all_remaining_target_count']}`",
                "- capture_template_count: "
                f"`{read_capture_template_artifact['capture_template_count']}`",
                "- capture_template_surface_ids: "
                f"`{json.dumps(read_capture_template_artifact['capture_template_surface_ids'])}`",
                "- capture_template_contract_ids: "
                f"`{json.dumps(read_capture_template_artifact['capture_template_contract_ids'])}`",
                "- payload_template_contract_ids: "
                f"`{json.dumps(read_capture_template_artifact['payload_template_contract_ids'])}`",
                "- payload_template_artifact_types: "
                f"`{json.dumps(read_capture_template_artifact['payload_template_artifact_types'])}`",
                "- payload_template_schema_ids: "
                f"`{json.dumps(read_capture_template_artifact['payload_template_schema_ids'])}`",
                "- payload_template_field_counts: "
                f"`{json.dumps(read_capture_template_artifact['payload_template_field_counts'])}`",
                "- required_operator_edits: "
                f"`{json.dumps(read_capture_template_artifact['required_operator_edits'])}`",
                f"- blocked_by: `{json.dumps(read_capture_template_artifact['blocked_by'])}`",
                "- record_unsafe_flags: "
                f"`{json.dumps(read_capture_template_artifact['record_unsafe_flags'])}`",
                "- focused_validation: "
                f"`{json.dumps(read_capture_template_artifact['focused_validation'])}`",
                f"- violations: `{json.dumps(read_capture_template_artifact['violations'])}`",
            ]
        )
    if wing_readonly_preflight is not None:
        lines.extend(
            [
                "",
                "## WING Read-Only Preflight Evidence",
                "",
                f"- schema_check: `{wing_readonly_preflight['status']}`",
                f"- read_only: `{str(wing_readonly_preflight['read_only']).lower()}`",
                f"- live_write: `{str(wing_readonly_preflight['live_write']).lower()}`",
                f"- queries_console: `{str(wing_readonly_preflight['queries_console']).lower()}`",
                f"- executes_tests: `{str(wing_readonly_preflight['executes_tests']).lower()}`",
                f"- operator_required: `{str(wing_readonly_preflight['operator_required']).lower()}`",
                f"- readiness_closure_allowed_by_evidence: `{str(wing_readonly_preflight['readiness_closure_allowed_by_evidence']).lower()}`",
                f"- live_authority_granted: `{str(wing_readonly_preflight['live_authority_granted']).lower()}`",
                f"- full_autonomous_ready: `{str(wing_readonly_preflight['full_autonomous_ready']).lower()}`",
                f"- evidence_status: `{wing_readonly_preflight['evidence_status']}`",
                f"- artifact_count: `{wing_readonly_preflight['artifact_count']}`",
                f"- valid_candidate_count: `{wing_readonly_preflight['valid_candidate_count']}`",
                f"- invalid_candidate_count: `{wing_readonly_preflight['invalid_candidate_count']}`",
                f"- readiness_update_candidate_available: `{str(wing_readonly_preflight['readiness_update_candidate_available']).lower()}`",
                "- mvp_blocker_matrix_ids: "
                f"`{json.dumps(wing_readonly_preflight['mvp_blocker_matrix_ids'])}`",
                "- mvp_blockers_closed_by_evidence: "
                f"`{json.dumps(wing_readonly_preflight['mvp_blockers_closed_by_evidence'])}`",
                "- latest_valid_candidate_run_id: "
                f"`{wing_readonly_preflight['latest_valid_candidate_run_id']}`",
                "- artifact_operator_next_steps: "
                f"`{json.dumps(wing_readonly_preflight['artifact_operator_next_steps'])}`",
                f"- capture_tool_path: `{wing_readonly_preflight['capture_tool_path']}`",
                "- capture_tool_available: "
                f"`{str(wing_readonly_preflight['capture_tool_available']).lower()}`",
                "- plan_command: "
                f"`{json.dumps(wing_readonly_preflight['plan_command'])}`",
                "- plan_command_uses_plan_only: "
                f"`{str(wing_readonly_preflight['plan_command_uses_plan_only']).lower()}`",
                f"- plan_command_output: `{wing_readonly_preflight['plan_command_output']}`",
                f"- plan_command_channels: `{wing_readonly_preflight['plan_command_channels']}`",
                "- plan_command_operator_confirmed_channels: "
                f"`{wing_readonly_preflight['plan_command_operator_confirmed_channels']}`",
                f"- plan_command_value_types: `{wing_readonly_preflight['plan_command_value_types']}`",
                f"- plan_command_artifact_root: `{wing_readonly_preflight['plan_command_artifact_root']}`",
                "- plan_command_matches_capture: "
                f"`{str(wing_readonly_preflight['plan_command_matches_capture']).lower()}`",
                "- prestart_backend_port_check_required: "
                f"`{str(wing_readonly_preflight['prestart_backend_port_check_required']).lower()}`",
                "- prestart_backend_port_check_must_pass_before_starting_backend: "
                f"`{str(wing_readonly_preflight['prestart_backend_port_check_must_pass_before_starting_backend']).lower()}`",
                "- prestart_backend_port_check_command: "
                f"`{json.dumps(wing_readonly_preflight['prestart_backend_port_check_command'])}`",
                "- prestart_backend_port_check_failure_detail_fields: "
                f"`{json.dumps(wing_readonly_preflight['prestart_backend_port_check_failure_detail_fields'])}`",
                "- capture_command_template: "
                f"`{json.dumps(wing_readonly_preflight['capture_command_template'])}`",
                f"- capture_command_channels: `{wing_readonly_preflight['capture_command_channels']}`",
                "- capture_command_operator_confirmed_channels: "
                f"`{wing_readonly_preflight['capture_command_operator_confirmed_channels']}`",
                "- capture_command_requires_operator_channel_confirmation: "
                f"`{str(wing_readonly_preflight['capture_command_requires_operator_channel_confirmation']).lower()}`",
                "- capture_command_operator_confirmation_matches_channels: "
                f"`{str(wing_readonly_preflight['capture_command_operator_confirmation_matches_channels']).lower()}`",
                "- capture_command_requires_physical_readback: "
                f"`{str(wing_readonly_preflight['capture_command_requires_physical_readback']).lower()}`",
                f"- capture_command_expected_plan: `{wing_readonly_preflight['capture_command_expected_plan']}`",
                "- capture_command_expected_plan_matches_plan_output: "
                f"`{str(wing_readonly_preflight['capture_command_expected_plan_matches_plan_output']).lower()}`",
                f"- review_tool_path: `{wing_readonly_preflight['review_tool_path']}`",
                "- review_tool_available: "
                f"`{str(wing_readonly_preflight['review_tool_available']).lower()}`",
                "- review_command: "
                f"`{json.dumps(wing_readonly_preflight['review_command'])}`",
                "- review_command_writes_decision_template: "
                f"`{str(wing_readonly_preflight['review_command_writes_decision_template']).lower()}`",
                "- review_exit_codes: "
                f"`{json.dumps(wing_readonly_preflight['review_exit_codes'], sort_keys=True)}`",
                "- review_manual_checklist_ids: "
                f"`{json.dumps(wing_readonly_preflight['review_manual_checklist_ids'])}`",
                "- review_manual_checklist_complete: "
                f"`{str(wing_readonly_preflight['review_manual_checklist_complete']).lower()}`",
                "- review_decision_template_required_fields: "
                f"`{json.dumps(wing_readonly_preflight['review_decision_template_required_fields'])}`",
                "- review_decision_template_contract_complete: "
                f"`{str(wing_readonly_preflight['review_decision_template_contract_complete']).lower()}`",
                f"- handoff_packet_tool_path: `{wing_readonly_preflight['handoff_packet_tool_path']}`",
                "- handoff_packet_tool_available: "
                f"`{str(wing_readonly_preflight['handoff_packet_tool_available']).lower()}`",
                "- handoff_packet_command: "
                f"`{json.dumps(wing_readonly_preflight['handoff_packet_command'])}`",
                f"- handoff_packet_output: `{wing_readonly_preflight['handoff_packet_output']}`",
                f"- handoff_packet_plan_output: `{wing_readonly_preflight['handoff_packet_plan_output']}`",
                f"- handoff_packet_url: `{wing_readonly_preflight['handoff_packet_url']}`",
                "- handoff_packet_artifact_root: "
                f"`{wing_readonly_preflight['handoff_packet_artifact_root']}`",
                f"- handoff_packet_channels: `{wing_readonly_preflight['handoff_packet_channels']}`",
                "- handoff_packet_value_types: "
                f"`{wing_readonly_preflight['handoff_packet_value_types']}`",
                "- handoff_packet_url_matches_capture: "
                f"`{str(wing_readonly_preflight['handoff_packet_url_matches_capture']).lower()}`",
                "- handoff_packet_artifact_root_matches_capture: "
                f"`{str(wing_readonly_preflight['handoff_packet_artifact_root_matches_capture']).lower()}`",
                "- handoff_packet_channel_matches_capture: "
                f"`{str(wing_readonly_preflight['handoff_packet_channel_matches_capture']).lower()}`",
                "- handoff_packet_value_types_match_capture: "
                f"`{str(wing_readonly_preflight['handoff_packet_value_types_match_capture']).lower()}`",
                f"- blocked_by: `{json.dumps(wing_readonly_preflight['blocked_by'])}`",
                f"- violations: `{json.dumps(wing_readonly_preflight['violations'])}`",
            ]
        )
    if wing_readonly_preflight_plan_artifact is not None:
        lines.extend(
            [
                "",
                "## WING Read-Only Preflight Plan Artifact",
                "",
                f"- schema_check: `{wing_readonly_preflight_plan_artifact['status']}`",
                f"- read_only: `{str(wing_readonly_preflight_plan_artifact['read_only']).lower()}`",
                f"- live_write: `{str(wing_readonly_preflight_plan_artifact['live_write']).lower()}`",
                f"- contacts_backend: `{str(wing_readonly_preflight_plan_artifact['contacts_backend']).lower()}`",
                f"- queries_console: `{str(wing_readonly_preflight_plan_artifact['queries_console']).lower()}`",
                "- operator_required: "
                f"`{str(wing_readonly_preflight_plan_artifact['operator_required']).lower()}`",
                "- expected_plan_required_for_real_run: "
                f"`{str(wing_readonly_preflight_plan_artifact['expected_plan_required_for_real_run']).lower()}`",
                "- physical_preflight_requires_existing_backend_connection: "
                f"`{str(wing_readonly_preflight_plan_artifact['physical_preflight_requires_existing_backend_connection']).lower()}`",
                "- backend_port_prestart_check_required: "
                f"`{str(wing_readonly_preflight_plan_artifact['backend_port_prestart_check_required']).lower()}`",
                "- backend_port_prestart_check_must_pass_before_starting_backend: "
                f"`{str(wing_readonly_preflight_plan_artifact['backend_port_prestart_check_must_pass_before_starting_backend']).lower()}`",
                "- backend_port_prestart_check_command: "
                f"`{json.dumps(wing_readonly_preflight_plan_artifact['backend_port_prestart_check_command'])}`",
                "- backend_port_prestart_check_failure_detail_fields: "
                f"`{json.dumps(wing_readonly_preflight_plan_artifact['backend_port_prestart_check_failure_detail_fields'])}`",
                "- manual_review_required_before_readiness_update: "
                f"`{str(wing_readonly_preflight_plan_artifact['manual_review_required_before_readiness_update']).lower()}`",
                "- operator_session_confirmation_required: "
                f"`{str(wing_readonly_preflight_plan_artifact['operator_session_confirmation_required']).lower()}`",
                "- operator_confirmed_session_ready: "
                f"`{str(wing_readonly_preflight_plan_artifact['operator_confirmed_session_ready']).lower()}`",
                "- execution_allowed_by_plan: "
                f"`{str(wing_readonly_preflight_plan_artifact['execution_allowed_by_plan']).lower()}`",
                "- live_authority_granted: "
                f"`{str(wing_readonly_preflight_plan_artifact['live_authority_granted']).lower()}`",
                f"- plan_path: `{wing_readonly_preflight_plan_artifact['plan_path']}`",
                f"- plan_status: `{wing_readonly_preflight_plan_artifact['plan_status']}`",
                f"- plan_output: `{wing_readonly_preflight_plan_artifact['plan_output']}`",
                "- selected_mvp_blocker: "
                f"`{wing_readonly_preflight_plan_artifact['selected_mvp_blocker']}`",
                "- selected_mvp_blocker_status: "
                f"`{wing_readonly_preflight_plan_artifact['selected_mvp_blocker_status']}`",
                "- channels: "
                f"`{json.dumps(wing_readonly_preflight_plan_artifact['channels'])}`",
                "- operator_confirmed_channels: "
                f"`{json.dumps(wing_readonly_preflight_plan_artifact['operator_confirmed_channels'])}`",
                "- value_types: "
                f"`{json.dumps(wing_readonly_preflight_plan_artifact['value_types'])}`",
                f"- request_count: `{wing_readonly_preflight_plan_artifact['request_count']}`",
                "- expected_request_count: "
                f"`{wing_readonly_preflight_plan_artifact['expected_request_count']}`",
                "- request_sequence_matches_expected: "
                f"`{str(wing_readonly_preflight_plan_artifact['request_sequence_matches_expected']).lower()}`",
                "- forbidden_request_types: "
                f"`{json.dumps(wing_readonly_preflight_plan_artifact['forbidden_request_types'])}`",
                "- fail_closed_conditions: "
                f"`{json.dumps(wing_readonly_preflight_plan_artifact['fail_closed_conditions'])}`",
                "- post_run_review_commands: "
                f"`{json.dumps(wing_readonly_preflight_plan_artifact['post_run_review_commands'])}`",
                f"- violations: `{json.dumps(wing_readonly_preflight_plan_artifact['violations'])}`",
            ]
        )
    if wing_readonly_preflight_handoff_packet_artifact is not None:
        lines.extend(
            [
                "",
                "## WING Read-Only Preflight Handoff Packet Artifact",
                "",
                "- schema_check: "
                f"`{wing_readonly_preflight_handoff_packet_artifact['status']}`",
                "- read_only: "
                f"`{str(wing_readonly_preflight_handoff_packet_artifact['read_only']).lower()}`",
                "- live_write: "
                f"`{str(wing_readonly_preflight_handoff_packet_artifact['live_write']).lower()}`",
                "- queries_console: "
                f"`{str(wing_readonly_preflight_handoff_packet_artifact['queries_console']).lower()}`",
                "- executes_tests: "
                f"`{str(wing_readonly_preflight_handoff_packet_artifact['executes_tests']).lower()}`",
                "- expected_plan_required_for_real_run: "
                f"`{str(wing_readonly_preflight_handoff_packet_artifact['expected_plan_required_for_real_run']).lower()}`",
                "- physical_preflight_requires_existing_backend_connection: "
                f"`{str(wing_readonly_preflight_handoff_packet_artifact['physical_preflight_requires_existing_backend_connection']).lower()}`",
                "- backend_port_prestart_check_required: "
                f"`{str(wing_readonly_preflight_handoff_packet_artifact['backend_port_prestart_check_required']).lower()}`",
                "- backend_port_prestart_check_must_pass_before_starting_backend: "
                f"`{str(wing_readonly_preflight_handoff_packet_artifact['backend_port_prestart_check_must_pass_before_starting_backend']).lower()}`",
                "- manual_review_required_before_readiness_update: "
                f"`{str(wing_readonly_preflight_handoff_packet_artifact['manual_review_required_before_readiness_update']).lower()}`",
                "- execution_allowed_by_packet: "
                f"`{str(wing_readonly_preflight_handoff_packet_artifact['execution_allowed_by_packet']).lower()}`",
                "- readiness_matrix_update_allowed_by_packet: "
                f"`{str(wing_readonly_preflight_handoff_packet_artifact['readiness_matrix_update_allowed_by_packet']).lower()}`",
                "- readiness_closure_allowed_by_packet: "
                f"`{str(wing_readonly_preflight_handoff_packet_artifact['readiness_closure_allowed_by_packet']).lower()}`",
                "- live_authority_granted: "
                f"`{str(wing_readonly_preflight_handoff_packet_artifact['live_authority_granted']).lower()}`",
                "- full_autonomous_ready: "
                f"`{str(wing_readonly_preflight_handoff_packet_artifact['full_autonomous_ready']).lower()}`",
                "- next_manual_artifact_required: "
                f"`{wing_readonly_preflight_handoff_packet_artifact['next_manual_artifact_required']}`",
                "- packet_path: "
                f"`{wing_readonly_preflight_handoff_packet_artifact['packet_path']}`",
                "- packet_status: "
                f"`{wing_readonly_preflight_handoff_packet_artifact['packet_status']}`",
                "- plan_artifact_status: "
                f"`{wing_readonly_preflight_handoff_packet_artifact['plan_artifact_status']}`",
                "- plan_artifact_violations: "
                f"`{json.dumps(wing_readonly_preflight_handoff_packet_artifact['plan_artifact_violations'])}`",
                "- plan_output: "
                f"`{wing_readonly_preflight_handoff_packet_artifact['plan_output']}`",
                "- command_names: "
                f"`{json.dumps(wing_readonly_preflight_handoff_packet_artifact['command_names'])}`",
                "- preview_command_uses_plan_only: "
                f"`{str(wing_readonly_preflight_handoff_packet_artifact['preview_command_uses_plan_only']).lower()}`",
                "- capture_command_requires_physical_readback: "
                f"`{str(wing_readonly_preflight_handoff_packet_artifact['capture_command_requires_physical_readback']).lower()}`",
                "- prestart_backend_port_check_command: "
                f"`{json.dumps(wing_readonly_preflight_handoff_packet_artifact['prestart_backend_port_check_command'])}`",
                "- backend_port_prestart_check_operator_sequence_step_index: "
                f"`{wing_readonly_preflight_handoff_packet_artifact['backend_port_prestart_check_operator_sequence_step_index']}`",
                "- backend_port_prestart_check_must_be_first_operator_step: "
                f"`{str(wing_readonly_preflight_handoff_packet_artifact['backend_port_prestart_check_must_be_first_operator_step']).lower()}`",
                "- backend_port_prestart_check_operator_sequence_step: "
                f"`{wing_readonly_preflight_handoff_packet_artifact['backend_port_prestart_check_operator_sequence_step']}`",
                "- backend_port_prestart_check_operator_sequence_step_is_first: "
                f"`{str(wing_readonly_preflight_handoff_packet_artifact['backend_port_prestart_check_operator_sequence_step_is_first']).lower()}`",
                "- backend_port_prestart_gate: "
                f"`{json.dumps(wing_readonly_preflight_handoff_packet_artifact['backend_port_prestart_gate'])}`",
                "- expected_backend_port_prestart_gate: "
                f"`{json.dumps(wing_readonly_preflight_handoff_packet_artifact['expected_backend_port_prestart_gate'])}`",
                "- backend_port_prestart_check_failure_detail_fields: "
                f"`{json.dumps(wing_readonly_preflight_handoff_packet_artifact['backend_port_prestart_check_failure_detail_fields'])}`",
                "- review_command_writes_decision_template: "
                f"`{str(wing_readonly_preflight_handoff_packet_artifact['review_command_writes_decision_template']).lower()}`",
                "- blockers_not_closed_by_packet: "
                f"`{json.dumps(wing_readonly_preflight_handoff_packet_artifact['blockers_not_closed_by_packet'])}`",
                "- mvp_blocker_anchor_matches_current_first_open: "
                f"`{str(wing_readonly_preflight_handoff_packet_artifact['mvp_blocker_anchor_matches_current_first_open']).lower()}`",
                "- mvp_blocker_anchor_sha256: "
                f"`{wing_readonly_preflight_handoff_packet_artifact['mvp_blocker_anchor_sha256']}`",
                "- expected_mvp_blocker_anchor_sha256: "
                f"`{wing_readonly_preflight_handoff_packet_artifact['expected_mvp_blocker_anchor_sha256']}`",
                "- mvp_blocker_anchor_sha256_matches_current_first_open: "
                f"`{str(wing_readonly_preflight_handoff_packet_artifact['mvp_blocker_anchor_sha256_matches_current_first_open']).lower()}`",
                "- mvp_blocker_anchor: "
                f"`{json.dumps(wing_readonly_preflight_handoff_packet_artifact['mvp_blocker_anchor'])}`",
                "- expected_mvp_blocker_anchor: "
                f"`{json.dumps(wing_readonly_preflight_handoff_packet_artifact['expected_mvp_blocker_anchor'])}`",
                "- operator_sequence_count: "
                f"`{wing_readonly_preflight_handoff_packet_artifact['operator_sequence_count']}`",
                "- operator_sequence_has_mvp_blocker_anchor_digest_review: "
                f"`{str(wing_readonly_preflight_handoff_packet_artifact['operator_sequence_has_mvp_blocker_anchor_digest_review']).lower()}`",
                "- operator_sequence_mvp_blocker_anchor_digest_review_step: "
                f"`{wing_readonly_preflight_handoff_packet_artifact['operator_sequence_mvp_blocker_anchor_digest_review_step']}`",
                "- operator_sequence_mvp_blocker_anchor_digest_review_step_index: "
                f"`{wing_readonly_preflight_handoff_packet_artifact['operator_sequence_mvp_blocker_anchor_digest_review_step_index']}`",
                "- operator_sequence_has_supervised_prestart_gate: "
                f"`{str(wing_readonly_preflight_handoff_packet_artifact['operator_sequence_has_supervised_prestart_gate']).lower()}`",
                "- operator_sequence_supervised_prestart_gate_step: "
                f"`{wing_readonly_preflight_handoff_packet_artifact['operator_sequence_supervised_prestart_gate_step']}`",
                "- operator_sequence_supervised_prestart_gate_step_index: "
                f"`{wing_readonly_preflight_handoff_packet_artifact['operator_sequence_supervised_prestart_gate_step_index']}`",
                "- operator_sequence_supervised_prestart_gate_is_final: "
                f"`{str(wing_readonly_preflight_handoff_packet_artifact['operator_sequence_supervised_prestart_gate_is_final']).lower()}`",
                "- required_supervised_prestart_gate_markers: "
                f"`{json.dumps(wing_readonly_preflight_handoff_packet_artifact['required_supervised_prestart_gate_markers'])}`",
                "- supervised_prestart_gate: "
                f"`{json.dumps(wing_readonly_preflight_handoff_packet_artifact['supervised_prestart_gate'])}`",
                "- expected_supervised_prestart_gate: "
                f"`{json.dumps(wing_readonly_preflight_handoff_packet_artifact['expected_supervised_prestart_gate'])}`",
                "- violations: "
                f"`{json.dumps(wing_readonly_preflight_handoff_packet_artifact['violations'])}`",
            ]
        )
    if wing_readonly_preflight_review_decision is not None:
        lines.extend(
            [
                "",
                "## WING Read-Only Preflight Review Decision Ledger",
                "",
                f"- schema_check: `{wing_readonly_preflight_review_decision['status']}`",
                f"- read_only: `{str(wing_readonly_preflight_review_decision['read_only']).lower()}`",
                f"- live_write: `{str(wing_readonly_preflight_review_decision['live_write']).lower()}`",
                "- readiness_matrix_update_allowed_by_decision: "
                f"`{str(wing_readonly_preflight_review_decision['readiness_matrix_update_allowed_by_decision']).lower()}`",
                "- readiness_closure_allowed_by_decision: "
                f"`{str(wing_readonly_preflight_review_decision['readiness_closure_allowed_by_decision']).lower()}`",
                "- live_authority_granted: "
                f"`{str(wing_readonly_preflight_review_decision['live_authority_granted']).lower()}`",
                "- full_autonomous_ready: "
                f"`{str(wing_readonly_preflight_review_decision['full_autonomous_ready']).lower()}`",
                f"- ledger_status: `{wing_readonly_preflight_review_decision['ledger_status']}`",
                f"- review_tool_path: `{wing_readonly_preflight_review_decision['review_tool_path']}`",
                "- review_tool_available: "
                f"`{str(wing_readonly_preflight_review_decision['review_tool_available']).lower()}`",
                "- review_command: "
                f"`{json.dumps(wing_readonly_preflight_review_decision['review_command'])}`",
                "- review_exit_codes: "
                f"`{json.dumps(wing_readonly_preflight_review_decision['review_exit_codes'], sort_keys=True)}`",
                "- preflight_evidence_status: "
                f"`{wing_readonly_preflight_review_decision['preflight_evidence_status']}`",
                "- decision_search_roots: "
                f"`{json.dumps(wing_readonly_preflight_review_decision['decision_search_roots'])}`",
                "- review_candidate_count: "
                f"`{wing_readonly_preflight_review_decision['review_candidate_count']}`",
                "- decision_record_count: "
                f"`{wing_readonly_preflight_review_decision['decision_record_count']}`",
                "- accepted_record_count: "
                f"`{wing_readonly_preflight_review_decision['accepted_record_count']}`",
                "- rejected_record_count: "
                f"`{wing_readonly_preflight_review_decision['rejected_record_count']}`",
                "- invalid_record_count: "
                f"`{wing_readonly_preflight_review_decision['invalid_record_count']}`",
                "- unreviewed_candidate_count: "
                f"`{wing_readonly_preflight_review_decision['unreviewed_candidate_count']}`",
                "- manual_readiness_update_candidate_available: "
                f"`{str(wing_readonly_preflight_review_decision['manual_readiness_update_candidate_available']).lower()}`",
                "- accepted_source_ids: "
                f"`{json.dumps(wing_readonly_preflight_review_decision['accepted_source_ids'])}`",
                "- accepted_candidate_prestart_contract_valid: "
                f"`{str(wing_readonly_preflight_review_decision['accepted_candidate_prestart_contract_valid']).lower()}`",
                "- accepted_candidate_prestart_source_ids: "
                f"`{json.dumps(wing_readonly_preflight_review_decision['accepted_candidate_prestart_source_ids'])}`",
                "- rejected_source_ids: "
                f"`{json.dumps(wing_readonly_preflight_review_decision['rejected_source_ids'])}`",
                "- invalid_record_ids: "
                f"`{json.dumps(wing_readonly_preflight_review_decision['invalid_record_ids'])}`",
                "- duplicate_source_ids: "
                f"`{json.dumps(wing_readonly_preflight_review_decision['duplicate_source_ids'])}`",
                "- conflicting_source_ids: "
                f"`{json.dumps(wing_readonly_preflight_review_decision['conflicting_source_ids'])}`",
                "- unreviewed_candidate_source_ids: "
                f"`{json.dumps(wing_readonly_preflight_review_decision['unreviewed_candidate_source_ids'])}`",
                "- candidate_reviews: "
                f"`{json.dumps(wing_readonly_preflight_review_decision['candidate_reviews'], sort_keys=True)}`",
                f"- blocked_by: `{json.dumps(wing_readonly_preflight_review_decision['blocked_by'])}`",
                f"- violations: `{json.dumps(wing_readonly_preflight_review_decision['violations'])}`",
            ]
        )
    if raw_mixer_contract is not None:
        lines.extend(
            [
                "",
                "## Raw Mixer State Physical Readback Contract",
                "",
                f"- schema_check: `{raw_mixer_contract['status']}`",
                f"- read_only: `{str(raw_mixer_contract['read_only']).lower()}`",
                f"- live_write: `{str(raw_mixer_contract['live_write']).lower()}`",
                f"- mutates_console_state: `{str(raw_mixer_contract['mutates_console_state']).lower()}`",
                f"- queries_console: `{str(raw_mixer_contract['queries_console']).lower()}`",
                f"- executes_tests: `{str(raw_mixer_contract['executes_tests']).lower()}`",
                f"- live_widening_allowed: `{str(raw_mixer_contract['live_widening_allowed']).lower()}`",
                f"- full_autonomous_ready: `{str(raw_mixer_contract['full_autonomous_ready']).lower()}`",
                f"- schema_id: `{raw_mixer_contract['schema_id']}`",
                f"- contract_status: `{raw_mixer_contract['contract_status']}`",
                f"- surface_id: `{raw_mixer_contract['surface_id']}`",
                f"- remediation_action_id: `{raw_mixer_contract['remediation_action_id']}`",
                "- next_physical_readback_surface_id: "
                f"`{raw_mixer_contract['next_physical_readback_surface_id']}`",
                "- next_physical_readback_action_id: "
                f"`{raw_mixer_contract['next_physical_readback_action_id']}`",
                f"- artifact_required: `{str(raw_mixer_contract['artifact_required']).lower()}`",
                "- physical_fixture_gap_remains: "
                f"`{str(raw_mixer_contract['physical_fixture_gap_remains']).lower()}`",
                "- accepted_as_physical_fixture_candidate: "
                f"`{str(raw_mixer_contract['accepted_as_physical_fixture_candidate']).lower()}`",
                "- live_authority_granted: "
                f"`{str(raw_mixer_contract['live_authority_granted']).lower()}`",
                "- required_top_level_field_count: "
                f"`{raw_mixer_contract['required_top_level_field_count']}`",
                f"- required_row_field_count: `{raw_mixer_contract['required_row_field_count']}`",
                "- allowed_sources: "
                f"`{json.dumps(raw_mixer_contract['allowed_sources'])}`",
                "- accepted_row_statuses: "
                f"`{json.dumps(raw_mixer_contract['accepted_row_statuses'])}`",
                "- forbidden_mutation_evidence_fields: "
                f"`{json.dumps(raw_mixer_contract['forbidden_mutation_evidence_fields'])}`",
                f"- target_consistency: `{str(raw_mixer_contract['target_consistency']).lower()}`",
                f"- violations: `{json.dumps(raw_mixer_contract['violations'])}`",
            ]
        )
    if raw_mixer_intake is not None:
        lines.extend(
            [
                "",
                "## Raw Mixer State Physical Readback Artifact Intake",
                "",
                f"- schema_check: `{raw_mixer_intake['status']}`",
                f"- read_only: `{str(raw_mixer_intake['read_only']).lower()}`",
                f"- live_write: `{str(raw_mixer_intake['live_write']).lower()}`",
                f"- mutates_console_state: `{str(raw_mixer_intake['mutates_console_state']).lower()}`",
                f"- queries_console: `{str(raw_mixer_intake['queries_console']).lower()}`",
                f"- executes_tests: `{str(raw_mixer_intake['executes_tests']).lower()}`",
                f"- operator_required: `{str(raw_mixer_intake['operator_required']).lower()}`",
                "- execution_allowed_by_intake: "
                f"`{str(raw_mixer_intake['execution_allowed_by_intake']).lower()}`",
                "- fixture_closure_allowed_by_intake: "
                f"`{str(raw_mixer_intake['fixture_closure_allowed_by_intake']).lower()}`",
                "- live_widening_allowed: "
                f"`{str(raw_mixer_intake['live_widening_allowed']).lower()}`",
                f"- full_autonomous_ready: `{str(raw_mixer_intake['full_autonomous_ready']).lower()}`",
                f"- intake_status: `{raw_mixer_intake['intake_status']}`",
                f"- schema_id: `{raw_mixer_intake['schema_id']}`",
                f"- surface_id: `{raw_mixer_intake['surface_id']}`",
                f"- remediation_action_id: `{raw_mixer_intake['remediation_action_id']}`",
                "- required_contract_method_id: "
                f"`{raw_mixer_intake['required_contract_method_id']}`",
                f"- artifact_count: `{raw_mixer_intake['artifact_count']}`",
                f"- valid_candidate_count: `{raw_mixer_intake['valid_candidate_count']}`",
                f"- invalid_candidate_count: `{raw_mixer_intake['invalid_candidate_count']}`",
                "- accepted_as_physical_fixture_candidate: "
                f"`{str(raw_mixer_intake['accepted_as_physical_fixture_candidate']).lower()}`",
                "- operator_review_required: "
                f"`{str(raw_mixer_intake['operator_review_required']).lower()}`",
                "- physical_fixture_gap_remains: "
                f"`{str(raw_mixer_intake['physical_fixture_gap_remains']).lower()}`",
                "- live_authority_granted: "
                f"`{str(raw_mixer_intake['live_authority_granted']).lower()}`",
                "- artifact_search_roots: "
                f"`{json.dumps(raw_mixer_intake['artifact_search_roots'])}`",
                "- artifact_source_ids: "
                f"`{json.dumps(raw_mixer_intake['artifact_source_ids'])}`",
                f"- blocked_by: `{json.dumps(raw_mixer_intake['blocked_by'])}`",
                "- intake_violations: "
                f"`{json.dumps(raw_mixer_intake['intake_violations'])}`",
                f"- violations: `{json.dumps(raw_mixer_intake['violations'])}`",
            ]
        )
    if channel_basic_contract is not None:
        lines.extend(
            [
                "",
                "## Channel Basic Controls Physical Readback Contract",
                "",
                f"- schema_check: `{channel_basic_contract['status']}`",
                f"- read_only: `{str(channel_basic_contract['read_only']).lower()}`",
                f"- live_write: `{str(channel_basic_contract['live_write']).lower()}`",
                f"- mutates_console_state: `{str(channel_basic_contract['mutates_console_state']).lower()}`",
                f"- queries_console: `{str(channel_basic_contract['queries_console']).lower()}`",
                f"- executes_tests: `{str(channel_basic_contract['executes_tests']).lower()}`",
                f"- live_widening_allowed: `{str(channel_basic_contract['live_widening_allowed']).lower()}`",
                f"- full_autonomous_ready: `{str(channel_basic_contract['full_autonomous_ready']).lower()}`",
                f"- schema_id: `{channel_basic_contract['schema_id']}`",
                f"- contract_status: `{channel_basic_contract['contract_status']}`",
                f"- surface_id: `{channel_basic_contract['surface_id']}`",
                f"- remediation_action_id: `{channel_basic_contract['remediation_action_id']}`",
                "- physical_readback_surface_id: "
                f"`{channel_basic_contract['physical_readback_surface_id']}`",
                "- physical_readback_action_id: "
                f"`{channel_basic_contract['physical_readback_action_id']}`",
                "- fixture_physical_gap_declared: "
                f"`{str(channel_basic_contract['fixture_physical_gap_declared']).lower()}`",
                "- remediation_physical_gap_declared: "
                f"`{str(channel_basic_contract['remediation_physical_gap_declared']).lower()}`",
                f"- artifact_required: `{str(channel_basic_contract['artifact_required']).lower()}`",
                "- physical_fixture_gap_remains: "
                f"`{str(channel_basic_contract['physical_fixture_gap_remains']).lower()}`",
                "- accepted_as_physical_fixture_candidate: "
                f"`{str(channel_basic_contract['accepted_as_physical_fixture_candidate']).lower()}`",
                "- live_authority_granted: "
                f"`{str(channel_basic_contract['live_authority_granted']).lower()}`",
                "- promotion_families_requiring_surface: "
                f"`{json.dumps(channel_basic_contract['promotion_families_requiring_surface'])}`",
                "- required_top_level_field_count: "
                f"`{channel_basic_contract['required_top_level_field_count']}`",
                "- required_channel_field_count: "
                f"`{channel_basic_contract['required_channel_field_count']}`",
                "- required_identity_fields: "
                f"`{json.dumps(channel_basic_contract['required_identity_fields'])}`",
                "- required_control_fields: "
                f"`{json.dumps(channel_basic_contract['required_control_fields'])}`",
                "- required_coverage_flags: "
                f"`{json.dumps(channel_basic_contract['required_coverage_flags'])}`",
                "- allowed_sources: "
                f"`{json.dumps(channel_basic_contract['allowed_sources'])}`",
                "- accepted_channel_statuses: "
                f"`{json.dumps(channel_basic_contract['accepted_channel_statuses'])}`",
                "- forbidden_mutation_evidence_fields: "
                f"`{json.dumps(channel_basic_contract['forbidden_mutation_evidence_fields'])}`",
                f"- target_consistency: `{str(channel_basic_contract['target_consistency']).lower()}`",
                f"- violations: `{json.dumps(channel_basic_contract['violations'])}`",
            ]
        )
    if channel_basic_intake is not None:
        lines.extend(
            [
                "",
                "## Channel Basic Controls Physical Readback Artifact Intake",
                "",
                f"- schema_check: `{channel_basic_intake['status']}`",
                f"- read_only: `{str(channel_basic_intake['read_only']).lower()}`",
                f"- live_write: `{str(channel_basic_intake['live_write']).lower()}`",
                f"- mutates_console_state: `{str(channel_basic_intake['mutates_console_state']).lower()}`",
                f"- queries_console: `{str(channel_basic_intake['queries_console']).lower()}`",
                f"- executes_tests: `{str(channel_basic_intake['executes_tests']).lower()}`",
                f"- operator_required: `{str(channel_basic_intake['operator_required']).lower()}`",
                "- execution_allowed_by_intake: "
                f"`{str(channel_basic_intake['execution_allowed_by_intake']).lower()}`",
                "- fixture_closure_allowed_by_intake: "
                f"`{str(channel_basic_intake['fixture_closure_allowed_by_intake']).lower()}`",
                "- live_widening_allowed: "
                f"`{str(channel_basic_intake['live_widening_allowed']).lower()}`",
                f"- full_autonomous_ready: `{str(channel_basic_intake['full_autonomous_ready']).lower()}`",
                f"- intake_status: `{channel_basic_intake['intake_status']}`",
                f"- schema_id: `{channel_basic_intake['schema_id']}`",
                f"- surface_id: `{channel_basic_intake['surface_id']}`",
                f"- remediation_action_id: `{channel_basic_intake['remediation_action_id']}`",
                "- required_contract_method_id: "
                f"`{channel_basic_intake['required_contract_method_id']}`",
                f"- artifact_count: `{channel_basic_intake['artifact_count']}`",
                f"- valid_candidate_count: `{channel_basic_intake['valid_candidate_count']}`",
                f"- invalid_candidate_count: `{channel_basic_intake['invalid_candidate_count']}`",
                f"- valid_channel_count: `{channel_basic_intake['valid_channel_count']}`",
                "- accepted_as_physical_fixture_candidate: "
                f"`{str(channel_basic_intake['accepted_as_physical_fixture_candidate']).lower()}`",
                "- operator_review_required: "
                f"`{str(channel_basic_intake['operator_review_required']).lower()}`",
                "- physical_fixture_gap_remains: "
                f"`{str(channel_basic_intake['physical_fixture_gap_remains']).lower()}`",
                "- live_authority_granted: "
                f"`{str(channel_basic_intake['live_authority_granted']).lower()}`",
                "- promotion_families_requiring_surface: "
                f"`{json.dumps(channel_basic_intake['promotion_families_requiring_surface'])}`",
                "- artifact_search_roots: "
                f"`{json.dumps(channel_basic_intake['artifact_search_roots'])}`",
                "- artifact_source_ids: "
                f"`{json.dumps(channel_basic_intake['artifact_source_ids'])}`",
                f"- blocked_by: `{json.dumps(channel_basic_intake['blocked_by'])}`",
                "- intake_violations: "
                f"`{json.dumps(channel_basic_intake['intake_violations'])}`",
                f"- violations: `{json.dumps(channel_basic_intake['violations'])}`",
            ]
        )
    if channel_processing_contract is not None:
        lines.extend(
            [
                "",
                "## Channel Processing Physical Readback Contract",
                "",
                f"- schema_check: `{channel_processing_contract['status']}`",
                f"- read_only: `{str(channel_processing_contract['read_only']).lower()}`",
                f"- live_write: `{str(channel_processing_contract['live_write']).lower()}`",
                f"- mutates_console_state: `{str(channel_processing_contract['mutates_console_state']).lower()}`",
                f"- queries_console: `{str(channel_processing_contract['queries_console']).lower()}`",
                f"- executes_tests: `{str(channel_processing_contract['executes_tests']).lower()}`",
                f"- live_widening_allowed: `{str(channel_processing_contract['live_widening_allowed']).lower()}`",
                f"- full_autonomous_ready: `{str(channel_processing_contract['full_autonomous_ready']).lower()}`",
                f"- schema_id: `{channel_processing_contract['schema_id']}`",
                f"- contract_status: `{channel_processing_contract['contract_status']}`",
                f"- surface_id: `{channel_processing_contract['surface_id']}`",
                f"- remediation_action_id: `{channel_processing_contract['remediation_action_id']}`",
                "- physical_readback_surface_id: "
                f"`{channel_processing_contract['physical_readback_surface_id']}`",
                "- physical_readback_action_id: "
                f"`{channel_processing_contract['physical_readback_action_id']}`",
                "- fixture_physical_gap_declared: "
                f"`{str(channel_processing_contract['fixture_physical_gap_declared']).lower()}`",
                "- remediation_physical_gap_declared: "
                f"`{str(channel_processing_contract['remediation_physical_gap_declared']).lower()}`",
                f"- artifact_required: `{str(channel_processing_contract['artifact_required']).lower()}`",
                "- physical_fixture_gap_remains: "
                f"`{str(channel_processing_contract['physical_fixture_gap_remains']).lower()}`",
                "- accepted_as_physical_fixture_candidate: "
                f"`{str(channel_processing_contract['accepted_as_physical_fixture_candidate']).lower()}`",
                "- live_authority_granted: "
                f"`{str(channel_processing_contract['live_authority_granted']).lower()}`",
                "- promotion_families_requiring_surface: "
                f"`{json.dumps(channel_processing_contract['promotion_families_requiring_surface'])}`",
                "- required_top_level_field_count: "
                f"`{channel_processing_contract['required_top_level_field_count']}`",
                "- required_channel_field_count: "
                f"`{channel_processing_contract['required_channel_field_count']}`",
                "- required_hpf_fields: "
                f"`{json.dumps(channel_processing_contract['required_hpf_fields'])}`",
                "- required_eq_fields: "
                f"`{json.dumps(channel_processing_contract['required_eq_fields'])}`",
                "- required_eq_band_fields: "
                f"`{json.dumps(channel_processing_contract['required_eq_band_fields'])}`",
                "- required_compressor_fields: "
                f"`{json.dumps(channel_processing_contract['required_compressor_fields'])}`",
                "- required_gate_fields: "
                f"`{json.dumps(channel_processing_contract['required_gate_fields'])}`",
                "- required_coverage_flags: "
                f"`{json.dumps(channel_processing_contract['required_coverage_flags'])}`",
                f"- minimum_eq_band_count: `{channel_processing_contract['minimum_eq_band_count']}`",
                "- allowed_sources: "
                f"`{json.dumps(channel_processing_contract['allowed_sources'])}`",
                "- accepted_channel_statuses: "
                f"`{json.dumps(channel_processing_contract['accepted_channel_statuses'])}`",
                "- forbidden_mutation_evidence_fields: "
                f"`{json.dumps(channel_processing_contract['forbidden_mutation_evidence_fields'])}`",
                f"- target_consistency: `{str(channel_processing_contract['target_consistency']).lower()}`",
                f"- violations: `{json.dumps(channel_processing_contract['violations'])}`",
            ]
        )
    if channel_processing_intake is not None:
        lines.extend(
            [
                "",
                "## Channel Processing Physical Readback Artifact Intake",
                "",
                f"- schema_check: `{channel_processing_intake['status']}`",
                f"- read_only: `{str(channel_processing_intake['read_only']).lower()}`",
                f"- live_write: `{str(channel_processing_intake['live_write']).lower()}`",
                f"- mutates_console_state: `{str(channel_processing_intake['mutates_console_state']).lower()}`",
                f"- queries_console: `{str(channel_processing_intake['queries_console']).lower()}`",
                f"- executes_tests: `{str(channel_processing_intake['executes_tests']).lower()}`",
                f"- operator_required: `{str(channel_processing_intake['operator_required']).lower()}`",
                "- execution_allowed_by_intake: "
                f"`{str(channel_processing_intake['execution_allowed_by_intake']).lower()}`",
                "- fixture_closure_allowed_by_intake: "
                f"`{str(channel_processing_intake['fixture_closure_allowed_by_intake']).lower()}`",
                "- live_widening_allowed: "
                f"`{str(channel_processing_intake['live_widening_allowed']).lower()}`",
                f"- full_autonomous_ready: `{str(channel_processing_intake['full_autonomous_ready']).lower()}`",
                f"- intake_status: `{channel_processing_intake['intake_status']}`",
                f"- schema_id: `{channel_processing_intake['schema_id']}`",
                f"- surface_id: `{channel_processing_intake['surface_id']}`",
                f"- remediation_action_id: `{channel_processing_intake['remediation_action_id']}`",
                "- required_contract_method_id: "
                f"`{channel_processing_intake['required_contract_method_id']}`",
                f"- artifact_count: `{channel_processing_intake['artifact_count']}`",
                f"- valid_candidate_count: `{channel_processing_intake['valid_candidate_count']}`",
                f"- invalid_candidate_count: `{channel_processing_intake['invalid_candidate_count']}`",
                f"- valid_channel_count: `{channel_processing_intake['valid_channel_count']}`",
                "- accepted_as_physical_fixture_candidate: "
                f"`{str(channel_processing_intake['accepted_as_physical_fixture_candidate']).lower()}`",
                "- operator_review_required: "
                f"`{str(channel_processing_intake['operator_review_required']).lower()}`",
                "- physical_fixture_gap_remains: "
                f"`{str(channel_processing_intake['physical_fixture_gap_remains']).lower()}`",
                "- live_authority_granted: "
                f"`{str(channel_processing_intake['live_authority_granted']).lower()}`",
                "- promotion_families_requiring_surface: "
                f"`{json.dumps(channel_processing_intake['promotion_families_requiring_surface'])}`",
                "- artifact_search_roots: "
                f"`{json.dumps(channel_processing_intake['artifact_search_roots'])}`",
                "- artifact_source_ids: "
                f"`{json.dumps(channel_processing_intake['artifact_source_ids'])}`",
                f"- blocked_by: `{json.dumps(channel_processing_intake['blocked_by'])}`",
                "- intake_violations: "
                f"`{json.dumps(channel_processing_intake['intake_violations'])}`",
                f"- violations: `{json.dumps(channel_processing_intake['violations'])}`",
            ]
        )
    if bus_group_main_dca_contract is not None:
        lines.extend(
            [
                "",
                "## Bus Group Main DCA Physical Readback Contract",
                "",
                f"- schema_check: `{bus_group_main_dca_contract['status']}`",
                f"- read_only: `{str(bus_group_main_dca_contract['read_only']).lower()}`",
                f"- live_write: `{str(bus_group_main_dca_contract['live_write']).lower()}`",
                f"- mutates_console_state: `{str(bus_group_main_dca_contract['mutates_console_state']).lower()}`",
                f"- queries_console: `{str(bus_group_main_dca_contract['queries_console']).lower()}`",
                f"- executes_tests: `{str(bus_group_main_dca_contract['executes_tests']).lower()}`",
                f"- live_widening_allowed: `{str(bus_group_main_dca_contract['live_widening_allowed']).lower()}`",
                f"- full_autonomous_ready: `{str(bus_group_main_dca_contract['full_autonomous_ready']).lower()}`",
                f"- schema_id: `{bus_group_main_dca_contract['schema_id']}`",
                f"- contract_status: `{bus_group_main_dca_contract['contract_status']}`",
                f"- surface_id: `{bus_group_main_dca_contract['surface_id']}`",
                f"- remediation_action_id: `{bus_group_main_dca_contract['remediation_action_id']}`",
                "- physical_readback_surface_id: "
                f"`{bus_group_main_dca_contract['physical_readback_surface_id']}`",
                "- physical_readback_action_id: "
                f"`{bus_group_main_dca_contract['physical_readback_action_id']}`",
                "- fixture_physical_gap_declared: "
                f"`{str(bus_group_main_dca_contract['fixture_physical_gap_declared']).lower()}`",
                "- remediation_physical_gap_declared: "
                f"`{str(bus_group_main_dca_contract['remediation_physical_gap_declared']).lower()}`",
                f"- artifact_required: `{str(bus_group_main_dca_contract['artifact_required']).lower()}`",
                "- physical_fixture_gap_remains: "
                f"`{str(bus_group_main_dca_contract['physical_fixture_gap_remains']).lower()}`",
                "- accepted_as_physical_fixture_candidate: "
                f"`{str(bus_group_main_dca_contract['accepted_as_physical_fixture_candidate']).lower()}`",
                "- live_authority_granted: "
                f"`{str(bus_group_main_dca_contract['live_authority_granted']).lower()}`",
                "- promotion_families_requiring_surface: "
                f"`{json.dumps(bus_group_main_dca_contract['promotion_families_requiring_surface'])}`",
                "- required_top_level_field_count: "
                f"`{bus_group_main_dca_contract['required_top_level_field_count']}`",
                "- required_bus_field_count: "
                f"`{bus_group_main_dca_contract['required_bus_field_count']}`",
                "- required_main_field_count: "
                f"`{bus_group_main_dca_contract['required_main_field_count']}`",
                "- required_dca_field_count: "
                f"`{bus_group_main_dca_contract['required_dca_field_count']}`",
                "- required_bus_control_fields: "
                f"`{json.dumps(bus_group_main_dca_contract['required_bus_control_fields'])}`",
                "- required_main_control_fields: "
                f"`{json.dumps(bus_group_main_dca_contract['required_main_control_fields'])}`",
                "- required_dca_control_fields: "
                f"`{json.dumps(bus_group_main_dca_contract['required_dca_control_fields'])}`",
                "- required_bus_assignment_fields: "
                f"`{json.dumps(bus_group_main_dca_contract['required_bus_assignment_fields'])}`",
                "- required_dca_assignment_fields: "
                f"`{json.dumps(bus_group_main_dca_contract['required_dca_assignment_fields'])}`",
                "- required_metering_fields: "
                f"`{json.dumps(bus_group_main_dca_contract['required_metering_fields'])}`",
                "- required_bus_coverage_flags: "
                f"`{json.dumps(bus_group_main_dca_contract['required_bus_coverage_flags'])}`",
                "- required_main_coverage_flags: "
                f"`{json.dumps(bus_group_main_dca_contract['required_main_coverage_flags'])}`",
                "- required_dca_coverage_flags: "
                f"`{json.dumps(bus_group_main_dca_contract['required_dca_coverage_flags'])}`",
                f"- minimum_bus_count: `{bus_group_main_dca_contract['minimum_bus_count']}`",
                f"- minimum_main_count: `{bus_group_main_dca_contract['minimum_main_count']}`",
                f"- minimum_dca_count: `{bus_group_main_dca_contract['minimum_dca_count']}`",
                "- allowed_sources: "
                f"`{json.dumps(bus_group_main_dca_contract['allowed_sources'])}`",
                "- accepted_surface_statuses: "
                f"`{json.dumps(bus_group_main_dca_contract['accepted_surface_statuses'])}`",
                "- forbidden_mutation_evidence_fields: "
                f"`{json.dumps(bus_group_main_dca_contract['forbidden_mutation_evidence_fields'])}`",
                f"- target_consistency: `{str(bus_group_main_dca_contract['target_consistency']).lower()}`",
                f"- violations: `{json.dumps(bus_group_main_dca_contract['violations'])}`",
            ]
        )
    if bus_group_main_dca_intake is not None:
        lines.extend(
            [
                "",
                "## Bus Group Main DCA Physical Readback Artifact Intake",
                "",
                f"- schema_check: `{bus_group_main_dca_intake['status']}`",
                f"- read_only: `{str(bus_group_main_dca_intake['read_only']).lower()}`",
                f"- live_write: `{str(bus_group_main_dca_intake['live_write']).lower()}`",
                f"- mutates_console_state: `{str(bus_group_main_dca_intake['mutates_console_state']).lower()}`",
                f"- queries_console: `{str(bus_group_main_dca_intake['queries_console']).lower()}`",
                f"- executes_tests: `{str(bus_group_main_dca_intake['executes_tests']).lower()}`",
                f"- operator_required: `{str(bus_group_main_dca_intake['operator_required']).lower()}`",
                "- execution_allowed_by_intake: "
                f"`{str(bus_group_main_dca_intake['execution_allowed_by_intake']).lower()}`",
                "- fixture_closure_allowed_by_intake: "
                f"`{str(bus_group_main_dca_intake['fixture_closure_allowed_by_intake']).lower()}`",
                "- live_widening_allowed: "
                f"`{str(bus_group_main_dca_intake['live_widening_allowed']).lower()}`",
                f"- full_autonomous_ready: `{str(bus_group_main_dca_intake['full_autonomous_ready']).lower()}`",
                f"- intake_status: `{bus_group_main_dca_intake['intake_status']}`",
                f"- schema_id: `{bus_group_main_dca_intake['schema_id']}`",
                f"- surface_id: `{bus_group_main_dca_intake['surface_id']}`",
                f"- remediation_action_id: `{bus_group_main_dca_intake['remediation_action_id']}`",
                "- required_contract_method_id: "
                f"`{bus_group_main_dca_intake['required_contract_method_id']}`",
                f"- artifact_count: `{bus_group_main_dca_intake['artifact_count']}`",
                f"- valid_candidate_count: `{bus_group_main_dca_intake['valid_candidate_count']}`",
                f"- invalid_candidate_count: `{bus_group_main_dca_intake['invalid_candidate_count']}`",
                f"- valid_bus_count: `{bus_group_main_dca_intake['valid_bus_count']}`",
                f"- valid_main_count: `{bus_group_main_dca_intake['valid_main_count']}`",
                f"- valid_dca_count: `{bus_group_main_dca_intake['valid_dca_count']}`",
                "- accepted_as_physical_fixture_candidate: "
                f"`{str(bus_group_main_dca_intake['accepted_as_physical_fixture_candidate']).lower()}`",
                "- operator_review_required: "
                f"`{str(bus_group_main_dca_intake['operator_review_required']).lower()}`",
                "- physical_fixture_gap_remains: "
                f"`{str(bus_group_main_dca_intake['physical_fixture_gap_remains']).lower()}`",
                "- live_authority_granted: "
                f"`{str(bus_group_main_dca_intake['live_authority_granted']).lower()}`",
                "- promotion_families_requiring_surface: "
                f"`{json.dumps(bus_group_main_dca_intake['promotion_families_requiring_surface'])}`",
                "- artifact_search_roots: "
                f"`{json.dumps(bus_group_main_dca_intake['artifact_search_roots'])}`",
                "- artifact_source_ids: "
                f"`{json.dumps(bus_group_main_dca_intake['artifact_source_ids'])}`",
                f"- blocked_by: `{json.dumps(bus_group_main_dca_intake['blocked_by'])}`",
                "- intake_violations: "
                f"`{json.dumps(bus_group_main_dca_intake['intake_violations'])}`",
                f"- violations: `{json.dumps(bus_group_main_dca_intake['violations'])}`",
            ]
        )
    if matrix_state_contract is not None:
        lines.extend(
            [
                "",
                "## Matrix State Physical Readback Contract",
                "",
                f"- schema_check: `{matrix_state_contract['status']}`",
                f"- read_only: `{str(matrix_state_contract['read_only']).lower()}`",
                f"- live_write: `{str(matrix_state_contract['live_write']).lower()}`",
                f"- mutates_console_state: `{str(matrix_state_contract['mutates_console_state']).lower()}`",
                f"- queries_console: `{str(matrix_state_contract['queries_console']).lower()}`",
                f"- executes_tests: `{str(matrix_state_contract['executes_tests']).lower()}`",
                f"- live_widening_allowed: `{str(matrix_state_contract['live_widening_allowed']).lower()}`",
                f"- full_autonomous_ready: `{str(matrix_state_contract['full_autonomous_ready']).lower()}`",
                f"- schema_id: `{matrix_state_contract['schema_id']}`",
                f"- contract_status: `{matrix_state_contract['contract_status']}`",
                f"- surface_id: `{matrix_state_contract['surface_id']}`",
                f"- remediation_action_id: `{matrix_state_contract['remediation_action_id']}`",
                "- physical_readback_surface_id: "
                f"`{matrix_state_contract['physical_readback_surface_id']}`",
                "- physical_readback_action_id: "
                f"`{matrix_state_contract['physical_readback_action_id']}`",
                "- fixture_physical_gap_declared: "
                f"`{str(matrix_state_contract['fixture_physical_gap_declared']).lower()}`",
                "- remediation_physical_gap_declared: "
                f"`{str(matrix_state_contract['remediation_physical_gap_declared']).lower()}`",
                f"- artifact_required: `{str(matrix_state_contract['artifact_required']).lower()}`",
                "- physical_fixture_gap_remains: "
                f"`{str(matrix_state_contract['physical_fixture_gap_remains']).lower()}`",
                "- accepted_as_physical_fixture_candidate: "
                f"`{str(matrix_state_contract['accepted_as_physical_fixture_candidate']).lower()}`",
                "- live_authority_granted: "
                f"`{str(matrix_state_contract['live_authority_granted']).lower()}`",
                "- promotion_families_requiring_surface: "
                f"`{json.dumps(matrix_state_contract['promotion_families_requiring_surface'])}`",
                "- required_top_level_field_count: "
                f"`{matrix_state_contract['required_top_level_field_count']}`",
                "- required_matrix_field_count: "
                f"`{matrix_state_contract['required_matrix_field_count']}`",
                "- required_control_fields: "
                f"`{json.dumps(matrix_state_contract['required_control_fields'])}`",
                "- required_eq_fields: "
                f"`{json.dumps(matrix_state_contract['required_eq_fields'])}`",
                "- required_eq_band_fields: "
                f"`{json.dumps(matrix_state_contract['required_eq_band_fields'])}`",
                "- required_dynamics_fields: "
                f"`{json.dumps(matrix_state_contract['required_dynamics_fields'])}`",
                "- required_insert_fields: "
                f"`{json.dumps(matrix_state_contract['required_insert_fields'])}`",
                "- required_metering_fields: "
                f"`{json.dumps(matrix_state_contract['required_metering_fields'])}`",
                "- required_coverage_flags: "
                f"`{json.dumps(matrix_state_contract['required_coverage_flags'])}`",
                f"- minimum_matrix_count: `{matrix_state_contract['minimum_matrix_count']}`",
                f"- minimum_eq_band_count: `{matrix_state_contract['minimum_eq_band_count']}`",
                "- allowed_sources: "
                f"`{json.dumps(matrix_state_contract['allowed_sources'])}`",
                "- accepted_matrix_statuses: "
                f"`{json.dumps(matrix_state_contract['accepted_matrix_statuses'])}`",
                "- forbidden_mutation_evidence_fields: "
                f"`{json.dumps(matrix_state_contract['forbidden_mutation_evidence_fields'])}`",
                f"- target_consistency: `{str(matrix_state_contract['target_consistency']).lower()}`",
                f"- violations: `{json.dumps(matrix_state_contract['violations'])}`",
            ]
        )
    if matrix_state_intake is not None:
        lines.extend(
            [
                "",
                "## Matrix State Physical Readback Artifact Intake",
                "",
                f"- schema_check: `{matrix_state_intake['status']}`",
                f"- read_only: `{str(matrix_state_intake['read_only']).lower()}`",
                f"- live_write: `{str(matrix_state_intake['live_write']).lower()}`",
                f"- mutates_console_state: `{str(matrix_state_intake['mutates_console_state']).lower()}`",
                f"- queries_console: `{str(matrix_state_intake['queries_console']).lower()}`",
                f"- executes_tests: `{str(matrix_state_intake['executes_tests']).lower()}`",
                f"- operator_required: `{str(matrix_state_intake['operator_required']).lower()}`",
                "- execution_allowed_by_intake: "
                f"`{str(matrix_state_intake['execution_allowed_by_intake']).lower()}`",
                "- fixture_closure_allowed_by_intake: "
                f"`{str(matrix_state_intake['fixture_closure_allowed_by_intake']).lower()}`",
                "- live_widening_allowed: "
                f"`{str(matrix_state_intake['live_widening_allowed']).lower()}`",
                f"- full_autonomous_ready: `{str(matrix_state_intake['full_autonomous_ready']).lower()}`",
                f"- intake_status: `{matrix_state_intake['intake_status']}`",
                f"- schema_id: `{matrix_state_intake['schema_id']}`",
                f"- surface_id: `{matrix_state_intake['surface_id']}`",
                f"- remediation_action_id: `{matrix_state_intake['remediation_action_id']}`",
                "- required_contract_method_id: "
                f"`{matrix_state_intake['required_contract_method_id']}`",
                f"- artifact_count: `{matrix_state_intake['artifact_count']}`",
                f"- valid_candidate_count: `{matrix_state_intake['valid_candidate_count']}`",
                f"- invalid_candidate_count: `{matrix_state_intake['invalid_candidate_count']}`",
                f"- valid_matrix_count: `{matrix_state_intake['valid_matrix_count']}`",
                "- accepted_as_physical_fixture_candidate: "
                f"`{str(matrix_state_intake['accepted_as_physical_fixture_candidate']).lower()}`",
                "- operator_review_required: "
                f"`{str(matrix_state_intake['operator_review_required']).lower()}`",
                "- physical_fixture_gap_remains: "
                f"`{str(matrix_state_intake['physical_fixture_gap_remains']).lower()}`",
                "- live_authority_granted: "
                f"`{str(matrix_state_intake['live_authority_granted']).lower()}`",
                "- promotion_families_requiring_surface: "
                f"`{json.dumps(matrix_state_intake['promotion_families_requiring_surface'])}`",
                "- artifact_search_roots: "
                f"`{json.dumps(matrix_state_intake['artifact_search_roots'])}`",
                "- artifact_source_ids: "
                f"`{json.dumps(matrix_state_intake['artifact_source_ids'])}`",
                f"- blocked_by: `{json.dumps(matrix_state_intake['blocked_by'])}`",
                "- intake_violations: "
                f"`{json.dumps(matrix_state_intake['intake_violations'])}`",
                f"- violations: `{json.dumps(matrix_state_intake['violations'])}`",
            ]
        )
    if fx_slots_inserts_contract is not None:
        lines.extend(
            [
                "",
                "## FX Slots Inserts Physical Readback Contract",
                "",
                f"- schema_check: `{fx_slots_inserts_contract['status']}`",
                f"- read_only: `{str(fx_slots_inserts_contract['read_only']).lower()}`",
                f"- live_write: `{str(fx_slots_inserts_contract['live_write']).lower()}`",
                "- mutates_console_state: "
                f"`{str(fx_slots_inserts_contract['mutates_console_state']).lower()}`",
                f"- queries_console: `{str(fx_slots_inserts_contract['queries_console']).lower()}`",
                f"- executes_tests: `{str(fx_slots_inserts_contract['executes_tests']).lower()}`",
                "- live_widening_allowed: "
                f"`{str(fx_slots_inserts_contract['live_widening_allowed']).lower()}`",
                "- full_autonomous_ready: "
                f"`{str(fx_slots_inserts_contract['full_autonomous_ready']).lower()}`",
                f"- schema_id: `{fx_slots_inserts_contract['schema_id']}`",
                f"- contract_status: `{fx_slots_inserts_contract['contract_status']}`",
                f"- surface_id: `{fx_slots_inserts_contract['surface_id']}`",
                "- remediation_action_id: "
                f"`{fx_slots_inserts_contract['remediation_action_id']}`",
                "- physical_readback_surface_id: "
                f"`{fx_slots_inserts_contract['physical_readback_surface_id']}`",
                "- physical_readback_action_id: "
                f"`{fx_slots_inserts_contract['physical_readback_action_id']}`",
                "- fixture_physical_gap_declared: "
                f"`{str(fx_slots_inserts_contract['fixture_physical_gap_declared']).lower()}`",
                "- remediation_physical_gap_declared: "
                f"`{str(fx_slots_inserts_contract['remediation_physical_gap_declared']).lower()}`",
                "- artifact_required: "
                f"`{str(fx_slots_inserts_contract['artifact_required']).lower()}`",
                "- physical_fixture_gap_remains: "
                f"`{str(fx_slots_inserts_contract['physical_fixture_gap_remains']).lower()}`",
                "- accepted_as_physical_fixture_candidate: "
                f"`{str(fx_slots_inserts_contract['accepted_as_physical_fixture_candidate']).lower()}`",
                "- live_authority_granted: "
                f"`{str(fx_slots_inserts_contract['live_authority_granted']).lower()}`",
                "- promotion_families_requiring_surface: "
                f"`{json.dumps(fx_slots_inserts_contract['promotion_families_requiring_surface'])}`",
                "- required_top_level_field_count: "
                f"`{fx_slots_inserts_contract['required_top_level_field_count']}`",
                "- required_fx_slot_field_count: "
                f"`{fx_slots_inserts_contract['required_fx_slot_field_count']}`",
                "- required_fx_slot_fields: "
                f"`{json.dumps(fx_slots_inserts_contract['required_fx_slot_fields'])}`",
                "- required_parameter_fields: "
                f"`{json.dumps(fx_slots_inserts_contract['required_parameter_fields'])}`",
                "- required_parameter_safe_range_fields: "
                f"`{json.dumps(fx_slots_inserts_contract['required_parameter_safe_range_fields'])}`",
                "- required_parameter_schema_fields: "
                f"`{json.dumps(fx_slots_inserts_contract['required_parameter_schema_fields'])}`",
                "- required_fx_slot_coverage_flags: "
                f"`{json.dumps(fx_slots_inserts_contract['required_fx_slot_coverage_flags'])}`",
                "- required_insert_fields: "
                f"`{json.dumps(fx_slots_inserts_contract['required_insert_fields'])}`",
                "- required_insert_coverage_flags: "
                f"`{json.dumps(fx_slots_inserts_contract['required_insert_coverage_flags'])}`",
                "- required_false_safety_flags: "
                f"`{json.dumps(fx_slots_inserts_contract['required_false_safety_flags'])}`",
                f"- minimum_fx_slot_count: `{fx_slots_inserts_contract['minimum_fx_slot_count']}`",
                "- minimum_fx_parameter_count: "
                f"`{fx_slots_inserts_contract['minimum_fx_parameter_count']}`",
                f"- minimum_insert_count: `{fx_slots_inserts_contract['minimum_insert_count']}`",
                "- allowed_sources: "
                f"`{json.dumps(fx_slots_inserts_contract['allowed_sources'])}`",
                "- accepted_row_statuses: "
                f"`{json.dumps(fx_slots_inserts_contract['accepted_row_statuses'])}`",
                "- allowed_insert_target_types: "
                f"`{json.dumps(fx_slots_inserts_contract['allowed_insert_target_types'])}`",
                "- allowed_insert_positions: "
                f"`{json.dumps(fx_slots_inserts_contract['allowed_insert_positions'])}`",
                "- forbidden_mutation_evidence_fields: "
                f"`{json.dumps(fx_slots_inserts_contract['forbidden_mutation_evidence_fields'])}`",
                "- target_consistency: "
                f"`{str(fx_slots_inserts_contract['target_consistency']).lower()}`",
                f"- violations: `{json.dumps(fx_slots_inserts_contract['violations'])}`",
            ]
        )
    if fx_slots_inserts_intake is not None:
        lines.extend(
            [
                "",
                "## FX Slots Inserts Physical Readback Artifact Intake",
                "",
                f"- schema_check: `{fx_slots_inserts_intake['status']}`",
                f"- read_only: `{str(fx_slots_inserts_intake['read_only']).lower()}`",
                f"- live_write: `{str(fx_slots_inserts_intake['live_write']).lower()}`",
                "- mutates_console_state: "
                f"`{str(fx_slots_inserts_intake['mutates_console_state']).lower()}`",
                f"- queries_console: `{str(fx_slots_inserts_intake['queries_console']).lower()}`",
                f"- executes_tests: `{str(fx_slots_inserts_intake['executes_tests']).lower()}`",
                "- operator_required: "
                f"`{str(fx_slots_inserts_intake['operator_required']).lower()}`",
                "- execution_allowed_by_intake: "
                f"`{str(fx_slots_inserts_intake['execution_allowed_by_intake']).lower()}`",
                "- fixture_closure_allowed_by_intake: "
                f"`{str(fx_slots_inserts_intake['fixture_closure_allowed_by_intake']).lower()}`",
                "- live_widening_allowed: "
                f"`{str(fx_slots_inserts_intake['live_widening_allowed']).lower()}`",
                "- full_autonomous_ready: "
                f"`{str(fx_slots_inserts_intake['full_autonomous_ready']).lower()}`",
                f"- intake_status: `{fx_slots_inserts_intake['intake_status']}`",
                f"- schema_id: `{fx_slots_inserts_intake['schema_id']}`",
                f"- surface_id: `{fx_slots_inserts_intake['surface_id']}`",
                "- remediation_action_id: "
                f"`{fx_slots_inserts_intake['remediation_action_id']}`",
                "- required_contract_method_id: "
                f"`{fx_slots_inserts_intake['required_contract_method_id']}`",
                f"- artifact_count: `{fx_slots_inserts_intake['artifact_count']}`",
                f"- valid_candidate_count: `{fx_slots_inserts_intake['valid_candidate_count']}`",
                f"- invalid_candidate_count: `{fx_slots_inserts_intake['invalid_candidate_count']}`",
                f"- valid_fx_slot_count: `{fx_slots_inserts_intake['valid_fx_slot_count']}`",
                f"- valid_insert_count: `{fx_slots_inserts_intake['valid_insert_count']}`",
                "- accepted_as_physical_fixture_candidate: "
                f"`{str(fx_slots_inserts_intake['accepted_as_physical_fixture_candidate']).lower()}`",
                "- operator_review_required: "
                f"`{str(fx_slots_inserts_intake['operator_review_required']).lower()}`",
                "- physical_fixture_gap_remains: "
                f"`{str(fx_slots_inserts_intake['physical_fixture_gap_remains']).lower()}`",
                "- live_authority_granted: "
                f"`{str(fx_slots_inserts_intake['live_authority_granted']).lower()}`",
                "- promotion_families_requiring_surface: "
                f"`{json.dumps(fx_slots_inserts_intake['promotion_families_requiring_surface'])}`",
                "- artifact_search_roots: "
                f"`{json.dumps(fx_slots_inserts_intake['artifact_search_roots'])}`",
                "- artifact_source_ids: "
                f"`{json.dumps(fx_slots_inserts_intake['artifact_source_ids'])}`",
                f"- blocked_by: `{json.dumps(fx_slots_inserts_intake['blocked_by'])}`",
                "- intake_violations: "
                f"`{json.dumps(fx_slots_inserts_intake['intake_violations'])}`",
                f"- violations: `{json.dumps(fx_slots_inserts_intake['violations'])}`",
            ]
        )
    if routing_sends_contract is not None:
        lines.extend(
            [
                "",
                "## Routing Sends Physical Readback Contract",
                "",
                f"- schema_check: `{routing_sends_contract['status']}`",
                f"- read_only: `{str(routing_sends_contract['read_only']).lower()}`",
                f"- live_write: `{str(routing_sends_contract['live_write']).lower()}`",
                "- mutates_console_state: "
                f"`{str(routing_sends_contract['mutates_console_state']).lower()}`",
                f"- queries_console: `{str(routing_sends_contract['queries_console']).lower()}`",
                f"- executes_tests: `{str(routing_sends_contract['executes_tests']).lower()}`",
                "- live_widening_allowed: "
                f"`{str(routing_sends_contract['live_widening_allowed']).lower()}`",
                "- full_autonomous_ready: "
                f"`{str(routing_sends_contract['full_autonomous_ready']).lower()}`",
                f"- schema_id: `{routing_sends_contract['schema_id']}`",
                f"- contract_status: `{routing_sends_contract['contract_status']}`",
                f"- surface_id: `{routing_sends_contract['surface_id']}`",
                f"- remediation_action_id: `{routing_sends_contract['remediation_action_id']}`",
                "- physical_readback_surface_id: "
                f"`{routing_sends_contract['physical_readback_surface_id']}`",
                "- physical_readback_action_id: "
                f"`{routing_sends_contract['physical_readback_action_id']}`",
                "- fixture_physical_gap_declared: "
                f"`{str(routing_sends_contract['fixture_physical_gap_declared']).lower()}`",
                "- remediation_physical_gap_declared: "
                f"`{str(routing_sends_contract['remediation_physical_gap_declared']).lower()}`",
                "- artifact_required: "
                f"`{str(routing_sends_contract['artifact_required']).lower()}`",
                "- physical_fixture_gap_remains: "
                f"`{str(routing_sends_contract['physical_fixture_gap_remains']).lower()}`",
                "- accepted_as_physical_fixture_candidate: "
                f"`{str(routing_sends_contract['accepted_as_physical_fixture_candidate']).lower()}`",
                "- live_authority_granted: "
                f"`{str(routing_sends_contract['live_authority_granted']).lower()}`",
                "- promotion_families_requiring_surface: "
                f"`{json.dumps(routing_sends_contract['promotion_families_requiring_surface'])}`",
                "- required_top_level_field_count: "
                f"`{routing_sends_contract['required_top_level_field_count']}`",
                "- required_static_route_field_count: "
                f"`{routing_sends_contract['required_static_route_field_count']}`",
                "- required_static_route_fields: "
                f"`{json.dumps(routing_sends_contract['required_static_route_fields'])}`",
                "- required_bus_send_fields: "
                f"`{json.dumps(routing_sends_contract['required_bus_send_fields'])}`",
                "- required_main_send_fields: "
                f"`{json.dumps(routing_sends_contract['required_main_send_fields'])}`",
                "- required_matrix_send_fields: "
                f"`{json.dumps(routing_sends_contract['required_matrix_send_fields'])}`",
                "- required_output_route_fields: "
                f"`{json.dumps(routing_sends_contract['required_output_route_fields'])}`",
                "- required_static_route_coverage_flags: "
                f"`{json.dumps(routing_sends_contract['required_static_route_coverage_flags'])}`",
                "- required_send_coverage_flags: "
                f"`{json.dumps(routing_sends_contract['required_send_coverage_flags'])}`",
                "- required_output_route_coverage_flags: "
                f"`{json.dumps(routing_sends_contract['required_output_route_coverage_flags'])}`",
                "- required_false_safety_flags: "
                f"`{json.dumps(routing_sends_contract['required_false_safety_flags'])}`",
                "- minimum_static_dante_route_count: "
                f"`{routing_sends_contract['minimum_static_dante_route_count']}`",
                "- minimum_channel_bus_send_count: "
                f"`{routing_sends_contract['minimum_channel_bus_send_count']}`",
                "- minimum_channel_main_send_count: "
                f"`{routing_sends_contract['minimum_channel_main_send_count']}`",
                "- minimum_channel_matrix_send_count: "
                f"`{routing_sends_contract['minimum_channel_matrix_send_count']}`",
                f"- minimum_output_route_count: `{routing_sends_contract['minimum_output_route_count']}`",
                "- allowed_sources: "
                f"`{json.dumps(routing_sends_contract['allowed_sources'])}`",
                "- accepted_row_statuses: "
                f"`{json.dumps(routing_sends_contract['accepted_row_statuses'])}`",
                "- forbidden_mutation_evidence_fields: "
                f"`{json.dumps(routing_sends_contract['forbidden_mutation_evidence_fields'])}`",
                "- target_consistency: "
                f"`{str(routing_sends_contract['target_consistency']).lower()}`",
                f"- violations: `{json.dumps(routing_sends_contract['violations'])}`",
            ]
        )
    if routing_sends_intake is not None:
        lines.extend(
            [
                "",
                "## Routing Sends Physical Readback Artifact Intake",
                "",
                f"- schema_check: `{routing_sends_intake['status']}`",
                f"- read_only: `{str(routing_sends_intake['read_only']).lower()}`",
                f"- live_write: `{str(routing_sends_intake['live_write']).lower()}`",
                "- mutates_console_state: "
                f"`{str(routing_sends_intake['mutates_console_state']).lower()}`",
                f"- queries_console: `{str(routing_sends_intake['queries_console']).lower()}`",
                f"- executes_tests: `{str(routing_sends_intake['executes_tests']).lower()}`",
                "- operator_required: "
                f"`{str(routing_sends_intake['operator_required']).lower()}`",
                "- execution_allowed_by_intake: "
                f"`{str(routing_sends_intake['execution_allowed_by_intake']).lower()}`",
                "- fixture_closure_allowed_by_intake: "
                f"`{str(routing_sends_intake['fixture_closure_allowed_by_intake']).lower()}`",
                "- live_widening_allowed: "
                f"`{str(routing_sends_intake['live_widening_allowed']).lower()}`",
                "- full_autonomous_ready: "
                f"`{str(routing_sends_intake['full_autonomous_ready']).lower()}`",
                f"- intake_status: `{routing_sends_intake['intake_status']}`",
                f"- schema_id: `{routing_sends_intake['schema_id']}`",
                f"- surface_id: `{routing_sends_intake['surface_id']}`",
                "- remediation_action_id: "
                f"`{routing_sends_intake['remediation_action_id']}`",
                "- required_contract_method_id: "
                f"`{routing_sends_intake['required_contract_method_id']}`",
                f"- artifact_count: `{routing_sends_intake['artifact_count']}`",
                f"- valid_candidate_count: `{routing_sends_intake['valid_candidate_count']}`",
                f"- invalid_candidate_count: `{routing_sends_intake['invalid_candidate_count']}`",
                "- valid_static_dante_route_count: "
                f"`{routing_sends_intake['valid_static_dante_route_count']}`",
                "- valid_channel_bus_send_count: "
                f"`{routing_sends_intake['valid_channel_bus_send_count']}`",
                "- valid_channel_main_send_count: "
                f"`{routing_sends_intake['valid_channel_main_send_count']}`",
                "- valid_channel_matrix_send_count: "
                f"`{routing_sends_intake['valid_channel_matrix_send_count']}`",
                f"- valid_output_route_count: `{routing_sends_intake['valid_output_route_count']}`",
                "- accepted_as_physical_fixture_candidate: "
                f"`{str(routing_sends_intake['accepted_as_physical_fixture_candidate']).lower()}`",
                "- operator_review_required: "
                f"`{str(routing_sends_intake['operator_review_required']).lower()}`",
                "- physical_fixture_gap_remains: "
                f"`{str(routing_sends_intake['physical_fixture_gap_remains']).lower()}`",
                "- live_authority_granted: "
                f"`{str(routing_sends_intake['live_authority_granted']).lower()}`",
                "- promotion_families_requiring_surface: "
                f"`{json.dumps(routing_sends_intake['promotion_families_requiring_surface'])}`",
                "- artifact_search_roots: "
                f"`{json.dumps(routing_sends_intake['artifact_search_roots'])}`",
                "- artifact_source_ids: "
                f"`{json.dumps(routing_sends_intake['artifact_source_ids'])}`",
                f"- blocked_by: `{json.dumps(routing_sends_intake['blocked_by'])}`",
                "- intake_violations: "
                f"`{json.dumps(routing_sends_intake['intake_violations'])}`",
                f"- violations: `{json.dumps(routing_sends_intake['violations'])}`",
            ]
        )
    if snapshots_scenes_contract is not None:
        lines.extend(
            [
                "",
                "## Snapshots Scenes Physical Readback Contract",
                "",
                f"- schema_check: `{snapshots_scenes_contract['status']}`",
                f"- read_only: `{str(snapshots_scenes_contract['read_only']).lower()}`",
                f"- live_write: `{str(snapshots_scenes_contract['live_write']).lower()}`",
                "- scene_load_allowed: "
                f"`{str(snapshots_scenes_contract['scene_load_allowed']).lower()}`",
                "- scene_save_allowed: "
                f"`{str(snapshots_scenes_contract['scene_save_allowed']).lower()}`",
                f"- schema_id: `{snapshots_scenes_contract['schema_id']}`",
                f"- contract_status: `{snapshots_scenes_contract['contract_status']}`",
                f"- surface_id: `{snapshots_scenes_contract['surface_id']}`",
                f"- remediation_action_id: `{snapshots_scenes_contract['remediation_action_id']}`",
                "- physical_readback_surface_id: "
                f"`{snapshots_scenes_contract['physical_readback_surface_id']}`",
                "- physical_readback_action_id: "
                f"`{snapshots_scenes_contract['physical_readback_action_id']}`",
                "- fixture_physical_gap_declared: "
                f"`{str(snapshots_scenes_contract['fixture_physical_gap_declared']).lower()}`",
                "- remediation_physical_gap_declared: "
                f"`{str(snapshots_scenes_contract['remediation_physical_gap_declared']).lower()}`",
                "- physical_fixture_gap_remains: "
                f"`{str(snapshots_scenes_contract['physical_fixture_gap_remains']).lower()}`",
                "- live_authority_granted: "
                f"`{str(snapshots_scenes_contract['live_authority_granted']).lower()}`",
                "- promotion_families_requiring_surface: "
                f"`{json.dumps(snapshots_scenes_contract['promotion_families_requiring_surface'])}`",
                "- required_snapshot_fields: "
                f"`{json.dumps(snapshots_scenes_contract['required_snapshot_fields'])}`",
                "- required_coverage_flags: "
                f"`{json.dumps(snapshots_scenes_contract['required_coverage_flags'])}`",
                "- required_false_safety_flags: "
                f"`{json.dumps(snapshots_scenes_contract['required_false_safety_flags'])}`",
                f"- minimum_snapshot_count: `{snapshots_scenes_contract['minimum_snapshot_count']}`",
                "- target_consistency: "
                f"`{str(snapshots_scenes_contract['target_consistency']).lower()}`",
                f"- violations: `{json.dumps(snapshots_scenes_contract['violations'])}`",
            ]
        )
    if snapshots_scenes_intake is not None:
        lines.extend(
            [
                "",
                "## Snapshots Scenes Physical Readback Artifact Intake",
                "",
                f"- schema_check: `{snapshots_scenes_intake['status']}`",
                f"- read_only: `{str(snapshots_scenes_intake['read_only']).lower()}`",
                f"- live_write: `{str(snapshots_scenes_intake['live_write']).lower()}`",
                "- mutates_console_state: "
                f"`{str(snapshots_scenes_intake['mutates_console_state']).lower()}`",
                f"- queries_console: `{str(snapshots_scenes_intake['queries_console']).lower()}`",
                f"- executes_tests: `{str(snapshots_scenes_intake['executes_tests']).lower()}`",
                "- operator_required: "
                f"`{str(snapshots_scenes_intake['operator_required']).lower()}`",
                "- execution_allowed_by_intake: "
                f"`{str(snapshots_scenes_intake['execution_allowed_by_intake']).lower()}`",
                "- fixture_closure_allowed_by_intake: "
                f"`{str(snapshots_scenes_intake['fixture_closure_allowed_by_intake']).lower()}`",
                "- scene_load_allowed: "
                f"`{str(snapshots_scenes_intake['scene_load_allowed']).lower()}`",
                "- scene_save_allowed: "
                f"`{str(snapshots_scenes_intake['scene_save_allowed']).lower()}`",
                "- live_widening_allowed: "
                f"`{str(snapshots_scenes_intake['live_widening_allowed']).lower()}`",
                "- full_autonomous_ready: "
                f"`{str(snapshots_scenes_intake['full_autonomous_ready']).lower()}`",
                f"- intake_status: `{snapshots_scenes_intake['intake_status']}`",
                f"- schema_id: `{snapshots_scenes_intake['schema_id']}`",
                f"- surface_id: `{snapshots_scenes_intake['surface_id']}`",
                "- remediation_action_id: "
                f"`{snapshots_scenes_intake['remediation_action_id']}`",
                "- required_contract_method_id: "
                f"`{snapshots_scenes_intake['required_contract_method_id']}`",
                f"- artifact_count: `{snapshots_scenes_intake['artifact_count']}`",
                f"- valid_candidate_count: `{snapshots_scenes_intake['valid_candidate_count']}`",
                f"- invalid_candidate_count: `{snapshots_scenes_intake['invalid_candidate_count']}`",
                f"- valid_snapshot_count: `{snapshots_scenes_intake['valid_snapshot_count']}`",
                "- accepted_as_physical_fixture_candidate: "
                f"`{str(snapshots_scenes_intake['accepted_as_physical_fixture_candidate']).lower()}`",
                "- operator_review_required: "
                f"`{str(snapshots_scenes_intake['operator_review_required']).lower()}`",
                "- physical_fixture_gap_remains: "
                f"`{str(snapshots_scenes_intake['physical_fixture_gap_remains']).lower()}`",
                "- live_authority_granted: "
                f"`{str(snapshots_scenes_intake['live_authority_granted']).lower()}`",
                "- promotion_families_requiring_surface: "
                f"`{json.dumps(snapshots_scenes_intake['promotion_families_requiring_surface'])}`",
                "- artifact_search_roots: "
                f"`{json.dumps(snapshots_scenes_intake['artifact_search_roots'])}`",
                "- artifact_source_ids: "
                f"`{json.dumps(snapshots_scenes_intake['artifact_source_ids'])}`",
                f"- blocked_by: `{json.dumps(snapshots_scenes_intake['blocked_by'])}`",
                "- intake_violations: "
                f"`{json.dumps(snapshots_scenes_intake['intake_violations'])}`",
                "- target_consistency: "
                f"`{str(snapshots_scenes_intake['target_consistency']).lower()}`",
                f"- violations: `{json.dumps(snapshots_scenes_intake['violations'])}`",
            ]
        )
    if user_presets_contract is not None:
        lines.extend(
            [
                "",
                "## User Presets Physical Readback Contract",
                "",
                f"- schema_check: `{user_presets_contract['status']}`",
                f"- read_only: `{str(user_presets_contract['read_only']).lower()}`",
                f"- live_write: `{str(user_presets_contract['live_write']).lower()}`",
                "- preset_recall_allowed: "
                f"`{str(user_presets_contract['preset_recall_allowed']).lower()}`",
                "- preset_store_allowed: "
                f"`{str(user_presets_contract['preset_store_allowed']).lower()}`",
                f"- schema_id: `{user_presets_contract['schema_id']}`",
                f"- contract_status: `{user_presets_contract['contract_status']}`",
                f"- surface_id: `{user_presets_contract['surface_id']}`",
                f"- remediation_action_id: `{user_presets_contract['remediation_action_id']}`",
                "- physical_readback_surface_id: "
                f"`{user_presets_contract['physical_readback_surface_id']}`",
                "- physical_readback_action_id: "
                f"`{user_presets_contract['physical_readback_action_id']}`",
                "- fixture_physical_gap_declared: "
                f"`{str(user_presets_contract['fixture_physical_gap_declared']).lower()}`",
                "- remediation_physical_gap_declared: "
                f"`{str(user_presets_contract['remediation_physical_gap_declared']).lower()}`",
                "- physical_fixture_gap_remains: "
                f"`{str(user_presets_contract['physical_fixture_gap_remains']).lower()}`",
                "- live_authority_granted: "
                f"`{str(user_presets_contract['live_authority_granted']).lower()}`",
                "- promotion_families_requiring_surface: "
                f"`{json.dumps(user_presets_contract['promotion_families_requiring_surface'])}`",
                "- required_preset_fields: "
                f"`{json.dumps(user_presets_contract['required_preset_fields'])}`",
                "- required_coverage_flags: "
                f"`{json.dumps(user_presets_contract['required_coverage_flags'])}`",
                "- required_false_safety_flags: "
                f"`{json.dumps(user_presets_contract['required_false_safety_flags'])}`",
                f"- minimum_preset_count: `{user_presets_contract['minimum_preset_count']}`",
                "- target_consistency: "
                f"`{str(user_presets_contract['target_consistency']).lower()}`",
                f"- violations: `{json.dumps(user_presets_contract['violations'])}`",
            ]
        )
    if user_presets_intake is not None:
        lines.extend(
            [
                "",
                "## User Presets Physical Readback Artifact Intake",
                "",
                f"- schema_check: `{user_presets_intake['status']}`",
                f"- read_only: `{str(user_presets_intake['read_only']).lower()}`",
                f"- live_write: `{str(user_presets_intake['live_write']).lower()}`",
                "- mutates_console_state: "
                f"`{str(user_presets_intake['mutates_console_state']).lower()}`",
                f"- queries_console: `{str(user_presets_intake['queries_console']).lower()}`",
                f"- executes_tests: `{str(user_presets_intake['executes_tests']).lower()}`",
                f"- operator_required: `{str(user_presets_intake['operator_required']).lower()}`",
                "- execution_allowed_by_intake: "
                f"`{str(user_presets_intake['execution_allowed_by_intake']).lower()}`",
                "- fixture_closure_allowed_by_intake: "
                f"`{str(user_presets_intake['fixture_closure_allowed_by_intake']).lower()}`",
                "- preset_recall_allowed: "
                f"`{str(user_presets_intake['preset_recall_allowed']).lower()}`",
                "- preset_store_allowed: "
                f"`{str(user_presets_intake['preset_store_allowed']).lower()}`",
                "- live_widening_allowed: "
                f"`{str(user_presets_intake['live_widening_allowed']).lower()}`",
                "- full_autonomous_ready: "
                f"`{str(user_presets_intake['full_autonomous_ready']).lower()}`",
                f"- intake_status: `{user_presets_intake['intake_status']}`",
                f"- schema_id: `{user_presets_intake['schema_id']}`",
                f"- surface_id: `{user_presets_intake['surface_id']}`",
                "- remediation_action_id: "
                f"`{user_presets_intake['remediation_action_id']}`",
                "- required_contract_method_id: "
                f"`{user_presets_intake['required_contract_method_id']}`",
                f"- artifact_count: `{user_presets_intake['artifact_count']}`",
                f"- valid_candidate_count: `{user_presets_intake['valid_candidate_count']}`",
                f"- invalid_candidate_count: `{user_presets_intake['invalid_candidate_count']}`",
                f"- valid_preset_count: `{user_presets_intake['valid_preset_count']}`",
                "- accepted_as_physical_fixture_candidate: "
                f"`{str(user_presets_intake['accepted_as_physical_fixture_candidate']).lower()}`",
                "- operator_review_required: "
                f"`{str(user_presets_intake['operator_review_required']).lower()}`",
                "- physical_fixture_gap_remains: "
                f"`{str(user_presets_intake['physical_fixture_gap_remains']).lower()}`",
                "- live_authority_granted: "
                f"`{str(user_presets_intake['live_authority_granted']).lower()}`",
                "- promotion_families_requiring_surface: "
                f"`{json.dumps(user_presets_intake['promotion_families_requiring_surface'])}`",
                "- artifact_search_roots: "
                f"`{json.dumps(user_presets_intake['artifact_search_roots'])}`",
                "- artifact_source_ids: "
                f"`{json.dumps(user_presets_intake['artifact_source_ids'])}`",
                f"- blocked_by: `{json.dumps(user_presets_intake['blocked_by'])}`",
                "- intake_violations: "
                f"`{json.dumps(user_presets_intake['intake_violations'])}`",
                "- target_consistency: "
                f"`{str(user_presets_intake['target_consistency']).lower()}`",
                f"- violations: `{json.dumps(user_presets_intake['violations'])}`",
            ]
        )
    if dlive_state_contract is not None:
        lines.extend(
            [
                "",
                "## dLive State Physical Readback Contract",
                "",
                f"- schema_check: `{dlive_state_contract['status']}`",
                f"- read_only: `{str(dlive_state_contract['read_only']).lower()}`",
                f"- live_write: `{str(dlive_state_contract['live_write']).lower()}`",
                "- midi_send_allowed: "
                f"`{str(dlive_state_contract['midi_send_allowed']).lower()}`",
                "- tcp_send_allowed: "
                f"`{str(dlive_state_contract['tcp_send_allowed']).lower()}`",
                "- dlive_write_allowed: "
                f"`{str(dlive_state_contract['dlive_write_allowed']).lower()}`",
                f"- schema_id: `{dlive_state_contract['schema_id']}`",
                f"- contract_status: `{dlive_state_contract['contract_status']}`",
                f"- surface_id: `{dlive_state_contract['surface_id']}`",
                f"- remediation_action_id: `{dlive_state_contract['remediation_action_id']}`",
                "- physical_readback_surface_id: "
                f"`{dlive_state_contract['physical_readback_surface_id']}`",
                "- physical_readback_action_id: "
                f"`{dlive_state_contract['physical_readback_action_id']}`",
                "- fixture_physical_gap_declared: "
                f"`{str(dlive_state_contract['fixture_physical_gap_declared']).lower()}`",
                "- remediation_physical_gap_declared: "
                f"`{str(dlive_state_contract['remediation_physical_gap_declared']).lower()}`",
                "- physical_fixture_gap_remains: "
                f"`{str(dlive_state_contract['physical_fixture_gap_remains']).lower()}`",
                "- live_authority_granted: "
                f"`{str(dlive_state_contract['live_authority_granted']).lower()}`",
                "- promotion_families_requiring_surface: "
                f"`{json.dumps(dlive_state_contract['promotion_families_requiring_surface'])}`",
                "- required_channel_fields: "
                f"`{json.dumps(dlive_state_contract['required_channel_fields'])}`",
                "- required_midi_state_fields: "
                f"`{json.dumps(dlive_state_contract['required_midi_state_fields'])}`",
                "- required_parity_fields: "
                f"`{json.dumps(dlive_state_contract['required_parity_fields'])}`",
                "- required_coverage_flags: "
                f"`{json.dumps(dlive_state_contract['required_coverage_flags'])}`",
                "- required_false_safety_flags: "
                f"`{json.dumps(dlive_state_contract['required_false_safety_flags'])}`",
                f"- minimum_channel_count: `{dlive_state_contract['minimum_channel_count']}`",
                "- target_consistency: "
                f"`{str(dlive_state_contract['target_consistency']).lower()}`",
                f"- violations: `{json.dumps(dlive_state_contract['violations'])}`",
            ]
        )
    if dlive_state_intake is not None:
        lines.extend(
            [
                "",
                "## dLive State Physical Readback Artifact Intake",
                "",
                f"- schema_check: `{dlive_state_intake['status']}`",
                f"- read_only: `{str(dlive_state_intake['read_only']).lower()}`",
                f"- live_write: `{str(dlive_state_intake['live_write']).lower()}`",
                "- mutates_console_state: "
                f"`{str(dlive_state_intake['mutates_console_state']).lower()}`",
                f"- queries_console: `{str(dlive_state_intake['queries_console']).lower()}`",
                f"- executes_tests: `{str(dlive_state_intake['executes_tests']).lower()}`",
                f"- operator_required: `{str(dlive_state_intake['operator_required']).lower()}`",
                "- execution_allowed_by_intake: "
                f"`{str(dlive_state_intake['execution_allowed_by_intake']).lower()}`",
                "- fixture_closure_allowed_by_intake: "
                f"`{str(dlive_state_intake['fixture_closure_allowed_by_intake']).lower()}`",
                "- midi_send_allowed: "
                f"`{str(dlive_state_intake['midi_send_allowed']).lower()}`",
                "- tcp_send_allowed: "
                f"`{str(dlive_state_intake['tcp_send_allowed']).lower()}`",
                "- dlive_write_allowed: "
                f"`{str(dlive_state_intake['dlive_write_allowed']).lower()}`",
                "- dlive_apply_allowed: "
                f"`{str(dlive_state_intake['dlive_apply_allowed']).lower()}`",
                "- live_widening_allowed: "
                f"`{str(dlive_state_intake['live_widening_allowed']).lower()}`",
                f"- full_autonomous_ready: `{str(dlive_state_intake['full_autonomous_ready']).lower()}`",
                f"- intake_status: `{dlive_state_intake['intake_status']}`",
                f"- schema_id: `{dlive_state_intake['schema_id']}`",
                f"- surface_id: `{dlive_state_intake['surface_id']}`",
                f"- remediation_action_id: `{dlive_state_intake['remediation_action_id']}`",
                "- required_contract_method_id: "
                f"`{dlive_state_intake['required_contract_method_id']}`",
                f"- artifact_count: `{dlive_state_intake['artifact_count']}`",
                f"- valid_candidate_count: `{dlive_state_intake['valid_candidate_count']}`",
                f"- invalid_candidate_count: `{dlive_state_intake['invalid_candidate_count']}`",
                f"- valid_channel_count: `{dlive_state_intake['valid_channel_count']}`",
                "- accepted_as_physical_fixture_candidate: "
                f"`{str(dlive_state_intake['accepted_as_physical_fixture_candidate']).lower()}`",
                "- operator_review_required: "
                f"`{str(dlive_state_intake['operator_review_required']).lower()}`",
                "- physical_fixture_gap_remains: "
                f"`{str(dlive_state_intake['physical_fixture_gap_remains']).lower()}`",
                "- live_authority_granted: "
                f"`{str(dlive_state_intake['live_authority_granted']).lower()}`",
                "- promotion_families_requiring_surface: "
                f"`{json.dumps(dlive_state_intake['promotion_families_requiring_surface'])}`",
                "- artifact_search_roots: "
                f"`{json.dumps(dlive_state_intake['artifact_search_roots'])}`",
                "- artifact_source_ids: "
                f"`{json.dumps(dlive_state_intake['artifact_source_ids'])}`",
                f"- blocked_by: `{json.dumps(dlive_state_intake['blocked_by'])}`",
                "- intake_violations: "
                f"`{json.dumps(dlive_state_intake['intake_violations'])}`",
                "- target_consistency: "
                f"`{str(dlive_state_intake['target_consistency']).lower()}`",
                f"- violations: `{json.dumps(dlive_state_intake['violations'])}`",
            ]
        )
    if artifact_review_queue is not None:
        lines.extend(
            [
                "",
                "## Read Surface Physical Readback Artifact Review Queue",
                "",
                f"- schema_check: `{artifact_review_queue['status']}`",
                f"- read_only: `{str(artifact_review_queue['read_only']).lower()}`",
                f"- live_write: `{str(artifact_review_queue['live_write']).lower()}`",
                "- mutates_console_state: "
                f"`{str(artifact_review_queue['mutates_console_state']).lower()}`",
                f"- queries_console: `{str(artifact_review_queue['queries_console']).lower()}`",
                f"- executes_tests: `{str(artifact_review_queue['executes_tests']).lower()}`",
                f"- operator_required: `{str(artifact_review_queue['operator_required']).lower()}`",
                "- execution_allowed_by_queue: "
                f"`{str(artifact_review_queue['execution_allowed_by_queue']).lower()}`",
                "- fixture_closure_allowed_by_queue: "
                f"`{str(artifact_review_queue['fixture_closure_allowed_by_queue']).lower()}`",
                "- live_widening_allowed: "
                f"`{str(artifact_review_queue['live_widening_allowed']).lower()}`",
                "- full_autonomous_ready: "
                f"`{str(artifact_review_queue['full_autonomous_ready']).lower()}`",
                f"- queue_status: `{artifact_review_queue['queue_status']}`",
                f"- intake_surface_count: `{artifact_review_queue['intake_surface_count']}`",
                f"- available_intake_count: `{artifact_review_queue['available_intake_count']}`",
                f"- artifact_count: `{artifact_review_queue['artifact_count']}`",
                f"- valid_candidate_count: `{artifact_review_queue['valid_candidate_count']}`",
                f"- invalid_candidate_count: `{artifact_review_queue['invalid_candidate_count']}`",
                f"- review_candidate_count: `{artifact_review_queue['review_candidate_count']}`",
                "- review_candidates_available: "
                f"`{str(artifact_review_queue['review_candidates_available']).lower()}`",
                "- missing_artifact_surface_count: "
                f"`{artifact_review_queue['missing_artifact_surface_count']}`",
                "- unsafe_authority_claim_count: "
                f"`{artifact_review_queue['unsafe_authority_claim_count']}`",
                "- candidate_source_id_missing_count: "
                f"`{artifact_review_queue['candidate_source_id_missing_count']}`",
                "- target_inconsistent_surface_count: "
                f"`{artifact_review_queue['target_inconsistent_surface_count']}`",
                "- operator_review_required: "
                f"`{str(artifact_review_queue['operator_review_required']).lower()}`",
                "- physical_fixture_gap_remains: "
                f"`{str(artifact_review_queue['physical_fixture_gap_remains']).lower()}`",
                "- live_authority_granted: "
                f"`{str(artifact_review_queue['live_authority_granted']).lower()}`",
                "- review_candidate_surface_ids: "
                f"`{json.dumps(artifact_review_queue['review_candidate_surface_ids'])}`",
                "- invalid_surface_ids: "
                f"`{json.dumps(artifact_review_queue['invalid_surface_ids'])}`",
                "- missing_artifact_surface_ids: "
                f"`{json.dumps(artifact_review_queue['missing_artifact_surface_ids'])}`",
                "- unsafe_authority_surface_ids: "
                f"`{json.dumps(artifact_review_queue['unsafe_authority_surface_ids'])}`",
                "- candidate_source_id_missing_surface_ids: "
                f"`{json.dumps(artifact_review_queue['candidate_source_id_missing_surface_ids'])}`",
                "- target_inconsistent_surface_ids: "
                f"`{json.dumps(artifact_review_queue['target_inconsistent_surface_ids'])}`",
                "- promotion_families_requiring_surface: "
                f"`{json.dumps(artifact_review_queue['promotion_families_requiring_surface'])}`",
                "- queue_item_method_ids: "
                f"`{json.dumps(artifact_review_queue['queue_item_method_ids'])}`",
                f"- blocked_by: `{json.dumps(artifact_review_queue['blocked_by'])}`",
                f"- violations: `{json.dumps(artifact_review_queue['violations'])}`",
            ]
        )
    if artifact_review_queue_artifact is not None:
        lines.extend(
            [
                "",
                "## Read Surface Physical Readback Artifact Review Queue Artifact",
                "",
                f"- schema_check: `{artifact_review_queue_artifact['status']}`",
                f"- read_only: `{str(artifact_review_queue_artifact['read_only']).lower()}`",
                f"- live_write: `{str(artifact_review_queue_artifact['live_write']).lower()}`",
                "- mutates_console_state: "
                f"`{str(artifact_review_queue_artifact['mutates_console_state']).lower()}`",
                "- queries_console: "
                f"`{str(artifact_review_queue_artifact['queries_console']).lower()}`",
                "- executes_tests: "
                f"`{str(artifact_review_queue_artifact['executes_tests']).lower()}`",
                "- operator_required: "
                f"`{str(artifact_review_queue_artifact['operator_required']).lower()}`",
                "- execution_allowed_by_queue: "
                f"`{str(artifact_review_queue_artifact['execution_allowed_by_queue']).lower()}`",
                "- fixture_closure_allowed_by_queue: "
                f"`{str(artifact_review_queue_artifact['fixture_closure_allowed_by_queue']).lower()}`",
                "- live_authority_granted: "
                f"`{str(artifact_review_queue_artifact['live_authority_granted']).lower()}`",
                "- full_autonomous_ready: "
                f"`{str(artifact_review_queue_artifact['full_autonomous_ready']).lower()}`",
                f"- queue_path: `{artifact_review_queue_artifact['queue_path']}`",
                f"- queue_status: `{artifact_review_queue_artifact['queue_status']}`",
                f"- intake_surface_count: `{artifact_review_queue_artifact['intake_surface_count']}`",
                f"- available_intake_count: `{artifact_review_queue_artifact['available_intake_count']}`",
                f"- artifact_count: `{artifact_review_queue_artifact['artifact_count']}`",
                f"- valid_candidate_count: `{artifact_review_queue_artifact['valid_candidate_count']}`",
                f"- invalid_candidate_count: `{artifact_review_queue_artifact['invalid_candidate_count']}`",
                f"- review_candidate_count: `{artifact_review_queue_artifact['review_candidate_count']}`",
                "- review_candidates_available: "
                f"`{str(artifact_review_queue_artifact['review_candidates_available']).lower()}`",
                "- missing_artifact_surface_count: "
                f"`{artifact_review_queue_artifact['missing_artifact_surface_count']}`",
                "- unsafe_authority_claim_count: "
                f"`{artifact_review_queue_artifact['unsafe_authority_claim_count']}`",
                "- candidate_source_id_missing_count: "
                f"`{artifact_review_queue_artifact['candidate_source_id_missing_count']}`",
                "- target_inconsistent_surface_count: "
                f"`{artifact_review_queue_artifact['target_inconsistent_surface_count']}`",
                "- physical_fixture_gap_remains: "
                f"`{str(artifact_review_queue_artifact['physical_fixture_gap_remains']).lower()}`",
                "- review_candidate_surface_ids: "
                f"`{json.dumps(artifact_review_queue_artifact['review_candidate_surface_ids'])}`",
                "- missing_artifact_surface_ids: "
                f"`{json.dumps(artifact_review_queue_artifact['missing_artifact_surface_ids'])}`",
                "- candidate_source_id_missing_surface_ids: "
                f"`{json.dumps(artifact_review_queue_artifact['candidate_source_id_missing_surface_ids'])}`",
                "- queue_item_method_ids: "
                f"`{json.dumps(artifact_review_queue_artifact['queue_item_method_ids'])}`",
                f"- blocked_by: `{json.dumps(artifact_review_queue_artifact['blocked_by'])}`",
                "- focused_validation: "
                f"`{json.dumps(artifact_review_queue_artifact['focused_validation'])}`",
                f"- violations: `{json.dumps(artifact_review_queue_artifact['violations'])}`",
            ]
        )
    if acceptance_template_artifact is not None:
        lines.extend(
            [
                "",
                "## Read Surface Physical Readback Acceptance Template Artifact",
                "",
                f"- schema_check: `{acceptance_template_artifact['status']}`",
                f"- read_only: `{str(acceptance_template_artifact['read_only']).lower()}`",
                f"- live_write: `{str(acceptance_template_artifact['live_write']).lower()}`",
                "- mutates_console_state: "
                f"`{str(acceptance_template_artifact['mutates_console_state']).lower()}`",
                "- queries_console: "
                f"`{str(acceptance_template_artifact['queries_console']).lower()}`",
                "- executes_tests: "
                f"`{str(acceptance_template_artifact['executes_tests']).lower()}`",
                "- operator_required: "
                f"`{str(acceptance_template_artifact['operator_required']).lower()}`",
                "- execution_allowed_by_template: "
                f"`{str(acceptance_template_artifact['execution_allowed_by_template']).lower()}`",
                "- fixture_closure_allowed_by_template: "
                f"`{str(acceptance_template_artifact['fixture_closure_allowed_by_template']).lower()}`",
                "- live_authority_granted: "
                f"`{str(acceptance_template_artifact['live_authority_granted']).lower()}`",
                "- full_autonomous_ready: "
                f"`{str(acceptance_template_artifact['full_autonomous_ready']).lower()}`",
                f"- template_path: `{acceptance_template_artifact['template_path']}`",
                f"- template_status: `{acceptance_template_artifact['template_status']}`",
                "- source_queue_status: "
                f"`{acceptance_template_artifact['source_queue_status']}`",
                "- review_candidate_count: "
                f"`{acceptance_template_artifact['review_candidate_count']}`",
                "- template_record_count: "
                f"`{acceptance_template_artifact['template_record_count']}`",
                "- candidate_trace_count: "
                f"`{acceptance_template_artifact['candidate_trace_count']}`",
                "- required_operator_edits: "
                f"`{json.dumps(acceptance_template_artifact['required_operator_edits'])}`",
                "- operator_sequence_count: "
                f"`{acceptance_template_artifact['operator_sequence_count']}`",
                f"- blocked_by: `{json.dumps(acceptance_template_artifact['blocked_by'])}`",
                "- record_unsafe_flags: "
                f"`{json.dumps(acceptance_template_artifact['record_unsafe_flags'])}`",
                f"- violations: `{json.dumps(acceptance_template_artifact['violations'])}`",
            ]
        )
    if fixture_acceptance_ledger is not None:
        lines.extend(
            [
                "",
                "## Read Surface Physical Readback Fixture Acceptance Ledger",
                "",
                f"- schema_check: `{fixture_acceptance_ledger['status']}`",
                f"- read_only: `{str(fixture_acceptance_ledger['read_only']).lower()}`",
                f"- live_write: `{str(fixture_acceptance_ledger['live_write']).lower()}`",
                "- mutates_console_state: "
                f"`{str(fixture_acceptance_ledger['mutates_console_state']).lower()}`",
                "- queries_console: "
                f"`{str(fixture_acceptance_ledger['queries_console']).lower()}`",
                "- executes_tests: "
                f"`{str(fixture_acceptance_ledger['executes_tests']).lower()}`",
                "- operator_required: "
                f"`{str(fixture_acceptance_ledger['operator_required']).lower()}`",
                "- execution_allowed_by_acceptance: "
                f"`{str(fixture_acceptance_ledger['execution_allowed_by_acceptance']).lower()}`",
                "- fixture_closure_allowed_by_acceptance: "
                f"`{str(fixture_acceptance_ledger['fixture_closure_allowed_by_acceptance']).lower()}`",
                "- live_widening_allowed: "
                f"`{str(fixture_acceptance_ledger['live_widening_allowed']).lower()}`",
                "- live_authority_granted: "
                f"`{str(fixture_acceptance_ledger['live_authority_granted']).lower()}`",
                "- full_autonomous_ready: "
                f"`{str(fixture_acceptance_ledger['full_autonomous_ready']).lower()}`",
                f"- ledger_status: `{fixture_acceptance_ledger['ledger_status']}`",
                f"- queue_status: `{fixture_acceptance_ledger['queue_status']}`",
                "- review_candidate_count: "
                f"`{fixture_acceptance_ledger['review_candidate_count']}`",
                "- acceptance_record_count: "
                f"`{fixture_acceptance_ledger['acceptance_record_count']}`",
                "- accepted_record_count: "
                f"`{fixture_acceptance_ledger['accepted_record_count']}`",
                "- rejected_record_count: "
                f"`{fixture_acceptance_ledger['rejected_record_count']}`",
                "- invalid_record_count: "
                f"`{fixture_acceptance_ledger['invalid_record_count']}`",
                "- candidate_trace_missing_count: "
                f"`{fixture_acceptance_ledger['candidate_trace_missing_count']}`",
                "- candidate_trace_mismatch_count: "
                f"`{fixture_acceptance_ledger['candidate_trace_mismatch_count']}`",
                "- unreviewed_candidate_count: "
                f"`{fixture_acceptance_ledger['unreviewed_candidate_count']}`",
                "- operator_acceptance_required: "
                f"`{str(fixture_acceptance_ledger['operator_acceptance_required']).lower()}`",
                "- operator_review_required: "
                f"`{str(fixture_acceptance_ledger['operator_review_required']).lower()}`",
                "- physical_fixture_gap_remains: "
                f"`{str(fixture_acceptance_ledger['physical_fixture_gap_remains']).lower()}`",
                "- accepted_surface_ids: "
                f"`{json.dumps(fixture_acceptance_ledger['accepted_surface_ids'])}`",
                "- rejected_surface_ids: "
                f"`{json.dumps(fixture_acceptance_ledger['rejected_surface_ids'])}`",
                "- invalid_record_ids: "
                f"`{json.dumps(fixture_acceptance_ledger['invalid_record_ids'])}`",
                "- unreviewed_candidate_surface_ids: "
                f"`{json.dumps(fixture_acceptance_ledger['unreviewed_candidate_surface_ids'])}`",
                f"- blocked_by: `{json.dumps(fixture_acceptance_ledger['blocked_by'])}`",
                f"- violations: `{json.dumps(fixture_acceptance_ledger['violations'])}`",
            ]
        )
    if fixture_closure_plan is not None:
        lines.extend(
            [
                "",
                "## Read Surface Physical Readback Fixture Closure Plan",
                "",
                f"- schema_check: `{fixture_closure_plan['status']}`",
                f"- read_only: `{str(fixture_closure_plan['read_only']).lower()}`",
                f"- live_write: `{str(fixture_closure_plan['live_write']).lower()}`",
                "- mutates_console_state: "
                f"`{str(fixture_closure_plan['mutates_console_state']).lower()}`",
                "- queries_console: "
                f"`{str(fixture_closure_plan['queries_console']).lower()}`",
                "- executes_tests: "
                f"`{str(fixture_closure_plan['executes_tests']).lower()}`",
                "- operator_required: "
                f"`{str(fixture_closure_plan['operator_required']).lower()}`",
                "- execution_allowed_by_plan: "
                f"`{str(fixture_closure_plan['execution_allowed_by_plan']).lower()}`",
                "- fixture_update_allowed_by_plan: "
                f"`{str(fixture_closure_plan['fixture_update_allowed_by_plan']).lower()}`",
                "- fixture_closure_allowed_by_plan: "
                f"`{str(fixture_closure_plan['fixture_closure_allowed_by_plan']).lower()}`",
                "- live_widening_allowed: "
                f"`{str(fixture_closure_plan['live_widening_allowed']).lower()}`",
                "- live_authority_granted: "
                f"`{str(fixture_closure_plan['live_authority_granted']).lower()}`",
                "- full_autonomous_ready: "
                f"`{str(fixture_closure_plan['full_autonomous_ready']).lower()}`",
                f"- plan_status: `{fixture_closure_plan['plan_status']}`",
                f"- ledger_status: `{fixture_closure_plan['ledger_status']}`",
                "- review_candidate_count: "
                f"`{fixture_closure_plan['review_candidate_count']}`",
                "- acceptance_record_count: "
                f"`{fixture_closure_plan['acceptance_record_count']}`",
                "- accepted_record_count: "
                f"`{fixture_closure_plan['accepted_record_count']}`",
                "- rejected_record_count: "
                f"`{fixture_closure_plan['rejected_record_count']}`",
                "- invalid_record_count: "
                f"`{fixture_closure_plan['invalid_record_count']}`",
                "- unreviewed_candidate_count: "
                f"`{fixture_closure_plan['unreviewed_candidate_count']}`",
                "- closure_candidate_count: "
                f"`{fixture_closure_plan['closure_candidate_count']}`",
                "- manual_fixture_closure_candidate_available: "
                f"`{str(fixture_closure_plan['manual_fixture_closure_candidate_available']).lower()}`",
                "- manual_fixture_closure_required: "
                f"`{str(fixture_closure_plan['manual_fixture_closure_required']).lower()}`",
                "- operator_review_required: "
                f"`{str(fixture_closure_plan['operator_review_required']).lower()}`",
                "- physical_fixture_gap_remains: "
                f"`{str(fixture_closure_plan['physical_fixture_gap_remains']).lower()}`",
                "- closure_candidate_surface_ids: "
                f"`{json.dumps(fixture_closure_plan['closure_candidate_surface_ids'])}`",
                "- rejected_surface_ids: "
                f"`{json.dumps(fixture_closure_plan['rejected_surface_ids'])}`",
                "- invalid_record_ids: "
                f"`{json.dumps(fixture_closure_plan['invalid_record_ids'])}`",
                "- unreviewed_candidate_surface_ids: "
                f"`{json.dumps(fixture_closure_plan['unreviewed_candidate_surface_ids'])}`",
                f"- blocked_by: `{json.dumps(fixture_closure_plan['blocked_by'])}`",
                f"- violations: `{json.dumps(fixture_closure_plan['violations'])}`",
            ]
        )
    if fixture_closure_plan_artifact is not None:
        lines.extend(
            [
                "",
                "## Read Surface Physical Readback Fixture Closure Plan Artifact",
                "",
                f"- schema_check: `{fixture_closure_plan_artifact['status']}`",
                f"- read_only: `{str(fixture_closure_plan_artifact['read_only']).lower()}`",
                f"- live_write: `{str(fixture_closure_plan_artifact['live_write']).lower()}`",
                "- mutates_console_state: "
                f"`{str(fixture_closure_plan_artifact['mutates_console_state']).lower()}`",
                "- queries_console: "
                f"`{str(fixture_closure_plan_artifact['queries_console']).lower()}`",
                "- executes_tests: "
                f"`{str(fixture_closure_plan_artifact['executes_tests']).lower()}`",
                "- operator_required: "
                f"`{str(fixture_closure_plan_artifact['operator_required']).lower()}`",
                "- execution_allowed_by_plan: "
                f"`{str(fixture_closure_plan_artifact['execution_allowed_by_plan']).lower()}`",
                "- fixture_update_allowed_by_plan: "
                f"`{str(fixture_closure_plan_artifact['fixture_update_allowed_by_plan']).lower()}`",
                "- fixture_closure_allowed_by_plan: "
                f"`{str(fixture_closure_plan_artifact['fixture_closure_allowed_by_plan']).lower()}`",
                "- live_authority_granted: "
                f"`{str(fixture_closure_plan_artifact['live_authority_granted']).lower()}`",
                "- full_autonomous_ready: "
                f"`{str(fixture_closure_plan_artifact['full_autonomous_ready']).lower()}`",
                f"- plan_path: `{fixture_closure_plan_artifact['plan_path']}`",
                f"- plan_status: `{fixture_closure_plan_artifact['plan_status']}`",
                f"- ledger_status: `{fixture_closure_plan_artifact['ledger_status']}`",
                "- acceptance_record_count: "
                f"`{fixture_closure_plan_artifact['acceptance_record_count']}`",
                "- accepted_record_count: "
                f"`{fixture_closure_plan_artifact['accepted_record_count']}`",
                "- invalid_record_count: "
                f"`{fixture_closure_plan_artifact['invalid_record_count']}`",
                "- unreviewed_candidate_count: "
                f"`{fixture_closure_plan_artifact['unreviewed_candidate_count']}`",
                "- closure_candidate_count: "
                f"`{fixture_closure_plan_artifact['closure_candidate_count']}`",
                "- manual_fixture_closure_candidate_available: "
                f"`{str(fixture_closure_plan_artifact['manual_fixture_closure_candidate_available']).lower()}`",
                "- physical_fixture_gap_remains: "
                f"`{str(fixture_closure_plan_artifact['physical_fixture_gap_remains']).lower()}`",
                "- closure_candidate_surface_ids: "
                f"`{json.dumps(fixture_closure_plan_artifact['closure_candidate_surface_ids'])}`",
                f"- blocked_by: `{json.dumps(fixture_closure_plan_artifact['blocked_by'])}`",
                f"- next_global_step: `{fixture_closure_plan_artifact['next_global_step']}`",
                "- focused_validation: "
                f"`{json.dumps(fixture_closure_plan_artifact['focused_validation'])}`",
                f"- violations: `{json.dumps(fixture_closure_plan_artifact['violations'])}`",
            ]
        )
    if next_plan is not None:
        lines.extend(
            [
                "",
                "## Next WING Proof Plan",
                "",
                f"- schema_check: `{next_plan['status']}`",
                f"- read_only: `{str(next_plan['read_only']).lower()}`",
                f"- live_write: `{str(next_plan['live_write']).lower()}`",
                f"- execution_allowed_by_plan: `{str(next_plan['execution_allowed_by_plan']).lower()}`",
                f"- operator_required: `{str(next_plan['operator_required']).lower()}`",
                f"- full_autonomous_ready: `{str(next_plan['full_autonomous_ready']).lower()}`",
                f"- plan_status: `{next_plan['plan_status']}`",
                f"- step_count: `{next_plan['step_count']}`",
                f"- planned_value_types: `{json.dumps(next_plan['planned_value_types'])}`",
                f"- planned_channels: `{json.dumps(next_plan['planned_channels'])}`",
                "- readonly_preflight_required: "
                f"`{str(next_plan['readonly_preflight_required']).lower()}`",
                "- readonly_preflight_candidate_available: "
                f"`{str(next_plan['readonly_preflight_candidate_available']).lower()}`",
                "- readonly_preflight_evidence_status: "
                f"`{next_plan['readonly_preflight_evidence_status']}`",
                "- readonly_preflight_blocked_by: "
                f"`{json.dumps(next_plan['readonly_preflight_blocked_by'])}`",
                "- readonly_preflight_prestart_check_required: "
                f"`{str(next_plan['readonly_preflight_prestart_check_required']).lower()}`",
                "- readonly_preflight_prestart_check_must_pass_before_starting_backend: "
                f"`{str(next_plan['readonly_preflight_prestart_check_must_pass_before_starting_backend']).lower()}`",
                "- readonly_preflight_prestart_check_command: "
                f"`{json.dumps(next_plan['readonly_preflight_prestart_check_command'])}`",
                "- readonly_preflight_prestart_check_failure_detail_fields: "
                f"`{json.dumps(next_plan['readonly_preflight_prestart_check_failure_detail_fields'])}`",
                "- readonly_preflight_step_count: "
                f"`{next_plan['readonly_preflight_step_count']}`",
                "- readonly_preflight_review_decision_required: "
                f"`{str(next_plan['readonly_preflight_review_decision_required']).lower()}`",
                "- readonly_preflight_review_decision_accepted: "
                f"`{str(next_plan['readonly_preflight_review_decision_accepted']).lower()}`",
                "- readonly_preflight_review_decision_status: "
                f"`{next_plan['readonly_preflight_review_decision_status']}`",
                "- readonly_preflight_review_decision_blocked_by: "
                f"`{json.dumps(next_plan['readonly_preflight_review_decision_blocked_by'])}`",
                "- readonly_preflight_review_decision_accepted_record_count: "
                f"`{next_plan['readonly_preflight_review_decision_accepted_record_count']}`",
                "- readonly_preflight_review_decision_accepted_source_ids: "
                f"`{json.dumps(next_plan['readonly_preflight_review_decision_accepted_source_ids'])}`",
                "- readonly_preflight_review_decision_candidate_prestart_contract_valid: "
                f"`{str(next_plan['readonly_preflight_review_decision_candidate_prestart_contract_valid']).lower()}`",
                "- readonly_preflight_review_decision_candidate_prestart_source_ids: "
                f"`{json.dumps(next_plan['readonly_preflight_review_decision_candidate_prestart_source_ids'])}`",
                "- readonly_preflight_review_decision_rejected_source_ids: "
                f"`{json.dumps(next_plan['readonly_preflight_review_decision_rejected_source_ids'])}`",
                "- readonly_preflight_review_decision_invalid_record_count: "
                f"`{next_plan['readonly_preflight_review_decision_invalid_record_count']}`",
                "- readonly_preflight_review_decision_invalid_record_ids: "
                f"`{json.dumps(next_plan['readonly_preflight_review_decision_invalid_record_ids'])}`",
                "- readonly_preflight_review_decision_duplicate_source_ids: "
                f"`{json.dumps(next_plan['readonly_preflight_review_decision_duplicate_source_ids'])}`",
                "- readonly_preflight_review_decision_conflicting_source_ids: "
                f"`{json.dumps(next_plan['readonly_preflight_review_decision_conflicting_source_ids'])}`",
                "- readonly_preflight_review_decision_unreviewed_candidate_count: "
                f"`{next_plan['readonly_preflight_review_decision_unreviewed_candidate_count']}`",
                f"- global_blockers: `{json.dumps(next_plan['global_blockers'])}`",
                f"- violations: `{json.dumps(next_plan['violations'])}`",
            ]
        )
    if failed:
        lines.extend(["", "## Failed Checks", ""])
        lines.extend(f"- `{name}`" for name in failed)
    _write_text(run_dir / "summary.md", "\n".join(lines) + "\n")
    return summary


def run_release_checks(
    *,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    include_frontend: bool = True,
    include_package: bool = False,
    quick: bool = False,
    continue_on_failure: bool = False,
    backend_port: int = DEFAULT_BACKEND_PORT,
    backend_listener_confirmation: Path | None = None,
    command_timeout_sec: float | None = DEFAULT_COMMAND_TIMEOUT_SEC,
) -> dict[str, object]:
    generated_at = utc_now()
    run_dir = artifact_root / f"{compact_stamp()}_no_wing_release_check"
    log_dir = run_dir / "logs"
    proof_report_path = proof_report_path_for_run(run_dir)

    checks = build_checks(
        run_dir=run_dir,
        include_frontend=include_frontend,
        include_package=include_package,
        quick=quick,
        command_timeout_sec=command_timeout_sec,
    )
    validate_repo_local_commands(checks)
    validate_no_live_commands(checks)

    results: list[CheckResult] = [
        check_port_free(
            backend_port,
            listener_confirmation_path=backend_listener_confirmation,
        )
    ]
    if results[-1].status != "passed" and not continue_on_failure:
        summary = write_summary(run_dir, results, generated_at=generated_at)
        summary["run_dir"] = str(run_dir)
        return summary

    log_dir.mkdir(parents=True, exist_ok=True)
    for check in checks:
        result = run_command_check(check, log_dir)
        results.append(result)
        if result.status != "passed" and not continue_on_failure:
            break
        if check.name == "wing_readonly_preflight_plan" and result.status == "passed":
            plan_artifact_result = validate_wing_readonly_preflight_plan_artifact_snapshot(
                wing_readonly_preflight_plan_path_for_run(run_dir)
            )
            results.append(plan_artifact_result)
            if plan_artifact_result.status != "passed" and not continue_on_failure:
                break
        if (
            check.name == "wing_readonly_preflight_handoff_packet"
            and result.status == "passed"
        ):
            handoff_artifact_result = (
                validate_wing_readonly_preflight_handoff_packet_artifact_snapshot(
                    wing_readonly_preflight_handoff_packet_path_for_run(run_dir),
                    plan_path=wing_readonly_preflight_plan_path_for_run(run_dir),
                )
            )
            results.append(handoff_artifact_result)
            if handoff_artifact_result.status != "passed" and not continue_on_failure:
                break
        if (
            check.name == "read_surface_physical_readback_capture_packet"
            and result.status == "passed"
        ):
            capture_packet_result = (
                validate_read_surface_physical_readback_capture_packet_artifact_snapshot(
                    read_surface_physical_readback_capture_packet_path_for_run(run_dir)
                )
            )
            results.append(capture_packet_result)
            if capture_packet_result.status != "passed" and not continue_on_failure:
                break
        if (
            check.name == "read_surface_physical_readback_capture_template"
            and result.status == "passed"
        ):
            capture_template_result = (
                validate_read_surface_physical_readback_capture_template_artifact_snapshot(
                    read_surface_physical_readback_capture_template_path_for_run(
                        run_dir
                    )
                )
            )
            results.append(capture_template_result)
            if capture_template_result.status != "passed" and not continue_on_failure:
                break
        if (
            check.name == "read_surface_physical_readback_artifact_review_queue"
            and result.status == "passed"
        ):
            review_queue_artifact_result = (
                validate_read_surface_physical_readback_artifact_review_queue_artifact_snapshot(
                    read_surface_physical_readback_artifact_review_queue_path_for_run(
                        run_dir
                    )
                )
            )
            results.append(review_queue_artifact_result)
            if (
                review_queue_artifact_result.status != "passed"
                and not continue_on_failure
            ):
                break
        if (
            check.name == "read_surface_physical_readback_acceptance_template"
            and result.status == "passed"
        ):
            acceptance_template_result = (
                validate_read_surface_physical_readback_acceptance_template_artifact_snapshot(
                    read_surface_physical_readback_acceptance_template_path_for_run(
                        run_dir
                    )
                )
            )
            results.append(acceptance_template_result)
            if acceptance_template_result.status != "passed" and not continue_on_failure:
                break
        if (
            check.name == "read_surface_physical_readback_fixture_closure_plan"
            and result.status == "passed"
        ):
            fixture_closure_artifact_result = (
                validate_read_surface_physical_readback_fixture_closure_plan_artifact_snapshot(
                    read_surface_physical_readback_fixture_closure_plan_path_for_run(
                        run_dir
                    )
                )
            )
            results.append(fixture_closure_artifact_result)
            if (
                fixture_closure_artifact_result.status != "passed"
                and not continue_on_failure
            ):
                break
        if check.name == "read_surface_fixture_evidence" and result.status == "passed":
            read_fixture_result = validate_read_surface_fixture_evidence_snapshot()
            results.append(read_fixture_result)
            if read_fixture_result.status != "passed" and not continue_on_failure:
                break
            read_remediation_result = validate_read_surface_remediation_plan_snapshot()
            results.append(read_remediation_result)
            if read_remediation_result.status != "passed" and not continue_on_failure:
                break
            read_capture_plan_result = (
                validate_read_surface_physical_readback_capture_plan_snapshot()
            )
            results.append(read_capture_plan_result)
            if read_capture_plan_result.status != "passed" and not continue_on_failure:
                break
            wing_readonly_preflight_result = (
                validate_wing_readonly_preflight_evidence_snapshot()
            )
            results.append(wing_readonly_preflight_result)
            if (
                wing_readonly_preflight_result.status != "passed"
                and not continue_on_failure
            ):
                break
            wing_readonly_preflight_review_decision_result = (
                validate_wing_readonly_preflight_review_decision_ledger_snapshot()
            )
            results.append(wing_readonly_preflight_review_decision_result)
            if (
                wing_readonly_preflight_review_decision_result.status != "passed"
                and not continue_on_failure
            ):
                break
            raw_mixer_contract_result = validate_raw_mixer_state_physical_readback_contract_snapshot()
            results.append(raw_mixer_contract_result)
            if raw_mixer_contract_result.status != "passed" and not continue_on_failure:
                break
            raw_mixer_intake_result = (
                validate_raw_mixer_state_physical_readback_artifact_intake_snapshot()
            )
            results.append(raw_mixer_intake_result)
            if raw_mixer_intake_result.status != "passed" and not continue_on_failure:
                break
            channel_basic_contract_result = (
                validate_channel_basic_controls_physical_readback_contract_snapshot()
            )
            results.append(channel_basic_contract_result)
            if channel_basic_contract_result.status != "passed" and not continue_on_failure:
                break
            channel_basic_intake_result = (
                validate_channel_basic_controls_physical_readback_artifact_intake_snapshot()
            )
            results.append(channel_basic_intake_result)
            if channel_basic_intake_result.status != "passed" and not continue_on_failure:
                break
            channel_processing_contract_result = (
                validate_channel_processing_physical_readback_contract_snapshot()
            )
            results.append(channel_processing_contract_result)
            if channel_processing_contract_result.status != "passed" and not continue_on_failure:
                break
            channel_processing_intake_result = (
                validate_channel_processing_physical_readback_artifact_intake_snapshot()
            )
            results.append(channel_processing_intake_result)
            if channel_processing_intake_result.status != "passed" and not continue_on_failure:
                break
            bus_group_main_dca_contract_result = (
                validate_bus_group_main_dca_physical_readback_contract_snapshot()
            )
            results.append(bus_group_main_dca_contract_result)
            if (
                bus_group_main_dca_contract_result.status != "passed"
                and not continue_on_failure
            ):
                break
            bus_group_main_dca_intake_result = (
                validate_bus_group_main_dca_physical_readback_artifact_intake_snapshot()
            )
            results.append(bus_group_main_dca_intake_result)
            if (
                bus_group_main_dca_intake_result.status != "passed"
                and not continue_on_failure
            ):
                break
            matrix_state_contract_result = (
                validate_matrix_state_physical_readback_contract_snapshot()
            )
            results.append(matrix_state_contract_result)
            if (
                matrix_state_contract_result.status != "passed"
                and not continue_on_failure
            ):
                break
            matrix_state_intake_result = (
                validate_matrix_state_physical_readback_artifact_intake_snapshot()
            )
            results.append(matrix_state_intake_result)
            if (
                matrix_state_intake_result.status != "passed"
                and not continue_on_failure
            ):
                break
            fx_slots_inserts_contract_result = (
                validate_fx_slots_inserts_physical_readback_contract_snapshot()
            )
            results.append(fx_slots_inserts_contract_result)
            if (
                fx_slots_inserts_contract_result.status != "passed"
                and not continue_on_failure
            ):
                break
            fx_slots_inserts_intake_result = (
                validate_fx_slots_inserts_physical_readback_artifact_intake_snapshot()
            )
            results.append(fx_slots_inserts_intake_result)
            if (
                fx_slots_inserts_intake_result.status != "passed"
                and not continue_on_failure
            ):
                break
            routing_sends_contract_result = (
                validate_routing_sends_physical_readback_contract_snapshot()
            )
            results.append(routing_sends_contract_result)
            if (
                routing_sends_contract_result.status != "passed"
                and not continue_on_failure
            ):
                break
            routing_sends_intake_result = (
                validate_routing_sends_physical_readback_artifact_intake_snapshot()
            )
            results.append(routing_sends_intake_result)
            if (
                routing_sends_intake_result.status != "passed"
                and not continue_on_failure
            ):
                break
            snapshots_scenes_contract_result = (
                validate_snapshots_scenes_physical_readback_contract_snapshot()
            )
            results.append(snapshots_scenes_contract_result)
            if (
                snapshots_scenes_contract_result.status != "passed"
                and not continue_on_failure
            ):
                break
            snapshots_scenes_intake_result = (
                validate_snapshots_scenes_physical_readback_artifact_intake_snapshot()
            )
            results.append(snapshots_scenes_intake_result)
            if (
                snapshots_scenes_intake_result.status != "passed"
                and not continue_on_failure
            ):
                break
            user_presets_contract_result = (
                validate_user_presets_physical_readback_contract_snapshot()
            )
            results.append(user_presets_contract_result)
            if (
                user_presets_contract_result.status != "passed"
                and not continue_on_failure
            ):
                break
            user_presets_intake_result = (
                validate_user_presets_physical_readback_artifact_intake_snapshot()
            )
            results.append(user_presets_intake_result)
            if (
                user_presets_intake_result.status != "passed"
                and not continue_on_failure
            ):
                break
            dlive_state_contract_result = (
                validate_dlive_state_physical_readback_contract_snapshot()
            )
            results.append(dlive_state_contract_result)
            if (
                dlive_state_contract_result.status != "passed"
                and not continue_on_failure
            ):
                break
            dlive_state_intake_result = (
                validate_dlive_state_physical_readback_artifact_intake_snapshot()
            )
            results.append(dlive_state_intake_result)
            if (
                dlive_state_intake_result.status != "passed"
                and not continue_on_failure
            ):
                break
            artifact_review_queue_result = (
                validate_read_surface_physical_readback_artifact_review_queue_snapshot()
            )
            results.append(artifact_review_queue_result)
            if (
                artifact_review_queue_result.status != "passed"
                and not continue_on_failure
            ):
                break
            fixture_acceptance_ledger_result = (
                validate_read_surface_physical_readback_fixture_acceptance_ledger_snapshot()
            )
            results.append(fixture_acceptance_ledger_result)
            if (
                fixture_acceptance_ledger_result.status != "passed"
                and not continue_on_failure
            ):
                break
            fixture_closure_plan_result = (
                validate_read_surface_physical_readback_fixture_closure_plan_snapshot()
            )
            results.append(fixture_closure_plan_result)
            if (
                fixture_closure_plan_result.status != "passed"
                and not continue_on_failure
            ):
                break
        if check.name == "supervised_proof_artifact_report" and result.status == "passed":
            gain_dry_run_result = validate_supervised_proof_report_gain_dry_run(proof_report_path)
            results.append(gain_dry_run_result)
            if gain_dry_run_result.status != "passed" and not continue_on_failure:
                break
            autonomy_overlay_result = validate_supervised_proof_report_autonomy_overlay(proof_report_path)
            results.append(autonomy_overlay_result)
            if autonomy_overlay_result.status != "passed" and not continue_on_failure:
                break
            next_plan_result = validate_supervised_proof_report_next_wing_plan(proof_report_path)
            results.append(next_plan_result)
            if next_plan_result.status != "passed" and not continue_on_failure:
                break
            wing_runner_policy_result = (
                validate_wing_operator_proposal_runner_physical_readback_policy()
            )
            results.append(wing_runner_policy_result)
            if wing_runner_policy_result.status != "passed" and not continue_on_failure:
                break
            evaluator_policy_result = (
                validate_supervised_proof_evaluator_physical_readback_policy()
            )
            results.append(evaluator_policy_result)
            if evaluator_policy_result.status != "passed" and not continue_on_failure:
                break
            artifact_result = validate_supervised_proof_report_gap_rollup(proof_report_path)
            results.append(artifact_result)
            if artifact_result.status != "passed" and not continue_on_failure:
                break

    summary = write_summary(run_dir, results, generated_at=generated_at)
    summary["run_dir"] = str(run_dir)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run no-WING Automixer readiness checks without connecting to mixer hardware."
    )
    parser.add_argument(
        "--artifact-root",
        default=str(DEFAULT_ARTIFACT_ROOT),
        help="Directory where release-check artifacts are written.",
    )
    parser.add_argument(
        "--skip-frontend",
        action="store_true",
        help="Run backend/offline checks only.",
    )
    parser.add_argument(
        "--include-package",
        action="store_true",
        help="Also run the Electron/macOS package build via npm run dist.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use focused backend tests instead of the full backend suite.",
    )
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Continue running later checks after a failed check.",
    )
    parser.add_argument(
        "--backend-port",
        type=int,
        default=DEFAULT_BACKEND_PORT,
        help="Local backend port that should be free before checks start.",
    )
    parser.add_argument(
        "--backend-listener-confirmation",
        type=Path,
        default=None,
        help=(
            "Optional operator JSON confirming an existing listener is the "
            "intended backend for this WING read-only preflight."
        ),
    )
    parser.add_argument(
        "--command-timeout-sec",
        type=float,
        default=DEFAULT_COMMAND_TIMEOUT_SEC,
        help="Timeout in seconds for each subprocess check. Use <= 0 to disable.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_release_checks(
        artifact_root=Path(args.artifact_root),
        include_frontend=not args.skip_frontend,
        include_package=args.include_package and not args.skip_frontend,
        quick=args.quick,
        continue_on_failure=args.continue_on_failure,
        backend_port=args.backend_port,
        backend_listener_confirmation=args.backend_listener_confirmation,
        command_timeout_sec=args.command_timeout_sec if args.command_timeout_sec > 0 else None,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if summary.get("status") == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
