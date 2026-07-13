import asyncio
import atexit
import copy
import errno
import fcntl
import hashlib
import inspect
import ipaddress
import math
import json
import logging
import os
import pwd
import re
import select
import signal
import subprocess
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    TYPE_CHECKING,
    Union,
)
from urllib.parse import urlsplit
import websockets
import numpy as np

if TYPE_CHECKING:
    from websockets.asyncio.server import ServerConnection as WebSocketServerProtocol
else:
    WebSocketServerProtocol = Any

from wing_client import WingClient
from dlive_client import DLiveClient
from osc.enhanced_osc_client import EnhancedOSCClient
from mixing_station_client import MixingStationClient, discover_mixing_station
from audio_devices import get_audio_devices
from dante_routing_config import get_routing_as_dict, get_module_signal_info
try:
    from voice_control_v2 import VoiceControlV2
except ImportError as exc:
    VoiceControlV2 = None
    _VOICE_CONTROL_V2_IMPORT_ERROR = str(exc)
else:
    _VOICE_CONTROL_V2_IMPORT_ERROR = None
try:
    from voice_control_sherpa import VoiceControlSherpa
except ImportError as exc:
    VoiceControlSherpa = None
    _VOICE_CONTROL_SHERPA_IMPORT_ERROR = str(exc)
else:
    _VOICE_CONTROL_SHERPA_IMPORT_ERROR = None
from lufs_gain_staging import LUFSGainStagingController, SafeGainCalibrator
from channel_recognizer import scan_and_recognize, recognize_instrument_spectral_fallback, AVAILABLE_PRESETS
from auto_eq import AutoEQController, InstrumentProfiles, MultiChannelAutoEQController
from phase_alignment import PhaseAlignmentController
from system_measurement import SystemMeasurementController, TargetBus
from auto_fader import AutoFaderController
from auto_compressor import AutoCompressorController
from backup_channels import backup_channel
from restore_channels import restore_from_backup_using_client
from bleed_service import BleedService
from audio_capture import (
    AudioCapture, AudioSourceType, AudioDeviceType,
    detect_audio_device, find_device_by_name, list_audio_devices, HAS_SOUNDDEVICE,
    LIVE_CAPTURE_MAX_CHANNELS, validate_live_capture_dimensions,
)
from feedback_detector import FeedbackDetector
from auto_soundcheck_engine import AutoSoundcheckEngine
from observation_mixer import ObservationMixerClient
from mixer_discovery import discover_mixers, discover_mixer_auto, DiscoveredMixer
from handlers import register_all_handlers
from user_config_store import (
    get_user_config_path,
    load_user_config,
    normalize_mixer_user_settings,
    normalize_user_config,
    save_user_config,
)
from ws_transport import (
    EMERGENCY_FAST_LANE_TYPES,
    QueuedIngressMessage,
    RejectionWindow,
    TokenBucket,
    broadcast_json,
    consume_token_buckets,
    is_connection_closed_error,
    send_json,
    websocket_message_type,
)
from operator_mode_policy import (
    DEFAULT_OPERATOR_MODE,
    build_operator_mode_status,
    MANUAL_LIVE_WRITE_KIND_ALIASES,
    SUPPORTED_MANUAL_LIVE_WRITE_KINDS,
    WING_DEPLOYMENT_LIVE_WRITE_KINDS,
    normalize_operator_mode,
)
from operator_proposal_queue import OperatorProposalQueue
from operator_recommendation_bridge import (
    SAFE_GAIN_SOURCE,
    SOUNDCHECK_SOURCE,
    build_safe_gain_operator_proposals,
    build_soundcheck_operator_proposals,
    import_operator_proposals,
)
from soundcheck_mix_readiness import (
    aggregate_soundcheck_readiness,
    assess_soundcheck_manifest,
    build_soundcheck_replan_draft,
    classify_transition_reason,
    coupled_safe_alternative_plan_errors,
    plan_preflight_assessment,
    soundcheck_manifest_logical_id_errors,
)
from soundcheck_transaction_plan import (
    bind_soundcheck_manifest_content_sha256,
    soundcheck_manifest_content_sha256,
)
from processing_proposal_adapters import build_operator_correction_payload
from automation_trace import (
    TRACE_PROVENANCE_SCHEMA,
    AutomationDecisionTrace,
    AutomationTraceStore,
)
from auth import AuthManager
from decision_memory import DecisionMemoryStore
from supervised_console_parameter_registry import FAMILY_CONTRACTS
from supervised_console_transaction import (
    AudioRecheckResult,
    ConsoleTransactionContext,
    ConsoleTransactionOperation,
    ConsoleTransactionPlan,
    SupervisedConsoleTransactionCoordinator,
)
from wing_supervised_console_adapters import build_wing_supervised_family_adapters
from runtime_identity import (
    PACKAGED_RUNTIME_SOURCE_MANIFEST_FILENAME,
    RUNTIME_IDENTITY_SCHEMA,
    capture_source_runtime_components,
    consume_proof_launch_binding_environment,
    load_packaged_runtime_source_components,
    runtime_identity_manifest_sha256,
)

_REPO_ROOT = os.path.abspath(
    os.environ.get("AUTOMIXER_REPO_ROOT")
    or os.path.join(os.path.dirname(__file__), "..")
)

_RUNTIME_BINDING_ENV_NAMES = {
    "AUTOMIXER_M005_LAUNCH_NONCE",
    "AUTOMIXER_M005_PRESTART_PROVENANCE_SCHEMA",
    "AUTOMIXER_M005_PRESTART_PROVENANCE_SHA256",
    "AUTOMIXER_M005_PRESTART_PROVENANCE_SIZE",
}
_NON_DOTENV_RUNTIME_ENV_NAMES = _RUNTIME_BINDING_ENV_NAMES | {
    "AUTOMIXER_WS_AUTH_TOKEN",
    "AUTOMIXER_WS_AUTH_TOKEN_FD",
    "AUTOMIXER_WS_TOKEN_PROCESS_LIFETIME",
}
WS_AUTH_TOKEN_PIPE_TIMEOUT_SECONDS = 5.0


def _load_repo_dotenv() -> None:
    """Load optional repo-root `.env` into ``os.environ`` (does not override existing vars)."""
    # M005 evidence is bound to the proof runner's explicit allowlisted child
    # environment. An ignored local `.env` is not part of release provenance
    # and must not alter its operator mode, write rails, or evidence roots.
    if os.environ.get("AUTOMIXER_M005_LAUNCH_NONCE"):
        return
    path = os.path.join(_REPO_ROOT, ".env")
    if not os.path.isfile(path):
        return
    try:
        with open(path, encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, _, val = line.partition("=")
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key in _NON_DOTENV_RUNTIME_ENV_NAMES:
                    continue
                if key and key not in os.environ:
                    os.environ[key] = val
    except OSError as exc:
        logging.getLogger(__name__).warning("Could not read .env: %s", exc)


_load_repo_dotenv()


def _consume_configured_ws_auth_token() -> Optional[str]:
    """Consume a launch token from env or a one-shot inherited pipe."""

    environment_token = os.environ.pop("AUTOMIXER_WS_AUTH_TOKEN", None)
    fd_text = os.environ.pop("AUTOMIXER_WS_AUTH_TOKEN_FD", None)
    if environment_token and fd_text:
        raise RuntimeError(
            "Configure exactly one WebSocket token transport: environment or inherited fd"
        )
    if not fd_text:
        return environment_token
    if not re.fullmatch(r"[3-9]|[1-9][0-9]{1,2}", str(fd_text)):
        raise RuntimeError("AUTOMIXER_WS_AUTH_TOKEN_FD is invalid")
    fd = int(fd_text)
    chunks: list[bytes] = []
    total = 0
    deadline = time.monotonic() + WS_AUTH_TOKEN_PIPE_TIMEOUT_SECONDS
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("Timed out waiting for WebSocket launch token pipe")
            readable, _, _ = select.select([fd], [], [], remaining)
            if not readable:
                raise RuntimeError("Timed out waiting for WebSocket launch token pipe")
            chunk = os.read(fd, 128)
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > 128:
                raise RuntimeError("WebSocket launch token pipe payload is too large")
    except OSError as exc:
        raise RuntimeError("Could not read WebSocket launch token pipe") from exc
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
    try:
        return b"".join(chunks).decode("ascii")
    except UnicodeDecodeError as exc:
        raise RuntimeError("WebSocket launch token pipe is not ASCII") from exc

_PROCESS_PACKAGED_BUILD_BINDING = None
if getattr(sys, "frozen", False):
    _PROCESS_SOURCE_RUNTIME_COMPONENTS, _PROCESS_PACKAGED_BUILD_BINDING = (
        load_packaged_runtime_source_components(
            Path(getattr(sys, "_MEIPASS"))
            / PACKAGED_RUNTIME_SOURCE_MANIFEST_FILENAME
        )
    )
else:
    try:
        _PROCESS_SOURCE_RUNTIME_COMPONENTS = capture_source_runtime_components(_REPO_ROOT)
    except Exception as exc:
        _PROCESS_SOURCE_RUNTIME_COMPONENTS = {
            "source": {
                "schema": "automixer.backend_runtime_source.v1",
                "observation": "unavailable",
                "error": type(exc).__name__,
                "worktree_dirty": None,
            },
            "code": {
                "scope": "automixer.live_safety_critical.v1",
                "files": [],
                "files_sha256": None,
                "error": type(exc).__name__,
            },
        }

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

WS_MAX_MESSAGE_BYTES = 256 * 1024
WS_PROTOCOL_MAX_QUEUE = 16
WS_NORMAL_QUEUE_SIZE = 16
WS_PER_CONNECTION_RATE = 10.0
WS_PER_CONNECTION_BURST = 30.0
WS_GLOBAL_RATE = 50.0
WS_GLOBAL_BURST = 100.0
WS_REJECT_CLOSE_THRESHOLD = 20
WS_REJECT_WINDOW_SECONDS = 5.0
WS_AUTH_ATTEMPT_RATE = 5.0
WS_AUTH_ATTEMPT_BURST = 20.0
WS_DEFAULT_MAX_CLIENTS = 10
WS_MAX_CONFIGURED_CLIENTS = 100
WS_MAX_PENDING_AUTH = 2
WS_AUTHENTICATION_TIMEOUT_SECONDS = 1.0
SYSTEM_MEASUREMENT_MIN_DURATION_SEC = 3.0
SYSTEM_MEASUREMENT_MAX_DURATION_SEC = 20.0
SYSTEM_MEASUREMENT_CAPTURE_GRACE_SEC = 1.0
SYSTEM_MEASUREMENT_CAPTURE_POLL_SEC = 0.05

# A source backend without a configured bearer token is intentionally useful
# for local read-only diagnostics. It must never become a control plane merely
# because the listener is bound to loopback: any local process can open a
# native WebSocket without a browser Origin header. Keep this list small and
# deny every other message before handler dispatch.
UNAUTHENTICATED_READ_ONLY_MESSAGE_TYPES = frozenset(
    {
        "connect_wing",
        "emergency_stop",
        "emergency_stop_agent",
        "get_action_history",
        "get_agent_communications_snapshot",
        "get_agent_status",
        "get_arrangement_automation_status",
        "get_audio_capture_status",
        "get_audio_devices",
        "get_auto_compressor_status",
        "get_auto_effects_status",
        "get_auto_engine_status",
        "get_auto_eq_status",
        "get_auto_fader_status",
        "get_auto_gate_status",
        "get_auto_panner_status",
        "get_auto_reverb_status",
        "get_auto_soundcheck_status",
        "get_bus_main_dca_snapshot",
        "get_channel_input_routing",
        "get_channel_intelligence",
        "get_channel_inventory",
        "get_channel_meters",
        "get_channel_processing_snapshot",
        "get_connection_topology",
        "get_cross_adaptive_eq_status",
        "get_dante_routing",
        "get_dashboard_snapshot",
        "get_decision_queue",
        "get_decision_traces",
        "get_dlive_state_snapshot",
        "get_eq_profiles",
        "get_freeze_status",
        "get_fx_inserts_snapshot",
        "get_fx_overview",
        "get_gain_staging_recommendations",
        "get_gain_staging_status",
        "get_groups_buses_snapshot",
        "get_insert_state",
        "get_live_input_trim_recommendations",
        "get_main_pipeline_preview",
        "get_manual_write_supervision",
        "get_matrix_snapshot",
        "get_memory_knowledge_snapshot",
        "get_operator_agent_roster",
        "get_operator_mode",
        "get_output_routing",
        "get_pending_actions",
        "get_phase_alignment_status",
        "get_processing_pipeline_snapshot",
        "query_mixer_parameter",
        "get_ready_for_live",
        "get_replay_runs",
        "get_routing_sends_snapshot",
        "get_rules_library_snapshot",
        "get_runtime_identity",
        "get_safety_monitor",
        "get_snapshots_scenes_snapshot",
        "get_state",
        "get_system_measurement_status",
        "get_telemetry_alerts_snapshot",
        "get_training_sessions_snapshot",
        "get_training_status",
        "get_user_presets_snapshot",
        "get_voice_control_status",
        "rescan_channel_names",
        "scan_audio_devices",
        "scan_channel_names",
        "scan_mixer_channel_names",
        "scan_mixers",
        "select_audio_device",
        "start_audio_capture_preview",
        "stop_audio_capture_preview",
        "subscribe_meters",
        "unsubscribe_meters",
    }
)

MANUAL_WRITE_WAL_TERMINAL_PHASES = frozenset(
    {
        "aborted_before_send",
        "committed",
        "reconciled_committed",
        "reconciled_no_change",
        "reconciled_rolled_back",
        "rolled_back",
    }
)

QUARANTINED_WRITE_SURFACE_OPERATOR_METHODS = {
    "bypass_mixer": "scene_preset_management_proposal",
    "start_realtime_correction": "lufs_gain_staging",
    "reset_trim": "lufs_gain_staging",
    "start_realtime_fader": "auto_fader_v1",
    "apply_auto_balance": "auto_fader_v1",
    "set_eq": "auto_eq",
    "apply_eq_correction": "auto_eq",
    "reset_eq": "auto_eq",
    "reset_all_eq": "auto_eq",
    "apply_channel_correction": "auto_eq",
    "apply_all_corrections": "auto_eq",
    "set_compressor": "auto_compressor",
    "start_auto_compressor": "auto_compressor",
    "start_auto_compressor_soundcheck": "auto_compressor",
    "start_auto_compressor_live": "auto_compressor",
    "set_auto_compressor_profile": "auto_compressor",
    "set_auto_compressor_manual": "auto_compressor",
    "set_pan": "auto_panner",
    "apply_phase_corrections": "phase_alignment",
    "reset_phase_delay": "phase_alignment",
    "set_insert": "fx_direct_apply",
    "set_fx_model": "fx_direct_apply",
    "set_fx_on": "fx_direct_apply",
    "set_fx_mix": "fx_direct_apply",
    "set_fx_parameter": "fx_direct_apply",
    "route_output": "routing_management_proposal",
    "route_multiple_outputs": "routing_management_proposal",
    "set_channel_input": "routing_management_proposal",
    "set_channel_alt_input": "routing_management_proposal",
    "load_snap": "scene_preset_management_proposal",
    "save_snap": "scene_preset_management_proposal",
    "restore_snapshot": "scene_preset_management_proposal",
    "set_mute": "mute_management_proposal",
    "mute_channel": "mute_management_proposal",
    "unmute_channel": "mute_management_proposal",
    "apply_system_measurement": "master_bus_optimization_proposal",
    "reset_system_measurement": "master_bus_optimization_proposal",
    "voice_command:set_fader": "auto_fader_v1",
    "voice_command:set_gain": "lufs_gain_staging",
    "voice_command:set_pan": "auto_panner",
    "voice_command:load_snap": "scene_preset_management_proposal",
    "voice_command:mute_channel": "mute_management_proposal",
    "voice_command:volume_up": "auto_fader_v1",
    "voice_command:volume_down": "auto_fader_v1",
    "voice_command:eq_on": "auto_eq",
    "voice_command:eq_band_up": "auto_eq",
    "voice_command:eq_band_down": "auto_eq",
    "voice_command:compressor_on": "auto_compressor",
    "voice_command:compressor_threshold": "auto_compressor",
    "voice_command:compressor_gain_up": "auto_compressor",
    "voice_command:compressor_gain_down": "auto_compressor",
}

MANUAL_WRITE_KIND_CONFIG = {
    "fader": {
        "address": "/ch/{channel}/fdr",
        "method": "set_channel_fader",
        "readback_tolerance": 0.05,
        "clamp": lambda value: max(-144.0, min(10.0, float(value))),
    },
    "gain": {
        "address": "/ch/{channel}/in/set/trim",
        "method": "set_channel_gain",
        "readback_tolerance": 0.06,
        "clamp": lambda value: max(-18.0, min(18.0, float(value))),
    },
    "mute": {
        "address": "/ch/{channel}/mute",
        "method": "set_channel_mute",
        "readback_tolerance": 0.0,
        "clamp": lambda value: 1.0 if bool(value) else 0.0,
    },
    "pan": {
        "address": "/ch/{channel}/pan",
        "method": "set_channel_pan",
        "readback_tolerance": 0.5,
        "clamp": lambda value: max(-100.0, min(100.0, float(value))),
    },
}

READONLY_PARAMETER_KIND_ALIASES = {
    **MANUAL_LIVE_WRITE_KIND_ALIASES,
    "eq_enabled": "eq_on",
    "eq_wet": "eq_mix",
    "dynamics_on": "dyn_on",
    "compressor_on": "dyn_on",
    "dynamics_threshold": "dyn_threshold",
    "compressor_threshold": "dyn_threshold",
    "gate_thr": "gate_threshold",
    "gate_threshold_db": "gate_threshold",
    "gate_mdl": "gate_model",
    "gate_att": "gate_attack",
    "gate_rel": "gate_release",
    "gate_filt": "gate_filter",
    "gate_freq": "gate_frequency",
}

READONLY_PARAMETER_KIND_CONFIG = {
    **MANUAL_WRITE_KIND_CONFIG,
    "eq_on": {"address": "/ch/{channel}/eq/on"},
    "eq_mix": {"address": "/ch/{channel}/eq/mix"},
    "eq_low_gain": {"address": "/ch/{channel}/eq/lg"},
    "eq_high_gain": {"address": "/ch/{channel}/eq/hg"},
    "dyn_on": {"address": "/ch/{channel}/dyn/on"},
    "dyn_threshold": {"address": "/ch/{channel}/dyn/thr"},
    "gate_on": {"address": "/ch/{channel}/gate/on"},
    "gate_threshold": {"address": "/ch/{channel}/gate/thr"},
    "gate_model": {"address": "/ch/{channel}/gate/mdl"},
    "gate_ratio": {"address": "/ch/{channel}/gate/ratio"},
    "gate_attack": {"address": "/ch/{channel}/gate/att"},
    "gate_release": {"address": "/ch/{channel}/gate/rel"},
    "gate_filter": {"address": "/ch/{channel}/gate/filt"},
    "gate_gain": {"address": "/ch/{channel}/gate/g"},
    "gate_frequency": {"address": "/ch/{channel}/gate/f"},
    "gate_q": {"address": "/ch/{channel}/gate/q"},
    "gate_mode": {"address": "/ch/{channel}/gate/mode"},
}

READONLY_ENUM_PARAMETER_KINDS = frozenset(
    {
        "gate_model",
        "gate_ratio",
        "gate_filter",
        "gate_mode",
    }
)

SUPPORTED_MANUAL_WRITE_KINDS_SET = frozenset(SUPPORTED_MANUAL_LIVE_WRITE_KINDS)


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def convert_numpy_types(obj: Any) -> Any:
    """
    Рекурсивно преобразует NumPy типы в нативные Python типы для JSON сериализации.

    Args:
        obj: Объект, который может содержать NumPy типы

    Returns:
        Объект с преобразованными типами
    """
    # Проверяем на NumPy скалярные типы
    if isinstance(obj, np.generic):
        if isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                           np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            # Для других NumPy типов пытаемся преобразовать в Python тип
            try:
                return obj.item()
            except (AttributeError, ValueError):
                return float(obj) if np.issubdtype(type(obj), np.floating) else int(obj) if np.issubdtype(type(obj), np.integer) else bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

class AutoMixerServer:
    """
    WebSocket server that bridges frontend UI with mixer control

    Supports two connection modes:
    1. Direct Wing OSC - connects directly to Wing mixer
    2. Mixing Station - connects via Mixing Station app (supports multiple mixers)
    """

    def __init__(self, ws_host: str = "localhost", ws_port: int = 8765):
        self.ws_host = ws_host
        self.ws_port = ws_port
        self._ws_allowed_origins = self._build_websocket_allowed_origins()
        # Read and remove the one-time bearer token before constructing runtime
        # identity. Packaged identity binds to its digest, never the plaintext.
        configured_token_from_fd = bool(os.environ.get("AUTOMIXER_WS_AUTH_TOKEN_FD"))
        configured_token = _consume_configured_ws_auth_token()
        runtime_backend_path = Path(
            sys.executable if getattr(sys, "frozen", False) else __file__
        ).resolve()
        runtime_backend_bytes = runtime_backend_path.read_bytes()
        try:
            runtime_backend_display_path = str(
                runtime_backend_path.relative_to(Path(_REPO_ROOT).resolve())
            )
        except ValueError:
            runtime_backend_display_path = str(runtime_backend_path)
        process_started_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        )
        process_instance_nonce = uuid.uuid4().hex
        proof_launch_binding = consume_proof_launch_binding_environment()
        if getattr(sys, "frozen", False):
            if proof_launch_binding["mode"] != "unbound":
                raise RuntimeError(
                    "Packaged backend cannot claim a source M005 proof-runner binding"
                )
            if not isinstance(_PROCESS_PACKAGED_BUILD_BINDING, dict):
                raise RuntimeError("Packaged backend build binding is unavailable")
            launch_binding = json.loads(json.dumps(_PROCESS_PACKAGED_BUILD_BINDING))
            if not re.fullmatch(r"[A-Za-z0-9_-]{43}", str(configured_token or "")):
                raise RuntimeError(
                    "Packaged backend requires an authenticated launcher token"
                )
            launch_binding.update(
                {
                    "authenticated_launcher_bound": True,
                    "launch_token_sha256": hashlib.sha256(
                        str(configured_token).encode("utf-8")
                    ).hexdigest(),
                }
            )
        else:
            launch_binding = proof_launch_binding
        runtime_source = json.loads(json.dumps(_PROCESS_SOURCE_RUNTIME_COMPONENTS))
        self._runtime_identity = {
            "type": "runtime_identity",
            "schema": RUNTIME_IDENTITY_SCHEMA,
            "runtime_kind": "packaged" if getattr(sys, "frozen", False) else "source",
            "instance_nonce": process_instance_nonce,
            "started_at_utc": process_started_at,
            "process": {
                "pid": os.getpid(),
                "instance_nonce": process_instance_nonce,
                "started_at_utc": process_started_at,
            },
            "source": runtime_source["source"],
            "code": runtime_source["code"],
            "backend": {
                "path": runtime_backend_display_path,
                "exists": True,
                "size": len(runtime_backend_bytes),
                "sha256": hashlib.sha256(runtime_backend_bytes).hexdigest(),
            },
            "launch_binding": launch_binding,
            "websocket": {
                "host": "127.0.0.1" if self._is_loopback_host(ws_host) else str(ws_host),
                "port": int(ws_port),
                "loopback_only": True,
            },
            "read_only_identity": True,
            "live_authority_granted": False,
        }
        self._runtime_identity["manifest_sha256"] = runtime_identity_manifest_sha256(
            self._runtime_identity
        )

        # A configured bearer token activates authentication even on loopback.
        # Packaged Electron uses this to isolate its private backend from other
        # local processes and browser pages. Pop the plaintext immediately after
        # reading it; AuthManager retains only an HMAC digest.
        if not self._is_loopback_host(ws_host):
            configured_token = None
            raise RuntimeError(
                "Non-loopback WebSocket binds are disabled; terminate HTTPS/WSS "
                "at a trusted reverse proxy and forward to 127.0.0.1"
            )
        self._ws_auth_required = bool(configured_token)
        self._ws_token_process_lifetime = self._parse_env_bool(
            os.environ.pop("AUTOMIXER_WS_TOKEN_PROCESS_LIFETIME", "false")
        )
        if self._ws_token_process_lifetime and (
            not configured_token
            or not configured_token_from_fd
            or not self._parse_env_bool(os.environ.get("AUTOMIXER_PACKAGED_APP", "false"))
        ):
            raise RuntimeError(
                "Process-lifetime WebSocket tokens require the packaged Electron FD launcher"
            )
        self._network_runtime_started = False
        self._ws_expected_client_id = os.environ.get(
            "AUTOMIXER_WS_CLIENT_ID", "automixer-ui"
        )
        self._authenticated_websocket_expiry: Dict[int, float] = {}
        self._authenticated_websocket_expiry_close_started: set[int] = set()
        self.auth_manager = AuthManager(enabled=self._ws_auth_required)
        if self._ws_auth_required:
            if not re.fullmatch(r"[A-Za-z0-9_-]{43}", str(configured_token or "")):
                raise RuntimeError(
                    "AUTOMIXER_WS_AUTH_TOKEN must be exactly 32 bytes encoded as "
                    "43-character base64url"
                )
            if not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", self._ws_expected_client_id):
                raise RuntimeError("AUTOMIXER_WS_CLIENT_ID is invalid")
            token_ttl_hours = (
                None
                if self._ws_token_process_lifetime
                else self._validated_ws_token_ttl_hours(
                    os.environ.get("AUTOMIXER_WS_TOKEN_TTL_HOURS", "24")
                )
            )
            self.auth_manager.register_token(
                configured_token,
                client_id=self._ws_expected_client_id,
                expires_hours=token_ttl_hours,
            )
            configured_token = None

        # Load configuration
        self.config = self._load_config()
        self._ws_max_clients = self._configured_ws_max_clients(self.config)
        self._ws_pending_admissions: set[int] = set()
        self._ws_global_ingress_bucket = TokenBucket(
            WS_GLOBAL_RATE,
            WS_GLOBAL_BURST,
        )
        self._ws_auth_attempt_bucket = TokenBucket(
            WS_AUTH_ATTEMPT_RATE,
            WS_AUTH_ATTEMPT_BURST,
        )
        # This ingress epoch advances synchronously when a priority stop frame
        # is accepted. Normal frames from the prior epoch are then stale even
        # while the asynchronous stop routine is still freezing controllers.
        self._ws_ingress_emergency_generation = 0
        self._ws_emergency_fast_lane_task: Optional[asyncio.Task[Any]] = None
        # Release/runtime smoke tests need a hard guarantee that no mixer
        # transport can be opened. Unlike observation mode, this process-level
        # lock cannot be weakened by a WebSocket request payload.
        self._packaged_offline_lock = self._parse_env_bool(
            os.getenv("AUTOMIXER_PACKAGED_OFFLINE_LOCK", "false")
        )

        # Unified mixer client (can be EnhancedOSCClient wrapping WingClient, DLiveClient, or MixingStationClient)
        self.mixer_client: Optional[Union[EnhancedOSCClient, WingClient, DLiveClient, MixingStationClient]] = None
        self.connection_mode: Optional[str] = None  # 'wing', 'dlive', or 'mixing_station'

        self.connected_clients: Set[WebSocketServerProtocol] = set()

        # Voice control
        # Voice runtimes are optional in the packaged backend. Missing native
        # audio/STT dependencies must disable only this feature, never prevent
        # the safety/control server from starting.
        self.voice_control: Optional[Any] = None
        self._voice_command_tasks: set[asyncio.Task[Any]] = set()

        # Gain staging controller (LUFS-based)
        self.gain_staging: Optional[LUFSGainStagingController] = None
        # Safe Gain Calibrator (новый метод: анализ → одноразовое применение)
        self.safe_gain_calibrator: Optional[SafeGainCalibrator] = None
        self._safe_gain_audio_thread_stop: Optional[threading.Event] = None
        self._safe_gain_audio_thread: Optional[threading.Thread] = None
        self.live_input_trim_controller = None
        # The arrangement automation prototype is intentionally not activated
        # by the product runtime. Its WebSocket surface returns explicit
        # fail-closed status until it is migrated to the supervised executor.
        self.arrangement_automation_controller = None

        # Auto-EQ controller
        self.auto_eq_controller: Optional[AutoEQController] = None
        self.multi_channel_auto_eq_controller: Optional[MultiChannelAutoEQController] = None

        # Phase alignment controller
        self.phase_alignment_controller: Optional[PhaseAlignmentController] = None

        # Master/system measurement controller
        self.system_measurement_controller: Optional[SystemMeasurementController] = None

        # Auto Fader controller
        self.auto_fader_controller: Optional[AutoFaderController] = None

        # Track last known fader values for relative changes
        self.last_fader_values: dict[int, float] = {}
        self.last_mute_values: dict[int, bool] = {}
        self.last_pan_values: dict[int, float] = {}

        # Auto Soundcheck state
        self.auto_soundcheck_running = False
        self.auto_soundcheck_observe_only = False
        self.auto_soundcheck_task: Optional[asyncio.Task] = None
        self.auto_soundcheck_websocket: Optional[WebSocketServerProtocol] = None
        self._observation_mixer_client: Optional[ObservationMixerClient] = None

        # Auto Compressor controller
        self.auto_compressor_controller: Optional[AutoCompressorController] = None

        # Centralized bleed detection service
        self.bleed_service = BleedService(self.config)

        # Unified audio capture service
        self.audio_capture: Optional[AudioCapture] = None
        self._audio_capture_preview_owner: Optional[WebSocketServerProtocol] = None
        self._audio_capture_lock = asyncio.Lock()
        self._system_measurement_task: Optional[asyncio.Task[Any]] = None
        self._controllers_cleanup_complete = False

        # Feedback detector
        self.feedback_detector: Optional[FeedbackDetector] = None

        # Auto soundcheck engine (headless auto-mixing)
        self.auto_soundcheck_engine: Optional[AutoSoundcheckEngine] = None
        self.last_soundcheck_readiness: Dict[str, Any] = {}
        self._soundcheck_manifests: Dict[str, Dict[str, Any]] = {}
        self._soundcheck_outcomes: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._soundcheck_global_blockers: Dict[str, List[str]] = {}
        self._soundcheck_preflight_assessments: Dict[str, Dict[str, Any]] = {}
        self._soundcheck_preflight_timestamps: Dict[str, float] = {}
        self._soundcheck_replan_drafts: Dict[str, Dict[str, Any]] = {}
        self._active_soundcheck_run_id = ""

        # AI Mixing Agent (rules + optional LLM, suggest-first by default)
        self.mixing_agent = None
        self.mixing_agent_task: Optional[asyncio.Task] = None
        self.agent_auto_apply_enabled = False
        self._agent_force_auto_apply = False

        # Online ML training service for internet-fed datasets
        self.agent_training_service = None
        if not self._packaged_offline_lock:
            self._init_training_service()
        else:
            logger.info("Packaged offline lock active: training service disabled")

        # Live concert mode: stricter limits, emergency stop
        self.live_mode = False
        self.operator_mode = normalize_operator_mode(
            os.getenv("AUTOMIXER_OPERATOR_MODE", DEFAULT_OPERATOR_MODE)
        )
        self._auto_mode_unlocked = self._parse_env_bool(
            os.getenv("AUTOMIXER_AUTO_MODE_UNLOCKED", "false")
        )
        self.operator_proposal_queue = OperatorProposalQueue()

        # Default safety mode for Wing test/observation runs
        self._wing_observation_only = self._parse_env_bool(
            os.getenv("AUTOMIXER_WING_OBSERVATION_MODE", "false")
        )
        self._wing_record_telemetry = self._parse_env_bool(
            os.getenv("AUTOMIXER_WING_RECORD_TELEMETRY", "false")
        )
        if self._packaged_offline_lock:
            self.operator_mode = normalize_operator_mode("analyze")
            self._auto_mode_unlocked = False
            self._wing_observation_only = True
            self._wing_record_telemetry = False
            self.auto_soundcheck_observe_only = True
        self._wing_telemetry_dir = os.getenv(
            "AUTOMIXER_WING_TELEMETRY_DIR",
            "artifacts/wing_telemetry",
        )
        shadow_reports_root = Path(
            os.getenv("AUTOMIXER_SHADOW_REPORTS_DIR", "artifacts")
        )
        if not shadow_reports_root.is_absolute():
            shadow_reports_root = Path(__file__).resolve().parents[1] / shadow_reports_root
        self._shadow_reports_root = shadow_reports_root
        replay_readiness_root = Path(
            os.getenv("AUTOMIXER_WING_REPLAY_READINESS_DIR", "artifacts")
        )
        if not replay_readiness_root.is_absolute():
            replay_readiness_root = Path(__file__).resolve().parents[1] / replay_readiness_root
        self._wing_replay_readiness_root = replay_readiness_root
        replay_artifacts_dir = Path(os.getenv("AUTOMIXER_REPLAY_ARTIFACTS_DIR", "artifacts/replay"))
        if not replay_artifacts_dir.is_absolute():
            replay_artifacts_dir = Path(__file__).resolve().parents[1] / replay_artifacts_dir
        self._replay_artifacts_dir = replay_artifacts_dir
        self._wing_deployment_write_boundary_enabled = self._parse_env_bool(
            os.getenv("AUTOMIXER_WING_DEPLOYMENT_WRITE_BOUNDARY_ENABLED", "true")
        )
        self._manual_write_emergency_stopped = False
        self._manual_write_emergency_generation = 0
        self._system_emergency_stop_in_progress = False
        self._system_emergency_stop_errors: list[dict[str, str]] = []
        self._system_emergency_teardown_complete = False
        self._latest_master_true_peak_dbtp: Optional[float] = None
        self._latest_master_true_peak_timestamp = 0.0
        self._master_capture_route_verified = self._parse_env_bool(
            os.getenv("AUTOMIXER_MASTER_CAPTURE_ROUTE_VERIFIED", "false")
        )
        self._master_capture_route_proof_id = str(
            os.getenv("AUTOMIXER_MASTER_CAPTURE_ROUTE_PROOF_ID", "")
        ).strip()
        self._master_capture_route_proof_path = str(
            os.getenv("AUTOMIXER_MASTER_CAPTURE_ROUTE_PROOF_PATH", "")
        ).strip()
        self._master_capture_route_proof_sha256 = str(
            os.getenv("AUTOMIXER_MASTER_CAPTURE_ROUTE_PROOF_SHA256", "")
        ).strip().lower()
        self._manual_write_cooldown_sec = max(
            0.0,
            self._parse_env_float(os.getenv("AUTOMIXER_MANUAL_WRITE_COOLDOWN_SEC"), 2.0),
        )
        self._manual_write_last_write_ts: Dict[tuple[int, str], float] = {}
        self._last_manual_write_result: Optional[Dict[str, Any]] = None
        manual_write_artifacts_dir = Path(
            os.getenv(
                "AUTOMIXER_MANUAL_WRITE_ARTIFACTS_DIR",
                "artifacts/wing_telemetry/manual_write_gate",
            )
        )
        if not manual_write_artifacts_dir.is_absolute():
            manual_write_artifacts_dir = Path(__file__).resolve().parents[1] / manual_write_artifacts_dir
        self._manual_write_artifacts_dir = manual_write_artifacts_dir
        # The physical-console mutex and recovery journal must share a stable
        # per-user scope. Configurable evidence roots must never split crash
        # recovery between two processes controlling the same console.
        self._console_runtime_dir = self._resolve_console_runtime_dir()
        self._console_writer_lock_path = self._console_runtime_dir / "console_writer.lock"
        self._console_write_wal_path = self._console_runtime_dir / "manual_write_wal.jsonl"
        self._soundcheck_parent_journal_path = (
            self._console_runtime_dir / "soundcheck_parent_journal.jsonl"
        )
        self._operator_apply_idempotency_keys: Dict[str, Dict[str, Any]] = {}
        self._durable_operator_apply_idempotency_keys: Dict[str, Dict[str, Any]] = {}
        self._durable_operator_proposal_ids: set[str] = set()
        self._supervised_transaction_idempotency_keys: Dict[str, str] = {}
        self._supervised_transaction_ids: Dict[str, str] = {}
        self._manual_write_prepared_transactions: Dict[str, Dict[str, Any]] = {}
        self._manual_write_recovery_transactions: Dict[str, Dict[str, Any]] = {}
        self._recover_manual_write_wal_state()
        self._soundcheck_parent_journal_error: Optional[str] = None
        self._recover_soundcheck_parent_journal_state()
        self.automation_trace_store = AutomationTraceStore(manual_write_artifacts_dir)
        self.decision_memory_store = DecisionMemoryStore(manual_write_artifacts_dir)
        self._console_write_lock = asyncio.Lock()
        self._console_write_owner_task: Optional[asyncio.Task[Any]] = None
        self._console_write_active_idempotency_keys: set[str] = set()

        # Snapshot for undo (path to last backup file)
        self._last_snapshot_path: Optional[str] = None

        # Shutdown flag (event created in start() to be in correct loop)
        self._shutdown_event: Optional[asyncio.Event] = None
        self._is_shutting_down = False
        self._shutdown_errors: list[dict[str, str]] = []

        # Build message-type → handler dispatch table (B15 decomposition)
        self._dispatch = register_all_handlers(self)

        logger.info(f"AutoMixer Server initialized on {ws_host}:{ws_port}")

    def get_runtime_identity_status(
        self,
        *,
        request_challenge_nonce: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return immutable process-start identity for attended proof binding."""
        identity = json.loads(json.dumps(self._runtime_identity))
        if request_challenge_nonce is not None:
            identity["request_challenge_nonce"] = str(request_challenge_nonce)
        return identity

    def _safe_cleanup_call(self, action, context: str, success_message: Optional[str] = None) -> bool:
        """Run cleanup logic without letting secondary stop errors break shutdown paths."""
        try:
            action()
            if success_message:
                logger.info(success_message)
            return True
        except Exception as exc:
            logger.warning("%s: %s", context, exc, exc_info=True)
            return False

    @staticmethod
    def _parse_env_bool(value: Any) -> bool:
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _is_loopback_host(host: str) -> bool:
        normalized = str(host or "").strip().lower()
        if normalized == "localhost":
            return True
        try:
            return ipaddress.ip_address(normalized).is_loopback
        except ValueError:
            return False

    @staticmethod
    def _validated_ws_token_ttl_hours(value: Any) -> float:
        try:
            ttl_hours = float(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "AUTOMIXER_WS_TOKEN_TTL_HOURS must be a finite value from 1 minute to 24 hours"
            ) from exc
        if not math.isfinite(ttl_hours) or not (1 / 60) <= ttl_hours <= 24:
            raise RuntimeError(
                "AUTOMIXER_WS_TOKEN_TTL_HOURS must be a finite value from 1 minute to 24 hours"
            )
        return ttl_hours

    @staticmethod
    def _configured_ws_max_clients(config: Mapping[str, Any]) -> int:
        websocket_config = config.get("websocket", {}) if isinstance(config, Mapping) else {}
        raw_value = (
            websocket_config.get("max_clients", WS_DEFAULT_MAX_CLIENTS)
            if isinstance(websocket_config, Mapping)
            else WS_DEFAULT_MAX_CLIENTS
        )
        try:
            if isinstance(raw_value, bool):
                raise ValueError
            parsed = int(raw_value)
            if isinstance(raw_value, float) and not raw_value.is_integer():
                raise ValueError
        except (TypeError, ValueError, OverflowError):
            logger.warning(
                "Invalid websocket.max_clients=%r; using fail-safe default %s",
                raw_value,
                WS_DEFAULT_MAX_CLIENTS,
            )
            return WS_DEFAULT_MAX_CLIENTS
        if not 1 <= parsed <= WS_MAX_CONFIGURED_CLIENTS:
            logger.warning(
                "websocket.max_clients=%r outside 1..%s; using fail-safe default %s",
                raw_value,
                WS_MAX_CONFIGURED_CLIENTS,
                WS_DEFAULT_MAX_CLIENTS,
            )
            return WS_DEFAULT_MAX_CLIENTS
        return parsed

    def _reserve_websocket_admission(self, websocket: WebSocketServerProtocol) -> bool:
        """Reserve one active-or-authenticating client slot without awaiting."""

        if self._is_shutting_down:
            return False
        connection_id = id(websocket)
        if connection_id in self._ws_pending_admissions:
            return True
        occupied = len(self.connected_clients) + len(self._ws_pending_admissions)
        pending_limit = min(WS_MAX_PENDING_AUTH, self._ws_max_clients)
        if (
            occupied >= self._ws_max_clients
            or len(self._ws_pending_admissions) >= pending_limit
        ):
            return False
        self._ws_pending_admissions.add(connection_id)
        return True

    def _release_websocket_admission(self, websocket: WebSocketServerProtocol) -> None:
        self._ws_pending_admissions.discard(id(websocket))

    async def _close_overloaded_websocket(
        self,
        websocket: WebSocketServerProtocol,
        *,
        reason: str,
    ) -> None:
        try:
            await websocket.close(code=1013, reason=reason)
        except Exception as exc:
            if not self._is_connection_closed_error(exc):
                logger.debug("Could not close overloaded WebSocket cleanly: %s", exc)

    async def _run_websocket_normal_lane(
        self,
        websocket: WebSocketServerProtocol,
        queue: asyncio.Queue[QueuedIngressMessage],
    ) -> None:
        """Run one FIFO worker so normal message handlers retain wire order."""

        while True:
            item = await queue.get()
            try:
                if self._is_shutting_down:
                    logger.warning("Dropped queued WebSocket message during shutdown")
                    continue
                if item.emergency_generation != self._ws_ingress_emergency_generation:
                    logger.warning(
                        "Dropped stale queued WebSocket message after emergency stop"
                    )
                    continue
                expires_at = self._authenticated_websocket_expiry.get(id(websocket))
                if expires_at is not None and time.time() >= expires_at:
                    logger.warning("Dropped queued WebSocket message after authentication expiry")
                    await self._close_expired_authenticated_websocket(websocket)
                    continue
                # A transport disconnect must stop queued work, but it must not
                # inject cancellation into a transaction already between
                # prepare/readback/apply/rollback stages. Shield the active
                # handler and drain it before the worker itself exits.
                active_handler = asyncio.create_task(
                    self.handle_client_message(websocket, item.payload)
                )
                try:
                    await asyncio.shield(active_handler)
                except asyncio.CancelledError:
                    with suppress(asyncio.CancelledError):
                        await active_handler
                    raise
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("Normal WebSocket ingress worker failed: %s", exc, exc_info=True)
            finally:
                queue.task_done()

    def _finish_websocket_emergency_task(self, task: asyncio.Task[Any]) -> None:
        if self._ws_emergency_fast_lane_task is task:
            self._ws_emergency_fast_lane_task = None
        try:
            task.result()
        except asyncio.CancelledError:
            logger.warning("Emergency WebSocket fast-lane task was cancelled")
        except Exception as exc:
            logger.error("Emergency WebSocket fast-lane task failed: %s", exc, exc_info=True)

    def _schedule_websocket_emergency(
        self,
        websocket: WebSocketServerProtocol,
        message: str | bytes,
    ) -> bool:
        """Schedule the first stop immediately and coalesce duplicate stop frames."""

        active_task = self._ws_emergency_fast_lane_task
        if (
            self._manual_write_emergency_stopped
            and not self._system_emergency_stop_errors
        ) or (
            active_task is not None and not active_task.done()
        ):
            logger.debug("Coalesced duplicate WebSocket emergency-stop frame")
            return False

        self._ws_ingress_emergency_generation += 1
        task = asyncio.create_task(self.handle_client_message(websocket, message))
        self._ws_emergency_fast_lane_task = task
        task.add_done_callback(self._finish_websocket_emergency_task)
        return True

    async def _record_websocket_ingress_rejection(
        self,
        websocket: WebSocketServerProtocol,
        rejection_window: RejectionWindow,
        *,
        reason: str,
        now: float,
    ) -> bool:
        if not rejection_window.record(now):
            logger.debug("Rejected WebSocket ingress frame: %s", reason)
            return False
        logger.warning(
            "Closing WebSocket after %s ingress rejects within %.1fs",
            rejection_window.count,
            WS_REJECT_WINDOW_SECONDS,
        )
        await self._close_overloaded_websocket(
            websocket,
            reason="ingress_overload",
        )
        return True

    @staticmethod
    def _resolve_console_runtime_dir() -> Path:
        test_override = os.environ.get("AUTOMIXER_TEST_CONSOLE_RUNTIME_DIR")
        if test_override:
            if not os.environ.get("PYTEST_CURRENT_TEST"):
                raise RuntimeError(
                    "AUTOMIXER_TEST_CONSOLE_RUNTIME_DIR is allowed only under pytest"
                )
            return Path(test_override).expanduser().resolve()
        return Path(pwd.getpwuid(os.getuid()).pw_dir) / ".automixer" / "runtime"

    @staticmethod
    def _build_websocket_allowed_origins() -> tuple[Optional[str], ...]:
        """Allow native clients and only explicitly trusted browser origins.

        A loopback bind isn't sufficient browser isolation: any web page can
        otherwise open a WebSocket to localhost. Native proof runners don't
        send an Origin header, represented by ``None`` in websockets.
        """

        origins: list[Optional[str]] = [
            None,
            "http://127.0.0.1:3000",
            "http://localhost:3000",
        ]
        configured = os.environ.get("AUTOMIXER_WS_ALLOWED_ORIGINS", "")
        for raw_origin in configured.split(","):
            origin = raw_origin.strip()
            if not origin:
                continue
            try:
                parsed = urlsplit(origin)
                parsed_port = parsed.port
            except ValueError as exc:
                raise RuntimeError(
                    "AUTOMIXER_WS_ALLOWED_ORIGINS entries must be exact http(s) origins"
                ) from exc
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or parsed.path
                or parsed.query
                or parsed.fragment
                or parsed.username is not None
                or parsed.password is not None
                or "*" in parsed.hostname
                or (parsed_port is not None and not 1 <= parsed_port <= 65535)
                or (
                    parsed.scheme == "http"
                    and not AutoMixerServer._is_loopback_host(parsed.hostname)
                )
            ):
                raise RuntimeError(
                    "AUTOMIXER_WS_ALLOWED_ORIGINS entries must be exact "
                    "http(s) origins without paths or wildcards; non-loopback "
                    "origins require HTTPS"
                )
            if origin not in origins:
                origins.append(origin)
        return tuple(origins)

    async def _authenticate_websocket(self, websocket: WebSocketServerProtocol) -> bool:
        """Authenticate the first frame before registering a remotely bound client."""
        if not self._ws_auth_required:
            return True
        if not self._ws_auth_attempt_bucket.consume():
            await websocket.close(code=1013, reason="authentication_rate_limited")
            return False
        try:
            raw_message = await asyncio.wait_for(
                websocket.recv(),
                timeout=WS_AUTHENTICATION_TIMEOUT_SECONDS,
            )
            request = json.loads(raw_message)
        except (asyncio.TimeoutError, json.JSONDecodeError, TypeError, ValueError):
            await websocket.close(code=1008, reason="authentication_required")
            return False

        token = request.get("token") if isinstance(request, dict) else None
        client_id = request.get("client_id") if isinstance(request, dict) else None
        authenticated = bool(
            isinstance(request, dict)
            and request.get("type") == "authenticate"
            and isinstance(token, str)
            and re.fullmatch(r"[A-Za-z0-9_-]{43}", token)
            and client_id == self._ws_expected_client_id
            and self.auth_manager.authenticate(token=token, client_id=client_id)
        )
        expires_at = self.auth_manager.token_expiry(token) if authenticated else None
        token = None
        if not authenticated:
            self._authenticated_websocket_expiry.pop(id(websocket), None)
            await websocket.close(code=1008, reason="authentication_failed")
            return False
        if expires_at is None and not self._ws_token_process_lifetime:
            await websocket.close(code=1008, reason="authentication_failed")
            return False
        if expires_at is not None:
            self._authenticated_websocket_expiry[id(websocket)] = expires_at
        else:
            self._authenticated_websocket_expiry.pop(id(websocket), None)

        await send_json(
            websocket,
            {"type": "authentication_status", "authenticated": True},
            converter=convert_numpy_types,
            logger_=logger,
        )
        return True

    async def _expire_authenticated_websocket(
        self,
        websocket: WebSocketServerProtocol,
        expires_at: float,
    ) -> None:
        await asyncio.sleep(max(0.0, expires_at - time.time()))
        if self._authenticated_websocket_expiry.get(id(websocket)) != expires_at:
            return
        await self._close_expired_authenticated_websocket(websocket)

    async def _close_expired_authenticated_websocket(
        self,
        websocket: WebSocketServerProtocol,
    ) -> None:
        connection_id = id(websocket)
        if connection_id in self._authenticated_websocket_expiry_close_started:
            return
        self._authenticated_websocket_expiry_close_started.add(connection_id)
        await websocket.close(code=1008, reason="authentication_expired")

    @staticmethod
    def _parse_env_float(value: Any, default: float) -> float:
        try:
            if value is None or str(value).strip() == "":
                return float(default)
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @classmethod
    def _normalize_tcp_port(cls, port: Any) -> Optional[int]:
        try:
            port_number = int(port)
        except (TypeError, ValueError):
            return None
        if 1 <= port_number <= 65535:
            return port_number
        return None

    def _ws_port_force_cleanup_enabled(self) -> bool:
        return self._parse_env_bool(os.getenv("AUTOMIXER_FORCE_CLEANUP_WS_PORT", "false"))

    def _pids_listening_on_port(self, port: Any) -> List[int]:
        port_number = self._normalize_tcp_port(port)
        if port_number is None:
            logger.warning("Refusing port cleanup for invalid WebSocket port: %r", port)
            return []
        try:
            completed = subprocess.run(
                ["lsof", "-ti", f":{port_number}"],
                check=False,
                capture_output=True,
                text=True,
                timeout=2.0,
            )
        except FileNotFoundError:
            logger.warning("Cannot inspect port %s because lsof is unavailable", port_number)
            return []
        except subprocess.TimeoutExpired:
            logger.warning("Timed out inspecting port %s with lsof", port_number)
            return []
        if completed.returncode not in (0, 1):
            logger.warning(
                "lsof failed while inspecting port %s: %s",
                port_number,
                (completed.stderr or "").strip(),
            )
            return []
        pids: List[int] = []
        for raw_pid in completed.stdout.splitlines():
            try:
                pids.append(int(raw_pid.strip()))
            except ValueError:
                logger.warning(
                    "Ignoring non-numeric lsof pid for port %s: %r",
                    port_number,
                    raw_pid,
                )
        return pids

    def _force_cleanup_ws_port(self) -> None:
        port_number = self._normalize_tcp_port(self.ws_port)
        if port_number is None:
            raise RuntimeError(f"Invalid WebSocket port for cleanup: {self.ws_port!r}")
        current_pid = os.getpid()
        pids = sorted(
            {pid for pid in self._pids_listening_on_port(port_number) if pid != current_pid}
        )
        if not pids:
            logger.warning("No external listener PID found for WebSocket port %s", port_number)
            return
        for pid in pids:
            logger.warning(
                "Force-stopping PID %s on WebSocket port %s after explicit operator opt-in",
                pid,
                port_number,
            )
            os.kill(pid, signal.SIGKILL)

    def _is_wing_deployment_write_boundary_active(self) -> bool:
        return (
            getattr(self, "connection_mode", "") == "wing"
            and isinstance(getattr(self, "mixer_client", None), (WingClient, EnhancedOSCClient))
        )

    def get_operator_mode_status(self) -> Dict[str, Any]:
        status = {
            "type": "operator_mode_status",
            **build_operator_mode_status(
                mode=self.operator_mode,
                connection_mode=self.connection_mode,
                wing_boundary_active=self._is_wing_deployment_write_boundary_active(),
                auto_mode_unlocked=self._auto_mode_unlocked,
            ),
        }
        status.update(
            {
                "packaged_offline_lock": self._packaged_offline_lock,
                "mixer_transport_locked": self._packaged_offline_lock,
                "websocket_auth_required": self._ws_auth_required,
                "unauthenticated_read_only": bool(
                    self._network_runtime_started and not self._ws_auth_required
                ),
            }
        )
        return status

    def _activate_network_runtime_safety_boundary(self) -> None:
        """Fail source runtimes without a token down to observation-only."""

        self._network_runtime_started = True
        if self._ws_auth_required:
            return
        self.operator_mode = normalize_operator_mode("analyze")
        self._auto_mode_unlocked = False
        self._wing_observation_only = True
        self._wing_record_telemetry = False
        self.auto_soundcheck_observe_only = True

    def _unauthenticated_message_is_read_only(
        self,
        message_type: Any,
        payload: Mapping[str, Any],
    ) -> bool:
        """Return whether an unauthenticated source frame is explicitly safe."""

        if self._ws_auth_required:
            return True
        if not isinstance(message_type, str) or not message_type:
            return False
        if message_type == "connect_wing":
            return (
                payload.get("observation_only") is True
                or payload.get("observe_only") is True
            )
        return message_type in UNAUTHENTICATED_READ_ONLY_MESSAGE_TYPES

    async def _block_unauthenticated_control_message(
        self,
        websocket: WebSocketServerProtocol,
        message_type: Any,
        payload: Mapping[str, Any],
    ) -> bool:
        if self._unauthenticated_message_is_read_only(message_type, payload):
            return False
        await self.send_to_client(
            websocket,
            {
                "type": "control_authentication_required",
                "status": "blocked",
                "success": False,
                "reason": "authenticated_control_session_required",
                "request_type": message_type if isinstance(message_type, str) else None,
                "unauthenticated_read_only": True,
                "live_authority_granted": False,
                "readiness_closure_allowed": False,
            },
        )
        return True

    def _packaged_offline_block_payload(self, operation: str) -> Dict[str, Any]:
        return {
            "type": "connection_status",
            "connected": False,
            "status": "blocked",
            "reason": "packaged_offline_lock_active",
            "operation": str(operation or "mixer_transport"),
            "packaged_offline_lock": True,
        }

    async def block_packaged_offline_transport(
        self,
        operation: str,
        *,
        websocket: Optional[WebSocketServerProtocol] = None,
    ) -> bool:
        """Reject mixer/discovery traffic while the immutable smoke lock is active."""

        if not self._packaged_offline_lock:
            return False
        payload = self._packaged_offline_block_payload(operation)
        if websocket is None:
            await self.broadcast(payload)
        else:
            await self.send_to_client(websocket, payload)
        logger.warning(
            "Packaged offline lock blocked mixer transport operation: %s",
            operation,
        )
        return True

    def set_operator_mode(self, mode: str, *, reason: str = "") -> Dict[str, Any]:
        previous_mode = self.operator_mode
        unauthenticated_read_only = bool(
            self._network_runtime_started and not self._ws_auth_required
        )
        if self._packaged_offline_lock or unauthenticated_read_only:
            self.operator_mode = normalize_operator_mode("analyze")
        else:
            self.operator_mode = normalize_operator_mode(mode)
        status = self.get_operator_mode_status()
        status.update(
            {
                "previous_mode": previous_mode,
                "requested_mode": str(mode or ""),
                "reason": (
                    "packaged_offline_lock_active"
                    if self._packaged_offline_lock
                    else "authenticated_control_session_required"
                    if unauthenticated_read_only
                    else str(reason or "operator_mode_change")
                ),
            }
        )
        return status

    def get_manual_write_supervision_status(self) -> Dict[str, Any]:
        operator_mode_status = self.get_operator_mode_status()
        approved_live_write_kinds = self._effective_live_write_kinds_for_manual_gate(
            operator_mode_status
        )
        status: Dict[str, Any] = {
            "deployment_write_boundary_enabled": self._wing_deployment_write_boundary_enabled,
            "deployment_write_boundary_active": self._is_wing_deployment_write_boundary_active(),
            "wing_observation_only": self._wing_observation_only,
            "wing_record_telemetry": self._wing_record_telemetry,
            "packaged_offline_lock": self._packaged_offline_lock,
            "mixer_transport_locked": self._packaged_offline_lock,
            "connection_mode": self.connection_mode,
            "approved_live_write_kinds": approved_live_write_kinds,
            "operator_mode": self.operator_mode,
            "requires_operator_mode": "supervised",
            "emergency_stopped": self._manual_write_emergency_stopped,
            "reset_required": self._manual_write_emergency_stopped,
            "emergency_reset_requires_operator_action": True,
            "emergency_stop_in_progress": self._system_emergency_stop_in_progress,
            "emergency_stop_teardown_complete": bool(
                self._manual_write_emergency_stopped
                and not self._system_emergency_stop_in_progress
                and not self._system_emergency_stop_errors
            ),
            "emergency_stop_teardown_errors": list(
                self._system_emergency_stop_errors
            ),
            "cooldown_sec": self._manual_write_cooldown_sec,
            "write_ahead_log": str(self._manual_write_wal_path()),
            "console_writer_lock": str(self._console_writer_lock_path),
            "console_writer_lock_scope": "per_user_global",
            "wal_recovery_required": bool(
                self._manual_write_recovery_transactions
            ),
            "wal_unresolved_transaction_ids": sorted(
                self._manual_write_recovery_transactions
            ),
        }
        if self.mixer_client and hasattr(self.mixer_client, "get_write_gate_status"):
            status["write_gate"] = self.mixer_client.get_write_gate_status()
        return status

    def reserve_operator_apply_idempotency_key(
        self,
        idempotency_key: Any,
        *,
        proposal_id: Any = None,
        approval_id: Any = None,
    ) -> Dict[str, Any]:
        key = str(idempotency_key or "").strip()
        if not key:
            return {
                "reserved": False,
                "reason": "missing_idempotency_key",
                "idempotency_key": None,
            }
        if key in self._operator_apply_idempotency_keys:
            return {
                "reserved": False,
                "reason": "duplicate_idempotency_key",
                "idempotency_key": key,
                "first_seen": self._operator_apply_idempotency_keys[key],
            }
        self._operator_apply_idempotency_keys[key] = {
            "created_at": _utc_timestamp(),
            "proposal_id": proposal_id,
            "approval_id": approval_id,
        }
        return {
            "reserved": True,
            "reason": "reserved",
            "idempotency_key": key,
        }

    def release_operator_apply_idempotency_key_if_no_send(
        self,
        idempotency_key: Any,
        *,
        proposal_id: Any = None,
    ) -> bool:
        key = str(idempotency_key or "").strip()
        current = self._operator_apply_idempotency_keys.get(key)
        if not key or not isinstance(current, Mapping):
            return False
        if proposal_id is not None and current.get("proposal_id") != proposal_id:
            return False
        if key in self._durable_operator_apply_idempotency_keys:
            return False
        self._operator_apply_idempotency_keys.pop(key, None)
        return True

    def record_automation_decision_trace(self, **payload: Any) -> Dict[str, Any]:
        try:
            known_fields = AutomationDecisionTrace.__dataclass_fields__
            trace_payload = {key: value for key, value in payload.items() if key in known_fields}
            extra_payload = {key: value for key, value in payload.items() if key not in known_fields}
            if extra_payload:
                metadata = dict(trace_payload.get("metadata") or {})
                for key, value in extra_payload.items():
                    metadata.setdefault(key, value)
                trace_payload["metadata"] = metadata
            metadata = dict(trace_payload.get("metadata") or {})
            runtime_identity = getattr(self, "_runtime_identity", {})
            metadata["trace_provenance"] = {
                "schema": TRACE_PROVENANCE_SCHEMA,
                "origin": "server_runtime",
                "runtime_instance_nonce": str(runtime_identity.get("instance_nonce") or ""),
                "runtime_kind": str(runtime_identity.get("runtime_kind") or "unknown"),
            }
            trace_payload["metadata"] = metadata
            trace = AutomationDecisionTrace(**trace_payload).to_dict()
            trace.update(extra_payload)
            recorded = self.automation_trace_store.append(trace)
            memory_store = getattr(self, "decision_memory_store", None)
            trace_store = getattr(self, "automation_trace_store", None)
            if (
                memory_store is not None
                and trace_store is not None
                and Path(getattr(memory_store, "root_dir", "")) == Path(getattr(trace_store, "root_dir", ""))
            ):
                try:
                    memory_store.append_from_trace(recorded)
                except Exception as memory_exc:
                    logger.warning("Could not record decision memory: %s", memory_exc, exc_info=True)
            return recorded
        except Exception as exc:
            logger.warning("Could not record automation decision trace: %s", exc, exc_info=True)
            return {
                "decision_id": payload.get("decision_id"),
                "status": "trace_write_failed",
                "error": str(exc),
            }

    def get_automation_decision_traces(self, limit: int = 50) -> List[Dict[str, Any]]:
        try:
            return self.automation_trace_store.read_recent(limit)
        except Exception as exc:
            logger.warning("Could not read automation decision traces: %s", exc, exc_info=True)
            return []

    def get_decision_memory_entries(self, limit: int = 50) -> List[Dict[str, Any]]:
        try:
            return self.decision_memory_store.read_recent(limit)
        except Exception as exc:
            logger.warning("Could not read decision memory: %s", exc, exc_info=True)
            return []

    def reset_manual_write_emergency_stop(
        self,
        *,
        approved: bool = False,
        approval_id: Optional[str] = None,
        reason: str = "",
    ) -> Dict[str, Any]:
        """Clear the manual-write emergency latch only after explicit operator approval.

        This does not unfreeze automation controllers. A separate operator review is
        required before returning the whole live system to a ready state.
        """

        if not self._manual_write_emergency_stopped:
            result = {
                "type": "manual_write_emergency_reset_result",
                "status": "not_required",
                "success": True,
                "reason": "emergency_stop_not_active",
                "approval_id": approval_id,
                "automation_remains_frozen": True,
                "manual_write_supervision": self.get_manual_write_supervision_status(),
                "ready_for_live": self.get_ready_for_live_status(),
            }
            artifact_path = self._record_manual_write_event(
                "manual_write_emergency_reset_not_required",
                result,
            )
            if artifact_path:
                result["artifact_path"] = artifact_path
            return result

        if approved is not True or not str(approval_id or "").strip():
            result = {
                "type": "manual_write_emergency_reset_result",
                "status": "blocked",
                "success": False,
                "reason": "operator_approval_required",
                "approval_id": approval_id,
                "automation_remains_frozen": True,
                "manual_write_supervision": self.get_manual_write_supervision_status(),
                "ready_for_live": self.get_ready_for_live_status(),
            }
            artifact_path = self._record_manual_write_event(
                "manual_write_emergency_reset_blocked",
                result,
            )
            if artifact_path:
                result["artifact_path"] = artifact_path
            return result

        if self._system_emergency_stop_in_progress:
            result = {
                "type": "manual_write_emergency_reset_result",
                "status": "blocked",
                "success": False,
                "reason": "emergency_stop_teardown_in_progress",
                "approval_id": str(approval_id),
                "automation_remains_frozen": True,
                "manual_write_supervision": self.get_manual_write_supervision_status(),
                "ready_for_live": self.get_ready_for_live_status(),
            }
            artifact_path = self._record_manual_write_event(
                "manual_write_emergency_reset_blocked",
                result,
            )
            if artifact_path:
                result["artifact_path"] = artifact_path
            return result

        if self._system_emergency_stop_errors:
            result = {
                "type": "manual_write_emergency_reset_result",
                "status": "blocked",
                "success": False,
                "reason": "emergency_stop_teardown_incomplete",
                "approval_id": str(approval_id),
                "automation_remains_frozen": True,
                "teardown_errors": list(self._system_emergency_stop_errors),
                "manual_write_supervision": self.get_manual_write_supervision_status(),
                "ready_for_live": self.get_ready_for_live_status(),
            }
            artifact_path = self._record_manual_write_event(
                "manual_write_emergency_reset_blocked",
                result,
            )
            if artifact_path:
                result["artifact_path"] = artifact_path
            return result

        if self._manual_write_recovery_transactions:
            result = {
                "type": "manual_write_emergency_reset_result",
                "status": "blocked",
                "success": False,
                "reason": "console_write_recovery_pending",
                "approval_id": str(approval_id),
                "automation_remains_frozen": True,
                "manual_write_supervision": self.get_manual_write_supervision_status(),
                "ready_for_live": self.get_ready_for_live_status(),
            }
            artifact_path = self._record_manual_write_event(
                "manual_write_emergency_reset_blocked",
                result,
            )
            if artifact_path:
                result["artifact_path"] = artifact_path
            return result

        self._manual_write_emergency_stopped = False
        self._system_emergency_teardown_complete = False
        result = {
            "type": "manual_write_emergency_reset_result",
            "status": "reset",
            "success": True,
            "reason": str(reason or "operator_approved_emergency_reset"),
            "approval_id": str(approval_id),
            "automation_remains_frozen": True,
            "manual_write_supervision": self.get_manual_write_supervision_status(),
            "ready_for_live": self.get_ready_for_live_status(),
        }
        artifact_path = self._record_manual_write_event(
            "manual_write_emergency_reset",
            result,
        )
        if artifact_path:
            result["artifact_path"] = artifact_path
        return result

    def _freeze_autonomous_writers_now(
        self,
        *,
        include_agent: bool = True,
    ) -> list[dict[str, str]]:
        """Synchronously remove live-write authority before any teardown await."""

        errors: list[dict[str, str]] = []
        self.agent_auto_apply_enabled = False
        self._agent_force_auto_apply = False
        self.auto_soundcheck_running = False
        self.auto_soundcheck_observe_only = True

        for controller, label in (
            (getattr(self, "gain_staging", None), "gain_staging"),
            (getattr(self, "auto_fader_controller", None), "auto_fader"),
            (getattr(self, "auto_fader_v2_controller", None), "auto_fader_v2"),
        ):
            if controller is None:
                continue
            try:
                freeze = getattr(controller, "set_automation_frozen", None)
                if callable(freeze):
                    freeze(True)
                elif hasattr(controller, "automation_frozen"):
                    controller.automation_frozen = True
                else:
                    raise RuntimeError("controller has no synchronous freeze gate")
            except Exception as exc:
                logger.error("Immediate %s freeze failed: %s", label, exc, exc_info=True)
                errors.append({"step": f"{label}_freeze", "error": type(exc).__name__})

        calibrator = getattr(self, "safe_gain_calibrator", None)
        if calibrator is not None:
            try:
                # A callback may otherwise auto-apply from the audio worker
                # after the server gate has latched. Detach both the callback
                # and its raw mixer reference synchronously.
                calibrator.on_suggestions_ready = None
                calibrator.on_progress_update = None
                calibrator.mixer_client = None
            except Exception as exc:
                logger.error("Immediate SafeGain freeze failed: %s", exc, exc_info=True)
                errors.append({"step": "safe_gain_freeze", "error": type(exc).__name__})
        safe_gain_stop = getattr(self, "_safe_gain_audio_thread_stop", None)
        if safe_gain_stop is not None:
            try:
                safe_gain_stop.set()
            except Exception as exc:
                errors.append({"step": "safe_gain_stop_signal", "error": type(exc).__name__})

        # Do not clear voice_control.is_listening here: both real voice
        # implementations use it to decide whether stop_listening must close
        # the PyAudio stream. Scheduling and execution are already blocked by
        # the server emergency generation below.
        for task in tuple(self._voice_command_tasks):
            if not task.done():
                task.cancel()

        agent = getattr(self, "mixing_agent", None)
        emergency_stop = getattr(agent, "emergency_stop", None)
        if include_agent and callable(emergency_stop):
            try:
                emergency_stop()
            except Exception as exc:
                logger.error("Immediate agent freeze failed: %s", exc, exc_info=True)
                errors.append({"step": "mixing_agent_emergency", "error": type(exc).__name__})
        return errors

    def _stop_safe_gain_runtime(self) -> None:
        """Stop and prove termination of the SafeGain callback/audio worker."""

        calibrator = getattr(self, "safe_gain_calibrator", None)
        stop_event = getattr(self, "_safe_gain_audio_thread_stop", None)
        thread = getattr(self, "_safe_gain_audio_thread", None)
        if calibrator is not None:
            calibrator.on_suggestions_ready = None
            calibrator.on_progress_update = None
            calibrator.mixer_client = None
        if stop_event is not None:
            stop_event.set()
        if thread is not None and thread is threading.current_thread():
            raise RuntimeError("SafeGain worker cannot join itself")
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        if thread is not None and thread.is_alive():
            raise RuntimeError("SafeGain audio worker did not terminate")
        if calibrator is not None:
            calibrator.reset()
        if self.safe_gain_calibrator is calibrator:
            self.safe_gain_calibrator = None
        if self._safe_gain_audio_thread is thread:
            self._safe_gain_audio_thread = None
        if self._safe_gain_audio_thread_stop is stop_event:
            self._safe_gain_audio_thread_stop = None

    def _finish_voice_command_task(self, task: asyncio.Task[Any]) -> None:
        self._voice_command_tasks.discard(task)
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.error("Voice command task failed: %s", exc, exc_info=True)

    def _schedule_voice_command(self, command: dict) -> None:
        if self._is_shutting_down or self._manual_write_emergency_stopped:
            logger.warning("Dropped voice command after live-write gate closed")
            return
        task = asyncio.create_task(self._execute_voice_command(command))
        self._voice_command_tasks.add(task)
        task.add_done_callback(self._finish_voice_command_task)

    async def _drain_voice_command_tasks(self) -> None:
        tasks = [task for task in tuple(self._voice_command_tasks) if task is not asyncio.current_task()]
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            failures = [
                result for result in results
                if isinstance(result, BaseException)
                and not isinstance(result, asyncio.CancelledError)
            ]
            if failures:
                raise ExceptionGroup("Voice command task drain failed", failures)
        self._voice_command_tasks = {
            task for task in self._voice_command_tasks if not task.done()
        }

    @staticmethod
    async def _await_task_terminal_without_cancelling(task: asyncio.Task[Any]) -> Any:
        """Wait for an owned task to finish while preserving caller cancellation."""

        if task is asyncio.current_task():
            raise RuntimeError("Cannot await the current task as a teardown dependency")
        caller_cancellation: Optional[asyncio.CancelledError] = None
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as exc:
                caller_cancellation = caller_cancellation or exc
                continue
            except BaseException:
                break
        error: Optional[BaseException] = None
        outcome = None
        try:
            outcome = task.result()
        except BaseException as exc:
            error = exc
        if caller_cancellation is not None:
            raise caller_cancellation
        if error is not None:
            raise error
        return outcome

    async def activate_system_emergency_stop(self, reason: str = "") -> Dict[str, Any]:
        """Latch the write gate and stop every active autonomous writer."""
        if self._system_emergency_stop_in_progress:
            return {
                "status": "stopping",
                "success": False,
                "reason": str(reason or "system_emergency_stop"),
                "generation": self._manual_write_emergency_generation,
                "coalesced": True,
                "teardown_complete": False,
            }
        latched_before = self._manual_write_emergency_stopped
        teardown_complete_before = self._system_emergency_teardown_complete
        if not latched_before:
            self._manual_write_emergency_stopped = True
            self._manual_write_emergency_generation += 1
        phase_zero_errors = self._freeze_autonomous_writers_now(
            include_agent=(
                not teardown_complete_before
                or bool(self._system_emergency_stop_errors)
            )
        )
        for run_id in list(getattr(self, "_soundcheck_manifests", {})):
            self._invalidate_soundcheck_console_fence(
                run_id,
                "emergency_stop_active",
            )
        if isinstance(self.last_soundcheck_readiness, Mapping):
            await self.broadcast({
                "type": "auto_soundcheck_status",
                "status": "readiness_invalidated",
                "running": False,
                "is_running": False,
                "message": "Emergency stop active; soundcheck readiness invalidated.",
                "soundcheck_readiness": dict(self.last_soundcheck_readiness),
            })
        if (
            latched_before
            and teardown_complete_before
            and not self._system_emergency_stop_errors
            and not phase_zero_errors
        ):
            return {
                "status": "stopped",
                "success": True,
                "reason": str(reason or "system_emergency_stop"),
                "generation": self._manual_write_emergency_generation,
                "coalesced": True,
                "teardown_complete": True,
            }
        self._system_emergency_stop_in_progress = True
        self._system_emergency_stop_errors = []
        errors: list[dict[str, str]] = list(phase_zero_errors)

        async def run_step(label: str, action) -> bool:
            try:
                await self._cancellation_safe_thread_call(action)
                return True
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("Emergency stop step %s failed: %s", label, exc, exc_info=True)
                errors.append({"step": label, "error": type(exc).__name__})
                return False

        try:
            try:
                await self._drain_voice_command_tasks()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                errors.append({
                    "step": "voice_command_tasks",
                    "error": type(exc).__name__,
                })

            if (
                self.safe_gain_calibrator is not None
                or self._safe_gain_audio_thread is not None
            ):
                await run_step("safe_gain_runtime", self._stop_safe_gain_runtime)

            engine = getattr(self, "auto_soundcheck_engine", None)
            if engine is not None:
                try:
                    await self._stop_auto_soundcheck_engine_async(observe_only=True)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.error("Emergency AutoSoundcheck stop failed: %s", exc, exc_info=True)
                    errors.append({
                        "step": "auto_soundcheck_engine",
                        "error": type(exc).__name__,
                    })

            legacy_soundcheck_task = getattr(self, "auto_soundcheck_task", None)
            if legacy_soundcheck_task is not None:
                if legacy_soundcheck_task is asyncio.current_task():
                    errors.append({
                        "step": "legacy_auto_soundcheck_task",
                        "error": "SelfDrainBlocked",
                    })
                else:
                    try:
                        await self._cancel_and_drain_async_task(
                            legacy_soundcheck_task
                        )
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        logger.error(
                            "Emergency legacy AutoSoundcheck drain failed: %s",
                            exc,
                            exc_info=True,
                        )
                        errors.append({
                            "step": "legacy_auto_soundcheck_task",
                            "error": type(exc).__name__,
                        })
                    finally:
                        if legacy_soundcheck_task.done() and (
                            self.auto_soundcheck_task is legacy_soundcheck_task
                        ):
                            self.auto_soundcheck_task = None

            agent = getattr(self, "mixing_agent", None)
            if agent is not None:
                agent_stopped = await run_step("mixing_agent_stop", agent.stop)
                if agent_stopped:
                    agent_task = self.mixing_agent_task
                    task_drained = True
                    if agent_task is not None:
                        if agent_task is asyncio.current_task():
                            task_drained = False
                            errors.append({
                                "step": "mixing_agent_task",
                                "error": "SelfDrainBlocked",
                            })
                        else:
                            try:
                                await self._cancel_and_drain_async_task(agent_task)
                            except asyncio.CancelledError:
                                raise
                            except Exception as exc:
                                task_drained = False
                                errors.append({
                                    "step": "mixing_agent_task",
                                    "error": type(exc).__name__,
                                })
                            finally:
                                if agent_task.done() and self.mixing_agent_task is agent_task:
                                    self.mixing_agent_task = None
                    if task_drained:
                        self.mixing_agent = None

            controllers = (
                (getattr(self, "gain_staging", None), "gain_staging", ("stop",)),
                (getattr(self, "auto_fader_controller", None), "auto_fader", ("stop",)),
                (getattr(self, "auto_fader_v2_controller", None), "auto_fader_v2", ("stop",)),
                (getattr(self, "auto_eq_controller", None), "auto_eq", ("stop",)),
                (
                    getattr(self, "multi_channel_auto_eq_controller", None),
                    "multi_channel_auto_eq",
                    ("stop_all", "stop"),
                ),
                (getattr(self, "auto_compressor_controller", None), "auto_compressor", ("stop",)),
                (
                    getattr(self, "phase_alignment_controller", None),
                    "phase_alignment",
                    ("stop_analysis",),
                ),
                (
                    getattr(self, "voice_control", None),
                    "voice_control",
                    ("stop_listening", "stop"),
                ),
            )
            for controller, label, stop_methods in controllers:
                if controller is None:
                    continue
                for stop_method in stop_methods:
                    stop = getattr(controller, stop_method, None)
                    if callable(stop):
                        await run_step(f"{label}_{stop_method}", stop)
                        break

            self.agent_auto_apply_enabled = False
        except asyncio.CancelledError:
            errors.append({"step": "emergency_teardown", "error": "CancelledError"})
            raise
        finally:
            self._system_emergency_stop_errors = errors
            self._system_emergency_stop_in_progress = False

        complete = not errors
        self._system_emergency_teardown_complete = complete
        return {
            "status": "stopped" if complete else "stop_incomplete",
            "success": complete,
            "reason": str(reason or "system_emergency_stop"),
            "generation": self._manual_write_emergency_generation,
            "coalesced": latched_before,
            "teardown_complete": complete,
            "teardown_errors": list(errors),
        }

    async def _stop_auto_soundcheck_engine_async(
        self,
        *,
        observe_only: bool,
    ) -> bool:
        """Detach and fully drain AutoSoundcheck without blocking the event loop."""

        async with self._audio_capture_lock:
            engine = self.auto_soundcheck_engine
            self.auto_soundcheck_running = False
            self.auto_soundcheck_observe_only = bool(observe_only)
            if engine is None:
                return False
            try:
                final_status = await self._cancellation_safe_thread_call(
                    engine.get_status
                )
            except Exception:
                final_status = {}
            readiness = (
                final_status.get("soundcheck_readiness")
                if isinstance(final_status, Mapping)
                else None
            )
            if isinstance(readiness, Mapping):
                retained = self.last_soundcheck_readiness
                retained_is_aggregate = bool(
                    isinstance(retained, Mapping)
                    and retained.get("run_id") == readiness.get("run_id")
                    and retained.get("plan_revision") == readiness.get("plan_revision")
                    and retained.get("plan_fingerprint")
                    == readiness.get("plan_fingerprint")
                    and retained.get("proposal_only") is False
                )
                if not retained_is_aggregate:
                    self.last_soundcheck_readiness = dict(readiness)
            try:
                await self._cancellation_safe_thread_call(engine.stop)
            except BaseException:
                if self.auto_soundcheck_engine is None:
                    self.auto_soundcheck_engine = engine
                raise
            if self.auto_soundcheck_engine is engine:
                self.auto_soundcheck_engine = None
            return True

    def import_safe_gain_suggestions_to_operator_queue(
        self,
        suggestions: Optional[Dict[Any, Any]] = None,
        *,
        source: str = SAFE_GAIN_SOURCE,
    ) -> Dict[str, Any]:
        if suggestions is None and self.safe_gain_calibrator:
            try:
                suggestions = self.safe_gain_calibrator.get_suggestions()
            except Exception as exc:
                return {
                    "type": "operator_proposals_imported",
                    "status": "unavailable",
                    "success": False,
                    "source": source,
                    "reason": "safe_gain_suggestions_unavailable",
                    "error": str(exc),
                    "operator_mode": self.get_operator_mode_status(),
                    "imported_count": 0,
                    "created_count": 0,
                    "existing_count": 0,
                    "blocked_count": 0,
                    "results": [],
                }

        channel_mapping = (
            getattr(self.safe_gain_calibrator, "channel_mapping", None)
            if self.safe_gain_calibrator
            else None
        )
        proposals = build_safe_gain_operator_proposals(
            suggestions or {},
            channel_mapping=channel_mapping,
            mixer_client=self.mixer_client,
            source=source,
        )
        summary = import_operator_proposals(
            self.operator_proposal_queue,
            proposals,
            self.get_operator_mode_status(),
            source=source,
        )
        if not suggestions:
            summary["reason"] = "no_safe_gain_suggestions"
        elif not proposals:
            summary["reason"] = "no_actionable_safe_gain_suggestions"
        return summary

    def import_soundcheck_recommendations_to_operator_queue(
        self,
        bundle: Optional[Dict[str, Any]] = None,
        *,
        source: str = SOUNDCHECK_SOURCE,
        trusted_engine_bundle: bool = False,
    ) -> Dict[str, Any]:
        if bundle is None:
            engine = getattr(self, "auto_soundcheck_engine", None)
            if engine and hasattr(engine, "get_soundcheck_recommendation_bundle"):
                try:
                    bundle = engine.get_soundcheck_recommendation_bundle()
                except Exception as exc:
                    return {
                        "type": "operator_proposals_imported",
                        "status": "unavailable",
                        "success": False,
                        "source": source,
                        "reason": "soundcheck_recommendation_bundle_unavailable",
                        "error": str(exc),
                        "operator_mode": self.get_operator_mode_status(),
                        "imported_count": 0,
                        "created_count": 0,
                        "existing_count": 0,
                        "blocked_count": 0,
                        "results": [],
                    }
                trusted_engine_bundle = True

        if (
            isinstance(bundle, Mapping)
            and isinstance(bundle.get("supervised_transaction_plans"), Mapping)
            and not trusted_engine_bundle
        ):
            return {
                "type": "operator_proposals_imported",
                "status": "blocked",
                "success": False,
                "complete": False,
                "source": source,
                "reason": "untrusted_supervised_soundcheck_manifest",
                "operator_mode": self.get_operator_mode_status(),
                "imported_count": 0,
                "created_count": 0,
                "existing_count": 0,
                "blocked_count": 0,
                "results": [],
            }

        manifest = (
            bundle.get("supervised_transaction_plans")
            if isinstance(bundle, Mapping)
            else None
        )
        manifest_error = None
        manifest_changed = False
        manifest_registration_required = False
        new_run_registration = False
        run_id = ""
        if isinstance(manifest, Mapping):
            run_id = str(manifest.get("run_id") or "").strip()
            revision = manifest.get("plan_revision")
            fingerprint = str(manifest.get("plan_fingerprint") or "").strip()
            content_sha256 = str(
                manifest.get("manifest_content_sha256") or ""
            ).strip()
            if (
                not run_id
                or isinstance(revision, bool)
                or not isinstance(revision, int)
                or revision < 1
                or not fingerprint
            ):
                manifest_error = "invalid_soundcheck_manifest_identity"
            if (
                manifest_error is None
                and manifest.get("schema_version")
                != "soundcheck_batch_manifest/v2"
            ):
                manifest_error = "unsupported_soundcheck_manifest_schema"
            elif (
                manifest_error is None
                and (
                    not content_sha256
                    or content_sha256
                    != soundcheck_manifest_content_sha256(manifest)
                )
            ):
                manifest_error = "soundcheck_manifest_content_digest_mismatch"
            logical_id_errors = soundcheck_manifest_logical_id_errors(manifest)
            if manifest_error is None and logical_id_errors:
                manifest_error = logical_id_errors[0]
            previous_manifest = self._soundcheck_manifests.get(run_id)
            new_run_registration = not isinstance(
                previous_manifest,
                Mapping,
            )
            manifest_registration_required = new_run_registration
            active_run_id = self._current_active_soundcheck_run_id()
            if (
                manifest_error is None
                and isinstance(previous_manifest, Mapping)
                and active_run_id
                and run_id != active_run_id
            ):
                manifest_error = "soundcheck_run_superseded"
            if manifest_error is None and isinstance(previous_manifest, Mapping):
                previous_revision = previous_manifest.get("plan_revision")
                previous_fingerprint = str(
                    previous_manifest.get("plan_fingerprint") or ""
                )
                if revision == previous_revision and fingerprint == previous_fingerprint:
                    previous_content = json.dumps(
                        previous_manifest,
                        sort_keys=True,
                        separators=(",", ":"),
                        default=str,
                    )
                    incoming_content = json.dumps(
                        manifest,
                        sort_keys=True,
                        separators=(",", ":"),
                        default=str,
                    )
                    if incoming_content != previous_content:
                        manifest_error = (
                            "soundcheck_manifest_same_identity_content_mismatch"
                        )
                    else:
                        manifest_changed = False
                elif revision == previous_revision and fingerprint != previous_fingerprint:
                    manifest_error = "soundcheck_manifest_revision_collision"
                elif revision != previous_revision + 1:
                    manifest_error = "soundcheck_manifest_revision_not_monotonic"
                elif content_sha256 == str(
                    previous_manifest.get("manifest_content_sha256") or ""
                ):
                    manifest_error = "soundcheck_manifest_revision_without_content_change"
                else:
                    manifest_changed = True
                    manifest_registration_required = True
            if (
                manifest_error is None
                and getattr(self, "_soundcheck_parent_journal_error", None)
            ):
                manifest_error = "soundcheck_parent_journal_recovery_failed"
        if manifest_error is not None:
            return {
                "type": "operator_proposals_imported",
                "status": "blocked",
                "success": False,
                "complete": False,
                "source": source,
                "reason": manifest_error,
                "operator_mode": self.get_operator_mode_status(),
                "imported_count": 0,
                "created_count": 0,
                "existing_count": 0,
                "blocked_count": 0,
                "results": [],
            }

        if new_run_registration and run_id:
            other_run_rows = [
                item
                for item in self.operator_proposal_queue.pending_actions()
                if isinstance(item, Mapping)
                and isinstance(item.get("raw"), Mapping)
                and str(item["raw"].get("soundcheck_run_id") or "").strip()
                and str(item["raw"].get("soundcheck_run_id") or "") != run_id
            ]
            if any(
                str(item.get("status") or "") == "applying"
                for item in other_run_rows
            ):
                return {
                    "type": "operator_proposals_imported",
                    "status": "blocked",
                    "success": False,
                    "complete": False,
                    "source": source,
                    "reason": "soundcheck_other_run_apply_in_progress",
                    "operator_mode": self.get_operator_mode_status(),
                    "imported_count": 0,
                    "created_count": 0,
                    "existing_count": 0,
                    "blocked_count": 0,
                    "results": [],
                }

        old_rows: List[Mapping[str, Any]] = []
        if manifest_changed and run_id:
            old_rows = [
                item
                for item in self.operator_proposal_queue.pending_actions()
                if isinstance(item, Mapping)
                and isinstance(item.get("raw"), Mapping)
                and str(item["raw"].get("soundcheck_run_id") or "") == run_id
            ]
            if any(str(item.get("status") or "") == "applying" for item in old_rows):
                return {
                    "type": "operator_proposals_imported",
                    "status": "blocked",
                    "success": False,
                    "complete": False,
                    "source": source,
                    "reason": "soundcheck_previous_revision_apply_in_progress",
                    "operator_mode": self.get_operator_mode_status(),
                    "imported_count": 0,
                    "created_count": 0,
                    "existing_count": 0,
                    "blocked_count": 0,
                    "results": [],
                }

        if (
            isinstance(manifest, Mapping)
            and run_id
            and manifest_registration_required
        ):
            journal_path = self._append_soundcheck_parent_journal(
                "manifest_registered",
                run_id,
                plan_revision=manifest.get("plan_revision"),
                plan_fingerprint=manifest.get("plan_fingerprint"),
                manifest=copy.deepcopy(manifest),
            )
            if not journal_path:
                return {
                    "type": "operator_proposals_imported",
                    "status": "blocked",
                    "success": False,
                    "complete": False,
                    "source": source,
                    "reason": "soundcheck_parent_journal_unavailable",
                    "operator_mode": self.get_operator_mode_status(),
                    "imported_count": 0,
                    "created_count": 0,
                    "existing_count": 0,
                    "blocked_count": 0,
                    "results": [],
                }

        if manifest_changed and run_id:
            for item in old_rows:
                self.operator_proposal_queue.dismiss(
                    proposal_id=str(item.get("id") or ""),
                    reason="superseded_soundcheck_manifest_revision",
                )

        proposals = build_soundcheck_operator_proposals(
            bundle or {},
            mixer_client=self.mixer_client,
            source=source,
        )
        summary = import_operator_proposals(
            self.operator_proposal_queue,
            proposals,
            self.get_operator_mode_status(),
            source=source,
        )
        if isinstance(manifest, Mapping):
            if run_id:
                self._soundcheck_manifests[run_id] = copy.deepcopy(manifest)
                if manifest_registration_required:
                    self._active_soundcheck_run_id = run_id
                    self._retire_soundcheck_console_change_invalidators_except(
                        None
                    )
                    for previous_run_id in list(self._soundcheck_manifests):
                        if previous_run_id == run_id:
                            continue
                        previous_blockers = self._soundcheck_global_blockers.setdefault(
                            previous_run_id,
                            [],
                        )
                        if "soundcheck_run_superseded" not in previous_blockers:
                            previous_blockers.append("soundcheck_run_superseded")
                    for item in self.operator_proposal_queue.pending_actions():
                        if not (
                            isinstance(item, Mapping)
                            and isinstance(item.get("raw"), Mapping)
                        ):
                            continue
                        previous_run_id = str(
                            item["raw"].get("soundcheck_run_id") or ""
                        ).strip()
                        if previous_run_id and previous_run_id != run_id:
                            self.operator_proposal_queue.dismiss(
                                proposal_id=str(item.get("id") or ""),
                                reason="superseded_by_new_soundcheck_run",
                            )
                if manifest_changed:
                    self._soundcheck_outcomes[run_id] = {}
                    self._soundcheck_global_blockers[run_id] = []
                else:
                    self._soundcheck_outcomes.setdefault(run_id, {})
                    self._soundcheck_global_blockers.setdefault(run_id, [])
                successful_proposal_ids = {
                    str((result.get("proposal") or {}).get("id") or "")
                    for result in summary.get("results") or []
                    if isinstance(result, Mapping)
                    and result.get("status") in {"created", "exists"}
                    and isinstance(result.get("proposal"), Mapping)
                }
                expected_child_ids = {
                    str(plan.get("transaction_id") or "")
                    for plan in manifest.get("plans") or []
                    if isinstance(plan, Mapping)
                    and str(plan.get("transaction_id") or "").strip()
                }
                missing_child_ids = sorted(
                    expected_child_ids - successful_proposal_ids
                )
                blockers = [
                    item
                    for item in self._soundcheck_global_blockers.get(run_id, [])
                    if not str(item).startswith("proposal_import_incomplete")
                    and str(item)
                    != "soundcheck_restart_requires_fresh_preflight_and_reimport"
                ]
                if missing_child_ids:
                    blockers.append(
                        "proposal_import_incomplete:" + ",".join(missing_child_ids)
                    )
                    summary["status"] = "partial" if successful_proposal_ids else "blocked"
                    summary["success"] = False
                    summary["complete"] = False
                    summary["missing_soundcheck_child_ids"] = missing_child_ids
                preflight = self._soundcheck_preflight_assessments.get(run_id)
                preflight_matches = bool(
                    isinstance(preflight, Mapping)
                    and preflight.get("run_id") == manifest.get("run_id")
                    and preflight.get("plan_revision") == manifest.get("plan_revision")
                    and preflight.get("plan_fingerprint")
                    == manifest.get("plan_fingerprint")
                )
                if not preflight_matches:
                    blockers.append("soundcheck_physical_preflight_missing")
                else:
                    blockers.extend(
                        str(item)
                        for item in preflight.get("global_blockers") or []
                        if str(item).strip()
                    )
                    outcomes = self._soundcheck_outcomes.setdefault(run_id, {})
                    for assessment in preflight.get("assessments") or []:
                        if not isinstance(assessment, Mapping):
                            continue
                        operation_id = str(
                            assessment.get("operation_id") or ""
                        ).strip()
                        disposition = str(
                            assessment.get("disposition") or ""
                        ).strip()
                        if not operation_id or disposition == "apply":
                            continue
                        outcomes[operation_id] = {
                            "disposition": disposition,
                            "reason": str(assessment.get("reason") or disposition),
                            "physical_verified": assessment.get("physical_verified")
                            is True,
                            "audible_state_verified": assessment.get(
                                "audible_state_verified"
                            )
                            is True,
                            "status": "preflight_classified",
                            "evidence_captured_at_epoch": preflight.get(
                                "captured_at_epoch"
                            ),
                            "evidence_max_age_sec": preflight.get("max_age_sec"),
                            "wal_terminal": "not_applicable",
                            "audio_recheck_passed": False,
                        }
                blockers = sorted(set(blockers))
                self._soundcheck_global_blockers[run_id] = blockers
                readiness = aggregate_soundcheck_readiness(
                    manifest,
                    outcomes=self._soundcheck_outcomes.get(run_id, {}),
                    proposal_only=True,
                    global_blockers=blockers,
                    source_coverage=(
                        preflight.get("source_coverage")
                        if isinstance(preflight, Mapping)
                        else manifest.get("source_coverage") or {}
                    ),
                    main_47_48_evidence=(
                        preflight.get("main_47_48_evidence")
                        if isinstance(preflight, Mapping)
                        else {}
                    ),
                )
                self.last_soundcheck_readiness = dict(readiness)
                summary["soundcheck_readiness"] = dict(readiness)
                draft = getattr(self, "_soundcheck_replan_drafts", {}).get(run_id)
                summary["soundcheck_replan"] = self._public_soundcheck_replan(
                    draft
                )
        if trusted_engine_bundle and isinstance(bundle, dict):
            engine = getattr(self, "auto_soundcheck_engine", None)
            acknowledge = getattr(
                engine,
                "acknowledge_published_soundcheck_bundle",
                None,
            )
            if callable(acknowledge):
                try:
                    if acknowledge(bundle):
                        summary["engine_bundle_acknowledged"] = True
                except Exception:
                    logger.exception(
                        "Could not acknowledge accepted soundcheck bundle"
                    )
        if not bundle:
            summary["reason"] = "no_soundcheck_recommendation_bundle"
        elif not proposals:
            summary["reason"] = "no_soundcheck_recommendations"
        return summary

    def _soundcheck_main_evidence_from_audio(
        self,
        audio_evidence: Mapping[str, Any] | None,
    ) -> Dict[str, Any]:
        evidence = dict(audio_evidence or {})
        peaks = list(evidence.get("master_sample_peaks_dbfs") or [])
        true_peak = self._coerce_finite_float(evidence.get("master_true_peak_dbtp"))
        age_sec = self._coerce_finite_float(evidence.get("age_sec"))
        max_age_sec = self._coerce_finite_float(evidence.get("max_age_sec"))
        ceiling = self._coerce_finite_float(evidence.get("ceiling_dbtp"))
        now = time.time()
        captured_at = now - max(0.0, age_sec) if age_sec is not None else None
        return {
            "route_verified": evidence.get("route_verified") is True,
            "fresh": bool(
                captured_at is not None
                and max_age_sec is not None
                and 0.0 <= now - captured_at <= max_age_sec
            ),
            "signal_present": bool(peaks and max(peaks) > -90.0),
            "headroom_safe": bool(
                true_peak is not None
                and ceiling is not None
                and true_peak <= ceiling
            ),
            "master_channels": list(evidence.get("master_channels") or []),
            "measurement_source": evidence.get("measurement_source"),
            "route_proof_id": evidence.get("route_proof_id"),
            "captured_at_epoch": captured_at,
            "max_age_sec": max_age_sec,
        }

    @staticmethod
    def _soundcheck_route_fence_values(
        audio_recheck: AudioRecheckResult | None,
    ) -> Dict[str, Any]:
        if audio_recheck is None or not isinstance(
            audio_recheck.evidence,
            Mapping,
        ):
            return {}
        route = audio_recheck.evidence.get("physical_route_readback")
        if not isinstance(route, Mapping) or route.get("verified") is not True:
            return {}
        readbacks = route.get("readbacks")
        if not isinstance(readbacks, Mapping):
            return {}
        return {
            str(address): row.get("value")
            for address, row in readbacks.items()
            if str(address).strip() and isinstance(row, Mapping)
        }

    @staticmethod
    def _soundcheck_fence_values_equal(
        expected: Any,
        actual: Any,
        tolerance: float,
    ) -> bool:
        try:
            left = float(expected)
            right = float(actual)
            return bool(
                math.isfinite(left)
                and math.isfinite(right)
                and abs(left - right) <= max(0.0, float(tolerance))
            )
        except (TypeError, ValueError):
            return str(expected).strip().upper() == str(actual).strip().upper()

    @staticmethod
    def _soundcheck_dca_ids_from_tags(tags: Any) -> list[int]:
        values: set[int] = set()

        def visit(item: Any) -> None:
            if item is None:
                return
            if isinstance(item, (list, tuple, set)):
                for nested in item:
                    visit(nested)
                return
            for token in str(item).replace(",", " ").replace(";", " ").split():
                normalized = token.strip().upper()
                if normalized.startswith("#D") and normalized[2:].isdigit():
                    dca_id = int(normalized[2:])
                    if 1 <= dca_id <= 16:
                        values.add(dca_id)

        visit(tags)
        return sorted(values)

    @staticmethod
    def _soundcheck_plan_requires_source_path_proof(
        plan_payload: Mapping[str, Any],
        source_coverage: Mapping[str, Any],
    ) -> bool:
        selected_channels = {
            int(channel)
            for channel in source_coverage.get("selected_channel_ids") or []
            if isinstance(channel, int)
            and not isinstance(channel, bool)
            and 1 <= channel <= 48
        }
        if not selected_channels:
            return False
        return any(
            isinstance(operation, Mapping)
            and (
                (
                    isinstance(operation.get("channel"), int)
                    and not isinstance(operation.get("channel"), bool)
                    and operation.get("channel") in selected_channels
                )
                or str(operation.get("family_id") or "")
                in {"bus_group_dca_matrix", "master_output"}
            )
            for operation in plan_payload.get("operations") or []
        )

    async def _read_selected_source_audible_paths_two_pass(
        self,
        *,
        run_id: str,
        source_coverage: Mapping[str, Any],
        channel_ids: Sequence[int],
        max_age_sec: float,
        expected_fader_db_by_channel: Mapping[int, float] | None = None,
    ) -> tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
        """Read and bind each selected source's fader and Main-audible path.

        This is read-only evidence. It deliberately does not turn an invalid
        source path into a parent-wide blocker; normal independent children may
        still be reviewed. A coupled balance group, however, must require every
        returned row to be verified immediately before apply.
        """

        client = getattr(self, "mixer_client", None)
        query = getattr(client, "query_parameter", None)
        intents = (
            dict(source_coverage.get("per_source_fader_intent") or {})
            if isinstance(
                source_coverage.get("per_source_fader_intent"),
                Mapping,
            )
            else {}
        )
        analysis_epoch = source_coverage.get("captured_at_epoch")
        evidence: Dict[str, Dict[str, Any]] = {}
        observed_values: Dict[str, Any] = {}
        tolerances = {
            "fader": 0.05,
            "mute": 0.0,
            "main_on": 0.0,
            "main_level": 0.05,
        }

        async def physical_query(address: str) -> Mapping[str, Any]:
            if not callable(query):
                return {}
            result = await asyncio.to_thread(query, address, 0.5)
            if inspect.isawaitable(result):
                result = await result
            if not isinstance(result, Mapping):
                return {}
            confirmed = result.get("confirmed_round_trips")
            if not (
                result.get("status") == "received"
                and result.get("address") == address
                and isinstance(confirmed, int)
                and not isinstance(confirmed, bool)
                and confirmed >= 2
            ):
                return {}
            return dict(result)

        for channel in sorted(set(channel_ids)):
            intent = intents.get(str(channel))
            baseline = (
                intent.get("analysis_baseline_db")
                if isinstance(intent, Mapping)
                else None
            )
            expected_fader = (
                expected_fader_db_by_channel.get(channel)
                if isinstance(expected_fader_db_by_channel, Mapping)
                and channel in expected_fader_db_by_channel
                else baseline
            )
            analysis_main_level = (
                intent.get("main_send_level_db")
                if isinstance(intent, Mapping)
                else None
            )
            analysis_main_pre = (
                intent.get("main_send_pre")
                if isinstance(intent, Mapping)
                else None
            )
            analysis_input_routing = (
                dict(intent.get("input_routing") or {})
                if isinstance(intent, Mapping)
                and isinstance(intent.get("input_routing"), Mapping)
                else {}
            )
            input_routing_valid = all(
                analysis_input_routing.get(key) is not None
                for key in (
                    "main_group",
                    "main_channel",
                    "alt_group",
                    "alt_channel",
                )
            )
            raw_dca_ids = (
                list(intent.get("assigned_dca_ids") or [])
                if isinstance(intent, Mapping)
                else []
            )
            assigned_dca_ids = sorted(
                {
                    int(dca_id)
                    for dca_id in raw_dca_ids
                    if isinstance(dca_id, int)
                    and not isinstance(dca_id, bool)
                    and 1 <= dca_id <= 16
                }
            )
            assigned_dca_states = (
                dict(intent.get("assigned_dca_states") or {})
                if isinstance(intent, Mapping)
                and isinstance(intent.get("assigned_dca_states"), Mapping)
                else {}
            )
            dca_analysis_valid = bool(
                len(assigned_dca_ids) == len(raw_dca_ids)
                and all(
                    isinstance(assigned_dca_states.get(str(dca_id)), Mapping)
                    and assigned_dca_states[str(dca_id)].get("muted") is False
                    and assigned_dca_states[str(dca_id)].get("locally_enabled") is True
                    and isinstance(
                        assigned_dca_states[str(dca_id)].get("fader_db"),
                        (int, float),
                    )
                    and not isinstance(
                        assigned_dca_states[str(dca_id)].get("fader_db"),
                        bool,
                    )
                    and math.isfinite(
                        float(assigned_dca_states[str(dca_id)]["fader_db"])
                    )
                    and float(assigned_dca_states[str(dca_id)]["fader_db"]) > -90.0
                    for dca_id in assigned_dca_ids
                )
            )
            intent_valid = bool(
                isinstance(intent, Mapping)
                and intent.get("mode") in {"absolute_target", "hold_current"}
                and intent.get("signal_verified") is True
                and intent.get("foundation_analyzed") is True
                and intent.get("channel_main_path_locally_enabled") is True
                and intent.get("downstream_dca_path_verified") is True
                and intent.get("analyzed_source_path_enabled") is True
                and intent.get("main_send_on") is True
                and analysis_main_pre is False
                and input_routing_valid
                and intent.get("analysis_run_id") == run_id
                and intent.get("analysis_captured_at_epoch") == analysis_epoch
                and isinstance(baseline, (int, float))
                and not isinstance(baseline, bool)
                and math.isfinite(float(baseline))
                and -90.0 < float(baseline) <= 0.0
                and isinstance(analysis_main_level, (int, float))
                and not isinstance(analysis_main_level, bool)
                and math.isfinite(float(analysis_main_level))
                and float(analysis_main_level) > -90.0
                and dca_analysis_valid
            )
            row: Dict[str, Any] = {
                "channel": channel,
                "analysis_baseline_db": (
                    float(baseline)
                    if isinstance(baseline, (int, float))
                    and not isinstance(baseline, bool)
                    else None
                ),
                "expected_current_fader_db": (
                    float(expected_fader)
                    if isinstance(expected_fader, (int, float))
                    and not isinstance(expected_fader, bool)
                    else None
                ),
                "analysis_main_send_level_db": (
                    float(analysis_main_level)
                    if isinstance(analysis_main_level, (int, float))
                    and not isinstance(analysis_main_level, bool)
                    else None
                ),
                "physical_verified": False,
                "two_pass_consistent": False,
                "matches_analysis_baseline": False,
                "matches_analysis_main_send": False,
                "channel_main_path_locally_enabled": False,
                "downstream_dca_path_verified": False,
                "captured_at_epoch": time.time(),
                "max_age_sec": max(0.0, float(max_age_sec)),
            }
            if not intent_valid:
                row["reason"] = "selected_source_analysis_intent_unverified"
                evidence[str(channel)] = row
                continue

            addresses = {
                "fader": f"/ch/{channel}/fdr",
                "mute": f"/ch/{channel}/mute",
                "main_on": f"/ch/{channel}/main/1/on",
                "main_level": f"/ch/{channel}/main/1/lvl",
                "main_pre": f"/ch/{channel}/main/1/pre",
                "input_main_group": f"/ch/{channel}/in/conn/grp",
                "input_main_channel": f"/ch/{channel}/in/conn/in",
                "input_alt_group": f"/ch/{channel}/in/conn/altgrp",
                "input_alt_channel": f"/ch/{channel}/in/conn/altin",
                "tags": f"/ch/{channel}/tags",
            }
            for dca_id in assigned_dca_ids:
                addresses[f"dca_{dca_id}_mute"] = f"/dca/{dca_id}/mute"
                addresses[f"dca_{dca_id}_fader"] = f"/dca/{dca_id}/fdr"
            passes: list[Dict[str, Any]] = []
            query_failed = False
            for _pass in range(2):
                values: Dict[str, Any] = {}
                address_items = list(addresses.items())
                results = await asyncio.gather(
                    *(
                        physical_query(address)
                        for _key, address in address_items
                    )
                )
                for (key, _address), result in zip(address_items, results):
                    if "value" not in result or result.get("value") is None:
                        query_failed = True
                        break
                    values[key] = result.get("value")
                if query_failed:
                    break
                passes.append(values)
            row["captured_at_epoch"] = time.time()
            if query_failed or len(passes) != 2:
                row["reason"] = "selected_source_audible_path_readback_failed"
                evidence[str(channel)] = row
                continue

            first, second = passes
            consistent = all(
                self._soundcheck_fence_values_equal(
                    first[key],
                    second[key],
                    (
                        0.0
                        if key.endswith("_mute")
                        else 0.05
                        if key.endswith("_fader")
                        else tolerances.get(key, 0.0)
                    ),
                )
                for key in addresses
            )
            fader = self._coerce_finite_float(second.get("fader"))
            main_level = self._coerce_finite_float(second.get("main_level"))
            matches_baseline = bool(
                consistent
                and fader is not None
                and isinstance(expected_fader, (int, float))
                and not isinstance(expected_fader, bool)
                and self._soundcheck_fence_values_equal(
                    fader,
                    expected_fader,
                    0.25,
                )
            )
            matches_main_send = bool(
                consistent
                and main_level is not None
                and self._soundcheck_fence_values_equal(
                    main_level,
                    analysis_main_level,
                    0.25,
                )
            )
            unmuted = self._soundcheck_fence_values_equal(
                second.get("mute"),
                0,
                0.0,
            )
            main_on = self._soundcheck_fence_values_equal(
                second.get("main_on"),
                1,
                0.0,
            )
            main_post_fader = self._soundcheck_fence_values_equal(
                second.get("main_pre"),
                0,
                0.0,
            )
            input_routing_matches = all(
                self._soundcheck_fence_values_equal(
                    second.get(f"input_{key}"),
                    analysis_input_routing.get(key),
                    0.0,
                )
                for key in (
                    "main_group",
                    "main_channel",
                    "alt_group",
                    "alt_channel",
                )
            )
            physical_dca_ids = self._soundcheck_dca_ids_from_tags(
                second.get("tags")
            )
            dca_topology_matches = physical_dca_ids == assigned_dca_ids
            dca_path_verified = True
            for dca_id in assigned_dca_ids:
                analysis_state = assigned_dca_states[str(dca_id)]
                dca_muted = not self._soundcheck_fence_values_equal(
                    second.get(f"dca_{dca_id}_mute"),
                    0,
                    0.0,
                )
                dca_fader = self._coerce_finite_float(
                    second.get(f"dca_{dca_id}_fader")
                )
                dca_path_verified = bool(
                    dca_path_verified
                    and not dca_muted
                    and dca_fader is not None
                    and dca_fader > -90.0
                    and self._soundcheck_fence_values_equal(
                        dca_fader,
                        analysis_state.get("fader_db"),
                        0.25,
                    )
                )
            audible = bool(
                consistent
                and fader is not None
                and -90.0 < fader <= 0.0
                and unmuted
                and main_on
                and main_level is not None
                and main_level > -90.0
                and main_post_fader
                and input_routing_matches
                and matches_baseline
                and matches_main_send
                and dca_topology_matches
                and dca_path_verified
            )
            row.update(
                {
                    "physical_verified": True,
                    "two_pass_consistent": consistent,
                    "matches_analysis_baseline": matches_baseline,
                    "matches_analysis_main_send": matches_main_send,
                    "channel_main_path_locally_enabled": bool(
                        unmuted
                        and main_on
                        and fader is not None
                        and -90.0 < fader <= 0.0
                        and main_level is not None
                        and main_level > -90.0
                        and main_post_fader
                        and input_routing_matches
                        and matches_baseline
                        and matches_main_send
                    ),
                    "downstream_dca_path_verified": bool(
                        dca_topology_matches and dca_path_verified
                    ),
                    "assigned_dca_ids": assigned_dca_ids,
                    "physical_dca_ids": physical_dca_ids,
                    "dca_topology_matches_analysis": dca_topology_matches,
                    "current_fader_db": fader,
                    "current_muted": not unmuted,
                    "current_main_send_on": main_on,
                    "current_main_send_level_db": main_level,
                    "current_main_send_pre": not main_post_fader,
                    "input_routing_matches_analysis": input_routing_matches,
                    "addresses": addresses,
                    "observed_values": {
                        addresses[key]: second[key] for key in addresses
                    },
                    "source": "fresh_physical_selected_source_path_readback",
                }
            )
            if not consistent:
                row["reason"] = "selected_source_audible_path_changed_during_readback"
            elif not matches_baseline:
                row["reason"] = "selected_source_fader_baseline_mismatch"
            elif not unmuted:
                row["reason"] = "selected_source_muted_after_analysis"
            elif not main_on:
                row["reason"] = "selected_source_main_send_off_after_analysis"
            elif main_level is None or main_level <= -90.0:
                row["reason"] = "selected_source_main_send_inactive_after_analysis"
            elif not main_post_fader:
                row["reason"] = "selected_source_main_send_pre_fader"
            elif not input_routing_matches:
                row["reason"] = "selected_source_input_routing_changed_after_analysis"
            elif not matches_main_send:
                row["reason"] = "selected_source_main_send_level_changed_after_analysis"
            elif not dca_topology_matches:
                row["reason"] = "selected_source_dca_topology_changed_after_analysis"
            elif not dca_path_verified:
                row["reason"] = "selected_source_dca_path_changed_or_inactive"
            else:
                row["reason"] = "selected_source_path_verified"
            evidence[str(channel)] = row
            if audible:
                observed_values.update(row["observed_values"])

        return evidence, observed_values

    async def _establish_soundcheck_console_change_fence(
        self,
        operations: Sequence[ConsoleTransactionOperation],
        snapshots: Mapping[str, Mapping[str, Any]],
        adapters: Mapping[str, Any],
        *,
        additional_observed_values: Mapping[str, Any] | None = None,
        capture_status: Mapping[str, Any] | None = None,
        require_global_generation: bool = False,
        quiet_window_sec: float = 0.1,
    ) -> Dict[str, Any]:
        """Bind sequential readback to WING's observed value-change stream."""

        client = getattr(self, "mixer_client", None)
        if client is None or getattr(client, "is_connected", False) is not True:
            return {
                "verified": False,
                "reason": "mixer_not_connected",
            }
        observer = getattr(
            client,
            "get_observed_state_revision",
            None,
        )
        if not callable(observer):
            return {
                "verified": False,
                "reason": "console_change_observer_unavailable",
            }
        address_specs: Dict[str, Dict[str, Any]] = {}
        for operation in operations:
            adapter = adapters.get(operation.family_id)
            if adapter is None:
                return {
                    "verified": False,
                    "reason": f"family_adapter_unavailable:{operation.family_id}",
                }
            prepared = adapter.prepare(operation)
            snapshot_values = dict(
                snapshots.get(operation.operation_id, {}).get("values") or {}
            )
            for spec in (
                *list(prepared.get("writes") or []),
                *list(prepared.get("audible_state_reads") or []),
            ):
                address = str(spec.get("address") or "")
                if not address or address not in snapshot_values:
                    return {
                        "verified": False,
                        "reason": "console_change_fence_snapshot_incomplete",
                        "operation_id": operation.operation_id,
                        "address": address,
                    }
                tolerance = float(spec.get("tolerance") or 0.0)
                previous = address_specs.get(address)
                address_specs[address] = {
                    "expected_value": snapshot_values[address],
                    "tolerance": (
                        min(float(previous["tolerance"]), tolerance)
                        if previous is not None
                        else tolerance
                    ),
                }
        for address, value in dict(additional_observed_values or {}).items():
            address = str(address or "")
            if address:
                address_specs.setdefault(
                    address,
                    {"expected_value": value, "tolerance": 0.0},
                )
        addresses = sorted(address_specs)
        if not addresses and not require_global_generation:
            return {
                "verified": False,
                "reason": "console_change_fence_has_no_addresses",
            }

        async def checkpoint() -> Mapping[str, Any]:
            value = observer(addresses)
            value = await self._maybe_await(value)
            return value if isinstance(value, Mapping) else {}

        first = await checkpoint()
        await asyncio.sleep(max(0.0, float(quiet_window_sec)))
        second = await checkpoint()
        first_rows = first.get("addresses") if isinstance(first, Mapping) else None
        second_rows = second.get("addresses") if isinstance(second, Mapping) else None
        if not isinstance(first_rows, Mapping) or not isinstance(second_rows, Mapping):
            return {
                "verified": False,
                "reason": "console_change_observer_contract_invalid",
            }
        changed_addresses = []
        mismatched_addresses = []
        address_generations: Dict[str, int] = {}
        observed_values: Dict[str, Any] = {}
        for address, spec in address_specs.items():
            first_row = first_rows.get(address)
            second_row = second_rows.get(address)
            if not isinstance(first_row, Mapping) or not isinstance(second_row, Mapping):
                mismatched_addresses.append(address)
                continue
            first_generation = first_row.get("generation")
            second_generation = second_row.get("generation")
            if (
                isinstance(first_generation, bool)
                or not isinstance(first_generation, int)
                or isinstance(second_generation, bool)
                or not isinstance(second_generation, int)
            ):
                mismatched_addresses.append(address)
                continue
            if first_generation != second_generation:
                changed_addresses.append(address)
            second_value = second_row.get("value")
            if not self._soundcheck_fence_values_equal(
                spec["expected_value"],
                second_value,
                float(spec["tolerance"]),
            ):
                mismatched_addresses.append(address)
            address_generations[address] = second_generation
            observed_values[address] = second_value
        first_global_generation = first.get("generation")
        second_global_generation = second.get("generation")
        global_generation_stable = bool(
            not require_global_generation
            or (
                isinstance(first_global_generation, int)
                and not isinstance(first_global_generation, bool)
                and isinstance(second_global_generation, int)
                and not isinstance(second_global_generation, bool)
                and first_global_generation == second_global_generation
            )
        )
        verified = bool(
            not changed_addresses
            and not mismatched_addresses
            and global_generation_stable
        )
        payload = {
            "verified": verified,
            "reason": (
                "console_change_fence_verified"
                if verified
                else "console_changed_during_readiness_fence"
            ),
            "addresses": addresses,
            "address_generations": address_generations,
            "observed_values": observed_values,
            "changed_addresses": sorted(set(changed_addresses)),
            "mismatched_addresses": sorted(set(mismatched_addresses)),
            "quiet_window_sec": max(0.0, float(quiet_window_sec)),
            "captured_at_epoch": second.get("captured_at_epoch"),
            "consistency_model": "observed_value_change_generation_fence",
            "valid_until": "next_observed_relevant_console_value_change",
            "global_generation_required": bool(require_global_generation),
            "global_generation": second_global_generation,
            "capture_identity": {
                key: capture_status.get(key)
                for key in (
                    "device_name",
                    "device_index",
                    "num_channels",
                    "sample_rate",
                    "source_type",
                )
                if isinstance(capture_status, Mapping)
            },
        }
        payload["fence_id"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()
        return payload

    def _soundcheck_capture_fence_still_current(
        self,
        fence: Mapping[str, Any] | None,
    ) -> bool:
        if not isinstance(fence, Mapping):
            return False
        expected = dict(fence.get("capture_identity") or {})
        if not expected:
            return True
        candidates = [
            getattr(self, "audio_capture", None),
            getattr(
                getattr(self, "auto_soundcheck_engine", None),
                "audio_capture",
                None,
            ),
        ]
        seen: set[int] = set()
        for capture in candidates:
            if capture is None or id(capture) in seen:
                continue
            seen.add(id(capture))
            if not getattr(capture, "running", False):
                continue
            get_status = getattr(capture, "get_status", None)
            if not callable(get_status):
                continue
            try:
                status = get_status()
            except Exception:
                continue
            callback_age = (
                status.get("callback_age_sec")
                if isinstance(status, Mapping)
                else None
            )
            if (
                isinstance(status, Mapping)
                and status.get("running") is True
                and status.get("real_input") is True
                and isinstance(callback_age, (int, float))
                and not isinstance(callback_age, bool)
                and 0.0 <= float(callback_age) <= 1.0
                and all(status.get(key) == value for key, value in expected.items())
            ):
                return True
        return False

    def _current_active_soundcheck_run_id(self) -> str:
        active = str(
            getattr(self, "_active_soundcheck_run_id", "") or ""
        ).strip()
        manifests = getattr(self, "_soundcheck_manifests", {})
        if active and active in manifests:
            return active
        readiness_run = str(
            (
                getattr(self, "last_soundcheck_readiness", {}) or {}
            ).get("run_id")
            or ""
        ).strip()
        if readiness_run and readiness_run in manifests:
            return readiness_run
        return str(next(reversed(manifests), "") or "").strip()

    def _retire_soundcheck_console_change_invalidators_except(
        self,
        active_run_id: str | None,
    ) -> None:
        subscriptions = getattr(
            self,
            "_soundcheck_console_change_subscriptions",
            {},
        )
        if not isinstance(subscriptions, dict):
            return
        for inactive_run_id, inactive in list(subscriptions.items()):
            if active_run_id is not None and inactive_run_id == active_run_id:
                continue
            if isinstance(inactive, tuple) and len(inactive) == 2:
                inactive_client, inactive_callback = inactive
                callbacks = getattr(inactive_client, "callbacks", {}).get("*")
                if isinstance(callbacks, list) and inactive_callback in callbacks:
                    callbacks.remove(inactive_callback)
            subscriptions.pop(inactive_run_id, None)

    def _soundcheck_console_fence_still_current(
        self,
        fence: Mapping[str, Any] | None,
        *,
        ignored_addresses: Iterable[str] = (),
        expected_mutations: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> bool:
        if not isinstance(fence, Mapping) or fence.get("verified") is not True:
            return False
        client = getattr(self, "mixer_client", None)
        if client is None or getattr(client, "is_connected", False) is not True:
            return False
        observer = getattr(
            client,
            "get_observed_state_revision",
            None,
        )
        if not callable(observer):
            return False
        current = observer(list(fence.get("addresses") or []))
        if not isinstance(current, Mapping):
            return False
        rows = current.get("addresses")
        if not isinstance(rows, Mapping):
            return False
        generations = dict(fence.get("address_generations") or {})
        values = dict(fence.get("observed_values") or {})
        ignored = {str(address) for address in ignored_addresses}
        mutation_expectations = dict(expected_mutations or {})
        if not self._soundcheck_capture_fence_still_current(fence):
            return False
        if (
            fence.get("global_generation_required") is True
            and not ignored
            and not mutation_expectations
        ):
            if current.get("generation") != fence.get("global_generation"):
                return False
        return all(
            address in ignored
            or (
                address in mutation_expectations
                and isinstance(rows.get(address), Mapping)
                and (
                    (
                        rows[address].get("generation") == generation
                        and rows[address].get("value") == values.get(address)
                    )
                    or self._soundcheck_fence_values_equal(
                        rows[address].get("value"),
                        mutation_expectations[address].get("target_value"),
                        float(
                            mutation_expectations[address].get("tolerance")
                            or 0.0
                        ),
                    )
                )
            )
            or (
            isinstance(rows.get(address), Mapping)
            and rows[address].get("generation") == generation
            and rows[address].get("value") == values.get(address)
            )
            for address, generation in generations.items()
        )

    def _invalidate_soundcheck_console_fence(
        self,
        run_id: str,
        reason: str = "console_state_changed_after_reconciliation",
    ) -> None:
        active_run_id = self._current_active_soundcheck_run_id()
        if active_run_id and run_id != active_run_id:
            blockers = self._soundcheck_global_blockers.setdefault(run_id, [])
            if "soundcheck_run_superseded" not in blockers:
                blockers.append("soundcheck_run_superseded")
            return
        blockers = self._soundcheck_global_blockers.setdefault(run_id, [])
        if reason not in blockers:
            blockers.append(reason)
        for outcome in self._soundcheck_outcomes.get(run_id, {}).values():
            if isinstance(outcome, dict):
                outcome["final_reconciliation_verified"] = False
                outcome["final_reconciliation_reason"] = reason
        readiness = self._aggregate_current_soundcheck_readiness(run_id)
        if isinstance(readiness, Mapping):
            self.last_soundcheck_readiness = dict(readiness)

    def _arm_soundcheck_console_change_invalidator(
        self,
        run_id: str,
        fence: Mapping[str, Any],
    ) -> bool:
        active_run_id = self._current_active_soundcheck_run_id()
        if active_run_id and run_id != active_run_id:
            return False
        client = getattr(self, "mixer_client", None)
        subscribe = getattr(client, "subscribe", None)
        if not callable(subscribe):
            return False
        subscriptions = getattr(
            self,
            "_soundcheck_console_change_subscriptions",
            None,
        )
        if not isinstance(subscriptions, dict):
            self._soundcheck_console_change_subscriptions = {}
            subscriptions = self._soundcheck_console_change_subscriptions
        self._retire_soundcheck_console_change_invalidators_except(run_id)
        previous = subscriptions.get(run_id)
        if isinstance(previous, tuple) and len(previous) == 2:
            previous_client, previous_callback = previous
            callbacks = getattr(previous_client, "callbacks", {}).get("*")
            if isinstance(callbacks, list) and previous_callback in callbacks:
                callbacks.remove(previous_callback)
        relevant = set(fence.get("addresses") or [])
        global_generation_required = (
            fence.get("global_generation_required") is True
        )
        try:
            event_loop = asyncio.get_running_loop()
        except RuntimeError:
            event_loop = None

        def on_console_update(address: str, *_args: Any) -> None:
            active_run_id = self._current_active_soundcheck_run_id()
            if active_run_id and run_id != active_run_id:
                return
            current_readiness = getattr(
                self,
                "last_soundcheck_readiness",
                {},
            )
            if (
                isinstance(current_readiness, Mapping)
                and str(current_readiness.get("run_id") or "")
                and str(current_readiness.get("run_id") or "") != run_id
            ):
                return
            if not global_generation_required and address not in relevant:
                return
            expected_mutations = getattr(
                self,
                "_soundcheck_expected_mutation_addresses",
                {},
            )
            run_expectations = expected_mutations.get(run_id, {})
            if isinstance(run_expectations, Mapping) and address in run_expectations:
                observed_value = _args[2] if len(_args) >= 3 else (
                    _args[0] if len(_args) == 1 else None
                )
                expectation = run_expectations[address]
                if self._soundcheck_fence_values_equal(
                    observed_value,
                    expectation.get("target_value"),
                    float(expectation.get("tolerance") or 0.0),
                ):
                    return
            if not self._soundcheck_console_fence_still_current(fence):
                self._invalidate_soundcheck_console_fence(run_id)
                broadcast = getattr(self, "broadcast", None)
                if event_loop is not None and callable(broadcast):
                    payload = {
                        "type": "auto_soundcheck_status",
                        "status": "readiness_invalidated",
                        "running": bool(
                            getattr(self, "auto_soundcheck_running", False)
                        ),
                        "is_running": bool(
                            getattr(self, "auto_soundcheck_running", False)
                        ),
                        "message": (
                            "Soundcheck readiness invalidated by a physical "
                            f"console change at {address}."
                        ),
                        "soundcheck_readiness": dict(
                            self.last_soundcheck_readiness or {}
                        ),
                    }
                    try:
                        running = asyncio.get_running_loop()
                    except RuntimeError:
                        running = None
                    if running is event_loop:
                        event_loop.create_task(broadcast(payload))
                    elif event_loop.is_running():
                        asyncio.run_coroutine_threadsafe(
                            broadcast(payload),
                            event_loop,
                        )

        subscribe("*", on_console_update)
        subscriptions[run_id] = (client, on_console_update)
        if not self._soundcheck_console_fence_still_current(fence):
            self._invalidate_soundcheck_console_fence(run_id)
            return False
        return True

    @staticmethod
    def _soundcheck_refreshable_global_blocker(reason: str) -> bool:
        return bool(
            reason in {
                "mixer_not_connected",
                "physical_readback_transport_unavailable",
                "emergency_stop_active",
                "soundcheck_physical_preflight_missing",
                "soundcheck_preflight_collection_stale",
                "soundcheck_final_reconciliation_collection_stale",
            }
            or reason.startswith("physical_snapshot_")
            or reason.startswith("soundcheck_preflight_consistency_mismatch:")
            or reason.startswith("soundcheck_preflight_change_fence_failed:")
            or reason.startswith("family_adapter_unavailable:")
            or reason.startswith("master_")
            or reason.startswith("stereo_master_")
            or reason.startswith("feedback_state_")
            or reason.startswith("soundcheck_final_reconciliation_")
            or reason == "console_state_changed_after_reconciliation"
        )

    async def preflight_soundcheck_manifest(
        self,
        manifest: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Collect fresh read-only snapshots and classify the whole parent plan."""
        preflight_started_at = time.time()
        preflight_max_age_sec = 30.0
        run_id = str(manifest.get("run_id") or "").strip()
        global_blockers: List[str] = []
        if not run_id:
            global_blockers.append("invalid_soundcheck_manifest_identity")
        if self._manual_write_emergency_stopped:
            global_blockers.append("emergency_stop_active")
        if not self.mixer_client or not getattr(self.mixer_client, "is_connected", False):
            global_blockers.append("mixer_not_connected")
        elif not callable(getattr(self.mixer_client, "query_parameter", None)):
            global_blockers.append("physical_readback_transport_unavailable")

        operations: List[ConsoleTransactionOperation] = []
        for raw_plan in manifest.get("plans") or []:
            if not isinstance(raw_plan, Mapping):
                global_blockers.append("invalid_soundcheck_child_plan")
                continue
            try:
                parsed = self._parse_supervised_transaction_plan(raw_plan)
            except (TypeError, ValueError) as exc:
                global_blockers.append(str(exc))
                continue
            operations.extend(parsed.operations)

        adapters = (
            build_wing_supervised_family_adapters(
                self.mixer_client,
                audio_recheck=self._supervised_transaction_audio_recheck,
            )
            if not global_blockers
            else {}
        )
        first_snapshots: Dict[str, Mapping[str, Any]] = {}
        if not global_blockers:
            for operation in operations:
                adapter = adapters.get(operation.family_id)
                if adapter is None:
                    global_blockers.append(
                        f"family_adapter_unavailable:{operation.family_id}"
                    )
                    break
                try:
                    snapshot = dict(await adapter.snapshot(operation))
                    snapshot["captured_at_epoch"] = time.time()
                    first_snapshots[operation.operation_id] = snapshot
                except asyncio.CancelledError:
                    raise
                except Exception:
                    first_snapshots[operation.operation_id] = {
                        "physical_verified": False,
                        "values": {},
                    }
                    global_blockers.append(
                        f"physical_snapshot_failed:{operation.operation_id}"
                    )
                    break
        snapshots: Dict[str, Mapping[str, Any]] = {}
        if not global_blockers:
            for operation in operations:
                adapter = adapters[operation.family_id]
                try:
                    snapshot = dict(await adapter.snapshot(operation))
                    snapshot["captured_at_epoch"] = time.time()
                    snapshots[operation.operation_id] = snapshot
                except asyncio.CancelledError:
                    raise
                except Exception:
                    snapshots[operation.operation_id] = {
                        "physical_verified": False,
                        "values": {},
                    }
                    global_blockers.append(
                        f"physical_snapshot_failed:{operation.operation_id}"
                    )
                    break
        if not global_blockers:
            for operation in operations:
                prepared = adapters[operation.family_id].prepare(operation)
                address_specs = [
                    *list(prepared.get("writes") or []),
                    *list(prepared.get("audible_state_reads") or []),
                ]
                first_values = dict(
                    first_snapshots.get(operation.operation_id, {}).get("values")
                    or {}
                )
                second_values = dict(
                    snapshots.get(operation.operation_id, {}).get("values")
                    or {}
                )
                for spec in address_specs:
                    address = str(spec.get("address") or "")
                    tolerance = float(spec.get("tolerance") or 0.0)
                    left = first_values.get(address)
                    right = second_values.get(address)
                    try:
                        equal = bool(
                            math.isfinite(float(left))
                            and math.isfinite(float(right))
                            and abs(float(left) - float(right)) <= tolerance
                        )
                    except (TypeError, ValueError):
                        equal = (
                            str(left).strip().upper()
                            == str(right).strip().upper()
                        )
                    if (
                        not address
                        or address not in first_values
                        or address not in second_values
                        or not equal
                    ):
                        global_blockers.append(
                            "soundcheck_preflight_consistency_mismatch:"
                            + operation.operation_id
                        )
                        break

        selected_source_fader_holds: Dict[str, Dict[str, Any]] = {}
        source_coverage = (
            dict(manifest.get("source_coverage") or {})
            if isinstance(manifest.get("source_coverage"), Mapping)
            else {}
        )
        selected_channels = {
            channel
            for channel in source_coverage.get("selected_channel_ids") or []
            if isinstance(channel, int)
            and not isinstance(channel, bool)
            and 1 <= channel <= 48
        }
        planned_fader_channels = {
            operation.channel
            for operation in operations
            if operation.family_id == "fader"
            and operation.write_kind == "fader"
            and operation.channel is not None
        }
        hold_channels = sorted(selected_channels - planned_fader_channels)
        fader_intents = (
            dict(source_coverage.get("per_source_fader_intent") or {})
            if isinstance(
                source_coverage.get("per_source_fader_intent"),
                Mapping,
            )
            else {}
        )
        selected_source_audible_paths: Dict[str, Dict[str, Any]] = {}
        if (
            not global_blockers
            and selected_channels
            and fader_intents
        ):
            selected_source_audible_paths, _path_values = (
                await self._read_selected_source_audible_paths_two_pass(
                    run_id=run_id,
                    source_coverage=source_coverage,
                    channel_ids=sorted(selected_channels),
                    max_age_sec=preflight_max_age_sec,
                )
            )
        if not global_blockers and hold_channels:
            for channel in hold_channels:
                intent = fader_intents.get(str(channel))
                baseline = (
                    intent.get("analysis_baseline_db")
                    if isinstance(intent, Mapping)
                    else None
                )
                if not (
                    isinstance(intent, Mapping)
                    and intent.get("mode") == "hold_current"
                    and intent.get("signal_verified") is True
                    and intent.get("foundation_analyzed") is True
                    and intent.get("channel_main_path_locally_enabled") is True
                    and intent.get("downstream_dca_path_verified") is True
                    and intent.get("analyzed_source_path_enabled") is True
                    and intent.get("analysis_run_id") == run_id
                    and intent.get("analysis_captured_at_epoch")
                    == source_coverage.get("captured_at_epoch")
                    and isinstance(baseline, (int, float))
                    and not isinstance(baseline, bool)
                    and math.isfinite(float(baseline))
                ):
                    selected_source_fader_holds[str(channel)] = {
                        "channel": channel,
                        "physical_verified": False,
                        "two_pass_consistent": False,
                        "matches_analysis_baseline": False,
                        "reason": "hold_fader_analysis_intent_unverified",
                    }
                    continue
                path = selected_source_audible_paths.get(str(channel), {})
                selected_source_fader_holds[str(channel)] = {
                    "channel": channel,
                    "address": f"/ch/{channel}/fdr",
                    "current_fader_db": path.get("current_fader_db"),
                    "physical_verified": path.get("physical_verified") is True,
                    "two_pass_consistent": path.get("two_pass_consistent") is True,
                    "matches_analysis_baseline": (
                        path.get("matches_analysis_baseline") is True
                    ),
                    "analysis_baseline_db": float(baseline),
                    "captured_at_epoch": path.get("captured_at_epoch"),
                    "max_age_sec": path.get("max_age_sec"),
                    "source": "fresh_physical_hold_current_readback",
                    "reason": path.get("reason"),
                }

        audio_preflight = None
        if not global_blockers and operations:
            audio_preflight = await self._supervised_transaction_audio_recheck(
                operations[0],
                snapshots.get(operations[0].operation_id, {}),
                {},
            )
            if not audio_preflight.passed:
                global_blockers.append(
                    str(audio_preflight.reason or "master_audio_preflight_failed")
                )
        if time.time() - preflight_started_at > preflight_max_age_sec:
            global_blockers.append("soundcheck_preflight_collection_stale")

        assessment = assess_soundcheck_manifest(
            manifest,
            snapshots_by_operation=snapshots,
            adapters=adapters,
            global_blockers=global_blockers,
        )
        audio_evidence = (
            dict(audio_preflight.evidence)
            if audio_preflight is not None
            and isinstance(audio_preflight.evidence, Mapping)
            else {}
        )
        assessment["source_coverage"] = dict(
            manifest.get("source_coverage") or {}
        )
        assessment["selected_source_fader_holds"] = (
            selected_source_fader_holds
        )
        assessment["selected_source_audible_paths"] = (
            selected_source_audible_paths
        )
        assessment["main_47_48_evidence"] = (
            self._soundcheck_main_evidence_from_audio(audio_evidence)
        )
        for item in assessment.get("assessments") or []:
            if not isinstance(item, dict):
                continue
            snapshot = snapshots.get(str(item.get("operation_id") or ""), {})
            item["captured_at_epoch"] = snapshot.get(
                "captured_at_epoch",
                preflight_started_at,
            )
            item["max_age_sec"] = preflight_max_age_sec
        replan_draft = build_soundcheck_replan_draft(
            manifest,
            assessment,
            snapshots_by_operation=snapshots,
            adapters=adapters,
        )
        assessment["replan"] = replan_draft
        if run_id:
            assessment["captured_at_epoch"] = preflight_started_at
            assessment["completed_at_epoch"] = time.time()
            assessment["max_age_sec"] = preflight_max_age_sec
            self._soundcheck_preflight_assessments[run_id] = assessment
            self._soundcheck_preflight_timestamps[run_id] = preflight_started_at
            replan_store = getattr(self, "_soundcheck_replan_drafts", None)
            if not isinstance(replan_store, dict):
                self._soundcheck_replan_drafts = {}
                replan_store = self._soundcheck_replan_drafts
            if replan_draft.get("draft_available") is True:
                replan_store[run_id] = dict(replan_draft)
            else:
                replan_store.pop(run_id, None)
        return assessment

    async def accept_soundcheck_replan_draft(
        self,
        *,
        run_id: Any,
        draft_id: Any,
        approved: bool,
        approval_id: Any,
    ) -> Dict[str, Any]:
        """Promote one server-generated no-write draft to a new review revision."""

        run_id = str(run_id or "").strip()
        draft_id = str(draft_id or "").strip()
        approval_id = str(approval_id or "").strip()
        active_run_id = self._current_active_soundcheck_run_id()
        if active_run_id and run_id != active_run_id:
            return {
                "type": "soundcheck_replan_result",
                "status": "blocked",
                "success": False,
                "reason": "soundcheck_run_superseded",
                "run_id": run_id or None,
                "active_run_id": active_run_id,
                "draft_id": draft_id or None,
            }
        if approved is not True or not approval_id:
            return {
                "type": "soundcheck_replan_result",
                "status": "blocked",
                "success": False,
                "reason": "operator_approval_required",
                "run_id": run_id or None,
                "draft_id": draft_id or None,
            }
        draft = getattr(self, "_soundcheck_replan_drafts", {}).get(run_id)
        current = self._soundcheck_manifests.get(run_id)
        revised = draft.get("revised_manifest") if isinstance(draft, Mapping) else None
        if not (
            isinstance(draft, Mapping)
            and draft.get("draft_id") == draft_id
            and isinstance(current, Mapping)
            and draft.get("from_plan_revision") == current.get("plan_revision")
            and isinstance(revised, Mapping)
            and revised.get("plan_revision") == current.get("plan_revision") + 1
        ):
            return {
                "type": "soundcheck_replan_result",
                "status": "blocked",
                "success": False,
                "reason": "stale_or_unknown_soundcheck_replan_draft",
                "run_id": run_id or None,
                "draft_id": draft_id or None,
            }

        preflight = await self.preflight_soundcheck_manifest(revised)
        active_run_id = self._current_active_soundcheck_run_id()
        if active_run_id and run_id != active_run_id:
            return {
                "type": "soundcheck_replan_result",
                "status": "blocked",
                "success": False,
                "reason": "soundcheck_run_superseded",
                "run_id": run_id,
                "active_run_id": active_run_id,
                "draft_id": draft_id,
            }
        by_operation = {
            str(item.get("operation_id") or ""): dict(item)
            for item in preflight.get("assessments") or []
            if isinstance(item, Mapping)
        }
        accepted_manifest = copy.deepcopy(revised)
        accepted_manifest["plans"] = [
            {
                **dict(plan),
                "preflight_assessment": plan_preflight_assessment(
                    plan,
                    by_operation,
                ),
            }
            for plan in revised.get("plans") or []
            if isinstance(plan, Mapping)
        ]
        accepted_manifest["normal_plans"] = [
            plan
            for plan in accepted_manifest["plans"]
            if plan.get("separate_approval_required") is not True
        ]
        accepted_manifest["separate_approval_plans"] = [
            plan
            for plan in accepted_manifest["plans"]
            if plan.get("separate_approval_required") is True
        ]
        bind_soundcheck_manifest_content_sha256(accepted_manifest)
        summary = self.import_soundcheck_recommendations_to_operator_queue(
            {
                "proposals": [],
                "supervised_transaction_plans": accepted_manifest,
                "soundcheck_preflight": preflight,
            },
            trusted_engine_bundle=True,
        )
        finalized = None
        if run_id in getattr(self, "_soundcheck_preflight_assessments", {}):
            finalized = await self.finalize_soundcheck_readiness_if_no_apply(run_id)
        if isinstance(finalized, Mapping):
            summary["soundcheck_readiness"] = dict(finalized)
        return {
            **summary,
            "type": "soundcheck_replan_result",
            "run_id": run_id,
            "draft_id": draft_id,
            "approval_id": approval_id,
            "from_plan_revision": draft.get("from_plan_revision"),
            "to_plan_revision": draft.get("to_plan_revision"),
            "new_operator_approval_required_for_each_child": True,
            "live_authority_changed": False,
        }

    async def reanalyze_soundcheck_stage(
        self,
        *,
        run_id: Any,
        approved: bool,
        approval_id: Any,
    ) -> Dict[str, Any]:
        """Prepare and durably publish one read-only stage-analysis revision."""

        run_id = str(run_id or "").strip()
        approval_id = str(approval_id or "").strip()

        def blocked(reason: str, **extra: Any) -> Dict[str, Any]:
            return {
                "type": "soundcheck_stage_reanalysis_result",
                "status": "blocked",
                "success": False,
                "reason": reason,
                "run_id": run_id or None,
                "live_authority_changed": False,
                **extra,
            }

        if approved is not True or not approval_id:
            return blocked("operator_approval_required")
        engine = getattr(self, "auto_soundcheck_engine", None)
        prepare = getattr(engine, "prepare_soundcheck_revision_candidate", None)
        commit = getattr(engine, "commit_soundcheck_revision_candidate", None)
        discard = getattr(engine, "discard_soundcheck_revision_candidate", None)
        discard_run = getattr(
            engine,
            "discard_soundcheck_revision_candidates_for_run",
            None,
        )
        validate_candidate = getattr(
            engine,
            "validate_soundcheck_revision_candidate",
            None,
        )
        publication_scope = getattr(
            engine,
            "soundcheck_revision_publication_scope",
            None,
        )
        if not all(
            callable(item)
            for item in (
                prepare,
                commit,
                discard,
                discard_run,
                validate_candidate,
                publication_scope,
            )
        ):
            return blocked(
                "soundcheck_reanalysis_engine_unavailable",
                fresh_analysis_required=True,
            )

        reanalysis_lock = getattr(self, "_soundcheck_reanalysis_lock", None)
        if reanalysis_lock is None:
            self._soundcheck_reanalysis_lock = asyncio.Lock()
            reanalysis_lock = self._soundcheck_reanalysis_lock
        async with reanalysis_lock:
            async with self._console_write_scope() as writer_acquired:
                if not writer_acquired:
                    return blocked(
                        "console_writer_scope_unavailable",
                        fresh_analysis_required=True,
                    )

                current = self._soundcheck_manifests.get(run_id)
                active_run_id = self._current_active_soundcheck_run_id()
                if not isinstance(current, Mapping) or active_run_id != run_id:
                    return blocked(
                        (
                            "soundcheck_parent_manifest_missing"
                            if not isinstance(current, Mapping)
                            else "soundcheck_run_superseded"
                        ),
                        active_run_id=active_run_id or None,
                    )
                applying_rows = [
                    item
                    for item in self.operator_proposal_queue.pending_actions()
                    if isinstance(item, Mapping)
                    and item.get("status") == "applying"
                    and isinstance(item.get("raw"), Mapping)
                    and str(item["raw"].get("soundcheck_run_id") or "") == run_id
                ]
                if applying_rows:
                    return blocked("soundcheck_child_apply_in_progress")

                base_revision = int(current.get("plan_revision") or 0)
                base_content_sha256 = str(
                    current.get("manifest_content_sha256") or ""
                )
                previous_preflight_present = (
                    run_id in self._soundcheck_preflight_assessments
                )
                previous_preflight = copy.deepcopy(
                    self._soundcheck_preflight_assessments.get(run_id)
                )
                previous_preflight_timestamp_present = (
                    run_id in self._soundcheck_preflight_timestamps
                )
                previous_preflight_timestamp = (
                    self._soundcheck_preflight_timestamps.get(run_id)
                )
                replan_store = getattr(self, "_soundcheck_replan_drafts", {})
                previous_replan_present = run_id in replan_store
                previous_replan = copy.deepcopy(replan_store.get(run_id))
                candidate_id = ""
                server_published = False
                engine_committed = False

                def restore_preflight_state() -> None:
                    if previous_preflight_present:
                        self._soundcheck_preflight_assessments[run_id] = (
                            previous_preflight
                        )
                    else:
                        self._soundcheck_preflight_assessments.pop(run_id, None)
                    if previous_preflight_timestamp_present:
                        self._soundcheck_preflight_timestamps[run_id] = (
                            previous_preflight_timestamp
                        )
                    else:
                        self._soundcheck_preflight_timestamps.pop(run_id, None)
                    if previous_replan_present:
                        self._soundcheck_replan_drafts[run_id] = previous_replan
                    else:
                        self._soundcheck_replan_drafts.pop(run_id, None)

                try:
                    try:
                        candidate = await self._cancellation_safe_thread_call(
                            prepare,
                            run_id,
                            expected_plan_revision=base_revision,
                            expected_manifest_content_sha256=base_content_sha256,
                        )
                    except asyncio.CancelledError as cancellation:
                        try:
                            await self._cancellation_safe_thread_call(
                                discard_run,
                                run_id,
                            )
                        except asyncio.CancelledError:
                            pass
                        raise cancellation

                    candidate_id = str(candidate.get("candidate_id") or "")
                    bundle = candidate.get("bundle")
                    revised = (
                        bundle.get("supervised_transaction_plans")
                        if isinstance(bundle, Mapping)
                        else None
                    )
                    if not (
                        candidate.get("run_id") == run_id
                        and candidate.get("base_plan_revision") == base_revision
                        and candidate.get("target_plan_revision")
                        == base_revision + 1
                        and candidate.get("base_manifest_content_sha256")
                        == base_content_sha256
                        and isinstance(revised, Mapping)
                        and revised.get("run_id") == run_id
                        and revised.get("plan_revision") == base_revision + 1
                        and revised.get("schema_version")
                        == "soundcheck_batch_manifest/v2"
                        and revised.get("manifest_content_sha256")
                        == soundcheck_manifest_content_sha256(revised)
                    ):
                        return blocked("soundcheck_reanalysis_revision_invalid")
                    if getattr(self, "auto_soundcheck_engine", None) is not engine:
                        return blocked("soundcheck_engine_superseded")
                    await self._cancellation_safe_thread_call(
                        validate_candidate,
                        candidate_id,
                    )

                    preflight = await self.preflight_soundcheck_manifest(revised)
                    latest = self._soundcheck_manifests.get(run_id)
                    if not (
                        self._current_active_soundcheck_run_id() == run_id
                        and isinstance(latest, Mapping)
                        and latest.get("plan_revision") == base_revision
                        and str(latest.get("manifest_content_sha256") or "")
                        == base_content_sha256
                    ):
                        return blocked("soundcheck_reanalysis_base_changed")
                    if getattr(self, "auto_soundcheck_engine", None) is not engine:
                        return blocked("soundcheck_engine_superseded")

                    by_operation = {
                        str(item.get("operation_id") or ""): dict(item)
                        for item in preflight.get("assessments") or []
                        if isinstance(item, Mapping)
                    }
                    accepted_manifest = copy.deepcopy(revised)
                    accepted_manifest["plans"] = [
                        {
                            **dict(plan),
                            "preflight_assessment": plan_preflight_assessment(
                                plan,
                                by_operation,
                            ),
                        }
                        for plan in revised.get("plans") or []
                        if isinstance(plan, Mapping)
                    ]
                    accepted_manifest["normal_plans"] = [
                        plan
                        for plan in accepted_manifest["plans"]
                        if plan.get("separate_approval_required") is not True
                    ]
                    accepted_manifest["separate_approval_plans"] = [
                        plan
                        for plan in accepted_manifest["plans"]
                        if plan.get("separate_approval_required") is True
                    ]
                    bind_soundcheck_manifest_content_sha256(accepted_manifest)
                    accepted_bundle = {
                        **dict(bundle),
                        "supervised_transaction_plans": accepted_manifest,
                        "soundcheck_preflight": preflight,
                    }
                    # No await is allowed from this lifecycle check through
                    # server publication and engine acknowledgement. A stop or
                    # child apply therefore cannot split the two revisions.
                    with publication_scope(candidate_id):
                        summary = self.import_soundcheck_recommendations_to_operator_queue(
                            accepted_bundle,
                            trusted_engine_bundle=True,
                        )
                        published_manifest = self._soundcheck_manifests.get(run_id)
                        server_published = bool(
                            isinstance(published_manifest, Mapping)
                            and published_manifest.get("plan_revision")
                            == accepted_manifest.get("plan_revision")
                            and published_manifest.get("manifest_content_sha256")
                            == accepted_manifest.get("manifest_content_sha256")
                        )
                        if server_published:
                            commit(
                                candidate_id,
                                accepted_manifest_content_sha256=accepted_manifest.get(
                                    "manifest_content_sha256"
                                ),
                                accepted_bundle=accepted_bundle,
                            )
                            engine_committed = True
                    if summary.get("success") is not True:
                        return {
                            **blocked(
                                str(summary.get("reason") or summary.get("status") or "blocked"),
                                from_plan_revision=base_revision,
                                to_plan_revision=(
                                    accepted_manifest.get("plan_revision")
                                    if server_published
                                    else None
                                ),
                                approval_id=approval_id,
                                operator_import=summary,
                                fresh_analysis_required=not server_published,
                                live_mixer_writes=False,
                            ),
                            "status": str(summary.get("status") or "blocked"),
                        }
                    if not server_published or not engine_committed:
                        return blocked(
                            "soundcheck_reanalysis_publish_incomplete",
                            fresh_analysis_required=True,
                        )
                    return {
                        "type": "soundcheck_stage_reanalysis_result",
                        "status": "ready_for_review",
                        "success": True,
                        "reason": summary.get("reason"),
                        "run_id": run_id,
                        "from_plan_revision": base_revision,
                        "to_plan_revision": accepted_manifest.get("plan_revision"),
                        "approval_id": approval_id,
                        "operator_import": summary,
                        "fresh_analysis_required": False,
                        "live_authority_changed": False,
                        "live_mixer_writes": False,
                    }
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    if server_published and not engine_committed:
                        blockers = self._soundcheck_global_blockers.setdefault(
                            run_id,
                            [],
                        )
                        if "soundcheck_engine_revision_commit_failed" not in blockers:
                            blockers.append("soundcheck_engine_revision_commit_failed")
                        return blocked(
                            "soundcheck_engine_revision_commit_failed",
                            error=type(exc).__name__,
                            fresh_analysis_required=True,
                        )
                    return blocked(
                        "soundcheck_stage_reanalysis_failed",
                        error=type(exc).__name__,
                        fresh_analysis_required=True,
                    )
                finally:
                    if candidate_id and not engine_committed:
                        try:
                            discard(candidate_id)
                        except BaseException:
                            pass
                    if not server_published:
                        restore_preflight_state()

    @staticmethod
    def _public_soundcheck_replan(draft: Any) -> Dict[str, Any]:
        """Bound the UI payload while retaining exact server-side draft identity."""

        if not isinstance(draft, Mapping) or draft.get("draft_available") is not True:
            return {
                "status": "not_required",
                "draft_available": False,
                "operator_requests": [],
                "safe_alternatives": [],
            }
        return {
            "status": str(draft.get("status") or "draft_available"),
            "draft_available": True,
            "draft_id": str(draft.get("draft_id") or ""),
            "run_id": str(draft.get("run_id") or ""),
            "from_plan_revision": draft.get("from_plan_revision"),
            "to_plan_revision": draft.get("to_plan_revision"),
            "split_operation_count": int(draft.get("split_operation_count") or 0),
            "safe_alternative_operation_count": int(
                draft.get("safe_alternative_operation_count") or 0
            ),
            "deferred_target_count": int(draft.get("deferred_target_count") or 0),
            "operator_requests": [
                dict(item)
                for item in draft.get("operator_requests") or []
                if isinstance(item, Mapping)
            ],
            "safe_alternatives": [
                dict(item)
                for item in draft.get("safe_alternatives") or []
                if isinstance(item, Mapping)
            ],
            "new_operator_approval_required": True,
            "live_authority_changed": False,
        }

    async def refresh_soundcheck_child_preflight(
        self,
        plan_payload: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Refresh one child and Main evidence immediately before approval/apply."""

        if not isinstance(plan_payload, Mapping):
            return {"refreshed": False, "reason": "soundcheck_child_plan_required"}
        refresh_started_at = time.time()
        run_id = str(plan_payload.get("run_id") or "").strip()
        transaction_id = str(plan_payload.get("transaction_id") or "").strip()
        manifest = self._soundcheck_manifests.get(run_id)
        active_run_id = self._current_active_soundcheck_run_id()
        if active_run_id and run_id != active_run_id:
            return {
                "refreshed": False,
                "reason": "soundcheck_run_superseded",
                "run_id": run_id,
                "active_run_id": active_run_id,
            }
        stored_plan = next(
            (
                item
                for item in (manifest or {}).get("plans") or []
                if isinstance(item, Mapping)
                and str(item.get("transaction_id") or "") == transaction_id
            ),
            None,
        )
        if not (
            isinstance(manifest, Mapping)
            and isinstance(stored_plan, Mapping)
            and dict(plan_payload) == dict(stored_plan)
        ):
            return {
                "refreshed": False,
                "reason": "stale_or_foreign_soundcheck_child",
                "run_id": run_id,
                "transaction_id": transaction_id,
            }

        requested_coupled_group = bool(
            stored_plan.get("coupled_safe_alternative_group") is True
            and stored_plan.get("atomic_group_commit_required") is True
        )
        requested_fader_balance_plan = any(
            isinstance(operation, Mapping)
            and operation.get("family_id") == "fader"
            and operation.get("write_kind") == "fader"
            for operation in stored_plan.get("operations") or []
        )
        manifest_source_coverage = (
            manifest.get("source_coverage")
            if isinstance(manifest.get("source_coverage"), Mapping)
            else {}
        )
        source_balance_proof_required = bool(
            requested_fader_balance_plan
            and list(manifest_source_coverage.get("selected_channel_ids") or [])
        )

        persistent_blockers = [
            str(item)
            for item in self._soundcheck_global_blockers.get(run_id, [])
            if str(item).strip()
            and not self._soundcheck_refreshable_global_blocker(str(item))
        ]
        current_blockers: List[str] = []
        child_local_blockers: List[str] = []
        if requested_coupled_group:
            coupled_contract_errors = coupled_safe_alternative_plan_errors(
                stored_plan,
                source_coverage=(
                    manifest.get("source_coverage")
                    if isinstance(manifest.get("source_coverage"), Mapping)
                    else None
                ),
            )
            if coupled_contract_errors:
                return {
                    "refreshed": False,
                    "reason": "coupled_safe_alternative_contract_invalid",
                    "contract_errors": coupled_contract_errors,
                    "run_id": run_id,
                    "transaction_id": transaction_id,
                }
        if self._manual_write_emergency_stopped:
            current_blockers.append("emergency_stop_active")
        if not self.mixer_client or not getattr(
            self.mixer_client,
            "is_connected",
            False,
        ):
            current_blockers.append("mixer_not_connected")
        elif not callable(getattr(self.mixer_client, "query_parameter", None)):
            current_blockers.append("physical_readback_transport_unavailable")

        current_stage_order = stored_plan.get("stage_order")
        if isinstance(current_stage_order, bool) or not isinstance(
            current_stage_order,
            int,
        ):
            current_blockers.append("soundcheck_stage_order_invalid")
            plans_to_refresh = [stored_plan]
        else:
            # Re-read every earlier stage together with the requested child.
            # This creates a fresh stage checkpoint; refreshing only the
            # current tonal/spatial child could otherwise let it run over a
            # drifted or inaudible foundation result.
            previous_preflight = self._soundcheck_preflight_assessments.get(
                run_id,
                {},
            )
            previous_by_operation = {
                str(item.get("operation_id") or ""): item
                for item in previous_preflight.get("assessments") or []
                if isinstance(item, Mapping)
            }
            manifest_plans = [
                item
                for item in manifest.get("plans") or []
                if isinstance(item, Mapping)
            ]
            current_plan_index = next(
                (
                    index
                    for index, item in enumerate(manifest_plans)
                    if str(item.get("transaction_id") or "") == transaction_id
                ),
                len(manifest_plans),
            )
            current_channels = {
                operation.get("channel")
                for operation in stored_plan.get("operations") or []
                if isinstance(operation, Mapping)
                and isinstance(operation.get("channel"), int)
                and not isinstance(operation.get("channel"), bool)
            }
            current_group_scope = any(
                str(operation.get("family_id") or "")
                in {"bus_group_dca_matrix", "master_output"}
                for operation in stored_plan.get("operations") or []
                if isinstance(operation, Mapping)
            )
            plans_to_refresh = []
            for item_index, item in enumerate(manifest_plans):
                if not isinstance(item, Mapping):
                    continue
                if str(item.get("transaction_id") or "") == transaction_id:
                    plans_to_refresh.append(item)
                    continue
                item_order = item.get("stage_order")
                if (
                    isinstance(item_order, bool)
                    or not isinstance(item_order, int)
                    or item_order > current_stage_order
                ):
                    continue
                same_stage = item_order == current_stage_order
                item_channels = {
                    operation.get("channel")
                    for operation in item.get("operations") or []
                    if isinstance(operation, Mapping)
                    and isinstance(operation.get("channel"), int)
                    and not isinstance(operation.get("channel"), bool)
                }
                same_source_dependency = bool(
                    current_channels.intersection(item_channels)
                )
                if same_stage and (
                    item_index >= current_plan_index
                    or not (same_source_dependency or current_group_scope)
                ):
                    continue
                logical_ids = [
                    str(value or "")
                    for value in item.get("logical_correction_ids") or []
                    if str(value or "")
                ]
                needs_checkpoint = any(
                    (
                        previous_by_operation.get(logical_id, {}).get("disposition")
                        in {"apply", "already_satisfied"}
                    )
                    or (
                        self._soundcheck_outcomes.get(run_id, {}).get(
                            logical_id,
                            {},
                        ).get("status")
                        == "applied"
                    )
                    or (
                        self._soundcheck_outcomes.get(run_id, {}).get(
                            logical_id,
                            {},
                        ).get("disposition")
                        == "already_satisfied"
                    )
                    for logical_id in logical_ids
                )
                if needs_checkpoint:
                    plans_to_refresh.append(item)

        parsed_plans: List[ConsoleTransactionPlan] = []
        for plan in plans_to_refresh:
            try:
                parsed_plan = self._parse_supervised_transaction_plan(plan)
            except (TypeError, ValueError) as exc:
                current_blockers.append(str(exc))
                continue
            coupled_group = bool(
                plan.get("coupled_safe_alternative_group") is True
                and plan.get("atomic_group_commit_required") is True
                and len(parsed_plan.operations) >= 2
                and all(
                    operation.family_id == "fader"
                    and operation.write_kind == "fader"
                    and operation.metadata.get(
                        "atomic_group_commit_required"
                    )
                    is True
                    for operation in parsed_plan.operations
                )
            )
            if len(parsed_plan.operations) != 1 and not coupled_group:
                current_blockers.append("soundcheck_child_must_have_one_operation")
                continue
            parsed_plans.append(parsed_plan)
        parsed = next(
            (
                item
                for item in parsed_plans
                if item.transaction_id == transaction_id
            ),
            None,
        )

        snapshots: Dict[str, Mapping[str, Any]] = {}
        adapters: Dict[str, Any] = {}
        operations = [
            operation
            for item in parsed_plans
            for operation in item.operations
        ]
        # A later tonal/spatial/group child can depend on an already applied
        # selected-source fader checkpoint.  Re-read the complete audible path
        # for that dependency even when the currently requested child is not a
        # fader operation; relying only on a cached change callback would miss
        # a dropped OSC mute/routing/DCA notification.
        dependency_fader_balance_present = any(
            refresh_operation.family_id == "fader"
            and refresh_operation.write_kind == "fader"
            for refresh_operation in operations
        )
        selected_channels_for_path_proof = {
            int(channel)
            for channel in manifest_source_coverage.get("selected_channel_ids") or []
            if isinstance(channel, int)
            and not isinstance(channel, bool)
            and 1 <= channel <= 48
        }
        current_selected_source_dependency_present = any(
            isinstance(refresh_operation.channel, int)
            and not isinstance(refresh_operation.channel, bool)
            and refresh_operation.channel in selected_channels_for_path_proof
            for refresh_operation in operations
        )
        selected_source_group_dependency_present = any(
            refresh_operation.family_id
            in {"bus_group_dca_matrix", "master_output"}
            for refresh_operation in operations
        )
        source_balance_proof_required = bool(
            selected_channels_for_path_proof
            and (
                dependency_fader_balance_present
                or current_selected_source_dependency_present
                or selected_source_group_dependency_present
            )
        )
        operation = parsed.operations[0] if parsed is not None and parsed.operations else None
        max_age_sec = 30.0
        if not current_blockers and operations:
            adapters = build_wing_supervised_family_adapters(
                self.mixer_client,
                audio_recheck=self._supervised_transaction_audio_recheck,
            )
            for refresh_operation in operations:
                adapter = adapters.get(refresh_operation.family_id)
                if adapter is None:
                    current_blockers.append(
                        f"family_adapter_unavailable:{refresh_operation.family_id}"
                    )
                    break
                try:
                    snapshot = dict(await adapter.snapshot(refresh_operation))
                    snapshot["captured_at_epoch"] = time.time()
                    snapshots[refresh_operation.operation_id] = snapshot
                except asyncio.CancelledError:
                    raise
                except Exception:
                    current_blockers.append(
                        f"physical_snapshot_failed:{refresh_operation.operation_id}"
                    )
                    break

        first_snapshots = dict(snapshots)
        if not current_blockers and operations:
            snapshots = {}
            for refresh_operation in operations:
                adapter = adapters[refresh_operation.family_id]
                try:
                    snapshot = dict(await adapter.snapshot(refresh_operation))
                    snapshot["captured_at_epoch"] = time.time()
                    snapshots[refresh_operation.operation_id] = snapshot
                except asyncio.CancelledError:
                    raise
                except Exception:
                    current_blockers.append(
                        f"physical_snapshot_failed:{refresh_operation.operation_id}"
                    )
                    break
        if not current_blockers and operations:
            for refresh_operation in operations:
                adapter = adapters[refresh_operation.family_id]
                prepared = adapter.prepare(refresh_operation)
                address_specs = [
                    *list(prepared.get("writes") or []),
                    *list(prepared.get("audible_state_reads") or []),
                ]
                first_values = dict(
                    first_snapshots.get(refresh_operation.operation_id, {}).get(
                        "values"
                    )
                    or {}
                )
                second_values = dict(
                    snapshots.get(refresh_operation.operation_id, {}).get("values")
                    or {}
                )
                for spec in address_specs:
                    address = str(spec.get("address") or "")
                    tolerance = float(spec.get("tolerance") or 0.0)
                    left = first_values.get(address)
                    right = second_values.get(address)
                    try:
                        equal = bool(
                            math.isfinite(float(left))
                            and math.isfinite(float(right))
                            and abs(float(left) - float(right)) <= tolerance
                        )
                    except (TypeError, ValueError):
                        equal = str(left).strip().upper() == str(right).strip().upper()
                    if (
                        not address
                        or address not in first_values
                        or address not in second_values
                        or not equal
                    ):
                        current_blockers.append(
                            "soundcheck_preflight_consistency_mismatch:"
                            + refresh_operation.operation_id
                        )
                        break
        if not current_blockers and requested_coupled_group:
            dynamic_group_errors = coupled_safe_alternative_plan_errors(
                stored_plan,
                snapshots_by_operation=snapshots,
                source_coverage=(
                    manifest.get("source_coverage")
                    if isinstance(manifest.get("source_coverage"), Mapping)
                    else None
                ),
            )
            child_local_blockers.extend(
                "coupled_group_contract:" + error
                for error in dynamic_group_errors
            )

        selected_source_audible_paths: Dict[str, Dict[str, Any]] = {}
        selected_path_fence_values: Dict[str, Any] = {}
        source_balance_analysis_deadline_epoch: float | None = None
        if not current_blockers and source_balance_proof_required:
            source_coverage = (
                dict(manifest.get("source_coverage") or {})
                if isinstance(manifest.get("source_coverage"), Mapping)
                else {}
            )
            captured_at = source_coverage.get("captured_at_epoch")
            declared_max_age = source_coverage.get("max_age_sec")
            declared_max_age_valid = bool(
                isinstance(declared_max_age, (int, float))
                and not isinstance(declared_max_age, bool)
                and math.isfinite(float(declared_max_age))
                and float(declared_max_age) >= 0.0
            )
            if (
                isinstance(captured_at, (int, float))
                and not isinstance(captured_at, bool)
                and declared_max_age_valid
            ):
                source_balance_analysis_deadline_epoch = float(captured_at) + min(
                    float(declared_max_age),
                    30.0,
                )
            source_age_valid = bool(
                source_balance_analysis_deadline_epoch is not None
                and float(captured_at) <= time.time()
                < source_balance_analysis_deadline_epoch
            )
            if not (
                source_coverage.get("verified") is True
                and source_age_valid
            ):
                child_local_blockers.append(
                    "source_balance_analysis_stale"
                )
            selected_channels = {
                int(channel)
                for channel in source_coverage.get("selected_channel_ids") or []
                if isinstance(channel, int)
                and not isinstance(channel, bool)
                and 1 <= channel <= 48
            }
            group_channels = {
                int(operation.channel)
                for operation in (parsed.operations if parsed is not None else ())
                if isinstance(operation.channel, int)
                and not isinstance(operation.channel, bool)
            }
            if requested_coupled_group and (
                not selected_channels or selected_channels != group_channels
            ):
                child_local_blockers.append(
                    "coupled_fader_selected_source_vector_mismatch"
                )
            if not current_blockers and not child_local_blockers:
                expected_faders_by_channel = None
                if not requested_fader_balance_plan:
                    fader_intents = (
                        dict(
                            source_coverage.get(
                                "per_source_fader_intent"
                            )
                            or {}
                        )
                        if isinstance(
                            source_coverage.get("per_source_fader_intent"),
                            Mapping,
                        )
                        else {}
                    )
                    expected_faders_by_channel = {
                        channel: float(
                            fader_intents[str(channel)][
                                "analysis_baseline_db"
                            ]
                        )
                        for channel in selected_channels
                        if isinstance(fader_intents.get(str(channel)), Mapping)
                        and isinstance(
                            fader_intents[str(channel)].get(
                                "analysis_baseline_db"
                            ),
                            (int, float),
                        )
                        and not isinstance(
                            fader_intents[str(channel)].get(
                                "analysis_baseline_db"
                            ),
                            bool,
                        )
                    }
                    for refresh_operation in operations:
                        if (
                            refresh_operation.family_id == "fader"
                            and refresh_operation.write_kind == "fader"
                            and isinstance(refresh_operation.channel, int)
                            and not isinstance(refresh_operation.channel, bool)
                            and isinstance(
                                refresh_operation.target_value,
                                (int, float),
                            )
                            and not isinstance(
                                refresh_operation.target_value,
                                bool,
                            )
                        ):
                            expected_faders_by_channel[
                                refresh_operation.channel
                            ] = float(refresh_operation.target_value)
                (
                    selected_source_audible_paths,
                    selected_path_fence_values,
                ) = await self._read_selected_source_audible_paths_two_pass(
                    run_id=run_id,
                    source_coverage=source_coverage,
                    channel_ids=sorted(selected_channels),
                    max_age_sec=max_age_sec,
                    expected_fader_db_by_channel=(
                        expected_faders_by_channel
                    ),
                )
                for channel in sorted(selected_channels):
                    path = selected_source_audible_paths.get(str(channel), {})
                    if not (
                        path.get("physical_verified") is True
                        and path.get("two_pass_consistent") is True
                        and path.get("matches_analysis_baseline") is True
                        and path.get("matches_analysis_main_send") is True
                        and path.get("channel_main_path_locally_enabled") is True
                        and path.get("downstream_dca_path_verified") is True
                    ):
                        blocker_prefix = (
                            "coupled_fader_selected_source_path_unverified"
                            if requested_fader_balance_plan
                            else "selected_source_dependency_path_unverified"
                        )
                        child_local_blockers.append(
                            f"{blocker_prefix}:"
                            f"{channel}:{path.get('reason') or 'missing'}"
                        )
                if (
                    source_balance_analysis_deadline_epoch is None
                    or time.time() >= source_balance_analysis_deadline_epoch
                ):
                    child_local_blockers.append(
                        "source_balance_analysis_expired_during_preflight"
                    )

        audio_preflight = None
        if not current_blockers and not child_local_blockers and operation is not None:
            audio_preflight = await self._supervised_transaction_audio_recheck(
                operation,
                snapshots.get(operation.operation_id, {}),
                {},
            )
            if not audio_preflight.passed:
                current_blockers.append(
                    str(audio_preflight.reason or "master_audio_preflight_failed")
                )

        console_change_fence: Dict[str, Any] = {}
        if not current_blockers and not child_local_blockers and operations:
            route_fence_values = self._soundcheck_route_fence_values(
                audio_preflight
            )
            console_change_fence = (
                await self._establish_soundcheck_console_change_fence(
                    operations,
                    snapshots,
                    adapters,
                    additional_observed_values={
                        **route_fence_values,
                        **selected_path_fence_values,
                    },
                    capture_status=(
                        dict(audio_preflight.evidence).get(
                            "capture_status_after"
                        )
                        if audio_preflight is not None
                        and isinstance(audio_preflight.evidence, Mapping)
                        else None
                    ),
                )
            )
            if console_change_fence.get("verified") is not True:
                current_blockers.append(
                    "soundcheck_preflight_change_fence_failed:"
                    + str(
                        console_change_fence.get("reason")
                        or "unknown_change_fence_failure"
                    )
                )
        if (
            source_balance_proof_required
            and not child_local_blockers
            and (
                source_balance_analysis_deadline_epoch is None
                or time.time() >= source_balance_analysis_deadline_epoch
            )
        ):
            child_local_blockers.append(
                "source_balance_analysis_expired_during_preflight"
            )

        refreshed_at = time.time()
        oldest_snapshot_at = min(
            (
                float(item.get("captured_at_epoch"))
                for item in snapshots.values()
                if isinstance(item.get("captured_at_epoch"), (int, float))
                and not isinstance(item.get("captured_at_epoch"), bool)
            ),
            default=refreshed_at,
        )
        if snapshots and refreshed_at - oldest_snapshot_at > max_age_sec:
            current_blockers.append("soundcheck_preflight_collection_stale")
        if refreshed_at - refresh_started_at > max_age_sec:
            current_blockers.append("soundcheck_preflight_collection_stale")

        child_manifest = {
            "run_id": manifest.get("run_id"),
            "plan_revision": manifest.get("plan_revision"),
            "plan_fingerprint": manifest.get("plan_fingerprint"),
            "plans": [dict(item) for item in plans_to_refresh],
            "unsupported_actions": [],
            "readiness_residuals": [],
        }
        blockers = sorted(set(persistent_blockers + current_blockers))
        child_assessment = assess_soundcheck_manifest(
            child_manifest,
            snapshots_by_operation=snapshots,
            adapters=adapters,
            global_blockers=blockers,
        )
        if child_local_blockers:
            current_group_operation_ids = {
                str(item.get("operation_id") or "")
                for item in stored_plan.get("operations") or []
                if isinstance(item, Mapping)
            }
            for item in child_assessment.get("assessments") or []:
                if (
                    isinstance(item, dict)
                    and str(item.get("operation_id") or "")
                    in current_group_operation_ids
                ):
                    item.update(
                        {
                            "scope": "operation",
                            "disposition": "operator_review",
                            "reason": child_local_blockers[0],
                            "executable": False,
                            "child_local_blockers": sorted(
                                set(child_local_blockers)
                            ),
                        }
                    )
        for item in child_assessment.get("assessments") or []:
            if isinstance(item, dict):
                snapshot = snapshots.get(str(item.get("operation_id") or ""), {})
                item["captured_at_epoch"] = snapshot.get(
                    "captured_at_epoch",
                    oldest_snapshot_at,
                )
                item["max_age_sec"] = max_age_sec

        previous = self._soundcheck_preflight_assessments.get(run_id, {})
        merged_assessments = {
            str(item.get("operation_id") or ""): dict(item)
            for item in previous.get("assessments") or []
            if isinstance(item, Mapping)
        }
        for item in child_assessment.get("assessments") or []:
            if isinstance(item, Mapping):
                merged_assessments[str(item.get("operation_id") or "")] = dict(item)
        audio_evidence = (
            dict(audio_preflight.evidence)
            if audio_preflight is not None
            and isinstance(audio_preflight.evidence, Mapping)
            else {}
        )
        local_blockers_by_transaction = dict(
            previous.get("child_local_blockers_by_transaction") or {}
        )
        if child_local_blockers:
            local_blockers_by_transaction[transaction_id] = sorted(
                set(child_local_blockers)
            )
        else:
            local_blockers_by_transaction.pop(transaction_id, None)
        source_deadlines_by_transaction = dict(
            previous.get("source_balance_deadline_by_transaction") or {}
        )
        if source_balance_proof_required:
            source_deadlines_by_transaction[transaction_id] = (
                source_balance_analysis_deadline_epoch
            )
        else:
            source_deadlines_by_transaction.pop(transaction_id, None)
        merged = {
            **dict(previous),
            "schema_version": child_assessment.get("schema_version"),
            "run_id": manifest.get("run_id"),
            "plan_revision": manifest.get("plan_revision"),
            "plan_fingerprint": manifest.get("plan_fingerprint"),
            "global_blockers": blockers,
            "child_local_blockers_by_transaction": (
                local_blockers_by_transaction
            ),
            "assessments": list(merged_assessments.values()),
            "source_coverage": dict(
                previous.get("source_coverage")
                or manifest.get("source_coverage")
                or {}
            ),
            "selected_source_audible_paths": (
                selected_source_audible_paths
                if source_balance_proof_required
                else dict(previous.get("selected_source_audible_paths") or {})
            ),
            "source_balance_analysis_deadline_epoch": (
                source_balance_analysis_deadline_epoch
                if source_balance_proof_required
                else previous.get("source_balance_analysis_deadline_epoch")
            ),
            "source_balance_deadline_by_transaction": (
                source_deadlines_by_transaction
            ),
            "main_47_48_evidence": self._soundcheck_main_evidence_from_audio(
                audio_evidence
            ),
            "console_change_fence": console_change_fence,
            "captured_at_epoch": oldest_snapshot_at,
            "completed_at_epoch": refreshed_at,
            "max_age_sec": max_age_sec,
        }
        active_run_id = self._current_active_soundcheck_run_id()
        if active_run_id and run_id != active_run_id:
            superseded = self._soundcheck_global_blockers.setdefault(run_id, [])
            if "soundcheck_run_superseded" not in superseded:
                superseded.append("soundcheck_run_superseded")
            return {
                "refreshed": False,
                "reason": "soundcheck_run_superseded",
                "run_id": run_id,
                "active_run_id": active_run_id,
            }
        self._soundcheck_preflight_assessments[run_id] = merged
        self._soundcheck_preflight_timestamps[run_id] = oldest_snapshot_at
        self._soundcheck_global_blockers[run_id] = blockers
        if console_change_fence.get("verified") is True:
            self._arm_soundcheck_console_change_invalidator(
                run_id,
                console_change_fence,
            )

        for item in child_assessment.get("assessments") or []:
            if not isinstance(item, Mapping):
                continue
            operation_id = str(item.get("operation_id") or "")
            existing_outcome = self._soundcheck_outcomes.setdefault(run_id, {}).get(
                operation_id,
                {},
            )
            if item.get("disposition") == "apply":
                if existing_outcome.get("status") == "preflight_classified":
                    self._soundcheck_outcomes[run_id].pop(operation_id, None)
                continue
            evidence_captured_at = item.get("captured_at_epoch")
            evidence_fresh = bool(
                isinstance(evidence_captured_at, (int, float))
                and not isinstance(evidence_captured_at, bool)
                and 0.0 <= refreshed_at - float(evidence_captured_at) <= max_age_sec
            )
            if existing_outcome.get("status") == "applied":
                terminal_verified = bool(
                    item.get("disposition") == "already_satisfied"
                    and item.get("physical_verified") is True
                    and item.get("audible_state_verified") is True
                    and evidence_fresh
                    and not blockers
                )
                existing_outcome["final_reconciliation_verified"] = terminal_verified
                existing_outcome["reconciliation_captured_at_epoch"] = (
                    evidence_captured_at
                )
                existing_outcome["reconciliation_max_age_sec"] = max_age_sec
                existing_outcome["audible_state_verified"] = (
                    item.get("audible_state_verified") is True
                )
                existing_outcome["final_reconciliation_reason"] = (
                    "prior_stage_target_and_audible_state_reconciled"
                    if terminal_verified
                    else str(item.get("reason") or "prior_stage_reconciliation_failed")
                )
                continue
            if existing_outcome.get("status") == "rolled_back":
                continue
            terminal_verified = bool(
                item.get("disposition") == "already_satisfied"
                and item.get("physical_verified") is True
                and item.get("audible_state_verified") is True
                and evidence_fresh
                and not blockers
            )
            self._soundcheck_outcomes[run_id][operation_id] = {
                "disposition": str(item.get("disposition") or "operator_review"),
                "reason": str(item.get("reason") or "preflight_classified"),
                "physical_verified": item.get("physical_verified") is True,
                "audible_state_verified": item.get("audible_state_verified") is True,
                "status": "preflight_classified",
                "evidence_captured_at_epoch": evidence_captured_at,
                "evidence_max_age_sec": max_age_sec,
                "evidence_fresh": evidence_fresh,
                "final_reconciliation_verified": terminal_verified,
                "reconciliation_captured_at_epoch": evidence_captured_at,
                "reconciliation_max_age_sec": max_age_sec,
                "wal_terminal": "not_applicable",
                "audio_recheck_passed": False,
            }
        return {**merged, "refreshed": True}

    def validate_soundcheck_child_for_apply(self, proposal: Any) -> Dict[str, Any]:
        """Reject stale, globally blocked or out-of-stage children before send."""
        raw = getattr(proposal, "raw", None)
        plan = raw.get("supervised_transaction_plan") if isinstance(raw, Mapping) else None
        if not isinstance(plan, Mapping):
            return {"allowed": False, "reason": "soundcheck_child_plan_required"}
        run_id = str(plan.get("run_id") or "").strip()
        manifest = self._soundcheck_manifests.get(run_id)
        if not run_id or not isinstance(manifest, Mapping):
            return {"allowed": False, "reason": "soundcheck_parent_manifest_missing"}
        active_run_id = self._current_active_soundcheck_run_id()
        if active_run_id and run_id != active_run_id:
            return {
                "allowed": False,
                "reason": "soundcheck_run_superseded",
                "run_id": run_id,
                "active_run_id": active_run_id,
            }
        transaction_id = str(plan.get("transaction_id") or "").strip()
        stored_plan = next(
            (
                item
                for item in manifest.get("plans") or []
                if isinstance(item, Mapping)
                and str(item.get("transaction_id") or "") == transaction_id
            ),
            None,
        )
        if not (
            isinstance(stored_plan, Mapping)
            and plan.get("run_id") == manifest.get("run_id")
            and plan.get("plan_revision") == manifest.get("plan_revision")
            and plan.get("plan_fingerprint") == manifest.get("plan_fingerprint")
            and dict(plan) == dict(stored_plan)
        ):
            return {
                "allowed": False,
                "reason": "stale_or_foreign_soundcheck_child",
                "run_id": run_id,
                "transaction_id": transaction_id,
            }
        coupled_contract_errors = coupled_safe_alternative_plan_errors(
            stored_plan,
            source_coverage=(
                manifest.get("source_coverage")
                if isinstance(manifest.get("source_coverage"), Mapping)
                else None
            ),
        )
        if coupled_contract_errors:
            return {
                "allowed": False,
                "reason": "coupled_safe_alternative_contract_invalid",
                "contract_errors": coupled_contract_errors,
            }
        blockers = list(self._soundcheck_global_blockers.get(run_id, []))
        if blockers:
            return {
                "allowed": False,
                "reason": "soundcheck_global_blocker_active",
                "global_blockers": blockers,
            }
        preflight = getattr(self, "_soundcheck_preflight_assessments", {}).get(
            run_id,
            {},
        )
        child_local_blockers = list(
            (
                preflight.get("child_local_blockers_by_transaction") or {}
            ).get(transaction_id, [])
        )
        if child_local_blockers:
            return {
                "allowed": False,
                "reason": "soundcheck_child_local_blocker_active",
                "child_local_blockers": child_local_blockers,
            }
        source_path_proof_required = self._soundcheck_plan_requires_source_path_proof(
            stored_plan,
            (
                manifest.get("source_coverage")
                if isinstance(manifest.get("source_coverage"), Mapping)
                else {}
            ),
        )
        if source_path_proof_required:
            source_deadline = (
                preflight.get("source_balance_deadline_by_transaction") or {}
            ).get(transaction_id)
            if not (
                isinstance(source_deadline, (int, float))
                and not isinstance(source_deadline, bool)
                and math.isfinite(float(source_deadline))
                and time.time() < float(source_deadline)
            ):
                return {
                    "allowed": False,
                    "reason": "source_balance_analysis_expired",
                    "fresh_analysis_required": True,
                    "reanalysis_entrypoint": "reanalyze_soundcheck_stage",
                }
        preflight_fence = (
            preflight.get("console_change_fence")
            if isinstance(preflight, Mapping)
            else None
        )
        if not self._soundcheck_console_fence_still_current(preflight_fence):
            return {
                "allowed": False,
                "reason": "soundcheck_console_change_fence_stale_or_missing",
            }
        assessment_by_operation = {
            str(item.get("operation_id") or ""): item
            for item in preflight.get("assessments") or []
            if isinstance(item, Mapping)
        }
        current_logical_ids = [
            str(item)
            for item in stored_plan.get("logical_correction_ids") or []
            if str(item).strip()
        ]
        current_assessments = [
            assessment_by_operation.get(logical_id, {})
            for logical_id in current_logical_ids
        ]
        stale_assessments = [
            assessment
            for assessment in current_assessments
            if not (
                isinstance(assessment.get("captured_at_epoch"), (int, float))
                and not isinstance(assessment.get("captured_at_epoch"), bool)
                and isinstance(assessment.get("max_age_sec"), (int, float))
                and not isinstance(assessment.get("max_age_sec"), bool)
                and 0.0
                <= time.time() - float(assessment["captured_at_epoch"])
                <= float(assessment["max_age_sec"])
            )
        ]
        if stale_assessments:
            return {
                "allowed": False,
                "reason": "soundcheck_physical_preflight_stale",
                "operation_ids": [
                    str(item.get("operation_id") or "")
                    for item in stale_assessments
                ],
            }
        coupled_group = bool(
            stored_plan.get("coupled_safe_alternative_group") is True
            and stored_plan.get("atomic_group_commit_required") is True
        )
        current_dispositions = {
            str(assessment.get("disposition") or "")
            for assessment in current_assessments
        }
        current_applyable = bool(
            current_assessments
            and (
                (
                    coupled_group
                    and current_dispositions.issubset(
                        {"apply", "already_satisfied"}
                    )
                    and "apply" in current_dispositions
                )
                or (
                    not coupled_group
                    and current_dispositions == {"apply"}
                )
            )
        )
        if not current_applyable:
            return {
                "allowed": False,
                "reason": "soundcheck_child_not_preflight_applyable",
                "preflight_assessments": current_assessments,
            }

        current_stage_order = stored_plan.get("stage_order")
        if isinstance(current_stage_order, bool) or not isinstance(
            current_stage_order,
            int,
        ):
            return {"allowed": False, "reason": "soundcheck_stage_order_invalid"}

        def outcome_resolved(logical_id: str) -> bool:
            outcome = self._soundcheck_outcomes.get(run_id, {}).get(logical_id, {})
            assessment = assessment_by_operation.get(logical_id, {})
            now = time.time()

            def fresh(payload: Mapping[str, Any], captured_key: str, max_age_key: str) -> bool:
                captured_at = payload.get(captured_key)
                max_age = payload.get(max_age_key)
                return bool(
                    isinstance(captured_at, (int, float))
                    and not isinstance(captured_at, bool)
                    and isinstance(max_age, (int, float))
                    and not isinstance(max_age, bool)
                    and float(max_age) >= 0.0
                    and 0.0 <= now - float(captured_at) <= float(max_age)
                )

            if not isinstance(outcome, Mapping) or not isinstance(assessment, Mapping):
                return False
            checkpoint_fresh = fresh(
                outcome,
                "reconciliation_captured_at_epoch",
                "reconciliation_max_age_sec",
            )
            assessment_fresh = fresh(
                assessment,
                "captured_at_epoch",
                "max_age_sec",
            )
            physical_target_current = bool(
                assessment.get("disposition") == "already_satisfied"
                and assessment.get("physical_verified") is True
                and assessment.get("audible_state_verified") is True
                and assessment_fresh
            )
            if not (
                physical_target_current
                and outcome.get("final_reconciliation_verified") is True
                and checkpoint_fresh
                and outcome.get("audible_state_verified") is True
            ):
                return False
            if outcome.get("disposition") == "already_satisfied":
                return bool(
                    outcome.get("physical_verified") is True
                    and fresh(
                        outcome,
                        "evidence_captured_at_epoch",
                        "evidence_max_age_sec",
                    )
                )
            return bool(
                outcome.get("status") == "applied"
                and outcome.get("physical_verified") is True
                and outcome.get("audio_recheck_passed") is True
                and outcome.get("wal_terminal") == "committed"
            )

        unresolved_prior = []
        dependency_reanalysis_required = []
        current_operations = [
            item
            for item in stored_plan.get("operations") or []
            if isinstance(item, Mapping)
        ]
        current_channels = {
            item.get("channel")
            for item in current_operations
            if isinstance(item.get("channel"), int)
            and not isinstance(item.get("channel"), bool)
        }
        current_is_group_scope = any(
            str(item.get("family_id") or "")
            in {"bus_group_dca_matrix", "master_output"}
            for item in current_operations
        )
        stage_order_by_name = {
            "foundation_level": 10,
            "source_control": 20,
            "phase_alignment_review": 30,
            "tonal_dynamics": 40,
            "group_balance": 50,
            "spatial_fx": 60,
            "destructive_high_risk": 70,
        }
        current_logical_ids = {
            str(logical_id or "")
            for logical_id in stored_plan.get("logical_correction_ids") or []
            if str(logical_id or "")
        }
        current_logical_ids.update(
            str(item.get("operation_id") or "")
            for item in current_operations
            if str(item.get("operation_id") or "")
        )
        dependency_residuals = [
            *list(manifest.get("readiness_residuals") or []),
            *list(manifest.get("unsupported_actions") or []),
        ]
        for residual in dependency_residuals:
            if not isinstance(residual, Mapping):
                continue
            if residual.get("musically_required") is False:
                continue
            residual_order = stage_order_by_name.get(
                str(residual.get("stage") or "")
            )
            residual_logical_id = str(
                residual.get("logical_correction_id")
                or f"soundcheck:residual:{residual.get('stage')}"
            )
            residual_current_step_id = str(
                residual.get("current_step_operation_id") or ""
            )
            if (
                residual_order is None
                or residual_order > current_stage_order
                or residual_logical_id in current_logical_ids
                or residual_current_step_id in current_logical_ids
            ):
                continue
            residual_channel = residual.get("channel")
            same_source = bool(
                isinstance(residual_channel, int)
                and not isinstance(residual_channel, bool)
                and residual_channel in current_channels
            )
            if same_source or current_is_group_scope:
                dependency_reanalysis_required.append(residual_logical_id)
        manifest_plans = [
            plan
            for plan in manifest.get("plans") or []
            if isinstance(plan, Mapping)
        ]
        current_transaction_id = str(
            stored_plan.get("transaction_id") or ""
        )
        current_plan_index = next(
            (
                index
                for index, plan in enumerate(manifest_plans)
                if str(plan.get("transaction_id") or "")
                == current_transaction_id
            ),
            len(manifest_plans),
        )
        for prior_index, prior_plan in enumerate(manifest_plans):
            if not isinstance(prior_plan, Mapping):
                continue
            prior_order = prior_plan.get("stage_order")
            if (
                not isinstance(prior_order, int)
                or prior_order > current_stage_order
                or (
                    prior_order == current_stage_order
                    and prior_index >= current_plan_index
                )
            ):
                continue
            for logical_id in prior_plan.get("logical_correction_ids") or []:
                logical_id = str(logical_id or "")
                raw_operation = next(
                    (
                        item
                        for item in prior_plan.get("operations") or []
                        if isinstance(item, Mapping)
                        and str(item.get("operation_id") or "") == logical_id
                    ),
                    {},
                )
                metadata = (
                    raw_operation.get("metadata")
                    if isinstance(raw_operation, Mapping)
                    and isinstance(raw_operation.get("metadata"), Mapping)
                    else {}
                )
                musically_required = metadata.get("musically_required") is not False
                assessment = assessment_by_operation.get(logical_id, {})
                outcome = self._soundcheck_outcomes.get(run_id, {}).get(
                    logical_id,
                    {},
                )
                requires_stage_checkpoint = bool(
                    assessment.get("disposition") in {"apply", "already_satisfied"}
                    or outcome.get("status") == "applied"
                    or outcome.get("disposition") == "already_satisfied"
                )
                prior_channel = raw_operation.get("channel")
                same_source_dependency = bool(
                    isinstance(prior_channel, int)
                    and not isinstance(prior_channel, bool)
                    and prior_channel in current_channels
                )
                local_or_group_dependency = bool(
                    same_source_dependency or current_is_group_scope
                )
                locally_unresolved = bool(
                    logical_id
                    and musically_required
                    and assessment.get("disposition")
                    in {"operator_review", "forbidden", "split_steps"}
                )
                if local_or_group_dependency and locally_unresolved:
                    dependency_reanalysis_required.append(logical_id)
                if (
                    logical_id
                    and musically_required
                    and requires_stage_checkpoint
                    and (
                        prior_order < current_stage_order
                        or local_or_group_dependency
                    )
                    and not outcome_resolved(logical_id)
                ):
                    unresolved_prior.append(logical_id)
        if dependency_reanalysis_required:
            return {
                "allowed": False,
                "reason": "soundcheck_dependency_reanalysis_required",
                "dependency_operation_ids": sorted(
                    set(dependency_reanalysis_required)
                ),
            }
        if unresolved_prior:
            return {
                "allowed": False,
                "reason": "soundcheck_prior_stage_unresolved",
                "unresolved_prior_operation_ids": sorted(set(unresolved_prior)),
            }
        return {
            "allowed": True,
            "reason": "soundcheck_child_identity_and_stage_valid",
            "run_id": run_id,
            "transaction_id": transaction_id,
        }

    def _aggregate_current_soundcheck_readiness(
        self,
        run_id: str,
        *,
        require_final_reconciliation: bool = True,
    ) -> Dict[str, Any] | None:
        manifest = self._soundcheck_manifests.get(run_id)
        if not isinstance(manifest, Mapping):
            return None
        preflight = getattr(self, "_soundcheck_preflight_assessments", {}).get(
            run_id,
            {},
        )
        source_coverage = (
            dict(preflight.get("source_coverage") or {})
            if isinstance(preflight, Mapping)
            else dict(manifest.get("source_coverage") or {})
        )
        if isinstance(preflight, Mapping):
            source_coverage["selected_source_path_evidence"] = dict(
                preflight.get("selected_source_audible_paths") or {}
            )
        return aggregate_soundcheck_readiness(
            manifest,
            outcomes=getattr(self, "_soundcheck_outcomes", {}).get(run_id, {}),
            proposal_only=False,
            global_blockers=getattr(self, "_soundcheck_global_blockers", {}).get(
                run_id,
                [],
            ),
            source_coverage=source_coverage,
            main_47_48_evidence=(
                preflight.get("main_47_48_evidence")
                if isinstance(preflight, Mapping)
                else {}
            ),
            require_final_reconciliation=require_final_reconciliation,
        )

    def refresh_soundcheck_readiness_truth(
        self,
        run_id: str | None = None,
    ) -> Dict[str, Any] | None:
        """Recompute clock-sensitive readiness without touching console I/O."""

        resolved_run_id = str(
            run_id
            or (
                self.last_soundcheck_readiness.get("run_id")
                if isinstance(self.last_soundcheck_readiness, Mapping)
                else ""
            )
            or next(reversed(self._soundcheck_manifests), "")
            or ""
        ).strip()
        if not resolved_run_id:
            return None
        active_run_id = self._current_active_soundcheck_run_id()
        if active_run_id and resolved_run_id != active_run_id:
            blockers = self._soundcheck_global_blockers.setdefault(
                resolved_run_id,
                [],
            )
            if "soundcheck_run_superseded" not in blockers:
                blockers.append("soundcheck_run_superseded")
            inactive = self._aggregate_current_soundcheck_readiness(
                resolved_run_id
            )
            return dict(inactive) if isinstance(inactive, Mapping) else None
        if getattr(self, "_manual_write_emergency_stopped", False):
            self._invalidate_soundcheck_console_fence(
                resolved_run_id,
                "emergency_stop_active",
            )
        elif not getattr(
            getattr(self, "mixer_client", None),
            "is_connected",
            False,
        ):
            self._invalidate_soundcheck_console_fence(
                resolved_run_id,
                "mixer_not_connected",
            )
        preflight = self._soundcheck_preflight_assessments.get(
            resolved_run_id,
            {},
        )
        fence = (
            preflight.get("console_change_fence")
            if isinstance(preflight, Mapping)
            else None
        )
        if isinstance(fence, Mapping) and fence.get("verified") is True:
            if not self._soundcheck_console_fence_still_current(fence):
                capture_current = self._soundcheck_capture_fence_still_current(
                    fence
                )
                self._invalidate_soundcheck_console_fence(
                    resolved_run_id,
                    (
                        "mixer_not_connected"
                        if not getattr(
                            getattr(self, "mixer_client", None),
                            "is_connected",
                            False,
                        )
                        else (
                            "console_state_changed_after_reconciliation"
                            if capture_current
                            else "master_capture_changed_or_stale_after_reconciliation"
                        )
                    ),
                )
        readiness = self._aggregate_current_soundcheck_readiness(resolved_run_id)
        if isinstance(readiness, Mapping):
            self.last_soundcheck_readiness = dict(readiness)
            return dict(readiness)
        return None

    async def reconcile_soundcheck_readiness(
        self,
        run_id: str,
    ) -> Dict[str, Any] | None:
        """Re-read every target and Main before promoting safe/show readiness."""

        manifest = self._soundcheck_manifests.get(run_id)
        if not isinstance(manifest, Mapping):
            return None
        active_run_id = self._current_active_soundcheck_run_id()
        if active_run_id and run_id != active_run_id:
            blockers = self._soundcheck_global_blockers.setdefault(run_id, [])
            if "soundcheck_run_superseded" not in blockers:
                blockers.append("soundcheck_run_superseded")
            inactive = self._aggregate_current_soundcheck_readiness(run_id)
            return dict(inactive) if isinstance(inactive, Mapping) else None
        operations: List[ConsoleTransactionOperation] = []
        for plan in manifest.get("plans") or []:
            if not isinstance(plan, Mapping):
                continue
            parsed = self._parse_supervised_transaction_plan(plan)
            operations.extend(parsed.operations)
        if not operations:
            blockers: List[str] = []
            if self._manual_write_emergency_stopped:
                blockers.append("emergency_stop_active")
            if not self.mixer_client or not getattr(
                self.mixer_client,
                "is_connected",
                False,
            ):
                blockers.append("mixer_not_connected")
            elif not callable(getattr(self.mixer_client, "query_parameter", None)):
                blockers.append("physical_readback_transport_unavailable")
            previous = self._soundcheck_preflight_assessments.get(run_id, {})
            source_coverage = dict(
                previous.get("source_coverage")
                or manifest.get("source_coverage")
                or {}
            )
            selected_channels = {
                int(channel)
                for channel in source_coverage.get("selected_channel_ids") or []
                if isinstance(channel, int)
                and not isinstance(channel, bool)
                and 1 <= channel <= 48
            }
            fader_intents = (
                dict(source_coverage.get("per_source_fader_intent") or {})
                if isinstance(
                    source_coverage.get("per_source_fader_intent"),
                    Mapping,
                )
                else {}
            )
            expected_faders = {
                channel: float(fader_intents[str(channel)]["analysis_baseline_db"])
                for channel in selected_channels
                if isinstance(fader_intents.get(str(channel)), Mapping)
                and isinstance(
                    fader_intents[str(channel)].get("analysis_baseline_db"),
                    (int, float),
                )
                and not isinstance(
                    fader_intents[str(channel)].get("analysis_baseline_db"),
                    bool,
                )
            }
            selected_source_audible_paths: Dict[str, Dict[str, Any]] = {}
            selected_source_path_fence_values: Dict[str, Any] = {}
            if not blockers and selected_channels:
                (
                    selected_source_audible_paths,
                    selected_source_path_fence_values,
                ) = await self._read_selected_source_audible_paths_two_pass(
                    run_id=run_id,
                    source_coverage=source_coverage,
                    channel_ids=sorted(selected_channels),
                    max_age_sec=30.0,
                    expected_fader_db_by_channel=expected_faders,
                )
            audio_recheck = None
            if not blockers:
                audio_recheck = await self._supervised_transaction_audio_recheck(
                    SimpleNamespace(
                        family_id="soundcheck_readiness",
                        operation_id="soundcheck:zero_correction_finalization",
                    ),
                    {},
                    {},
                )
                if not audio_recheck.passed:
                    blockers.append(
                        str(
                            audio_recheck.reason
                            or "master_audio_preflight_failed"
                        )
                    )
            console_change_fence: Dict[str, Any] = {}
            if not blockers:
                console_change_fence = (
                    await self._establish_soundcheck_console_change_fence(
                        (),
                        {},
                        {},
                        additional_observed_values=(
                            {
                                **self._soundcheck_route_fence_values(
                                    audio_recheck
                                ),
                                **selected_source_path_fence_values,
                            }
                        ),
                        capture_status=(
                            dict(audio_recheck.evidence).get(
                                "capture_status_after"
                            )
                            if audio_recheck is not None
                            and isinstance(audio_recheck.evidence, Mapping)
                            else None
                        ),
                        require_global_generation=True,
                    )
                )
                if console_change_fence.get("verified") is not True:
                    blockers.append(
                        "soundcheck_final_reconciliation_change_fence_failed:"
                        + str(
                            console_change_fence.get("reason")
                            or "unknown_change_fence_failure"
                        )
                    )
            active_run_id = self._current_active_soundcheck_run_id()
            if active_run_id and run_id != active_run_id:
                superseded = self._soundcheck_global_blockers.setdefault(
                    run_id,
                    [],
                )
                if "soundcheck_run_superseded" not in superseded:
                    superseded.append("soundcheck_run_superseded")
                inactive = self._aggregate_current_soundcheck_readiness(run_id)
                return (
                    dict(inactive)
                    if isinstance(inactive, Mapping)
                    else None
                )
            audio_evidence = (
                dict(audio_recheck.evidence)
                if audio_recheck is not None
                and isinstance(audio_recheck.evidence, Mapping)
                else {}
            )
            assessment = {
                "schema_version": "soundcheck_preflight_assessment/v1",
                "run_id": manifest.get("run_id"),
                "plan_revision": manifest.get("plan_revision"),
                "plan_fingerprint": manifest.get("plan_fingerprint"),
                "global_blockers": sorted(set(blockers)),
                "assessments": [],
                "source_coverage": source_coverage,
                "selected_source_audible_paths": (
                    selected_source_audible_paths
                ),
                "main_47_48_evidence": self._soundcheck_main_evidence_from_audio(
                    audio_evidence
                ),
                "console_change_fence": console_change_fence,
                "captured_at_epoch": time.time(),
                "completed_at_epoch": time.time(),
                "max_age_sec": 30.0,
            }
            self._soundcheck_preflight_assessments[run_id] = assessment
            self._soundcheck_preflight_timestamps[run_id] = assessment[
                "captured_at_epoch"
            ]
            self._soundcheck_global_blockers[run_id] = sorted(set(blockers))
            readiness = self._aggregate_current_soundcheck_readiness(run_id)
            if isinstance(readiness, Mapping):
                self.last_soundcheck_readiness = dict(readiness)
                if (
                    console_change_fence.get("verified") is True
                    and readiness.get("mix_readiness")
                    in {"safe_core_ready", "show_mix_ready"}
                ):
                    if not self._arm_soundcheck_console_change_invalidator(
                        run_id,
                        console_change_fence,
                    ):
                        readiness = self._aggregate_current_soundcheck_readiness(
                            run_id
                        )
                        if isinstance(readiness, Mapping):
                            self.last_soundcheck_readiness = dict(readiness)
            return readiness

        persistent_blockers = [
            str(item)
            for item in self._soundcheck_global_blockers.get(run_id, [])
            if str(item).strip()
            and not self._soundcheck_refreshable_global_blocker(str(item))
        ]
        blockers: List[str] = []
        if self._manual_write_emergency_stopped:
            blockers.append("emergency_stop_active")
        if not self.mixer_client or not getattr(
            self.mixer_client,
            "is_connected",
            False,
        ):
            blockers.append("mixer_not_connected")
        elif not callable(getattr(self.mixer_client, "query_parameter", None)):
            blockers.append("physical_readback_transport_unavailable")

        reconciliation_started_at = time.time()
        adapters = (
            build_wing_supervised_family_adapters(
                self.mixer_client,
                audio_recheck=self._supervised_transaction_audio_recheck,
            )
            if not blockers
            else {}
        )
        first_snapshots: Dict[str, Mapping[str, Any]] = {}
        if not blockers:
            for operation in operations:
                adapter = adapters.get(operation.family_id)
                if adapter is None:
                    blockers.append(
                        f"family_adapter_unavailable:{operation.family_id}"
                    )
                    break
                try:
                    snapshot = dict(await adapter.snapshot(operation))
                    snapshot["captured_at_epoch"] = time.time()
                    first_snapshots[operation.operation_id] = snapshot
                except asyncio.CancelledError:
                    raise
                except Exception:
                    blockers.append(
                        f"physical_snapshot_failed:{operation.operation_id}"
                    )
                    break

        # A single sequential sweep can miss a human/control-surface change
        # that happens immediately after an early channel was read.  Require a
        # second complete pass and compare every mutation plus audible-state
        # address before promoting readiness.
        snapshots: Dict[str, Mapping[str, Any]] = {}
        if not blockers:
            for operation in operations:
                adapter = adapters.get(operation.family_id)
                try:
                    snapshot = dict(await adapter.snapshot(operation))
                    snapshot["captured_at_epoch"] = time.time()
                    snapshots[operation.operation_id] = snapshot
                except asyncio.CancelledError:
                    raise
                except Exception:
                    blockers.append(
                        f"physical_snapshot_failed:{operation.operation_id}"
                    )
                    break
        if not blockers:
            for operation in operations:
                adapter = adapters[operation.family_id]
                prepared = adapter.prepare(operation)
                addresses = [
                    *list(prepared.get("writes") or []),
                    *list(prepared.get("audible_state_reads") or []),
                ]
                first_values = dict(
                    first_snapshots.get(operation.operation_id, {}).get("values")
                    or {}
                )
                second_values = dict(
                    snapshots.get(operation.operation_id, {}).get("values") or {}
                )
                consistent = True
                for spec in addresses:
                    address = str(spec.get("address") or "")
                    tolerance = float(spec.get("tolerance") or 0.0)
                    left = first_values.get(address)
                    right = second_values.get(address)
                    try:
                        equal = bool(
                            math.isfinite(float(left))
                            and math.isfinite(float(right))
                            and abs(float(left) - float(right)) <= tolerance
                        )
                    except (TypeError, ValueError):
                        equal = str(left).strip().upper() == str(right).strip().upper()
                    if not address or address not in first_values or address not in second_values or not equal:
                        consistent = False
                        break
                if not consistent:
                    blockers.append(
                        "soundcheck_final_reconciliation_consistency_mismatch:"
                        + operation.operation_id
                    )

        final_selected_source_paths: Dict[str, Dict[str, Any]] = {}
        final_source_path_fence_values: Dict[str, Any] = {}
        final_selected_source_channels: set[int] = set()
        fader_balance_plans = [
            plan
            for plan in manifest.get("plans") or []
            if isinstance(plan, Mapping)
            and any(
                isinstance(raw, Mapping)
                and raw.get("family_id") == "fader"
                and raw.get("write_kind") == "fader"
                for raw in plan.get("operations") or []
            )
        ]
        source_coverage = dict(
            self._soundcheck_preflight_assessments.get(run_id, {}).get(
                "source_coverage"
            )
            or manifest.get("source_coverage")
            or {}
        )
        selected_channels = {
            int(channel)
            for channel in source_coverage.get("selected_channel_ids") or []
            if isinstance(channel, int)
            and not isinstance(channel, bool)
            and 1 <= channel <= 48
        }
        fader_intents = (
            dict(source_coverage.get("per_source_fader_intent") or {})
            if isinstance(
                source_coverage.get("per_source_fader_intent"),
                Mapping,
            )
            else {}
        )
        selected_source_path_required = bool(
            selected_channels and fader_intents
        )
        if not blockers and selected_source_path_required:
            final_selected_source_channels = set(selected_channels)
            expected_faders = {
                int(raw["channel"]): float(raw["target_value"])
                for plan in fader_balance_plans
                for raw in plan.get("operations") or []
                if isinstance(raw, Mapping)
                and isinstance(raw.get("channel"), int)
                and not isinstance(raw.get("channel"), bool)
                and isinstance(raw.get("target_value"), (int, float))
                and not isinstance(raw.get("target_value"), bool)
            }
            for channel in selected_channels - set(expected_faders):
                intent = fader_intents.get(str(channel))
                baseline = (
                    intent.get("analysis_baseline_db")
                    if isinstance(intent, Mapping)
                    else None
                )
                if (
                    isinstance(baseline, (int, float))
                    and not isinstance(baseline, bool)
                ):
                    expected_faders[channel] = float(baseline)
            if selected_channels == set(expected_faders):
                (
                    final_selected_source_paths,
                    final_source_path_fence_values,
                ) = await self._read_selected_source_audible_paths_two_pass(
                    run_id=run_id,
                    source_coverage=source_coverage,
                    channel_ids=sorted(selected_channels),
                    max_age_sec=30.0,
                    expected_fader_db_by_channel=expected_faders,
                )

        audio_recheck = None
        if not blockers:
            audio_recheck = await self._supervised_transaction_audio_recheck(
                operations[0],
                snapshots.get(operations[0].operation_id, {}),
                {},
            )
            if not audio_recheck.passed:
                blockers.append(
                    str(audio_recheck.reason or "master_audio_preflight_failed")
                )
        console_change_fence: Dict[str, Any] = {}
        if not blockers:
            console_change_fence = (
                await self._establish_soundcheck_console_change_fence(
                    operations,
                    snapshots,
                    adapters,
                    additional_observed_values={
                        **self._soundcheck_route_fence_values(audio_recheck),
                        **final_source_path_fence_values,
                    },
                    capture_status=(
                        dict(audio_recheck.evidence).get(
                            "capture_status_after"
                        )
                        if audio_recheck is not None
                        and isinstance(audio_recheck.evidence, Mapping)
                        else None
                    ),
                    require_global_generation=True,
                )
            )
            if console_change_fence.get("verified") is not True:
                blockers.append(
                    "soundcheck_final_reconciliation_change_fence_failed:"
                    + str(
                        console_change_fence.get("reason")
                        or "unknown_change_fence_failure"
                    )
                )
        active_run_id = self._current_active_soundcheck_run_id()
        if active_run_id and run_id != active_run_id:
            superseded = self._soundcheck_global_blockers.setdefault(run_id, [])
            if "soundcheck_run_superseded" not in superseded:
                superseded.append("soundcheck_run_superseded")
            inactive = self._aggregate_current_soundcheck_readiness(run_id)
            return dict(inactive) if isinstance(inactive, Mapping) else None
        reconciled_at = time.time()
        reconciliation_max_age_sec = 30.0
        oldest_snapshot_at = min(
            (
                float(item.get("captured_at_epoch"))
                for item in snapshots.values()
                if isinstance(item.get("captured_at_epoch"), (int, float))
                and not isinstance(item.get("captured_at_epoch"), bool)
            ),
            default=reconciled_at,
        )
        if snapshots and reconciled_at - oldest_snapshot_at > reconciliation_max_age_sec:
            blockers.append("soundcheck_final_reconciliation_collection_stale")
        if reconciled_at - reconciliation_started_at > reconciliation_max_age_sec:
            blockers.append("soundcheck_final_reconciliation_collection_stale")
        all_blockers = sorted(set(persistent_blockers + blockers))
        assessment = assess_soundcheck_manifest(
            manifest,
            snapshots_by_operation=snapshots,
            adapters=adapters,
            global_blockers=all_blockers,
        )
        assessment_by_id = {
            str(item.get("operation_id") or ""): item
            for item in assessment.get("assessments") or []
            if isinstance(item, Mapping)
        }
        outcomes = self._soundcheck_outcomes.setdefault(run_id, {})
        for operation in operations:
            outcome = outcomes.get(operation.operation_id)
            if not isinstance(outcome, dict):
                continue
            preliminarily_terminal = bool(
                (
                    outcome.get("status") == "applied"
                    and outcome.get("physical_verified") is True
                    and outcome.get("audio_recheck_passed") is True
                    and outcome.get("wal_terminal") == "committed"
                )
                or (
                    outcome.get("disposition") == "already_satisfied"
                    and outcome.get("physical_verified") is True
                )
            )
            if not preliminarily_terminal:
                continue
            current = assessment_by_id.get(operation.operation_id, {})
            snapshot_captured_at = snapshots.get(operation.operation_id, {}).get(
                "captured_at_epoch"
            )
            snapshot_fresh = bool(
                isinstance(snapshot_captured_at, (int, float))
                and not isinstance(snapshot_captured_at, bool)
                and 0.0
                <= reconciled_at - float(snapshot_captured_at)
                <= reconciliation_max_age_sec
            )
            source_path_verified = True
            source_path_reason = ""
            if (
                operation.family_id == "fader"
                and operation.write_kind == "fader"
                and operation.channel in final_selected_source_channels
            ):
                source_path = final_selected_source_paths.get(
                    str(operation.channel),
                    {},
                )
                source_path_captured_at = source_path.get("captured_at_epoch")
                source_path_verified = bool(
                    source_path.get("physical_verified") is True
                    and source_path.get("two_pass_consistent") is True
                    and source_path.get("matches_analysis_baseline") is True
                    and source_path.get("matches_analysis_main_send") is True
                    and source_path.get("channel_main_path_locally_enabled") is True
                    and source_path.get("downstream_dca_path_verified") is True
                    and isinstance(source_path_captured_at, (int, float))
                    and not isinstance(source_path_captured_at, bool)
                    and 0.0
                    <= reconciled_at - float(source_path_captured_at)
                    <= reconciliation_max_age_sec
                )
                source_path_reason = str(
                    source_path.get("reason")
                    or "selected_source_audible_path_unverified"
                )
            verified = bool(
                current.get("disposition") == "already_satisfied"
                and current.get("physical_verified") is True
                and current.get("audible_state_verified") is True
                and snapshot_fresh
                and source_path_verified
                and not all_blockers
            )
            outcome["final_reconciliation_verified"] = verified
            outcome["reconciliation_captured_at_epoch"] = snapshot_captured_at
            outcome["reconciliation_max_age_sec"] = reconciliation_max_age_sec
            outcome["audible_state_verified"] = (
                current.get("audible_state_verified") is True
            )
            outcome["final_reconciliation_reason"] = (
                "physical_target_and_audible_state_reconciled"
                if verified
                else (
                    source_path_reason
                    if not source_path_verified
                    else str(
                        current.get("reason")
                        or "final_physical_reconciliation_failed"
                    )
                )
            )

        audio_evidence = (
            dict(audio_recheck.evidence)
            if audio_recheck is not None
            and isinstance(audio_recheck.evidence, Mapping)
            else {}
        )
        for item in assessment.get("assessments") or []:
            if isinstance(item, dict):
                snapshot = snapshots.get(str(item.get("operation_id") or ""), {})
                item["captured_at_epoch"] = snapshot.get(
                    "captured_at_epoch",
                    oldest_snapshot_at,
                )
                item["max_age_sec"] = reconciliation_max_age_sec
        previous = self._soundcheck_preflight_assessments.get(run_id, {})
        assessment["source_coverage"] = dict(
            previous.get("source_coverage")
            or manifest.get("source_coverage")
            or {}
        )
        assessment["main_47_48_evidence"] = (
            self._soundcheck_main_evidence_from_audio(audio_evidence)
        )
        assessment["selected_source_audible_paths"] = (
            final_selected_source_paths
            if selected_source_path_required
            else dict(previous.get("selected_source_audible_paths") or {})
        )
        assessment["console_change_fence"] = console_change_fence
        assessment["source_balance_analysis_deadline_epoch"] = previous.get(
            "source_balance_analysis_deadline_epoch"
        )
        assessment["source_balance_deadline_by_transaction"] = dict(
            previous.get("source_balance_deadline_by_transaction") or {}
        )
        assessment["captured_at_epoch"] = oldest_snapshot_at
        assessment["completed_at_epoch"] = time.time()
        assessment["max_age_sec"] = reconciliation_max_age_sec
        self._soundcheck_preflight_assessments[run_id] = assessment
        self._soundcheck_preflight_timestamps[run_id] = oldest_snapshot_at
        self._soundcheck_global_blockers[run_id] = all_blockers
        readiness = self._aggregate_current_soundcheck_readiness(run_id)
        if isinstance(readiness, Mapping):
            self.last_soundcheck_readiness = dict(readiness)
            if (
                console_change_fence.get("verified") is True
                and readiness.get("mix_readiness")
                in {"safe_core_ready", "show_mix_ready"}
            ):
                fence_armed = self._arm_soundcheck_console_change_invalidator(
                    run_id,
                    console_change_fence,
                )
                if not fence_armed:
                    invalidated = self._aggregate_current_soundcheck_readiness(
                        run_id
                    )
                    if isinstance(invalidated, Mapping):
                        self.last_soundcheck_readiness = dict(invalidated)
                        return dict(invalidated)
            return dict(readiness)
        return None

    async def finalize_soundcheck_readiness_if_no_apply(
        self,
        run_id: str,
    ) -> Dict[str, Any] | None:
        """Run the read-only terminal checkpoint when no child needs a write."""

        manifest = self._soundcheck_manifests.get(str(run_id or "").strip())
        if not isinstance(manifest, Mapping):
            return None
        active_run_id = self._current_active_soundcheck_run_id()
        if active_run_id and run_id != active_run_id:
            blockers = self._soundcheck_global_blockers.setdefault(run_id, [])
            if "soundcheck_run_superseded" not in blockers:
                blockers.append("soundcheck_run_superseded")
            inactive = self._aggregate_current_soundcheck_readiness(run_id)
            return dict(inactive) if isinstance(inactive, Mapping) else None
        preflight = self._soundcheck_preflight_assessments.get(run_id, {})
        assessments = [
            item
            for item in preflight.get("assessments") or []
            if isinstance(item, Mapping)
        ]
        if any(item.get("disposition") == "apply" for item in assessments):
            return self._aggregate_current_soundcheck_readiness(run_id)
        candidate = self._aggregate_current_soundcheck_readiness(
            run_id,
            require_final_reconciliation=False,
        )
        if not isinstance(candidate, Mapping):
            return None
        logical_counts = candidate.get("logical_counts") or {}
        nonterminal = sum(
            int(logical_counts.get(key) or 0)
            for key in ("pending", "operator_review", "forbidden")
        )
        if nonterminal:
            self.last_soundcheck_readiness = dict(candidate)
            return dict(candidate)
        return await self.reconcile_soundcheck_readiness(run_id)

    def record_soundcheck_child_result(
        self,
        proposal: Any,
        apply_result: Mapping[str, Any],
    ) -> Dict[str, Any] | None:
        """Aggregate one terminal child result without inferring mix success."""
        raw = getattr(proposal, "raw", None)
        if not isinstance(raw, Mapping):
            return None
        plan = raw.get("supervised_transaction_plan")
        if not isinstance(plan, Mapping):
            return None
        run_id = str(plan.get("run_id") or "").strip()
        manifest = self._soundcheck_manifests.get(run_id)
        if not run_id or not isinstance(manifest, Mapping):
            return None
        active_run_id = self._current_active_soundcheck_run_id()
        if active_run_id and run_id != active_run_id:
            blockers = self._soundcheck_global_blockers.setdefault(run_id, [])
            if "soundcheck_run_superseded" not in blockers:
                blockers.append("soundcheck_run_superseded")
            return None
        transaction_id = str(plan.get("transaction_id") or "").strip()
        stored_plan = next(
            (
                item
                for item in manifest.get("plans") or []
                if isinstance(item, Mapping)
                and str(item.get("transaction_id") or "") == transaction_id
            ),
            None,
        )
        identity_matches = bool(
            isinstance(stored_plan, Mapping)
            and plan.get("run_id") == manifest.get("run_id")
            and plan.get("plan_revision") == manifest.get("plan_revision")
            and plan.get("plan_fingerprint") == manifest.get("plan_fingerprint")
            and dict(plan) == dict(stored_plan)
        )
        if not identity_matches:
            logger.warning(
                "Rejected stale or foreign soundcheck child result: run=%s transaction=%s",
                run_id,
                transaction_id,
            )
            return None
        logical_ids = [
            str(item)
            for item in stored_plan.get("logical_correction_ids") or []
            if str(item).strip()
        ]
        stored_operation_ids = [
            str(operation.get("operation_id") or "")
            for operation in stored_plan.get("operations") or []
            if isinstance(operation, Mapping)
        ]
        if logical_ids != stored_operation_ids or not logical_ids:
            logger.warning(
                "Rejected soundcheck child with inconsistent logical identity: %s",
                transaction_id,
            )
            return None
        status = str(apply_result.get("status") or "")
        reason = str(apply_result.get("reason") or status or "unknown_child_result")
        applied_operation_ids = {
            str(value or "").strip()
            for value in apply_result.get("applied_operation_ids") or []
            if str(value or "").strip()
        }
        skipped_operation_ids = {
            str(event.get("operation_id") or "").strip()
            for event in apply_result.get("events") or []
            if isinstance(event, Mapping)
            and event.get("phase") == "apply"
            and isinstance(event.get("result"), Mapping)
            and event["result"].get("skipped") is True
        }
        if (
            status == "applied"
            and not applied_operation_ids
            and not skipped_operation_ids
            and len(logical_ids) == 1
        ):
            applied_operation_ids = {logical_ids[0]}
        base_outcome: Dict[str, Any] = {
            "status": status,
            "reason": reason,
            "physical_verified": apply_result.get("physical_verified") is True,
            "audio_recheck_passed": apply_result.get("audio_recheck_passed") is True,
            "audible_state_verified": apply_result.get("audible_state_verified") is True,
            "wal_terminal": str(apply_result.get("wal_terminal") or "not_committed"),
            "write_state_unknown": apply_result.get("write_state_unknown") is True,
            "physical_step_count": 0,
            "transaction_physical_step_count": (
                len(applied_operation_ids)
            ),
            "final_reconciliation_verified": False,
        }
        if status != "applied":
            scope, disposition, _required_evidence = classify_transition_reason(reason)
            base_outcome["disposition"] = disposition
            if scope == "global":
                blockers = self._soundcheck_global_blockers.setdefault(run_id, [])
                if reason not in blockers:
                    blockers.append(reason)
        for logical_id in logical_ids:
            normalized = str(logical_id or "").strip()
            if normalized:
                member_outcome = dict(base_outcome)
                if status == "applied" and normalized in skipped_operation_ids:
                    member_outcome.update(
                        {
                            "status": "preflight_classified",
                            "disposition": "already_satisfied",
                            "reason": "physical_target_already_satisfied",
                            "physical_verified": True,
                            "audio_recheck_passed": False,
                            "audible_state_verified": True,
                            "wal_terminal": "not_applicable",
                            "physical_step_count": 0,
                        }
                    )
                elif status == "applied" and normalized in applied_operation_ids:
                    member_outcome["physical_step_count"] = 1
                elif status == "applied":
                    member_outcome.update(
                        {
                            "status": "manual_review_required",
                            "disposition": "operator_review",
                            "reason": "transaction_member_outcome_missing",
                            "physical_verified": False,
                        }
                    )
                self._soundcheck_outcomes.setdefault(run_id, {})[normalized] = dict(
                    member_outcome
                )
                journal_path = self._append_soundcheck_parent_journal(
                    "child_outcome",
                    run_id,
                    plan_revision=manifest.get("plan_revision"),
                    plan_fingerprint=manifest.get("plan_fingerprint"),
                    manifest_content_sha256=manifest.get(
                        "manifest_content_sha256"
                    ),
                    transaction_id=transaction_id,
                    operation_id=normalized,
                    outcome=member_outcome,
                )
                if not journal_path:
                    blockers = self._soundcheck_global_blockers.setdefault(run_id, [])
                    reason = "durable_parent_outcome_record_failed"
                    if reason not in blockers:
                        blockers.append(reason)

        readiness = self._aggregate_current_soundcheck_readiness(run_id)
        if readiness is None:
            return None
        self.last_soundcheck_readiness = readiness
        return readiness

    def _quarantined_write_operator_correction(
        self,
        *,
        surface: str,
        result_type: str,
        request: Optional[Dict[str, Any]],
        extra: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        method_id = QUARANTINED_WRITE_SURFACE_OPERATOR_METHODS.get(surface)
        if not method_id:
            return None

        request_payload = dict(request or {})
        extra_payload = dict(extra or {})
        target = (
            request_payload.get("channel")
            or request_payload.get("target_bus")
            or request_payload.get("snap_name")
            or request_payload.get("snapshot_path")
            or request_payload.get("output_group")
            or surface
        )
        blockers = [
            "deployment_write_surface_quarantined",
            "not_in_supervised_fader_gain_executor",
            "hardware_readback_and_rollback_not_proven",
        ]
        proposed_action = {
            "action_type": f"blocked_direct_write:{surface}",
            "surface": surface,
            "result_type": result_type,
            "request": request_payload,
            "operator_confirmation_required": True,
        }
        proposal = {
            "id": f"quarantined_wing_write:{surface}",
            "method_id": method_id,
            "source": "wing_deployment_boundary",
            "title": f"{surface}: blocked WING write",
            "reason": (
                "Direct WING write surface is quarantined; this is "
                "operator-review evidence only."
            ),
            "target": target,
            "kind": "quarantined_write_surface",
            "severity": "high",
            "confidence": 1.0,
            "blockers": blockers,
            "raw": {
                "surface": surface,
                "result_type": result_type,
                "request": request_payload,
                "extra": extra_payload,
            },
            "evidence": {
                "live_write": False,
                "queue_mutation": False,
                "deployment_write_surface_quarantined": True,
                "readback_required_before_promotion": True,
                "rollback_required_before_promotion": True,
            },
        }
        return build_operator_correction_payload(
            proposal=proposal,
            method_id=method_id,
            target=target,
            proposed_action=proposed_action,
            blockers=blockers,
        )

    async def _block_quarantined_wing_write_surface(
        self,
        websocket: Optional[WebSocketServerProtocol],
        *,
        surface: str,
        result_type: str,
        request: Optional[Dict[str, Any]] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if not self._is_wing_deployment_write_boundary_active():
            return False

        payload: Dict[str, Any] = {
            "type": result_type,
            "status": "blocked",
            "success": False,
            "reason": "deployment_write_surface_quarantined",
            "surface": surface,
            "message": (
                "WING deployment path is limited to approved manual fader/gain writes; "
                f"{surface} is quarantined."
            ),
            "manual_write_supervision": self.get_manual_write_supervision_status(),
        }
        if request:
            payload["request"] = request
        if extra:
            payload.update(extra)
        operator_correction = self._quarantined_write_operator_correction(
            surface=surface,
            result_type=result_type,
            request=request,
            extra=extra,
        )
        if operator_correction:
            payload["operator_correction"] = operator_correction

        logger.warning(
            "Blocked quarantined WING deployment write surface",
            extra={
                "event": "wing_deployment_boundary_blocked",
                "surface": surface,
                "result_type": result_type,
            },
        )
        if websocket is not None:
            await self.send_to_client(websocket, payload)
        return True

    async def _block_unpromoted_maintenance_surface(
        self,
        websocket: Optional[WebSocketServerProtocol],
        *,
        surface: str,
        result_type: str,
        requested_channel_count: int = 0,
        dry_run: Any = None,
        confirm_maintenance: bool = False,
        rollback_snapshot_path: Any = None,
    ) -> None:
        """Fail closed for legacy bulk writes not migrated to the safe executor.

        These message types used to call direct multi-parameter mutation methods.
        The handlers now route through ``request_*`` entrypoints, but those
        entrypoints must not silently restore the bypass around the supervised
        WING transaction path.  Keep the legacy implementation callable for
        offline inventory tests while the WebSocket product surface remains
        explicitly unavailable.
        """

        request_summary = {
            "requested_channel_count": max(0, int(requested_channel_count or 0)),
            "dry_run_requested": dry_run is True,
            "maintenance_confirmation_supplied": confirm_maintenance is True,
            "rollback_plan_present": bool(rollback_snapshot_path),
        }
        if await self._block_quarantined_wing_write_surface(
            websocket,
            surface=surface,
            result_type=result_type,
            request=request_summary,
        ):
            return

        await self.send_to_client(
            websocket,
            {
                "type": result_type,
                "status": "blocked",
                "success": False,
                "blocked": True,
                "reason": "legacy_maintenance_surface_not_promoted",
                "blocked_reason": "legacy_maintenance_surface_not_promoted",
                "surface": surface,
                "send_attempted": False,
                "dry_run_only": dry_run is True,
                **request_summary,
                "message": (
                    "This bulk maintenance action is not exposed because it has "
                    "not been migrated to the supervised readback/rollback executor."
                ),
            },
        )

    async def request_reset_trim(
        self,
        websocket,
        channels: list,
        *,
        dry_run: Any = None,
        confirm_maintenance: bool = False,
        rollback_snapshot_path: Any = None,
    ) -> None:
        await self._block_unpromoted_maintenance_surface(
            websocket,
            surface="reset_trim",
            result_type="reset_trim_result",
            requested_channel_count=len(channels) if isinstance(channels, list) else 0,
            dry_run=dry_run,
            confirm_maintenance=confirm_maintenance,
            rollback_snapshot_path=rollback_snapshot_path,
        )

    async def request_bypass_mixer(
        self,
        websocket,
        *,
        dry_run: Any = None,
        confirm_maintenance: bool = False,
        rollback_snapshot_path: Any = None,
    ) -> None:
        await self._block_unpromoted_maintenance_surface(
            websocket,
            surface="bypass_mixer",
            result_type="bypass_result",
            requested_channel_count=40,
            dry_run=dry_run,
            confirm_maintenance=confirm_maintenance,
            rollback_snapshot_path=rollback_snapshot_path,
        )

    async def request_reset_all_eq(
        self,
        websocket,
        data: Optional[Dict[str, Any]] = None,
        *,
        dry_run: Any = None,
        confirm_maintenance: bool = False,
        rollback_snapshot_path: Any = None,
    ) -> None:
        channels = data.get("channels", []) if isinstance(data, dict) else []
        await self._block_unpromoted_maintenance_surface(
            websocket,
            surface="reset_all_eq",
            result_type="reset_all_eq_result",
            requested_channel_count=len(channels) if isinstance(channels, list) else 0,
            dry_run=dry_run,
            confirm_maintenance=confirm_maintenance,
            rollback_snapshot_path=rollback_snapshot_path,
        )

    async def request_reset_phase_delay(
        self,
        websocket,
        channels: Optional[List[int]] = None,
        *,
        dry_run: Any = None,
        confirm_maintenance: bool = False,
        rollback_snapshot_path: Any = None,
    ) -> None:
        await self._block_unpromoted_maintenance_surface(
            websocket,
            surface="reset_phase_delay",
            result_type="phase_alignment_reset_result",
            requested_channel_count=len(channels) if isinstance(channels, list) else 0,
            dry_run=dry_run,
            confirm_maintenance=confirm_maintenance,
            rollback_snapshot_path=rollback_snapshot_path,
        )

    def _is_voice_command_write_surface(self, cmd_type: Any) -> bool:
        return str(cmd_type or "").strip().lower() in {
            "set_fader",
            "set_gain",
            "set_pan",
            "load_snap",
            "mute_channel",
            "volume_up",
            "volume_down",
            "eq_on",
            "eq_band_up",
            "eq_band_down",
            "compressor_on",
            "compressor_threshold",
            "compressor_gain_up",
            "compressor_gain_down",
        }

    def _normalize_voice_command_payload(self, command: Any) -> Dict[str, Any]:
        if isinstance(command, Mapping):
            return dict(command)
        return {"raw_command": command}

    async def _block_quarantined_voice_command_if_needed(self, command: Dict[str, Any]) -> bool:
        command = self._normalize_voice_command_payload(command)
        cmd_type = str(command.get("type") or "").strip().lower()
        if (
            not self._is_wing_deployment_write_boundary_active()
            or not self._is_voice_command_write_surface(cmd_type)
        ):
            return False

        request_payload = dict(command)
        surface = f"voice_command:{cmd_type}"
        payload = {
            "type": "voice_command_blocked",
            "status": "blocked",
            "success": False,
            "reason": "deployment_write_surface_quarantined",
            "surface": surface,
            "command": cmd_type,
            "request": request_payload,
            "message": (
                "WING deployment path is limited to Product Layer proposals and "
                "approved manual fader/gain/pan writes; direct voice-command writes "
                "are quarantined."
            ),
            "manual_write_supervision": self.get_manual_write_supervision_status(),
        }
        operator_correction = self._quarantined_write_operator_correction(
            surface=surface,
            result_type="voice_command_blocked",
            request=request_payload,
            extra={"command": cmd_type},
        )
        if operator_correction:
            payload["operator_correction"] = operator_correction
        artifact_path = self._record_manual_write_event("voice_command_blocked", payload)
        if artifact_path:
            payload["artifact_path"] = artifact_path
        await self.broadcast(payload)
        return True

    async def _broadcast_voice_command_write_failed(
        self,
        command: Dict[str, Any],
        *,
        value_type: str,
        requested_value: Any,
        reason: str = "mixer_write_failed",
    ) -> None:
        command = self._normalize_voice_command_payload(command)
        cmd_type = str(command.get("type") or "").strip().lower()
        payload = {
            "type": "voice_command_blocked",
            "status": "blocked",
            "success": False,
            "reason": reason,
            "surface": f"voice_command:{cmd_type}",
            "command": cmd_type,
            "channel": command.get("channel"),
            "value_type": value_type,
            "requested_value": requested_value,
            "live_write": False,
            "live_authority_granted": False,
            "readiness_closure_allowed": False,
            "request": dict(command),
        }
        await self.broadcast(payload)

    async def _broadcast_voice_command_error(
        self,
        command: Dict[str, Any],
        *,
        reason: str,
        error: Optional[str] = None,
    ) -> None:
        command = self._normalize_voice_command_payload(command)
        cmd_type = str(command.get("type") or "").strip().lower()
        payload = {
            "type": "voice_command_error",
            "status": "error",
            "success": False,
            "reason": reason,
            "surface": f"voice_command:{cmd_type}" if cmd_type else "voice_command",
            "command": command,
            "command_type": cmd_type,
            "channel": command.get("channel"),
            "request": dict(command),
            "live_write": False,
            "live_authority_granted": False,
            "readiness_closure_allowed": False,
        }
        if error is not None:
            payload["error"] = error
        await self.broadcast(payload)

    def _manual_write_blocked_result(
        self,
        *,
        reason: str,
        channel: Any = None,
        value: Any = None,
        value_type: Any = None,
        source: str = "",
        approval_id: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
        event_type: str = "manual_write_blocked",
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "status": "blocked",
            "success": False,
            "reason": reason,
            "channel": channel,
            "value_type": value_type,
            "requested_value": value,
            "proof_status": "blocked",
            "source": source,
            "approval_id": approval_id,
            "manual_write_supervision": self.get_manual_write_supervision_status(),
        }
        if extra:
            result.update(extra)
        artifact_path = self._record_manual_write_event(event_type, result)
        if artifact_path:
            result["artifact_path"] = artifact_path
        return result

    def _record_manual_write_event(self, event_type: str, result: Dict[str, Any]) -> Optional[str]:
        try:
            self._manual_write_artifacts_dir.mkdir(parents=True, exist_ok=True)
            path = self._manual_write_artifacts_dir / "manual_write_events.jsonl"
            event = {
                "timestamp": _utc_timestamp(),
                "event_type": str(event_type),
                "result": convert_numpy_types(result),
            }
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            return str(path)
        except Exception as exc:
            logger.warning("Could not record manual write event: %s", exc, exc_info=True)
            return None

    def _manual_write_wal_path(self) -> Path:
        return self._console_write_wal_path

    def _append_soundcheck_parent_journal(
        self,
        event_type: str,
        run_id: str,
        **payload: Any,
    ) -> Optional[str]:
        """Durably bind parent revisions and logical child outcomes."""

        event_type = str(event_type or "").strip()
        run_id = str(run_id or "").strip()
        if not event_type or not run_id:
            return None
        try:
            self._ensure_durable_directory(self._console_runtime_dir)
            path = self._soundcheck_parent_journal_path
            event = {
                "schema": "automixer.soundcheck_parent_journal.v1",
                "timestamp": _utc_timestamp(),
                "event_type": event_type,
                "run_id": run_id,
                **convert_numpy_types(payload),
            }
            with path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n"
                )
                handle.flush()
                os.fsync(handle.fileno())
            self._fsync_directory(path.parent)
            return str(path)
        except Exception as exc:
            logger.error(
                "Could not durably append soundcheck parent journal: %s",
                exc,
                exc_info=True,
            )
            return None

    def _recover_soundcheck_parent_journal_state(self) -> None:
        """Restore identity/outcome truth but require new physical preflight."""

        path = self._soundcheck_parent_journal_path
        if not path.exists():
            return
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, raw_line in enumerate(handle, start=1):
                    if not raw_line.strip():
                        continue
                    event = json.loads(raw_line)
                    if not isinstance(event, Mapping) or event.get("schema") != (
                        "automixer.soundcheck_parent_journal.v1"
                    ):
                        raise ValueError(
                            f"invalid_soundcheck_parent_journal_record:{line_number}"
                        )
                    run_id = str(event.get("run_id") or "").strip()
                    if not run_id:
                        raise ValueError(
                            f"soundcheck_parent_journal_run_id_missing:{line_number}"
                        )
                    event_type = str(event.get("event_type") or "")
                    if event_type == "manifest_registered":
                        manifest = event.get("manifest")
                        if not isinstance(manifest, Mapping):
                            raise ValueError(
                                f"soundcheck_parent_manifest_missing:{line_number}"
                            )
                        revision = manifest.get("plan_revision")
                        fingerprint = str(
                            manifest.get("plan_fingerprint") or ""
                        ).strip()
                        if (
                            manifest.get("run_id") != run_id
                            or isinstance(revision, bool)
                            or not isinstance(revision, int)
                            or revision < 1
                            or not fingerprint
                            or soundcheck_manifest_logical_id_errors(manifest)
                            or manifest.get("schema_version")
                            != "soundcheck_batch_manifest/v2"
                            or not str(
                                manifest.get("manifest_content_sha256") or ""
                            ).strip()
                            or manifest.get("manifest_content_sha256")
                            != soundcheck_manifest_content_sha256(manifest)
                        ):
                            raise ValueError(
                                f"invalid_soundcheck_parent_manifest:{line_number}"
                            )
                        current = self._soundcheck_manifests.get(run_id)
                        if not isinstance(current, Mapping):
                            self._soundcheck_manifests[run_id] = copy.deepcopy(
                                manifest
                            )
                            self._soundcheck_outcomes[run_id] = {}
                            self._soundcheck_global_blockers[run_id] = []
                        else:
                            current_revision = int(
                                current.get("plan_revision") or 0
                            )
                            if revision == current_revision:
                                if dict(manifest) != dict(current):
                                    raise ValueError(
                                        "soundcheck_parent_manifest_revision_collision:"
                                        f"{line_number}"
                                    )
                            elif revision == current_revision + 1:
                                self._soundcheck_manifests[run_id] = (
                                    copy.deepcopy(manifest)
                                )
                                self._soundcheck_outcomes[run_id] = {}
                                self._soundcheck_global_blockers[run_id] = []
                            else:
                                raise ValueError(
                                    "soundcheck_parent_manifest_revision_not_monotonic:"
                                    f"{line_number}"
                                )
                    elif event_type == "child_outcome":
                        manifest = self._soundcheck_manifests.get(run_id)
                        transaction_id = str(
                            event.get("transaction_id") or ""
                        ).strip()
                        operation_id = str(
                            event.get("operation_id") or ""
                        ).strip()
                        outcome = event.get("outcome")
                        manifest_child = next(
                            (
                                plan
                                for plan in (manifest or {}).get("plans") or []
                                if isinstance(plan, Mapping)
                                and str(plan.get("transaction_id") or "")
                                == transaction_id
                            ),
                            None,
                        )
                        child_operation_ids = {
                            str(operation.get("operation_id") or "")
                            for operation in (
                                manifest_child.get("operations") or []
                                if isinstance(manifest_child, Mapping)
                                else []
                            )
                            if isinstance(operation, Mapping)
                            and str(operation.get("operation_id") or "").strip()
                        }
                        if not (
                            isinstance(manifest, Mapping)
                            and event.get("plan_revision")
                            == manifest.get("plan_revision")
                            and event.get("plan_fingerprint")
                            == manifest.get("plan_fingerprint")
                            and event.get("manifest_content_sha256")
                            == manifest.get("manifest_content_sha256")
                            and isinstance(manifest_child, Mapping)
                            and transaction_id
                            and operation_id
                            and operation_id in child_operation_ids
                            and isinstance(outcome, Mapping)
                        ):
                            continue
                        restored = dict(outcome)
                        restored["final_reconciliation_verified"] = False
                        restored["restored_from_parent_journal"] = True
                        self._soundcheck_outcomes.setdefault(run_id, {})[
                            operation_id
                        ] = restored
            for run_id in self._soundcheck_manifests:
                blockers = self._soundcheck_global_blockers.setdefault(run_id, [])
                if "soundcheck_restart_requires_fresh_preflight_and_reimport" not in blockers:
                    blockers.append(
                        "soundcheck_restart_requires_fresh_preflight_and_reimport"
                    )
            if self._soundcheck_manifests:
                recovered_run_id = next(reversed(self._soundcheck_manifests))
                self._active_soundcheck_run_id = recovered_run_id
                recovered_readiness = self._aggregate_current_soundcheck_readiness(
                    recovered_run_id
                )
                if isinstance(recovered_readiness, Mapping):
                    self.last_soundcheck_readiness = dict(recovered_readiness)
        except Exception as exc:
            self._soundcheck_parent_journal_error = str(exc)
            logger.error(
                "Soundcheck parent journal recovery failed: %s",
                exc,
                exc_info=True,
            )

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        """Persist directory entries after creating or extending durable state."""

        flags = os.O_RDONLY
        flags |= getattr(os, "O_DIRECTORY", 0)
        flags |= getattr(os, "O_CLOEXEC", 0)
        directory_fd = os.open(path, flags)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)

    @classmethod
    def _ensure_durable_directory(cls, path: Path, *, mode: int = 0o700) -> None:
        """Create a directory tree and persist every newly created parent entry."""

        path = Path(path)
        missing: list[Path] = []
        cursor = path
        while not cursor.exists():
            missing.append(cursor)
            if cursor.parent == cursor:
                break
            cursor = cursor.parent
        path.mkdir(parents=True, exist_ok=True, mode=mode)
        if path.is_symlink() or not path.is_dir():
            raise RuntimeError("console runtime path must be a real directory")
        os.chmod(path, mode)
        # mkdir(parents=True) creates outermost first. Persist each new child
        # entry in its parent in the same order.
        for created in reversed(missing):
            cls._fsync_directory(created.parent)

    def _append_manual_write_wal(
        self,
        phase: str,
        transaction_id: str,
        **payload: Any,
    ) -> Optional[str]:
        """Durably append one state transition before/after a physical write."""

        transaction_id = str(transaction_id or "").strip()
        phase = str(phase or "").strip()
        if not transaction_id or not phase:
            return None
        try:
            self._ensure_durable_directory(self._console_runtime_dir)
            path = self._manual_write_wal_path()
            event = {
                "schema": "automixer.manual_write_wal.v1",
                "timestamp": _utc_timestamp(),
                "transaction_id": transaction_id,
                "phase": phase,
                **convert_numpy_types(payload),
            }
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._fsync_directory(path.parent)
            if phase in MANUAL_WRITE_WAL_TERMINAL_PHASES:
                self._manual_write_recovery_transactions.pop(transaction_id, None)
                self._manual_write_prepared_transactions.pop(transaction_id, None)
            else:
                if phase == "prepared":
                    self._manual_write_prepared_transactions[transaction_id] = event
                prepared = self._manual_write_prepared_transactions.get(
                    transaction_id,
                    {},
                )
                self._manual_write_recovery_transactions[transaction_id] = {
                    **prepared,
                    "phase": phase,
                    "latest_event": event,
                }
            proposal_id = str(event.get("proposal_id") or "").strip()
            if proposal_id:
                self._durable_operator_proposal_ids.add(proposal_id)
            idempotency_key = str(event.get("idempotency_key") or "").strip()
            if (
                phase == "prepared"
                and idempotency_key
                and event.get("scope") == "operator_proposal"
            ):
                self._durable_operator_apply_idempotency_keys[idempotency_key] = {
                    "created_at": event.get("timestamp"),
                    "proposal_id": event.get("proposal_id"),
                    "approval_id": event.get("approval_id"),
                    "restored_from_wal": True,
                    "transaction_id": transaction_id,
                }
            return str(path)
        except Exception as exc:
            logger.error(
                "Could not durably append manual write WAL: %s",
                exc,
                exc_info=True,
            )
            return None

    def _recover_manual_write_wal_state(self) -> None:
        """Latch writes when the prior process left a non-terminal write record."""

        path = self._manual_write_wal_path()
        if not path.exists():
            return
        latest: Dict[str, Dict[str, Any]] = {}
        prepared_records: Dict[str, Dict[str, Any]] = {}
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line_number, raw_line in enumerate(handle, start=1):
                    if not raw_line.strip():
                        continue
                    try:
                        event = json.loads(raw_line)
                    except (TypeError, ValueError, json.JSONDecodeError) as exc:
                        latest[f"wal_parse_error:{line_number}"] = {
                            "phase": "wal_parse_error",
                            "error": str(exc),
                            "line_number": line_number,
                        }
                        continue
                    transaction_id = str(event.get("transaction_id") or "").strip()
                    phase = str(event.get("phase") or "").strip()
                    if not transaction_id or not phase:
                        latest[f"wal_invalid_record:{line_number}"] = {
                            "phase": "wal_invalid_record",
                            "line_number": line_number,
                        }
                        continue
                    idempotency_key = str(
                        event.get("idempotency_key") or ""
                    ).strip()
                    proposal_id = str(event.get("proposal_id") or "").strip()
                    if proposal_id:
                        self._durable_operator_proposal_ids.add(proposal_id)
                    if idempotency_key:
                        if event.get("scope") == "supervised_transaction":
                            self._supervised_transaction_idempotency_keys.setdefault(
                                idempotency_key,
                                transaction_id,
                            )
                            self._supervised_transaction_ids.setdefault(
                                transaction_id,
                                idempotency_key,
                            )
                        else:
                            durable_record = {
                                "created_at": event.get("timestamp"),
                                "proposal_id": event.get("proposal_id"),
                                "approval_id": event.get("approval_id"),
                                "restored_from_wal": True,
                                "transaction_id": transaction_id,
                            }
                            self._durable_operator_apply_idempotency_keys[
                                idempotency_key
                            ] = durable_record
                            self._operator_apply_idempotency_keys.setdefault(
                                idempotency_key,
                                durable_record,
                            )
                    if phase == "prepared":
                        prepared_records[transaction_id] = event
                    latest[transaction_id] = event
        except OSError as exc:
            latest["wal_read_error"] = {
                "phase": "wal_read_error",
                "error": str(exc),
            }

        unresolved = {
            transaction_id: {
                **prepared_records.get(transaction_id, {}),
                "phase": str(event.get("phase") or ""),
                "latest_event": event,
            }
            for transaction_id, event in latest.items()
            if str(event.get("phase") or "")
            not in MANUAL_WRITE_WAL_TERMINAL_PHASES
        }
        self._manual_write_prepared_transactions = {
            transaction_id: prepared_records[transaction_id]
            for transaction_id in unresolved
            if transaction_id in prepared_records
        }
        self._manual_write_recovery_transactions = unresolved
        if unresolved:
            self._manual_write_emergency_stopped = True
            self._manual_write_emergency_generation += 1
            logger.error(
                "Manual write WAL recovery required for %d transaction(s)",
                len(unresolved),
            )

    @staticmethod
    def _wal_values_match(expected: Any, actual: Any, tolerance: float) -> bool:
        if isinstance(expected, bool):
            parsed = AutoMixerServer._coerce_optional_bool(actual)
            return parsed is expected
        try:
            expected_number = float(expected)
            actual_number = float(actual)
        except (TypeError, ValueError):
            return str(expected).strip().upper() == str(actual).strip().upper()
        return (
            math.isfinite(expected_number)
            and math.isfinite(actual_number)
            and abs(expected_number - actual_number) <= tolerance
        )

    def _wal_recovery_writes(
        self,
        recovery: Mapping[str, Any],
    ) -> List[Dict[str, Any]]:
        writes: List[Dict[str, Any]] = []
        if recovery.get("scope") == "supervised_transaction":
            for operation in recovery.get("operations") or []:
                if not isinstance(operation, Mapping):
                    continue
                preimages = dict(operation.get("preimage_values") or {})
                for write in operation.get("writes") or []:
                    if not isinstance(write, Mapping):
                        continue
                    address = str(write.get("address") or "").strip()
                    if not address or address not in preimages:
                        continue
                    writes.append(
                        {
                            "address": address,
                            "old_value": preimages[address],
                            "target_value": write.get("target_value"),
                            "tolerance": float(write.get("tolerance") or 0.05),
                        }
                    )
            return writes

        address = str(recovery.get("address") or "").strip()
        if not address or "old_value" not in recovery:
            return []
        value_type = str(recovery.get("value_type") or "")
        tolerance = float(
            MANUAL_WRITE_KIND_CONFIG.get(value_type, {}).get(
                "readback_tolerance",
                0.05,
            )
        )
        return [
            {
                "address": address,
                "old_value": recovery.get("old_value"),
                "target_value": recovery.get("effective_value"),
                "tolerance": tolerance,
            }
        ]

    async def reconcile_console_write_recovery(
        self,
        *,
        transaction_id: Any,
        approved: bool = False,
        approval_id: Any = None,
    ) -> Dict[str, Any]:
        """Close an unresolved WAL row only from fresh physical readback."""

        transaction_id = str(transaction_id or "").strip()
        approval_id = str(approval_id or "").strip()
        if approved is not True or not approval_id:
            return {
                "type": "console_write_recovery_result",
                "status": "blocked",
                "success": False,
                "reason": "operator_approval_required",
                "transaction_id": transaction_id or None,
            }
        async with self._console_write_scope():
            recovery = self._manual_write_recovery_transactions.get(transaction_id)
            if not recovery:
                return {
                    "type": "console_write_recovery_result",
                    "status": "blocked",
                    "success": False,
                    "reason": "unresolved_transaction_not_found",
                    "transaction_id": transaction_id or None,
                }
            writes = self._wal_recovery_writes(recovery)
            if not writes:
                return {
                    "type": "console_write_recovery_result",
                    "status": "blocked",
                    "success": False,
                    "reason": "recovery_preimage_or_target_missing",
                    "transaction_id": transaction_id,
                }
            query = getattr(self.mixer_client, "query_parameter", None)
            if not callable(query):
                return {
                    "type": "console_write_recovery_result",
                    "status": "blocked",
                    "success": False,
                    "reason": "physical_readback_transport_unavailable",
                    "transaction_id": transaction_id,
                }

            evidence = []
            matches_target = []
            matches_old = []
            for write in writes:
                try:
                    result = await self._cancellation_safe_thread_call(
                        query,
                        write["address"],
                        0.75,
                    )
                except Exception as exc:
                    return {
                        "type": "console_write_recovery_result",
                        "status": "blocked",
                        "success": False,
                        "reason": "physical_recovery_read_failed",
                        "transaction_id": transaction_id,
                        "address": write["address"],
                        "error": str(exc),
                    }
                confirmed_round_trips = (
                    result.get("confirmed_round_trips")
                    if isinstance(result, Mapping)
                    else None
                )
                if (
                    not isinstance(result, Mapping)
                    or result.get("status") != "received"
                    or result.get("address") != write["address"]
                    or isinstance(confirmed_round_trips, bool)
                    or not isinstance(confirmed_round_trips, int)
                    or confirmed_round_trips < 2
                ):
                    return {
                        "type": "console_write_recovery_result",
                        "status": "blocked",
                        "success": False,
                        "reason": "physical_recovery_read_unverified",
                        "transaction_id": transaction_id,
                        "address": write["address"],
                        "query_result": convert_numpy_types(result),
                    }
                actual = result.get("value")
                target_match = self._wal_values_match(
                    write["target_value"],
                    actual,
                    write["tolerance"],
                )
                old_match = self._wal_values_match(
                    write["old_value"],
                    actual,
                    write["tolerance"],
                )
                matches_target.append(target_match)
                matches_old.append(old_match)
                evidence.append(
                    {
                        "address": write["address"],
                        "actual": actual,
                        "target_match": target_match,
                        "preimage_match": old_match,
                        "confirmed_round_trips": confirmed_round_trips,
                    }
                )

            if all(matches_target) and all(matches_old):
                terminal_phase = "reconciled_no_change"
            elif all(matches_target):
                terminal_phase = "reconciled_committed"
            elif all(matches_old):
                terminal_phase = "reconciled_rolled_back"
            else:
                return {
                    "type": "console_write_recovery_result",
                    "status": "blocked",
                    "success": False,
                    "reason": "recovery_state_mixed_or_changed",
                    "transaction_id": transaction_id,
                    "physical_readback": evidence,
                }

            wal_path = self._append_manual_write_wal(
                terminal_phase,
                transaction_id,
                approval_id=approval_id,
                physical_readback=evidence,
            )
            if not wal_path:
                return {
                    "type": "console_write_recovery_result",
                    "status": "blocked",
                    "success": False,
                    "reason": "recovery_terminal_record_failed",
                    "transaction_id": transaction_id,
                }
            return {
                "type": "console_write_recovery_result",
                "status": "reconciled",
                "success": True,
                "reason": terminal_phase,
                "transaction_id": transaction_id,
                "approval_id": approval_id,
                "wal_path": wal_path,
                "physical_readback": evidence,
                "emergency_reset_still_required": True,
            }

    @staticmethod
    def _normalize_manual_write_kind(value_type: Any) -> str:
        value = str(value_type or "").strip().lower()
        if value in MANUAL_LIVE_WRITE_KIND_ALIASES:
            return MANUAL_LIVE_WRITE_KIND_ALIASES[value]
        return value

    @staticmethod
    def _coerce_manual_write_channel(value: Any) -> Optional[int]:
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            parsed = value
        elif isinstance(value, float):
            if not math.isfinite(value) or not value.is_integer():
                return None
            parsed = int(value)
        elif isinstance(value, str):
            raw = value.strip()
            if not raw or not raw.lstrip("+-").isdigit():
                return None
            parsed = int(raw)
        else:
            return None
        return parsed if 1 <= parsed <= 48 else None

    @staticmethod
    def _normalize_readonly_parameter_kind(value_type: Any) -> str:
        value = str(value_type or "").strip().lower().replace("-", "_")
        if value in READONLY_PARAMETER_KIND_ALIASES:
            return READONLY_PARAMETER_KIND_ALIASES[value]
        return value

    @classmethod
    def _get_supported_manual_write_kinds(cls) -> tuple[str, ...]:
        return tuple(SUPPORTED_MANUAL_LIVE_WRITE_KINDS)

    def _effective_live_write_kinds_for_manual_gate(
        self,
        operator_mode: Dict[str, Any],
    ) -> list[str]:
        operator_capabilities = operator_mode.get("capabilities", {})
        allowed = list(operator_capabilities.get("allowed_live_write_kinds") or [])
        if self._is_wing_deployment_write_boundary_active():
            wing_allowed = set(WING_DEPLOYMENT_LIVE_WRITE_KINDS)
            allowed = [kind for kind in allowed if kind in wing_allowed]
        return allowed

    @staticmethod
    def _manual_write_address(channel: int, value_type: str) -> str:
        config = MANUAL_WRITE_KIND_CONFIG.get(value_type)
        if not config:
            return ""
        return config["address"].format(channel=channel)

    @staticmethod
    def _readonly_parameter_address(channel: int, value_type: str) -> str:
        config = READONLY_PARAMETER_KIND_CONFIG.get(value_type)
        if not config:
            return ""
        return config["address"].format(channel=channel)

    @staticmethod
    def _parse_manual_write_address(address: Any) -> Optional[tuple[int, str]]:
        if isinstance(address, os.PathLike):
            address = address.__fspath__()
        if isinstance(address, (bytes, bytearray)):
            try:
                address = address.decode("utf-8")
            except UnicodeDecodeError:
                return None
        normalized = str(address or "").strip()
        if not normalized:
            return None
        for value_type, config in MANUAL_WRITE_KIND_CONFIG.items():
            template = config.get("address", "")
            if "{channel}" not in template:
                continue
            pattern = re.escape(template).replace(r"\{channel\}", r"(?P<channel>\d+)")
            match = re.fullmatch(pattern, normalized)
            if match is not None:
                try:
                    return int(match.group("channel")), value_type
                except (TypeError, ValueError):
                    return None
        return None

    @staticmethod
    def _parse_readonly_parameter_address(address: Any) -> Optional[tuple[int, str]]:
        if isinstance(address, os.PathLike):
            address = address.__fspath__()
        if isinstance(address, (bytes, bytearray)):
            try:
                address = address.decode("utf-8")
            except UnicodeDecodeError:
                return None
        normalized = str(address or "").strip()
        if not normalized:
            return None
        for value_type, config in READONLY_PARAMETER_KIND_CONFIG.items():
            template = config.get("address", "")
            if "{channel}" not in template:
                continue
            pattern = re.escape(template).replace(r"\{channel\}", r"(?P<channel>\d+)")
            match = re.fullmatch(pattern, normalized)
            if match is not None:
                try:
                    return int(match.group("channel")), value_type
                except (TypeError, ValueError):
                    return None
        return None

    @staticmethod
    def _coerce_optional_bool(value: Any) -> Optional[bool]:
        if value is None:
            return None
        if isinstance(value, bool):
            return value
        if str(type(value).__module__).startswith("unittest.mock"):
            return None
        if isinstance(value, (int, float)):
            if value in {0, 1, 0.0, 1.0}:
                return bool(int(value))
            return None
        if isinstance(value, str):
            text = value.strip().lower()
            if text in {"1", "true", "on", "yes"}:
                return True
            if text in {"0", "false", "off", "no"}:
                return False
            if text == "mute":
                return True
            if text == "unmute":
                return False
        return None

    @staticmethod
    def _coerce_epoch_seconds(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, datetime):
            timestamp = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
            return timestamp.timestamp()
        if isinstance(value, (int, float)):
            numeric = float(value)
            return numeric / 1000.0 if numeric > 1_000_000_000_000 else numeric
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            try:
                return AutoMixerServer._coerce_epoch_seconds(float(text))
            except ValueError:
                pass
            try:
                return AutoMixerServer._coerce_epoch_seconds(datetime.fromisoformat(text.replace("Z", "+00:00")))
            except ValueError:
                return None
        return None

    @staticmethod
    def _iso_from_epoch(value: Optional[float]) -> Optional[str]:
        if value is None:
            return None
        return datetime.fromtimestamp(value, timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    def _manual_write_cache_freshness(
        self,
        address: str,
        *,
        minimum_updated_at_epoch: Optional[float] = None,
    ) -> Dict[str, Any]:
        client = self.mixer_client
        state = getattr(client, "state", None) if client else None
        present = bool(address and isinstance(state, dict) and state.get(address) is not None)
        updated_epoch = None
        if address:
            for attr in ("state_timestamps", "_state_timestamps", "state_updated_at", "_state_updated_at"):
                timestamps = getattr(client, attr, None) if client else None
                if not isinstance(timestamps, dict):
                    continue
                raw = timestamps.get(address)
                if isinstance(raw, dict):
                    raw = raw.get("updated_at") or raw.get("timestamp") or raw.get("ts")
                updated_epoch = self._coerce_epoch_seconds(raw)
                if updated_epoch is not None:
                    break

        now = datetime.now(timezone.utc).timestamp()
        age_sec = round(max(0.0, now - updated_epoch), 3) if updated_epoch is not None else None
        stale_after_sec = 30.0
        if not address:
            status = "unsupported"
        elif not present:
            status = "missing"
        elif updated_epoch is None:
            status = "unknown"
        elif age_sec is not None and age_sec > stale_after_sec:
            status = "stale"
        elif minimum_updated_at_epoch is not None and updated_epoch < minimum_updated_at_epoch:
            status = "not_after_write"
        else:
            status = "fresh"
        return {
            "status": status,
            "address": address or None,
            "source": "state_timestamps" if updated_epoch is not None else "unavailable",
            "updated_at": self._iso_from_epoch(updated_epoch),
            "minimum_updated_at": self._iso_from_epoch(minimum_updated_at_epoch),
            "updated_after_minimum": (
                None
                if minimum_updated_at_epoch is None or updated_epoch is None
                else updated_epoch >= minimum_updated_at_epoch
            ),
            "age_sec": age_sec,
            "stale_after_sec": stale_after_sec,
            "read_only": True,
            "live_write": False,
            "queries_console": False,
        }

    @staticmethod
    def _cache_freshness_blocks_pre_read(read_result: Optional[Dict[str, Any]]) -> bool:
        if not isinstance(read_result, dict):
            return False
        if read_result.get("source") != "mixer_state_cache":
            return False
        cache_freshness = read_result.get("cache_freshness")
        return not isinstance(cache_freshness, dict) or cache_freshness.get("status") != "fresh"

    @staticmethod
    def _clamp_manual_write_value(value: float, value_type: str) -> float:
        clamp = MANUAL_WRITE_KIND_CONFIG.get(value_type, {}).get("clamp")
        if clamp is None:
            return float(value)
        return clamp(value)

    @staticmethod
    def _coerce_optional_float(value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            if str(type(value).__module__).startswith("unittest.mock"):
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _coerce_finite_float(value: Any) -> Optional[float]:
        if isinstance(value, bool):
            return None
        parsed = AutoMixerServer._coerce_optional_float(value)
        if parsed is None or not math.isfinite(parsed):
            return None
        return parsed

    def _manual_write_cooldown_remaining(self, channel: int, value_type: str) -> float:
        if self._manual_write_cooldown_sec <= 0:
            return 0.0
        last_write = self._manual_write_last_write_ts.get((int(channel), value_type))
        if last_write is None:
            return 0.0
        elapsed = time.monotonic() - last_write
        return max(0.0, self._manual_write_cooldown_sec - elapsed)

    def _manual_write_retry_delay_sec(self) -> float:
        """Delay between manual-write attempts when the mixer client drops a write.

        WingClient rate-limits repeated OSC writes to the same address and
        returns False for a throttled write, so an approved apply followed by an
        immediate rollback to the same address can lose the rollback. The retry
        must wait two full throttle intervals: with trailing-edge throttle
        semantics a queued value is flushed at last_send + interval, and a retry
        arriving less than one interval after that flush would be throttled
        again.
        """
        client = self.mixer_client
        if getattr(client, "_osc_throttle_enabled", False):
            try:
                throttle_hz = float(getattr(client, "_osc_throttle_hz", 0.0) or 0.0)
            except (TypeError, ValueError):
                throttle_hz = 0.0
            if throttle_hz > 0:
                return min(1.0, (2.0 / throttle_hz) + 0.05)
        return 0.25

    def _manual_write_readback_status(
        self,
        *,
        old_value: Any,
        readback_value: Any,
        effective_value: Any,
        value_type: str,
        readback_source: str = "mixer_state_cache",
        readback_cache_freshness: Optional[Dict[str, Any]] = None,
    ) -> str:
        if value_type == "mute":
            readback_bool = self._coerce_optional_bool(readback_value)
            if readback_bool is None:
                return "readback_unavailable"
            if readback_source == "mixer_state_cache":
                if not isinstance(readback_cache_freshness, dict) or readback_cache_freshness.get("status") != "fresh":
                    return "readback_cache_not_fresh"
            if readback_bool == bool(effective_value):
                return (
                    "verified_from_physical_readback"
                    if readback_source == "physical_readback"
                    else "verified_from_cache"
                )
            old_bool = self._coerce_optional_bool(old_value)
            if old_bool is not None and old_bool == readback_bool:
                return "not_verified"
            if readback_source == "physical_readback":
                return "mismatch_from_physical_readback"
            return "mismatch_from_cache"

        readback_float = self._coerce_optional_float(readback_value)
        if readback_float is None:
            return "readback_unavailable"
        if readback_source == "mixer_state_cache":
            if not isinstance(readback_cache_freshness, dict) or readback_cache_freshness.get("status") != "fresh":
                return "readback_cache_not_fresh"
        config = MANUAL_WRITE_KIND_CONFIG.get(value_type, {})
        tolerance = config.get("readback_tolerance", 0.05)
        if abs(readback_float - effective_value) <= tolerance:
            if readback_source == "physical_readback":
                return "verified_from_physical_readback"
            return "verified_from_cache"

        old_float = self._coerce_optional_float(old_value)
        if old_float is not None and abs(readback_float - old_float) <= tolerance:
            return "not_verified"

        if readback_source == "physical_readback":
            return "mismatch_from_physical_readback"
        return "mismatch_from_cache"

    async def _maybe_await(self, value: Any) -> Any:
        if asyncio.iscoroutine(value) or hasattr(value, "__await__"):
            return await value
        return value

    def _extract_manual_write_read_value(self, raw_value: Any) -> Any:
        if str(type(raw_value).__module__).startswith("unittest.mock"):
            return None
        if isinstance(raw_value, dict):
            for key in ("value", "readback_value", "current_value", "result"):
                if raw_value.get(key) is not None:
                    return raw_value.get(key)
            return None
        if isinstance(raw_value, (list, tuple)):
            for item in reversed(raw_value):
                if self._coerce_optional_float(item) is not None:
                    return item
            return None
        return raw_value

    @staticmethod
    def _readonly_read_value_available(value_type: str, value: Any) -> bool:
        if AutoMixerServer._coerce_optional_float(value) is not None:
            return True
        if value_type in READONLY_ENUM_PARAMETER_KINDS:
            return bool(str(value or "").strip())
        return False

    @staticmethod
    async def _cancellation_safe_thread_call(function, *args, **kwargs):
        """Drain a blocking worker before cancellation can release its scope."""

        worker = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
        cancellation: Optional[asyncio.CancelledError] = None
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError as exc:
                cancellation = cancellation or exc
                continue
            except BaseException:
                if cancellation is None:
                    raise
                break
        outcome = None
        error: Optional[BaseException] = None
        try:
            outcome = worker.result()
        except BaseException as exc:
            error = exc
        if cancellation is not None:
            raise cancellation
        if error is not None:
            raise error
        return outcome

    @staticmethod
    async def _cancel_and_drain_async_task(task: asyncio.Task[Any]) -> None:
        """Cancel a child task and wait until its terminal cleanup completes."""

        if task is asyncio.current_task():
            raise RuntimeError("Cannot drain the current task")
        task.cancel()
        caller_cancellation: Optional[asyncio.CancelledError] = None
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError as exc:
                if task.done():
                    break
                caller_cancellation = caller_cancellation or exc
                continue
            except BaseException:
                break
        child_error: Optional[BaseException] = None
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except BaseException as exc:
            child_error = exc
        if caller_cancellation is not None:
            raise caller_cancellation
        if child_error is not None:
            raise child_error

    async def _manual_write_read_value(
        self,
        *,
        channel: int,
        value_type: str,
        address: str,
        prefer_physical: bool = False,
        allow_cache_fallback: bool = True,
        minimum_cache_updated_at_epoch: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Read a whitelisted WING parameter for proof metadata without writing.

        Physical readback is used only when the mixer client exposes an explicit
        query helper. Otherwise this falls back to the local mixer state cache.
        """

        client = self.mixer_client
        state = getattr(client, "state", {}) if client else {}
        cached_value = state.get(address) if isinstance(state, dict) else None

        async def physical_read() -> Optional[Dict[str, Any]]:
            query_parameter = getattr(client, "query_parameter", None) if client else None
            if not callable(query_parameter):
                return None
            try:
                if inspect.iscoroutinefunction(query_parameter):
                    raw = await query_parameter(address)
                else:
                    raw = await self._cancellation_safe_thread_call(
                        query_parameter,
                        address,
                    )
                    raw = await self._maybe_await(raw)
            except Exception as exc:
                return {
                    "value": None,
                    "status": "readback_unavailable",
                    "source": "physical_readback",
                    "address": address,
                    "error": str(exc),
                }
            confirmed_round_trips = (
                raw.get("confirmed_round_trips")
                if isinstance(raw, Mapping)
                else None
            )
            if (
                not isinstance(raw, Mapping)
                or raw.get("status") != "received"
                or raw.get("address") != address
                or isinstance(confirmed_round_trips, bool)
                or not isinstance(confirmed_round_trips, int)
                or confirmed_round_trips < 2
            ):
                return {
                    "value": None,
                    "status": "readback_unavailable",
                    "source": "physical_readback",
                    "address": address,
                    "error": "physical_query_contract_unverified",
                    "raw": raw,
                }
            value = self._extract_manual_write_read_value(raw)
            if not self._readonly_read_value_available(value_type, value):
                return {
                    "value": None,
                    "status": "readback_unavailable",
                    "source": "physical_readback",
                    "address": address,
                    "raw": raw,
                }
            return {
                "value": value,
                "status": "verified_from_physical_readback",
                "source": "physical_readback",
                "address": address,
                "raw": raw,
            }

        if prefer_physical:
            physical = await physical_read()
            if physical and physical.get("value") is not None:
                return physical
            if not allow_cache_fallback:
                return physical or {
                    "value": None,
                    "status": "readback_unavailable",
                    "source": "physical_readback",
                    "address": address,
                    "error": "physical_query_unavailable",
                }
            if minimum_cache_updated_at_epoch is not None:
                return physical or {
                    "value": None,
                    "status": "readback_unavailable",
                    "source": "physical_readback",
                    "address": address,
                    "error": "physical_query_unavailable",
                }

        if self._readonly_read_value_available(value_type, cached_value):
            return {
                "value": cached_value,
                "status": "verified_from_cache",
                "source": "mixer_state_cache",
                "address": address,
                "cache_freshness": self._manual_write_cache_freshness(
                    address,
                    minimum_updated_at_epoch=minimum_cache_updated_at_epoch,
                ),
            }

        physical = await physical_read()
        if physical:
            return physical

        return {
            "value": None,
            "status": "readback_unavailable",
            "source": "unavailable",
            "address": address,
        }

    @staticmethod
    def _manual_write_proof_status(readback_status: str) -> str:
        if readback_status == "verified_from_physical_readback":
            return "verified"
        return "partial"

    @asynccontextmanager
    async def _console_write_scope(
        self,
        *,
        deadline_epoch: Optional[float] = None,
    ):
        """Reentrant task/process scope for all console snapshots and mutations."""
        task = asyncio.current_task()
        if task is not None and self._console_write_owner_task is task:
            yield True
            return
        local_lock_acquired = False
        try:
            if deadline_epoch is None:
                await self._console_write_lock.acquire()
            else:
                remaining = float(deadline_epoch) - time.time()
                if remaining <= 0.0:
                    yield False
                    return
                try:
                    await asyncio.wait_for(
                        self._console_write_lock.acquire(),
                        timeout=remaining,
                    )
                except asyncio.TimeoutError:
                    yield False
                    return
            local_lock_acquired = True
            lock_path = self._console_writer_lock_path
            self._ensure_durable_directory(self._console_runtime_dir)
            open_flags = os.O_CREAT | os.O_RDWR
            open_flags |= getattr(os, "O_CLOEXEC", 0)
            open_flags |= getattr(os, "O_NOFOLLOW", 0)
            lock_fd = os.open(lock_path, open_flags, 0o600)
            process_lock_acquired = False
            try:
                os.fchmod(lock_fd, 0o600)
                while True:
                    try:
                        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        process_lock_acquired = True
                        break
                    except BlockingIOError:
                        if (
                            deadline_epoch is not None
                            and time.time() >= float(deadline_epoch)
                        ):
                            break
                        await asyncio.sleep(0.05)
                if not process_lock_acquired:
                    os.close(lock_fd)
                    yield False
                    return
                self._recover_manual_write_wal_state()
            except BaseException:
                os.close(lock_fd)
                raise
            self._console_write_owner_task = task
            self._console_write_active_idempotency_keys = set()
            try:
                yield True
            finally:
                self._console_write_active_idempotency_keys.clear()
                self._console_write_owner_task = None
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                os.close(lock_fd)
        finally:
            if local_lock_acquired:
                self._console_write_lock.release()

    async def _apply_manual_console_write(
        self,
        websocket: Optional[WebSocketServerProtocol],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Serialize every physical console mutation through one server lock."""
        async with self._console_write_scope():
            return await self._apply_manual_console_write_unlocked(
                websocket,
                **kwargs,
            )

    async def _apply_manual_console_write_unlocked(
        self,
        websocket: Optional[WebSocketServerProtocol],
        *,
        channel: Any,
        value: Any,
        value_type: Any,
        source: str = "",
        reason: str = "",
        approved: bool = False,
        approval_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        proposal_id: Optional[str] = None,
        expected_current_value: Any = None,
        bypass_cooldown: bool = False,
        record_event: bool = True,
        emergency_recovery: bool = False,
    ) -> Dict[str, Any]:
        write_kind = self._normalize_manual_write_kind(value_type)
        if self._packaged_offline_lock:
            return self._manual_write_blocked_result(
                reason="packaged_offline_lock_active",
                channel=channel,
                value=value,
                value_type=write_kind or value_type,
                source=source,
                approval_id=approval_id,
                extra={
                    "packaged_offline_lock": True,
                    "mixer_transport_locked": True,
                },
            )
        supported_manual_write_kinds = self._get_supported_manual_write_kinds()
        if write_kind not in supported_manual_write_kinds:
            return self._manual_write_blocked_result(
                reason="unsupported_value_type",
                channel=channel,
                value=value,
                value_type=write_kind or value_type,
                source=source,
                approval_id=approval_id,
                extra={"allowed_live_write_kinds": list(supported_manual_write_kinds)},
            )

        operator_mode = self.get_operator_mode_status()
        operator_capabilities = operator_mode.get("capabilities", {})
        allowed_live_write_kinds = self._effective_live_write_kinds_for_manual_gate(
            operator_mode
        )
        if not operator_capabilities.get("can_apply_to_console"):
            return self._manual_write_blocked_result(
                reason="operator_mode_blocks_live_write",
                channel=channel,
                value=value,
                value_type=write_kind,
                source=source,
                approval_id=approval_id,
                extra={"allowed_live_write_kinds": allowed_live_write_kinds},
            )
        if write_kind not in allowed_live_write_kinds:
            return self._manual_write_blocked_result(
                reason="live_write_kind_not_allowed_in_mode",
                channel=channel,
                value=value,
                value_type=write_kind,
                source=source,
                approval_id=approval_id,
                extra={"allowed_live_write_kinds": allowed_live_write_kinds},
            )

        if self.connection_mode == "wing" and self._wing_observation_only:
            return self._manual_write_blocked_result(
                reason="wing_observation_only_active",
                channel=channel,
                value=value,
                value_type=write_kind,
                source=source,
                approval_id=approval_id,
                extra={
                    "observation_only": True,
                    "record_telemetry": self._wing_record_telemetry,
                },
            )

        if self._manual_write_emergency_stopped and not emergency_recovery:
            return self._manual_write_blocked_result(
                reason="emergency_stop_active",
                channel=channel,
                value=value,
                value_type=write_kind,
                source=source,
                approval_id=approval_id,
            )

        if approved is not True:
            return self._manual_write_blocked_result(
                reason="approval_required",
                channel=channel,
                value=value,
                value_type=write_kind,
                source=source,
                approval_id=approval_id,
                extra={"allowed_live_write_kinds": allowed_live_write_kinds},
            )

        channel_int = self._coerce_manual_write_channel(channel)
        if channel_int is None:
            return self._manual_write_blocked_result(
                reason="invalid_channel",
                channel=channel,
                value=value,
                value_type=write_kind,
                source=source,
                approval_id=approval_id,
            )
        try:
            if write_kind == "mute":
                parsed_mute = self._coerce_optional_bool(value)
                if parsed_mute is None:
                    raise ValueError("invalid_mute_value")
                requested_value = bool(parsed_mute)
                effective_value = self._clamp_manual_write_value(requested_value, write_kind)
            else:
                if isinstance(value, bool):
                    raise ValueError("invalid_numeric_value")
                requested_value = float(value)
                if not math.isfinite(requested_value):
                    raise ValueError("non_finite_value")
                effective_value = self._clamp_manual_write_value(requested_value, write_kind)
        except (TypeError, ValueError):
            return self._manual_write_blocked_result(
                reason="invalid_value",
                channel=channel_int,
                value=value,
                value_type=write_kind,
                source=source,
                approval_id=approval_id,
            )

        idempotency_key_text = str(idempotency_key or "").strip()
        if (
            idempotency_key_text
            and idempotency_key_text
            not in self._console_write_active_idempotency_keys
        ):
            durable_duplicate = self._durable_operator_apply_idempotency_keys.get(
                idempotency_key_text
            )
            if durable_duplicate:
                return self._manual_write_blocked_result(
                    reason="duplicate_idempotency_key",
                    channel=channel_int,
                    value=requested_value,
                    value_type=write_kind,
                    source=source,
                    approval_id=approval_id,
                    extra={
                        "idempotency_key": idempotency_key_text,
                        "first_seen": durable_duplicate,
                    },
                )
            self._console_write_active_idempotency_keys.add(idempotency_key_text)

        if not self.mixer_client or not getattr(self.mixer_client, "is_connected", False):
            return self._manual_write_blocked_result(
                reason="mixer_not_connected",
                channel=channel_int,
                value=requested_value,
                value_type=write_kind,
                source=source,
                approval_id=approval_id,
            )

        method_name = MANUAL_WRITE_KIND_CONFIG.get(write_kind, {}).get("method", "")
        write_method = getattr(self.mixer_client, method_name, None)
        if not callable(write_method):
            return self._manual_write_blocked_result(
                reason="manual_write_method_unavailable",
                channel=channel_int,
                value=requested_value,
                value_type=write_kind,
                source=source,
                approval_id=approval_id,
                extra={"method": method_name},
            )

        address = self._manual_write_address(channel_int, write_kind)
        pre_read = await self._manual_write_read_value(
            channel=channel_int,
            value_type=write_kind,
            address=address,
            prefer_physical=True,
        )
        old_value = pre_read.get("value")
        old_value_float = self._coerce_optional_float(old_value)
        old_value_bool = self._coerce_optional_bool(old_value)
        pre_read_status = str(pre_read.get("status") or "readback_unavailable")
        pre_read_source = str(pre_read.get("source") or "unavailable")
        pre_read_cache_freshness = pre_read.get("cache_freshness")
        if old_value_float is None and old_value_bool is None:
            return self._manual_write_blocked_result(
                reason="pre_read_unavailable",
                channel=channel_int,
                value=requested_value,
                value_type=write_kind,
                source=source,
                approval_id=approval_id,
                extra={
                    "address": address,
                    "effective_value": effective_value,
                    "pre_read_status": pre_read_status,
                    "pre_read_source": pre_read_source,
                    "pre_read_value": old_value,
                    "pre_read_error": pre_read.get("error"),
                },
            )

        if self._cache_freshness_blocks_pre_read(pre_read):
            return self._manual_write_blocked_result(
                reason="pre_read_cache_not_fresh",
                channel=channel_int,
                value=requested_value,
                value_type=write_kind,
                source=source,
                approval_id=approval_id,
                extra={
                    "address": address,
                    "effective_value": effective_value,
                    "pre_read_status": pre_read_status,
                    "pre_read_source": pre_read_source,
                    "pre_read_value": old_value,
                    "cache_freshness": pre_read_cache_freshness,
                },
            )

        cooldown_remaining = 0.0 if bypass_cooldown else self._manual_write_cooldown_remaining(channel_int, write_kind)
        if cooldown_remaining > 0:
            return self._manual_write_blocked_result(
                reason="cooldown",
                channel=channel_int,
                value=requested_value,
                value_type=write_kind,
                source=source,
                approval_id=approval_id,
                extra={
                    "address": address,
                    "effective_value": effective_value,
                    "old_value": old_value,
                    "pre_read_status": pre_read_status,
                    "pre_read_source": pre_read_source,
                    "pre_read_value": old_value,
                    "cooldown_sec": self._manual_write_cooldown_sec,
                    "cooldown_remaining_sec": round(cooldown_remaining, 3),
                },
            )

        write_started_epoch = datetime.now(timezone.utc).timestamp()
        write_emergency_generation = self._manual_write_emergency_generation
        transaction_id = str(uuid.uuid4())
        max_write_attempts = 3
        retry_delay_sec = self._manual_write_retry_delay_sec()
        write_attempts = 0
        write_result: Any = False
        wal_path = self._append_manual_write_wal(
            "prepared",
            transaction_id,
            scope="operator_proposal" if idempotency_key else "manual_write",
            idempotency_key=str(idempotency_key or "").strip() or None,
            proposal_id=str(proposal_id or "").strip() or None,
            channel=channel_int,
            value_type=write_kind,
            address=address,
            requested_value=requested_value,
            effective_value=effective_value,
            old_value=old_value,
            expected_current_value=expected_current_value,
            source=source,
            approval_id=approval_id,
            emergency_generation=write_emergency_generation,
        )
        if not wal_path:
            await self.activate_system_emergency_stop(
                f"manual_write_wal_prepare_failed:{transaction_id}"
            )
            return self._manual_write_blocked_result(
                reason="manual_write_wal_prepare_failed",
                channel=channel_int,
                value=requested_value,
                value_type=write_kind,
                source=source,
                approval_id=approval_id,
                extra={
                    "transaction_id": transaction_id,
                    "address": address,
                    "effective_value": effective_value,
                    "old_value": old_value,
                    "emergency_stop_activated": True,
                },
            )
        if expected_current_value is not None:
            cas_read = await self._manual_write_read_value(
                channel=channel_int,
                value_type=write_kind,
                address=address,
                prefer_physical=True,
                allow_cache_fallback=False,
            )
            tolerance = float(
                MANUAL_WRITE_KIND_CONFIG.get(write_kind, {}).get(
                    "readback_tolerance",
                    0.05,
                )
            )
            cas_verified = (
                cas_read.get("source") == "physical_readback"
                and self._wal_values_match(
                    expected_current_value,
                    cas_read.get("value"),
                    tolerance,
                )
            )
            if not cas_verified:
                self._append_manual_write_wal(
                    "aborted_before_send",
                    transaction_id,
                    reason="manual_write_cas_target_changed_or_unverified",
                    expected_current_value=expected_current_value,
                    current_readback_value=cas_read.get("value"),
                    current_readback_status=cas_read.get("status"),
                    current_readback_source=cas_read.get("source"),
                )
                return self._manual_write_blocked_result(
                    reason="manual_write_cas_target_changed_or_unverified",
                    channel=channel_int,
                    value=requested_value,
                    value_type=write_kind,
                    source=source,
                    approval_id=approval_id,
                    extra={
                        "transaction_id": transaction_id,
                        "address": address,
                        "expected_current_value": expected_current_value,
                        "current_readback_value": cas_read.get("value"),
                        "current_readback_status": cas_read.get("status"),
                        "current_readback_source": cas_read.get("source"),
                    },
                )
        while True:
            if (
                self._manual_write_emergency_generation != write_emergency_generation
                or (self._manual_write_emergency_stopped and not emergency_recovery)
            ):
                phase = (
                    "aborted_before_send"
                    if write_attempts == 0
                    else "manual_review_required"
                )
                self._append_manual_write_wal(
                    phase,
                    transaction_id,
                    reason="emergency_stop_active",
                    write_attempts=write_attempts,
                )
                return self._manual_write_blocked_result(
                    reason="emergency_stop_active",
                    channel=channel_int,
                    value=requested_value,
                    value_type=write_kind,
                    source=source,
                    approval_id=approval_id,
                    extra={"transaction_id": transaction_id},
                )
            write_attempts += 1
            if not self._append_manual_write_wal(
                "send_started",
                transaction_id,
                write_attempt=write_attempts,
                method=method_name,
                channel=channel_int,
                effective_value=effective_value,
            ):
                await self.activate_system_emergency_stop(
                    f"manual_write_wal_send_start_failed:{transaction_id}"
                )
                return self._manual_write_blocked_result(
                    reason="manual_write_wal_send_start_failed",
                    channel=channel_int,
                    value=requested_value,
                    value_type=write_kind,
                    source=source,
                    approval_id=approval_id,
                    extra={
                        "transaction_id": transaction_id,
                        "address": address,
                        "effective_value": effective_value,
                        "old_value": old_value,
                        "write_attempts": write_attempts - 1,
                        "emergency_stop_activated": True,
                    },
                )
            try:
                write_result = write_method(channel_int, effective_value)
            except Exception as exc:
                logger.error("Manual console write failed: %s", exc, exc_info=True)
                self._append_manual_write_wal(
                    "manual_review_required",
                    transaction_id,
                    reason="manual_write_exception",
                    error=str(exc),
                    write_attempt=write_attempts,
                )
                await self.activate_system_emergency_stop(
                    f"manual_write_exception:{transaction_id}"
                )
                error_result = {
                    "status": "error",
                    "success": False,
                    "reason": "manual_write_exception",
                    "error": str(exc),
                    "channel": channel_int,
                    "value_type": write_kind,
                    "requested_value": requested_value,
                    "effective_value": effective_value,
                    "old_value": old_value,
                    "pre_read_status": pre_read_status,
                    "pre_read_source": pre_read_source,
                    "pre_read_value": old_value,
                    "proof_status": "blocked",
                    "source": source,
                    "approval_id": approval_id,
                    "transaction_id": transaction_id,
                    "write_attempts": write_attempts,
                    "wal_path": wal_path,
                    "emergency_stop_activated": True,
                    "manual_write_supervision": self.get_manual_write_supervision_status(),
                }
                if record_event:
                    artifact_path = self._record_manual_write_event("manual_write_error", error_result)
                    if artifact_path:
                        error_result["artifact_path"] = artifact_path
                return error_result
            send_phase = (
                "send_result_false" if write_result is False else "send_completed"
            )
            if not self._append_manual_write_wal(
                send_phase,
                transaction_id,
                write_attempt=write_attempts,
                write_result=write_result,
            ):
                await self.activate_system_emergency_stop(
                    f"manual_write_wal_send_result_failed:{transaction_id}"
                )
                return self._manual_write_blocked_result(
                    reason="manual_write_wal_send_result_failed",
                    channel=channel_int,
                    value=requested_value,
                    value_type=write_kind,
                    source=source,
                    approval_id=approval_id,
                    extra={
                        "transaction_id": transaction_id,
                        "address": address,
                        "effective_value": effective_value,
                        "old_value": old_value,
                        "write_attempts": write_attempts,
                        "emergency_stop_activated": True,
                    },
                )
            if write_result is not False or write_attempts >= max_write_attempts:
                break
            logger.warning(
                "Manual console write for ch%s %s returned False (attempt %d/%d); "
                "retrying in %.3fs in case a same-address write rate limit dropped it",
                channel_int,
                write_kind,
                write_attempts,
                max_write_attempts,
                retry_delay_sec,
            )
            try:
                await asyncio.sleep(retry_delay_sec)
            except asyncio.CancelledError:
                self._append_manual_write_wal(
                    "manual_review_required",
                    transaction_id,
                    reason="cancelled_between_write_attempts",
                    write_attempts=write_attempts,
                )
                await self.activate_system_emergency_stop(
                    f"manual_write_cancelled:{transaction_id}"
                )
                raise

        if write_result is False:
            self._append_manual_write_wal(
                "manual_review_required",
                transaction_id,
                reason="mixer_write_failed",
                write_attempts=write_attempts,
            )
            await self.activate_system_emergency_stop(
                f"manual_write_result_uncertain:{transaction_id}"
            )
            return self._manual_write_blocked_result(
                reason="mixer_write_failed",
                channel=channel_int,
                value=requested_value,
                value_type=write_kind,
                source=source,
                approval_id=approval_id,
                extra={
                    "address": address,
                    "effective_value": effective_value,
                    "old_value": old_value,
                    "pre_read_status": pre_read_status,
                    "pre_read_source": pre_read_source,
                    "pre_read_value": old_value,
                    "method": method_name,
                    "write_attempts": write_attempts,
                    "transaction_id": transaction_id,
                    "wal_path": wal_path,
                    "emergency_stop_activated": True,
                },
            )

        self._manual_write_last_write_ts[(channel_int, write_kind)] = time.monotonic()
        try:
            readback = await self._manual_write_read_value(
                channel=channel_int,
                value_type=write_kind,
                address=address,
                prefer_physical=True,
                minimum_cache_updated_at_epoch=write_started_epoch,
            )
        except asyncio.CancelledError:
            self._append_manual_write_wal(
                "manual_review_required",
                transaction_id,
                reason="cancelled_during_physical_readback",
                write_attempts=write_attempts,
            )
            await self.activate_system_emergency_stop(
                f"manual_write_cancelled:{transaction_id}"
            )
            raise
        except Exception as exc:
            self._append_manual_write_wal(
                "manual_review_required",
                transaction_id,
                reason="physical_readback_exception",
                error=str(exc),
                write_attempts=write_attempts,
            )
            await self.activate_system_emergency_stop(
                f"manual_write_readback_exception:{transaction_id}"
            )
            return self._manual_write_blocked_result(
                reason="physical_readback_exception",
                channel=channel_int,
                value=requested_value,
                value_type=write_kind,
                source=source,
                approval_id=approval_id,
                extra={
                    "transaction_id": transaction_id,
                    "address": address,
                    "effective_value": effective_value,
                    "old_value": old_value,
                    "write_attempts": write_attempts,
                    "wal_path": wal_path,
                    "error": str(exc),
                    "emergency_stop_activated": True,
                },
            )
        readback_value = readback.get("value")
        readback_source = str(readback.get("source") or "unavailable")
        readback_cache_freshness = readback.get("cache_freshness")
        readback_status = self._manual_write_readback_status(
            old_value=old_value,
            readback_value=readback_value,
            effective_value=effective_value,
            value_type=write_kind,
            readback_source=readback_source,
            readback_cache_freshness=readback_cache_freshness,
        )
        proof_status = self._manual_write_proof_status(readback_status)
        rollback_value = self._coerce_optional_float(old_value)
        rollback_available = rollback_value is not None

        physically_verified = readback_status == "verified_from_physical_readback"
        result = {
            "status": "applied" if physically_verified else "sent_unverified",
            "success": physically_verified,
            "reason": (
                str(reason or "manual_supervised_write")
                if physically_verified
                else "physical_readback_not_verified"
            ),
            "transaction_id": transaction_id,
            "channel": channel_int,
            "value_type": write_kind,
            "requested_value": requested_value,
            "effective_value": effective_value,
            "old_value": old_value,
            "pre_read_value": old_value,
            "pre_read_status": pre_read_status,
            "pre_read_source": pre_read_source,
            "pre_read_cache_freshness": pre_read_cache_freshness,
            "readback_value": readback_value,
            "readback_status": readback_status,
            "readback_source": readback_source,
            "readback_cache_freshness": readback_cache_freshness,
            "proof_status": proof_status,
            "rollback_available": rollback_available,
            "rollback_value": rollback_value,
            "rollback_address": address if rollback_available else None,
            "rollback_status": "not_attempted",
            "cooldown_sec": self._manual_write_cooldown_sec,
            "source": source,
            "requested_reason": str(reason or "manual_supervised_write"),
            "approval_id": approval_id,
            "method": method_name,
            "write_attempts": write_attempts,
            "wal_path": wal_path,
        }
        durably_committed = False
        if physically_verified:
            durably_committed = bool(
                self._append_manual_write_wal(
                    "committed",
                    transaction_id,
                    readback_status=readback_status,
                    readback_source=readback_source,
                    readback_value=readback_value,
                    write_attempts=write_attempts,
                )
            )
            if not durably_committed:
                result.update(
                    {
                        "status": "sent_unverified",
                        "success": False,
                        "reason": "manual_write_wal_commit_failed",
                        "proof_status": "blocked",
                    }
                )
        else:
            self._append_manual_write_wal(
                "manual_review_required",
                transaction_id,
                reason="physical_readback_not_verified",
                readback_status=readback_status,
                readback_source=readback_source,
                readback_value=readback_value,
                write_attempts=write_attempts,
            )

        if not durably_committed:
            await self.activate_system_emergency_stop(
                f"manual_write_unverified:{transaction_id}"
            )
            result["emergency_stop_activated"] = True
        result["manual_write_supervision"] = self.get_manual_write_supervision_status()
        self._last_manual_write_result = dict(result)
        if record_event:
            artifact_path = self._record_manual_write_event(
                "manual_write_applied" if durably_committed else "manual_write_unverified",
                result,
            )
            if artifact_path:
                result["artifact_path"] = artifact_path
                self._last_manual_write_result["artifact_path"] = artifact_path
        return result

    async def _rollback_last_manual_console_write(
        self,
        websocket: Optional[WebSocketServerProtocol],
        *,
        source: str = "manual_rollback",
        reason: str = "",
        approved: bool = False,
        approval_id: Optional[str] = None,
        transaction_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        async with self._console_write_scope():
            return await self._rollback_last_manual_console_write_unlocked(
                websocket,
                source=source,
                reason=reason,
                approved=approved,
                approval_id=approval_id,
                transaction_id=transaction_id,
            )

    async def _rollback_last_manual_console_write_unlocked(
        self,
        websocket: Optional[WebSocketServerProtocol],
        *,
        source: str = "manual_rollback",
        reason: str = "",
        approved: bool = False,
        approval_id: Optional[str] = None,
        transaction_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        previous = dict(self._last_manual_write_result or {})
        if not previous or previous.get("status") not in {"applied", "sent_unverified"}:
            return self._manual_write_blocked_result(
                reason="no_rollback_available",
                source=source,
                approval_id=approval_id,
                event_type="manual_write_rollback_blocked",
            )

        previous_transaction_id = str(previous.get("transaction_id") or "")
        requested_transaction_id = str(transaction_id or "")
        if not requested_transaction_id:
            return self._manual_write_blocked_result(
                reason="rollback_transaction_id_required",
                channel=previous.get("channel"),
                value_type=previous.get("value_type"),
                source=source,
                approval_id=approval_id,
                extra={"rollback_of_transaction_id": previous_transaction_id},
                event_type="manual_write_rollback_blocked",
            )
        if requested_transaction_id != previous_transaction_id:
            return self._manual_write_blocked_result(
                reason="rollback_transaction_mismatch",
                channel=previous.get("channel"),
                value_type=previous.get("value_type"),
                source=source,
                approval_id=approval_id,
                extra={
                    "rollback_of_transaction_id": previous_transaction_id,
                    "requested_transaction_id": requested_transaction_id,
                },
                event_type="manual_write_rollback_blocked",
            )

        if not previous.get("rollback_available"):
            return self._manual_write_blocked_result(
                reason="rollback_value_unavailable",
                channel=previous.get("channel"),
                value_type=previous.get("value_type"),
                source=source,
                approval_id=approval_id,
                extra={"rollback_of": previous.get("approval_id")},
                event_type="manual_write_rollback_blocked",
            )

        rollback_address = str(previous.get("rollback_address") or "")
        current_read = await self._manual_write_read_value(
            channel=int(previous.get("channel")),
            value_type=str(previous.get("value_type")),
            address=rollback_address,
            prefer_physical=True,
        )
        current_status = self._manual_write_readback_status(
            old_value=previous.get("effective_value"),
            readback_value=current_read.get("value"),
            effective_value=previous.get("effective_value"),
            value_type=str(previous.get("value_type")),
            readback_source=str(current_read.get("source") or "unavailable"),
            readback_cache_freshness=current_read.get("cache_freshness"),
        )
        if current_status != "verified_from_physical_readback":
            return self._manual_write_blocked_result(
                reason="rollback_intervening_change_or_unverified_state",
                channel=previous.get("channel"),
                value_type=previous.get("value_type"),
                source=source,
                approval_id=approval_id,
                extra={
                    "rollback_of_transaction_id": previous_transaction_id,
                    "current_readback_value": current_read.get("value"),
                    "current_readback_source": current_read.get("source"),
                    "current_readback_status": current_status,
                    "expected_current_value": previous.get("effective_value"),
                },
                event_type="manual_write_rollback_blocked",
            )

        rollback_result = await self._apply_manual_console_write(
            websocket,
            channel=previous.get("channel"),
            value=previous.get("rollback_value"),
            value_type=previous.get("value_type"),
            source=source,
            reason=reason or f"rollback:{previous.get('approval_id') or previous.get('source') or 'manual_write'}",
            approved=approved,
            approval_id=approval_id or f"rollback:{previous.get('approval_id') or 'manual_write'}",
            expected_current_value=previous.get("effective_value"),
            bypass_cooldown=True,
            record_event=False,
            emergency_recovery=True,
        )
        rollback_result.update(
            {
                "rollback": True,
                "rollback_of": previous.get("approval_id"),
                "rollback_of_transaction_id": previous_transaction_id,
                "rollback_source": previous.get("source"),
            }
        )
        if rollback_result.get("status") in {"applied", "sent_unverified"}:
            rollback_proof_status = rollback_result.get("proof_status") or self._manual_write_proof_status(
                str(rollback_result.get("readback_status") or "")
            )
            rollback_result["rollback_readback_status"] = rollback_result.get("readback_status")
            rollback_result["rollback_readback_source"] = rollback_result.get("readback_source")
            rollback_result["rollback_proof_status"] = rollback_proof_status
            rollback_result["rollback_status"] = "applied" if rollback_proof_status == "verified" else "unverified"
        rollback_event_type = (
            "manual_write_rollback_applied"
            if rollback_result.get("status") == "applied"
            else "manual_write_rollback_blocked"
        )
        if rollback_result.get("status") == "error":
            rollback_event_type = "manual_write_rollback_error"
        elif rollback_result.get("status") == "sent_unverified":
            rollback_event_type = "manual_write_rollback_unverified"
        artifact_path = self._record_manual_write_event(rollback_event_type, rollback_result)
        if artifact_path:
            rollback_result["artifact_path"] = artifact_path
            if rollback_result.get("status") == "applied":
                self._last_manual_write_result["artifact_path"] = artifact_path
        return rollback_result

    def cleanup_all_controllers(self):
        """Cleanup all active controllers - call before shutdown or on error."""
        if self._controllers_cleanup_complete:
            logger.debug("Controller cleanup already completed")
            return
        logger.info("Cleaning up all controllers...")
        errors: list[tuple[str, BaseException]] = []

        if self.safe_gain_calibrator is not None or self._safe_gain_audio_thread is not None:
            try:
                self._stop_safe_gain_runtime()
                logger.info("SafeGain runtime stopped")
            except Exception as exc:
                logger.error("Error stopping SafeGain runtime: %s", exc, exc_info=True)
                errors.append(("safe_gain_runtime", exc))

        live_voice_tasks = [
            task for task in tuple(self._voice_command_tasks) if not task.done()
        ]
        for task in live_voice_tasks:
            task.cancel()
        self._voice_command_tasks = {
            task for task in self._voice_command_tasks if not task.done()
        }
        if live_voice_tasks:
            errors.append((
                "voice_command_tasks",
                RuntimeError("tasks cancelled but not drained"),
            ))

        # Stop gain staging
        if self.gain_staging:
            stopped = self._safe_cleanup_call(
                self.gain_staging.stop,
                "Error stopping gain staging",
                "Gain staging stopped",
            )
            if stopped:
                self.gain_staging = None
            else:
                errors.append(("gain_staging", RuntimeError("stop failed")))

        # Stop voice control
        if self.voice_control:
            try:
                self.voice_control.stop_listening()
                logger.info("Voice control stopped")
            except Exception as e:
                logger.error(f"Error stopping voice control: {e}")
                errors.append(("voice_control", e))
            else:
                self.voice_control = None

        # Stop Auto-EQ
        if self.auto_eq_controller:
            try:
                self.auto_eq_controller.stop()
                logger.info("Auto-EQ stopped")
            except Exception as e:
                logger.error(f"Error stopping Auto-EQ: {e}")
                errors.append(("auto_eq", e))
            else:
                self.auto_eq_controller = None

        # Stop Multi-channel Auto-EQ
        if self.multi_channel_auto_eq_controller:
            stopped = self._safe_cleanup_call(
                self.multi_channel_auto_eq_controller.stop_all,
                "Error stopping multi-channel Auto-EQ",
                "Multi-channel Auto-EQ stopped",
            )
            if stopped:
                self.multi_channel_auto_eq_controller = None
            else:
                errors.append(("multi_channel_auto_eq", RuntimeError("stop failed")))

        # Stop Phase alignment
        if self.phase_alignment_controller:
            stopped = self._safe_cleanup_call(
                self.phase_alignment_controller.stop_analysis,
                "Error stopping phase alignment",
                "Phase alignment stopped",
            )
            if stopped:
                self.phase_alignment_controller = None
            else:
                errors.append(("phase_alignment", RuntimeError("stop failed")))

        if self.system_measurement_controller:
            self.system_measurement_controller = None

        # Stop Auto Fader
        if self.auto_fader_controller:
            stopped = self._safe_cleanup_call(
                self.auto_fader_controller.stop,
                "Error stopping Auto Fader",
                "Auto Fader stopped",
            )
            if stopped:
                self.auto_fader_controller = None
            else:
                errors.append(("auto_fader", RuntimeError("stop failed")))

        auto_fader_v2 = getattr(self, "auto_fader_v2_controller", None)
        if auto_fader_v2 is not None:
            stopped = self._safe_cleanup_call(
                auto_fader_v2.stop,
                "Error stopping Auto Fader V2",
                "Auto Fader V2 stopped",
            )
            if stopped:
                self.auto_fader_v2_controller = None
            else:
                errors.append(("auto_fader_v2", RuntimeError("stop failed")))

        # A synchronous fallback cannot truthfully drain a live asyncio task.
        if self.auto_soundcheck_task is not None:
            if self.auto_soundcheck_task.done():
                self.auto_soundcheck_task = None
                self.auto_soundcheck_websocket = None
            else:
                self.auto_soundcheck_task.cancel()
                errors.append((
                    "legacy_auto_soundcheck_task",
                    RuntimeError("task cancelled but not drained"),
                ))
        if self.auto_soundcheck_task is None:
            self.auto_soundcheck_running = False
            self.auto_soundcheck_observe_only = False

        # Stop Auto Compressor
        if self.auto_compressor_controller:
            try:
                self.auto_compressor_controller.stop()
                logger.info("Auto Compressor stopped")
            except Exception as e:
                logger.error(f"Error stopping Auto Compressor: {e}")
                errors.append(("auto_compressor", e))
            else:
                self.auto_compressor_controller = None

        # Stop audio capture
        if self.audio_capture:
            capture = self.audio_capture
            try:
                capture.stop()
                logger.info("Audio capture stopped")
            except Exception as e:
                logger.error(f"Error stopping audio capture: {e}")
                errors.append(("audio_capture", e))
            else:
                if self.audio_capture is capture:
                    self.audio_capture = None
                self._audio_capture_preview_owner = None

        # Stop auto soundcheck engine
        if self.auto_soundcheck_engine:
            engine = self.auto_soundcheck_engine
            try:
                engine.stop()
                logger.info("Auto soundcheck engine stopped")
            except Exception as e:
                logger.error(f"Error stopping auto soundcheck engine: {e}")
                errors.append(("auto_soundcheck_engine", e))
            else:
                if self.auto_soundcheck_engine is engine:
                    self.auto_soundcheck_engine = None
        if self.auto_soundcheck_engine is None:
            self.auto_soundcheck_running = False
            self.auto_soundcheck_observe_only = False

        # Stop AI mixing agent
        if self.mixing_agent:
            agent = self.mixing_agent
            try:
                agent.stop()
                logger.info("AI mixing agent stopped")
            except Exception as e:
                logger.error(f"Error stopping AI mixing agent: {e}")
                errors.append(("mixing_agent", e))
            else:
                if self.mixing_agent is agent:
                    self.mixing_agent = None
        if self.mixing_agent is None and self.mixing_agent_task is not None:
            if self.mixing_agent_task.done():
                self.mixing_agent_task = None
            else:
                self.mixing_agent_task.cancel()
                errors.append((
                    "mixing_agent_task",
                    RuntimeError("task cancelled but not drained"),
                ))

        # Disconnect mixer
        if self.mixer_client:
            mixer = self.mixer_client
            try:
                mixer.disconnect()
                logger.info("Mixer disconnected")
            except Exception as e:
                logger.error(f"Error disconnecting mixer: {e}")
                errors.append(("mixer_disconnect", e))
            else:
                if self.mixer_client is mixer:
                    self.mixer_client = None

        # Stop online training service
        if self.agent_training_service:
            try:
                service = self.agent_training_service
                try:
                    asyncio.get_running_loop()
                except RuntimeError:
                    asyncio.run(service.stop())
                    self.agent_training_service = None
                else:
                    errors.append((
                        "training_service",
                        RuntimeError("async service not drained"),
                    ))
            except Exception as e:
                logger.error("Error stopping training service: %s", e)
                errors.append(("training_service", e))

        self._controllers_cleanup_complete = not errors
        if errors:
            detail = ", ".join(
                f"{label}:{type(error).__name__}" for label, error in errors
            )
            raise RuntimeError(f"Controller cleanup incomplete ({detail})")
        logger.info("All controllers cleaned up")

    async def graceful_shutdown(self, sig=None):
        """Handle graceful shutdown."""
        if self._is_shutting_down:
            return
        self._is_shutting_down = True
        # Phase zero is synchronous and fail-closed: invalidate every queued
        # ingress item and every manual-write epoch before the first shutdown
        # await. This prevents an approved transaction from starting while a
        # controller or the mixer transport is being torn down.
        latched_before = self._manual_write_emergency_stopped
        if not latched_before:
            self._manual_write_emergency_stopped = True
            self._manual_write_emergency_generation += 1
        self._ws_ingress_emergency_generation += 1
        errors: list[dict[str, str]] = self._freeze_autonomous_writers_now(
            include_agent=not latched_before
        )

        if sig:
            logger.info(f"Received signal {sig.name}, shutting down...")
        else:
            logger.info("Initiating graceful shutdown...")

        try:
            emergency_task = self._ws_emergency_fast_lane_task
            if emergency_task is not None and emergency_task is not asyncio.current_task():
                try:
                    await self._await_task_terminal_without_cancelling(emergency_task)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    errors.append({
                        "step": "websocket_emergency_task",
                        "error": type(exc).__name__,
                    })

            try:
                await self._drain_voice_command_tasks()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                errors.append({
                    "step": "voice_command_tasks",
                    "error": type(exc).__name__,
                })

            if self.connected_clients:
                logger.info(
                    "Closing %d client connections before controller teardown...",
                    len(self.connected_clients),
                )
                clients = list(self.connected_clients)
                close_results = await asyncio.gather(
                    *(ws.close(1001, "Server shutting down") for ws in clients),
                    return_exceptions=True,
                )
                for result in close_results:
                    if isinstance(result, BaseException):
                        errors.append({
                            "step": "websocket_close",
                            "error": type(result).__name__,
                        })
                self.connected_clients.clear()

            # Every physical manual write holds this lock through readback and
            # WAL finalization. Waiting for a lock handoff proves that no such
            # transaction is still racing the later mixer disconnect. New
            # contenders see the emergency latch above and fail closed.
            async with self._console_write_lock:
                pass

            measurement_task = self._system_measurement_task
            if (
                measurement_task is not None
                and measurement_task is not asyncio.current_task()
                and not measurement_task.done()
            ):
                try:
                    await self._cancel_and_drain_async_task(measurement_task)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.error("System measurement shutdown failed: %s", exc, exc_info=True)
                    errors.append({
                        "step": "system_measurement",
                        "error": type(exc).__name__,
                    })

            legacy_task = self.auto_soundcheck_task
            if (
                legacy_task is not None
                and legacy_task is not asyncio.current_task()
                and not legacy_task.done()
            ):
                try:
                    await self._cancel_and_drain_async_task(legacy_task)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    errors.append({
                        "step": "legacy_auto_soundcheck_task",
                        "error": type(exc).__name__,
                    })
                finally:
                    if legacy_task.done() and self.auto_soundcheck_task is legacy_task:
                        self.auto_soundcheck_task = None
                        self.auto_soundcheck_websocket = None

            agent = self.mixing_agent
            if agent is not None:
                try:
                    await self._cancellation_safe_thread_call(agent.stop)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    errors.append({
                        "step": "mixing_agent",
                        "error": type(exc).__name__,
                    })
                else:
                    agent_task = self.mixing_agent_task
                    if agent_task is not None and agent_task is not asyncio.current_task():
                        try:
                            await self._cancel_and_drain_async_task(agent_task)
                        except asyncio.CancelledError:
                            raise
                        except Exception as exc:
                            errors.append({
                                "step": "mixing_agent_task",
                                "error": type(exc).__name__,
                            })
                        finally:
                            if agent_task.done() and self.mixing_agent_task is agent_task:
                                self.mixing_agent_task = None
                    if self.mixing_agent_task is None:
                        self.mixing_agent = None

            training_service = self.agent_training_service
            if training_service is not None:
                try:
                    await training_service.stop()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    errors.append({
                        "step": "training_service",
                        "error": type(exc).__name__,
                    })
                else:
                    if self.agent_training_service is training_service:
                        self.agent_training_service = None

            try:
                await self._stop_auto_soundcheck_engine_async(observe_only=False)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error("AutoSoundcheck shutdown failed: %s", exc, exc_info=True)
                errors.append({
                    "step": "auto_soundcheck_engine",
                    "error": type(exc).__name__,
                })

            async with self._audio_capture_lock:
                capture = self.audio_capture
                if capture is not None:
                    try:
                        await self._cancellation_safe_thread_call(capture.stop)
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        logger.error("Audio capture shutdown failed: %s", exc, exc_info=True)
                        errors.append({
                            "step": "audio_capture",
                            "error": type(exc).__name__,
                        })
                    else:
                        if self.audio_capture is capture:
                            self.audio_capture = None
                        self._audio_capture_preview_owner = None

            try:
                self.cleanup_all_controllers()
            except Exception as exc:
                logger.error("Controller cleanup failed: %s", exc, exc_info=True)
                errors.append({
                    "step": "controller_cleanup",
                    "error": type(exc).__name__,
                })

        except asyncio.CancelledError:
            errors.append({"step": "graceful_shutdown", "error": "CancelledError"})
            raise
        finally:
            self._shutdown_errors = errors
            if self._shutdown_event:
                self._shutdown_event.set()

        if errors:
            logger.error("Graceful shutdown incomplete: %s", errors)
        else:
            logger.info("Graceful shutdown complete")

    def _merge_automixer_yaml(self, config: dict) -> dict:
        """Overlay ``ai``, ``agent``, ``audio``, ``mixer``, ``websocket``, ``safety`` from automixer.yaml."""
        yaml_path = os.path.join(_REPO_ROOT, "config", "automixer.yaml")
        if not os.path.isfile(yaml_path):
            return config
        try:
            import yaml
            with open(yaml_path, encoding="utf-8") as f:
                yml = yaml.safe_load(f)
            if not isinstance(yml, dict):
                return config
            for key in ("ai", "agent", "audio", "mixer", "websocket", "safety", "training", "muq_eval"):
                if key not in yml or not isinstance(yml[key], dict):
                    continue
                if key not in config or not isinstance(config[key], dict):
                    config[key] = {}
                config[key].update(yml[key])
            logger.info(
                "Merged settings from %s (ai/agent/audio/mixer/websocket/safety/training/muq_eval)",
                yaml_path,
            )
        except Exception as e:
            logger.warning("automixer.yaml merge skipped: %s", e)
        return config

    def _apply_env_config_overrides(self, config: dict) -> dict:
        """Apply runtime env overrides for server-loaded config sections."""
        ai = config.setdefault("ai", {})
        training = config.setdefault("training", {})
        env_map = {
            "AUTOMIXER_LLM_BACKEND": (ai, "llm_backend", str),
            "AUTOMIXER_LLM_MODEL": (ai, "llm_model", str),
            "AUTOMIXER_OLLAMA_URL": (ai, "ollama_url", str),
            "OPENAI_API_KEY": (ai, "openai_api_key", str),
            "AUTOMIXER_OPENAI_API_KEY": (ai, "openai_api_key", str),
            "KIMI_CLI_PATH": (ai, "kimi_cli_path", str),
            "AUTOMIXER_KIMI_CLI": (ai, "kimi_cli_path", str),
            "AUTOMIXER_KIMI_WORK_DIR": (ai, "kimi_work_dir", str),
            "AUTOMIXER_KIMI_TIMEOUT": (ai, "kimi_timeout_sec", float),
            "AUTOMIXER_PERPLEXITY_KEY": (ai, "perplexity_api_key", str),
        }
        for env_key, (section, key, converter) in env_map.items():
            value = os.environ.get(env_key)
            if value is None:
                continue
            try:
                section[key] = converter(value)
            except (TypeError, ValueError):
                logger.warning("Ignoring invalid %s=%r", env_key, value)

        fallbacks = os.environ.get("AUTOMIXER_MODEL_FALLBACKS")
        if fallbacks is not None:
            ai["model_fallbacks"] = [
                item.strip()
                for item in fallbacks.split(",")
                if item.strip()
            ]
        training_enabled = os.environ.get("AUTOMIXER_TRAINING_ENABLED")
        if training_enabled is not None:
            training["enabled"] = training_enabled.lower() in {"1", "true", "yes", "on"}
        training_interval = os.environ.get("AUTOMIXER_TRAINING_INTERVAL_MINUTES")
        if training_interval is not None:
            try:
                training["interval_minutes"] = int(training_interval)
            except (TypeError, ValueError):
                logger.warning("Invalid AUTOMIXER_TRAINING_INTERVAL_MINUTES=%r", training_interval)
        training_manifest = os.environ.get("AUTOMIXER_TRAINING_MANIFEST_URL")
        if training_manifest is not None:
            training["manifest_url"] = training_manifest
        training_max_dataset_bytes = os.environ.get("AUTOMIXER_TRAINING_MAX_DATASET_BYTES")
        if training_max_dataset_bytes is not None:
            try:
                training["max_dataset_bytes"] = int(training_max_dataset_bytes)
            except (TypeError, ValueError):
                logger.warning("Invalid AUTOMIXER_TRAINING_MAX_DATASET_BYTES=%r", training_max_dataset_bytes)
        training_cleanup_downloads = os.environ.get("AUTOMIXER_TRAINING_CLEANUP_DOWNLOADS")
        if training_cleanup_downloads is not None:
            training["cleanup_downloads"] = training_cleanup_downloads.lower() in {"1", "true", "yes", "on"}
        training_safe_autostart = os.environ.get("AUTOMIXER_TRAINING_SAFE_AUTOSTART")
        if training_safe_autostart is not None:
            training["safe_autostart"] = training_safe_autostart.lower() in {"1", "true", "yes", "on"}
        training_study_enabled = os.environ.get("AUTOMIXER_TRAINING_STUDY_ENABLED")
        if training_study_enabled is not None:
            training.setdefault("study", {})["enabled"] = training_study_enabled.lower() in {"1", "true", "yes", "on"}
        training_study_max_resources = os.environ.get("AUTOMIXER_TRAINING_STUDY_MAX_RESOURCES")
        if training_study_max_resources is not None:
            try:
                training.setdefault("study", {})["max_resources_per_run"] = int(training_study_max_resources)
            except (TypeError, ValueError):
                logger.warning("Invalid AUTOMIXER_TRAINING_STUDY_MAX_RESOURCES=%r", training_study_max_resources)
        training_study_max_resources_per_type = os.environ.get("AUTOMIXER_TRAINING_STUDY_MAX_RESOURCES_PER_TYPE")
        if training_study_max_resources_per_type is not None:
            try:
                training.setdefault("study", {})["max_resources_per_type"] = int(training_study_max_resources_per_type)
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid AUTOMIXER_TRAINING_STUDY_MAX_RESOURCES_PER_TYPE=%r",
                    training_study_max_resources_per_type,
                )
        training_study_max_bytes = os.environ.get("AUTOMIXER_TRAINING_STUDY_MAX_BYTES")
        if training_study_max_bytes is not None:
            try:
                training.setdefault("study", {})["max_resource_bytes"] = int(training_study_max_bytes)
            except (TypeError, ValueError):
                logger.warning("Invalid AUTOMIXER_TRAINING_STUDY_MAX_BYTES=%r", training_study_max_bytes)
        training_study_timeout = os.environ.get("AUTOMIXER_TRAINING_STUDY_TIMEOUT")
        if training_study_timeout is not None:
            try:
                training.setdefault("study", {})["timeout_sec"] = float(training_study_timeout)
            except (TypeError, ValueError):
                logger.warning("Invalid AUTOMIXER_TRAINING_STUDY_TIMEOUT=%r", training_study_timeout)
        training_study_allowed_domains = os.environ.get("AUTOMIXER_TRAINING_STUDY_ALLOWED_DOMAINS")
        if training_study_allowed_domains is not None:
            parsed = [item.strip() for item in training_study_allowed_domains.split(",") if item.strip()]
            training.setdefault("study", {})["allowed_domains"] = parsed

        discovery = training.setdefault("discovery", {})
        training_discovery_enabled = os.environ.get("AUTOMIXER_TRAINING_DISCOVERY_ENABLED")
        if training_discovery_enabled is not None:
            discovery["enabled"] = training_discovery_enabled.lower() in {"1", "true", "yes", "on"}
        training_hf_search_limit = os.environ.get("AUTOMIXER_TRAINING_HF_SEARCH_LIMIT")
        if training_hf_search_limit is not None:
            try:
                discovery["hf_search_limit"] = int(training_hf_search_limit)
            except (TypeError, ValueError):
                logger.warning("Invalid AUTOMIXER_TRAINING_HF_SEARCH_LIMIT=%r", training_hf_search_limit)
        training_discovery_candidates = os.environ.get("AUTOMIXER_TRAINING_MAX_CANDIDATES_PER_TARGET")
        if training_discovery_candidates is not None:
            try:
                discovery["max_candidates_per_target"] = int(training_discovery_candidates)
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid AUTOMIXER_TRAINING_MAX_CANDIDATES_PER_TARGET=%r",
                    training_discovery_candidates,
                )
        training_discovery_candidate_urls = os.environ.get("AUTOMIXER_TRAINING_DISCOVERY_CANDIDATE_URLS")
        if training_discovery_candidate_urls is not None:
            candidate_urls: List[str] = []
            raw_urls = training_discovery_candidate_urls.strip()
            if raw_urls:
                try:
                    parsed = json.loads(raw_urls)
                    if isinstance(parsed, list):
                        for item in parsed:
                            if isinstance(item, str) and item.strip():
                                candidate_urls.append(item.strip())
                    else:
                        raise ValueError("candidate_urls must be json array")
                except Exception:
                    candidate_urls = [item.strip() for item in raw_urls.split(",") if item.strip()]
            discovery["candidate_urls"] = candidate_urls
        return config

    def _init_training_service(self) -> None:
        try:
            from ml.agent_training_service import AgentTrainingService
        except Exception as exc:
            logger.warning("Training service module unavailable: %s", exc)
            return
        try:
            self.agent_training_service = AgentTrainingService(
                self.config.get("training", {}),
                _REPO_ROOT,
            )
        except Exception as exc:
            logger.error("Failed to initialize training service: %s", exc, exc_info=True)
            self.agent_training_service = None

    def _load_config(self) -> dict:
        """Load configuration from default_config.json and overlay config/automixer.yaml."""
        config_path = os.path.join(_REPO_ROOT, "config", "default_config.json")
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
                logger.info("Configuration loaded from %s", config_path)
                return self._apply_env_config_overrides(self._merge_automixer_yaml(config))
        except Exception as e:
            logger.warning("Failed to load config from %s: %s. Using defaults.", config_path, e)
            base = {
                "automation": {
                    "auto_gain": {
                        "bleeding_rejection": {
                            "enabled": True,
                            "correlation_threshold": 0.7,
                            "level_difference_threshold_db": 8.0,
                        }
                    }
                }
            }
            return self._apply_env_config_overrides(self._merge_automixer_yaml(base))

    def _get_user_config_path(self) -> str:
        """Get path to user config file."""
        return get_user_config_path(os.path.dirname(os.path.dirname(__file__)))

    def _load_user_config(self) -> dict:
        """Load user-saved settings from user_config.json."""
        return load_user_config(self._get_user_config_path())

    def _normalize_mixer_user_settings(self, settings: dict) -> dict:
        """Normalize mixer settings so backend and frontend can share one payload."""
        return normalize_mixer_user_settings(settings)

    def _normalize_user_config(self, data: dict) -> dict:
        """Normalize persisted user config sections for frontend/backend compatibility."""
        return normalize_user_config(data)

    def _save_user_config(self, section: str, settings: dict):
        """Save user settings to user_config.json under a given section."""
        try:
            save_user_config(self._get_user_config_path(), section, settings)
        except Exception as e:
            logger.error(f"Failed to save user config: {e}")
            raise

    async def register_client(self, websocket: WebSocketServerProtocol):
        self.connected_clients.add(websocket)
        logger.info(f"Client connected. Total clients: {len(self.connected_clients)}")

        if self.mixer_client and self.mixer_client.is_connected:
            await self.send_to_client(websocket, {
                "type": "connection_status",
                "connected": True,
                "mode": self.connection_mode,
                "state": self.mixer_client.get_state()
            })

    async def unregister_client(self, websocket: WebSocketServerProtocol):
        self.connected_clients.discard(websocket)
        await self._stop_owned_audio_capture_preview(websocket)
        logger.info(f"Client disconnected. Total clients: {len(self.connected_clients)}")

    async def _stop_owned_audio_capture_preview(
        self,
        websocket: Optional[WebSocketServerProtocol] = None,
    ) -> bool:
        """Stop only a preview owned by the requester (or during shutdown)."""
        async with self._audio_capture_lock:
            owner = self._audio_capture_preview_owner
            if owner is None or (websocket is not None and owner is not websocket):
                return False
            capture = self.audio_capture
            if capture is not None:
                stop_outcome = {"completed": False}

                def stop_capture() -> None:
                    capture.stop()
                    stop_outcome["completed"] = True

                try:
                    await self._cancellation_safe_thread_call(stop_capture)
                finally:
                    # The helper drains the worker even when this coroutine is
                    # cancelled. Release ownership only after that worker has
                    # actually stopped the exact capture; retain both refs when
                    # stop failed so shutdown can retry and report the failure.
                    if stop_outcome["completed"]:
                        if self.audio_capture is capture:
                            self.audio_capture = None
                        if self._audio_capture_preview_owner is owner:
                            self._audio_capture_preview_owner = None
                return True
            self._audio_capture_preview_owner = None
            return True

    def _is_connection_closed_error(self, e: Exception) -> bool:
        """True if the exception indicates the client already closed the connection."""
        return is_connection_closed_error(e)

    async def send_to_client(self, websocket: WebSocketServerProtocol, message: dict):
        return await send_json(
            websocket,
            message,
            converter=convert_numpy_types,
            logger_=logger,
        )

    async def _send_runtime_error(self, websocket: WebSocketServerProtocol, error: Exception, *, payload_type: str = "error") -> None:
        """Best-effort runtime error reporting that does not mask cancellation or closed sockets."""
        try:
            await self.send_to_client(websocket, {
                "type": payload_type,
                "error": str(error),
            })
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self._is_connection_closed_error(exc):
                logger.info("Skipped runtime error delivery to closed client: %s", exc)
            else:
                logger.warning("Failed to deliver runtime error to client: %s", exc, exc_info=True)

    def _handle_external_fader_update(self, address: str, args: tuple[Any, ...]) -> None:
        """Track manual fader/trim moves and forward them to live controllers when possible."""
        parsed = self._parse_manual_write_address(address)
        if parsed is None:
            return
        channel, value_type = parsed
        if value_type not in {"fader", "gain", "mute", "pan"} or not args:
            return
        if not 1 <= channel <= 48:
            logger.debug("Ignoring external fader update for out-of-range channel %s: %s", channel, address)
            return

        if value_type == "mute":
            parsed_mute = self._coerce_optional_bool(args[0])
            if parsed_mute is None:
                return
            safe_value = bool(self._clamp_manual_write_value(bool(parsed_mute), "mute"))
            self.last_mute_values[channel] = safe_value
            return
        try:
            value = float(args[0])
        except (TypeError, ValueError, IndexError, AttributeError):
            return
        if not math.isfinite(value):
            logger.debug(
                "Ignoring external update with non-finite value for channel %s: %s",
                channel,
                value,
            )
            return
        value = float(value)

        if value_type == "fader":
            safe_value = self._clamp_manual_write_value(value, "fader")
            if safe_value != value:
                logger.debug(
                    "Clamped external fader update value for channel %s: %s -> %s",
                    channel,
                    value,
                    safe_value,
                )
            self.last_fader_values[channel] = safe_value
            controller = self.auto_fader_controller
            if controller and controller.realtime_enabled:
                try:
                    controller.handle_external_fader_change(channel, safe_value)
                except Exception as exc:
                    logger.warning(
                        "Auto Fader external change handler failed for channel %s: %s",
                        channel,
                        exc,
                        exc_info=True,
                    )
            return

        # value_type == "gain" (trim)
        if value_type == "gain":
            safe_value = self._clamp_manual_write_value(value, "gain")
            if safe_value != value:
                logger.debug(
                    "Clamped external gain update value for channel %s: %s -> %s",
                    channel,
                    value,
                    safe_value,
                )
            gain_controller = self.gain_staging
            if gain_controller and gain_controller.realtime_correction_enabled:
                try:
                    external_trim_handler = getattr(gain_controller, "handle_external_trim_change", None)
                    if external_trim_handler:
                        external_trim_handler(channel, safe_value)
                except Exception as exc:
                    logger.warning(
                        "Auto gain-staging external trim handler failed for channel %s: %s",
                        channel,
                        exc,
                        exc_info=True,
                    )
            return

        # value_type == "pan" (read-only for now; parse/validate for safety)
        if value_type == "pan":
            safe_value = self._clamp_manual_write_value(value, "pan")
            if safe_value != value:
                logger.debug(
                    "Clamped external pan update value for channel %s: %s -> %s",
                    channel,
                    value,
                    safe_value,
                )
            self.last_pan_values[channel] = safe_value
            return

    def _schedule_mixer_update_broadcast(
        self,
        loop: asyncio.AbstractEventLoop,
        address: str,
        args: tuple[Any, ...],
    ) -> None:
        """Queue mixer updates back onto the asyncio loop from transport callbacks."""
        try:
            loop.call_soon_threadsafe(
                lambda: asyncio.create_task(self.broadcast({
                    "type": "mixer_update",
                    "address": address,
                    "values": args,
                }))
            )
        except RuntimeError as exc:
            logger.warning("Dropped mixer update broadcast for %s: %s", address, exc)

    async def broadcast(self, message: dict):
        logger.debug("Broadcasting message: %s to %s clients", message.get("type"), len(self.connected_clients))
        disconnected = await broadcast_json(
            self.connected_clients,
            message,
            sender=self.send_to_client,
            logger_=logger,
        )
        for client in disconnected:
            if client in self.connected_clients:
                self.connected_clients.discard(client)
                logger.info("Removed disconnected client from broadcast list. Total clients: %s", len(self.connected_clients))

    def _rebind_controller_mixer_clients(self, client):
        """Point long-lived controllers at the active mixer client."""
        for attr_name in (
            "gain_staging",
            "safe_gain_calibrator",
            "auto_eq_controller",
            "multi_channel_auto_eq_controller",
            "phase_alignment_controller",
            "auto_fader_controller",
            "auto_compressor_controller",
        ):
            controller = getattr(self, attr_name, None)
            if controller is not None and hasattr(controller, "mixer_client"):
                controller.mixer_client = client
        if self.mixing_agent is not None:
            self.mixing_agent.mixer = client

    async def _send_soundcheck_observation(self, websocket, message: str = "", operation: dict = None, summary: dict = None):
        """Send a dry-run observation event to the initiating client."""
        if websocket is None:
            return
        payload = {
            "type": "auto_soundcheck_observation",
            "message": message,
        }
        if operation is not None:
            payload["operation"] = operation
        if summary is not None:
            payload["summary"] = summary
        await self.send_to_client(websocket, payload)

    def _ensure_audio_capture(
        self,
        device_id: Optional[Union[str, int]],
        *,
        num_channels: int,
        sample_rate: int = 48000,
        buffer_seconds: float = 8.0,
    ) -> AudioCapture:
        """Ensure unified AudioCapture is running for the requested input device."""
        num_channels, sample_rate, buffer_seconds = validate_live_capture_dimensions(
            num_channels,
            sample_rate,
            buffer_seconds,
        )
        if not HAS_SOUNDDEVICE:
            raise RuntimeError(
                "Real audio input is unavailable; synthetic silence cannot satisfy live meters"
            )
        source_type = AudioSourceType.SOUNDDEVICE
        if device_id in (None, "", "default"):
            requested_device = None
        elif isinstance(device_id, str):
            normalized_device = device_id.strip()
            if normalized_device == "" or normalized_device.lower() == "default":
                requested_device = None
            elif normalized_device.isdigit():
                requested_device = int(normalized_device)
            else:
                requested_device = normalized_device
        else:
            requested_device = device_id

        needs_restart = (
            self.audio_capture is None or
            not getattr(self.audio_capture, "running", False) or
            getattr(self.audio_capture, "device_name", None) != requested_device or
            getattr(self.audio_capture, "num_channels", 0) < num_channels or
            getattr(self.audio_capture, "sample_rate", sample_rate) != sample_rate or
            float(getattr(self.audio_capture, "buffer_seconds", 0.0)) < buffer_seconds
        )

        if needs_restart:
            if self.audio_capture:
                previous_capture = self.audio_capture
                stopped = self._safe_cleanup_call(
                    previous_capture.stop,
                    "Error stopping audio capture before reconfigure",
                )
                if not stopped:
                    raise RuntimeError(
                        "Existing audio capture did not stop; refusing replacement"
                    )
                if self.audio_capture is previous_capture:
                    self.audio_capture = None
            self.audio_capture = AudioCapture(
                num_channels=num_channels,
                sample_rate=sample_rate,
                buffer_seconds=buffer_seconds,
                source_type=source_type,
                device_name=requested_device,
            )
            self.audio_capture.allow_silence_fallback = False
            try:
                self.audio_capture.start()
            except Exception:
                failed_capture = self.audio_capture
                capture_thread = getattr(failed_capture, "_capture_thread", None)
                cleanup_incomplete = bool(
                    isinstance(failed_capture, AudioCapture)
                    and (
                        getattr(failed_capture, "_stream", None) is not None
                        or (
                            capture_thread is not None
                            and callable(getattr(capture_thread, "is_alive", None))
                            and capture_thread.is_alive() is True
                        )
                    )
                )
                if not cleanup_incomplete and self.audio_capture is failed_capture:
                    self.audio_capture = None
                raise
            if self.audio_capture.source_type != AudioSourceType.SOUNDDEVICE:
                self.audio_capture.stop()
                self.audio_capture = None
                raise RuntimeError(
                    "Real audio input did not start; synthetic capture is not accepted"
                )

        return self.audio_capture

    async def _ensure_audio_capture_async(
        self,
        device_id: Optional[Union[str, int]],
        *,
        num_channels: int,
        sample_rate: int = 48000,
        buffer_seconds: float = 8.0,
    ) -> AudioCapture:
        """Start/reconfigure a device off-loop and drain it on cancellation."""

        previous_capture = self.audio_capture
        try:
            return await self._cancellation_safe_thread_call(
                self._ensure_audio_capture,
                device_id,
                num_channels=num_channels,
                sample_rate=sample_rate,
                buffer_seconds=buffer_seconds,
            )
        except asyncio.CancelledError:
            # The worker is drained before cancellation is re-raised. If it
            # created a replacement, no caller has assigned ownership yet, so
            # stop that exact orphan before releasing the ownership lock.
            orphan = self.audio_capture
            if orphan is not None and orphan is not previous_capture:
                try:
                    await self._cancellation_safe_thread_call(orphan.stop)
                except Exception as cleanup_error:
                    logger.error(
                        "Cancelled audio start left a retryable capture: %s",
                        cleanup_error,
                        exc_info=True,
                    )
                else:
                    if self.audio_capture is orphan:
                        self.audio_capture = None
                    self._audio_capture_preview_owner = None
            raise

    def _activate_observation_mode(self, websocket, loop):
        """Wrap the live mixer client with a dry-run proxy for the soundcheck cycle."""
        if not self.mixer_client:
            return None

        def emit(coro):
            try:
                running_loop = asyncio.get_running_loop()
            except RuntimeError:
                running_loop = None

            if running_loop is loop:
                task = loop.create_task(coro)
                task.add_done_callback(
                    lambda t: logger.warning(
                        "Observation emit failed: %s",
                        t.exception(),
                    ) if t.exception() else None
                )
                return

            future = asyncio.run_coroutine_threadsafe(coro, loop)
            future.add_done_callback(
                lambda f: logger.warning(
                    "Observation emit failed: %s",
                    f.exception(),
                ) if f.exception() else None
            )

        def on_command(operation: dict):
            emit(self._send_soundcheck_observation(
                websocket,
                message=f"[OBSERVE] {operation['message']}",
                operation=operation,
            ))

        original_client = self.mixer_client
        proxy = ObservationMixerClient(original_client, on_command=on_command)
        self._observation_mixer_client = proxy
        self.mixer_client = proxy
        self._rebind_controller_mixer_clients(proxy)
        return original_client

    async def _deactivate_observation_mode(self, websocket, original_client):
        """Restore the real mixer client and emit a final dry-run summary."""
        proxy = self._observation_mixer_client
        if proxy is not None:
            summary = proxy.get_summary()
            await self._send_soundcheck_observation(
                websocket,
                message=(
                    f"[OBSERVE] Cycle complete. Intercepted "
                    f"{summary['total_operations']} mixer write operation(s)."
                ),
                summary=summary,
            )

        if original_client is not None:
            self.mixer_client = original_client
            self._rebind_controller_mixer_clients(original_client)

        self._observation_mixer_client = None

    def _is_real_mixer_connected(self) -> bool:
        return bool(
            self.mixer_client
            and getattr(self.mixer_client, "is_connected", False)
            and self.connection_mode in {"wing", "dlive", "mixing_station"}
        )

    def _is_agent_dry_run(self) -> bool:
        return bool(self._is_real_mixer_connected())

    def _enforce_agent_dry_run_mode(self):
        """Keep agent in suggest-only mode while a real mixer is connected."""
        from ai.agent import AgentMode

        if not self.mixing_agent:
            return
        if self._is_agent_dry_run():
            self.mixing_agent.set_mode(AgentMode.SUGGEST)
            self.agent_auto_apply_enabled = False
            logger.info("Real mixer connected: forcing AI agent into SUGGEST (dry-run) mode")

    def _apply_llm_dry_run_overrides(self, mode: str = None, allow_auto_apply: bool = True, force_auto_apply: bool = False):
        if force_auto_apply:
            logger.warning("Ignoring removed force_auto_apply authority")
        if self._is_real_mixer_connected():
            if allow_auto_apply:
                logger.info(
                    "Real mixer connected: forcing AI agent to dry-run mode (AUTO disabled)."
                )
            allow_auto_apply = False
            if mode is None:
                mode = "suggest"
            elif str(mode).lower() == "auto":
                mode = "suggest"
        return mode, allow_auto_apply

    def _normalize_agent_mode_request(self, mode: str | None) -> str:
        """Normalize free-form AI agent mode aliases into a canonical value."""
        normalized = str(mode or "").strip().lower().replace("-", "_").replace(" ", "_")
        normalized = "_".join(filter(None, normalized.split("_")))
        if not normalized:
            return "auto"

        agent_mode_aliases = {
            "auto": "auto",
            "automatic": "auto",
            "full_autonomous": "auto",
            "full_autonomous_mixing": "auto",
            "full_autonomous_mixing_mode": "auto",
            "full_autonomous_mode": "auto",
            "auto_mode": "auto",
            "autonomous": "auto",
            "suggest": "suggest",
            "suggestion": "suggest",
            "suggest_mode": "suggest",
            "assist": "suggest",
            "assisted": "suggest",
            "assist_mode": "suggest",
            "assisted_mode": "suggest",
            "analysis": "suggest",
            "analysis_mode": "suggest",
            "analysismode": "suggest",
            "analysis_mode_mode": "suggest",
            "manual": "manual",
            "manual_mode": "manual",
            "manualmode": "manual",
            "supervised": "suggest",
            "supervised_automix": "suggest",
            "supervised_auto_mix": "suggest",
            "supervised_automix_mode": "suggest",
            "supervised_mode": "suggest",
            "supervised_auto_mode": "suggest",
            "full_manual": "manual",
            "full_manual_mode": "manual",
        }
        return agent_mode_aliases.get(normalized, normalized)

    def _coerce_agent_mode(self, mode: str = None, allow_auto_apply: bool = True):
        """Return AgentMode for current run; AUTO is the default test-stage behavior."""
        from ai.agent import AgentMode

        requested = self._normalize_agent_mode_request(
            mode if mode is not None else self.config.get("agent", {}).get("mode", "auto")
        )
        try:
            agent_mode = AgentMode(requested)
        except ValueError:
            agent_mode = AgentMode.AUTO

        if agent_mode == AgentMode.AUTO and not allow_auto_apply:
            logger.warning("AI agent AUTO mode requested with allow_auto_apply=false; using SUGGEST mode")
            return AgentMode.SUGGEST
        return agent_mode

    def _normalize_agent_instrument(self, preset: str = None, name: str = "") -> str:
        """Normalize project preset names to the AI rule-engine vocabulary."""
        value = (preset or name or "unknown").strip()
        mapping = {
            "leadVocal": "lead_vocal",
            "backVocal": "backing_vocal",
            "electricGuitar": "electric_guitar",
            "acousticGuitar": "acoustic_guitar",
            "bass": "bass_guitar",
            "hihat": "hi_hat",
            "overheads": "overhead",
            "keys": "keys_piano",
        }
        if value in mapping:
            return mapping[value]
        normalized = re.sub(r"(?<!^)(?=[A-Z])", "_", value).lower()
        normalized = re.sub(r"[^a-z0-9]+", "_", normalized).strip("_")
        return mapping.get(normalized, normalized or "unknown")

    def _selected_agent_channels(self, channels=None) -> List[int]:
        """Choose a bounded channel list for the agent seed pass."""
        if channels:
            selected = []
            for ch in channels:
                try:
                    channel = int(ch)
                except (TypeError, ValueError):
                    continue
                if channel > 0:
                    selected.append(channel)
            return sorted(set(selected))

        if self.auto_soundcheck_engine:
            selected = list(
                getattr(self.auto_soundcheck_engine, "selected_channels", [])
                or []
            )
            if selected:
                return [int(ch) for ch in selected]

        if self.audio_capture and getattr(self.audio_capture, "running", False):
            return list(range(1, min(self.audio_capture.num_channels, 48) + 1))

        return list(range(1, 9))

    def _agent_channel_state(self, channel: int, name: str = "", preset: str = "", metrics: dict = None) -> dict:
        """Build one conservative channel observation for MixingAgent."""
        metrics = metrics or {}
        peak = metrics.get("peak_db")
        rms = metrics.get("rms_db")
        lufs = metrics.get("lufs", metrics.get("lufs_momentary"))
        has_real_meter = peak is not None or rms is not None or lufs is not None

        peak_db = float(peak if peak is not None else -12.0)
        rms_db = float(rms if rms is not None else -30.0)
        lufs_db = float(lufs if lufs is not None else rms_db)
        instrument = self._normalize_agent_instrument(preset, name)

        muted = False
        if self.mixer_client and hasattr(self.mixer_client, "get_mute"):
            try:
                muted = bool(self.mixer_client.get_mute(channel))
            except Exception:
                muted = False

        return {
            "channel_id": channel,
            "name": name or f"Ch {channel}",
            "instrument": instrument,
            "rms_db": rms_db,
            "peak_db": peak_db,
            "true_peak_db": peak_db,
            "lufs_momentary": lufs_db,
            "dynamic_range_db": max(0.0, peak_db - rms_db),
            "is_muted": muted,
            "channel_armed": bool(has_real_meter and (rms_db > -50.0 or peak_db > -45.0)),
            "needs_attention": bool(
                has_real_meter and
                (rms_db > -50.0 or peak_db > -45.0) and
                (peak_db > -6.0 or peak_db < -30.0)
            ),
        }

    def collect_agent_channel_states(self, channels=None) -> Dict[int, dict]:
        """Collect current channel observations for the AI MixingAgent."""
        self._sync_runtime_from_auto_soundcheck()
        selected_channels = self._selected_agent_channels(channels)
        names: Dict[int, str] = {}

        for ch in selected_channels:
            if self.mixer_client and hasattr(self.mixer_client, "get_channel_name"):
                try:
                    names[ch] = self.mixer_client.get_channel_name(ch) or f"Ch {ch}"
                except Exception:
                    names[ch] = f"Ch {ch}"
            else:
                names[ch] = f"Ch {ch}"

        recognized = {}
        try:
            recognized = scan_and_recognize(names)
        except Exception as e:
            logger.debug("Agent channel recognition failed: %s", e)

        engine_channels = {}
        if self.auto_soundcheck_engine:
            try:
                engine_channels = {
                    channel: {
                        "peak_db": info.peak_db,
                        "rms_db": info.rms_db,
                        "lufs": info.lufs,
                    }
                    for channel, info in dict(
                        self.auto_soundcheck_engine.channels
                    ).items()
                }
            except Exception:
                engine_channels = {}

        states: Dict[int, dict] = {}
        for ch in selected_channels:
            metrics = {}
            engine_info = engine_channels.get(ch) or engine_channels.get(str(ch)) or {}
            if engine_info:
                metrics.update({
                    "peak_db": engine_info.get("peak_db"),
                    "rms_db": engine_info.get("rms_db"),
                    "lufs": engine_info.get("lufs"),
                })

            if self.audio_capture and getattr(self.audio_capture, "running", False):
                try:
                    metrics["rms_db"] = self.audio_capture.get_rms(ch)
                    metrics["peak_db"] = self.audio_capture.get_peak(ch)
                    metrics["lufs"] = self.audio_capture.get_lufs(ch)
                except Exception:
                    pass

            rec = recognized.get(ch, {})
            preset = engine_info.get("preset") or rec.get("preset") or ""
            states[ch] = self._agent_channel_state(ch, names.get(ch), preset, metrics)

        return states

    def update_mixing_agent_channel(self, channel: int, data: dict):
        """Feed one channel update from the soundcheck engine into the AI agent."""
        if not self.mixing_agent:
            return
        self._sync_runtime_from_auto_soundcheck()
        self.mixing_agent.mixer = self.mixer_client
        state = self._agent_channel_state(
            int(channel),
            name=data.get("name", f"Ch {channel}"),
            preset=data.get("preset", ""),
            metrics={
                "peak_db": data.get("peak_db"),
                "rms_db": data.get("rms_db"),
                "lufs": data.get("lufs"),
            },
        )
        self.mixing_agent.update_channel_state(int(channel), state)

    async def init_mixing_agent(self, mode: str = None, channels=None, use_llm: bool = True,
                                allow_auto_apply: bool = True, start: bool = False,
                                force_auto_apply: bool = False):
        """Create or reconfigure the AI MixingAgent (AUTO apply by default for testing)."""
        from ai.agent import MixingAgent
        from ai.knowledge_base import KnowledgeBase
        from ai.llm_client import LLMClient
        from ai.rule_engine import RuleEngine

        self._sync_runtime_from_auto_soundcheck()
        self._agent_force_auto_apply = False
        mode, allow_auto_apply = self._apply_llm_dry_run_overrides(
            mode=mode,
            allow_auto_apply=allow_auto_apply,
            force_auto_apply=force_auto_apply,
        )
        agent_config = self.config.get("agent", {})
        ai_config = self.config.get("ai", {})
        agent_mode = self._coerce_agent_mode(mode, allow_auto_apply=allow_auto_apply)
        self.agent_auto_apply_enabled = bool(allow_auto_apply and agent_mode.value == "auto")

        llm_client = None
        if use_llm:
            llm_client = LLMClient(
                backend=ai_config.get("llm_backend", "auto"),
                model=ai_config.get("llm_model", "gpt-5.4"),
                ollama_url=ai_config.get("ollama_url", "http://localhost:11434"),
                perplexity_api_key=ai_config.get("perplexity_api_key") or None,
                openai_api_key=ai_config.get("openai_api_key") or None,
                openai_url=ai_config.get("openai_url", "https://api.openai.com/v1/responses"),
                model_fallbacks=ai_config.get("model_fallbacks") or None,
                openai_reasoning_effort=ai_config.get("openai_reasoning_effort", "low"),
                kimi_cli_path=(ai_config.get("kimi_cli_path") or None),
                kimi_work_dir=(ai_config.get("kimi_work_dir") or None),
                kimi_timeout_sec=float(ai_config.get("kimi_timeout_sec", 120)),
            )

        if self.mixing_agent is None:
            knowledge_dir = ai_config.get("knowledge_dir") or None
            self.mixing_agent = MixingAgent(
                knowledge_base=KnowledgeBase(
                    knowledge_dir=knowledge_dir,
                    use_vector_db=False,
                    allowed_categories=KnowledgeBase.AGENT_RUNTIME_CATEGORIES,
                ),
                rule_engine=RuleEngine(),
                llm_client=llm_client,
                mixer_client=self.mixer_client,
                mode=agent_mode,
                cycle_interval=float(agent_config.get("cycle_interval", 0.5)),
            )
        else:
            self.mixing_agent.set_mode(agent_mode)
            self.mixing_agent.llm = llm_client
            self.mixing_agent.mixer = self.mixer_client

        self.mixing_agent.set_confidence_threshold(float(agent_config.get("confidence_threshold", 0.6)))
        self.mixing_agent._max_actions_per_cycle = int(agent_config.get("max_actions_per_cycle", 5))
        self.mixing_agent.configure_safety_limits(
            max_fader_step_db=agent_config.get("max_fader_step_db", 1.0),
            max_eq_step_db=agent_config.get("max_eq_step_db", 2.0),
            max_comp_threshold_step_db=agent_config.get("max_comp_threshold_step_db", 3.0),
            max_pan_step=agent_config.get("max_pan_step", 0.25),
        )
        self.mixing_agent.update_channel_states_batch(self.collect_agent_channel_states(channels))

        if start and not self.mixing_agent.state.is_running:
            self.mixing_agent_task = asyncio.create_task(self.mixing_agent.start())
            self.mixing_agent_task.add_done_callback(
                lambda t: logger.warning("AI mixing agent task failed: %s", t.exception())
                if not t.cancelled() and t.exception() else None
            )

        self._enforce_agent_dry_run_mode()

        return self.mixing_agent

    def _sync_runtime_from_auto_soundcheck(self):
        """Preserve server ownership of mixer/audio runtime services.

        Soundcheck receives already-running shared services when available. An
        engine-owned fallback must never replace the server references because
        it is stopped and disconnected with the engine lifecycle.
        """
        return

    async def stop_mixing_agent(self):
        """Stop the AI MixingAgent loop while preserving its last status/history."""
        if self.mixing_agent:
            await self._cancellation_safe_thread_call(self.mixing_agent.stop)
        if self.mixing_agent_task:
            self.mixing_agent_task.cancel()
            self.mixing_agent_task = None
        self.agent_auto_apply_enabled = False

    async def handle_client_message(
        self,
        websocket: WebSocketServerProtocol,
        message: str | bytes,
    ):
        try:
            data = json.loads(message)
            message_size = (
                len(message)
                if isinstance(message, bytes)
                else len(message.encode("utf-8", errors="replace"))
            )
            if not isinstance(data, Mapping):
                await self.send_to_client(
                    websocket,
                    {
                        "type": "error",
                        "error": "WebSocket message must be a JSON object",
                    },
                )
                return
            msg_type = data.get("type")
            logger.debug(
                "Received WebSocket message type=%s bytes=%s",
                msg_type,
                message_size,
            )
            if self._is_shutting_down and msg_type not in EMERGENCY_FAST_LANE_TYPES:
                logger.warning("Rejected WebSocket message during shutdown")
                return
            if await self._block_unauthenticated_control_message(
                websocket,
                msg_type,
                data,
            ):
                return

            handler = self._dispatch.get(msg_type)
            if handler is not None:
                await handler(websocket, data)
            else:
                logger.debug("Unknown WebSocket message type: %s", msg_type)

        except json.JSONDecodeError as e:
            logger.debug("Invalid WebSocket JSON rejected: %s", e)
            await self.send_to_client(websocket, {
                "type": "error",
                "error": f"Invalid JSON: {e}"
            })
        except asyncio.CancelledError:
            logger.info("Client message handling cancelled")
            raise
        except Exception as e:
            logger.error(f"Error handling message: {e}", exc_info=True)
            await self._send_runtime_error(websocket, e)

    async def connect_wing(
        self,
        ip: str,
        send_port: int = 2223,
        receive_port: int = 2223,
        *,
        observation_only: Optional[bool] = None,
        record_telemetry: Optional[bool] = None,
        telemetry_session_label: Optional[str] = None,
    ):
        """
        Connect directly to Wing mixer via OSC

        Args:
            ip: Wing IP address
            send_port: Wing OSC port (default 2223)
        """
        if await self.block_packaged_offline_transport("connect_wing"):
            return False
        if self._network_runtime_started and not self._ws_auth_required:
            observation_only = True
            record_telemetry = False
        # Disconnect existing connection if any
        await self.disconnect_mixer()
        self._agent_force_auto_apply = False
        if observation_only is not None:
            self._wing_observation_only = bool(observation_only)
        if record_telemetry is not None:
            self._wing_record_telemetry = bool(record_telemetry)

        try:
            osc_port = receive_port or send_port or 2223
            if send_port and receive_port and send_port != receive_port:
                logger.info(
                    "Wing connect requested with send_port=%s receive_port=%s; using OSC port %s",
                    send_port,
                    receive_port,
                    osc_port,
                )

            self.mixer_client = EnhancedOSCClient(ip=ip, port=osc_port)
            self.connection_mode = 'wing'

            # Store the event loop reference for thread-safe callback
            loop = asyncio.get_running_loop()

            def on_mixer_update(address: str, *args):
                self._handle_external_fader_update(address, args)
                self._schedule_mixer_update_broadcast(loop, address, args)

            self.mixer_client.subscribe("*", on_mixer_update)

            success = self.mixer_client.connect()

            if success:
                if self.mixing_agent is not None:
                    self.mixing_agent.mixer = self.mixer_client
                self._enforce_agent_dry_run_mode()
                await self.broadcast({
                    "type": "connection_status",
                    "connected": True,
                    "mode": "wing",
                    "ip": ip,
                    "state": self.mixer_client.get_state(),
                    "observation_only": self._wing_observation_only,
                    "record_telemetry": self._wing_record_telemetry,
                    "telemetry_session_label": telemetry_session_label,
                })
                logger.info("Wing connected successfully (direct OSC)")
            else:
                disconnect_failed_client = getattr(self.mixer_client, "disconnect", None)
                if callable(disconnect_failed_client):
                    self._safe_cleanup_call(
                        disconnect_failed_client,
                        "Error stopping failed Wing connection attempt",
                    )
                self.mixer_client = None
                self.connection_mode = None
                await self.broadcast({
                    "type": "connection_status",
                    "connected": False,
                    "error": "Connection failed - Wing did not respond"
                })

        except asyncio.CancelledError:
            self.mixer_client = None
            self.connection_mode = None
            raise
        except Exception as e:
            logger.error(f"Error connecting to Wing: {e}")
            self.mixer_client = None
            self.connection_mode = None
            await self.broadcast({
                "type": "connection_status",
                "connected": False,
                "error": str(e)
            })

    async def connect_dlive(self, ip: str = "192.168.3.70", port: int = 51328,
                            tls: bool = False, midi_base_channel: int = 0,
                            tls_ca_file: Optional[str] = None,
                            tls_server_hostname: Optional[str] = None):
        """Connect to Allen & Heath dLive mixer via MIDI over TCP."""
        if await self.block_packaged_offline_transport("connect_dlive"):
            return False
        await self.disconnect_mixer()
        self._agent_force_auto_apply = False

        try:
            client = DLiveClient(
                ip=ip,
                port=port,
                tls=tls,
                tls_ca_file=tls_ca_file,
                tls_server_hostname=tls_server_hostname,
                midi_base_channel=midi_base_channel,
            )
            success = client.connect()

            if success:
                self.mixer_client = client
                self.connection_mode = 'dlive'
                if self.mixing_agent is not None:
                    self.mixing_agent.mixer = self.mixer_client
                self._enforce_agent_dry_run_mode()

                # Request channel names from dLive
                try:
                    client.request_channel_names(48)
                except Exception as e:
                    logger.warning(f"Could not request channel names: {e}")

                await self.broadcast({
                    "type": "connection_status",
                    "connected": True,
                    "mode": "dlive",
                    "ip": ip,
                    "state": client.get_state()
                })
                logger.info(f"dLive connected at {ip}:{port}")
            else:
                await self.broadcast({
                    "type": "connection_status",
                    "connected": False,
                    "error": "dLive connection failed"
                })

        except asyncio.CancelledError:
            self.mixer_client = None
            self.connection_mode = None
            raise
        except Exception as e:
            logger.error(f"Error connecting to dLive: {e}")
            self.mixer_client = None
            self.connection_mode = None
            await self.broadcast({
                "type": "connection_status",
                "connected": False,
                "error": str(e)
            })

    async def connect_mixing_station(self, host: str = "127.0.0.1", osc_port: int = 8000, rest_port: int = 8080):
        """
        Connect to mixer via Mixing Station app

        Args:
            host: Mixing Station host (usually localhost)
            osc_port: Mixing Station OSC port
            rest_port: Mixing Station REST API port
        """
        if await self.block_packaged_offline_transport("connect_mixing_station"):
            return False
        # Disconnect existing connection if any
        await self.disconnect_mixer()
        self._agent_force_auto_apply = False

        try:
            self.mixer_client = MixingStationClient(host, osc_port, rest_port)
            self.connection_mode = 'mixing_station'

            # Store the event loop reference for thread-safe callback
            loop = asyncio.get_running_loop()

            def on_mixer_update(address: str, *args):
                self._handle_external_fader_update(address, args)
                self._schedule_mixer_update_broadcast(loop, address, args)

            self.mixer_client.subscribe("*", on_mixer_update)

            success = self.mixer_client.connect()

            if success:
                if self.mixing_agent is not None:
                    self.mixing_agent.mixer = self.mixer_client
                self._enforce_agent_dry_run_mode()
                await self.broadcast({
                    "type": "connection_status",
                    "connected": True,
                    "mode": "mixing_station",
                    "host": host,
                    "osc_port": osc_port,
                    "state": self.mixer_client.get_state()
                })
                logger.info("Connected to Mixing Station successfully")
            else:
                self.mixer_client = None
                self.connection_mode = None
                await self.broadcast({
                    "type": "connection_status",
                    "connected": False,
                    "error": "Connection failed - Mixing Station not responding. Ensure API is enabled in settings."
                })

        except asyncio.CancelledError:
            self.mixer_client = None
            self.connection_mode = None
            raise
        except Exception as e:
            logger.error(f"Error connecting to Mixing Station: {e}")
            self.mixer_client = None
            self.connection_mode = None
            await self.broadcast({
                "type": "connection_status",
                "connected": False,
                "error": str(e)
            })

    async def discover_mixing_station(self):
        """Try to auto-discover Mixing Station on common ports"""
        if await self.block_packaged_offline_transport("discover_mixing_station"):
            return False
        logger.info("Discovering Mixing Station...")

        result = discover_mixing_station()

        if result:
            await self.broadcast({
                "type": "mixing_station_discovered",
                "found": True,
                "osc_port": result['osc_port'],
                "rest_port": result['rest_port']
            })
            logger.info(f"Mixing Station discovered on port {result['osc_port']}")
        else:
            await self.broadcast({
                "type": "mixing_station_discovered",
                "found": False,
                "error": "Mixing Station not found. Ensure it's running and API is enabled."
            })
            logger.info("Mixing Station not found")

    async def disconnect_mixer(self):
        """Disconnect from current mixer (Wing or Mixing Station)"""
        self._agent_force_auto_apply = False
        if self.mixer_client:
            self.mixer_client.disconnect()
            self.mixer_client = None
            if self.mixing_agent is not None:
                self.mixing_agent.mixer = None
            mode = self.connection_mode
            self.connection_mode = None

            for run_id in list(getattr(self, "_soundcheck_manifests", {})):
                self._invalidate_soundcheck_console_fence(
                    run_id,
                    "mixer_not_connected",
                )

            await self.broadcast({
                "type": "connection_status",
                "connected": False
            })
            if isinstance(self.last_soundcheck_readiness, Mapping):
                await self.broadcast({
                    "type": "auto_soundcheck_status",
                    "status": "readiness_invalidated",
                    "running": False,
                    "is_running": False,
                    "message": "Mixer disconnected; soundcheck readiness invalidated.",
                    "soundcheck_readiness": dict(
                        self.last_soundcheck_readiness
                    ),
                })
            logger.info(f"Disconnected from mixer (was: {mode})")

    async def start_voice_control(self, model_size: str = "small", language: str = "ru",
                                  device_id: Optional[str] = None, channel: int = 0):
        """Start voice control listening"""
        logger.info("=" * 60)
        logger.info("START_VOICE_CONTROL CALLED")
        logger.info(f"Parameters: model={model_size}, language={language}, device_id={device_id}, channel={channel}")
        logger.info("=" * 60)
        try:

            if self.voice_control and self.voice_control.is_listening:
                logger.warning("Voice control already active")
                await self.broadcast({
                    "type": "voice_control_status",
                    "active": True,
                    "message": "Voice control already active"
                })
                return

            # Convert empty language string to None for auto-detect
            if language == "":
                language = None

            # Convert device_id to index if provided
            input_device_index = None
            if device_id:
                try:
                    input_device_index = int(device_id)
                    logger.info(f"Using audio device index: {input_device_index}")
                except ValueError:
                    logger.warning(f"Invalid device_id: {device_id}, using default device")

            # Initialize voice control (using Sherpa-ONNX with GigaAM - best for Russian!)
            logger.info("Step 1: Initializing VoiceControlSherpa (GigaAM)...")
            if VoiceControlSherpa is not None:
                try:
                    self.voice_control = VoiceControlSherpa(
                        input_device_index=input_device_index,
                        input_channel=channel
                    )
                    logger.info("Step 1: ✅ VoiceControlSherpa initialized")

                    # Auto-configure channel aliases from Wing channel names
                    if self.mixer_client and self.mixer_client.is_connected:
                        self._setup_voice_aliases_from_wing()

                except Exception as e:
                    logger.error(f"Step 1: ❌ Error initializing VoiceControlSherpa: {e}", exc_info=True)
                    self.voice_control = None
            else:
                logger.info(
                    "VoiceControlSherpa optional runtime unavailable: %s",
                    _VOICE_CONTROL_SHERPA_IMPORT_ERROR,
                )

            if self.voice_control is None:
                if VoiceControlV2 is None:
                    raise RuntimeError(
                        "Voice control is unavailable because optional audio/STT "
                        "dependencies are not installed"
                    )
                logger.info("Falling back to VoiceControlV2 (Whisper)...")
                self.voice_control = VoiceControlV2(
                    model_size=model_size,
                    language=language,
                    input_device_index=input_device_index,
                    input_channel=channel
                )

            # Get event loop for thread-safe callback
            loop = asyncio.get_running_loop()

            # Define callback for recognized commands
            def on_command_recognized(command: dict):
                """Handle recognized voice command"""
                logger.info("=" * 60)
                logger.info(f"🎤 VOICE COMMAND RECOGNIZED: {command}")
                logger.info("=" * 60)
                # Schedule task from thread using call_soon_threadsafe
                loop.call_soon_threadsafe(
                    lambda cmd=command: self._schedule_voice_command(cmd)
                )

            # Start listening
            logger.info("Step 2: Starting voice control listening...")
            try:
                # Load model first if needed (this might block, so do it in executor)
                # Check for either 'model' (Whisper) or 'recognizer' (Sherpa) attribute
                model_loaded = getattr(self.voice_control, 'model', None) or getattr(self.voice_control, 'recognizer', None)
                if not model_loaded:
                    logger.info("Model not loaded, loading in executor (this may take a moment)...")
                    import concurrent.futures
                    loop = asyncio.get_running_loop()
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        await loop.run_in_executor(executor, self.voice_control.load_model)
                    logger.info("✅ Model loaded")
                else:
                    logger.info("Model already loaded")

                # Start listening (this should be fast as it just starts threads)
                logger.info("Calling start_listening...")
                self.voice_control.start_listening(on_command_recognized)
                logger.info("Step 2: ✅ Listening started")
            except Exception as e:
                logger.error(f"Step 2: ❌ Error starting listening: {e}", exc_info=True)
                raise

            # Verify it's actually listening
            if not self.voice_control.is_listening:
                raise Exception("Voice control failed to start listening")

            logger.info("Step 3: Broadcasting success message...")
            try:
                await self.broadcast({
                    "type": "voice_control_status",
                    "active": True,
                    "message": "Voice control started successfully"
                })
                logger.info("Step 3: ✅ Broadcast sent")
            except Exception as e:
                logger.error(f"Step 3: ❌ Error broadcasting: {e}", exc_info=True)
                logger.warning("Voice control is active, but startup status broadcast failed")

            logger.info("=" * 60)
            logger.info("VOICE CONTROL STARTED SUCCESSFULLY")
            logger.info("=" * 60)

        except asyncio.CancelledError:
            if self.voice_control and not getattr(self.voice_control, "is_listening", False):
                self.voice_control = None
            raise
        except Exception as e:
            logger.error(f"Error starting voice control: {e}", exc_info=True)
            if self.voice_control and not getattr(self.voice_control, "is_listening", False):
                self.voice_control = None
            await self.broadcast({
                "type": "voice_control_status",
                "active": False,
                "error": str(e)
            })

    def _setup_voice_aliases_from_wing(self):
        """
        Auto-configure voice control channel aliases from Wing channel names.
        Maps channel names to channel numbers for voice commands.
        """
        if not self.mixer_client or not self.voice_control:
            return

        try:
            logger.info("Setting up voice aliases from Wing channel names...")

            # Get channel names from Wing (only works with WingClient)
            if not isinstance(self.mixer_client, (WingClient, EnhancedOSCClient)):
                logger.info("Channel aliases only supported for Wing mixer")
                return

            channel_names = self.mixer_client.get_all_channel_names(40)

            if not channel_names:
                logger.info("No channel names found on Wing")
                return

            # Build aliases from channel names
            aliases = {}

            # Common name variations and normalizations
            name_normalizations = {
                # Drums - kick (bd = bass drum)
                "kick": ["kick", "кик", "бочка", "bass drum", "bd"],
                "bd": ["kick", "кик", "бочка", "bass drum", "bd"],
                "бочка": ["kick", "кик", "бочка", "bass drum", "bd"],
                # Drums - snare (малый барабан)
                "snare": ["snare", "снэйр", "снейр", "малый", "рабочий", "sn", "малый барабан", "малыйбарабан"],
                "малый": ["snare", "снэйр", "снейр", "малый", "рабочий", "sn", "малый барабан", "малыйбарабан"],
                "рабочий": ["snare", "снэйр", "снейр", "малый", "рабочий", "sn", "малый барабан", "малыйбарабан"],
                "малый барабан": ["snare", "снэйр", "снейр", "малый", "рабочий", "sn", "малый барабан", "малыйбарабан"],
                # Drums - hi-hat
                "hihat": ["hihat", "hi-hat", "hh", "хайхэт", "хэт"],
                "хайхэт": ["hihat", "хайхэт", "хэт"],
                "hh": ["hihat", "хайхэт", "хэт", "hh"],
                # Drums - toms
                "tom": ["tom", "том"],
                "том": ["tom", "том"],
                "floor": ["floor", "флор", "флортом"],
                "флортом": ["floor", "флор", "флортом"],
                # Drums - overhead
                "overhead": ["overhead", "oh", "овер", "оверхэд"],
                "oh": ["overhead", "oh", "овер", "оверхэд"],
                "оверхэд": ["overhead", "oh", "овер", "оверхэд"],
                # Bass
                "bass": ["bass", "бас", "басгитара"],
                "бас": ["bass", "бас", "басгитара"],
                # Guitar
                "guitar": ["guitar", "gtr", "гитара", "электрогитара"],
                "гитара": ["guitar", "gtr", "гитара", "электрогитара"],
                "gtr": ["guitar", "gtr", "гитара"],
                # Keys
                "keys": ["keys", "клавиши", "клавишные", "kbd"],
                "клавиши": ["keys", "клавиши", "клавишные"],
                "piano": ["piano", "пиано", "рояль"],
                "synth": ["synth", "синт", "синтезатор"],
                # Accordion
                "accordion": ["accordion", "аккордеон", "баян", "гармошка"],
                "аккордеон": ["accordion", "аккордеон", "баян", "гармошка"],
                "баян": ["accordion", "аккордеон", "баян", "гармошка"],
                # Vocals
                "vocal": ["vocal", "vox", "вокал", "голос"],
                "вокал": ["vocal", "vox", "вокал", "голос"],
                "vox": ["vocal", "vox", "вокал"],
                # Playback
                "playback": ["playback", "плейбэк", "pb", "минусовка"],
                "плейбэк": ["playback", "плейбэк", "минусовка"],
            }

            for ch_num, ch_name in channel_names.items():
                if not ch_name:
                    continue

                # Normalize the name
                name_lower = ch_name.lower().strip()

                # Add direct name as alias
                aliases[name_lower] = ch_num

                # Check if name matches any known patterns
                for key, variations in name_normalizations.items():
                    if key in name_lower or name_lower in variations:
                        # Add all variations as aliases for this channel
                        for variation in variations:
                            if variation not in aliases:
                                aliases[variation] = ch_num
                        break

            if aliases:
                # Update voice control aliases
                if hasattr(self.voice_control, 'set_channel_aliases'):
                    self.voice_control.set_channel_aliases(aliases)
                    logger.info(f"Configured {len(aliases)} voice aliases from Wing channel names")

                    # Log the mappings
                    for alias, ch in sorted(aliases.items(), key=lambda x: x[1]):
                        logger.info(f"  '{alias}' -> channel {ch}")
            else:
                logger.info("No aliases created from Wing channel names")

        except Exception as e:
            logger.error(f"Error setting up voice aliases: {e}")

    async def rescan_channel_names(self):
        """
        Force rescan of all channel names from Wing mixer and update voice aliases.
        Useful when channel names have changed on the mixer.
        """
        if not self.mixer_client or not isinstance(self.mixer_client, (WingClient, EnhancedOSCClient)):
            logger.warning("Rescan only available for Wing mixer")
            target = getattr(self, "auto_soundcheck_websocket", None) or next(iter(self.connected_clients), None)
            if target is not None:
                await self.send_to_client(target, {
                    "type": "voice_control_status",
                    "active": bool(self.voice_control and getattr(self.voice_control, "is_listening", False)),
                    "error": "Rescan only available for Wing mixer"
                })
            return

        logger.info("Rescanning channel names from Wing...")
        try:
            for ch in range(1, 41):
                self.mixer_client.send(f"/ch/{ch}/name")
                await asyncio.sleep(0.01)

            await asyncio.sleep(0.5)
            self._setup_voice_aliases_from_wing()
            logger.info("Channel names rescanned successfully")

            target = getattr(self, "auto_soundcheck_websocket", None) or next(iter(self.connected_clients), None)
            if target is not None:
                await self.send_to_client(target, {
                    "type": "voice_control_status",
                    "active": bool(self.voice_control and getattr(self.voice_control, "is_listening", False)),
                    "message": "Channel names rescanned successfully"
                })
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Error rescanning channel names: {e}", exc_info=True)
            target = getattr(self, "auto_soundcheck_websocket", None) or next(iter(self.connected_clients), None)
            if target is not None:
                await self.send_to_client(target, {
                    "type": "voice_control_status",
                    "active": bool(self.voice_control and getattr(self.voice_control, "is_listening", False)),
                    "error": str(e)
                })

    async def stop_voice_control(self):
        """Stop voice control listening"""
        try:
            if not self.voice_control:
                await self.send_to_client(getattr(self, "auto_soundcheck_websocket", None) or next(iter(self.connected_clients), None), {
                    "type": "voice_control_status",
                    "active": False,
                    "message": "Voice control is not running"
                })
                return

            self.voice_control.stop_listening()
            await self.broadcast({
                "type": "voice_control_status",
                "active": False,
                "message": "Voice control stopped"
            })
            logger.info("Voice control stopped")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Error stopping voice control: {e}", exc_info=True)
            target = getattr(self, "auto_soundcheck_websocket", None) or next(iter(self.connected_clients), None)
            if target is not None:
                await self.send_to_client(target, {
                    "type": "voice_control_status",
                    "active": False,
                    "error": str(e)
                })

    # ========== Real-time Peak Correction Methods ==========

    def get_gain_staging_status(self) -> dict:
        """Get current gain staging status."""
        # Если используется SafeGainCalibrator, возвращаем его статус
        if self.safe_gain_calibrator:
            safe_status = self.safe_gain_calibrator.get_status()
            return {
                "active": safe_status.get('state') != 'idle',
                "realtime_enabled": safe_status.get('state') == 'learning',
                "safe_gain_mode": True,
                "state": safe_status.get('state'),
                "learning_progress": safe_status.get('learning_progress', 0.0),
                "channels_count": safe_status.get('channels_count', 0),
                "suggestions_ready": safe_status.get('suggestions_ready', False),
                "target_lufs": safe_status.get('target_lufs'),
                "max_peak_limit": safe_status.get('max_peak_limit'),
                "live_mode": self.live_mode,
                "automation_frozen": False
            }

        # Старый метод (для обратной совместимости)
        if not self.gain_staging:
            return {"active": False, "live_mode": self.live_mode, "automation_frozen": False, "safe_gain_mode": False}
        st = self.gain_staging.get_status()
        st["live_mode"] = getattr(self.gain_staging, "live_mode", False)
        st["automation_frozen"] = getattr(self.gain_staging, "automation_frozen", False)
        st["safe_gain_mode"] = False
        return st

    def get_gain_staging_recommendations(self) -> dict:
        """Return SafeGain recommendations without granting apply authority."""

        calibrator = getattr(self, "safe_gain_calibrator", None)
        recommendations: Dict[int, Dict[str, Any]] = {}
        reason = "safe_gain_calibrator_not_active"
        if calibrator is not None:
            getter = getattr(calibrator, "get_recommendations", None)
            if callable(getter):
                try:
                    candidate = getter()
                    if isinstance(candidate, dict):
                        recommendations = candidate
                        reason = (
                            "recommendations_available"
                            if recommendations
                            else "no_recommendations_available"
                        )
                    else:
                        reason = "invalid_recommendation_payload"
                except Exception:
                    logger.exception("Could not read SafeGain recommendations")
                    reason = "recommendations_unavailable"
            else:
                reason = "recommendation_provider_unavailable"
        return {
            "active": calibrator is not None,
            "source": "safe_gain_calibrator",
            "status": "available" if recommendations else "unavailable",
            "reason": reason,
            "live_apply_enabled": False,
            "live_authority_granted": False,
            "dry_run": True,
            "recommendation_count": len(recommendations),
            "recommendations": recommendations,
        }

    def get_live_input_trim_recommendations(self) -> dict:
        """Return current soundcheck AutoGain proposals as a read-only view."""

        controller = getattr(self, "live_input_trim_controller", None)
        recommendations: Dict[int, Dict[str, Any]] = {}
        reason = "live_input_trim_controller_not_active"
        if controller is not None:
            getter = getattr(controller, "get_recommendations", None)
            if callable(getter):
                try:
                    candidate = getter()
                    if isinstance(candidate, dict):
                        recommendations = candidate
                        reason = (
                            "recommendations_available"
                            if recommendations
                            else "no_recommendations_available"
                        )
                    else:
                        reason = "invalid_recommendation_payload"
                except Exception:
                    logger.exception("Could not read live input-trim recommendations")
                    reason = "recommendations_unavailable"
            else:
                reason = "recommendation_provider_unavailable"
        return {
            "active": controller is not None,
            "source": "live_input_trim_controller",
            "status": "available" if recommendations else "unavailable",
            "reason": reason,
            "live_apply_enabled": False,
            "live_authority_granted": False,
            "dry_run": True,
            "recommendation_count": len(recommendations),
            "recommendations": recommendations,
        }

    def get_arrangement_automation_status(self) -> dict:
        return {
            "status": "blocked",
            "active": False,
            "analysis_only_mode": True,
            "live_apply_enabled": False,
            "live_authority_granted": False,
            "reason": "arrangement_automation_not_promoted",
            "message": (
                "Arrangement automation is an offline/prototype method and is "
                "not active in the product WebSocket runtime."
            ),
        }

    async def _send_arrangement_automation_blocked(self, websocket, operation: str) -> None:
        await self.send_to_client(
            websocket,
            {
                "type": "arrangement_automation_status",
                "status_type": "blocked",
                "operation": operation,
                **self.get_arrangement_automation_status(),
            },
        )

    async def start_arrangement_automation(
        self,
        websocket,
        *,
        channels=None,
        channel_mapping=None,
        channel_names=None,
        settings=None,
    ) -> None:
        await self._send_arrangement_automation_blocked(
            websocket, "start_arrangement_automation"
        )

    async def stop_arrangement_automation(self, websocket) -> None:
        await self._send_arrangement_automation_blocked(
            websocket, "stop_arrangement_automation"
        )

    async def run_arrangement_automation_tick(self, websocket, data=None) -> None:
        await self._send_arrangement_automation_blocked(
            websocket, "run_arrangement_automation_tick"
        )

    async def set_arrangement_automation_live_apply(
        self,
        websocket,
        *,
        enabled: bool = False,
        analysis_only_mode: Any = None,
        confirm_live_apply: Any = None,
    ) -> None:
        await self._send_arrangement_automation_blocked(
            websocket, "set_arrangement_automation_live_apply"
        )

    async def update_safe_gain_settings(self, settings: dict):
        """Обновить настройки Safe Gain Calibration."""
        if settings:
            automation_config = self.config.setdefault('automation', {})
            safe_config = automation_config.setdefault('safe_gain_calibration', {})
            for key, value in settings.items():
                safe_config[key] = value

        if not self.safe_gain_calibrator:
            logger.info("Safe gain settings stored for the next calibration run")
            await self.broadcast({
                "type": "safe_gain_settings_updated",
                "settings": settings
            })
            return

        # Обновляем настройки в калибраторе
        self.safe_gain_calibrator.update_settings(settings)

        # Обновляем конфиг
        if 'learning_duration_sec' in settings:
            if 'automation' not in self.config:
                self.config['automation'] = {}
            if 'safe_gain_calibration' not in self.config['automation']:
                self.config['automation']['safe_gain_calibration'] = {}
            self.config['automation']['safe_gain_calibration']['learning_duration_sec'] = settings['learning_duration_sec']
            logger.info(f"Updated learning_duration_sec to {settings['learning_duration_sec']} seconds")

        # Отправляем подтверждение клиенту
        await self.broadcast({
            "type": "safe_gain_settings_updated",
            "settings": settings
        })

    async def start_realtime_correction(self, device_id: str = None, channels: list = None,
                                        channel_settings: dict = None, channel_mapping: dict = None,
                                        mode: str = "lufs", learning_duration_sec: float = None,
                                        safe_gain_settings: dict = None):
        """Start Safe Gain Staging calibration (новый метод: анализ → одноразовое применение)."""
        logger.info("Starting Safe Gain Staging calibration (new method)...")
        if self._is_shutting_down or self._manual_write_emergency_stopped:
            await self.broadcast({
                "type": "gain_staging_status",
                "active": False,
                "realtime_enabled": False,
                "status": "blocked",
                "reason": "emergency_stop_active",
            })
            return
        request_payload = {
            "device_id": device_id,
            "channels": channels or [],
            "mode": mode,
            "safe_gain_settings": safe_gain_settings or {},
        }
        extra_payload = {"active": False, "realtime_enabled": False}
        if await self._block_quarantined_wing_write_surface(
            None,
            surface="start_realtime_correction",
            result_type="gain_staging_status",
            request=request_payload,
            extra=extra_payload,
        ):
            payload = {
                "type": "gain_staging_status",
                "active": False,
                "realtime_enabled": False,
                "status": "blocked",
                "reason": "deployment_write_surface_quarantined",
                "surface": "start_realtime_correction",
            }
            operator_correction = self._quarantined_write_operator_correction(
                surface="start_realtime_correction",
                result_type="gain_staging_status",
                request=request_payload,
                extra=extra_payload,
            )
            if operator_correction:
                payload["operator_correction"] = operator_correction
            await self.broadcast(payload)
            return

        # Check mixer connection
        if not self.mixer_client or not self.mixer_client.is_connected:
            await self.broadcast({
                "type": "gain_staging_status",
                "active": False,
                "realtime_enabled": False,
                "error": "Mixer not connected"
            })
            return

        # Check if we have required parameters
        if not device_id or not channels or not channel_settings:
            await self.broadcast({
                "type": "gain_staging_status",
                "active": False,
                "realtime_enabled": False,
                "error": "Missing required parameters: device_id, channels, or channel_settings"
            })
            logger.warning("Missing required parameters for Safe Gain calibration")
            return

        # Останавливаем старый калибратор, если он активен
        if self.safe_gain_calibrator is not None or self._safe_gain_audio_thread is not None:
            logger.info("Stopping existing SafeGainCalibrator before starting new one...")
            await self._cancellation_safe_thread_call(self._stop_safe_gain_runtime)

        self._controllers_cleanup_complete = False

        try:
            if safe_gain_settings:
                if 'automation' not in self.config:
                    self.config['automation'] = {}
                if 'safe_gain_calibration' not in self.config['automation']:
                    self.config['automation']['safe_gain_calibration'] = {}
                self.config['automation']['safe_gain_calibration'].update(dict(safe_gain_settings))
                logger.info(f"Applying SafeGain settings for this measurement: {safe_gain_settings}")

            # Обновляем learning_duration_sec в конфиге, если передан
            if learning_duration_sec is not None:
                if 'automation' not in self.config:
                    self.config['automation'] = {}
                if 'safe_gain_calibration' not in self.config['automation']:
                    self.config['automation']['safe_gain_calibration'] = {}
                self.config['automation']['safe_gain_calibration']['learning_duration_sec'] = float(learning_duration_sec)
                logger.info(f"Applying learning_duration_sec={learning_duration_sec} seconds for this measurement")

            # Initialize SafeGainCalibrator
            import pyaudio
            pa = pyaudio.PyAudio()
            device_info = pa.get_device_info_by_index(int(device_id))
            sample_rate = int(device_info.get('defaultSampleRate', 48000))
            pa.terminate()

            logger.info("Initializing SafeGainCalibrator...")

            # Логируем значение из конфига перед созданием калибратора
            cal_config = self.config.get('automation', {}).get('safe_gain_calibration', {})
            logger.info(f"Config learning_duration_sec before calibrator init: {cal_config.get('learning_duration_sec', 'NOT SET')}")
            logger.info(f"Requested learning_duration_sec: {learning_duration_sec}")

            self.safe_gain_calibrator = SafeGainCalibrator(
                mixer_client=self.mixer_client,
                sample_rate=sample_rate,
                config=self.config
            )

            # Логируем значение после создания калибратора
            logger.info(f"Calibrator learning_duration after init: {self.safe_gain_calibrator.learning_duration} seconds")

            # Применяем learning_duration_sec, если передан (обновляем после создания)
            if learning_duration_sec is not None:
                old_value = self.safe_gain_calibrator.learning_duration
                self.safe_gain_calibrator.learning_duration = float(learning_duration_sec)
                logger.info(f"Updated learning_duration: {old_value} -> {self.safe_gain_calibrator.learning_duration} seconds")

            # Setup callbacks
            loop = asyncio.get_running_loop()

            def on_progress_update(status: dict):
                message = {
                    "type": "gain_staging_status",
                    "status_type": "safe_gain_progress",
                    **status
                }
                loop.call_soon_threadsafe(
                    lambda msg=message: asyncio.create_task(self.broadcast(msg))
                )

            def on_suggestions_ready(suggestions: dict):
                if self._is_shutting_down or self._manual_write_emergency_stopped:
                    logger.warning("SafeGain completion ignored after live-write gate closed")
                    return
                proposal_import = self.import_safe_gain_suggestions_to_operator_queue(
                    suggestions,
                    source=SAFE_GAIN_SOURCE,
                )
                message = {
                    "type": "gain_staging_status",
                    "status_type": "safe_gain_ready",
                    "suggestions": suggestions,
                    "operator_proposal_import": proposal_import,
                    "message": "Analysis complete. Evaluating SafeGain trim actions..."
                }
                loop.call_soon_threadsafe(
                    lambda msg=message: asyncio.create_task(self.broadcast(msg))
                )
                loop.call_soon_threadsafe(
                    lambda summary=proposal_import: asyncio.create_task(self.broadcast(summary))
                )
                operator_mode = self.get_operator_mode_status()
                resolved_policy = self._active_soundcheck_live_policy or {}
                auto_apply_allowed = bool(resolved_policy.get("live_apply_allowed")) or bool(
                    operator_mode["capabilities"].get("can_autonomous_apply")
                )
                if not auto_apply_allowed:
                    loop.call_soon_threadsafe(
                        lambda summary=proposal_import, mode=operator_mode: asyncio.create_task(self.broadcast({
                            "type": "gain_staging_status",
                            "status_type": "safe_gain_proposals_queued",
                            "message": "SafeGain analysis queued proposals. Current operator mode blocks automatic apply.",
                            "operator_mode": mode,
                            "operator_proposal_import": summary,
                        }))
                    )
                    return

                # Automatic apply remains available only for explicit live policies or unlocked autonomous mode.
                if (
                    self.safe_gain_calibrator
                    and not self._is_shutting_down
                    and not self._manual_write_emergency_stopped
                ):
                    applied = self.safe_gain_calibrator.apply_corrections()
                    if applied:
                        logger.info("Safe Gain corrections applied automatically")
                        loop.call_soon_threadsafe(
                            lambda: asyncio.create_task(self.broadcast({
                                "type": "gain_staging_status",
                                "status_type": "safe_gain_applied",
                                "message": "Corrections applied successfully"
                            }))
                        )

            self.safe_gain_calibrator.on_progress_update = on_progress_update
            self.safe_gain_calibrator.on_suggestions_ready = on_suggestions_ready

            # Add channels to SafeGainCalibrator
            for audio_ch_str in channels:
                audio_ch = int(audio_ch_str) if isinstance(audio_ch_str, str) else audio_ch_str
                mixer_ch = channel_mapping.get(audio_ch, audio_ch) if channel_mapping else audio_ch
                self.safe_gain_calibrator.add_channel(audio_channel=audio_ch, mixer_channel=mixer_ch)

            # Initialize audio analyzer (используем gain_staging только для захвата аудио)
            if not self.gain_staging:
                logger.info("Initializing audio analyzer for Safe Gain calibration...")

                self.gain_staging = LUFSGainStagingController(
                    mixer_client=self.mixer_client,
                    config=self.config,
                    bleed_service=self.bleed_service
                )
            else:
                self.gain_staging.mixer_client = self.mixer_client

            # Convert channel_settings and channel_mapping for audio analyzer
            converted_settings = {}
            instrument_types_for_bleed = {}
            for ch_str, settings in channel_settings.items():
                ch = int(ch_str) if isinstance(ch_str, str) else ch_str
                converted_settings[ch] = settings
                # Extract instrument type for bleed service
                instrument_type = settings.get('instrumentType') or settings.get('preset') or 'custom'
                instrument_types_for_bleed[ch] = 'tom' if instrument_type == 'toms' else instrument_type

            # Configure bleed service with instrument types
            if self.bleed_service and instrument_types_for_bleed:
                self.bleed_service.configure(instrument_types_for_bleed)

            converted_mapping = {}
            if channel_mapping:
                for audio_ch_str, mixer_ch in channel_mapping.items():
                    audio_ch = int(audio_ch_str) if isinstance(audio_ch_str, str) else audio_ch_str
                    converted_mapping[audio_ch] = mixer_ch
            else:
                for audio_ch in channels:
                    audio_ch_int = int(audio_ch) if isinstance(audio_ch, str) else audio_ch
                    converted_mapping[audio_ch_int] = audio_ch_int

            self.gain_staging.channel_settings = converted_settings
            self.gain_staging.channel_mapping = converted_mapping

            # Set channel_settings for SafeGainCalibrator to use preset-based metrics
            self.safe_gain_calibrator.channel_settings = converted_settings
            # Stop existing audio analyzer if running
            if self.gain_staging.is_active:
                logger.info("Stopping existing audio analyzer...")
                self.gain_staging.stop()

            # Start audio analyzer for SafeGainCalibrator
            logger.info("Starting audio analyzer for Safe Gain calibration...")

            # Поток для передачи аудио данных в SafeGainCalibrator
            self._safe_gain_audio_thread_stop = threading.Event()

            def safe_gain_audio_processor():
                """Периодически передает аудио данные из буферов в SafeGainCalibrator"""
                import time
                while not self._safe_gain_audio_thread_stop.is_set():
                    if not self.safe_gain_calibrator or self.safe_gain_calibrator.state.value != 'learning':
                        time.sleep(0.1)
                        continue

                    if not self.gain_staging or not hasattr(self.gain_staging, '_audio_buffers'):
                        time.sleep(0.1)
                        continue

                    # Process audio for each channel from audio buffers
                    for audio_ch in channels:
                        audio_ch_int = int(audio_ch) if isinstance(audio_ch, str) else audio_ch
                        if audio_ch_int in self.gain_staging._audio_buffers:
                            buffer = self.gain_staging._audio_buffers[audio_ch_int]
                            if buffer and len(buffer) > 0:
                                # Get latest audio chunks (copy to avoid race conditions)
                                chunks = list(buffer)
                                if chunks:
                                    try:
                                        samples = np.concatenate(chunks)
                                        if len(samples) > 0:
                                            self.safe_gain_calibrator.process_audio(audio_ch_int, samples)
                                    except Exception as e:
                                        logger.debug(f"Error processing audio for SafeGain: {e}")

                    time.sleep(0.05)  # Process every 50ms

            self._safe_gain_audio_thread = threading.Thread(
                target=safe_gain_audio_processor,
                daemon=True
            )

            success = self.gain_staging.start(
                device_id=device_id,
                channels=[int(ch) for ch in channels],
                channel_settings=converted_settings,
                channel_mapping=converted_mapping,
                on_status_callback=None
            )

            if not success:
                await self.broadcast({
                    "type": "gain_staging_status",
                    "active": False,
                    "realtime_enabled": False,
                    "error": "Failed to start audio analyzer"
                })
                await self._cancellation_safe_thread_call(self._stop_safe_gain_runtime)
                return

            # Start audio processing thread
            self._safe_gain_audio_thread.start()
            logger.info("Audio processing thread started for Safe Gain calibration")

            # Start Safe Gain analysis
            analysis_started = self.safe_gain_calibrator.start_analysis()

            if not analysis_started:
                await self.broadcast({
                    "type": "gain_staging_status",
                    "active": False,
                    "realtime_enabled": False,
                    "error": "Failed to start Safe Gain analysis"
                })
                await self._cancellation_safe_thread_call(self._stop_safe_gain_runtime)
                return

            await self.broadcast({
                "type": "gain_staging_status",
                "active": True,
                "realtime_enabled": True,
                "status_type": "safe_gain_started",
                "message": f"Safe Gain analysis started (learning for {self.safe_gain_calibrator.learning_duration}s)"
            })
            logger.info("Safe Gain calibration started successfully")
        except Exception as e:
            logger.error(f"Error starting real-time correction: {e}", exc_info=True)
            try:
                await self._cancellation_safe_thread_call(self._stop_safe_gain_runtime)
            except Exception as cleanup_error:
                logger.error(
                    "SafeGain failed-start cleanup incomplete: %s",
                    cleanup_error,
                    exc_info=True,
                )
            await self.broadcast({
                "type": "gain_staging_status",
                "active": self.gain_staging.is_active if self.gain_staging else False,
                "realtime_enabled": False,
                "error": f"Error: {str(e)}"
            })

    async def stop_realtime_correction(self):
        """Stop Safe Gain calibration or real-time TRIM correction."""
        logger.info("Stopping gain staging...")

        # Останавливаем SafeGainCalibrator если активен
        if self.safe_gain_calibrator is not None or self._safe_gain_audio_thread is not None:
            logger.info("Stopping Safe Gain calibration...")
            await self._cancellation_safe_thread_call(self._stop_safe_gain_runtime)

        # Останавливаем аудио анализатор
        if self.gain_staging:
            if self.gain_staging.realtime_correction_enabled:
                self.gain_staging.stop_realtime_correction()
            if self.gain_staging.is_active:
                self.gain_staging.stop()

        await self.broadcast({
            "type": "gain_staging_status",
            "active": False,
            "realtime_enabled": False,
            "message": "Gain staging stopped"
        })

        logger.info("Real-time correction stopped")

    async def scan_and_recognize_channels(self, websocket, channels: list):
        """
        Scan channel names from mixer and recognize instrument types.

        Args:
            websocket: WebSocket connection to respond to
            channels: List of channel numbers to scan (empty = all selected)
        """
        logger.info(f"Scanning channel names for recognition. Requested channels: {channels}")

        if not self.mixer_client or not self.mixer_client.is_connected:
            await self.send_to_client(websocket, {
                "type": "channel_scan_result",
                "error": "Mixer not connected",
                "results": {}
            })
            return

        try:
            # Get channel names from mixer
            if isinstance(self.mixer_client, (WingClient, EnhancedOSCClient)):
                # Per Wing Remote Protocols v3.0.5:
                # /ch/X/name - для установки имени
                # /ch/X/$name - для чтения имени (read-only, отражает связанный источник)
                logger.info(f"Querying channel names using /ch/X/$name for channels: {channels}")

                # Send queries for $name (read-only address that reflects linked source)
                for ch in channels:
                    self.mixer_client.send(f"/ch/{ch}/$name")
                    await asyncio.sleep(0.02)

                # Wait for responses
                await asyncio.sleep(0.5)

                # Collect names from state
                channel_names = {}
                for ch in channels:
                    # Try $name first (read-only, reflects linked source)
                    name = self.mixer_client.state.get(f"/ch/{ch}/$name")
                    if not name or name == '':
                        # Fallback to regular name
                        name = self.mixer_client.state.get(f"/ch/{ch}/name")

                    if name and name != '':
                        channel_names[ch] = name
                        logger.info(f"Channel {ch} name: '{name}'")
                    else:
                        channel_names[ch] = f"Ch {ch}"
                        logger.debug(f"Channel {ch}: no name found")

                logger.info(f"Retrieved {len(channel_names)} channel names")

                # Run recognition
                results = scan_and_recognize(channel_names)

                await self.send_to_client(websocket, {
                    "type": "channel_scan_result",
                    "results": results,
                    "available_presets": AVAILABLE_PRESETS
                })

                recognized_count = sum(1 for r in results.values() if r.get('recognized'))
                logger.info(f"Channel scan complete: {recognized_count}/{len(results)} recognized")

            else:
                await self.send_to_client(websocket, {
                    "type": "channel_scan_result",
                    "error": "Channel scanning only supported for Wing mixer",
                    "results": {}
                })

        except Exception as e:
            logger.error(f"Error scanning channels: {e}", exc_info=True)
            await self.send_to_client(websocket, {
                "type": "channel_scan_result",
                "error": str(e),
                "results": {}
            })

    async def scan_mixer_channel_names(self, websocket):
        """
        Scan all channel names from mixer (channels 1-40) and return them for Audio Device tab.
        Updates channel names in the available channels list.
        """
        logger.info("Scanning mixer channel names for Audio Device tab...")

        if not self.mixer_client or not self.mixer_client.is_connected:
            await self.send_to_client(websocket, {
                "type": "mixer_channel_names",
                "error": "Mixer not connected",
                "channel_names": {}
            })
            return

        try:
            if isinstance(self.mixer_client, (WingClient, EnhancedOSCClient)):
                # Scan all channels (1-40) using async approach
                logger.info("Querying channel names for all 40 channels...")

                # Send queries for $name (read-only address that reflects linked source)
                for ch in range(1, 41):
                    self.mixer_client.send(f"/ch/{ch}/$name")
                    await asyncio.sleep(0.02)

                # Wait longer for responses to arrive
                await asyncio.sleep(1.0)

                # Collect names from state
                channel_names = {}
                for ch in range(1, 41):
                    # Try $name first (read-only, reflects linked source)
                    name = self.mixer_client.state.get(f"/ch/{ch}/$name")
                    if not name or name == '':
                        # Fallback to regular name
                        name = self.mixer_client.state.get(f"/ch/{ch}/name")

                    # Handle tuple response (Wing sometimes returns tuples)
                    if isinstance(name, tuple) and len(name) > 0:
                        name = name[0]

                    if name and str(name).strip():
                        channel_names[ch] = str(name).strip()
                        logger.debug(f"Channel {ch} name: '{channel_names[ch]}'")
                    else:
                        channel_names[ch] = f"Ch {ch}"
                        logger.debug(f"Channel {ch}: no name found, using default")

                logger.info(f"Retrieved {len(channel_names)} channel names from mixer")
                logger.info(f"Sample names: {dict(list(channel_names.items())[:5])}")

                await self.send_to_client(websocket, {
                    "type": "mixer_channel_names",
                    "channel_names": channel_names
                })
            else:
                await self.send_to_client(websocket, {
                    "type": "mixer_channel_names",
                    "error": "Channel name scanning only supported for Wing mixer",
                    "channel_names": {}
                })

        except Exception as e:
            logger.error(f"Error scanning mixer channel names: {e}", exc_info=True)
            await self.send_to_client(websocket, {
                "type": "mixer_channel_names",
                "error": str(e),
                "channel_names": {}
            })

    async def reset_trim(self, websocket, channels: list):
        """
        Reset TRIM to 0dB for selected channels.

        Args:
            websocket: WebSocket connection to respond to
            channels: List of channel numbers to reset (mixer channels)
        """
        logger.info(f"Resetting TRIM to 0dB for channels: {channels}")
        if await self._block_quarantined_wing_write_surface(
            websocket,
            surface="reset_trim",
            result_type="reset_trim_result",
            request={"channels": channels},
        ):
            return

        if not self.mixer_client or not self.mixer_client.is_connected:
            await self.send_to_client(websocket, {
                "type": "reset_trim_result",
                "success": False,
                "error": "Mixer not connected",
                "channels": channels
            })
            return

        if not channels:
            await self.send_to_client(websocket, {
                "type": "reset_trim_result",
                "success": False,
                "error": "No channels specified",
                "channels": []
            })
            return

        # Use asyncio.wait_for to prevent hanging
        async def reset_trim_task():
            results = {}
            success_count = 0
            failed_channels = []

            logger.info(f"Starting TRIM reset for {len(channels)} channels: {channels}")
            logger.info(f"Mixer client type: {type(self.mixer_client).__name__}")
            logger.info(f"Mixer connected: {self.mixer_client.is_connected if hasattr(self.mixer_client, 'is_connected') else 'N/A'}")

            # Set TRIM to 0 dB for all channels
            for ch in channels:
                try:
                    # Set TRIM to 0 dB using direct OSC command for reliability
                    if isinstance(self.mixer_client, (WingClient, EnhancedOSCClient)):
                        # Use direct OSC address for TRIM
                        address = f"/ch/{ch}/in/set/trim"
                        logger.info(f"Channel {ch}: Sending TRIM reset to {address} = 0.0 dB")
                        result = self.mixer_client.send(address, 0.0)

                        if result:
                            success_count += 1
                            results[ch] = {
                                "success": True,
                                "new_trim": 0.0
                            }
                            logger.info(f"Channel {ch}: TRIM reset command sent successfully")

                            # Small delay to allow mixer to process
                            await asyncio.sleep(0.1)

                            # Query the trim value to verify and trigger state update
                            logger.debug(f"Channel {ch}: Requesting TRIM value update from mixer")
                            self.mixer_client.send(f"/ch/{ch}/in/set/trim")
                        else:
                            # Fallback to set_channel_gain method
                            logger.warning(f"Channel {ch}: Direct OSC failed, trying set_channel_gain method")
                            result = self.mixer_client.set_channel_gain(ch, 0.0)
                            if result:
                                success_count += 1
                                results[ch] = {
                                    "success": True,
                                    "new_trim": 0.0
                                }
                            else:
                                failed_channels.append(ch)
                                results[ch] = {
                                    "success": False,
                                    "error": "Failed to send TRIM reset command"
                                }
                                logger.warning(f"Channel {ch}: Failed to send TRIM reset command")
                    else:
                        # For other mixer clients, use standard method
                        result = self.mixer_client.set_channel_gain(ch, 0.0)
                        if result:
                            success_count += 1
                            results[ch] = {
                                "success": True,
                                "new_trim": 0.0
                            }
                            logger.info(f"Channel {ch}: TRIM reset command sent successfully")
                        else:
                            failed_channels.append(ch)
                            results[ch] = {
                                "success": False,
                                "error": "Failed to send TRIM reset command"
                            }
                            logger.warning(f"Channel {ch}: Failed to send TRIM reset command")

                    # Delay between commands to allow mixer to process
                    await asyncio.sleep(0.1)  # Increased delay for better reliability

                except Exception as e:
                    failed_channels.append(ch)
                    results[ch] = {
                        "success": False,
                        "error": str(e)
                    }
                    logger.error(f"Error resetting TRIM for channel {ch}: {e}", exc_info=True)

            # Wait longer for mixer to process all commands
            await asyncio.sleep(0.5)

            # Verify TRIM values if possible (for WingClient)
            if isinstance(self.mixer_client, (WingClient, EnhancedOSCClient)):
                logger.info("Verifying TRIM reset...")
                await asyncio.sleep(0.5)  # Increased delay for state updates
                verification_success = 0
                for ch in channels:
                    try:
                        # Query current TRIM value by requesting it
                        self.mixer_client.send(f"/ch/{ch}/in/set/trim")
                        await asyncio.sleep(0.1)  # Wait for response

                        # Check if TRIM is actually 0.0 (or close to it)
                        current_trim = self.mixer_client.get_channel_gain(ch)
                        if current_trim is not None:
                            if abs(current_trim) < 0.1:  # Allow small tolerance
                                verification_success += 1
                                logger.info(f"Channel {ch}: TRIM verified at {current_trim:.2f} dB (reset successful)")
                            else:
                                logger.warning(f"Channel {ch}: TRIM reset may have failed - current value: {current_trim:.2f} dB, expected: 0.0 dB")
                                # Try to reset again
                                logger.info(f"Channel {ch}: Retrying TRIM reset...")
                                self.mixer_client.set_channel_gain(ch, 0.0)
                                await asyncio.sleep(0.1)
                        else:
                            logger.warning(f"Channel {ch}: Could not verify TRIM value (no response from mixer)")
                    except Exception as e:
                        logger.debug(f"Could not verify TRIM for channel {ch}: {e}")

                logger.info(f"TRIM reset verification: {verification_success}/{len(channels)} channels verified")

            # Send response immediately
            await self.send_to_client(websocket, {
                "type": "reset_trim_result",
                "success": success_count > 0,
                "success_count": success_count,
                "total_count": len(channels),
                "failed_channels": failed_channels,
                "results": results,
                "message": f"Reset TRIM to 0dB for {success_count}/{len(channels)} channels"
            })

            logger.info(f"TRIM reset completed: {success_count}/{len(channels)} channels successful")
            if failed_channels:
                logger.warning(f"Failed to reset TRIM for channels: {failed_channels}")

        try:
            # Set 5 second timeout to prevent hanging
            await asyncio.wait_for(reset_trim_task(), timeout=5.0)
        except asyncio.CancelledError:
            logger.info("Reset TRIM cancelled")
            raise
        except asyncio.TimeoutError:
            logger.error("Timeout while resetting TRIM")
            await self.send_to_client(websocket, {
                "type": "reset_trim_result",
                "success": False,
                "error": "Operation timed out",
                "channels": channels
            })
        except Exception as e:
            logger.error(f"Error resetting TRIM: {e}", exc_info=True)
            await self.send_to_client(websocket, {
                "type": "reset_trim_result",
                "success": False,
                "error": str(e),
                "channels": channels
            })

    async def bypass_mixer(self, websocket):
        """
        Bypass mixer: Disable all modules and set faders to 0dB for all 40 channels.

        Args:
            websocket: WebSocket connection to respond to
        """
        logger.info("Starting bypass operation: disabling all modules and setting faders to 0dB")
        if await self._block_quarantined_wing_write_surface(
            websocket,
            surface="bypass_mixer",
            result_type="bypass_result",
        ):
            return

        if not self.mixer_client or not self.mixer_client.is_connected:
            await self.send_to_client(websocket, {
                "type": "bypass_result",
                "success": False,
                "error": "Mixer not connected"
            })
            return

        if not isinstance(self.mixer_client, (WingClient, EnhancedOSCClient)):
            await self.send_to_client(websocket, {
                "type": "bypass_result",
                "success": False,
                "error": "Bypass operation only supported for Wing mixer"
            })
            return

        async def bypass_task():
            success_count = 0
            failed_channels = []

            try:
                # Process all 40 channels
                for ch in range(1, 41):
                    try:
                        # 1. Set fader to 0dB
                        self.mixer_client.set_channel_fader(ch, 0.0)
                        await asyncio.sleep(0.01)

                        # 2. Disable EQ
                        self.mixer_client.set_eq_on(ch, 0)
                        await asyncio.sleep(0.01)

                        # 3. Disable PreEQ
                        self.mixer_client.send(f"/ch/{ch}/peq/on", 0)
                        await asyncio.sleep(0.01)

                        # 4. Disable Compressor/Dynamics
                        self.mixer_client.set_compressor_on(ch, 0)
                        await asyncio.sleep(0.01)

                        # 5. Disable Gate
                        self.mixer_client.set_gate_on(ch, 0)
                        await asyncio.sleep(0.01)

                        # 6. Disable Filters
                        self.mixer_client.set_low_cut(ch, enabled=0)
                        await asyncio.sleep(0.01)
                        self.mixer_client.set_high_cut(ch, enabled=0)
                        await asyncio.sleep(0.01)

                        # 7. Disable Inserts
                        self.mixer_client.send(f"/ch/{ch}/preins/on", 0)
                        await asyncio.sleep(0.01)
                        self.mixer_client.send(f"/ch/{ch}/postins/on", 0)
                        await asyncio.sleep(0.01)

                        success_count += 1

                        if ch % 10 == 0:
                            logger.info(f"Bypass progress: {ch}/40 channels processed")

                    except Exception as e:
                        failed_channels.append(ch)
                        logger.error(f"Error bypassing channel {ch}: {e}", exc_info=True)

                # Send response
                await self.send_to_client(websocket, {
                    "type": "bypass_result",
                    "success": success_count > 0,
                    "success_count": success_count,
                    "total_count": 40,
                    "failed_channels": failed_channels,
                    "message": f"Bypass completed: {success_count}/40 channels processed"
                })

                logger.info(f"Bypass operation completed: {success_count}/40 channels successful")
                if failed_channels:
                    logger.warning(f"Failed to bypass channels: {failed_channels}")

            except Exception as e:
                logger.error(f"Error in bypass operation: {e}", exc_info=True)
                await self.send_to_client(websocket, {
                    "type": "bypass_result",
                    "success": False,
                    "error": str(e)
                })

        # Run bypass task with timeout
        try:
            await asyncio.wait_for(bypass_task(), timeout=30.0)
        except asyncio.CancelledError:
            logger.info("Bypass operation cancelled")
            raise
        except asyncio.TimeoutError:
            logger.error("Bypass operation timed out")
            await self.send_to_client(websocket, {
                "type": "bypass_result",
                "success": False,
                "error": "Operation timed out"
            })
        except Exception as e:
            logger.error(f"Error executing bypass: {e}", exc_info=True)
            await self.send_to_client(websocket, {
                "type": "bypass_result",
                "success": False,
                "error": str(e)
            })

    # ========== Auto-EQ Methods ==========

    def get_auto_eq_status(self) -> dict:
        """Get current Auto-EQ status."""
        if not self.auto_eq_controller:
            return {"active": False}
        return self.auto_eq_controller.get_status()

    async def start_auto_eq(self, websocket, device_id: str = None, channel: int = None,
                            profile: str = "custom", auto_apply: bool = False,
                            monitored_channels: List[int] = None):
        """Start Auto-EQ analysis for a channel."""
        monitored_channels = monitored_channels or []
        auto_apply_forced_off = False
        if self._is_wing_deployment_write_boundary_active() and auto_apply:
            auto_apply = False
            auto_apply_forced_off = True
            logger.warning(
                "Forced Auto-EQ auto_apply off for WING deployment boundary",
                extra={"event": "wing_deployment_boundary_auto_apply_forced_off"},
            )
        logger.info(f"Starting Auto-EQ: device={device_id}, channel={channel}, profile={profile}, monitored_channels={monitored_channels}")

        if not self.mixer_client or not self.mixer_client.is_connected:
            await self.send_to_client(websocket, {
                "type": "auto_eq_status",
                "active": False,
                "error": "Mixer not connected"
            })
            return

        if not device_id or not channel:
            await self.send_to_client(websocket, {
                "type": "auto_eq_status",
                "active": False,
                "error": "Missing required parameters: device_id or channel"
            })
            return

        try:
            # Initialize controller if needed
            if not self.auto_eq_controller:
                self.auto_eq_controller = AutoEQController(
                    mixer_client=self.mixer_client,
                    bleed_service=self.bleed_service
                )
            else:
                # Update mixer client reference
                self.auto_eq_controller.mixer_client = self.mixer_client

            # Get event loop for thread-safe callbacks
            loop = asyncio.get_running_loop()

            def on_spectrum_update(spectrum_data: dict):
                """Handle spectrum data updates."""
                loop.call_soon_threadsafe(
                    lambda: asyncio.create_task(self.broadcast({
                        "type": "auto_eq_spectrum",
                        **spectrum_data
                    }))
                )

            def on_corrections_calculated(corrections: list):
                """Handle EQ corrections updates."""
                loop.call_soon_threadsafe(
                    lambda: asyncio.create_task(self.broadcast({
                        "type": "auto_eq_corrections",
                        "corrections": corrections
                    }))
                )

            def on_status_update(status: dict):
                """Handle status updates."""
                loop.call_soon_threadsafe(
                    lambda: asyncio.create_task(self.broadcast({
                        "type": "auto_eq_status",
                        **status
                    }))
                )

            # Start analysis
            success = self.auto_eq_controller.start(
                device_id=int(device_id),
                channel=int(channel),
                profile_name=profile,
                auto_apply=auto_apply,
                monitored_channels=monitored_channels,
                on_spectrum_callback=on_spectrum_update,
                on_corrections_callback=on_corrections_calculated,
                on_status_callback=on_status_update
            )

            if success:
                await self.send_to_client(websocket, {
                    "type": "auto_eq_status",
                    "active": True,
                    "channel": channel,
                    "profile": profile,
                    "auto_apply": auto_apply,
                    "auto_apply_forced_off": auto_apply_forced_off,
                    "message": f"Auto-EQ started for channel {channel}"
                })
                logger.info(f"Auto-EQ started successfully for channel {channel}")
            else:
                await self.send_to_client(websocket, {
                    "type": "auto_eq_status",
                    "active": False,
                    "error": "Failed to start Auto-EQ analysis"
                })

        except Exception as e:
            logger.error(f"Error starting Auto-EQ: {e}", exc_info=True)
            await self.send_to_client(websocket, {
                "type": "auto_eq_status",
                "active": False,
                "error": str(e)
            })

    async def stop_auto_eq(self, websocket):
        """Stop Auto-EQ analysis."""
        logger.info("Stopping Auto-EQ...")

        if self.auto_eq_controller:
            self.auto_eq_controller.stop()

        await self.send_to_client(websocket, {
            "type": "auto_eq_status",
            "active": False,
            "message": "Auto-EQ stopped"
        })

        logger.info("Auto-EQ stopped")

    async def set_eq_profile(self, websocket, profile: str):
        """Change the EQ profile."""
        if not self.auto_eq_controller:
            await self.send_to_client(websocket, {
                "type": "auto_eq_status",
                "error": "Auto-EQ not initialized"
            })
            return

        self.auto_eq_controller.set_profile(profile)

        await self.send_to_client(websocket, {
            "type": "auto_eq_status",
            "profile": profile,
            "message": f"Profile changed to {profile}"
        })

        logger.info(f"EQ profile changed to: {profile}")

    async def apply_eq_correction(self, websocket):
        """Apply calculated EQ corrections to the mixer."""
        if await self._block_quarantined_wing_write_surface(
            websocket,
            surface="apply_eq_correction",
            result_type="auto_eq_apply_result",
        ):
            return

        if not self.auto_eq_controller:
            await self.send_to_client(websocket, {
                "type": "auto_eq_apply_result",
                "success": False,
                "error": "Auto-EQ not initialized"
            })
            return

        try:
            success = self.auto_eq_controller.apply_to_mixer()

            await self.send_to_client(websocket, {
                "type": "auto_eq_apply_result",
                "success": success,
                "message": "EQ corrections applied to mixer" if success else "Failed to apply EQ"
            })

        except Exception as e:
            logger.error(f"Error applying EQ: {e}")
            await self.send_to_client(websocket, {
                "type": "auto_eq_apply_result",
                "success": False,
                "error": str(e)
            })

    async def reset_eq(self, websocket, data: dict = None):
        """Reset EQ to flat."""
        if data is None:
            data = {}
        if await self._block_quarantined_wing_write_surface(
            websocket,
            surface="reset_eq",
            result_type="auto_eq_reset_result",
            request=data,
        ):
            return

        # Try to get channel from message, auto_eq_controller, or use default
        channel = None
        if data.get("channel"):
            channel = data.get("channel")
            logger.info(f"Reset EQ: Using channel from message: {channel}")
        elif self.auto_eq_controller and self.auto_eq_controller.current_channel:
            channel = self.auto_eq_controller.current_channel
            logger.info(f"Reset EQ: Using channel from auto_eq_controller: {channel}")
        else:
            channel = 1  # Default fallback
            logger.info(f"Reset EQ: Using default channel: {channel}")

        # If we have auto_eq_controller with current_channel, use it
        if self.auto_eq_controller and self.auto_eq_controller.current_channel:
            try:
                success = self.auto_eq_controller.reset_eq()
                logger.info(f"Reset EQ via auto_eq_controller: success={success}")
                await self.send_to_client(websocket, {
                    "type": "auto_eq_reset_result",
                    "success": success,
                    "message": "EQ reset to flat" if success else "Failed to reset EQ"
                })
                return
            except Exception as e:
                logger.error(f"Error resetting EQ via auto_eq_controller: {e}", exc_info=True)
                # Fall through to direct reset

        # Direct reset via mixer_client
        if self.mixer_client and hasattr(self.mixer_client, 'send'):
            try:
                logger.info(f"Reset EQ: Resetting channel {channel} directly via mixer_client")

                # Ensure EQ is enabled (needed for changes to take effect)
                if hasattr(self.mixer_client, 'set_eq_on'):
                    logger.info(f"Reset EQ: Enabling EQ for channel {channel}")
                    self.mixer_client.set_eq_on(channel, 1)
                    await asyncio.sleep(0.05)

                # Reset EQ bands to flat using direct OSC addresses (only reset gain, don't change Q or frequency)
                logger.info(f"Reset EQ: Sending gain=0 commands for channel {channel}")

                # Low shelf gain
                self.mixer_client.send(f"/ch/{channel}/eq/lg", 0.0)
                logger.debug(f"Reset EQ: Sent /ch/{channel}/eq/lg = 0.0")
                await asyncio.sleep(0.05)

                # Band 1-4 gains
                for band in [1, 2, 3, 4]:
                    self.mixer_client.send(f"/ch/{channel}/eq/{band}g", 0.0)
                    logger.debug(f"Reset EQ: Sent /ch/{channel}/eq/{band}g = 0.0")
                    await asyncio.sleep(0.05)

                # High shelf gain
                self.mixer_client.send(f"/ch/{channel}/eq/hg", 0.0)
                logger.debug(f"Reset EQ: Sent /ch/{channel}/eq/hg = 0.0")

                logger.info(f"Reset EQ: Successfully reset channel {channel}")
                await self.send_to_client(websocket, {
                    "type": "auto_eq_reset_result",
                    "success": True,
                    "message": f"EQ reset to flat for channel {channel}"
                })
                return
            except Exception as e:
                logger.error(f"Error resetting EQ directly: {e}", exc_info=True)
                await self.send_to_client(websocket, {
                    "type": "auto_eq_reset_result",
                    "success": False,
                    "error": str(e)
                })
                return

        # No way to reset
        logger.warning("Reset EQ: No mixer client available")
        await self.send_to_client(websocket, {
            "type": "auto_eq_reset_result",
            "success": False,
            "error": "Mixer not connected"
        })

    async def reset_all_eq(self, websocket, data: dict = None):
        """Reset EQ to flat for multiple channels."""
        if data is None:
            data = {}
        if await self._block_quarantined_wing_write_surface(
            websocket,
            surface="reset_all_eq",
            result_type="reset_all_eq_result",
            request=data,
        ):
            return

        channels = data.get("channels", [])
        if not channels:
            await self.send_to_client(websocket, {
                "type": "reset_all_eq_result",
                "success": False,
                "error": "No channels specified"
            })
            return

        if not self.mixer_client or not hasattr(self.mixer_client, 'send'):
            await self.send_to_client(websocket, {
                "type": "reset_all_eq_result",
                "success": False,
                "error": "Mixer not connected"
            })
            return

        logger.info(f"Reset All EQ: Resetting {len(channels)} channels")
        reset_count = 0
        errors = []

        try:
            for channel in channels:
                try:
                    # Ensure EQ is enabled
                    if hasattr(self.mixer_client, 'set_eq_on'):
                        self.mixer_client.set_eq_on(channel, 1)
                        await asyncio.sleep(0.05)

                    # Reset all EQ bands to 0 dB
                    self.mixer_client.send(f"/ch/{channel}/eq/lg", 0.0)
                    await asyncio.sleep(0.05)

                    for band in [1, 2, 3, 4]:
                        self.mixer_client.send(f"/ch/{channel}/eq/{band}g", 0.0)
                        await asyncio.sleep(0.05)

                    self.mixer_client.send(f"/ch/{channel}/eq/hg", 0.0)
                    reset_count += 1
                    logger.debug(f"Reset All EQ: Successfully reset channel {channel}")

                except Exception as e:
                    error_msg = f"Channel {channel}: {str(e)}"
                    errors.append(error_msg)
                    logger.error(f"Reset All EQ: Error resetting channel {channel}: {e}")

            success = reset_count > 0
            message = f"Reset EQ for {reset_count}/{len(channels)} channels"
            if errors:
                message += f". Errors: {len(errors)}"

            await self.send_to_client(websocket, {
                "type": "reset_all_eq_result",
                "success": success,
                "reset_count": reset_count,
                "total_count": len(channels),
                "message": message,
                "errors": errors if errors else None
            })

            logger.info(f"Reset All EQ: Completed. Reset {reset_count}/{len(channels)} channels")

        except Exception as e:
            logger.error(f"Reset All EQ: Unexpected error: {e}", exc_info=True)
            await self.send_to_client(websocket, {
                "type": "reset_all_eq_result",
                "success": False,
                "error": str(e)
            })

    async def start_multi_channel_auto_eq(self, websocket, device_id: str = None, channels_config: List[Dict] = None, mode: str = 'soundcheck'):
        """Start multi-channel Auto-EQ analysis.

        Args:
            mode: 'soundcheck' (analyze once, then stop) or 'live' (continuous)
        """
        channels_config = channels_config or []
        logger.info(f"Starting Multi-Channel Auto-EQ: device={device_id}, channels={len(channels_config)}, mode={mode}")
        logger.info(f"Channels config: {channels_config}")

        if not self.mixer_client or not self.mixer_client.is_connected:
            await self.send_to_client(websocket, {
                "type": "multi_channel_auto_eq_status",
                "active": False,
                "error": "Mixer not connected"
            })
            return

        if device_id is None or not channels_config or len(channels_config) == 0:
            await self.send_to_client(websocket, {
                "type": "multi_channel_auto_eq_status",
                "active": False,
                "error": f"Missing required parameters: device_id={device_id}, channels_config length={len(channels_config)}"
            })
            return

        # Validate channels_config structure
        for i, config in enumerate(channels_config):
            if 'channel' not in config:
                await self.send_to_client(websocket, {
                    "type": "multi_channel_auto_eq_status",
                    "active": False,
                    "error": f"Invalid channel config at index {i}: missing 'channel' field"
                })
                return

        try:
            # Initialize controller if needed
            if not self.multi_channel_auto_eq_controller:
                self.multi_channel_auto_eq_controller = MultiChannelAutoEQController(
                    mixer_client=self.mixer_client,
                    bleed_service=self.bleed_service
                )
            else:
                self.multi_channel_auto_eq_controller.mixer_client = self.mixer_client

            # Get event loop for thread-safe callbacks
            loop = asyncio.get_running_loop()

            def on_spectrum_update(spectrum_data: dict):
                """Handle spectrum data updates."""
                loop.call_soon_threadsafe(
                    lambda: asyncio.create_task(self.broadcast({
                        "type": "multi_channel_spectrum",
                        **spectrum_data
                    }))
                )

            def on_corrections_calculated(corrections_data: dict):
                """Handle EQ corrections updates."""
                loop.call_soon_threadsafe(
                    lambda: asyncio.create_task(self.broadcast({
                        "type": "multi_channel_corrections",
                        **corrections_data
                    }))
                )

            def on_status_update(status_data: dict):
                """Handle status updates."""
                loop.call_soon_threadsafe(
                    lambda: asyncio.create_task(self.broadcast({
                        "type": "multi_channel_status",
                        **status_data
                    }))
                )

            # Start multi-channel analysis
            success = self.multi_channel_auto_eq_controller.start_multi_channel(
                device_id=int(device_id),
                channels_config=channels_config,
                mode=mode,
                on_spectrum_callback=on_spectrum_update,
                on_corrections_callback=on_corrections_calculated,
                on_status_callback=on_status_update
            )

            if success:
                await self.send_to_client(websocket, {
                    "type": "multi_channel_auto_eq_status",
                    "active": True,
                    "channels_count": len(channels_config),
                    "message": f"Multi-channel Auto-EQ started for {len(channels_config)} channels"
                })
                logger.info(f"Multi-channel Auto-EQ started successfully")
            else:
                await self.send_to_client(websocket, {
                    "type": "multi_channel_auto_eq_status",
                    "active": False,
                    "error": "Failed to start multi-channel Auto-EQ"
                })

        except Exception as e:
            logger.error(f"Error starting multi-channel Auto-EQ: {e}", exc_info=True)
            await self.send_to_client(websocket, {
                "type": "multi_channel_auto_eq_status",
                "active": False,
                "error": str(e)
            })

    async def stop_multi_channel_auto_eq(self, websocket):
        """Stop multi-channel Auto-EQ analysis."""
        if self.multi_channel_auto_eq_controller:
            self.multi_channel_auto_eq_controller.stop_all()

        await self.send_to_client(websocket, {
            "type": "multi_channel_auto_eq_status",
            "active": False,
            "message": "Multi-channel Auto-EQ stopped"
        })

        logger.info("Multi-channel Auto-EQ stopped")

    async def set_channel_profile(self, websocket, channel: int = None, profile: str = None):
        """Change profile for a specific channel in multi-channel mode."""
        if not self.multi_channel_auto_eq_controller:
            await self.send_to_client(websocket, {
                "type": "multi_channel_status",
                "error": "Multi-channel Auto-EQ not initialized"
            })
            return

        if channel is None or profile is None:
            await self.send_to_client(websocket, {
                "type": "multi_channel_status",
                "error": "Missing channel or profile parameter"
            })
            return

        self.multi_channel_auto_eq_controller.set_channel_profile(channel, profile)

        await self.send_to_client(websocket, {
            "type": "multi_channel_status",
            "channel": channel,
            "profile": profile,
            "message": f"Profile changed for channel {channel} to {profile}"
        })

        logger.info(f"Channel {channel} profile changed to: {profile}")

    async def apply_channel_correction(self, websocket, channel: int = None):
        """Apply corrections for a specific channel."""
        if await self._block_quarantined_wing_write_surface(
            websocket,
            surface="apply_channel_correction",
            result_type="multi_channel_apply_result",
            request={"channel": channel},
        ):
            return

        if not self.multi_channel_auto_eq_controller:
            await self.send_to_client(websocket, {
                "type": "multi_channel_apply_result",
                "success": False,
                "error": "Multi-channel Auto-EQ not initialized"
            })
            return

        if channel is None:
            await self.send_to_client(websocket, {
                "type": "multi_channel_apply_result",
                "success": False,
                "error": "Missing channel parameter"
            })
            return

        try:
            success = self.multi_channel_auto_eq_controller.apply_channel_correction(channel)

            await self.send_to_client(websocket, {
                "type": "multi_channel_apply_result",
                "success": success,
                "channel": channel,
                "message": f"EQ corrections applied to channel {channel}" if success else f"Failed to apply EQ to channel {channel}"
            })

        except Exception as e:
            logger.error(f"Error applying EQ to channel {channel}: {e}")
            await self.send_to_client(websocket, {
                "type": "multi_channel_apply_result",
                "success": False,
                "channel": channel,
                "error": str(e)
            })

    async def apply_all_corrections(self, websocket):
        """Apply corrections for all channels."""
        if await self._block_quarantined_wing_write_surface(
            websocket,
            surface="apply_all_corrections",
            result_type="multi_channel_apply_result",
        ):
            return

        if not self.multi_channel_auto_eq_controller:
            await self.send_to_client(websocket, {
                "type": "multi_channel_apply_result",
                "success": False,
                "error": "Multi-channel Auto-EQ not initialized. Start batch analysis first."
            })
            return

        # Check if there are any channels with corrections
        if not self.multi_channel_auto_eq_controller.active_channels:
            await self.send_to_client(websocket, {
                "type": "multi_channel_apply_result",
                "success": False,
                "error": "No channels with corrections available. Run analysis first."
            })
            return

        try:
            results = self.multi_channel_auto_eq_controller.apply_all_corrections()
            success_count = sum(1 for v in results.values() if v)

            await self.send_to_client(websocket, {
                "type": "multi_channel_apply_result",
                "success": success_count > 0,
                "results": results,
                "message": f"Applied corrections to {success_count}/{len(results)} channels"
            })

        except Exception as e:
            logger.error(f"Error applying all corrections: {e}")
            await self.send_to_client(websocket, {
                "type": "multi_channel_apply_result",
                "success": False,
                "error": str(e)
            })

    def get_system_measurement_status(self) -> dict:
        """Get current system/master measurement state."""
        controller = self.system_measurement_controller
        if not controller:
            return {"active": False}

        result = controller.last_result
        return {
            "active": True,
            "quality": getattr(result, "overall_quality", 0.0) if result else 0.0,
            "recommended_target": controller.get_recommended_target(),
            "num_corrections": len(controller.applied_corrections),
            "has_result": result is not None,
        }

    def _parse_target_bus(self, target_bus: str) -> TargetBus:
        normalized = (target_bus or "master").strip().lower()
        aliases = {
            "master": TargetBus.MASTER,
            "main": TargetBus.MASTER,
            "group": TargetBus.GROUP,
            "bus": TargetBus.GROUP,
            "matrix": TargetBus.MATRIX,
            "mtx": TargetBus.MATRIX,
        }
        if normalized not in aliases:
            raise ValueError(f"Unsupported target bus: {target_bus}")
        return aliases[normalized]

    async def start_system_measurement(
        self,
        websocket,
        device_id: str = None,
        reference_channel: int = None,
        measurement_channel: int = None,
        duration_sec: float = 6.0,
        target_bus: str = "master",
        target_id: int = 1,
        correction_mode: str = "flat",
        reference_curve: str = "pink_noise_live_pa",
    ):
        """Capture reference + measurement mic and estimate system transfer function."""
        if not self.mixer_client or not self.mixer_client.is_connected:
            await self.send_to_client(websocket, {
                "type": "system_measurement_result",
                "success": False,
                "error": "Mixer not connected",
            })
            return

        if reference_channel is None or measurement_channel is None:
            await self.send_to_client(websocket, {
                "type": "system_measurement_result",
                "success": False,
                "error": "reference_channel and measurement_channel are required",
            })
            return

        try:
            reference_channel = int(reference_channel)
            measurement_channel = int(measurement_channel)
            target_id = int(target_id)
            duration_sec = float(duration_sec)
            target = self._parse_target_bus(target_bus)
            correction_mode = (correction_mode or "flat").strip().lower()
            reference_curve = (reference_curve or "pink_noise_live_pa").strip().lower()
        except (TypeError, ValueError) as exc:
            await self.send_to_client(websocket, {
                "type": "system_measurement_result",
                "success": False,
                "error": str(exc),
            })
            return

        if (
            reference_channel < 1
            or measurement_channel < 1
            or reference_channel > LIVE_CAPTURE_MAX_CHANNELS
            or measurement_channel > LIVE_CAPTURE_MAX_CHANNELS
            or reference_channel == measurement_channel
            or target_id < 1
            or not math.isfinite(duration_sec)
            or duration_sec < SYSTEM_MEASUREMENT_MIN_DURATION_SEC
            or duration_sec > SYSTEM_MEASUREMENT_MAX_DURATION_SEC
        ):
            await self.send_to_client(websocket, {
                "type": "system_measurement_result",
                "success": False,
                "error": (
                    "System measurement requires distinct positive channels, "
                    f"a positive target id and duration between "
                    f"{SYSTEM_MEASUREMENT_MIN_DURATION_SEC:g} and "
                    f"{SYSTEM_MEASUREMENT_MAX_DURATION_SEC:g} seconds"
                ),
            })
            return

        current_task = asyncio.current_task()
        active_task = self._system_measurement_task
        if active_task is not None and not active_task.done():
            await self.send_to_client(websocket, {
                "type": "system_measurement_result",
                "success": False,
                "error": "system_measurement_already_running",
            })
            return

        self._system_measurement_task = current_task
        self.system_measurement_controller = None
        capture: Optional[AudioCapture] = None
        measurement_owns_capture = False

        try:
            async with self._audio_capture_lock:
                preview_owner = self._audio_capture_preview_owner
                engine = self.auto_soundcheck_engine
                engine_state = getattr(getattr(engine, "state", None), "value", None)
                if self.auto_soundcheck_running or (
                    engine is not None and engine_state not in ("stopped", "error")
                ):
                    raise RuntimeError("audio_capture_in_use_by_auto_soundcheck")
                if preview_owner is not None and preview_owner is not websocket:
                    raise RuntimeError("audio_capture_in_use_by_another_runtime_owner")

                measurement_owns_capture = preview_owner is None
                capture = await self._ensure_audio_capture_async(
                    device_id,
                    num_channels=max(reference_channel, measurement_channel),
                    buffer_seconds=max(8.0, duration_sec + 2.0),
                )
                start_sample_count = int(
                    getattr(capture, "captured_sample_count", 0)
                )
                required_samples = int(capture.sample_rate * duration_sec)

                await self.broadcast({
                    "type": "system_measurement_status",
                    "active": True,
                    "message": "Capturing reference and measurement mic...",
                })

                deadline = (
                    time.monotonic()
                    + duration_sec
                    + SYSTEM_MEASUREMENT_CAPTURE_GRACE_SEC
                )
                while (
                    int(getattr(capture, "captured_sample_count", 0))
                    - start_sample_count
                    < required_samples
                ):
                    if time.monotonic() >= deadline:
                        raise RuntimeError(
                            "Real audio input did not provide the full requested "
                            "measurement duration"
                        )
                    await asyncio.sleep(SYSTEM_MEASUREMENT_CAPTURE_POLL_SEC)

                capture_status = capture.get_status()
                callback_age = capture_status.get("callback_age_sec")
                if (
                    capture_status.get("real_input") is not True
                    or not isinstance(callback_age, (int, float))
                    or isinstance(callback_age, bool)
                    or not math.isfinite(float(callback_age))
                    or float(callback_age) > 1.0
                ):
                    raise RuntimeError(
                        "Real audio input is stale or unavailable after measurement"
                    )
                synchronized = capture.get_synchronized_buffers(
                    [reference_channel, measurement_channel],
                    required_samples,
                )
                available_samples = synchronized.get("available_samples", {})
                for channel in (reference_channel, measurement_channel):
                    if int(available_samples.get(channel, 0)) < required_samples:
                        raise RuntimeError(
                            f"Channel {channel} does not contain the full requested "
                            "measurement duration"
                        )

                reference_signal = np.asarray(
                    synchronized.get("buffers", {}).get(reference_channel, []),
                    dtype=np.float64,
                )
                measurement_signal = np.asarray(
                    synchronized.get("buffers", {}).get(measurement_channel, []),
                    dtype=np.float64,
                )
                if (
                    len(reference_signal) != required_samples
                    or len(measurement_signal) != required_samples
                ):
                    raise RuntimeError(
                        "Audio capture returned a truncated system measurement"
                    )

                ref_rms = 20.0 * np.log10(
                    np.sqrt(np.mean(reference_signal ** 2)) + 1e-10
                )
                mic_rms = 20.0 * np.log10(
                    np.sqrt(np.mean(measurement_signal ** 2)) + 1e-10
                )
                if ref_rms < -50.0:
                    raise RuntimeError("Reference channel level is too low for measurement")
                if mic_rms < -60.0:
                    raise RuntimeError("Measurement mic level is too low for analysis")

                controller = SystemMeasurementController(
                    mixer_client=self.mixer_client,
                    sample_rate=capture.sample_rate,
                )
                analysis = await self._cancellation_safe_thread_call(
                    controller.analyze_reference_measurement,
                    reference_signal,
                    measurement_signal,
                    correction_mode=correction_mode,
                    reference_curve=reference_curve,
                )
                self.system_measurement_controller = controller

            await self.broadcast({
                "type": "system_measurement_result",
                "success": True,
                "target_bus": target.value,
                "target_id": target_id,
                "correction_mode": analysis.get("correction_mode", correction_mode),
                "reference_curve": analysis.get("reference_curve", reference_curve),
                "reference_channel": reference_channel,
                "measurement_channel": measurement_channel,
                "recommended_target": controller.get_recommended_target(),
                "captured_samples": required_samples,
                "capture_real_input": True,
                **analysis,
            })
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("System measurement failed: %s", e, exc_info=True)
            await self.send_to_client(websocket, {
                "type": "system_measurement_result",
                "success": False,
                "error": str(e),
            })
        finally:
            try:
                if measurement_owns_capture and capture is not None:
                    async with self._audio_capture_lock:
                        if self.audio_capture is capture:
                            await self._cancellation_safe_thread_call(capture.stop)
                            if self.audio_capture is capture:
                                self.audio_capture = None
            finally:
                if self._system_measurement_task is current_task:
                    self._system_measurement_task = None

    async def apply_system_measurement(self, websocket, target_bus: str = "master", target_id: int = 1):
        """Apply previously calculated system EQ corrections to the target bus."""
        if await self._block_quarantined_wing_write_surface(
            websocket,
            surface="apply_system_measurement",
            result_type="system_measurement_apply_result",
            request={"target_bus": target_bus, "target_id": target_id},
        ):
            return

        if not self.system_measurement_controller:
            await self.send_to_client(websocket, {
                "type": "system_measurement_apply_result",
                "success": False,
                "error": "No measurement result available",
            })
            return

        try:
            target = self._parse_target_bus(target_bus)
            target_id = int(target_id)
            success = self.system_measurement_controller.apply_corrections(target, target_id)
            await self.broadcast({
                "type": "system_measurement_apply_result",
                "success": success,
                "target_bus": target.value,
                "target_id": target_id,
                "corrections": [c.to_dict() for c in self.system_measurement_controller.applied_corrections],
                "message": "System EQ corrections applied" if success else "Failed to apply system EQ corrections",
            })
        except Exception as e:
            logger.error("Failed to apply system measurement corrections: %s", e, exc_info=True)
            await self.send_to_client(websocket, {
                "type": "system_measurement_apply_result",
                "success": False,
                "error": str(e),
            })

    async def reset_system_measurement(self, websocket, target_bus: str = "master", target_id: int = 1):
        """Reset EQ on the target bus used for system measurement."""
        if await self._block_quarantined_wing_write_surface(
            websocket,
            surface="reset_system_measurement",
            result_type="system_measurement_reset_result",
            request={"target_bus": target_bus, "target_id": target_id},
        ):
            return

        if not self.system_measurement_controller:
            self.system_measurement_controller = SystemMeasurementController(mixer_client=self.mixer_client)

        try:
            target = self._parse_target_bus(target_bus)
            target_id = int(target_id)
            success = self.system_measurement_controller.reset_corrections(target, target_id)
            await self.broadcast({
                "type": "system_measurement_reset_result",
                "success": success,
                "target_bus": target.value,
                "target_id": target_id,
                "message": "System EQ reset" if success else "Failed to reset system EQ",
            })
        except Exception as e:
            logger.error("Failed to reset system measurement corrections: %s", e, exc_info=True)
            await self.send_to_client(websocket, {
                "type": "system_measurement_reset_result",
                "success": False,
                "error": str(e),
            })

    def get_phase_alignment_status(self) -> dict:
        """Get current phase alignment status."""
        if not self.phase_alignment_controller:
            return {"active": False}
        return self.phase_alignment_controller.get_status()

    async def start_phase_alignment(self, websocket, device_id: str = None,
                                   reference_channel: int = None, channels: List[int] = None,
                                   settings: dict = None,
                                   apply_once: bool = False,
                                   apply_once_duration_sec: float = 5.0):
        """Start phase alignment analysis. If apply_once=True, one measurement then auto-stop and apply (for soundcheck)."""
        channels = channels or []
        logger.info(f"Starting Phase Alignment: device={device_id}, ref={reference_channel}, channels={channels}")

        if not self.mixer_client or not self.mixer_client.is_connected:
            await self.send_to_client(websocket, {
                "type": "phase_alignment_status",
                "active": False,
                "error": "Mixer not connected"
            })
            return

        if device_id is None or reference_channel is None or not channels:
            await self.send_to_client(websocket, {
                "type": "phase_alignment_status",
                "active": False,
                "error": "Missing required parameters: device_id, reference_channel, or channels"
            })
            return

        try:
            # Initialize controller if needed
            settings = settings or {}
            if not self.phase_alignment_controller:
                self.phase_alignment_controller = PhaseAlignmentController(
                    mixer_client=self.mixer_client,
                    bleed_service=self.bleed_service,
                    settings=settings
                )
            else:
                self.phase_alignment_controller.mixer_client = self.mixer_client
                # Update settings for future analyzer instances
                self.phase_alignment_controller.settings = settings

            # Get event loop for thread-safe callbacks
            loop = asyncio.get_running_loop()

            def on_measurement_update(measurements: dict):
                """Handle measurement updates."""
                import numpy as np

                # Конвертируем tuple ключи в строки и NumPy типы в Python типы для JSON сериализации
                measurements_serializable = {}
                for key, value in measurements.items():
                    if isinstance(key, tuple):
                        # Конвертируем tuple в строку вида "(ref_ch, ch)"
                        serializable_key = f"({key[0]}, {key[1]})"
                    else:
                        serializable_key = str(key)

                    # Конвертируем значения измерения в JSON-совместимые типы
                    serializable_value = {}
                    if isinstance(value, dict):
                        for k, v in value.items():
                            # Конвертируем NumPy типы в Python типы
                            if hasattr(v, 'item'):  # NumPy scalar
                                serializable_value[k] = v.item()
                            elif isinstance(v, (np.integer, np.floating)):
                                serializable_value[k] = float(v) if isinstance(v, np.floating) else int(v)
                            elif isinstance(v, (list, tuple)):
                                # Конвертируем списки/кортежи с NumPy типами
                                serializable_value[k] = [
                                    item.item() if hasattr(item, 'item') else
                                    (float(item) if isinstance(item, np.floating) else int(item) if isinstance(item, np.integer) else item)
                                    for item in v
                                ]
                            else:
                                serializable_value[k] = v
                    else:
                        serializable_value = value

                    measurements_serializable[serializable_key] = serializable_value

                loop.call_soon_threadsafe(
                    lambda: asyncio.create_task(self.broadcast({
                        "type": "phase_alignment_measurements",
                        "measurements": measurements_serializable
                    }))
                )

            # Convert device_id to int if needed
            device_index = int(device_id) if isinstance(device_id, (str, int)) else None

            # Run start_analysis in a thread pool to avoid blocking the event loop
            # (PyAudio's pa.open() can block at the C/CoreAudio level if device is busy)
            loop = asyncio.get_running_loop()
            success = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self.phase_alignment_controller.start_analysis(
                        device_id=device_index,
                        reference_channel=reference_channel,
                        channels=channels,
                        on_measurement_callback=on_measurement_update,
                        apply_once=apply_once,
                        apply_once_duration_sec=apply_once_duration_sec,
                        settings=settings
                    )
                ),
                timeout=10.0
            )

            if success:
                await self.broadcast({
                    "type": "phase_alignment_status",
                    "active": True,
                    "reference_channel": reference_channel,
                    "channels": channels,
                    "message": "Phase alignment analysis started"
                })
            else:
                await self.send_to_client(websocket, {
                    "type": "phase_alignment_status",
                    "active": False,
                    "error": "Failed to start phase alignment analysis"
                })

        except asyncio.TimeoutError:
            logger.error("Phase alignment start timed out (audio device may be busy)")
            # Force cleanup if timeout
            if self.phase_alignment_controller and self.phase_alignment_controller.analyzer:
                analyzer = self.phase_alignment_controller.analyzer
                self._safe_cleanup_call(
                    lambda: setattr(analyzer, "is_running", False),
                    "Error stopping phase alignment analyzer after timeout",
                )
                self.phase_alignment_controller.analyzer = None
                self.phase_alignment_controller.is_active = False
            await self.send_to_client(websocket, {
                "type": "phase_alignment_status",
                "active": False,
                "error": "Phase alignment start timed out (audio device busy)"
            })
        except Exception as e:
            logger.error(f"Error starting phase alignment: {e}", exc_info=True)
            await self.send_to_client(websocket, {
                "type": "phase_alignment_status",
                "active": False,
                "error": str(e)
            })

    async def stop_phase_alignment(self, websocket):
        """Stop phase alignment analysis and apply corrections."""
        logger.info("Stopping Phase Alignment...")

        measurements = None
        if self.phase_alignment_controller:
            # Получаем измерения перед остановкой
            if self.phase_alignment_controller.analyzer:
                measurements = self.phase_alignment_controller.analyzer.get_measurements()
                logger.info(f"Got measurements before stop: {measurements}")

            # Run stop_analysis in thread pool to avoid blocking event loop
            # (PyAudio stream.close() can block at C/CoreAudio level)
            loop = asyncio.get_running_loop()
            try:
                await asyncio.wait_for(
                    loop.run_in_executor(None, self.phase_alignment_controller.stop_analysis),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                logger.warning("Phase alignment stop_analysis timed out after 5s, forcing cleanup...")
                # Force cleanup
                if self.phase_alignment_controller.analyzer:
                    self.phase_alignment_controller.analyzer.is_running = False
                    self.phase_alignment_controller.analyzer = None
                self.phase_alignment_controller.is_active = False
            except Exception as e:
                logger.error(f"Error in stop_analysis: {e}")

        await self.broadcast({
            "type": "phase_alignment_status",
            "active": False,
            "message": "Phase alignment analysis stopped"
        })

        # Автоматически применяем коррекции после остановки, если есть измерения
        if measurements and len(measurements) > 0:
            logger.info("Auto-applying corrections after stop...")
            # Конвертируем tuple ключи в строки для JSON
            measurements_serializable = {}
            import numpy as np
            for key, value in measurements.items():
                if isinstance(key, tuple):
                    serializable_key = f"({key[0]}, {key[1]})"
                else:
                    serializable_key = str(key)

                # Конвертируем NumPy типы
                serializable_value = {}
                if isinstance(value, dict):
                    for k, v in value.items():
                        if hasattr(v, 'item'):
                            serializable_value[k] = v.item()
                        elif isinstance(v, (np.integer, np.floating)):
                            serializable_value[k] = float(v) if isinstance(v, np.floating) else int(v)
                        else:
                            serializable_value[k] = v
                else:
                    serializable_value = value

                measurements_serializable[serializable_key] = serializable_value

            # Применяем коррекции
            await self.apply_phase_corrections(websocket, measurements_serializable)

    async def apply_phase_corrections(self, websocket, measurements: dict = None):
        """Apply phase/delay corrections to mixer."""
        logger.info(f"apply_phase_corrections called. Measurements type: {type(measurements)}, value: {measurements}")
        if await self._block_quarantined_wing_write_surface(
            websocket,
            surface="apply_phase_corrections",
            result_type="phase_alignment_apply_result",
            request={"measurements": measurements},
        ):
            return

        if not self.mixer_client or not self.mixer_client.is_connected:
            await self.send_to_client(websocket, {
                "type": "phase_alignment_apply_result",
                "success": False,
                "error": "Mixer not connected"
            })
            return

        # Если контроллер не инициализирован, но есть измерения, применяем напрямую
        if not self.phase_alignment_controller:
            logger.warning("Phase alignment controller not initialized, applying corrections directly")
            if not measurements:
                await self.send_to_client(websocket, {
                    "type": "phase_alignment_apply_result",
                    "success": False,
                    "error": "No measurements provided and controller not initialized"
                })
                return
            # Применяем напрямую через mixer_client
            try:
                result = await self._apply_phase_corrections_direct(measurements)
                success = result.get("applied", False) if isinstance(result, dict) else bool(result)
                await self.broadcast({
                    "type": "phase_alignment_apply_result",
                    "success": success,
                    "blocked": result.get("blocked", False) if isinstance(result, dict) else False,
                    "result": result if isinstance(result, dict) else None,
                    "message": f"Phase/delay corrections applied to {len(measurements)} channel pairs" if success else "Phase/delay corrections blocked or failed"
                })
            except Exception as e:
                logger.error(f"Error applying phase corrections directly: {e}", exc_info=True)
                await self.send_to_client(websocket, {
                    "type": "phase_alignment_apply_result",
                    "success": False,
                    "error": str(e)
                })
            return

        try:
            logger.info(f"Applying corrections through controller. Measurements type: {type(measurements)}, value: {measurements}")

            # Проверка наличия измерений
            if not measurements:
                logger.warning("No measurements provided, trying to get from analyzer")
                if self.phase_alignment_controller.analyzer:
                    measurements = self.phase_alignment_controller.analyzer.get_measurements()
                    logger.info(f"Got measurements from analyzer: {measurements}")
                else:
                    await self.send_to_client(websocket, {
                        "type": "phase_alignment_apply_result",
                        "success": False,
                        "error": "No measurements available. Run analysis first."
                    })
                    return

            if not isinstance(measurements, dict):
                await self.send_to_client(websocket, {
                    "type": "phase_alignment_apply_result",
                    "success": False,
                    "error": f"Invalid measurements format: expected dict, got {type(measurements)}"
                })
                return

            if len(measurements) == 0:
                await self.send_to_client(websocket, {
                    "type": "phase_alignment_apply_result",
                    "success": False,
                    "error": "Measurements dictionary is empty. No corrections to apply."
                })
                return

            result = self.phase_alignment_controller.apply_corrections(measurements)
            success = result.get("applied", False) if isinstance(result, dict) else bool(result)

            await self.broadcast({
                "type": "phase_alignment_apply_result",
                "success": success,
                "blocked": result.get("blocked", False) if isinstance(result, dict) else False,
                "result": result if isinstance(result, dict) else None,
                "corrections": self.phase_alignment_controller.corrections.copy() if self.phase_alignment_controller.corrections else {},
                "message": "Phase/delay corrections applied to mixer" if success else "Phase/delay corrections blocked or failed"
            })

        except Exception as e:
            logger.error(f"Error applying phase corrections: {e}", exc_info=True)
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            await self.send_to_client(websocket, {
                "type": "phase_alignment_apply_result",
                "success": False,
                "error": f"Error: {str(e)}"
            })

    async def _apply_phase_corrections_direct(self, measurements: dict):
        """Return blocked phase/delay evidence instead of writing directly."""
        controller = self.phase_alignment_controller or PhaseAlignmentController(
            mixer_client=self.mixer_client
        )
        return controller.apply_corrections(measurements)

    async def reset_phase_delay(self, websocket, channels: List[int] = None):
        """Reset and disable all phase inversions and delays on specified channels."""
        if await self._block_quarantined_wing_write_surface(
            websocket,
            surface="reset_phase_delay",
            result_type="phase_alignment_reset_result",
            request={"channels": channels or []},
        ):
            return

        if not self.mixer_client or not self.mixer_client.is_connected:
            await self.send_to_client(websocket, {
                "type": "phase_alignment_reset_result",
                "success": False,
                "error": "Mixer not connected"
            })
            return

        if not channels or len(channels) == 0:
            await self.send_to_client(websocket, {
                "type": "phase_alignment_reset_result",
                "success": False,
                "error": "No channels specified"
            })
            return

        try:
            # Если контроллер инициализирован, используем его метод
            if self.phase_alignment_controller:
                result = self.phase_alignment_controller.reset_all_phase_delay(channels)
            else:
                # Иначе сбрасываем напрямую через mixer_client
                result = await self._reset_phase_delay_direct(channels)
            success = result.get("applied", False) if isinstance(result, dict) else bool(result)

            await self.broadcast({
                "type": "phase_alignment_reset_result",
                "success": success,
                "blocked": result.get("blocked", False) if isinstance(result, dict) else False,
                "result": result if isinstance(result, dict) else None,
                "message": f"Phase and delay reset for {len(channels)} channels" if success else "Phase/delay reset blocked or failed"
            })

        except Exception as e:
            logger.error(f"Error resetting phase/delay: {e}", exc_info=True)
            await self.send_to_client(websocket, {
                "type": "phase_alignment_reset_result",
                "success": False,
                "error": str(e)
            })

    async def _reset_phase_delay_direct(self, channels: List[int]):
        """Return blocked phase/delay reset evidence instead of writing directly."""
        controller = self.phase_alignment_controller or PhaseAlignmentController(
            mixer_client=self.mixer_client
        )
        return controller.reset_all_phase_delay(channels)

    # ========== Auto Fader Methods ==========

    def get_auto_fader_status(self) -> dict:
        """Get current Auto Fader status."""
        if not self.auto_fader_controller:
            return {"active": False, "live_mode": self.live_mode}
        st = self.auto_fader_controller.get_status()
        st["live_mode"] = self.live_mode
        st["freeze"] = self.auto_fader_controller.get_freeze_status()
        return st

    def get_ready_for_live_status(self) -> dict:
        """Checklist: ready for live (all systems calibrated, no clipping)."""
        mixer_connected = bool(self.mixer_client and self.mixer_client.is_connected)
        gain_available = bool(self.gain_staging)
        gain_automation_frozen = False
        if self.gain_staging:
            raw_gain_frozen = getattr(self.gain_staging, "automation_frozen", False)
            gain_automation_frozen = raw_gain_frozen if isinstance(raw_gain_frozen, bool) else False
        fader_available = False
        fader_automation_frozen = False
        if self.auto_fader_controller:
            try:
                fader_status = self.auto_fader_controller.get_status() or {}
                fader_available = bool(
                    fader_status.get("active")
                    or fader_status.get("realtime_enabled")
                    or fader_status.get("collecting")
                )
            except Exception as exc:
                logger.warning("Failed to read Auto Fader readiness status: %s", exc)
            try:
                freeze_status = self.auto_fader_controller.get_freeze_status()
                if isinstance(freeze_status, dict):
                    fader_automation_frozen = bool(
                        freeze_status.get("automation_frozen")
                        or freeze_status.get("global_frozen")
                        or freeze_status.get("frozen")
                        or freeze_status.get("all_frozen")
                    )
            except Exception as exc:
                logger.warning("Failed to read Auto Fader freeze status: %s", exc)
        manual_emergency_clear = not self._manual_write_emergency_stopped
        automation_unfrozen = not (gain_automation_frozen or fader_automation_frozen)
        master_true_peak = self._coerce_finite_float(self._latest_master_true_peak_dbtp)
        master_true_peak_age_sec = (
            max(0.0, time.time() - self._latest_master_true_peak_timestamp)
            if self._latest_master_true_peak_timestamp > 0
            else None
        )
        master_true_peak_fresh = bool(
            master_true_peak is not None
            and master_true_peak_age_sec is not None
            and master_true_peak_age_sec <= 10.0
        )
        no_clipping_threshold_dbtp = -3.0
        no_clipping = bool(
            master_true_peak_fresh
            and master_true_peak is not None
            and master_true_peak <= no_clipping_threshold_dbtp
        )
        ready = (
            mixer_connected
            and gain_available
            and fader_available
            and manual_emergency_clear
            and automation_unfrozen
            and no_clipping
        )
        blocked_reasons = []
        if not mixer_connected:
            blocked_reasons.append("mixer_not_connected")
        if not gain_available:
            blocked_reasons.append("gain_staging_unavailable")
        if not fader_available:
            blocked_reasons.append("auto_fader_unavailable")
        if not manual_emergency_clear:
            blocked_reasons.append("manual_write_emergency_stop_active")
        if not automation_unfrozen:
            blocked_reasons.append("automation_frozen_after_emergency_stop")
        if not master_true_peak_fresh:
            blocked_reasons.append("master_true_peak_unavailable_or_stale")
        elif not no_clipping:
            blocked_reasons.append("master_true_peak_headroom_insufficient")
        return {
            "ready": ready,
            "blocked_reasons": blocked_reasons,
            "checks": {
                "mixer_connected": mixer_connected,
                "gain_staging_available": gain_available,
                "auto_fader_available": fader_available,
                "manual_write_emergency_clear": manual_emergency_clear,
                "automation_unfrozen": automation_unfrozen,
                "no_clipping": no_clipping,
                "master_true_peak_fresh": master_true_peak_fresh,
            },
            "master_true_peak": {
                "value_dbtp": master_true_peak,
                "threshold_dbtp": no_clipping_threshold_dbtp,
                "age_sec": round(master_true_peak_age_sec, 3) if master_true_peak_age_sec is not None else None,
                "max_age_sec": 10.0,
            },
        }

    def record_master_true_peak_evidence(self, value_dbtp: Any) -> bool:
        """Record a measured master true peak; callers must not pass channel peaks."""
        value = self._coerce_finite_float(value_dbtp)
        if value is None:
            return False
        self._latest_master_true_peak_dbtp = value
        self._latest_master_true_peak_timestamp = time.time()
        return True

    def _verified_master_capture_route_proof(self) -> Dict[str, Any]:
        path_text = self._master_capture_route_proof_path
        expected_sha256 = self._master_capture_route_proof_sha256
        if not path_text or not expected_sha256:
            return {
                "verified": False,
                "reason": "proof_path_and_sha256_required",
            }
        path = Path(path_text).expanduser()
        try:
            raw = path.read_bytes()
        except OSError as exc:
            return {
                "verified": False,
                "reason": "proof_artifact_unreadable",
                "path": str(path),
                "error": str(exc),
            }
        actual_sha256 = hashlib.sha256(raw).hexdigest()
        if actual_sha256 != expected_sha256:
            return {
                "verified": False,
                "reason": "proof_sha256_mismatch",
                "path": str(path),
                "expected_sha256": expected_sha256,
                "actual_sha256": actual_sha256,
            }
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            return {
                "verified": False,
                "reason": "proof_artifact_invalid_json",
                "path": str(path),
                "actual_sha256": actual_sha256,
                "error": str(exc),
            }
        if not isinstance(payload, Mapping):
            return {
                "verified": False,
                "reason": "proof_artifact_contract_failed",
                "path": str(path),
                "actual_sha256": actual_sha256,
                "missing_or_invalid": ["object_payload_required"],
            }
        required_text = (
            "proof_id",
            "console_serial",
            "firmware_version",
            "capture_device",
        )
        missing = [
            key for key in required_text if not str(payload.get(key) or "").strip()
        ]
        channels = payload.get("master_channels")
        capture_device_index = payload.get("capture_device_index")
        capture_channel_count = payload.get("capture_channel_count")
        capture_sample_rate_hz = payload.get("capture_sample_rate_hz")
        if payload.get("accepted") is not True:
            missing.append("accepted_true")
        if channels != [47, 48]:
            missing.append("master_channels_47_48")
        if (
            isinstance(capture_device_index, bool)
            or not isinstance(capture_device_index, int)
            or capture_device_index < 0
        ):
            missing.append("capture_device_index_nonnegative_integer")
        if capture_channel_count != 48 or isinstance(capture_channel_count, bool):
            missing.append("capture_channel_count_exactly_48")
        if (
            isinstance(capture_sample_rate_hz, bool)
            or not isinstance(capture_sample_rate_hz, int)
            or capture_sample_rate_hz not in {44_100, 48_000}
        ):
            missing.append("capture_sample_rate_hz_44100_or_48000")
        if missing:
            return {
                "verified": False,
                "reason": "proof_artifact_contract_failed",
                "path": str(path),
                "actual_sha256": actual_sha256,
                "missing_or_invalid": missing,
            }
        return {
            "verified": True,
            "reason": "immutable_route_proof_verified",
            "path": str(path),
            "sha256": actual_sha256,
            "proof_id": str(payload["proof_id"]),
            "console_serial": str(payload["console_serial"]),
            "firmware_version": str(payload["firmware_version"]),
            "capture_device": str(payload["capture_device"]),
            "capture_device_index": int(capture_device_index),
            "capture_channel_count": int(capture_channel_count),
            "capture_sample_rate_hz": int(capture_sample_rate_hz),
            "master_channels": [47, 48],
        }

    async def _fresh_master_usb_route_readback(
        self,
        route_proof: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Prove that the connected console currently routes Main L/R to USB 47/48."""

        expected = {
            "/$syscfg/$serial": str(route_proof.get("console_serial") or ""),
            "/$syscfg/$firmware": str(route_proof.get("firmware_version") or ""),
            "/io/out/USB/47/grp": "MAIN",
            # WING writes route selectors 1-based but reports them 0-based.
            "/io/out/USB/47/in": 0,
            "/io/out/USB/48/grp": "MAIN",
            "/io/out/USB/48/in": 1,
        }
        client = getattr(self, "mixer_client", None)
        query = getattr(client, "query_parameter", None) if client else None
        if not (
            client
            and getattr(client, "is_connected", False)
            and callable(query)
        ):
            return {
                "verified": False,
                "reason": "master_capture_route_physical_readback_unavailable",
                "expected": expected,
                "readbacks": {},
                "captured_at_epoch": time.time(),
            }

        def expected_value_matches(expected_value: Any, actual: Any) -> bool:
            if isinstance(expected_value, str):
                return bool(
                    expected_value
                    and str(actual or "").strip().upper()
                    == expected_value.strip().upper()
                )
            return bool(
                isinstance(actual, (int, float))
                and not isinstance(actual, bool)
                and math.isfinite(float(actual))
                and float(actual).is_integer()
                and int(actual) == expected_value
            )

        async def collect_pass() -> tuple[Dict[str, Any], Dict[str, Any], List[str]]:
            pass_readbacks: Dict[str, Any] = {}
            pass_values: Dict[str, Any] = {}
            pass_mismatches: List[str] = []
            for address, expected_value in expected.items():
                try:
                    if inspect.iscoroutinefunction(query):
                        raw = await query(address, 0.5)
                    else:
                        raw = await self._cancellation_safe_thread_call(
                            query,
                            address,
                            0.5,
                        )
                        raw = await self._maybe_await(raw)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    raw = {
                        "status": "exception",
                        "address": address,
                        "error": type(exc).__name__,
                    }
                captured_at = time.time()
                row = dict(raw) if isinstance(raw, Mapping) else {
                    "status": "invalid_response",
                    "response_type": type(raw).__name__,
                }
                row["captured_at_epoch"] = captured_at
                pass_readbacks[address] = row
                confirmed = (
                    raw.get("confirmed_round_trips")
                    if isinstance(raw, Mapping)
                    else None
                )
                actual = raw.get("value") if isinstance(raw, Mapping) else None
                pass_values[address] = actual
                valid_transport = bool(
                    isinstance(raw, Mapping)
                    and raw.get("status") == "received"
                    and raw.get("address") == address
                    and isinstance(confirmed, int)
                    and not isinstance(confirmed, bool)
                    and confirmed >= 2
                )
                if not valid_transport or not expected_value_matches(
                    expected_value,
                    actual,
                ):
                    pass_mismatches.append(address)
            return pass_readbacks, pass_values, pass_mismatches

        collection_started_at = time.time()
        first_readbacks, first_values, first_mismatches = await collect_pass()
        readbacks, values, mismatches = await collect_pass()
        consistency_mismatches = [
            address
            for address in expected
            if not expected_value_matches(first_values.get(address), values.get(address))
        ]
        collection_completed_at = time.time()
        oldest_read_at = min(
            (
                float(row.get("captured_at_epoch"))
                for pass_rows in (first_readbacks, readbacks)
                for row in pass_rows.values()
                if isinstance(row.get("captured_at_epoch"), (int, float))
            ),
            default=collection_completed_at,
        )
        collection_max_age_sec = 5.0
        collection_stale = bool(
            collection_completed_at - collection_started_at > collection_max_age_sec
            or collection_completed_at - oldest_read_at > collection_max_age_sec
        )
        mismatches = sorted(set(first_mismatches + mismatches))
        verified = not mismatches and not consistency_mismatches and not collection_stale
        if collection_stale:
            reason = "master_capture_route_readback_collection_stale"
        elif consistency_mismatches:
            reason = "master_capture_route_readback_consistency_mismatch"
        elif mismatches:
            reason = "master_capture_route_physical_readback_mismatch"
        else:
            reason = "fresh_master_usb_route_physical_readback_verified"
        return {
            "verified": verified,
            "reason": reason,
            "expected": expected,
            "readbacks": readbacks,
            "first_pass_readbacks": first_readbacks,
            "mismatched_addresses": mismatches,
            "consistency_mismatched_addresses": consistency_mismatches,
            "captured_at_epoch": oldest_read_at,
            "completed_at_epoch": collection_completed_at,
            "max_age_sec": collection_max_age_sec,
            "confirmed_round_trips_required": 2,
            "complete_passes_required": 2,
        }

    async def _supervised_transaction_audio_recheck(
        self,
        operation: ConsoleTransactionOperation,
        snapshot: Mapping[str, Any],
        apply_evidence: Mapping[str, Any],
    ) -> AudioRecheckResult:
        """Fail closed unless a fresh conservative master-peak estimate is safe."""
        route_proof = self._verified_master_capture_route_proof()
        physical_route_before = (
            await self._fresh_master_usb_route_readback(route_proof)
            if route_proof.get("verified") is True
            else {
                "verified": False,
                "reason": "immutable_route_proof_required_before_physical_readback",
            }
        )
        capture = getattr(self, "audio_capture", None)
        capture_owner = "server_audio_capture"
        if capture is None or not getattr(capture, "running", False):
            engine_capture = getattr(
                getattr(self, "auto_soundcheck_engine", None),
                "audio_capture",
                None,
            )
            if engine_capture is not None and getattr(
                engine_capture,
                "running",
                False,
            ):
                capture = engine_capture
                capture_owner = "active_auto_soundcheck_engine"
        capture_status_before: Dict[str, Any] = {}
        capture_status_after: Dict[str, Any] = {}
        capture_progressed = False

        def capture_identity_matches(status: Mapping[str, Any]) -> bool:
            return bool(
                str(status.get("device_name") or "").strip()
                == str(route_proof.get("capture_device") or "").strip()
                and isinstance(status.get("device_index"), int)
                and not isinstance(status.get("device_index"), bool)
                and status.get("device_index")
                == route_proof.get("capture_device_index")
                and status.get("num_channels")
                == route_proof.get("capture_channel_count")
                and not isinstance(status.get("num_channels"), bool)
                and status.get("sample_rate")
                == route_proof.get("capture_sample_rate_hz")
                and not isinstance(status.get("sample_rate"), bool)
            )

        def capture_status_is_fresh(status: Mapping[str, Any]) -> bool:
            callback_age = status.get("callback_age_sec")
            sample_count = status.get("captured_sample_count")
            return bool(
                status.get("running") is True
                and status.get("real_input") is True
                and isinstance(callback_age, (int, float))
                and not isinstance(callback_age, bool)
                and math.isfinite(float(callback_age))
                and 0.0 <= float(callback_age) <= 1.0
                and isinstance(sample_count, int)
                and not isinstance(sample_count, bool)
                and sample_count >= 0
            )

        if (
            route_proof.get("verified") is True
            and physical_route_before.get("verified") is True
            and capture is not None
            and getattr(capture, "running", False)
            and callable(getattr(capture, "get_status", None))
        ):
            try:
                capture_status_before = dict(capture.get_status() or {})
            except Exception as exc:
                capture_status_before = {
                    "status_error": type(exc).__name__,
                }
            if (
                capture_status_is_fresh(capture_status_before)
                and capture_identity_matches(capture_status_before)
            ):
                start_count = int(capture_status_before["captured_sample_count"])
                deadline = time.monotonic() + 0.25
                while time.monotonic() < deadline:
                    await asyncio.sleep(0.025)
                    try:
                        capture_status_after = dict(capture.get_status() or {})
                    except Exception as exc:
                        capture_status_after = {
                            "status_error": type(exc).__name__,
                        }
                        break
                    if (
                        capture_status_is_fresh(capture_status_after)
                        and capture_identity_matches(capture_status_after)
                        and int(capture_status_after["captured_sample_count"])
                        > start_count
                    ):
                        capture_progressed = True
                        break

        capture_peaks: List[float] = []
        capture_peaks_by_channel: Dict[int, float] = {}
        master_channels: List[int] = []
        capture_fresh = bool(
            capture_status_is_fresh(capture_status_before)
            and capture_status_is_fresh(capture_status_after)
            and capture_identity_matches(capture_status_before)
            and capture_identity_matches(capture_status_after)
            and capture_progressed
        )
        if capture_fresh:
            channel_count = int(getattr(capture, "num_channels", 0) or 0)
            master_channels = [channel for channel in (47, 48) if channel <= channel_count]
            for channel in master_channels:
                try:
                    peak = self._coerce_finite_float(capture.get_peak(channel))
                except Exception as exc:
                    logger.warning("Master peak read failed for channel %s: %s", channel, exc)
                    peak = None
                if peak is not None:
                    capture_peaks.append(peak)
                    capture_peaks_by_channel[channel] = peak
        physical_route_after: Dict[str, Any] = {}
        if capture_fresh and capture_peaks:
            physical_route_after = await self._fresh_master_usb_route_readback(
                route_proof
            )
        route_verified_around_measurement = bool(
            route_proof.get("verified") is True
            and physical_route_before.get("verified") is True
            and physical_route_after.get("verified") is True
        )
        if capture_fresh and capture_peaks and route_verified_around_measurement:
            # AudioCapture exposes sample peak. Add the same conservative
            # inter-sample allowance used by the gain-staging heuristics.
            self.record_master_true_peak_evidence(max(capture_peaks) + 3.0)
        true_peak = self._coerce_finite_float(self._latest_master_true_peak_dbtp)
        age_sec = (
            max(0.0, time.time() - self._latest_master_true_peak_timestamp)
            if self._latest_master_true_peak_timestamp > 0
            else None
        )
        evidence = {
            "family_id": operation.family_id,
            "operation_id": operation.operation_id,
            "master_true_peak_dbtp": true_peak,
            "age_sec": round(age_sec, 3) if age_sec is not None else None,
            "max_age_sec": 5.0,
            "ceiling_dbtp": -1.0,
            "measurement_source": "master_channels_sample_peak_plus_3db",
            "master_channels": master_channels,
            "master_sample_peaks_dbfs": capture_peaks,
            "route_verified": bool(
                route_verified_around_measurement
            ),
            "route_proof_id": route_proof.get("proof_id"),
            "route_proof": route_proof,
            "physical_route_readback": physical_route_after,
            "physical_route_readback_before_audio": physical_route_before,
            "physical_route_readback_after_audio": physical_route_after,
            "capture_status_before": capture_status_before,
            "capture_status_after": capture_status_after,
            "capture_owner": capture_owner,
            "capture_real_input_fresh": capture_fresh,
            "capture_identity_matches_proof": bool(
                capture_identity_matches(capture_status_before)
                and capture_identity_matches(capture_status_after)
            ),
            "capture_sample_progressed": capture_progressed,
            "legacy_route_flag_ignored": getattr(
                self,
                "_master_capture_route_verified",
                False,
            ),
        }
        if route_proof.get("verified") is not True:
            return AudioRecheckResult(
                "unavailable",
                "master_capture_route_proof_required",
                evidence,
            )
        if physical_route_before.get("verified") is not True:
            return AudioRecheckResult(
                "unavailable",
                str(
                    physical_route_before.get("reason")
                    or "master_capture_route_physical_readback_required"
                ),
                evidence,
            )
        if not capture_status_is_fresh(capture_status_before):
            return AudioRecheckResult(
                "unavailable",
                "master_capture_real_input_stale_or_unavailable",
                evidence,
            )
        if not capture_identity_matches(capture_status_before):
            return AudioRecheckResult(
                "unavailable",
                "master_capture_device_identity_mismatch",
                evidence,
            )
        if not capture_fresh:
            return AudioRecheckResult(
                "unavailable",
                "master_capture_sample_progression_missing",
                evidence,
            )
        if physical_route_after.get("verified") is not True:
            return AudioRecheckResult(
                "unavailable",
                str(
                    physical_route_after.get("reason")
                    or "master_capture_route_changed_during_audio_measurement"
                ),
                evidence,
            )
        if master_channels != [47, 48] or set(capture_peaks_by_channel) != {47, 48}:
            return AudioRecheckResult(
                "unavailable",
                "stereo_master_capture_channels_required",
                evidence,
            )
        if max(capture_peaks) <= -90.0:
            return AudioRecheckResult(
                "failed",
                "master_audio_signal_missing",
                evidence,
            )
        if true_peak is None or age_sec is None or age_sec > 5.0:
            return AudioRecheckResult(
                "unavailable",
                "master_true_peak_unavailable_or_stale",
                evidence,
            )
        if true_peak > -1.0:
            return AudioRecheckResult(
                "failed",
                "master_true_peak_headroom_insufficient",
                evidence,
            )
        return AudioRecheckResult("passed", "master_true_peak_safe", evidence)

    @staticmethod
    def _parse_supervised_transaction_plan(
        payload: Any,
    ) -> ConsoleTransactionPlan:
        if not isinstance(payload, Mapping):
            raise ValueError("transaction_plan_required")
        raw_operations = payload.get("operations")
        if not isinstance(raw_operations, list) or not raw_operations:
            raise ValueError("transaction_operations_required")
        operations = []
        for raw in raw_operations:
            if not isinstance(raw, Mapping):
                raise ValueError("invalid_transaction_operation")
            operations.append(
                ConsoleTransactionOperation(
                    operation_id=str(raw.get("operation_id") or "").strip(),
                    family_id=str(raw.get("family_id") or "").strip(),
                    write_kind=str(raw.get("write_kind") or "").strip(),
                    target_value=raw.get("target_value"),
                    channel=(
                        raw["channel"]
                        if raw.get("channel") is not None
                        else None
                    ),
                    parameter=str(raw.get("parameter") or "").strip(),
                    metadata=(
                        dict(raw.get("metadata") or {})
                        if isinstance(raw.get("metadata"), Mapping)
                        else {}
                    ),
                )
            )
            if operations[-1].channel is not None and (
                isinstance(operations[-1].channel, bool)
                or not isinstance(operations[-1].channel, int)
                or operations[-1].channel < 1
                or operations[-1].channel > 48
            ):
                raise ValueError("transaction_channel_must_be_integer_1_to_48")
        return ConsoleTransactionPlan(
            transaction_id=str(payload.get("transaction_id") or "").strip(),
            operations=tuple(operations),
            reason=str(payload.get("reason") or "supervised soundcheck transaction"),
        )

    @staticmethod
    def _supervised_transaction_no_send_blocked(
        reason: Any,
        **extra: Any,
    ) -> Dict[str, Any]:
        return {
            "status": "blocked",
            "success": False,
            "reason": str(reason or "supervised_transaction_blocked"),
            "no_console_send_proven": True,
            **extra,
        }

    async def apply_supervised_soundcheck_transaction(
        self,
        plan_payload: Any,
        *,
        approved: bool,
        approval_id: Any,
        idempotency_key: Any,
        proof_run: bool = False,
        approval_expires_at_epoch: Any = None,
    ) -> Dict[str, Any]:
        """Serialize and deduplicate one reviewed multi-parameter transaction."""
        approval_deadline = self._coerce_finite_float(approval_expires_at_epoch)
        parent_soundcheck_run = bool(
            isinstance(plan_payload, Mapping)
            and str(plan_payload.get("run_id") or "").strip()
        )
        if proof_run is not True and parent_soundcheck_run and approval_deadline is None:
            return self._supervised_transaction_no_send_blocked(
                "operator_approval_deadline_required"
            )
        if approval_deadline is not None and approval_deadline <= time.time():
            return self._supervised_transaction_no_send_blocked(
                "operator_approval_expired"
            )
        writer_wait_deadline = (
            min(approval_deadline, time.time() + 2.0)
            if approval_deadline is not None
            else time.time() + 2.0
        )
        async with self._console_write_scope(
            deadline_epoch=writer_wait_deadline,
        ) as writer_acquired:
            if writer_acquired is not True:
                return self._supervised_transaction_no_send_blocked(
                    (
                        "operator_approval_expired"
                        if approval_deadline is not None
                        and approval_deadline <= time.time()
                        else "console_writer_conflict_timeout"
                    )
                )
            if approval_deadline is not None and approval_deadline <= time.time():
                return self._supervised_transaction_no_send_blocked(
                    "operator_approval_expired"
                )
            if isinstance(plan_payload, Mapping) and str(
                plan_payload.get("run_id") or ""
            ).strip():
                refresh_result = await self.refresh_soundcheck_child_preflight(
                    plan_payload
                )
                if refresh_result.get("refreshed") is not True:
                    return self._supervised_transaction_no_send_blocked(
                        refresh_result.get("reason"),
                        soundcheck_child_preflight=refresh_result,
                    )
                if approval_deadline is not None and approval_deadline <= time.time():
                    return self._supervised_transaction_no_send_blocked(
                        "operator_approval_expired",
                        soundcheck_child_preflight=refresh_result,
                    )
                preapply_gate = self.validate_soundcheck_child_for_apply(
                    SimpleNamespace(
                        raw={"supervised_transaction_plan": dict(plan_payload)}
                    )
                )
                if preapply_gate.get("allowed") is not True:
                    return self._supervised_transaction_no_send_blocked(
                        preapply_gate.get("reason"),
                        soundcheck_child_gate=preapply_gate,
                    )
            if approval_deadline is not None and approval_deadline <= time.time():
                return self._supervised_transaction_no_send_blocked(
                    "operator_approval_expired"
                )
            payload = await self._apply_supervised_soundcheck_transaction_unlocked(
                plan_payload,
                approved=approved,
                approval_id=approval_id,
                idempotency_key=idempotency_key,
                proof_run=proof_run,
                approval_deadline_epoch=approval_deadline,
            )
            if isinstance(plan_payload, Mapping) and str(
                plan_payload.get("run_id") or ""
            ).strip():
                readiness = self.record_soundcheck_child_result(
                    SimpleNamespace(
                        raw={"supervised_transaction_plan": dict(plan_payload)}
                    ),
                    payload,
                )
                run_id = str(plan_payload.get("run_id") or "").strip()
                candidate = self._aggregate_current_soundcheck_readiness(
                    run_id,
                    require_final_reconciliation=False,
                )
                if (
                    payload.get("status") == "applied"
                    and isinstance(candidate, Mapping)
                    and candidate.get("mix_readiness")
                    in {"safe_core_ready", "show_mix_ready"}
                ):
                    reconciled = await self.reconcile_soundcheck_readiness(run_id)
                    if isinstance(reconciled, Mapping):
                        readiness = dict(reconciled)
                elif payload.get("status") == "applied":
                    manifest = self._soundcheck_manifests.get(run_id, {})
                    deferred_split_present = any(
                        isinstance(raw, Mapping)
                        and isinstance(raw.get("metadata"), Mapping)
                        and "final_target_value" in raw["metadata"]
                        and raw.get("target_value")
                        != raw["metadata"].get("final_target_value")
                        for plan in manifest.get("plans") or []
                        if isinstance(plan, Mapping)
                        for raw in plan.get("operations") or []
                    )
                    if deferred_split_present:
                        next_preflight = await self.preflight_soundcheck_manifest(
                            manifest
                        )
                        payload["next_soundcheck_replan"] = dict(
                            next_preflight.get("replan") or {}
                        )
                if isinstance(readiness, Mapping):
                    payload["soundcheck_readiness"] = dict(readiness)
            return payload

    async def _apply_supervised_soundcheck_transaction_unlocked(
        self,
        plan_payload: Any,
        *,
        approved: bool,
        approval_id: Any,
        idempotency_key: Any,
        proof_run: bool = False,
        approval_deadline_epoch: float | None = None,
    ) -> Dict[str, Any]:
        """Run one reviewed plan through the canonical physical transaction gate."""
        if proof_run:
            return self._supervised_transaction_no_send_blocked(
                "hardware_proof_requires_local_proof_runner"
            )
        operator_mode = self.get_operator_mode_status()
        if not operator_mode.get("capabilities", {}).get("can_apply_to_console"):
            return self._supervised_transaction_no_send_blocked(
                "operator_mode_blocks_live_write",
                operator_mode=operator_mode,
            )
        if self.connection_mode == "wing" and self._wing_observation_only:
            return self._supervised_transaction_no_send_blocked(
                "wing_observation_only_active",
                observation_only=True,
            )
        if self._manual_write_emergency_stopped:
            return self._supervised_transaction_no_send_blocked(
                "emergency_stop_active"
            )
        try:
            plan = self._parse_supervised_transaction_plan(plan_payload)
        except (TypeError, ValueError) as exc:
            return self._supervised_transaction_no_send_blocked(str(exc))
        if not plan.transaction_id:
            return self._supervised_transaction_no_send_blocked(
                "transaction_id_required"
            )
        soundcheck_run_id = (
            str(plan_payload.get("run_id") or "").strip()
            if isinstance(plan_payload, Mapping)
            else ""
        )
        if soundcheck_run_id:
            active_run_id = self._current_active_soundcheck_run_id()
            if active_run_id and soundcheck_run_id != active_run_id:
                return self._supervised_transaction_no_send_blocked(
                    "soundcheck_run_superseded",
                    run_id=soundcheck_run_id,
                    active_run_id=active_run_id,
                )

        allowed_live_write_kinds = set(
            self._effective_live_write_kinds_for_manual_gate(operator_mode)
        )
        disallowed_write_kinds = sorted(
            {
                operation.write_kind
                for operation in plan.operations
                if operation.write_kind not in allowed_live_write_kinds
            }
        )
        if disallowed_write_kinds:
            return self._supervised_transaction_no_send_blocked(
                "live_write_kind_not_allowed_in_mode",
                disallowed_live_write_kinds=disallowed_write_kinds,
                allowed_live_write_kinds=sorted(allowed_live_write_kinds),
            )
        if not self.mixer_client or not getattr(self.mixer_client, "is_connected", False):
            return self._supervised_transaction_no_send_blocked(
                "mixer_not_connected"
            )
        if not callable(getattr(self.mixer_client, "query_parameter", None)):
            return self._supervised_transaction_no_send_blocked(
                "physical_readback_transport_unavailable"
            )

        key = str(idempotency_key or "").strip()
        if not key:
            return self._supervised_transaction_no_send_blocked(
                "missing_idempotency_key"
            )
        if key in self._supervised_transaction_idempotency_keys:
            return self._supervised_transaction_no_send_blocked(
                "duplicate_idempotency_key",
                idempotency_key=key,
            )
        if plan.transaction_id in self._supervised_transaction_ids:
            return self._supervised_transaction_no_send_blocked(
                "duplicate_transaction_id",
                transaction_id=plan.transaction_id,
                first_idempotency_key=self._supervised_transaction_ids[
                    plan.transaction_id
                ],
            )
        self._supervised_transaction_idempotency_keys[key] = plan.transaction_id
        self._supervised_transaction_ids[plan.transaction_id] = key
        if len(self._supervised_transaction_idempotency_keys) > 1000:
            oldest_key = next(iter(self._supervised_transaction_idempotency_keys))
            oldest_transaction_id = self._supervised_transaction_idempotency_keys.pop(
                oldest_key
            )
            if self._supervised_transaction_ids.get(oldest_transaction_id) == oldest_key:
                self._supervised_transaction_ids.pop(oldest_transaction_id, None)

        contracts = {contract.family_id: contract for contract in FAMILY_CONTRACTS}
        transaction_emergency_generation = self._manual_write_emergency_generation
        initial_soundcheck_manifest = (
            self._soundcheck_manifests.get(soundcheck_run_id, {})
            if soundcheck_run_id
            else {}
        )
        initial_soundcheck_preflight = (
            self._soundcheck_preflight_assessments.get(
                soundcheck_run_id,
                {},
            )
            if soundcheck_run_id
            else {}
        )
        source_coverage = (
            initial_soundcheck_manifest.get("source_coverage")
            if isinstance(initial_soundcheck_manifest, Mapping)
            and isinstance(
                initial_soundcheck_manifest.get("source_coverage"),
                Mapping,
            )
            else {}
        )
        source_balance_deadline_required = bool(
            soundcheck_run_id
            and isinstance(plan_payload, Mapping)
            and self._soundcheck_plan_requires_source_path_proof(
                plan_payload,
                source_coverage,
            )
        )
        initial_source_balance_deadline = (
            (
                initial_soundcheck_preflight.get(
                    "source_balance_deadline_by_transaction"
                )
                or {}
            ).get(plan.transaction_id)
            if isinstance(initial_soundcheck_preflight, Mapping)
            else None
        )
        def write_authority_current() -> bool:
            fence_current = True
            no_soundcheck_blockers = True
            source_analysis_current = True
            no_child_local_blockers = True
            if soundcheck_run_id:
                preflight = self._soundcheck_preflight_assessments.get(
                    soundcheck_run_id,
                    {},
                )
                fence_current = self._soundcheck_console_fence_still_current(
                    preflight.get("console_change_fence")
                    if isinstance(preflight, Mapping)
                    else None,
                    expected_mutations=getattr(
                        self,
                        "_soundcheck_expected_mutation_addresses",
                        {},
                    ).get(soundcheck_run_id, {}),
                )
                no_soundcheck_blockers = not bool(
                    self._soundcheck_global_blockers.get(soundcheck_run_id, [])
                )
                no_child_local_blockers = not bool(
                    (
                        preflight.get("child_local_blockers_by_transaction")
                        or {}
                    ).get(plan.transaction_id, [])
                )
                if source_balance_deadline_required:
                    source_deadline = (
                        preflight.get(
                            "source_balance_deadline_by_transaction"
                        )
                        or {}
                    ).get(plan.transaction_id)
                    source_analysis_current = bool(
                        isinstance(source_deadline, (int, float))
                        and not isinstance(source_deadline, bool)
                        and math.isfinite(float(source_deadline))
                        and isinstance(
                            initial_source_balance_deadline,
                            (int, float),
                        )
                        and not isinstance(
                            initial_source_balance_deadline,
                            bool,
                        )
                        and math.isfinite(
                            float(initial_source_balance_deadline)
                        )
                        and float(source_deadline)
                        == float(initial_source_balance_deadline)
                        and time.time() < float(source_deadline)
                    )
            current_operator_mode = self.get_operator_mode_status()
            current_allowed_kinds = set(
                self._effective_live_write_kinds_for_manual_gate(
                    current_operator_mode
                )
            )
            current_mode_allows = bool(
                current_operator_mode.get("capabilities", {}).get(
                    "can_apply_to_console"
                )
                and all(
                    operation.write_kind in current_allowed_kinds
                    for operation in plan.operations
                )
                and not (
                    self.connection_mode == "wing"
                    and self._wing_observation_only
                )
                and self.mixer_client is not None
                and getattr(self.mixer_client, "is_connected", False) is True
            )
            return bool(
                not self._manual_write_emergency_stopped
                and self._manual_write_emergency_generation
                == transaction_emergency_generation
                and (
                    not soundcheck_run_id
                    or self._current_active_soundcheck_run_id()
                    == soundcheck_run_id
                )
                and fence_current
                and no_soundcheck_blockers
                and no_child_local_blockers
                and source_analysis_current
                and current_mode_allows
                and (
                    approval_deadline_epoch is None
                    or time.time() < approval_deadline_epoch
                )
            )

        adapters = build_wing_supervised_family_adapters(
            self.mixer_client,
            audio_recheck=self._supervised_transaction_audio_recheck,
            write_authority_current=write_authority_current,
        )
        wal_state = {"attempted": False, "prepared": False}

        def persist_transaction_prepare(
            prepared_plan: ConsoleTransactionPlan,
            prepared_context: ConsoleTransactionContext,
            snapshots: tuple[Mapping[str, Any], ...],
        ) -> str:
            wal_state["attempted"] = True
            operations = []
            for operation, snapshot in zip(prepared_plan.operations, snapshots):
                prepared = adapters[operation.family_id].prepare(operation)
                operations.append(
                    {
                        "operation_id": operation.operation_id,
                        "family_id": operation.family_id,
                        "write_kind": operation.write_kind,
                        "channel": operation.channel,
                        "parameter": operation.parameter,
                        "writes": list(prepared.get("writes") or ()),
                        "preimage_values": dict(snapshot.get("values") or {}),
                    }
                )
            path = self._append_manual_write_wal(
                "prepared",
                prepared_plan.transaction_id,
                scope="supervised_transaction",
                approval_id=prepared_context.approval_id,
                idempotency_key=prepared_context.idempotency_key,
                emergency_generation=prepared_context.emergency_generation,
                reason=prepared_plan.reason,
                operations=operations,
            )
            if not path:
                raise RuntimeError("supervised_transaction_wal_prepare_failed")
            wal_state["prepared"] = True
            return path

        coordinator = SupervisedConsoleTransactionCoordinator(
            family_contracts=contracts,
            adapters=adapters,
            emergency_generation_getter=lambda: self._manual_write_emergency_generation,
            emergency_stop=self.activate_system_emergency_stop,
            prepared_callback=persist_transaction_prepare,
        )
        context = ConsoleTransactionContext(
            approved=approved is True,
            approval_id=str(approval_id or "").strip(),
            idempotency_key=str(idempotency_key or "").strip(),
            emergency_generation=self._manual_write_emergency_generation,
            proof_run=False,
            approval_deadline_epoch=approval_deadline_epoch,
        )
        current_write_expectations = {
            str(spec.get("address") or ""): {
                "target_value": spec.get("target_value"),
                "tolerance": float(spec.get("tolerance") or 0.0),
            }
            for operation in plan.operations
            for spec in adapters[operation.family_id].prepare(operation).get(
                "writes",
                (),
            )
            if str(spec.get("address") or "")
        }
        expected_mutations = getattr(
            self,
            "_soundcheck_expected_mutation_addresses",
            None,
        )
        if not isinstance(expected_mutations, dict):
            self._soundcheck_expected_mutation_addresses = {}
            expected_mutations = self._soundcheck_expected_mutation_addresses
        if soundcheck_run_id:
            expected_mutations[soundcheck_run_id] = current_write_expectations
        try:
            result = await coordinator.execute(plan, context)
        finally:
            if soundcheck_run_id:
                expected_mutations.pop(soundcheck_run_id, None)
        payload = result.to_dict()

        def apply_event_may_have_sent(event: Any) -> bool:
            if not isinstance(event, Mapping) or event.get("phase") != "apply":
                return False
            apply_payload = event.get("result")
            if not isinstance(apply_payload, Mapping):
                return True
            if apply_payload.get("skipped") is True:
                return False
            evidence = (
                apply_payload.get("evidence")
                if isinstance(apply_payload.get("evidence"), Mapping)
                else {}
            )
            writes = evidence.get("writes")
            return bool(
                (isinstance(writes, list) and writes)
                or apply_payload.get("success") is True
                or evidence.get("write_state_unknown") is True
            )

        console_send_may_have_occurred = any(
            apply_event_may_have_sent(event)
            for event in payload.get("events") or []
        )
        terminal_phase = "not_prepared"
        if wal_state["attempted"] and not wal_state["prepared"]:
            await self.activate_system_emergency_stop(
                f"supervised_transaction_wal_prepare_failed:{plan.transaction_id}"
            )
            payload.update(
                {
                    "status": "blocked",
                    "reason": "durable_write_journal_unavailable",
                    "emergency_stop_activated": True,
                }
            )
        elif wal_state["prepared"]:
            if result.status == "applied":
                terminal_phase = "committed"
            elif result.status in {"rolled_back", "proof_passed"}:
                terminal_phase = "rolled_back"
            elif not console_send_may_have_occurred:
                terminal_phase = "aborted_before_send"
            else:
                terminal_phase = "manual_review_required"
            terminal_path = self._append_manual_write_wal(
                terminal_phase,
                plan.transaction_id,
                scope="supervised_transaction",
                result_status=result.status,
                result_reason=result.reason,
                applied_operation_ids=list(result.applied_operation_ids),
                rollback_operation_ids=list(result.rollback_operation_ids),
            )
            if not terminal_path:
                terminal_phase = "terminal_record_failed"
                await self.activate_system_emergency_stop(
                    f"supervised_transaction_wal_terminal_failed:{plan.transaction_id}"
                )
                payload.update(
                    {
                        "status": "manual_review_required",
                        "reason": "durable_terminal_record_failed",
                        "emergency_stop_activated": True,
                    }
                )
            elif terminal_phase == "manual_review_required":
                payload["manual_review_required"] = True
        audible_requirements = {
            str(requirement)
            for operation in plan.operations
            for requirement in (
                dict(operation.metadata).get("audible_state_requirements") or []
            )
            if str(requirement).strip()
        }
        audible_state_verified = not audible_requirements
        if audible_requirements and result.status == "applied":
            verified_values: Dict[str, Any] = {}
            for event in payload.get("events", []):
                if not isinstance(event, Mapping) or event.get("phase") != "verify":
                    continue
                verify_result = event.get("result")
                evidence = (
                    verify_result.get("evidence")
                    if isinstance(verify_result, Mapping)
                    else None
                )
                values = (
                    evidence.get("values")
                    if isinstance(evidence, Mapping)
                    else None
                )
                if isinstance(values, Mapping):
                    verified_values.update(values)
            requirement_results = []
            if "eq_enabled_physical_readback" in audible_requirements:
                requirement_results.append(
                    any(
                        str(address).endswith("/eq/on")
                        and value in {1, 1.0, True, "1"}
                        for address, value in verified_values.items()
                    )
                )
            if "compressor_enabled_physical_readback" in audible_requirements:
                requirement_results.append(
                    any(
                        str(address).endswith("/dyn/on")
                        and value in {1, 1.0, True, "1"}
                        for address, value in verified_values.items()
                    )
                )
            audible_state_verified = bool(
                requirement_results and all(requirement_results)
            )
        payload["wal_terminal"] = (
            terminal_phase if wal_state["prepared"] else "not_prepared"
        )
        payload["physical_verified"] = result.status == "applied"
        payload["audio_recheck_passed"] = result.status == "applied"
        payload["audible_state_verified"] = audible_state_verified
        payload["success"] = result.status == "applied"
        if payload.get("status") != "applied":
            payload["success"] = False
        payload["proof_run"] = False
        no_console_send_proven = bool(
            not console_send_may_have_occurred
            and terminal_phase not in {
                "manual_review_required",
                "terminal_record_failed",
            }
            and payload.get("write_state_unknown") is not True
        )
        payload["no_console_send_proven"] = no_console_send_proven
        if (
            result.status == "rolled_back"
            and terminal_phase == "rolled_back"
            and console_send_may_have_occurred
        ):
            payload.update(
                {
                    "physical_state_restored": True,
                    "terminal_after_console_send": True,
                    "fresh_analysis_required": True,
                }
            )
        if no_console_send_proven:
            if self._supervised_transaction_idempotency_keys.get(key) == plan.transaction_id:
                self._supervised_transaction_idempotency_keys.pop(key, None)
            if self._supervised_transaction_ids.get(plan.transaction_id) == key:
                self._supervised_transaction_ids.pop(plan.transaction_id, None)
            payload["supervised_idempotency_released"] = True
        artifact_path = self._record_manual_write_event(
            "supervised_soundcheck_transaction",
            payload,
        )
        if artifact_path:
            payload["artifact_path"] = artifact_path
        return payload

    async def create_snapshot(self, websocket, channels: list = None):
        """Create backup snapshot (for undo). Uses current mixer client."""
        if not self.mixer_client or not self.mixer_client.is_connected:
            await self.send_to_client(websocket, {"type": "snapshot_result", "success": False, "error": "Mixer not connected"})
            return
        try:
            client = self.mixer_client
            ch_list = list(channels) if channels else list(range(1, 41))

            def do_backup():
                out = {"timestamp": datetime.now().isoformat(), "channels": {}}
                for c in ch_list:
                    try:
                        out["channels"][c] = backup_channel(client, c)
                    except Exception as e:
                        out["channels"][c] = {"error": str(e)}
                return out

            loop = asyncio.get_running_loop()
            backup_data = await loop.run_in_executor(None, do_backup)
            channel_results = backup_data.get("channels", {})
            success_channels = [
                channel for channel, payload in channel_results.items()
                if isinstance(payload, dict) and "error" not in payload
            ]
            failed_channels = [
                channel for channel, payload in channel_results.items()
                if not isinstance(payload, dict) or "error" in payload
            ]
            os.makedirs("presets", exist_ok=True)
            path = os.path.join("presets", f"channel_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(backup_data, f, indent=2, ensure_ascii=False)
            if success_channels:
                self._last_snapshot_path = path
            await self.send_to_client(websocket, {
                "type": "snapshot_result",
                "success": bool(success_channels),
                "path": path,
                "success_count": len(success_channels),
                "failed_channels": failed_channels,
                "error": None if success_channels else "Failed to back up any selected channels",
            })
        except Exception as e:
            logger.error(f"Create snapshot error: {e}", exc_info=True)
            await self.send_to_client(websocket, {"type": "snapshot_result", "success": False, "error": str(e)})

    async def restore_snapshot(
        self,
        websocket,
        snapshot_path: str = None,
        *,
        dry_run: Any = None,
        confirm_maintenance: bool = False,
        rollback_snapshot_path: str = None,
        source: str = "manual_ui",
    ):
        """Restore from last snapshot (undo)."""
        if await self._block_quarantined_wing_write_surface(
            websocket,
            surface="restore_snapshot",
            result_type="restore_result",
            request={
                "snapshot_path": snapshot_path,
                "dry_run": dry_run,
                "confirm_maintenance": confirm_maintenance,
                "rollback_snapshot_path": rollback_snapshot_path,
                "source": source,
            },
        ):
            return

        if dry_run is not False or confirm_maintenance is not True:
            await self.send_to_client(
                websocket,
                {
                    "type": "restore_result",
                    "status": "blocked",
                    "success": False,
                    "reason": "maintenance_confirmation_required",
                    "dry_run": dry_run is not False,
                    "source": str(source or "manual_ui"),
                },
            )
            return
        if not rollback_snapshot_path or not os.path.isfile(rollback_snapshot_path):
            await self.send_to_client(
                websocket,
                {
                    "type": "restore_result",
                    "status": "blocked",
                    "success": False,
                    "reason": "rollback_snapshot_required",
                    "rollback_snapshot_path": rollback_snapshot_path,
                    "source": str(source or "manual_ui"),
                },
            )
            return

        path = snapshot_path or self._last_snapshot_path
        if not path or not os.path.isfile(path):
            await self.send_to_client(websocket, {"type": "restore_result", "success": False, "error": "No snapshot file"})
            return
        if not self.mixer_client or not self.mixer_client.is_connected:
            await self.send_to_client(websocket, {"type": "restore_result", "success": False, "error": "Mixer not connected"})
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                backup_data = json.load(f)

            channel_entries = backup_data.get("channels", {})
            restorable_channels = [
                channel for channel, payload in channel_entries.items()
                if isinstance(payload, dict) and "error" not in payload
            ]
            if not restorable_channels:
                await self.send_to_client(websocket, {
                    "type": "restore_result",
                    "success": False,
                    "error": "Snapshot does not contain any restorable channels",
                    "path": path,
                })
                return

            async with self._console_write_scope():
                success = await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: restore_from_backup_using_client(
                        self.mixer_client,
                        path,
                        skip_confirm=True,
                    ),
                )
            await self.send_to_client(websocket, {
                "type": "restore_result",
                "success": bool(success),
                "path": path,
                "restorable_count": len(restorable_channels),
                "error": None if success else "Restore failed",
            })
        except Exception as e:
            logger.error(f"Restore snapshot error: {e}", exc_info=True)
            await self.send_to_client(websocket, {"type": "restore_result", "success": False, "error": str(e)})

    async def start_realtime_fader(self, websocket, device_id: str = None,
                                    channels: List[int] = None,
                                    channel_settings: Dict = None,
                                    channel_mapping: Dict = None,
                                    settings: Dict = None):
        """Start Real-Time Fader mode."""
        channels = channels or []
        channel_settings = channel_settings or {}
        channel_mapping = channel_mapping or {}
        settings = settings or {}
        if not isinstance(settings, dict):
            settings = {}

        logger.info(f"Starting Real-Time Fader: device={device_id}, channels={channels}")
        if await self._block_quarantined_wing_write_surface(
            websocket,
            surface="start_realtime_fader",
            result_type="auto_fader_status",
            request={"device_id": device_id, "channels": channels, "settings": settings},
            extra={"status_type": "blocked", "active": False, "realtime_enabled": False},
        ):
            return

        if not self.mixer_client or not self.mixer_client.is_connected:
            await self.send_to_client(websocket, {
                "type": "auto_fader_status",
                "status_type": "error",
                "active": False,
                "error": "Mixer not connected"
            })
            return

        if device_id is None or not channels:
            await self.send_to_client(websocket, {
                "type": "auto_fader_status",
                "status_type": "error",
                "active": False,
                "error": "Missing required parameters: device_id or channels"
            })
            return

        try:
            # Stop existing Auto Fader if running
            if self.auto_fader_controller:
                self.auto_fader_controller.stop()

            # Create new controller
            self.auto_fader_controller = AutoFaderController(
                mixer_client=self.mixer_client,
                sample_rate=48000,
                config=self.config,
                bleed_service=self.bleed_service
            )

            # Apply settings
            if settings:
                self.auto_fader_controller.update_settings(
                    fader_range_db=settings.get('faderRangeDb'),
                    avg_window_sec=settings.get('avgWindowSec'),
                    sensitivity=settings.get('sensitivity'),
                    attack_ms=settings.get('attackMs'),
                    release_ms=settings.get('releaseMs'),
                    gate_threshold=settings.get('gateThreshold', -50.0)
                )

            # Set up callbacks
            loop = asyncio.get_running_loop()

            def on_status_update(status_data):
                try:
                    # Extract 'type' field and use it as 'status_type'
                    # Create a copy to avoid modifying the original dict
                    status_data_copy = dict(status_data)
                    status_type = status_data_copy.pop('type', 'status_update')
                    loop.call_soon_threadsafe(
                        lambda: asyncio.create_task(self.broadcast({
                            "type": "auto_fader_status",
                            "status_type": status_type,
                            **status_data_copy
                        }))
                    )
                except Exception as e:
                    logger.error(f"Error broadcasting auto fader status: {e}")

            def on_levels_updated(levels_data):
                try:
                    loop.call_soon_threadsafe(
                        lambda: asyncio.create_task(self.broadcast({
                            "type": "auto_fader_status",
                            "status_type": "levels_update",
                            "channels": levels_data
                        }))
                    )
                except Exception as e:
                    logger.error(f"Error broadcasting auto fader levels: {e}")

            self.auto_fader_controller.on_status_update = on_status_update
            self.auto_fader_controller.on_levels_updated = on_levels_updated

            # Convert channel_settings keys to int
            int_channel_settings = {}
            instrument_types_for_bleed = {}
            for ch_str, ch_data in channel_settings.items():
                try:
                    ch_id = int(ch_str)
                    instrument_type = ch_data.get('instrumentType', 'custom')
                    int_channel_settings[ch_id] = {
                        'instrument_type': instrument_type
                    }
                    # Map 'toms' -> 'tom' for BleedDetector INSTRUMENT_BANDS
                    instrument_types_for_bleed[ch_id] = 'tom' if instrument_type == 'toms' else instrument_type
                except (ValueError, TypeError):
                    pass

            # Configure bleed service with instrument types
            if self.bleed_service and instrument_types_for_bleed:
                self.bleed_service.configure(instrument_types_for_bleed)

            # Convert channel_mapping keys to int
            int_channel_mapping = {}
            for ch_str, mixer_ch in channel_mapping.items():
                try:
                    int_channel_mapping[int(ch_str)] = int(mixer_ch)
                except (ValueError, TypeError):
                    pass

            # Start audio capture
            success = self.auto_fader_controller.start(
                device_id=int(device_id),
                channels=[int(ch) for ch in channels],
                channel_settings=int_channel_settings,
                channel_mapping=int_channel_mapping,
                on_status_callback=on_status_update
            )

            if not success:
                self._safe_cleanup_call(
                    self.auto_fader_controller.stop,
                    "Error stopping Auto Fader after failed audio capture start",
                )
                self.auto_fader_controller = None
                await self.send_to_client(websocket, {
                    "type": "auto_fader_status",
                    "status_type": "error",
                    "active": False,
                    "error": "Failed to start audio capture"
                })
                return

            # Start real-time fader control
            success = self.auto_fader_controller.start_realtime_fader()

            if success:
                await self.broadcast({
                    "type": "auto_fader_status",
                    "status_type": "realtime_fader_started",
                    "active": True,
                    "realtime_enabled": True,
                    "mode": "realtime"
                })
            else:
                self._safe_cleanup_call(
                    self.auto_fader_controller.stop,
                    "Error stopping Auto Fader after realtime start failure",
                )
                self.auto_fader_controller = None
                await self.send_to_client(websocket, {
                    "type": "auto_fader_status",
                    "status_type": "error",
                    "error": "Failed to start real-time fader"
                })

        except asyncio.CancelledError:
            if self.auto_fader_controller:
                self._safe_cleanup_call(
                    self.auto_fader_controller.stop,
                    "Error stopping Auto Fader after cancellation",
                )
                self.auto_fader_controller = None
            raise
        except Exception as e:
            logger.error(f"Error starting Real-Time Fader: {e}", exc_info=True)
            if self.auto_fader_controller:
                self._safe_cleanup_call(
                    self.auto_fader_controller.stop,
                    "Error stopping Auto Fader after startup exception",
                )
                self.auto_fader_controller = None
            await self.send_to_client(websocket, {
                "type": "auto_fader_status",
                "status_type": "error",
                "active": False,
                "error": str(e)
            })

    async def stop_realtime_fader(self, websocket):
        """Stop Real-Time Fader mode."""
        logger.info("Stopping Real-Time Fader")

        try:
            if not self.auto_fader_controller:
                await self.send_to_client(websocket, {
                    "type": "auto_fader_status",
                    "status_type": "error",
                    "error": "Auto Fader controller not initialized"
                })
                return

            self.auto_fader_controller.stop_realtime_fader()

            await self.broadcast({
                "type": "auto_fader_status",
                "status_type": "realtime_fader_stopped",
                "active": False,
                "realtime_enabled": False
            })

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Error stopping Real-Time Fader: {e}", exc_info=True)
            await self.send_to_client(websocket, {
                "type": "auto_fader_status",
                "status_type": "error",
                "error": str(e)
            })

    async def start_auto_balance(self, websocket, device_id: str = None,
                                  channels: List[int] = None,
                                  channel_settings: Dict = None,
                                  channel_mapping: Dict = None,
                                  duration: float = 15,
                                  bleed_threshold: float = -50):
        """Start Auto Balance collection (LEARN phase)."""
        channels = channels or []
        channel_settings = channel_settings or {}
        channel_mapping = channel_mapping or {}

        logger.info(f"Starting Auto Balance LEARN: device={device_id}, channels={channels}, "
                    f"duration={duration}s, bleed_threshold={bleed_threshold} LUFS")

        if not self.mixer_client or not self.mixer_client.is_connected:
            await self.send_to_client(websocket, {
                "type": "auto_fader_status",
                "status_type": "error",
                "active": False,
                "error": "Mixer not connected"
            })
            return

        if device_id is None or not channels:
            await self.send_to_client(websocket, {
                "type": "auto_fader_status",
                "status_type": "error",
                "active": False,
                "error": "Missing required parameters: device_id or channels"
            })
            return

        try:
            # Stop existing Auto Fader if running
            if self.auto_fader_controller:
                self.auto_fader_controller.stop()

            # Create new controller
            self.auto_fader_controller = AutoFaderController(
                mixer_client=self.mixer_client,
                sample_rate=48000,
                config=self.config,
                bleed_service=self.bleed_service
            )

            # Set up callbacks
            loop = asyncio.get_running_loop()

            def on_status_update(status_data):
                try:
                    # Extract 'type' field and use it as 'status_type'
                    # Create a copy to avoid modifying the original dict
                    status_data_copy = dict(status_data)
                    status_type = status_data_copy.pop('type', 'status_update')
                    loop.call_soon_threadsafe(
                        lambda: asyncio.create_task(self.broadcast({
                            "type": "auto_fader_status",
                            "status_type": status_type,
                            **status_data_copy
                        }))
                    )
                except Exception as e:
                    logger.error(f"Error broadcasting auto balance status: {e}")

            def on_levels_updated(levels_data):
                try:
                    loop.call_soon_threadsafe(
                        lambda: asyncio.create_task(self.broadcast({
                            "type": "auto_fader_status",
                            "status_type": "levels_update",
                            "channels": levels_data
                        }))
                    )
                except Exception as e:
                    logger.error(f"Error broadcasting auto balance levels: {e}")

            self.auto_fader_controller.on_status_update = on_status_update
            self.auto_fader_controller.on_levels_updated = on_levels_updated

            # Convert channel_settings keys to int
            # Support both 'instrumentType' (from AutoFaderTab) and 'preset' (from AutoSoundcheckTab)
            int_channel_settings = {}
            instrument_types_for_bleed = {}
            for ch_str, ch_data in channel_settings.items():
                try:
                    ch_id = int(ch_str)
                    # Try instrumentType first (from AutoFaderTab), then preset (from AutoSoundcheckTab), default to 'custom'
                    instrument_type = ch_data.get('instrumentType') or ch_data.get('preset', 'custom')
                    int_channel_settings[ch_id] = {
                        'instrument_type': instrument_type
                    }
                    # Map 'toms' -> 'tom' for BleedDetector INSTRUMENT_BANDS
                    instrument_types_for_bleed[ch_id] = 'tom' if instrument_type == 'toms' else instrument_type
                except (ValueError, TypeError):
                    pass

            # Configure bleed service with instrument types
            if self.bleed_service and instrument_types_for_bleed:
                self.bleed_service.configure(instrument_types_for_bleed)

            logger.info(f"Processed channel_settings for Auto Balance: {int_channel_settings}")

            # Convert channel_mapping keys to int
            int_channel_mapping = {}
            for ch_str, mixer_ch in channel_mapping.items():
                try:
                    int_channel_mapping[int(ch_str)] = int(mixer_ch)
                except (ValueError, TypeError):
                    pass

            # Start audio capture
            success = self.auto_fader_controller.start(
                device_id=int(device_id),
                channels=[int(ch) for ch in channels],
                channel_settings=int_channel_settings,
                channel_mapping=int_channel_mapping,
                on_status_callback=on_status_update
            )

            if not success:
                self._safe_cleanup_call(
                    self.auto_fader_controller.stop,
                    "Error stopping Auto Balance after failed audio capture start",
                )
                self.auto_fader_controller = None
                await self.send_to_client(websocket, {
                    "type": "auto_fader_status",
                    "status_type": "error",
                    "active": False,
                    "error": "Failed to start audio capture"
                })
                return

            # Start auto balance collection (uses same algorithms and methods as Auto Balance)
            # All settings (fader_range_db, avg_window_sec, sensitivity, etc.) are loaded from config
            # in AutoFaderController.__init__()
            success = self.auto_fader_controller.start_auto_balance(duration=duration, bleed_threshold=bleed_threshold)

            if success:
                await self.broadcast({
                    "type": "auto_fader_status",
                    "status_type": "auto_balance_started",
                    "active": True,
                    "collecting": True,
                    "duration": duration,
                    "mode": "static"
                })
            else:
                self._safe_cleanup_call(
                    self.auto_fader_controller.stop,
                    "Error stopping Auto Balance after learn start failure",
                )
                self.auto_fader_controller = None
                await self.send_to_client(websocket, {
                    "type": "auto_fader_status",
                    "status_type": "error",
                    "error": "Failed to start auto balance collection"
                })

        except asyncio.CancelledError:
            if self.auto_fader_controller:
                self._safe_cleanup_call(
                    self.auto_fader_controller.stop,
                    "Error stopping Auto Balance after cancellation",
                )
                self.auto_fader_controller = None
            raise
        except Exception as e:
            logger.error(f"Error starting Auto Balance: {e}", exc_info=True)
            if self.auto_fader_controller:
                self._safe_cleanup_call(
                    self.auto_fader_controller.stop,
                    "Error stopping Auto Balance after startup exception",
                )
                self.auto_fader_controller = None
            await self.send_to_client(websocket, {
                "type": "auto_fader_status",
                "status_type": "error",
                "active": False,
                "error": str(e)
            })

    async def apply_auto_balance(self, websocket):
        """Apply Auto Balance results to mixer."""
        logger.info("Applying Auto Balance")
        if await self._block_quarantined_wing_write_surface(
            websocket,
            surface="apply_auto_balance",
            result_type="auto_fader_status",
            extra={"status_type": "blocked", "active": False},
        ):
            return

        try:
            if not self.auto_fader_controller:
                await self.send_to_client(websocket, {
                    "type": "auto_fader_status",
                    "status_type": "error",
                    "error": "Auto Fader controller not initialized"
                })
                return

            success = self.auto_fader_controller.apply_auto_balance()

            if success:
                status = self.auto_fader_controller.get_status()
                auto_balance_result = status.get('auto_balance_result') or {}
                await self.broadcast({
                    "type": "auto_fader_status",
                    "status_type": "auto_balance_applied",
                    "applied_count": len(auto_balance_result),
                    "total_count": len(auto_balance_result),
                    "pass_number": status.get('auto_balance_pass', 1)
                })
            else:
                await self.send_to_client(websocket, {
                    "type": "auto_fader_status",
                    "status_type": "error",
                    "error": "Failed to apply auto balance"
                })

        except Exception as e:
            logger.error(f"Error applying Auto Balance: {e}", exc_info=True)
            await self.send_to_client(websocket, {
                "type": "auto_fader_status",
                "status_type": "error",
                "error": str(e)
            })

    async def cancel_auto_balance(self, websocket):
        """Cancel Auto Balance collection."""
        logger.info("Cancelling Auto Balance")

        try:
            if not self.auto_fader_controller:
                await self.send_to_client(websocket, {
                    "type": "auto_fader_status",
                    "status_type": "error",
                    "error": "Auto Fader controller not initialized"
                })
                return

            self.auto_fader_controller.cancel_auto_balance()

            await self.broadcast({
                "type": "auto_fader_status",
                "status_type": "auto_balance_cancelled"
            })

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Error cancelling Auto Balance: {e}", exc_info=True)
            await self.send_to_client(websocket, {
                "type": "auto_fader_status",
                "status_type": "error",
                "error": str(e)
            })

    async def lock_auto_balance_channel(self, websocket, channel):
        """Lock a channel so it won't be changed on subsequent Auto Balance passes."""
        try:
            if not self.auto_fader_controller:
                await self.send_to_client(websocket, {
                    "type": "auto_fader_status",
                    "status_type": "error",
                    "error": "Auto Fader controller not initialized"
                })
                return

            ch = int(channel) if channel is not None else None
            if ch is None:
                await self.send_to_client(websocket, {
                    "type": "auto_fader_status",
                    "status_type": "error",
                    "error": "Missing channel"
                })
                return

            success = self.auto_fader_controller.lock_channel(ch)
            if success:
                await self.broadcast({
                    "type": "auto_fader_status",
                    "status_type": "channel_lock_changed",
                    "channel": ch,
                    "locked": True
                })
            else:
                await self.send_to_client(websocket, {
                    "type": "auto_fader_status",
                    "status_type": "error",
                    "error": f"Channel {ch} not found"
                })
        except asyncio.CancelledError:
            raise
        except (TypeError, ValueError) as e:
            logger.error(f"Error locking channel: {e}", exc_info=True)
            await self.send_to_client(websocket, {
                "type": "auto_fader_status",
                "status_type": "error",
                "error": f"Invalid channel: {channel}"
            })
        except Exception as e:
            logger.error(f"Error locking channel: {e}", exc_info=True)
            await self.send_to_client(websocket, {
                "type": "auto_fader_status",
                "status_type": "error",
                "error": str(e)
            })

    async def unlock_auto_balance_channel(self, websocket, channel):
        """Unlock a channel so it can be changed on subsequent Auto Balance passes."""
        try:
            if not self.auto_fader_controller:
                await self.send_to_client(websocket, {
                    "type": "auto_fader_status",
                    "status_type": "error",
                    "error": "Auto Fader controller not initialized"
                })
                return

            ch = int(channel) if channel is not None else None
            if ch is None:
                await self.send_to_client(websocket, {
                    "type": "auto_fader_status",
                    "status_type": "error",
                    "error": "Missing channel"
                })
                return

            success = self.auto_fader_controller.unlock_channel(ch)
            if success:
                await self.broadcast({
                    "type": "auto_fader_status",
                    "status_type": "channel_lock_changed",
                    "channel": ch,
                    "locked": False
                })
            else:
                await self.send_to_client(websocket, {
                    "type": "auto_fader_status",
                    "status_type": "error",
                    "error": f"Channel {ch} not found"
                })
        except asyncio.CancelledError:
            raise
        except (TypeError, ValueError) as e:
            logger.error(f"Error unlocking channel: {e}", exc_info=True)
            await self.send_to_client(websocket, {
                "type": "auto_fader_status",
                "status_type": "error",
                "error": f"Invalid channel: {channel}"
            })
        except Exception as e:
            logger.error(f"Error unlocking channel: {e}", exc_info=True)
            await self.send_to_client(websocket, {
                "type": "auto_fader_status",
                "status_type": "error",
                "error": str(e)
            })

    async def set_auto_fader_profile(self, websocket, profile: str):
        """Set Auto Fader genre profile."""
        logger.info(f"Setting Auto Fader profile: {profile}")

        try:
            if not self.auto_fader_controller:
                await self.send_to_client(websocket, {
                    "type": "auto_fader_status",
                    "status_type": "error",
                    "error": "Auto Fader controller not initialized"
                })
                return

            self.auto_fader_controller.set_profile(profile)

            await self.send_to_client(websocket, {
                "type": "auto_fader_status",
                "message": f"Profile set to: {profile}"
            })

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Error setting Auto Fader profile: {e}", exc_info=True)
            await self.send_to_client(websocket, {
                "type": "auto_fader_status",
                "status_type": "error",
                "error": str(e)
            })

    async def update_auto_fader_settings(self, websocket, settings: Dict):
        """Update Auto Fader settings."""
        logger.info(f"Updating Auto Fader settings: {settings}")

        try:
            if not self.auto_fader_controller:
                await self.send_to_client(websocket, {
                    "type": "auto_fader_status",
                    "status_type": "error",
                    "error": "Auto Fader controller not initialized"
                })
                return

            self.auto_fader_controller.update_settings(
                target_lufs=settings.get('targetLufs'),
                max_adjustment_db=settings.get('maxAdjustmentDb'),
                ratio=settings.get('ratio'),
                attack_ms=settings.get('attackMs'),
                release_ms=settings.get('releaseMs'),
                hold_ms=settings.get('holdMs'),
            )

            await self.send_to_client(websocket, {
                "type": "auto_fader_status",
                "message": "Settings updated"
            })

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Error updating Auto Fader settings: {e}", exc_info=True)
            await self.send_to_client(websocket, {
                "type": "auto_fader_status",
                "status_type": "error",
                "error": str(e)
            })

    async def save_auto_fader_defaults(self, websocket, settings: Dict):
        """Save Auto Fader settings as user defaults."""
        try:
            self._save_user_config('auto_fader', settings)
            await self.send_to_client(websocket, {
                "type": "auto_fader_defaults_saved",
                "success": True
            })
            logger.info(f"Auto Fader defaults saved: {settings}")
        except Exception as e:
            logger.error(f"Error saving Auto Fader defaults: {e}", exc_info=True)
            await self.send_to_client(websocket, {
                "type": "auto_fader_defaults_saved",
                "success": False,
                "error": str(e)
            })

    async def load_auto_fader_defaults(self, websocket):
        """Load saved Auto Fader user defaults."""
        try:
            user_config = self._load_user_config()
            saved_settings = user_config.get('auto_fader', None)
            await self.send_to_client(websocket, {
                "type": "auto_fader_defaults_loaded",
                "settings": saved_settings
            })
            logger.info(f"Auto Fader defaults loaded: {saved_settings}")
        except Exception as e:
            logger.error(f"Error loading Auto Fader defaults: {e}", exc_info=True)
            await self.send_to_client(websocket, {
                "type": "auto_fader_defaults_loaded",
                "settings": None,
                "error": str(e)
            })

    async def save_all_settings(self, websocket, settings: Dict):
        """Save all application settings to user config."""
        try:
            # Save each section individually
            for section, section_settings in settings.items():
                self._save_user_config(section, section_settings)
            await self.send_to_client(websocket, {
                "type": "all_settings_saved",
                "success": True
            })
            logger.info(f"All settings saved: sections={list(settings.keys())}")
        except Exception as e:
            logger.error(f"Error saving all settings: {e}", exc_info=True)
            await self.send_to_client(websocket, {
                "type": "all_settings_saved",
                "success": False,
                "error": str(e)
            })

    async def load_all_settings(self, websocket):
        """Load all saved user settings."""
        try:
            user_config = self._load_user_config()
            await self.send_to_client(websocket, {
                "type": "all_settings_loaded",
                "settings": user_config
            })
            logger.info(f"All settings loaded: sections={list(user_config.keys())}")
        except Exception as e:
            logger.error(f"Error loading all settings: {e}", exc_info=True)
            await self.send_to_client(websocket, {
                "type": "all_settings_loaded",
                "settings": {},
                "error": str(e)
            })

    # ========== Auto Soundcheck Methods ==========

    async def reset_all_functions_to_defaults(self, websocket, channels: List[int]):
        """Reset all functions to default state for selected channels."""
        logger.info(f"Resetting all functions to defaults for channels: {channels}")

        if not channels or len(channels) == 0:
            logger.warning("No channels provided for reset")
            return

        try:
            # Stop all active controllers (non-blocking)
            logger.info("Stopping all active controllers...")
            stop_tasks = []

            if self.gain_staging:
                async def stop_gain_staging():
                    try:
                        self.gain_staging.stop()
                        logger.info("Gain staging stopped")
                    except Exception as e:
                        logger.warning(f"Error stopping gain staging: {e}")
                stop_tasks.append(asyncio.create_task(stop_gain_staging()))

            if self.phase_alignment_controller:
                async def stop_phase_alignment():
                    try:
                        await self._cancellation_safe_thread_call(
                            self.phase_alignment_controller.stop_analysis
                        )
                        logger.info("Phase alignment stopped")
                    except Exception as e:
                        logger.warning(f"Error stopping phase alignment: {e}")
                stop_tasks.append(asyncio.create_task(stop_phase_alignment()))

            if self.multi_channel_auto_eq_controller:
                async def stop_multi_eq():
                    try:
                        self.multi_channel_auto_eq_controller.stop_all()
                        logger.info("Multi-channel auto EQ stopped")
                    except Exception as e:
                        logger.warning(f"Error stopping multi-channel auto EQ: {e}")
                stop_tasks.append(asyncio.create_task(stop_multi_eq()))

            if self.auto_fader_controller:
                async def stop_auto_fader():
                    try:
                        self.auto_fader_controller.stop()
                        logger.info("Auto fader stopped")
                    except Exception as e:
                        logger.warning(f"Error stopping auto fader: {e}")
                stop_tasks.append(asyncio.create_task(stop_auto_fader()))

            # Wait for all stops with timeout
            if stop_tasks:
                try:
                    await asyncio.wait_for(asyncio.gather(*stop_tasks, return_exceptions=True), timeout=2.0)
                except asyncio.TimeoutError:
                    logger.warning("Some controllers did not stop in time, continuing...")

            await asyncio.sleep(0.2)  # Small delay after stopping

            # Reset TRIM to 0dB (with timeout and error handling)
            logger.info("Resetting TRIM to 0dB...")
            try:
                await asyncio.wait_for(self.reset_trim(websocket, channels), timeout=15.0)
                logger.info("TRIM reset completed")
            except asyncio.TimeoutError:
                logger.warning("TRIM reset timed out after 15s, continuing...")
            except Exception as e:
                logger.error(f"Error resetting TRIM: {e}", exc_info=True)
            await asyncio.sleep(0.3)

            # Reset all faders to 0dB (all 40 channels)
            logger.info("Resetting all faders to 0dB...")
            try:
                if self.mixer_client and self.mixer_client.is_connected:
                    reset_count = 0
                    errors = []
                    # Reset all 40 channels
                    for ch in range(1, 41):
                        try:
                            self.mixer_client.set_channel_fader(ch, 0.0)
                            reset_count += 1
                            await asyncio.sleep(0.02)  # Small delay between channels
                        except Exception as e:
                            errors.append(f"Channel {ch}: {str(e)}")
                            logger.warning(f"Error resetting fader for channel {ch}: {e}")

                    logger.info(f"Faders reset completed: {reset_count}/40 channels")
                    if errors:
                        logger.warning(f"Fader reset errors: {len(errors)} channels failed")
                else:
                    logger.warning("Mixer not connected, skipping fader reset")
            except Exception as e:
                logger.error(f"Error resetting faders: {e}", exc_info=True)
            await asyncio.sleep(0.3)

            # Reset EQ to flat (with timeout)
            logger.info("Resetting EQ to flat...")
            try:
                await asyncio.wait_for(self.reset_all_eq(websocket, {"channels": channels}), timeout=15.0)
                logger.info("EQ reset completed")
            except asyncio.TimeoutError:
                logger.warning("EQ reset timed out after 15s, continuing...")
            except Exception as e:
                logger.error(f"Error resetting EQ: {e}", exc_info=True)
            await asyncio.sleep(0.3)

            # Reset phase/delay (with timeout)
            logger.info("Resetting phase/delay...")
            try:
                await asyncio.wait_for(self.reset_phase_delay(websocket, channels), timeout=15.0)
                logger.info("Phase/delay reset completed")
            except asyncio.TimeoutError:
                logger.warning("Phase/delay reset timed out after 15s, continuing...")
            except Exception as e:
                logger.error(f"Error resetting phase/delay: {e}", exc_info=True)
            await asyncio.sleep(0.3)

            logger.info("All functions reset to defaults successfully")

        except Exception as e:
            logger.error(f"Error resetting functions to defaults: {e}", exc_info=True)
            import traceback
            logger.error(traceback.format_exc())
            # Don't raise - continue with cycle even if reset fails

    async def start_auto_soundcheck(self, websocket, device_id: str = None, channels: list = None,
                                    channel_settings: dict = None, channel_mapping: dict = None,
                                    timings: dict = None, observe_only: bool = False):
        """Start automatic soundcheck cycle."""
        logger.info("=" * 60)
        logger.info("RECEIVED start_auto_soundcheck MESSAGE")
        logger.info(f"device_id: {device_id}")
        logger.info(f"channels: {channels}")
        logger.info(f"timings: {timings}")
        logger.info(f"observe_only: {observe_only}")
        logger.info("=" * 60)
        if await self._block_quarantined_wing_write_surface(
            websocket,
            surface="start_auto_soundcheck",
            result_type="auto_soundcheck_status",
            request={"device_id": device_id, "channels": channels or [], "timings": timings or {}},
            extra={"is_running": False},
        ):
            return

        if self.auto_soundcheck_running:
            logger.warning("Auto soundcheck is already running")
            await self.send_to_client(websocket, {
                "type": "auto_soundcheck_status",
                "is_running": True,
                "error": "Auto soundcheck is already running"
            })
            return

        # Normalize timings - frontend sends gain_staging, backend config uses gain_staging_duration
        if not timings:
            config_timings = self.config.get("automation", {}).get("auto_soundcheck", {
                "gain_staging_duration": 30,
                "phase_alignment_duration": 30,
                "auto_eq_duration": 30,
                "auto_fader_duration": 30
            })
            # Convert config format to internal format
            timings = {
                "gain_staging": config_timings.get("gain_staging_duration", 30),
                "phase_alignment": config_timings.get("phase_alignment_duration", 30),
                "auto_eq": config_timings.get("auto_eq_duration", 30),
                "auto_fader": config_timings.get("auto_fader_duration", 30)
            }
        else:
            # Normalize timings from frontend (may use either format)
            normalized_timings = {}
            for key in ["gain_staging", "phase_alignment", "auto_eq", "auto_fader"]:
                # Try direct key first
                if key in timings:
                    normalized_timings[key] = timings[key]
                # Try with _duration suffix
                elif f"{key}_duration" in timings:
                    normalized_timings[key] = timings[f"{key}_duration"]
                else:
                    # Default value
                    normalized_timings[key] = 30
            timings = normalized_timings

        logger.info(f"Normalized timings: {timings}")

        if not device_id:
            logger.error("Missing device_id")
            await self.send_to_client(websocket, {
                "type": "auto_soundcheck_status",
                "is_running": False,
                "error": "Missing device_id"
            })
            return

        if not channels or len(channels) == 0:
            logger.error("No channels specified")
            await self.send_to_client(websocket, {
                "type": "auto_soundcheck_status",
                "is_running": False,
                "error": "No channels specified"
            })
            return

        self.auto_soundcheck_running = True
        self.auto_soundcheck_observe_only = observe_only
        self.auto_soundcheck_websocket = websocket

        # Send initial status immediately before starting task
        await self._send_soundcheck_status(websocket, None, 0, 0, "Soundcheck cycle starting...")
        await asyncio.sleep(0.05)  # Small delay to ensure message is sent

        # Start the cycle in a background task
        logger.info("Starting auto soundcheck cycle task...")
        try:
            self.auto_soundcheck_task = asyncio.create_task(
                self._run_auto_soundcheck_cycle(websocket, device_id, channels, channel_settings,
                                               channel_mapping, timings, observe_only=observe_only)
            )
            logger.info("Auto soundcheck cycle task created successfully")

            # Send status that cycle has started
            await self._send_soundcheck_status(websocket, None, 5, 0, "Soundcheck cycle initialized, starting reset...")

        except Exception as e:
            logger.error(f"Error creating auto soundcheck task: {e}", exc_info=True)
            import traceback
            logger.error(traceback.format_exc())
            self.auto_soundcheck_running = False
            self.auto_soundcheck_observe_only = False
            await self.send_to_client(websocket, {
                "type": "auto_soundcheck_status",
                "is_running": False,
                "error": f"Failed to start cycle: {str(e)}"
            })

    async def stop_auto_soundcheck(self, websocket):
        """Stop the auto soundcheck cycle."""
        if not self.auto_soundcheck_running:
            await self.send_to_client(websocket, {
                "type": "auto_soundcheck_status",
                "is_running": False,
                "message": "Auto soundcheck is not running"
            })
            return

        logger.info("Stopping auto soundcheck cycle...")
        self.auto_soundcheck_running = False
        self.auto_soundcheck_observe_only = False

        # Cancel the task
        if self.auto_soundcheck_task:
            self.auto_soundcheck_task.cancel()
            try:
                await self.auto_soundcheck_task
            except asyncio.CancelledError:
                pass
            self.auto_soundcheck_task = None

        # Stop all active controllers
        if self.gain_staging:
            self._safe_cleanup_call(
                self.gain_staging.stop,
                "Error stopping gain staging during auto soundcheck shutdown",
            )

        if self.phase_alignment_controller:
            self._safe_cleanup_call(
                self.phase_alignment_controller.stop_analysis,
                "Error stopping phase alignment during auto soundcheck shutdown",
            )

        if self.multi_channel_auto_eq_controller:
            self._safe_cleanup_call(
                self.multi_channel_auto_eq_controller.stop_all,
                "Error stopping multi-channel Auto-EQ during auto soundcheck shutdown",
            )

        if self.auto_fader_controller:
            self._safe_cleanup_call(
                self.auto_fader_controller.stop,
                "Error stopping auto fader during auto soundcheck shutdown",
            )

        await self.send_to_client(websocket, {
            "type": "auto_soundcheck_status",
            "is_running": False,
            "message": "Auto soundcheck stopped"
        })

    # ========== Auto Compressor ==========

    def _blocked_compressor_reset_result(self, mixer_ch: int) -> Dict[str, Any]:
        return {
            "status": "blocked",
            "success": False,
            "applied": False,
            "blocked": True,
            "surface": "auto_compressor_reset",
            "reason": "auto_compressor_reset_direct_apply_quarantined",
            "channel": mixer_ch,
            "write_policy": "proposal_only_quarantined",
            "live_write_allowed": False,
            "can_apply_live": False,
            "mutates_mixer": False,
            "blocked_calls": [
                {
                    "cmd": "set_compressor_on",
                    "channel": mixer_ch,
                    "on": 0,
                    "semantic": "reset_compressor_disable",
                },
                {
                    "cmd": "set_compressor",
                    "channel": mixer_ch,
                    "threshold": -10.0,
                    "ratio": "3.0",
                    "attack": 10.0,
                    "release": 100.0,
                    "gain": 0.0,
                    "mix": 100.0,
                    "knee": 0,
                    "semantic": "reset_compressor_defaults",
                },
            ],
        }

    def _reset_compressor_on_channel(self, mixer_ch: int):
        """Return blocked reset evidence instead of mutating compressor state."""
        result = self._blocked_compressor_reset_result(int(mixer_ch))
        logger.warning(
            "Blocked quarantined Auto Compressor reset helper",
            extra={
                "event": "auto_compressor_reset_blocked",
                "surface": result["surface"],
                "channel": result["channel"],
            },
        )
        return result

    async def start_auto_compressor(self, websocket, device_id, channels: list, channel_mapping: dict, channel_names: dict = None):
        """Start Auto Compressor: open audio capture and keep reset writes as blocked evidence."""
        if await self._block_quarantined_wing_write_surface(
            websocket,
            surface="start_auto_compressor",
            result_type="auto_compressor_status",
            request={
                "device_id": device_id,
                "channels": channels or [],
                "channel_mapping": channel_mapping or {},
            },
            extra={"active": False},
        ):
            return

        if not self.mixer_client or not self.mixer_client.is_connected:
            await self.send_to_client(websocket, {"type": "auto_compressor_status", "error": "Mixer not connected"})
            return
        if not device_id and device_id != 0:
            await self.send_to_client(websocket, {"type": "auto_compressor_status", "error": "Audio device required"})
            return
        channels = [int(c) for c in (channels or [])]
        channel_mapping = {int(k): int(v) for k, v in (channel_mapping or {}).items()}
        channel_names = {int(k): str(v) for k, v in (channel_names or {}).items()}
        if self.auto_compressor_controller:
            self.auto_compressor_controller.stop()
        # Preserve reset intent as blocked evidence; compressor live writes remain quarantined.
        mixer_channels = list({channel_mapping.get(ac, ac) for ac in channels})
        loop = asyncio.get_running_loop()
        for ch in mixer_channels:
            await loop.run_in_executor(None, self._reset_compressor_on_channel, ch)
        cfg = self.config.get("automation", {}).get("auto_compressor", {})
        duration = cfg.get("soundcheck_duration_per_channel", 7.0)
        target_gr = cfg.get("target_gr_db", 6.0)
        self.auto_compressor_controller = AutoCompressorController(
            self.mixer_client, sample_rate=48000,
            soundcheck_duration_per_channel=duration, target_gr_db=target_gr,
            bleed_service=self.bleed_service)
        loop = asyncio.get_running_loop()
        def on_status(data):
            asyncio.run_coroutine_threadsafe(
                self.broadcast({"type": "auto_compressor_status", **data}), loop)
        self.auto_compressor_controller.on_status = on_status
        ok = self.auto_compressor_controller.start(int(device_id), channels, channel_mapping, channel_names)
        if not ok:
            self.auto_compressor_controller = None
            await self.send_to_client(websocket, {"type": "auto_compressor_status", "error": "Failed to start audio capture"})
            return
        await self.send_to_client(websocket, {"type": "auto_compressor_status", "active": True, "message": "Auto Compressor started (post-fader)"})

    async def stop_auto_compressor(self, websocket):
        """Stop Auto Compressor and release audio."""
        if self.auto_compressor_controller:
            self.auto_compressor_controller.stop()
            self.auto_compressor_controller = None
        await self.broadcast({"type": "auto_compressor_status", "active": False, "message": "Stopped"})

    async def get_auto_compressor_status(self, websocket, request_id=None):
        """Return current Auto Compressor status."""
        if not self.auto_compressor_controller:
            await self.send_to_client(websocket, {"type": "auto_compressor_status", "request_id": request_id, "active": False})
            return
        await self.send_to_client(websocket, {"type": "auto_compressor_status", **self.auto_compressor_controller.get_status()})

    async def get_auto_gate_status(self, websocket, request_id=None):
        """Return Auto Gate status."""
        await self.send_to_client(websocket, {
            "type": "auto_gate_status",
            "request_id": request_id,
            "active": False,
            "message": "Auto Gate not implemented"
        })

    async def get_auto_panner_status(self, websocket, request_id=None):
        """Return Auto Panner status."""
        await self.send_to_client(websocket, {
            "type": "auto_panner_status",
            "request_id": request_id,
            "active": False,
            "message": "Auto Panner not implemented"
        })

    async def get_auto_reverb_status(self, websocket, request_id=None):
        """Return Auto Reverb status."""
        await self.send_to_client(websocket, {
            "type": "auto_reverb_status",
            "request_id": request_id,
            "active": False,
            "message": "Auto Reverb not implemented"
        })

    async def get_auto_effects_status(self, websocket, request_id=None):
        """Return Auto Effects status."""
        await self.send_to_client(websocket, {
            "type": "auto_effects_status",
            "request_id": request_id,
            "active": False,
            "message": "Auto Effects not implemented"
        })

    async def get_cross_adaptive_eq_status(self, websocket, request_id=None):
        """Return Cross-Adaptive EQ status."""
        await self.send_to_client(websocket, {
            "type": "cross_adaptive_eq_status",
            "request_id": request_id,
            "active": False,
            "message": "Cross-Adaptive EQ not implemented"
        })

    async def start_auto_compressor_soundcheck(
        self, websocket,
        genre: str = "unknown",
        style: str = "live",
        genre_factor: float = 1.0,
        mix_density_factor: float = 1.0,
        bpm=None
    ):
        """Run soundcheck: record each channel, analyze, adapt, apply."""
        if await self._block_quarantined_wing_write_surface(
            websocket,
            surface="start_auto_compressor_soundcheck",
            result_type="auto_compressor_status",
        ):
            return

        if not self.auto_compressor_controller or not self.auto_compressor_controller.is_active:
            await self.send_to_client(websocket, {"type": "auto_compressor_status", "error": "Start Auto Compressor first"})
            return
        task = asyncio.create_task(self.auto_compressor_controller.start_soundcheck(
            genre=genre,
            style=style,
            genre_factor=genre_factor,
            mix_density_factor=mix_density_factor,
            bpm=bpm
        ))
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def stop_auto_compressor_soundcheck(self, websocket):
        if self.auto_compressor_controller:
            self.auto_compressor_controller.stop_soundcheck()
        await self.broadcast({"type": "auto_compressor_status", "soundcheck_running": False})

    async def start_auto_compressor_live(self, websocket, auto_correct: bool = True):
        if await self._block_quarantined_wing_write_surface(
            websocket,
            surface="start_auto_compressor_live",
            result_type="auto_compressor_status",
        ):
            return

        if not self.auto_compressor_controller or not self.auto_compressor_controller.is_active:
            await self.send_to_client(websocket, {"type": "auto_compressor_status", "error": "Start Auto Compressor first"})
            return
        await self.auto_compressor_controller.start_live(auto_correct=auto_correct)
        await self.broadcast({"type": "auto_compressor_status", "live_running": True})

    async def stop_auto_compressor_live(self, websocket):
        if self.auto_compressor_controller:
            self.auto_compressor_controller.stop_live()
        await self.broadcast({"type": "auto_compressor_status", "live_running": False})

    async def set_auto_compressor_profile(self, websocket, channel, profile: str):
        """Apply preset profile (punch, control, gentle, etc.) for one channel."""
        if await self._block_quarantined_wing_write_surface(
            websocket,
            surface="set_auto_compressor_profile",
            result_type="auto_compressor_status",
            request={"channel": channel, "profile": profile},
        ):
            return

        if not self.auto_compressor_controller or not self.mixer_client:
            await self.send_to_client(websocket, {"type": "auto_compressor_status", "error": "Not ready"})
            return
        from compressor_presets import get_preset
        from channel_recognizer import recognize_instrument
        from compressor_adaptation import adapt_params
        ch = int(channel)
        name = self.auto_compressor_controller.channel_names.get(ch, "")
        inst = recognize_instrument(name) or "custom"
        preset = get_preset(inst, profile or "base")
        features = self.auto_compressor_controller.channel_features.get(ch)
        if features:
            params = adapt_params(features, preset, profile or "base")
        else:
            params = {"threshold": preset.get("threshold", -15), "ratio_wing": "3.0", "attack_ms": preset.get("attack_ms", 15),
                      "release_ms": preset.get("release_ms", 150), "knee": preset.get("knee", 2), "gain": preset.get("makeup_gain", 0)}
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: self.auto_compressor_controller._apply_compressor_smooth(ch, params))
        await self.broadcast({"type": "auto_compressor_status", "profile_applied": ch, "profile": profile})

    async def set_auto_compressor_manual(self, websocket, channel, params: dict):
        """Apply manual compressor params to one channel."""
        if await self._block_quarantined_wing_write_surface(
            websocket,
            surface="set_auto_compressor_manual",
            result_type="auto_compressor_status",
            request={"channel": channel, "params": params or {}},
        ):
            return

        if not self.auto_compressor_controller or not self.mixer_client:
            await self.send_to_client(websocket, {"type": "auto_compressor_status", "error": "Not ready"})
            return
        ch = int(channel)
        from compressor_adaptation import ratio_float_to_wing
        p = dict(params)
        if "ratio" in p and isinstance(p["ratio"], (int, float)):
            p["ratio_wing"] = ratio_float_to_wing(float(p["ratio"]))
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: self.auto_compressor_controller._apply_compressor_smooth(ch, p))
        await self.broadcast({"type": "auto_compressor_status", "manual_applied": ch})

    async def _run_auto_soundcheck_cycle(self, websocket, device_id: str, channels: list,
                                         channel_settings: dict, channel_mapping: dict, timings: dict,
                                         observe_only: bool = False):
        """Execute the full auto soundcheck cycle."""
        observation_base_client = None
        try:
            logger.info("=" * 60)
            logger.info("Starting auto soundcheck cycle execution...")
            logger.info(f"device_id: {device_id}")
            logger.info(f"channels: {channels}")
            logger.info(f"timings: {timings}")
            logger.info(f"observe_only: {observe_only}")
            logger.info("=" * 60)

            if observe_only:
                observation_base_client = self._activate_observation_mode(
                    websocket,
                    asyncio.get_running_loop(),
                )
                await self._send_soundcheck_status(
                    websocket,
                    None,
                    0,
                    0,
                    "Observation mode active: all mixer writes are intercepted and logged.",
                )

            # Send initial status
            await self._send_soundcheck_status(websocket, None, 0, 0, "Initializing soundcheck cycle...")
            await asyncio.sleep(0.1)  # Small delay to ensure message is sent

            # Step 1: Reset to defaults
            logger.info("Step 1: Resetting all functions to defaults...")
            await self._send_soundcheck_status(websocket, "reset", 10, 0, "Resetting all functions to defaults...")
            await asyncio.sleep(0.1)

            # Run reset in a way that doesn't block
            reset_completed = False
            reset_error = None
            try:
                await self._send_soundcheck_status(websocket, "reset", 30, 0, "Stopping active controllers...")
                await asyncio.sleep(0.1)

                await self._send_soundcheck_status(websocket, "reset", 50, 0, "Resetting TRIM, EQ, Phase...")
                await asyncio.sleep(0.1)

                await self.reset_all_functions_to_defaults(websocket, channels)
                reset_completed = True
                logger.info("Step 1 completed: Reset successful")
            except asyncio.CancelledError:
                raise
            except Exception as e:
                reset_error = str(e)
                logger.error(f"Error in reset step: {e}", exc_info=True)
                await self._send_soundcheck_status(websocket, "reset", 0, 0, f"Reset error: {reset_error}", error=reset_error)
                # Continue anyway - reset errors are not critical

            if reset_completed:
                await self._send_soundcheck_status(websocket, "reset", 100, 0, "Reset complete")
            await asyncio.sleep(0.3)  # Reduced delay

            if not self.auto_soundcheck_running:
                logger.info("Auto soundcheck stopped after reset")
                return

            # Step 1.5: Channel recognizer (scan names + optional spectral verification)
            await self._send_soundcheck_status(websocket, "channel_recognizer", 0, 5, "Scanning channel names...")
            channel_names_for_recognizer = {}
            if self.mixer_client and self.mixer_client.is_connected and channels:
                try:
                    channel_names_for_recognizer = self.mixer_client.get_all_channel_names(max(channels) if channels else 40)
                    rec = scan_and_recognize(channel_names_for_recognizer)
                    await self.send_to_client(websocket, {"type": "auto_soundcheck_channel_recognizer", "recognition": rec})
                    await asyncio.sleep(0.5)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.warning(f"Channel recognizer step: {e}")
            await self._send_soundcheck_status(websocket, "channel_recognizer", 100, 0, "Channel scan complete")
            await asyncio.sleep(0.3)

            if not self.auto_soundcheck_running:
                return

            # Step 2: GAIN STAGING
            duration = timings.get("gain_staging", 30)
            logger.info(f"Step 2: Starting GAIN STAGING for {duration} seconds...")
            logger.info(f"Step 2: device_id={device_id}, channels={channels}, duration={duration}")
            await self._send_soundcheck_status(websocket, "gain_staging", 0, duration, "Starting GAIN STAGING...")

            # Ensure channel_settings and channel_mapping are not None
            if channel_settings is None:
                channel_settings = {}
            if channel_mapping is None:
                channel_mapping = {}

            # Stop any existing gain staging to ensure clean start
            if self.gain_staging and (self.gain_staging.is_active or self.gain_staging.is_audio_stream_active):
                logger.info("Stopping existing gain staging before starting new one in soundcheck cycle...")
                try:
                    self.gain_staging.stop()
                    await asyncio.sleep(0.5)  # Wait for cleanup
                    logger.info("Existing gain staging stopped")
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.warning(f"Error stopping existing gain staging: {e}")

            logger.info(f"Calling start_realtime_correction with device_id={device_id}, channels={channels}, channel_settings keys: {list(channel_settings.keys()) if channel_settings else None}, channel_mapping keys: {list(channel_mapping.keys()) if channel_mapping else None}")
            logger.info(f"Before start_realtime_correction: mixer_client={self.mixer_client is not None}, mixer_connected={self.mixer_client.is_connected if self.mixer_client else False}")

            try:
                await self.start_realtime_correction(device_id, channels, channel_settings, channel_mapping)
                logger.info("start_realtime_correction completed successfully")
                logger.info(f"After start_realtime_correction: gain_staging={self.gain_staging is not None}, is_active={self.gain_staging.is_active if self.gain_staging else False}, is_audio_stream_active={self.gain_staging.is_audio_stream_active if self.gain_staging else False}")
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Error starting realtime correction: {e}", exc_info=True)
                await self._send_soundcheck_status(websocket, "gain_staging", 0, 0, f"Error starting GAIN STAGING: {str(e)}", error=str(e))
                raise

            await self._wait_with_progress(websocket, "gain_staging", duration)

            if not self.auto_soundcheck_running:
                logger.info("Auto soundcheck stopped during GAIN STAGING")
                try:
                    await self.stop_realtime_correction()
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error(f"Error stopping realtime correction: {e}")
                return

            try:
                await self.stop_realtime_correction()
                logger.info("stop_realtime_correction completed successfully")

                # TRIM corrections are already applied on the mixer in real-time by gain staging.
                # Do NOT re-read from WingClient cache (get_channel_gain returns stale values
                # because send() doesn't update the state cache).
                logger.info("GAIN STAGING corrections already applied on mixer in real-time, preserving values")

                # Fully stop gain staging to release audio device for next steps (phase alignment)
                if self.gain_staging:
                    logger.info("Fully stopping gain staging controller to release audio device...")
                    try:
                        self.gain_staging.stop()
                        await asyncio.sleep(1.0)  # Wait for audio device to be fully released
                        logger.info("Gain staging controller fully stopped, audio device released")
                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        logger.warning(f"Error fully stopping gain staging: {e}")

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Error stopping realtime correction: {e}", exc_info=True)
                # Still try to release audio device
                if self.gain_staging:
                    try:
                        self.gain_staging.stop()
                        await asyncio.sleep(1.0)
                    except asyncio.CancelledError:
                        raise
                    except Exception as e2:
                        logger.warning(f"Error force-stopping gain staging: {e2}")

            await self._send_soundcheck_status(websocket, "gain_staging", 100, 0, "GAIN STAGING complete")
            await asyncio.sleep(1)

            if not self.auto_soundcheck_running:
                return

            # Step 3: PHASE ALIGNMENT
            duration = timings.get("phase_alignment", 30)
            await self._send_soundcheck_status(websocket, "phase_alignment", 0, duration, "Starting PHASE ALIGNMENT...")

            # Get reference channel (first channel)
            reference_channel = channels[0] if channels else None
            channels_to_align = channels[1:] if len(channels) > 1 else []

            if reference_channel and channels_to_align:
                logger.info(f"Starting phase alignment (apply_once): ref={reference_channel}, channels={channels_to_align}")
                try:
                    apply_once_sec = min(duration, 15)
                    await asyncio.wait_for(
                        self.start_phase_alignment(
                            websocket, device_id, reference_channel, channels_to_align,
                            apply_once=True, apply_once_duration_sec=apply_once_sec
                        ),
                        timeout=15.0
                    )
                    logger.info("Phase alignment started successfully")
                except asyncio.TimeoutError:
                    logger.warning("Phase alignment start timed out, continuing...")
                    await self._send_soundcheck_status(websocket, "phase_alignment", 0, duration, "Phase alignment start timed out, skipping...")
                    # Skip phase alignment if start fails
                    await self._send_soundcheck_status(websocket, "phase_alignment", 100, 0, "PHASE ALIGNMENT skipped (start timeout)")
                    await asyncio.sleep(0.5)
                    # Continue to next step
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error(f"Error starting phase alignment: {e}", exc_info=True)
                    await self._send_soundcheck_status(websocket, "phase_alignment", 0, duration, f"Phase alignment error: {str(e)}, skipping...", error=str(e))
                    await self._send_soundcheck_status(websocket, "phase_alignment", 100, 0, "PHASE ALIGNMENT skipped (error)")
                    await asyncio.sleep(0.5)
                    # Continue to next step
                else:
                    # Only wait if start was successful
                    await self._wait_with_progress(websocket, "phase_alignment", duration)

                    if not self.auto_soundcheck_running:
                        logger.info("Auto soundcheck stopped during PHASE ALIGNMENT")
                        try:
                            await asyncio.wait_for(self.stop_phase_alignment(websocket), timeout=10.0)
                        except asyncio.CancelledError:
                            raise
                        except asyncio.TimeoutError as e:
                            logger.error(f"Error stopping phase alignment: {e}")
                        except Exception as e:
                            logger.error(f"Error stopping phase alignment: {e}")
                        return

                    try:
                        # Stop phase alignment with timeout
                        await asyncio.wait_for(self.stop_phase_alignment(websocket), timeout=10.0)
                        logger.info("Phase alignment stopped successfully")
                        await self._send_soundcheck_status(websocket, "phase_alignment", 100, 0, "PHASE ALIGNMENT complete")
                    except asyncio.CancelledError:
                        raise
                    except asyncio.TimeoutError:
                        logger.warning("Phase alignment stop timed out, continuing...")
                        await self._send_soundcheck_status(websocket, "phase_alignment", 100, 0, "PHASE ALIGNMENT stopped (timeout)")
                    except Exception as e:
                        logger.error(f"Error stopping phase alignment: {e}", exc_info=True)
                        await self._send_soundcheck_status(websocket, "phase_alignment", 100, 0, f"PHASE ALIGNMENT error: {str(e)}")
            else:
                await self._send_soundcheck_status(websocket, "phase_alignment", 100, 0, "PHASE ALIGNMENT skipped (need at least 2 channels)")

            await asyncio.sleep(1)

            if not self.auto_soundcheck_running:
                return

            # Step 4: AUTO EQ
            duration = timings.get("auto_eq", 30)
            await self._send_soundcheck_status(websocket, "auto_eq", 0, duration, "Starting AUTO EQ...")

            # Build channels config for multi-channel auto EQ
            channels_config = []
            for channel_id in channels:
                profile = "custom"
                if channel_settings:
                    profile = channel_settings.get(channel_id, {}).get("preset", "custom")
                channels_config.append({
                    "channel": channel_id,
                    "profile": profile,
                    "auto_apply": False
                })

            await self.start_multi_channel_auto_eq(websocket, device_id, channels_config)
            await self._wait_with_progress(websocket, "auto_eq", duration)

            if not self.auto_soundcheck_running:
                await self.stop_multi_channel_auto_eq(websocket)
                return

            await self.stop_multi_channel_auto_eq(websocket)

            # Apply EQ corrections
            logger.info("Applying AUTO EQ corrections...")
            try:
                await self.apply_all_corrections(websocket)
                logger.info("AUTO EQ corrections applied successfully")
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Error applying AUTO EQ corrections: {e}", exc_info=True)
                await self._send_soundcheck_status(websocket, "auto_eq", 100, 0, f"AUTO EQ complete (apply error: {str(e)})")

            await self._send_soundcheck_status(websocket, "auto_eq", 100, 0, "AUTO EQ complete")
            await asyncio.sleep(1)

            if not self.auto_soundcheck_running:
                return

            # Step 5: AUTO FADER (Auto Balance)
            duration = timings.get("auto_fader", 30)
            await self._send_soundcheck_status(websocket, "auto_fader", 0, duration, "Starting AUTO FADER (Auto Balance)...")

            # Use same settings as Auto Balance from config
            auto_fader_config = self.config.get("automation", {}).get("auto_fader", {})
            bleed_threshold = auto_fader_config.get("bleed_threshold", -50.0)

            logger.info(f"Starting Auto Balance in soundcheck: duration={duration}s, bleed_threshold={bleed_threshold} LUFS, "
                       f"channels={channels}, channel_settings keys: {list(channel_settings.keys()) if channel_settings else []}, "
                       f"channel_mapping keys: {list(channel_mapping.keys()) if channel_mapping else []}")

            # Use same methods and settings as Auto Balance
            await self.start_auto_balance(websocket, device_id, channels,
                                         channel_settings or {},
                                         channel_mapping or {},
                                         duration, bleed_threshold)

            # Wait for auto balance LEARN phase to complete
            # The start_auto_balance method handles the LEARN phase internally
            # We show progress while waiting
            await self._wait_with_progress(websocket, "auto_fader", duration)

            if not self.auto_soundcheck_running:
                await self.cancel_auto_balance(websocket)
                return

            # Small delay to ensure LEARN phase is fully complete
            await asyncio.sleep(1)

            # Apply auto balance corrections
            await self.apply_auto_balance(websocket)
            await self._send_soundcheck_status(websocket, "auto_fader", 100, 0, "AUTO FADER complete")
            await asyncio.sleep(1)

            # Complete
            await self._send_soundcheck_status(websocket, None, 100, 0, "Soundcheck cycle complete!", complete=True)

        except asyncio.CancelledError:
            logger.info("Auto soundcheck cycle cancelled")
            try:
                await asyncio.shield(
                    self._send_soundcheck_status(
                        websocket,
                        None,
                        0,
                        0,
                        "Auto soundcheck cancelled",
                        error="Cancelled",
                    )
                )
            except Exception as exc:
                logger.warning("Failed to send cancellation status for auto soundcheck: %s", exc)
            raise
        except Exception as e:
            logger.error(f"Error in auto soundcheck cycle: {e}", exc_info=True)
            await self._send_soundcheck_status(websocket, None, 0, 0, f"Error: {str(e)}", error=str(e))
        finally:
            if observation_base_client is not None:
                await self._deactivate_observation_mode(websocket, observation_base_client)
            logger.info("Auto soundcheck cycle finished, cleaning up...")
            self.auto_soundcheck_running = False
            self.auto_soundcheck_observe_only = False
            self.auto_soundcheck_task = None
            self.auto_soundcheck_websocket = None

    async def _wait_with_progress(self, websocket, step: str, duration: float):
        """Wait for specified duration while sending progress updates."""
        start_time = time.time()
        update_interval = 0.5  # Update every 0.5 seconds

        while self.auto_soundcheck_running:
            elapsed = time.time() - start_time
            if elapsed >= duration:
                break

            progress = (elapsed / duration) * 100
            remaining = duration - elapsed

            await self._send_soundcheck_status(websocket, step, progress, remaining,
                                             f"Running {step.upper().replace('_', ' ')}...")

            await asyncio.sleep(update_interval)

    async def _send_soundcheck_status(self, websocket, current_step: str = None,
                                     step_progress: float = 0, step_time_remaining: float = 0,
                                     message: str = "", error: str = None, complete: bool = False):
        """Send auto soundcheck status update."""
        try:
            if websocket is None:
                logger.warning("Cannot send soundcheck status: websocket is None")
                return

            status_data = {
                "type": "auto_soundcheck_status",
                "is_running": self.auto_soundcheck_running and not complete,
                "running": self.auto_soundcheck_running and not complete,
                "observe_only": self.auto_soundcheck_observe_only,
                "current_step": current_step,
                "step_progress": step_progress,
                "progress": step_progress,
                "step_time_remaining": step_time_remaining,
                "message": message,
                "error": error,
                "complete": complete
            }

            logger.debug(f"Sending soundcheck status: step={current_step}, progress={step_progress:.1f}%, message={message}")
            await self.send_to_client(websocket, status_data)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"Error sending soundcheck status: {e}", exc_info=True)
            # Don't raise - we don't want to stop the cycle if status update fails

    async def _execute_voice_command(self, command: dict):
        """Execute a recognized voice command"""
        command = self._normalize_voice_command_payload(command)
        command_generation = self._manual_write_emergency_generation
        if self._is_shutting_down or self._manual_write_emergency_stopped:
            await self._broadcast_voice_command_error(
                command,
                reason="emergency_stop_active",
            )
            return
        if "raw_command" in command and "type" not in command:
            await self._broadcast_voice_command_error(
                command,
                reason="malformed_voice_command",
            )
            return
        if not self.mixer_client:
            logger.warning("No mixer connected, cannot execute voice command")
            await self._broadcast_voice_command_error(
                command,
                reason="mixer_not_connected",
            )
            return

        try:
            cmd_type = str(command.get("type") or "").strip().lower()
            channel = command.get("channel")
            if await self._block_quarantined_voice_command_if_needed(command):
                return
            if (
                self._is_shutting_down
                or self._manual_write_emergency_stopped
                or self._manual_write_emergency_generation != command_generation
            ):
                await self._broadcast_voice_command_error(
                    command,
                    reason="emergency_stop_active",
                )
                return

            if cmd_type == "set_fader":
                if channel is None:
                    await self._broadcast_voice_command_error(command, reason="missing_channel")
                    return
                value = self._coerce_finite_float(command.get("value", 0.5))
                if value is None:
                    await self._broadcast_voice_command_error(command, reason="invalid_value")
                    return
                write_result = self.mixer_client.set_channel_fader(channel, value)
                if write_result is False:
                    await self._broadcast_voice_command_write_failed(
                        command,
                        value_type="fader",
                        requested_value=value,
                    )
                    return
                await self.broadcast({
                    "type": "voice_command_executed",
                    "command": cmd_type,
                    "channel": channel,
                    "value": value
                })

            elif cmd_type == "set_gain":
                if channel is None:
                    await self._broadcast_voice_command_error(command, reason="missing_channel")
                    return
                value = self._coerce_finite_float(command.get("value", 0.0))
                if value is None:
                    await self._broadcast_voice_command_error(command, reason="invalid_value")
                    return
                write_result = self.mixer_client.set_channel_gain(channel, value)
                if write_result is False:
                    await self._broadcast_voice_command_write_failed(
                        command,
                        value_type="gain",
                        requested_value=value,
                    )
                    return
                await self.broadcast({
                    "type": "voice_command_executed",
                    "command": cmd_type,
                    "channel": channel,
                    "value": value
                })

            elif cmd_type == "set_pan":
                if channel is None:
                    await self._broadcast_voice_command_error(command, reason="missing_channel")
                    return
                raw_value = command.get("value", command.get("pan", command.get("pan_percent", 0.0)))
                value = self._coerce_finite_float(raw_value)
                if value is None:
                    await self._broadcast_voice_command_error(command, reason="invalid_value")
                    return
                value = self._clamp_manual_write_value(value, "pan")
                write_result = self.mixer_client.set_channel_pan(channel, value)
                if write_result is False:
                    await self._broadcast_voice_command_write_failed(
                        command,
                        value_type="pan",
                        requested_value=value,
                    )
                    return
                await self.broadcast({
                    "type": "voice_command_executed",
                    "command": cmd_type,
                    "channel": channel,
                    "value": value,
                })

            elif cmd_type == "load_snap":
                snap_name = command.get("snap_name")
                if not snap_name:
                    await self._broadcast_voice_command_error(command, reason="missing_snap_name")
                    return
                if not isinstance(self.mixer_client, (WingClient, EnhancedOSCClient)):
                    await self._broadcast_voice_command_error(command, reason="unsupported_mixer_client")
                    return
                success = self.mixer_client.load_snap(snap_name)
                await self.broadcast({
                    "type": "voice_command_executed",
                    "command": cmd_type,
                    "snap_name": snap_name,
                    "success": success
                })

            elif cmd_type == "mute_channel":
                if channel is None:
                    await self._broadcast_voice_command_error(command, reason="missing_channel")
                    return
                muted = command.get("muted", True)
                parsed_mute = self._coerce_optional_bool(muted)
                if parsed_mute is None:
                    parsed_mute = bool(muted)
                write_result = self.mixer_client.set_channel_mute(channel, int(parsed_mute))
                if write_result is False:
                    await self._broadcast_voice_command_write_failed(
                        command,
                        value_type="mute",
                        requested_value=bool(parsed_mute),
                    )
                    return
                await self.broadcast({
                    "type": "voice_command_executed",
                    "command": cmd_type,
                    "channel": channel,
                    "muted": bool(parsed_mute)
                    })

            elif cmd_type == "volume_up":
                if channel is not None:
                    db_change = self._coerce_finite_float(command.get("db", 3))
                    if db_change is None:
                        await self._broadcast_voice_command_error(command, reason="invalid_db")
                        return
                    # Try to get current value from tracked values first, then from mixer
                    current_db = self.last_fader_values.get(channel)

                    if current_db is None:
                        # Request fresh value from Wing
                        self.mixer_client.send(f"/ch/{channel}/fdr")
                        await asyncio.sleep(0.1)  # Wait for response
                        if (
                            self._is_shutting_down
                            or self._manual_write_emergency_stopped
                            or self._manual_write_emergency_generation != command_generation
                        ):
                            await self._broadcast_voice_command_error(
                                command,
                                reason="emergency_stop_active",
                            )
                            return
                        current_db = self.mixer_client.get_channel_fader(channel)

                    if current_db is not None:
                        # Add dB directly (Wing uses dB values)
                        new_db = min(10.0, current_db + db_change)
                        write_result = self.mixer_client.set_channel_fader(channel, new_db)
                        if write_result is False:
                            await self._broadcast_voice_command_write_failed(
                                command,
                                value_type="fader",
                                requested_value=new_db,
                            )
                            return
                        # Update tracked value
                        self.last_fader_values[channel] = new_db
                        logger.info(f"Increased channel {channel} fader: {current_db:.1f} dB -> {new_db:.1f} dB (+{db_change} dB)")
                    else:
                        # Fallback: use last known or default to 0
                        last_known = self.last_fader_values.get(channel, 0.0)
                        new_db = min(10.0, last_known + db_change)
                        write_result = self.mixer_client.set_channel_fader(channel, new_db)
                        if write_result is False:
                            await self._broadcast_voice_command_write_failed(
                                command,
                                value_type="fader",
                                requested_value=new_db,
                            )
                            return
                        self.last_fader_values[channel] = new_db
                        logger.info(f"Increased channel {channel} fader by {db_change} dB (to {new_db:.1f} dB, using tracked value)")

                    await self.broadcast({
                        "type": "voice_command_executed",
                        "command": cmd_type,
                        "channel": channel,
                        "db_change": db_change,
                        "new_db": new_db
                    })
                else:
                    await self._broadcast_voice_command_error(command, reason="missing_channel")
                    return

            elif cmd_type == "volume_down":
                if channel is not None:
                    db_change = self._coerce_finite_float(command.get("db", 3))
                    if db_change is None:
                        await self._broadcast_voice_command_error(command, reason="invalid_db")
                        return
                    # Try to get current value from tracked values first, then from mixer
                    current_db = self.last_fader_values.get(channel)

                    if current_db is None:
                        # Request fresh value from Wing
                        self.mixer_client.send(f"/ch/{channel}/fdr")
                        await asyncio.sleep(0.1)  # Wait for response
                        if (
                            self._is_shutting_down
                            or self._manual_write_emergency_stopped
                            or self._manual_write_emergency_generation != command_generation
                        ):
                            await self._broadcast_voice_command_error(
                                command,
                                reason="emergency_stop_active",
                            )
                            return
                        current_db = self.mixer_client.get_channel_fader(channel)

                    if current_db is not None:
                        # Subtract dB directly (Wing uses dB values)
                        new_db = max(-144.0, current_db - db_change)
                        write_result = self.mixer_client.set_channel_fader(channel, new_db)
                        if write_result is False:
                            await self._broadcast_voice_command_write_failed(
                                command,
                                value_type="fader",
                                requested_value=new_db,
                            )
                            return
                        # Update tracked value
                        self.last_fader_values[channel] = new_db
                        logger.info(f"Decreased channel {channel} fader: {current_db:.1f} dB -> {new_db:.1f} dB (-{db_change} dB)")
                    else:
                        # Fallback: use last known or default to 0
                        last_known = self.last_fader_values.get(channel, 0.0)
                        new_db = max(-144.0, last_known - db_change)
                        write_result = self.mixer_client.set_channel_fader(channel, new_db)
                        if write_result is False:
                            await self._broadcast_voice_command_write_failed(
                                command,
                                value_type="fader",
                                requested_value=new_db,
                            )
                            return
                        self.last_fader_values[channel] = new_db
                        logger.info(f"Decreased channel {channel} fader by {db_change} dB (to {new_db:.1f} dB, using tracked value)")

                    await self.broadcast({
                        "type": "voice_command_executed",
                        "command": cmd_type,
                        "channel": channel,
                        "db_change": db_change,
                        "new_db": new_db
                    })
                else:
                    await self._broadcast_voice_command_error(command, reason="missing_channel")
                    return

            elif cmd_type == "eq_on":
                on = command.get("on", 1)
                if channel is None:
                    await self._broadcast_voice_command_error(command, reason="missing_channel")
                    return
                if not isinstance(self.mixer_client, (WingClient, EnhancedOSCClient)):
                    await self._broadcast_voice_command_error(command, reason="unsupported_mixer_client")
                    return
                if channel is not None and isinstance(self.mixer_client, (WingClient, EnhancedOSCClient)):
                    write_result = self.mixer_client.set_eq_on(channel, on)
                    if write_result is False:
                        await self._broadcast_voice_command_write_failed(
                            command,
                            value_type="eq_on",
                            requested_value=on,
                        )
                        return
                    logger.info(f"Set EQ {'on' if on else 'off'} for channel {channel}")
                    await self.broadcast({
                        "type": "voice_command_executed",
                        "command": cmd_type,
                        "channel": channel,
                        "on": on
                    })

            elif cmd_type == "eq_band_up" or cmd_type == "eq_band_down":
                band = command.get("band")
                if channel is None:
                    await self._broadcast_voice_command_error(command, reason="missing_channel")
                    return
                if not band:
                    await self._broadcast_voice_command_error(command, reason="missing_eq_band")
                    return
                db_change = self._coerce_finite_float(command.get("db", 3))
                if db_change is None:
                    await self._broadcast_voice_command_error(command, reason="invalid_db")
                    return
                if not isinstance(self.mixer_client, (WingClient, EnhancedOSCClient)):
                    await self._broadcast_voice_command_error(command, reason="unsupported_mixer_client")
                    return
                if channel is not None and band and isinstance(self.mixer_client, (WingClient, EnhancedOSCClient)):
                    # Get current gain
                    band_gain_key = f"{band}g" if band in ["1", "2", "3", "4"] else ("lg" if band == "low" else "hg")
                    current_gain = self.mixer_client.get_eq_band_gain(channel, band_gain_key) or 0.0

                    if cmd_type == "eq_band_up":
                        new_gain = min(15.0, current_gain + db_change)
                    else:
                        new_gain = max(-15.0, current_gain - db_change)

                    write_result = self.mixer_client.set_eq_band_gain(channel, band_gain_key, new_gain)
                    if write_result is False:
                        await self._broadcast_voice_command_write_failed(
                            command,
                            value_type="eq_band_gain",
                            requested_value=new_gain,
                        )
                        return
                    logger.info(f"Set EQ band {band} for channel {channel}: {current_gain:.1f} dB -> {new_gain:.1f} dB")
                    await self.broadcast({
                        "type": "voice_command_executed",
                        "command": cmd_type,
                        "channel": channel,
                        "band": band,
                        "new_gain": new_gain
                    })

            elif cmd_type == "compressor_on":
                on = command.get("on", 1)
                if channel is None:
                    await self._broadcast_voice_command_error(command, reason="missing_channel")
                    return
                if not isinstance(self.mixer_client, (WingClient, EnhancedOSCClient)):
                    await self._broadcast_voice_command_error(command, reason="unsupported_mixer_client")
                    return
                if channel is not None and isinstance(self.mixer_client, (WingClient, EnhancedOSCClient)):
                    write_result = self.mixer_client.set_compressor_on(channel, on)
                    if write_result is False:
                        await self._broadcast_voice_command_write_failed(
                            command,
                            value_type="compressor_on",
                            requested_value=on,
                        )
                        return
                    logger.info(f"Set compressor {'on' if on else 'off'} for channel {channel}")
                    await self.broadcast({
                        "type": "voice_command_executed",
                        "command": cmd_type,
                        "channel": channel,
                        "on": on
                    })

            elif cmd_type == "compressor_threshold":
                if channel is None:
                    await self._broadcast_voice_command_error(command, reason="missing_channel")
                    return
                threshold = command.get("threshold")
                if threshold is None:
                    await self._broadcast_voice_command_error(command, reason="missing_compressor_threshold")
                    return
                threshold = self._coerce_finite_float(threshold)
                if threshold is None:
                    await self._broadcast_voice_command_error(command, reason="invalid_compressor_threshold")
                    return
                if not isinstance(self.mixer_client, (WingClient, EnhancedOSCClient)):
                    await self._broadcast_voice_command_error(command, reason="unsupported_mixer_client")
                    return
                if channel is not None and threshold is not None and isinstance(self.mixer_client, (WingClient, EnhancedOSCClient)):
                    write_result = self.mixer_client.set_compressor_threshold(channel, threshold)
                    if write_result is False:
                        await self._broadcast_voice_command_write_failed(
                            command,
                            value_type="compressor_threshold",
                            requested_value=threshold,
                        )
                        return
                    logger.info(f"Set compressor threshold for channel {channel}: {threshold:.1f} dB")
                    await self.broadcast({
                        "type": "voice_command_executed",
                        "command": cmd_type,
                        "channel": channel,
                        "threshold": threshold
                    })

            elif cmd_type == "compressor_gain_up" or cmd_type == "compressor_gain_down":
                if channel is None:
                    await self._broadcast_voice_command_error(command, reason="missing_channel")
                    return
                db_change = self._coerce_finite_float(command.get("db", 3))
                if db_change is None:
                    await self._broadcast_voice_command_error(command, reason="invalid_db")
                    return
                if not isinstance(self.mixer_client, (WingClient, EnhancedOSCClient)):
                    await self._broadcast_voice_command_error(command, reason="unsupported_mixer_client")
                    return
                if channel is not None and isinstance(self.mixer_client, (WingClient, EnhancedOSCClient)):
                    current_gain = self.mixer_client.get_compressor_gain(channel) or 0.0
                    if cmd_type == "compressor_gain_up":
                        new_gain = min(12.0, current_gain + db_change)
                    else:
                        new_gain = max(-6.0, current_gain - db_change)

                    write_result = self.mixer_client.set_compressor_gain(channel, new_gain)
                    if write_result is False:
                        await self._broadcast_voice_command_write_failed(
                            command,
                            value_type="compressor_gain",
                            requested_value=new_gain,
                        )
                        return
                    logger.info(f"Set compressor gain for channel {channel}: {current_gain:.1f} dB -> {new_gain:.1f} dB")
                    await self.broadcast({
                        "type": "voice_command_executed",
                        "command": cmd_type,
                        "channel": channel,
                        "new_gain": new_gain
                    })
            else:
                await self._broadcast_voice_command_error(
                    command,
                    reason="unknown_voice_command",
                )
                return

            logger.info("=" * 60)
            logger.info(f"✅ EXECUTED VOICE COMMAND: {cmd_type}")
            if channel is not None:
                logger.info(f"   Channel: {channel}")
            logger.info("=" * 60)

        except asyncio.CancelledError:
            logger.info("Voice command execution cancelled")
            raise
        except Exception as e:
            logger.error("=" * 60)
            logger.error(f"❌ ERROR EXECUTING VOICE COMMAND: {e}")
            logger.error(f"   Command: {command}")
            logger.error("=" * 60)
            await self._broadcast_voice_command_error(
                command,
                reason="voice_command_exception",
                error=str(e),
            )

    async def handler(self, websocket: WebSocketServerProtocol):
        if not self._reserve_websocket_admission(websocket):
            logger.warning(
                "Rejected WebSocket connection at configured client cap=%s",
                self._ws_max_clients,
            )
            await self._close_overloaded_websocket(
                websocket,
                reason="server_capacity",
            )
            return

        try:
            authenticated = await self._authenticate_websocket(websocket)
        finally:
            self._release_websocket_admission(websocket)
        if not authenticated:
            return

        registered = False
        try:
            await self.register_client(websocket)
            registered = True
        except Exception:
            await self.unregister_client(websocket)
            raise
        expiry_task = None
        expires_at = self._authenticated_websocket_expiry.get(id(websocket))
        if expires_at is not None:
            expiry_task = asyncio.create_task(
                self._expire_authenticated_websocket(websocket, expires_at)
            )

        normal_queue: asyncio.Queue[QueuedIngressMessage] = asyncio.Queue(
            maxsize=WS_NORMAL_QUEUE_SIZE
        )
        normal_worker = asyncio.create_task(
            self._run_websocket_normal_lane(websocket, normal_queue)
        )
        connection_bucket = TokenBucket(
            WS_PER_CONNECTION_RATE,
            WS_PER_CONNECTION_BURST,
        )
        rejection_window = RejectionWindow(
            WS_REJECT_CLOSE_THRESHOLD,
            WS_REJECT_WINDOW_SECONDS,
        )

        try:
            async for message in websocket:
                msg_type = websocket_message_type(message)
                if self._is_shutting_down:
                    await websocket.close(code=1001, reason="server_shutdown")
                    break
                if msg_type in EMERGENCY_FAST_LANE_TYPES:
                    # Exact stop message types bypass both token buckets and the
                    # normal FIFO. The first accepted stop also invalidates all
                    # normal messages already queued under the previous epoch.
                    self._schedule_websocket_emergency(websocket, message)
                    continue

                if expires_at is not None and time.time() >= expires_at:
                    await self._close_expired_authenticated_websocket(websocket)
                    break

                now = time.monotonic()
                if not consume_token_buckets(
                    (connection_bucket, self._ws_global_ingress_bucket),
                    now=now,
                ):
                    should_close = await self._record_websocket_ingress_rejection(
                        websocket,
                        rejection_window,
                        reason="rate_limited",
                        now=now,
                    )
                    if should_close:
                        break
                    continue

                try:
                    normal_queue.put_nowait(
                        QueuedIngressMessage(
                            payload=message,
                            emergency_generation=self._ws_ingress_emergency_generation,
                        )
                    )
                except asyncio.QueueFull:
                    should_close = await self._record_websocket_ingress_rejection(
                        websocket,
                        rejection_window,
                        reason="normal_queue_full",
                        now=now,
                    )
                    if should_close:
                        break
        except websockets.exceptions.ConnectionClosed:
            logger.info("Client connection closed")
        except asyncio.CancelledError:
            logger.info("WebSocket handler cancelled")
            raise
        except Exception as e:
            logger.error(f"Handler error: {e}", exc_info=True)
        finally:
            normal_worker.cancel()
            with suppress(asyncio.CancelledError):
                await normal_worker
            if expiry_task is not None:
                expiry_task.cancel()
                with suppress(asyncio.CancelledError):
                    await expiry_task
            self._authenticated_websocket_expiry.pop(id(websocket), None)
            self._authenticated_websocket_expiry_close_started.discard(id(websocket))
            if registered:
                await self.unregister_client(websocket)

    async def start(self):
        logger.info(f"Starting WebSocket server on {self.ws_host}:{self.ws_port}")
        self._activate_network_runtime_safety_boundary()

        # Create shutdown event in the correct event loop
        self._shutdown_event = asyncio.Event()

        # Set up signal handlers for graceful shutdown
        loop = asyncio.get_running_loop()

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(
                    sig,
                    lambda s=sig: asyncio.create_task(self.graceful_shutdown(s))
                )
            except NotImplementedError:
                # Windows doesn't support add_signal_handler
                pass

        # Create server with SO_REUSEADDR enabled
        import socket

        # Custom create_server with reuse_address
        try:
            server = await websockets.serve(
                self.handler,
                self.ws_host,
                self.ws_port,
                reuse_address=True,  # Allow port reuse
                origins=self._ws_allowed_origins,
                max_size=WS_MAX_MESSAGE_BYTES,
                max_queue=WS_PROTOCOL_MAX_QUEUE,
            )

            logger.info(f"WebSocket server started on {self.ws_host}:{self.ws_port}")

            if self.agent_training_service:
                await self.agent_training_service.start_scheduler()

            # Wait for shutdown signal
            await self._shutdown_event.wait()

            # Close the server
            server.close()
            await server.wait_closed()
            if self._shutdown_errors:
                raise RuntimeError(
                    f"Graceful shutdown incomplete: {self._shutdown_errors}"
                )

        except OSError as e:
            if e.errno == errno.EADDRINUSE:
                if not self._ws_port_force_cleanup_enabled():
                    raise RuntimeError(
                        f"Port {self.ws_port} is already in use. Refusing automatic kill; "
                        "set AUTOMIXER_FORCE_CLEANUP_WS_PORT=true only after confirming "
                        "the listener is safe to stop."
                    ) from e
                logger.error(
                    "Port %s is already in use. Explicit cleanup opt-in is enabled; "
                    "attempting safe PID cleanup.",
                    self.ws_port,
                )
                try:
                    self._force_cleanup_ws_port()
                    await asyncio.sleep(1)
                    # Retry
                    server = await websockets.serve(
                        self.handler,
                        self.ws_host,
                        self.ws_port,
                        reuse_address=True,
                        origins=self._ws_allowed_origins,
                        max_size=WS_MAX_MESSAGE_BYTES,
                        max_queue=WS_PROTOCOL_MAX_QUEUE,
                    )
                    logger.info(f"WebSocket server started on {self.ws_host}:{self.ws_port} (after cleanup)")
                    await self._shutdown_event.wait()
                    server.close()
                    await server.wait_closed()
                    if self._shutdown_errors:
                        raise RuntimeError(
                            f"Graceful shutdown incomplete: {self._shutdown_errors}"
                        )
                except Exception as retry_error:
                    logger.error(f"Failed to start server after cleanup: {retry_error}")
                    raise
            else:
                raise

        logger.info("Server stopped")


# Global server instance for atexit cleanup
_server_instance: Optional[AutoMixerServer] = None

def _atexit_cleanup():
    """Cleanup function called on exit."""
    global _server_instance
    if _server_instance:
        logger.info("Atexit cleanup triggered")
        _server_instance.cleanup_all_controllers()


def main() -> int:
    """Run the backend and return a process exit code supervisors can trust."""

    global _server_instance
    atexit.register(_atexit_cleanup)

    server: Optional[AutoMixerServer] = None
    exit_code = 0
    try:
        ws_host = os.environ.get("AUTOMIXER_WS_HOST", "127.0.0.1")
        ws_port = int(os.environ.get("AUTOMIXER_WS_PORT", "8765"))
        server = AutoMixerServer(ws_host=ws_host, ws_port=ws_port)
        _server_instance = server
        asyncio.run(server.start())
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        exit_code = 1
    finally:
        if server is not None:
            try:
                server.cleanup_all_controllers()
            except Exception as cleanup_error:
                logger.error(
                    "Server cleanup failed: %s",
                    cleanup_error,
                    exc_info=True,
                )
                exit_code = 1
            finally:
                if _server_instance is server:
                    _server_instance = None
        logger.info("Server shutdown complete")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
