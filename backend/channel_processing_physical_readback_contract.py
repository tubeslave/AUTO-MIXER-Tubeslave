"""Contract for channel EQ/dynamics/gate physical readback artifacts.

This module is transport-free. It defines the JSON-like artifact shape a future
operator-run physical readback must produce before the
``channel_eq_dynamics_gate`` read surface can be treated as physical fixture
evidence for EQ, compressor, gate, de-esser, and unmasking promotion decisions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Tuple


CHANNEL_PROCESSING_PHYSICAL_READBACK_CONTRACT_METHOD_ID = (
    "channel_processing_physical_readback_contract"
)
CHANNEL_PROCESSING_PHYSICAL_READBACK_SCHEMA_ID = "channel_processing_physical_readback.v1"
CHANNEL_PROCESSING_PHYSICAL_READBACK_ARTIFACT_TYPE = "channel_processing_physical_readback"
CHANNEL_PROCESSING_SURFACE_ID = "channel_eq_dynamics_gate"
CHANNEL_PROCESSING_REMEDIATION_ACTION_ID = "remediate_channel_eq_dynamics_gate"

CHANNEL_PROCESSING_REQUIRED_TOP_LEVEL_FIELDS = (
    "type",
    "schema_id",
    "surface_id",
    "remediation_action_id",
    "captured_at",
    "source",
    "console_model",
    "channel_count",
    "channels",
    "safety",
)
CHANNEL_PROCESSING_REQUIRED_CHANNEL_FIELDS = (
    "channel_id",
    "channel_label",
    "status",
    "source",
    "captured_at",
    "hpf",
    "eq",
    "compressor",
    "gate",
    "coverage",
)
CHANNEL_PROCESSING_REQUIRED_HPF_FIELDS = (
    "enabled",
    "frequency_hz",
)
CHANNEL_PROCESSING_REQUIRED_EQ_FIELDS = (
    "enabled",
    "bands",
)
CHANNEL_PROCESSING_REQUIRED_EQ_BAND_FIELDS = (
    "band",
    "frequency_hz",
    "gain_db",
    "q",
)
CHANNEL_PROCESSING_REQUIRED_COMPRESSOR_FIELDS = (
    "enabled",
    "model",
    "threshold_db",
    "ratio",
    "attack_ms",
    "release_ms",
    "makeup_db",
    "mix_pct",
    "detector",
    "gain_reduction_db",
)
CHANNEL_PROCESSING_REQUIRED_GATE_FIELDS = (
    "enabled",
    "model",
    "threshold_db",
    "range_db",
    "attack_ms",
    "hold_ms",
    "release_ms",
    "accent",
    "ratio",
)
CHANNEL_PROCESSING_REQUIRED_COVERAGE_FLAGS = (
    "hpf",
    "eq",
    "compressor",
    "gate",
    "cache_matches_physical_readback",
)
CHANNEL_PROCESSING_PHYSICAL_READBACK_SOURCES = (
    "physical_readback",
    "wing_physical_readback",
    "dlive_physical_readback",
)
CHANNEL_PROCESSING_VERIFIED_PHYSICAL_STATUSES = (
    "verified_from_physical_readback",
)
CHANNEL_PROCESSING_REQUIRED_FALSE_SAFETY_FLAGS = (
    "live_write",
    "mutates_console_state",
    "live_widening_allowed",
    "full_autonomous_ready",
)
CHANNEL_PROCESSING_FORBIDDEN_MUTATION_EVIDENCE_FIELDS = (
    "write_events",
    "manual_write_events",
    "apply_result",
    "rollback_result",
    "proposal_apply_result",
    "eq_write_events",
    "compressor_write_events",
    "gate_write_events",
    "hpf_write_events",
    "reset_channel_processing_events",
)
MINIMUM_EQ_BAND_COUNT = 4


@dataclass(frozen=True)
class ChannelProcessingPhysicalReadbackValidation:
    method_id: str
    schema_id: str
    surface_id: str
    remediation_action_id: str
    status: str
    accepted_as_physical_fixture_candidate: bool
    live_authority_granted: bool
    full_autonomous_ready: bool
    channel_count: int
    violations: Tuple[str, ...]
    evidence: Dict[str, Any]

    def __post_init__(self) -> None:
        if self.live_authority_granted:
            raise ValueError("Channel processing readback validation must not grant live authority")
        if self.full_autonomous_ready:
            raise ValueError("Channel processing readback validation must not claim full autonomous readiness")

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["violations"] = list(self.violations)
        return payload


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _bool_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _string(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _enabled_value(value: Any) -> bool:
    return isinstance(value, bool) or (
        isinstance(value, int) and not isinstance(value, bool) and value in (0, 1)
    )


def _append_required_field_violations(payload: Mapping[str, Any], violations: list[str]) -> None:
    for key in CHANNEL_PROCESSING_REQUIRED_TOP_LEVEL_FIELDS:
        if key not in payload:
            violations.append(f"missing_top_level_field:{key}")


def _append_identity_violations(payload: Mapping[str, Any], violations: list[str]) -> None:
    if payload.get("type") != CHANNEL_PROCESSING_PHYSICAL_READBACK_ARTIFACT_TYPE:
        violations.append("artifact_type_mismatch")
    if payload.get("schema_id") != CHANNEL_PROCESSING_PHYSICAL_READBACK_SCHEMA_ID:
        violations.append("schema_id_mismatch")
    if payload.get("surface_id") != CHANNEL_PROCESSING_SURFACE_ID:
        violations.append("surface_id_mismatch")
    if payload.get("remediation_action_id") != CHANNEL_PROCESSING_REMEDIATION_ACTION_ID:
        violations.append("remediation_action_id_mismatch")
    if payload.get("source") not in CHANNEL_PROCESSING_PHYSICAL_READBACK_SOURCES:
        violations.append("source_not_physical_readback")
    if not _string(payload.get("captured_at")):
        violations.append("captured_at_missing")
    if not _string(payload.get("console_model")):
        violations.append("console_model_missing")


def _append_safety_violations(payload: Mapping[str, Any], violations: list[str]) -> None:
    safety = _mapping(payload.get("safety"))
    if safety.get("read_only") is not True:
        violations.append("safety_read_only_not_true")
    for key in CHANNEL_PROCESSING_REQUIRED_FALSE_SAFETY_FLAGS:
        if safety.get(key) is not False:
            violations.append(f"safety_{key}_not_false")
        if _bool_true(payload.get(key)):
            violations.append(f"top_level_{key}_not_false")
    for key in CHANNEL_PROCESSING_FORBIDDEN_MUTATION_EVIDENCE_FIELDS:
        if key in payload:
            violations.append(f"mutation_evidence_field_present:{key}")


def _append_nested_field_violations(
    *,
    channel_index: int,
    mapping: Mapping[str, Any],
    required_fields: Tuple[str, ...],
    label: str,
    violations: list[str],
) -> None:
    for key in required_fields:
        if key not in mapping:
            violations.append(f"channel_{channel_index}_{label}_missing_field:{key}")


def _append_numeric_field_violations(
    *,
    channel_index: int,
    mapping: Mapping[str, Any],
    fields: Tuple[str, ...],
    label: str,
    violations: list[str],
) -> None:
    for key in fields:
        if key in mapping and not _finite_number(mapping.get(key)):
            violations.append(f"channel_{channel_index}_{label}_{key}_not_number")


def _append_enabled_field_violation(
    *,
    channel_index: int,
    mapping: Mapping[str, Any],
    label: str,
    violations: list[str],
) -> None:
    if "enabled" in mapping and not _enabled_value(mapping.get("enabled")):
        violations.append(f"channel_{channel_index}_{label}_enabled_not_bool_or_0_1")


def _append_eq_band_violations(
    *,
    channel_index: int,
    eq: Mapping[str, Any],
    violations: list[str],
) -> None:
    bands = eq.get("bands")
    if not isinstance(bands, list) or len(bands) < MINIMUM_EQ_BAND_COUNT:
        violations.append(f"channel_{channel_index}_eq_bands_missing")
        return
    seen_bands: set[int] = set()
    for band_index, band_value in enumerate(bands):
        band = _mapping(band_value)
        if not band:
            violations.append(f"channel_{channel_index}_eq_band_{band_index}_not_object")
            continue
        for key in CHANNEL_PROCESSING_REQUIRED_EQ_BAND_FIELDS:
            if key not in band:
                violations.append(f"channel_{channel_index}_eq_band_{band_index}_missing_field:{key}")
        band_number = band.get("band")
        if not isinstance(band_number, int) or band_number < 1:
            violations.append(f"channel_{channel_index}_eq_band_{band_index}_band_invalid")
        elif band_number in seen_bands:
            violations.append(f"channel_{channel_index}_eq_band_{band_index}_band_duplicate")
        else:
            seen_bands.add(band_number)
        for key in ("frequency_hz", "gain_db", "q"):
            if key in band and not _finite_number(band.get(key)):
                violations.append(f"channel_{channel_index}_eq_band_{band_index}_{key}_not_number")


def _append_processing_violations(
    *,
    channel_index: int,
    channel: Mapping[str, Any],
    violations: list[str],
) -> None:
    hpf = _mapping(channel.get("hpf"))
    eq = _mapping(channel.get("eq"))
    compressor = _mapping(channel.get("compressor"))
    gate = _mapping(channel.get("gate"))
    coverage = _mapping(channel.get("coverage"))

    _append_nested_field_violations(
        channel_index=channel_index,
        mapping=hpf,
        required_fields=CHANNEL_PROCESSING_REQUIRED_HPF_FIELDS,
        label="hpf",
        violations=violations,
    )
    _append_enabled_field_violation(
        channel_index=channel_index,
        mapping=hpf,
        label="hpf",
        violations=violations,
    )
    _append_numeric_field_violations(
        channel_index=channel_index,
        mapping=hpf,
        fields=("frequency_hz",),
        label="hpf",
        violations=violations,
    )

    _append_nested_field_violations(
        channel_index=channel_index,
        mapping=eq,
        required_fields=CHANNEL_PROCESSING_REQUIRED_EQ_FIELDS,
        label="eq",
        violations=violations,
    )
    _append_enabled_field_violation(
        channel_index=channel_index,
        mapping=eq,
        label="eq",
        violations=violations,
    )
    _append_eq_band_violations(channel_index=channel_index, eq=eq, violations=violations)

    _append_nested_field_violations(
        channel_index=channel_index,
        mapping=compressor,
        required_fields=CHANNEL_PROCESSING_REQUIRED_COMPRESSOR_FIELDS,
        label="compressor",
        violations=violations,
    )
    _append_enabled_field_violation(
        channel_index=channel_index,
        mapping=compressor,
        label="compressor",
        violations=violations,
    )
    for key in ("model", "ratio", "detector"):
        if key in compressor and not _string(compressor.get(key)):
            violations.append(f"channel_{channel_index}_compressor_{key}_missing")
    _append_numeric_field_violations(
        channel_index=channel_index,
        mapping=compressor,
        fields=(
            "threshold_db",
            "attack_ms",
            "release_ms",
            "makeup_db",
            "mix_pct",
            "gain_reduction_db",
        ),
        label="compressor",
        violations=violations,
    )

    _append_nested_field_violations(
        channel_index=channel_index,
        mapping=gate,
        required_fields=CHANNEL_PROCESSING_REQUIRED_GATE_FIELDS,
        label="gate",
        violations=violations,
    )
    _append_enabled_field_violation(
        channel_index=channel_index,
        mapping=gate,
        label="gate",
        violations=violations,
    )
    for key in ("model", "ratio"):
        if key in gate and not _string(gate.get(key)):
            violations.append(f"channel_{channel_index}_gate_{key}_missing")
    _append_numeric_field_violations(
        channel_index=channel_index,
        mapping=gate,
        fields=(
            "threshold_db",
            "range_db",
            "attack_ms",
            "hold_ms",
            "release_ms",
            "accent",
        ),
        label="gate",
        violations=violations,
    )

    _append_nested_field_violations(
        channel_index=channel_index,
        mapping=coverage,
        required_fields=CHANNEL_PROCESSING_REQUIRED_COVERAGE_FLAGS,
        label="coverage",
        violations=violations,
    )
    for key in CHANNEL_PROCESSING_REQUIRED_COVERAGE_FLAGS:
        if coverage.get(key) is not True:
            violations.append(f"channel_{channel_index}_coverage_{key}_not_true")


def _append_channel_violations(payload: Mapping[str, Any], violations: list[str]) -> int:
    channels = payload.get("channels")
    if not isinstance(channels, list) or not channels:
        violations.append("channels_missing")
        return 0

    declared_count = payload.get("channel_count")
    if not isinstance(declared_count, int) or declared_count != len(channels):
        violations.append("channel_count_mismatch")

    seen_channel_ids: set[int] = set()
    for index, channel_value in enumerate(channels):
        channel = _mapping(channel_value)
        if not channel:
            violations.append(f"channel_{index}_not_object")
            continue
        for key in CHANNEL_PROCESSING_REQUIRED_CHANNEL_FIELDS:
            if key not in channel:
                violations.append(f"channel_{index}_missing_field:{key}")

        channel_id = channel.get("channel_id")
        if not isinstance(channel_id, int) or channel_id < 1:
            violations.append(f"channel_{index}_channel_id_invalid")
        elif channel_id in seen_channel_ids:
            violations.append(f"channel_{index}_channel_id_duplicate")
        else:
            seen_channel_ids.add(channel_id)
        if not _string(channel.get("channel_label")):
            violations.append(f"channel_{index}_channel_label_missing")
        if channel.get("status") not in CHANNEL_PROCESSING_VERIFIED_PHYSICAL_STATUSES:
            violations.append(f"channel_{index}_status_not_physical_verified")
        if channel.get("source") not in CHANNEL_PROCESSING_PHYSICAL_READBACK_SOURCES:
            violations.append(f"channel_{index}_source_not_physical_readback")
        if not _string(channel.get("captured_at")):
            violations.append(f"channel_{index}_captured_at_missing")
        _append_processing_violations(
            channel_index=index,
            channel=channel,
            violations=violations,
        )
    return len(channels)


def build_channel_processing_physical_readback_contract() -> Dict[str, Any]:
    """Return the non-executing artifact contract for channel processing proof."""

    return {
        "type": "channel_processing_physical_readback_contract",
        "generated_at": _utc_now(),
        "method_id": CHANNEL_PROCESSING_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
        "schema_id": CHANNEL_PROCESSING_PHYSICAL_READBACK_SCHEMA_ID,
        "status": "contract_ready_physical_fixture_missing",
        "surface_id": CHANNEL_PROCESSING_SURFACE_ID,
        "remediation_action_id": CHANNEL_PROCESSING_REMEDIATION_ACTION_ID,
        "safety": {
            "read_only": True,
            "live_write": False,
            "mutates_console_state": False,
            "queries_console": False,
            "executes_tests": False,
            "live_widening_allowed": False,
            "full_autonomous_ready": False,
        },
        "artifact_contract": {
            "type": CHANNEL_PROCESSING_PHYSICAL_READBACK_ARTIFACT_TYPE,
            "schema_id": CHANNEL_PROCESSING_PHYSICAL_READBACK_SCHEMA_ID,
            "required_top_level_fields": list(CHANNEL_PROCESSING_REQUIRED_TOP_LEVEL_FIELDS),
            "required_channel_fields": list(CHANNEL_PROCESSING_REQUIRED_CHANNEL_FIELDS),
            "required_hpf_fields": list(CHANNEL_PROCESSING_REQUIRED_HPF_FIELDS),
            "required_eq_fields": list(CHANNEL_PROCESSING_REQUIRED_EQ_FIELDS),
            "required_eq_band_fields": list(CHANNEL_PROCESSING_REQUIRED_EQ_BAND_FIELDS),
            "required_compressor_fields": list(CHANNEL_PROCESSING_REQUIRED_COMPRESSOR_FIELDS),
            "required_gate_fields": list(CHANNEL_PROCESSING_REQUIRED_GATE_FIELDS),
            "required_coverage_flags": list(CHANNEL_PROCESSING_REQUIRED_COVERAGE_FLAGS),
            "required_false_safety_flags": list(CHANNEL_PROCESSING_REQUIRED_FALSE_SAFETY_FLAGS),
            "allowed_sources": list(CHANNEL_PROCESSING_PHYSICAL_READBACK_SOURCES),
            "accepted_channel_statuses": list(CHANNEL_PROCESSING_VERIFIED_PHYSICAL_STATUSES),
            "forbidden_mutation_evidence_fields": list(
                CHANNEL_PROCESSING_FORBIDDEN_MUTATION_EVIDENCE_FIELDS
            ),
            "minimum_channel_count": 1,
            "minimum_eq_band_count": MINIMUM_EQ_BAND_COUNT,
        },
        "summary": {
            "physical_readback_surface_id": CHANNEL_PROCESSING_SURFACE_ID,
            "physical_readback_action_id": CHANNEL_PROCESSING_REMEDIATION_ACTION_ID,
            "artifact_required": True,
            "physical_fixture_gap_remains": True,
            "accepted_as_physical_fixture_candidate": False,
            "live_authority_granted": False,
            "promotion_families_requiring_surface": [
                "eq",
                "compressor",
                "gate",
                "deessing",
                "frequency_separation",
                "cross_channel_unmasking",
            ],
            "claim": (
                "contract only; a separate operator-run channel processing "
                "physical readback artifact is still required"
            ),
        },
    }


def validate_channel_processing_physical_readback_artifact(
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    """Validate a future channel processing artifact without granting authority."""

    violations: list[str] = []
    artifact = _mapping(payload)
    _append_required_field_violations(artifact, violations)
    _append_identity_violations(artifact, violations)
    _append_safety_violations(artifact, violations)
    channel_count = _append_channel_violations(artifact, violations)

    status = (
        "valid_physical_readback_artifact"
        if not violations
        else "invalid_physical_readback_artifact"
    )
    validation = ChannelProcessingPhysicalReadbackValidation(
        method_id=CHANNEL_PROCESSING_PHYSICAL_READBACK_CONTRACT_METHOD_ID,
        schema_id=CHANNEL_PROCESSING_PHYSICAL_READBACK_SCHEMA_ID,
        surface_id=str(artifact.get("surface_id") or ""),
        remediation_action_id=str(artifact.get("remediation_action_id") or ""),
        status=status,
        accepted_as_physical_fixture_candidate=not violations,
        live_authority_granted=False,
        full_autonomous_ready=False,
        channel_count=channel_count,
        violations=tuple(sorted(set(violations))),
        evidence={
            "source": artifact.get("source"),
            "captured_at": artifact.get("captured_at"),
            "console_model": artifact.get("console_model"),
            "channel_count": artifact.get("channel_count"),
            "validated_at": _utc_now(),
        },
    )
    return validation.to_dict()
