"""Fail-closed contract for no-live-transport integration evidence.

This contract validates offline evidence that an integration candidate did not
open live transport, contact mixer hardware or send live writes. Passing
results can support offline integration promotion work, but they cannot grant
live authority, close readiness or replace real hardware proof.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping


SCHEMA_VERSION = "automixer.no_live_transport_contract.v1"

REQUIRED_TRANSPORT_CONTROLS = (
    "offline_executor_only",
    "artifact_only_outputs",
    "no_wing_client",
    "no_osc_client",
    "no_websocket_session",
    "no_midi_transport",
    "no_backend_contact",
    "no_hardware_contact",
    "no_live_write_requests",
    "audit_log_required",
)
REQUIRED_TRANSPORT_EVIDENCE = (
    "source_audit_artifact",
    "runtime_config_artifact",
    "network_trace_artifact",
    "artifact_manifest",
    "operator_report_reference",
    "audit_reference",
)
REQUIRED_ZERO_RUNTIME_OBSERVATIONS = (
    "websocket_opened",
    "osc_messages_sent",
    "live_write_requests_sent",
    "mixer_client_calls",
    "network_connections_opened",
)


def _as_bool(value: Any) -> bool:
    return bool(value)


def _as_list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _dedupe(items: Iterable[str]) -> List[str]:
    return sorted(dict.fromkeys(items))


def _nonempty_string(value: Any) -> bool:
    return bool(str(value or "").strip())


def _passed(value: Any) -> bool:
    if value is True:
        return True
    if not isinstance(value, Mapping):
        return False
    status = str(value.get("status") or value.get("result") or "").strip().lower()
    return status in {"passed", "approved", "complete"}


def _evidence_bound(value: Any) -> bool:
    if not isinstance(value, Mapping) or not _passed(value):
        return False
    return any(
        _nonempty_string(value.get(key))
        for key in ("artifact", "artifact_path", "audit_reference", "sha256")
    )


def _is_zero_or_false(value: Any) -> bool:
    if isinstance(value, bool):
        return value is False
    if isinstance(value, (int, float)):
        return value == 0
    return False


def evaluate_no_live_transport_contract(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    """Evaluate offline evidence that no live transport was opened."""

    manifest = _as_mapping(manifest)
    blockers: List[str] = []

    offline_only = _as_bool(manifest.get("offline_only"))
    live_write = _as_bool(manifest.get("live_write"))
    contacts_wing = _as_bool(manifest.get("contacts_wing"))
    contacts_backend = _as_bool(manifest.get("contacts_backend"))
    hardware_evidence = _as_bool(manifest.get("hardware_evidence"))
    claimed_readiness_closure = _as_bool(manifest.get("readiness_closure_allowed"))
    claimed_live_authority = _as_bool(manifest.get("live_authority_granted"))

    if not offline_only:
        blockers.append("offline_only_required")
    if live_write:
        blockers.append("no_live_transport_forbids_live_write")
    if contacts_wing:
        blockers.append("no_live_transport_forbids_wing_contact")
    if contacts_backend:
        blockers.append("no_live_transport_forbids_backend_contact")
    if hardware_evidence:
        blockers.append("no_live_transport_must_not_be_labeled_hardware")
    if claimed_readiness_closure:
        blockers.append("no_live_transport_cannot_close_readiness")
    if claimed_live_authority:
        blockers.append("no_live_transport_cannot_grant_live_authority")

    transport_controls = _as_mapping(manifest.get("transport_controls"))
    missing_transport_controls = [
        item
        for item in REQUIRED_TRANSPORT_CONTROLS
        if not _passed(transport_controls.get(item))
    ]
    blockers.extend(
        f"missing_transport_control:{item}" for item in missing_transport_controls
    )

    transport_evidence = _as_mapping(manifest.get("transport_evidence"))
    missing_transport_evidence = [
        item
        for item in REQUIRED_TRANSPORT_EVIDENCE
        if not _evidence_bound(transport_evidence.get(item))
    ]
    blockers.extend(
        f"missing_transport_evidence:{item}" for item in missing_transport_evidence
    )

    runtime_observations = _as_mapping(manifest.get("runtime_observations"))
    missing_runtime_observations: List[str] = []
    nonzero_runtime_observations: List[str] = []
    for item in REQUIRED_ZERO_RUNTIME_OBSERVATIONS:
        if item not in runtime_observations:
            missing_runtime_observations.append(item)
        elif not _is_zero_or_false(runtime_observations.get(item)):
            nonzero_runtime_observations.append(item)
    blockers.extend(
        f"missing_runtime_observation:{item}"
        for item in missing_runtime_observations
    )
    blockers.extend(
        f"live_transport_observed:{item}" for item in nonzero_runtime_observations
    )

    forbidden_transport_tokens = [
        str(item).strip()
        for item in _as_list(manifest.get("forbidden_transport_tokens"))
        if str(item).strip()
    ]
    blockers.extend(
        f"forbidden_transport_token:{item}" for item in forbidden_transport_tokens
    )

    blocker_list = _dedupe(blockers)
    ready_for_no_live_transport_use = not blocker_list

    return {
        "schema": SCHEMA_VERSION,
        "transport_id": str(manifest.get("transport_id") or ""),
        "status": "ready_for_no_live_transport_use"
        if ready_for_no_live_transport_use
        else "blocked",
        "ready_for_no_live_transport_use": ready_for_no_live_transport_use,
        "offline_only": offline_only,
        "hardware_evidence": hardware_evidence,
        "live_write": live_write,
        "authority": {
            "live_authority_granted": False,
            "readiness_closure_allowed": False,
            "contacts_wing": contacts_wing,
            "contacts_backend": contacts_backend,
        },
        "summary": {
            "required_transport_controls": list(REQUIRED_TRANSPORT_CONTROLS),
            "missing_transport_controls": missing_transport_controls,
            "required_transport_evidence": list(REQUIRED_TRANSPORT_EVIDENCE),
            "missing_transport_evidence": missing_transport_evidence,
            "required_zero_runtime_observations": list(
                REQUIRED_ZERO_RUNTIME_OBSERVATIONS
            ),
            "missing_runtime_observations": missing_runtime_observations,
            "nonzero_runtime_observations": nonzero_runtime_observations,
            "forbidden_transport_tokens": forbidden_transport_tokens,
        },
        "blockers": blocker_list,
    }
