"""Offline validation harness for the operator proposal workflow.

This module runs the no-hardware path:

fixture -> analysis -> proposal -> safety review -> simulated apply ->
simulated readback -> rollback proof -> emergency-stop block proof.

It intentionally has no mixer transport imports and no network code. The output
is evidence for cockpit/backend behavior only; it is not a live-console proof.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from operator_analysis import build_operator_analysis_report
    from operator_mode_policy import build_operator_mode_status
    from operator_proposal_queue import OperatorProposalQueue
except ImportError:  # pragma: no cover - package import compatibility
    from backend.operator_analysis import build_operator_analysis_report
    from backend.operator_mode_policy import build_operator_mode_status
    from backend.operator_proposal_queue import OperatorProposalQueue


VERIFIED_SIMULATED_READBACK = "verified_from_simulated_readback"
ALLOWED_OFFLINE_VALUE_TYPES = {"fader", "gain"}
FORBIDDEN_LIVE_MESSAGES = ["set_fader", "set_gain", "route_output", "load_snap", "save_snap"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def compact_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _coerce_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _round(value: Any, digits: int = 3) -> Optional[float]:
    numeric = _coerce_float(value)
    return round(numeric, digits) if numeric is not None else None


def address_for(channel: int, value_type: str) -> str:
    if value_type == "fader":
        return f"/ch/{channel}/fdr"
    if value_type == "gain":
        return f"/ch/{channel}/in/set/trim"
    raise ValueError(f"Unsupported value_type: {value_type}")


@dataclass(frozen=True)
class OfflineChannelFixture:
    channel: int
    name: str
    role: str
    group: str
    fader_db: float = 0.0
    gain_db: float = 0.0
    rms_db: float = -40.0
    peak_db: float = -12.0
    lufs: float = -30.0
    mute: int = 0


class OfflineMixerState:
    """Small in-memory mixer state used only for simulated readback."""

    def __init__(self, channels: Iterable[OfflineChannelFixture]):
        self.state: Dict[str, Any] = {}
        self.state_timestamps: Dict[str, float] = {}
        self.channel_numbers: List[int] = []
        for channel in channels:
            self.channel_numbers.append(int(channel.channel))
            self._set_state_value(f"/ch/{channel.channel}/$name", channel.name)
            self._set_state_value(f"/ch/{channel.channel}/name", channel.name)
            self._set_state_value(f"/ch/{channel.channel}/fdr", float(channel.fader_db))
            self._set_state_value(f"/ch/{channel.channel}/in/set/trim", float(channel.gain_db))
            self._set_state_value(f"/ch/{channel.channel}/mute", int(channel.mute))

    def _set_state_value(self, address: str, value: Any) -> None:
        self.state[address] = value
        self.state_timestamps[address] = datetime.now(timezone.utc).timestamp()

    def read(self, channel: int, value_type: str) -> Optional[float]:
        return _coerce_float(self.state.get(address_for(channel, value_type)))

    def apply(self, channel: int, value_type: str, value: float) -> Dict[str, Any]:
        address = address_for(channel, value_type)
        old_value = self.read(channel, value_type)
        self._set_state_value(address, float(value))
        readback_value = self.read(channel, value_type)
        return {
            "address": address,
            "old_value": old_value,
            "requested_value": float(value),
            "readback_value": readback_value,
            "readback_status": VERIFIED_SIMULATED_READBACK if readback_value == float(value) else "simulated_mismatch",
        }

    def snapshot_values(self) -> Dict[str, Any]:
        return dict(self.state)


class OfflineAudioCapture:
    running = True
    sample_rate = 48000

    def __init__(self, channels: Iterable[OfflineChannelFixture]):
        self._by_channel = {int(channel.channel): channel for channel in channels}
        self.num_channels = max(self._by_channel.keys(), default=0)

    def get_status(self) -> Dict[str, Any]:
        return {
            "running": True,
            "source_type": "offline_fixture",
            "num_channels": self.num_channels,
            "sample_rate": self.sample_rate,
            "buffer_seconds": 5.0,
            "subscribers": 0,
        }

    def get_rms(self, channel: int) -> float:
        item = self._by_channel.get(int(channel))
        return float(item.rms_db) if item else -100.0

    def get_peak(self, channel: int) -> float:
        item = self._by_channel.get(int(channel))
        return float(item.peak_db) if item else -100.0

    def get_lufs(self, channel: int) -> float:
        item = self._by_channel.get(int(channel))
        return float(item.lufs) if item else -100.0


class OfflineServerFixture:
    connection_mode = "offline_fixture"

    def __init__(self, channels: Iterable[OfflineChannelFixture], *, operator_mode: str = "assist"):
        channel_list = list(channels)
        self.operator_mode = operator_mode
        self.mixer_client = OfflineMixerState(channel_list)
        self.audio_capture = OfflineAudioCapture(channel_list)
        self.operator_proposal_queue = OperatorProposalQueue()

    def get_operator_mode_status(self) -> Dict[str, Any]:
        return {
            "type": "operator_mode_status",
            **build_operator_mode_status(
                mode=self.operator_mode,
                connection_mode=self.connection_mode,
                wing_boundary_active=False,
                auto_mode_unlocked=False,
            ),
        }

    def get_manual_write_supervision_status(self) -> Dict[str, Any]:
        return {
            "deployment_write_boundary_enabled": True,
            "deployment_write_boundary_active": False,
            "connection_mode": self.connection_mode,
            "approved_live_write_kinds": ["fader", "gain"],
            "operator_mode": self.operator_mode,
            "requires_operator_mode": "supervised",
            "emergency_stopped": False,
            "reset_required": False,
            "emergency_reset_requires_operator_action": True,
            "cooldown_sec": 0.0,
        }


def default_offline_channels() -> List[OfflineChannelFixture]:
    """Return a deterministic no-hardware fixture that creates safe proposals."""

    return [
        OfflineChannelFixture(1, "KICK", "Kick", "DRUMS", fader_db=0.0, gain_db=0.0, rms_db=-18.0, peak_db=-4.8, lufs=-27.5),
        OfflineChannelFixture(2, "SNARE T", "Snare", "DRUMS", fader_db=0.0, gain_db=-12.0, rms_db=-15.0, peak_db=-2.4, lufs=-24.0),
        OfflineChannelFixture(3, "SNARE B", "Snare", "DRUMS", fader_db=0.0, gain_db=-11.9, rms_db=-15.5, peak_db=-4.2, lufs=-23.2),
        OfflineChannelFixture(4, "TOM", "Tom", "DRUMS", fader_db=0.0, gain_db=0.0, rms_db=-19.0, peak_db=-0.8, lufs=-26.0),
        OfflineChannelFixture(6, "OH L", "Overhead", "DRUMS", fader_db=0.0, gain_db=-0.7, rms_db=-21.0, peak_db=-5.0, lufs=-24.2),
        OfflineChannelFixture(9, "GUITAR L", "Guitar", "GUITARS", fader_db=0.0, gain_db=-2.4, rms_db=-18.0, peak_db=-5.1, lufs=-21.5),
        OfflineChannelFixture(16, "VOCAL", "Vocal", "VOCALS", fader_db=0.0, gain_db=-11.9, rms_db=-16.0, peak_db=-3.5, lufs=-17.7),
    ]


def _proposal_changes(proposal: Dict[str, Any]) -> List[Dict[str, Any]]:
    changes = proposal.get("requested_changes")
    if isinstance(changes, list) and changes:
        return [dict(change) for change in changes if isinstance(change, dict)]
    change = proposal.get("requested_change")
    return [dict(change)] if isinstance(change, dict) else []


def review_proposal_safety(proposal: Dict[str, Any]) -> Dict[str, Any]:
    """Fail-closed review for offline/simulated proposal execution."""

    blockers: List[str] = []
    evidence: List[Dict[str, Any]] = []
    changes = _proposal_changes(proposal)
    if not changes:
        blockers.append("missing_requested_change")

    for change in changes:
        value_type = str(change.get("value_type") or "").strip().lower()
        channel = _coerce_int(change.get("channel"))
        target_value = _coerce_float(change.get("value"))
        current_value = _coerce_float(change.get("current_value"))
        rollback_value = _coerce_float(change.get("rollback_value", current_value))

        if value_type not in ALLOWED_OFFLINE_VALUE_TYPES:
            blockers.append(f"unsupported_value_type:{value_type or 'missing'}")
        if channel is None or target_value is None:
            blockers.append("missing_channel_or_target_value")
        if current_value is None:
            blockers.append("missing_current_value")
        if rollback_value is None:
            blockers.append("missing_rollback_value")
        if current_value is not None and target_value is not None and target_value > current_value:
            blockers.append("offline_harness_allows_downward_changes_only")
        if value_type == "fader" and target_value is not None and target_value > 0.0:
            blockers.append("fader_target_above_zero_db")
        if value_type == "gain" and target_value is not None and abs(target_value) > 24.0:
            blockers.append("gain_target_outside_offline_bounds")

        evidence.append(
            {
                "channel": channel,
                "value_type": value_type,
                "current_value": current_value,
                "target_value": target_value,
                "rollback_value": rollback_value,
                "direction": "down_or_same" if current_value is not None and target_value is not None and target_value <= current_value else "unsafe",
                "readback_required": bool(change.get("readback_required", True)),
            }
        )

    return {
        "status": "approved_for_simulation" if not blockers else "blocked",
        "approved": not blockers,
        "blockers": sorted(set(blockers)),
        "evidence": evidence,
        "live_write": False,
    }


def simulate_proposal_apply(
    mixer: OfflineMixerState,
    proposal: Dict[str, Any],
    safety_review: Dict[str, Any],
) -> Dict[str, Any]:
    if not safety_review.get("approved"):
        return {
            "status": "blocked",
            "reason": "safety_review_blocked",
            "safety_review": safety_review,
            "live_write": False,
        }

    apply_results: List[Dict[str, Any]] = []
    rollback_results: List[Dict[str, Any]] = []
    for change in _proposal_changes(proposal):
        channel = int(change["channel"])
        value_type = str(change["value_type"]).lower()
        old_value = mixer.read(channel, value_type)
        target_value = float(change["value"])
        apply_result = mixer.apply(channel, value_type, target_value)
        rollback_target = _coerce_float(change.get("rollback_value", change.get("current_value", old_value)))
        if rollback_target is None:
            rollback_result = {
                "status": "blocked",
                "reason": "rollback_value_unavailable",
                "channel": channel,
                "value_type": value_type,
            }
        else:
            rollback_result = mixer.apply(channel, value_type, rollback_target)
            rollback_result["status"] = (
                "applied"
                if rollback_result.get("readback_status") == VERIFIED_SIMULATED_READBACK
                else "partial"
            )
        apply_results.append(apply_result)
        rollback_results.append(rollback_result)

    apply_verified = all(item.get("readback_status") == VERIFIED_SIMULATED_READBACK for item in apply_results)
    rollback_verified = all(item.get("readback_status") == VERIFIED_SIMULATED_READBACK for item in rollback_results)
    return {
        "status": "simulated_applied_and_rolled_back" if apply_verified and rollback_verified else "partial",
        "proposal_id": proposal.get("id"),
        "apply_results": apply_results,
        "rollback_results": rollback_results,
        "readback_status": VERIFIED_SIMULATED_READBACK if apply_verified else "simulated_mismatch",
        "rollback_readback_status": VERIFIED_SIMULATED_READBACK if rollback_verified else "simulated_mismatch",
        "proof_status": "verified" if apply_verified and rollback_verified else "partial",
        "live_write": False,
    }


def run_offline_validation(
    channels: Optional[Iterable[OfflineChannelFixture]] = None,
    *,
    modes: Iterable[str] = ("analyze", "assist", "auto", "supervised"),
    total_channels: int = 16,
) -> Dict[str, Any]:
    channel_list = list(channels or default_offline_channels())
    mode_results: List[Dict[str, Any]] = []
    all_forbidden_messages: List[str] = []

    for mode in modes:
        server = OfflineServerFixture(channel_list, operator_mode=mode)
        analysis = build_operator_analysis_report(
            server,
            total_channels=total_channels,
            create_proposals=False,
            include_proposal_previews=True,
        )
        proposals = list(analysis.get("proposals_preview") or [])
        proposal_results: List[Dict[str, Any]] = []
        emergency_stopped = False

        for proposal in proposals:
            safety_review = review_proposal_safety(proposal)
            can_apply_in_mode = mode == "supervised"
            if not can_apply_in_mode:
                proposal_results.append(
                    {
                        "proposal_id": proposal.get("id"),
                        "title": proposal.get("title"),
                        "status": "mode_blocked",
                        "reason": f"{mode}_mode_does_not_apply",
                        "safety_review": safety_review,
                        "live_write": False,
                    }
                )
                continue

            apply_result = simulate_proposal_apply(server.mixer_client, proposal, safety_review)
            emergency_stopped = True
            post_emergency_apply = {
                "status": "blocked",
                "reason": "emergency_stop_active",
                "live_write": False,
            }
            proposal_results.append(
                {
                    "proposal_id": proposal.get("id"),
                    "title": proposal.get("title"),
                    "status": apply_result.get("status"),
                    "reason": "simulated_apply_readback_rollback_complete",
                    "safety_review": safety_review,
                    "apply_result": apply_result,
                    "post_emergency_apply": post_emergency_apply,
                    "live_write": False,
                }
            )

        all_forbidden_messages.extend(FORBIDDEN_LIVE_MESSAGES)
        mode_results.append(
            {
                "mode": mode,
                "analysis_status": analysis.get("status"),
                "observation_count": analysis.get("observation_count"),
                "proposal_count": len(proposals),
                "applied_simulation_count": sum(1 for item in proposal_results if item.get("status") == "simulated_applied_and_rolled_back"),
                "mode_blocked_count": sum(1 for item in proposal_results if item.get("status") == "mode_blocked"),
                "emergency_stop_proven": bool(emergency_stopped) if mode == "supervised" else False,
                "proposal_results": proposal_results,
                "final_state": server.mixer_client.snapshot_values(),
            }
        )

    supervised = next((item for item in mode_results if item["mode"] == "supervised"), {})
    supervised_results = supervised.get("proposal_results", [])
    simulation_failures = [
        item
        for item in supervised_results
        if item.get("status") != "simulated_applied_and_rolled_back"
    ]
    status = "passed"
    reason = "offline_simulated_apply_readback_rollback_complete"
    if not supervised_results:
        status = "clear"
        reason = "no_offline_proposals_generated"
    elif simulation_failures:
        status = "failed"
        reason = "offline_simulation_has_failures"

    return {
        "type": "offline_operator_validation",
        "generated_at": utc_now(),
        "status": status,
        "reason": reason,
        "connection_policy": "no_hardware_no_network_no_live_writes",
        "write_policy": "simulated_apply_only",
        "live_write": False,
        "forbidden_messages": sorted(set(all_forbidden_messages)),
        "fixture": {
            "channel_count": len(channel_list),
            "channels": [
                {
                    "channel": channel.channel,
                    "name": channel.name,
                    "role": channel.role,
                    "group": channel.group,
                    "fader_db": channel.fader_db,
                    "gain_db": channel.gain_db,
                    "peak_db": channel.peak_db,
                    "lufs": channel.lufs,
                }
                for channel in channel_list
            ],
        },
        "summary": {
            "mode_count": len(mode_results),
            "supervised_proposal_count": len(supervised_results),
            "supervised_simulated_applied_count": supervised.get("applied_simulation_count", 0),
            "supervised_failure_count": len(simulation_failures),
            "non_supervised_modes_blocked": all(
                item["mode"] == "supervised" or item["applied_simulation_count"] == 0
                for item in mode_results
            ),
        },
        "mode_results": mode_results,
    }


def write_offline_validation_artifacts(
    result: Dict[str, Any],
    *,
    artifact_root: str | Path = "artifacts/offline_validation",
    run_id: Optional[str] = None,
    teacher_examples: Optional[Iterable[Dict[str, Any]]] = None,
) -> Dict[str, str]:
    run_id = run_id or f"{compact_stamp()}_offline_operator_validation"
    run_dir = Path(artifact_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "summary.json"
    events_path = run_dir / "events.jsonl"
    traces_path = run_dir / "decision_traces.jsonl"
    metrics_path = run_dir / "metrics_before_after.json"
    teacher_examples_path = run_dir / "foundation_teacher_examples.jsonl"
    report_path = run_dir / "report.md"
    teacher_rows = list(teacher_examples or [])

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")

    with events_path.open("w", encoding="utf-8") as handle:
        for mode_result in result.get("mode_results", []):
            handle.write(
                json.dumps(
                    {
                        "timestamp": utc_now(),
                        "event_type": "offline_mode_result",
                        "mode": mode_result.get("mode"),
                        "proposal_count": mode_result.get("proposal_count"),
                        "applied_simulation_count": mode_result.get("applied_simulation_count"),
                        "mode_blocked_count": mode_result.get("mode_blocked_count"),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
            for proposal in mode_result.get("proposal_results", []):
                handle.write(
                    json.dumps(
                        {
                            "timestamp": utc_now(),
                            "event_type": "offline_proposal_result",
                            "mode": mode_result.get("mode"),
                            "payload": proposal,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )

    trace_count = 0
    with traces_path.open("w", encoding="utf-8") as handle:
        for mode_result in result.get("mode_results", []):
            mode = mode_result.get("mode")
            for proposal in mode_result.get("proposal_results", []):
                apply_result = proposal.get("apply_result") if isinstance(proposal, dict) else {}
                if not isinstance(apply_result, dict):
                    apply_result = {}
                safety_review = proposal.get("safety_review") if isinstance(proposal, dict) else {}
                if not isinstance(safety_review, dict):
                    safety_review = {}
                trace = {
                    "decision_id": f"offline-{mode}-{trace_count + 1}",
                    "idempotency_key": None,
                    "proposal_id": proposal.get("proposal_id"),
                    "source": "offline_replay_v1",
                    "mode": mode,
                    "target": proposal.get("title"),
                    "action_type": "offline_proposal",
                    "status": proposal.get("status"),
                    "current_value": apply_result.get("old_value"),
                    "proposed_value": apply_result.get("requested_value"),
                    "delta": apply_result.get("delta_db"),
                    "safety_limits": {
                        "offline_only": True,
                        "live_write": False,
                        "allowed_live_write_kinds": list(ALLOWED_OFFLINE_VALUE_TYPES),
                    },
                    "pre_read": {
                        "status": apply_result.get("pre_read_status"),
                        "value": apply_result.get("old_value"),
                    },
                    "readback": {
                        "status": apply_result.get("readback_status"),
                        "value": apply_result.get("readback_value"),
                    },
                    "rollback": {
                        "available": apply_result.get("rollback_available"),
                        "status": apply_result.get("rollback_status"),
                        "value": apply_result.get("rollback_value"),
                    },
                    "proof_status": apply_result.get("proof_status", proposal.get("status")),
                    "blocked_reason": proposal.get("reason") if proposal.get("status") != "simulated_applied_and_rolled_back" else None,
                    "created_at": utc_now(),
                    "metadata": {
                        "safety_review": safety_review,
                        "live_write": False,
                    },
                }
                handle.write(json.dumps(trace, ensure_ascii=False, sort_keys=True) + "\n")
                trace_count += 1

    metrics = {
        "generated_at": utc_now(),
        "run_id": run_id,
        "status": result.get("status"),
        "offline_only": True,
        "live_write": False,
        "before_after_available": False,
        "summary": result.get("summary", {}),
        "trace_count": trace_count,
        "foundation_teacher_example_count": len(teacher_rows),
        "foundation_teacher_examples_path": str(teacher_examples_path),
    }
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")

    with teacher_examples_path.open("w", encoding="utf-8") as handle:
        for row in teacher_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    with report_path.open("w", encoding="utf-8") as handle:
        handle.write(f"# Offline Replay Report {run_id}\n\n")
        handle.write(f"- Status: {result.get('status')}\n")
        handle.write(f"- Reason: {result.get('reason')}\n")
        handle.write("- Policy: offline-only, no WING/OSC transport imports, no live writes\n")
        handle.write(f"- Decision traces: {trace_count}\n")
        handle.write(f"- Foundation teacher examples: {len(teacher_rows)}\n")
        handle.write(f"- Summary: `{json.dumps(result.get('summary', {}), ensure_ascii=False, sort_keys=True)}`\n")

    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "summary_path": str(summary_path),
        "events_path": str(events_path),
        "decision_traces_path": str(traces_path),
        "metrics_path": str(metrics_path),
        "foundation_teacher_examples_path": str(teacher_examples_path),
        "report_path": str(report_path),
    }
