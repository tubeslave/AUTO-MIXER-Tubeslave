"""Replay write-intent adapter for dry-run-first WING automation paths."""

from __future__ import annotations

from itertools import count
from typing import Any, Dict, Optional

from ai_mixing_pipeline.decision_layer.replay_proposal_ranking import (
    build_fader_proposals,
    build_gain_proposals,
    build_replay_graph_checkpoint,
    build_replay_timeline_artifact,
    rank_replay_proposals,
)
from soundcheck_autonomy_contract import build_action_envelope


def _compact(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


class ReplayWriteIntentAdapter:
    """Emit replay-safe intent artifacts before legacy automation can touch WING."""

    def __init__(self, *, source_system: str):
        self.source_system = str(source_system or "unknown")
        self._records: list[dict[str, Any]] = []
        self._sequence = count(1)

    def list_records(self) -> list[dict[str, Any]]:
        return [dict(record) for record in self._records]

    def emit_fader_intent(
        self,
        *,
        mixer_client: Any,
        mixer_type: Optional[str],
        channel: int,
        current_fader_db: float,
        target_fader_db: float,
        readback_source: str,
        context_metadata: Optional[dict[str, Any]] = None,
        extra_payload: Optional[dict[str, Any]] = None,
        replay_correlation_id: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        payload = _compact(
            {
                "source": self.source_system,
                "mixer_channel": int(channel),
                "current_fader_db": float(current_fader_db),
                "recommended_target_fader_db": float(target_fader_db),
                "confidence": 1.0,
                "readback_source": str(readback_source or "controller_snapshot"),
                "replay_correlation_id": replay_correlation_id
                or self._next_correlation_id("fader", channel),
                "requires_confirmation": True,
                "approval_status": "pending",
                "live_apply": False,
                "requested_live_write": True,
                "auto_apply_blocked": True,
                "safety": {
                    "requires_confirmation": True,
                    "approval_status": "pending",
                    "write_intent_only": True,
                    "console_family": "wing",
                    "write_kind": "fader",
                },
            }
        )
        if extra_payload:
            payload.update(_compact(dict(extra_payload)))
        payload["action_envelope"] = build_action_envelope(
            envelope_id=f"fader::{payload['replay_correlation_id']}",
            source_system=self.source_system,
            kind="fader",
            family="fader_adjustment",
            action_type="set_fader_db",
            target={"mixer_channel": int(channel)},
            current_state={"fader_db": float(current_fader_db)},
            requested_state={"fader_db": float(target_fader_db)},
            confidence=float(payload.get("confidence", 1.0)),
            replay_correlation_id=str(payload["replay_correlation_id"]),
            context_metadata=context_metadata,
            metadata=payload,
            rationale={},
            audit={"readback_source": str(readback_source or "controller_snapshot")},
        )
        payload["autonomy_mode"] = payload["action_envelope"]["autonomy_mode"]
        payload["trust_gates"] = payload["action_envelope"]["trust_gates"]
        payload["required_approval_level"] = payload["action_envelope"]["required_approval_level"]
        payload["rollback_handle"] = payload["action_envelope"]["rollback_handle"]
        payload["risk"] = payload["action_envelope"]["risk"]
        return self._emit(
            mixer_client=mixer_client,
            mixer_type=mixer_type,
            write_kind="fader",
            channel=channel,
            proposal_builder=build_fader_proposals,
            proposal_arg_name="fader_updates",
            payload=payload,
            context_metadata=context_metadata,
        )

    def emit_gain_intent(
        self,
        *,
        mixer_client: Any,
        mixer_type: Optional[str],
        channel: int,
        current_trim_db: float,
        target_trim_db: float,
        readback_source: str,
        context_metadata: Optional[dict[str, Any]] = None,
        extra_payload: Optional[dict[str, Any]] = None,
        replay_correlation_id: Optional[str] = None,
    ) -> Optional[dict[str, Any]]:
        payload = _compact(
            {
                "source": self.source_system,
                "mixer_channel": int(channel),
                "current_trim_db": float(current_trim_db),
                "recommended_target_trim_db": float(target_trim_db),
                "confidence": 1.0,
                "readback_source": str(readback_source or "controller_snapshot"),
                "replay_correlation_id": replay_correlation_id
                or self._next_correlation_id("gain", channel),
                "requires_confirmation": True,
                "approval_status": "pending",
                "live_apply": False,
                "requested_live_write": True,
                "auto_apply_blocked": True,
                "safety": {
                    "requires_confirmation": True,
                    "approval_status": "pending",
                    "write_intent_only": True,
                    "console_family": "wing",
                    "write_kind": "gain",
                },
            }
        )
        if extra_payload:
            payload.update(_compact(dict(extra_payload)))
        payload["action_envelope"] = build_action_envelope(
            envelope_id=f"gain::{payload['replay_correlation_id']}",
            source_system=self.source_system,
            kind="input_gain",
            family="gain_adjustment",
            action_type="set_input_trim_db",
            target={"mixer_channel": int(channel)},
            current_state={"trim_db": float(current_trim_db)},
            requested_state={"trim_db": float(target_trim_db)},
            confidence=float(payload.get("confidence", 1.0)),
            replay_correlation_id=str(payload["replay_correlation_id"]),
            context_metadata=context_metadata,
            metadata=payload,
            rationale={},
            audit={"readback_source": str(readback_source or "controller_snapshot")},
        )
        payload["autonomy_mode"] = payload["action_envelope"]["autonomy_mode"]
        payload["trust_gates"] = payload["action_envelope"]["trust_gates"]
        payload["required_approval_level"] = payload["action_envelope"]["required_approval_level"]
        payload["rollback_handle"] = payload["action_envelope"]["rollback_handle"]
        payload["risk"] = payload["action_envelope"]["risk"]
        return self._emit(
            mixer_client=mixer_client,
            mixer_type=mixer_type,
            write_kind="gain",
            channel=channel,
            proposal_builder=build_gain_proposals,
            proposal_arg_name="recommendations",
            payload=payload,
            context_metadata=context_metadata,
        )

    def _emit(
        self,
        *,
        mixer_client: Any,
        mixer_type: Optional[str],
        write_kind: str,
        channel: int,
        proposal_builder,
        proposal_arg_name: str,
        payload: dict[str, Any],
        context_metadata: Optional[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        if not self._is_wing_target(mixer_client, mixer_type):
            return None

        proposals = proposal_builder(**{proposal_arg_name: [payload]})
        proposal = proposals[0]
        ranking = rank_replay_proposals(proposals)
        metadata = _compact(
            {
                "source_system": self.source_system,
                "write_kind": write_kind,
                "mixer_channel": int(channel),
                "transport_decision": "dry_run_only",
                "target_console": "wing",
                **dict(context_metadata or {}),
            }
        )
        timeline_artifact = build_replay_timeline_artifact(
            proposals,
            ranking,
            metadata=metadata,
        )
        graph_checkpoint = build_replay_graph_checkpoint(
            proposals,
            ranking,
            metadata=metadata,
        )
        record = {
            "source_system": self.source_system,
            "write_kind": write_kind,
            "mixer_channel": int(channel),
            "decision": "dry_run_only",
            "allow_live_write": False,
            "replay_correlation_id": proposal.replay_correlation_id,
            "transport_policy": timeline_artifact["events"][0]["transport_policy"],
            "live_mixer_writes": timeline_artifact["events"][0]["live_mixer_writes"],
            "proposal": proposal.to_dict(),
            "ranking": ranking.to_dict(),
            "timeline_artifact": timeline_artifact,
            "graph_checkpoint": graph_checkpoint.to_dict(),
            "metadata": metadata,
        }
        self._records.append(record)
        return record

    def _next_correlation_id(self, write_kind: str, channel: int) -> str:
        return (
            f"corr::{self.source_system}::{write_kind}::"
            f"ch{int(channel)}::{next(self._sequence)}"
        )

    @staticmethod
    def _is_wing_target(mixer_client: Any, mixer_type: Optional[str]) -> bool:
        if mixer_type and str(mixer_type).strip().lower() == "wing":
            return True
        if mixer_client is None:
            return False
        type_name = type(mixer_client).__name__.lower()
        module_name = type(mixer_client).__module__.lower()
        return "wing" in type_name or "wing" in module_name
