"""
Auto FX planning for live-style spatial mixing.

The planner keeps creative effects conservative and reversible: shared stereo
FX returns, post-fader sends, filtered returns, and vocal-priority ducking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    from processing_proposal_adapters import build_operator_correction_payload
except ImportError:  # pragma: no cover - package import compatibility
    from backend.processing_proposal_adapters import build_operator_correction_payload


AUTO_FX_METHOD_ID = "auto_fx"
FX_READ_REQUIRED_SOURCE = "cached_or_physical_fx_read_required_before_promotion"


@dataclass(frozen=True)
class FXBusDecision:
    bus_id: int
    name: str
    fx_type: str
    fx_slot: str
    model: str
    return_level_db: float
    params: dict[str, float | str | bool] = field(default_factory=dict)
    hpf_hz: float = 200.0
    lpf_hz: float = 7000.0
    duck_source: str | None = None
    duck_depth_db: float = 0.0
    reason: str = ""


@dataclass(frozen=True)
class FXSendDecision:
    channel_id: int
    instrument: str
    bus_id: int
    send_db: float
    post_fader: bool = True
    reason: str = ""


@dataclass(frozen=True)
class FXPlan:
    buses: list[FXBusDecision]
    sends: list[FXSendDecision]
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "buses": [bus.__dict__ for bus in self.buses],
            "sends": [send.__dict__ for send in self.sends],
            "notes": list(self.notes),
        }


class AutoFXPlanner:
    """Rule-based FX planner for reverb, delay, chorus/doubler, and placement."""

    def __init__(self, tempo_bpm: float = 120.0, vocal_priority: bool = True):
        self.tempo_bpm = max(40.0, min(float(tempo_bpm), 240.0))
        self.vocal_priority = bool(vocal_priority)
        self.live_write_allowed = False
        self.write_policy = "proposal_only_quarantined"
        self.last_apply_result: dict[str, Any] = {}

    def create_plan(self, channels: dict[int, str]) -> FXPlan:
        buses = self._default_buses()
        sends: list[FXSendDecision] = []

        for channel_id, instrument in sorted(channels.items()):
            normalized = self._normalize_instrument(instrument)
            sends.extend(self._sends_for_instrument(channel_id, normalized))

        return FXPlan(
            buses=buses,
            sends=sends,
            notes=[
                "Shared stereo FX returns; no per-channel insert reverbs by default.",
                "FX sends are post-fader so agent balance moves preserve wet/dry ratio.",
                "Returns are filtered to keep low end clear and vocal presence readable.",
                "Vocal plate and tempo delay are ducked from lead vocal activity.",
            ],
        )

    def apply_to_mixer(self, mixer_client: Any, plan: FXPlan) -> dict[str, Any]:
        """Return blocked apply evidence without mutating a mixer client.

        The FX planner is useful for assisted mixing, but direct FX/send writes are
        quarantined until a supervised apply contract has physical readback and
        rollback proof. Keep this method for legacy callers, but fail closed.
        """
        blocked_calls = self._build_apply_calls(plan)
        reason = "fx_direct_apply_quarantined" if mixer_client else "no_mixer_client"
        send_levels = [send.send_db for send in plan.sends]
        return_levels = [bus.return_level_db for bus in plan.buses]
        loudest_send_db = max(send_levels) if send_levels else None
        loudest_return_db = max(return_levels) if return_levels else None
        all_sends_post_fader = all(send.post_fader for send in plan.sends)
        send_level_limit = {
            "status": "passed"
            if send_levels and all(-36.0 <= send_db <= -12.0 for send_db in send_levels)
            else "blocked",
            "min_send_db": min(send_levels) if send_levels else None,
            "max_send_db": loudest_send_db,
            "allowed_min_send_db": -36.0,
            "allowed_max_send_db": -12.0,
            "post_fader_sends_only": all_sends_post_fader,
            "source": "legacy_auto_fx_plan_blocked_calls",
        }
        return_level_safety = {
            "status": "passed"
            if loudest_return_db is not None and loudest_return_db <= -3.0
            else "blocked",
            "loudest_return_db": loudest_return_db,
            "max_return_level_db": -3.0,
            "source": "legacy_auto_fx_plan_blocked_calls",
            "physical_return_readback_required_before_live_promotion": True,
        }
        fx_return_headroom = {
            "status": "blocked_until_physical_return_readback",
            "planned_loudest_return_db": loudest_return_db,
            "min_return_headroom_db": 3.0,
            "return_level_safety_status": return_level_safety["status"],
            "source": FX_READ_REQUIRED_SOURCE,
            "required_before_live_promotion": True,
        }
        dry_wet_bounds = {
            "fx_mix_percent": 100.0,
            "post_fader_sends_only": all_sends_post_fader,
            "loudest_send_db": loudest_send_db,
            "loudest_return_db": loudest_return_db,
            "send_level_limit_status": send_level_limit["status"],
            "return_level_safety_status": return_level_safety["status"],
        }
        current_fx_parameter_schema_hints = {
            "status": "required_before_live_promotion",
            "source": FX_READ_REQUIRED_SOURCE,
            "planned_fx_slots": {
                bus.fx_slot: {"model": bus.model, "fx_type": bus.fx_type}
                for bus in plan.buses
            },
            "requires_physical_schema_readback": True,
            "live_write_authority": False,
        }
        rollback_fx_readback = {
            "status": "required_before_live_promotion",
            "source": FX_READ_REQUIRED_SOURCE,
            "send_level_readback_required": True,
            "return_level_readback_required": True,
            "fx_parameter_readback_required": True,
            "monitor_send_readback_required": True,
        }
        source_identity_downstream_trust = {
            "status": "legacy_multi_source_unverified",
            "trusted_for_downstream": False,
            "source_roles": sorted({send.instrument for send in plan.sends}),
            "operator_review_required": True,
        }
        blockers = [
            "unsupported_value_type:fx_send",
            "not_in_supervised_fader_gain_executor",
            "hardware_readback_and_rollback_not_proven",
            "fx_return_and_send_readback_not_proven",
            "monitor_safety_not_proven",
            "feedback_intelligibility_not_proven",
            reason,
        ]
        proposed_action = {
            "action_type": "auto_fx_apply_blocked",
            "bus_count": len(plan.buses),
            "send_count": len(plan.sends),
            "target_send_db": loudest_send_db,
            "return_level_db": loudest_return_db,
            "send_level_db": loudest_send_db,
            "hpf_hz": min((bus.hpf_hz for bus in plan.buses), default=None),
            "lpf_hz": max((bus.lpf_hz for bus in plan.buses), default=None),
            "send_level_limit": send_level_limit,
            "fx_return_headroom": fx_return_headroom,
            "return_level_safety": return_level_safety,
            "dry_wet_bounds": dry_wet_bounds,
            "rollback_fx_readback": rollback_fx_readback,
            "blocked_write_count": len(blocked_calls),
        }
        result = {
            "applied": False,
            "blocked": bool(blocked_calls),
            "calls": [],
            "blocked_calls": blocked_calls,
            "blocked_call_count": len(blocked_calls),
            "reason": reason,
            "write_policy": self.write_policy,
            "live_write_allowed": self.live_write_allowed,
            "live_write": False,
            "queue_mutation": False,
            "can_apply_live": False,
            "can_import_to_queue": False,
            "queue_import_allowed": False,
            "live_apply_allowed": False,
            "read_only_or_dry_run_only": True,
            "live_authority_granted": False,
            "readiness_closure_allowed": False,
            "queue_action": "not_queueable_until_promoted",
            "mixer_client_present": mixer_client is not None,
            "mutates_mixer": False,
        }
        proposal = {
            "id": f"{AUTO_FX_METHOD_ID}:plan:{reason}",
            "method_id": AUTO_FX_METHOD_ID,
            "source": AUTO_FX_METHOD_ID,
            "title": "AutoFX blocked apply evidence",
            "reason": (
                "AutoFX generated blocked FX/send write evidence. It is "
                "operator review only until FX send/return readback, monitor "
                "safety, feedback/intelligibility checks and rollback proof exist."
            ),
            "target": "fx_plan",
            "kind": "fx_send",
            "severity": "medium" if blocked_calls else "low",
            "confidence": 0.4,
            "queue_action": result["queue_action"],
            "live_write": False,
            "queue_mutation": False,
            "can_import_to_queue": False,
            "can_apply_live": False,
            "blockers": blockers,
            "proposed_action": proposed_action,
            "evidence": {
                "blocked_calls": blocked_calls,
                "blocked_call_count": len(blocked_calls),
                "write_policy": self.write_policy,
                "old_value_source": FX_READ_REQUIRED_SOURCE,
                "rollback_value_source": FX_READ_REQUIRED_SOURCE,
                "live_write": False,
                "queue_mutation": False,
                "can_apply_live": False,
                "can_import_to_queue": False,
                "queue_import_allowed": False,
                "live_apply_allowed": False,
                "read_only_or_dry_run_only": True,
                "live_authority_granted": False,
                "readiness_closure_allowed": False,
                "queue_action": "not_queueable_until_promoted",
                "mutates_mixer": False,
                "readback_required_before_promotion": True,
                "rollback_required_before_promotion": True,
                "return_level_db": loudest_return_db,
                "send_level_limit": send_level_limit,
                "fx_return_headroom": fx_return_headroom,
                "return_level_safety": return_level_safety,
                "dry_wet_bounds": dry_wet_bounds,
                "current_fx_send_return_cache_freshness": "not_provided",
                "current_fx_send_return_cache_no_live_authority": True,
                "current_fx_parameter_schema_hints": current_fx_parameter_schema_hints,
                "fx_parameter_schema_no_live_authority": True,
                "current_fx_parameter_schema_no_live_authority": True,
                "source_role": "multi_source_fx_plan",
                "source_identity_downstream_trust": source_identity_downstream_trust,
                "rollback_fx_readback": rollback_fx_readback,
                "monitor_safety": "not_measured_in_legacy_auto_fx_apply_path",
                "feedback_risk": "not_measured_in_legacy_auto_fx_apply_path",
                "intelligibility_risk": "not_measured_in_legacy_auto_fx_apply_path",
            },
        }
        result["operator_correction"] = build_operator_correction_payload(
            proposal=proposal,
            method_id=AUTO_FX_METHOD_ID,
            target="fx_plan",
            proposed_action=proposed_action,
            blockers=blockers,
        )
        self.last_apply_result = result
        return result

    def _build_apply_calls(self, plan: FXPlan) -> list[dict[str, Any]]:
        """Build the would-be FX write plan as blocked, reviewable evidence."""
        calls: list[dict[str, Any]] = []
        for bus in plan.buses:
            calls.append({"cmd": "set_fx_model", "slot": bus.fx_slot, "model": bus.model})
            calls.append({"cmd": "set_fx_on", "slot": bus.fx_slot, "on": 1})
            calls.append({"cmd": "set_fx_mix", "slot": bus.fx_slot, "mix": 100.0})
            calls.append({"cmd": "set_fx_return_level", "bus": bus.bus_id, "level_db": bus.return_level_db})

            for parameter, value in self._wing_parameter_map(bus).items():
                calls.append({
                    "cmd": "set_fx_parameter",
                    "slot": bus.fx_slot,
                    "parameter": parameter,
                    "value": value,
                })

        for send in plan.sends:
            mode = "POST" if send.post_fader else "PRE"
            calls.append({
                "cmd": "set_channel_send",
                "channel": send.channel_id,
                "bus": send.bus_id,
                "level_db": send.send_db,
                "on": 1,
                "mode": mode,
            })

        return calls

    def _default_buses(self) -> list[FXBusDecision]:
        quarter_ms = 60000.0 / self.tempo_bpm
        dotted_eighth_ms = quarter_ms * 0.75
        return [
            FXBusDecision(
                bus_id=13,
                name="Vocal Plate",
                fx_type="reverb",
                fx_slot="FX1",
                model="PLATE",
                return_level_db=-4.0,
                params={"decay_s": 1.55, "predelay_ms": 42.0, "density": 0.72, "brightness": 0.58},
                hpf_hz=220.0,
                lpf_hz=7800.0,
                duck_source="lead_vocal" if self.vocal_priority else None,
                duck_depth_db=1.8 if self.vocal_priority else 0.0,
                reason="Lead vocal needs depth while dry vocal remains forward.",
            ),
            FXBusDecision(
                bus_id=14,
                name="Drum Room",
                fx_type="reverb",
                fx_slot="FX2",
                model="ROOM",
                return_level_db=-4.0,
                params={"decay_s": 0.72, "predelay_ms": 10.0, "density": 0.86, "brightness": 0.42},
                hpf_hz=180.0,
                lpf_hz=5600.0,
                reason="Short dense drum space supports snare/toms without washing overheads.",
            ),
            FXBusDecision(
                bus_id=15,
                name="Tempo Delay",
                fx_type="delay",
                fx_slot="FX3",
                model="STEREO_DELAY",
                return_level_db=-0.5,
                params={
                    "left_delay_ms": dotted_eighth_ms,
                    "right_delay_ms": quarter_ms,
                    "feedback": 0.22,
                    "width": 0.85,
                },
                hpf_hz=260.0,
                lpf_hz=5200.0,
                duck_source="lead_vocal" if self.vocal_priority else None,
                duck_depth_db=4.0 if self.vocal_priority else 0.0,
                reason="Tempo delay is audible in vocal gaps, not over lyric consonants.",
            ),
            FXBusDecision(
                bus_id=16,
                name="Mod Doubler",
                fx_type="chorus",
                fx_slot="FX4",
                model="CHORUS",
                return_level_db=-11.0,
                params={"left_delay_ms": 11.0, "right_delay_ms": 17.0, "depth": 0.16, "rate_hz": 0.45},
                hpf_hz=220.0,
                lpf_hz=8500.0,
                reason="Subtle stereo modulation widens backing parts without moving the lead center.",
            ),
        ]

    def _sends_for_instrument(self, channel_id: int, instrument: str) -> list[FXSendDecision]:
        send_map: dict[str, list[tuple[int, float, str]]] = {
            "lead_vocal": [
                (13, -18.0, "lead vocal plate depth"),
                (15, -24.0, "ducked tempo delay for vocal phrases"),
            ],
            "backing_vocal": [
                (13, -15.0, "backing vocals sit behind lead"),
                (16, -23.0, "subtle width for backing vocals"),
            ],
            "snare": [
                (14, -17.0, "short drum room/plate support"),
            ],
            "rack_tom": [
                (14, -19.0, "tom depth follows drum room"),
            ],
            "floor_tom": [
                (14, -19.0, "floor tom depth follows drum room"),
            ],
            "overhead": [
                (14, -30.0, "tiny cohesion only; overheads already contain room"),
            ],
            "hi_hat": [
                (14, -32.0, "very low room only to avoid harsh wash"),
            ],
            "ride": [
                (14, -31.0, "very low room only to avoid harsh wash"),
            ],
            "electric_guitar": [
                (13, -25.0, "small shared plate to place guitars behind vocal"),
                (16, -26.0, "subtle width without hard center masking"),
            ],
            "accordion": [
                (13, -23.0, "shared plate for melodic depth"),
                (15, -29.0, "low tempo echo for phrase tails"),
            ],
            "playback": [
                (13, -29.0, "minimal glue; playback likely already processed"),
            ],
        }
        return [
            FXSendDecision(
                channel_id=channel_id,
                instrument=instrument,
                bus_id=bus_id,
                send_db=send_db,
                post_fader=True,
                reason=reason,
            )
            for bus_id, send_db, reason in send_map.get(instrument, [])
        ]

    def _wing_parameter_map(self, bus: FXBusDecision) -> dict[int, float]:
        """Generic FX parameter slots used by our Wing adapter wrapper."""
        if bus.fx_type == "reverb":
            return {
                1: float(bus.params.get("decay_s", 1.2)),
                2: float(bus.params.get("predelay_ms", 30.0)),
                3: float(bus.params.get("density", 0.7)),
                4: float(bus.params.get("brightness", 0.5)),
                5: float(bus.hpf_hz),
                6: float(bus.lpf_hz),
            }
        if bus.fx_type == "delay":
            return {
                1: float(bus.params.get("left_delay_ms", 375.0)),
                2: float(bus.params.get("right_delay_ms", 500.0)),
                3: float(bus.params.get("feedback", 0.2)),
                4: float(bus.params.get("width", 0.8)),
                5: float(bus.hpf_hz),
                6: float(bus.lpf_hz),
            }
        if bus.fx_type == "chorus":
            return {
                1: float(bus.params.get("rate_hz", 0.45)),
                2: float(bus.params.get("depth", 0.16)),
                3: float(bus.params.get("left_delay_ms", 11.0)),
                4: float(bus.params.get("right_delay_ms", 17.0)),
                5: float(bus.hpf_hz),
                6: float(bus.lpf_hz),
            }
        return {}

    @staticmethod
    def _normalize_instrument(instrument: str) -> str:
        mapping = {
            "leadVocal": "lead_vocal",
            "backingVocal": "backing_vocal",
            "bass": "bass_guitar",
            "electricGuitar": "electric_guitar",
            "tom": "rack_tom",
            "hihat": "hi_hat",
        }
        return mapping.get(instrument, instrument)
