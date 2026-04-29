"""
Auto FX planning for live-style spatial mixing.

The planner keeps creative effects conservative and reversible: shared stereo
FX returns, post-fader sends, filtered returns, and vocal-priority ducking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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
        """Apply an FX plan to a Wing-like mixer client when supported."""
        calls: list[dict[str, Any]] = []
        if not mixer_client:
            return {"applied": False, "calls": calls, "reason": "no_mixer_client"}

        for bus in plan.buses:
            if hasattr(mixer_client, "set_fx_model"):
                mixer_client.set_fx_model(bus.fx_slot, bus.model)
                calls.append({"cmd": "set_fx_model", "slot": bus.fx_slot, "model": bus.model})
            if hasattr(mixer_client, "set_fx_on"):
                mixer_client.set_fx_on(bus.fx_slot, 1)
                calls.append({"cmd": "set_fx_on", "slot": bus.fx_slot, "on": 1})
            if hasattr(mixer_client, "set_fx_mix"):
                mixer_client.set_fx_mix(bus.fx_slot, 100.0)
                calls.append({"cmd": "set_fx_mix", "slot": bus.fx_slot, "mix": 100.0})
            if hasattr(mixer_client, "set_fx_return_level"):
                mixer_client.set_fx_return_level(bus.bus_id, bus.return_level_db)
                calls.append({"cmd": "set_fx_return_level", "bus": bus.bus_id, "level_db": bus.return_level_db})

            for parameter, value in self._wing_parameter_map(bus).items():
                if hasattr(mixer_client, "set_fx_parameter"):
                    mixer_client.set_fx_parameter(bus.fx_slot, parameter, value)
                    calls.append({"cmd": "set_fx_parameter", "slot": bus.fx_slot, "parameter": parameter, "value": value})

        for send in plan.sends:
            mode = "POST" if send.post_fader else "PRE"
            if hasattr(mixer_client, "set_channel_send"):
                mixer_client.set_channel_send(send.channel_id, send.bus_id, level=send.send_db, on=1, mode=mode)
                calls.append({
                    "cmd": "set_channel_send",
                    "channel": send.channel_id,
                    "bus": send.bus_id,
                    "level_db": send.send_db,
                    "mode": mode,
                })
            elif hasattr(mixer_client, "set_send_level"):
                mixer_client.set_send_level(send.channel_id, send.bus_id, send.send_db)
                calls.append({
                    "cmd": "set_send_level",
                    "channel": send.channel_id,
                    "bus": send.bus_id,
                    "level_db": send.send_db,
                })

        return {"applied": bool(calls), "calls": calls}

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
