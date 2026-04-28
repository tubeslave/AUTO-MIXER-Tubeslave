"""Manual-rule action planner for offline decision/correction candidates."""

from __future__ import annotations

from typing import Any

from .action_schema import (
    CandidateActionSet,
    CompressorAction,
    EQAction,
    FXSendAction,
    GainAction,
    NoChangeAction,
    ensure_no_change_candidate,
)


class DecisionActionPlanner:
    """Create conservative engineering candidates without applying them."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = dict(config or {})
        self.safety = dict(self.config.get("safety", {}) or {})

    def plan(
        self,
        channels: dict[str, str],
        technical_profile: dict[str, Any] | None = None,
    ) -> list[CandidateActionSet]:
        technical_profile = dict(technical_profile or {})
        candidates: list[CandidateActionSet] = [
            CandidateActionSet(
                candidate_id="candidate_000_no_change",
                actions=[NoChangeAction()],
                description="Required no-change baseline.",
                source="manual_rule",
                safety_limits_snapshot=self._limits(),
            )
        ]
        vocal_channels = self._roles(channels, "vocal")
        lead_vocal_channels = [cid for cid in vocal_channels if "back" not in cid.lower()]
        backing_vocal_channels = [cid for cid in vocal_channels if "back" in cid.lower()]
        bass_channels = self._roles(channels, "bass")
        kick_channels = self._roles(channels, "kick")
        guitar_channels = self._roles(channels, "guitar")
        keys_channels = [cid for cid, role in channels.items() if role in {"keys", "piano", "synth"}]
        drum_channels = [cid for cid, role in channels.items() if role in {"kick", "snare", "drums"}]

        if vocal_channels:
            candidates.append(self._gain_candidate("candidate_001_vocal_up_0_5db", vocal_channels, 0.5, "Small vocal lift."))
            candidates.append(self._gain_candidate("candidate_002_vocal_down_0_5db", vocal_channels, -0.5, "Small vocal trim."))

        mud = float(technical_profile.get("muddiness_proxy", technical_profile.get("mud", 0.25)) or 0.0)
        if mud >= 0.18:
            targets = [cid for cid, role in channels.items() if role not in {"kick", "bass"}]
            candidates.append(
                CandidateActionSet(
                    candidate_id="candidate_003_low_mid_cleanup",
                    actions=[
                        EQAction(cid, "low_mid", 250.0, -1.0, 0.9, "peaking")
                        for cid in targets
                    ],
                    description="Low-mid cleanup on masking channels.",
                    source="manual_rule",
                    safety_limits_snapshot=self._limits(),
                )
            )

        harsh = float(technical_profile.get("harshness_proxy", technical_profile.get("harshness", 0.0)) or 0.0)
        if harsh >= 0.12:
            candidates.append(
                CandidateActionSet(
                    candidate_id="candidate_004_harshness_reduction",
                    actions=[
                        EQAction(cid, "harshness", 3500.0, -1.0, 1.2, "peaking")
                        for cid, role in channels.items()
                        if role in {"vocal", "guitars", "snare", "drums"}
                    ],
                    description="Small harshness cut on likely bright sources.",
                    source="manual_rule",
                    safety_limits_snapshot=self._limits(),
                )
            )

        if bass_channels and kick_channels:
            candidates.append(
                CandidateActionSet(
                    candidate_id="candidate_005_bass_kick_balance",
                    actions=[
                        *[GainAction(cid, -0.5) for cid in bass_channels],
                        *[GainAction(cid, 0.5) for cid in kick_channels],
                    ],
                    description="Micro bass/kick balance candidate.",
                    source="manual_rule",
                    safety_limits_snapshot=self._limits(),
                )
            )

        balance_actions = [
            *[GainAction(cid, 0.8) for cid in lead_vocal_channels],
            *[GainAction(cid, -0.8) for cid in backing_vocal_channels],
            *[GainAction(cid, 0.4) for cid in kick_channels],
            *[GainAction(cid, -0.3) for cid in bass_channels],
            *[GainAction(cid, -0.5) for cid in guitar_channels],
            *[GainAction(cid, -0.6) for cid in keys_channels if "playback" in cid.lower()],
        ]
        if balance_actions:
            candidates.append(
                CandidateActionSet(
                    candidate_id="candidate_009_gain_balance_polish",
                    actions=balance_actions,
                    description="Role-aware gain balance candidate around the static rough mix.",
                    source="manual_rule",
                    safety_limits_snapshot=self._limits(),
                )
            )

        overcompressed = bool(technical_profile.get("overcompressed", False))
        if drum_channels and not overcompressed:
            candidates.append(
                CandidateActionSet(
                    candidate_id="candidate_006_light_bus_compression",
                    actions=[
                        CompressorAction("master", -18.0, 1.25, 20.0, 180.0, 0.0)
                    ],
                    description="Offline-only light bus compression probe.",
                    source="manual_rule",
                    safety_limits_snapshot=self._limits(),
                )
            )

        if vocal_channels and (mud >= 0.18 or technical_profile.get("clarity_drop")):
            candidates.append(
                CandidateActionSet(
                    candidate_id="candidate_007_fx_less_wet",
                    actions=[FXSendAction(cid, "reverb_delay", -0.8) for cid in vocal_channels],
                    description="Reduce vocal FX wash when clarity or mud is risky.",
                    source="manual_rule",
                    safety_limits_snapshot=self._limits(),
                )
            )

        if vocal_channels and bool(technical_profile.get("too_dry", False)):
            candidates.append(
                CandidateActionSet(
                    candidate_id="candidate_008_fx_more_wet",
                    actions=[FXSendAction(cid, "reverb_delay", 0.5) for cid in vocal_channels],
                    description="Slight ambience candidate for dry mixes.",
                    source="manual_rule",
                    safety_limits_snapshot=self._limits(),
                )
            )
        return ensure_no_change_candidate([candidate for candidate in candidates if candidate.actions])

    def _gain_candidate(self, candidate_id: str, channels: list[str], gain_db: float, description: str) -> CandidateActionSet:
        return CandidateActionSet(
            candidate_id=candidate_id,
            actions=[GainAction(channel_id, gain_db) for channel_id in channels],
            description=description,
            source="manual_rule",
            safety_limits_snapshot=self._limits(),
        )

    @staticmethod
    def _roles(channels: dict[str, str], role_name: str) -> list[str]:
        return [channel_id for channel_id, role in channels.items() if role_name in role]

    def _limits(self) -> dict[str, Any]:
        return {
            "max_gain_change_db_per_step": self.safety.get("max_gain_change_db_per_step", 1.0),
            "max_eq_change_db_per_step": self.safety.get("max_eq_change_db_per_step", 1.5),
            "max_true_peak_dbfs": self.safety.get("max_true_peak_dbfs", -1.0),
        }
