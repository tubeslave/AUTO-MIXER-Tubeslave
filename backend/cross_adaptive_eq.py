"""
Cross-adaptive mirror EQ utilities.

Based on IMP 7.3 (Intelligent Music Production, De Man, Reiss & Stables):

Full mirror equalization strategy:
- "Mirrored EQ involves boosting one track at a particular frequency
  and cutting the other track at the same frequency" [98, 127].
- "preference for boosting the masked track or for full mirror EQ,
  over just cutting the masker" [154].
- "subtractive EQ (cuts) are generally performed with a higher quality
  factor than additive EQ (boosts)" [12].
- Target frequency contour for the complete mix is similar to pink
  noise [274, 275].

References:
  [12] Pestana & Reiss (2014)
  [40] Perez Gonzalez - cross-adaptive multitrack EQ
  [52] Ma et al. - intelligent target EQ
  [61] Hafezi & Reiss - multitrack intelligent EQ
  [98, 127] - mirror equalization technique
  [154] Wakefield & Dewey - preference for boost + mirror over cut-only
  [274, 275] - pink noise target spectrum
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

try:
    from source_identity_downstream_gate import evaluate_source_identity_downstream_trust
except ImportError:  # pragma: no cover - package import compatibility
    from backend.source_identity_downstream_gate import evaluate_source_identity_downstream_trust


METHOD_ID = "cross_adaptive_eq"
READ_REQUIRED_SOURCE = "cached_or_physical_eq_band_read_required_before_promotion"


@dataclass
class MirrorEQAdjustment:
    channel_id: int
    frequency_hz: float
    gain_db: float
    q_factor: float
    paired_channel_id: Optional[int] = None
    band: str = ""
    role: str = ""
    overlap_strength: float = 0.0
    masked_channel_id: Optional[int] = None
    masker_channel_id: Optional[int] = None


@dataclass
class CrossAdaptiveEQProposalResult:
    method_id: str
    adjustments: Tuple[MirrorEQAdjustment, ...]
    suggested_actions: Tuple[Dict[str, Any], ...]
    warnings: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)


# Q factors per IMP [12]:
# Subtractive EQ (cuts) use higher Q (narrower, surgical).
# Additive EQ (boosts) use lower Q (broader, more musical).
Q_SUBTRACTIVE = 4.0
Q_ADDITIVE = 2.0


def _freshness_status(payload: Optional[Mapping[str, Any]]) -> str:
    if payload is None:
        return "not_provided"
    if not isinstance(payload, Mapping):
        return "unknown"
    status = str(payload.get("status") or "").strip().lower()
    if status:
        return status
    field_count = int(payload.get("field_count") or 0)
    stale_count = int(payload.get("stale_field_count") or 0)
    if field_count > 0 and stale_count == 0:
        return "fresh"
    if stale_count > 0:
        return "stale"
    return "unknown"


def _freshness_payload(payload: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if payload is None:
        return {"status": "not_provided"}
    if not isinstance(payload, Mapping):
        return {"status": "unknown", "raw_type": type(payload).__name__}
    normalized = dict(payload)
    normalized["status"] = _freshness_status(payload)
    return normalized


def _identity_payload_for_channel(
    channel_id: int,
    channel_source_identity: Optional[Mapping[Any, Mapping[str, Any]]],
) -> Mapping[str, Any]:
    if not isinstance(channel_source_identity, Mapping):
        return {}
    return (
        channel_source_identity.get(channel_id)
        or channel_source_identity.get(str(channel_id))
        or {}
    )


def _source_identity_gate_for_channel(
    channel_id: int,
    channel_source_identity: Optional[Mapping[Any, Mapping[str, Any]]],
):
    payload = _identity_payload_for_channel(channel_id, channel_source_identity)
    if not isinstance(payload, Mapping):
        payload = {}
    return evaluate_source_identity_downstream_trust(
        source_id=str(payload.get("source_id") or payload.get("id") or f"channel:{channel_id}"),
        source_type=str(payload.get("source_type") or payload.get("role") or "unknown"),
        trust_state=payload.get("trust_state", "not_provided"),
        trust_confidence=payload.get("trust_confidence", 0.0),
        replay_outcome_count=payload.get("replay_outcome_count", 0),
        current_state_cache_freshness=payload.get("current_state_cache_freshness")
        or payload.get("source_identity_calibration_freshness"),
    )


def _source_identity_gate_by_channel(
    channel_ids: Tuple[int, ...],
    channel_source_identity: Optional[Mapping[Any, Mapping[str, Any]]],
) -> Tuple[Dict[int, Any], Tuple[Dict[str, Any], ...], Tuple[str, ...]]:
    decisions = {}
    blocked = []
    blockers = set()
    for channel_id in channel_ids:
        decision = _source_identity_gate_for_channel(channel_id, channel_source_identity)
        decisions[channel_id] = decision
        if decision.usable_for_downstream:
            continue
        blocked.append({"channel_id": channel_id, **decision.to_dict()})
        blockers.update(decision.blockers)
    return decisions, tuple(blocked), tuple(sorted(blockers))


class CrossAdaptiveEQ:
    """
    Full mirror-EQ processor for multitrack band-energy dictionaries.

    Per IMP 7.3 [98, 127, 154]:
    - Cut the lower-priority (masker) channel in overlapping bands (Q=4.0).
    - Boost the higher-priority (masked) channel in overlapping bands (Q=2.0).
    - Boost amount is half the cut amount to err on the conservative side.
    """

    BAND_CENTERS = {
        "sub": 40.0,
        "bass": 120.0,
        "low_mid": 375.0,
        "mid": 1250.0,
        "high_mid": 3000.0,
        "high": 6000.0,
        "air": 12000.0,
    }

    def __init__(
        self,
        min_band_level_db=-80.0,
        overlap_tolerance_db=6.0,
        max_cut_db=-6.0,
        max_boost_db=3.0,
        q_subtractive=Q_SUBTRACTIVE,
        q_additive=Q_ADDITIVE,
    ):
        self.min_band_level_db = min_band_level_db
        self.overlap_tolerance_db = overlap_tolerance_db
        self.max_cut_db = max_cut_db
        self.max_boost_db = max_boost_db
        self.q_subtractive = q_subtractive
        self.q_additive = q_additive

    def calculate_corrections(
        self,
        channel_band_energy,
        channel_priorities,
    ):
        """
        Compute full mirror-EQ adjustments for overlapping bands.

        Per IMP 7.3 [98, 127, 154]:
        - For each overlapping band between two channels with different
          priorities, apply a CUT on the masker (lower-priority) and a
          BOOST on the masked (higher-priority) channel.
        - Cuts use Q=4.0 (subtractive, narrower) [12].
        - Boosts use Q=2.0 (additive, broader) [12].

        Lower numeric priority means higher importance.
        """
        channels = list(channel_band_energy.keys())
        out = []

        for i, ch_a in enumerate(channels):
            for ch_b in channels[i + 1:]:
                bands_a = channel_band_energy.get(ch_a, {})
                bands_b = channel_band_energy.get(ch_b, {})
                if not bands_a or not bands_b:
                    continue

                prio_a = channel_priorities.get(ch_a, 3)
                prio_b = channel_priorities.get(ch_b, 3)

                if prio_a == prio_b:
                    continue

                # Identify masked (higher priority) and masker (lower priority).
                if prio_a < prio_b:
                    masked_ch = ch_a
                    masker_ch = ch_b
                    masked_bands = bands_a
                    masker_bands = bands_b
                else:
                    masked_ch = ch_b
                    masker_ch = ch_a
                    masked_bands = bands_b
                    masker_bands = bands_a

                for band, center_hz in self.BAND_CENTERS.items():
                    masked_db = masked_bands.get(band, -100.0)
                    masker_db = masker_bands.get(band, -100.0)
                    if masked_db < self.min_band_level_db or masker_db < self.min_band_level_db:
                        continue

                    # Overlap when both bands are close in magnitude.
                    diff = abs(masked_db - masker_db)
                    if diff > self.overlap_tolerance_db:
                        continue

                    # Stronger overlap -> stronger correction (bounded).
                    overlap_strength = 1.0 - (diff / max(self.overlap_tolerance_db, 1e-6))

                    # CUT the masker channel [98, 127] with Q=4.0 [12].
                    cut_db = -min(abs(self.max_cut_db), 1.0 + 5.0 * overlap_strength)
                    out.append(
                        MirrorEQAdjustment(
                            channel_id=masker_ch,
                            frequency_hz=center_hz,
                            gain_db=cut_db,
                            q_factor=self.q_subtractive,
                            paired_channel_id=masked_ch,
                            band=band,
                            role="masker_cut",
                            overlap_strength=overlap_strength,
                            masked_channel_id=masked_ch,
                            masker_channel_id=masker_ch,
                        )
                    )

                    # BOOST the masked channel [154] with Q=2.0 [12].
                    # Boost is half the cut amount (conservative) [154].
                    boost_db = min(self.max_boost_db, abs(cut_db) * 0.5)
                    out.append(
                        MirrorEQAdjustment(
                            channel_id=masked_ch,
                            frequency_hz=center_hz,
                            gain_db=boost_db,
                            q_factor=self.q_additive,
                            paired_channel_id=masker_ch,
                            band=band,
                            role="masked_source_boost",
                            overlap_strength=overlap_strength,
                            masked_channel_id=masked_ch,
                            masker_channel_id=masker_ch,
                        )
                    )

        return out

    def analyze_unmasking_proposals(
        self,
        channel_band_energy: Mapping[int, Mapping[str, float]],
        channel_priorities: Mapping[int, int],
        *,
        current_state_cache_freshness: Optional[Mapping[str, Any]] = None,
        channel_source_identity: Optional[Mapping[Any, Mapping[str, Any]]] = None,
    ) -> CrossAdaptiveEQProposalResult:
        """Build proposal-only cross-channel unmasking actions.

        `calculate_corrections` remains the compatibility API for older offline
        and live code paths. This wrapper adds the proposal-only contract fields
        needed before any future operator review or supervised promotion.
        """

        current_state_freshness = _freshness_payload(current_state_cache_freshness)
        freshness_status = str(current_state_freshness.get("status") or "unknown")
        metadata: Dict[str, Any] = {
            "method_id": METHOD_ID,
            "proposal_only": True,
            "live_write_allowed": False,
            "can_import_to_queue": False,
            "current_state_cache_freshness_status": freshness_status,
            "current_state_cache_freshness": current_state_freshness,
            "old_value_source": READ_REQUIRED_SOURCE,
            "readback_required": True,
            "rollback_required": True,
            "blockers": (),
        }
        channel_ids = tuple(sorted(int(channel_id) for channel_id in channel_band_energy))
        source_identity_by_channel, blocked_source_identity, source_identity_blockers = (
            _source_identity_gate_by_channel(channel_ids, channel_source_identity)
        )
        metadata["source_identity_downstream_trust"] = [
            {"channel_id": channel_id, **decision.to_dict()}
            for channel_id, decision in source_identity_by_channel.items()
        ]
        metadata["source_identity_blocked_channels"] = list(blocked_source_identity)
        metadata["source_identity_blocked_channel_count"] = len(blocked_source_identity)

        if freshness_status not in {"fresh", "not_provided"}:
            metadata["adjustment_count"] = 0
            metadata["action_count"] = 0
            metadata["blockers"] = ("cross_adaptive_eq_context_cache_not_fresh",)
            return CrossAdaptiveEQProposalResult(
                method_id=METHOD_ID,
                adjustments=(),
                suggested_actions=(),
                warnings=(
                    "cross-channel masking context cache is not fresh; "
                    "collect fresh band-energy and role evidence before EQ proposals",
                ),
                metadata=metadata,
            )
        if source_identity_blockers:
            metadata["adjustment_count"] = 0
            metadata["action_count"] = 0
            metadata["blockers"] = source_identity_blockers
            return CrossAdaptiveEQProposalResult(
                method_id=METHOD_ID,
                adjustments=(),
                suggested_actions=(),
                warnings=(
                    "source identity replay calibration blocks cross-channel unmasking; "
                    "review source identity trust before linked EQ proposals",
                ),
                metadata=metadata,
            )

        adjustments = tuple(
            self.calculate_corrections(
                channel_band_energy=channel_band_energy,
                channel_priorities=channel_priorities,
            )
        )
        actions = tuple(
            self._proposal_action(item, freshness_status, source_identity_by_channel)
            for item in adjustments
        )
        metadata["adjustment_count"] = len(adjustments)
        metadata["action_count"] = len(actions)
        metadata["masking_pair_count"] = len(
            {
                (item.masked_channel_id, item.masker_channel_id, item.band)
                for item in adjustments
            }
        )

        return CrossAdaptiveEQProposalResult(
            method_id=METHOD_ID,
            adjustments=adjustments,
            suggested_actions=actions,
            metadata=metadata,
        )

    def _proposal_action(
        self,
        adjustment: MirrorEQAdjustment,
        freshness_status: str,
        source_identity_by_channel: Mapping[int, Any],
    ) -> Dict[str, Any]:
        gain_db = float(adjustment.gain_db)
        gain_limit = abs(self.max_cut_db) if gain_db < 0.0 else abs(self.max_boost_db)
        bounded = abs(gain_db) <= gain_limit + 1e-9
        masked_trust = source_identity_by_channel.get(adjustment.masked_channel_id)
        masker_trust = source_identity_by_channel.get(adjustment.masker_channel_id)
        source_identity_trust = {
            "masked_channel": masked_trust.to_dict() if masked_trust else None,
            "masker_channel": masker_trust.to_dict() if masker_trust else None,
        }
        return {
            "action_type": "cross_channel_unmasking_eq_candidate",
            "method_id": METHOD_ID,
            "value_type": "eq",
            "target_kind": "masking_eq_band",
            "channel_id": int(adjustment.channel_id),
            "paired_channel_id": adjustment.paired_channel_id,
            "frequency_hz": round(float(adjustment.frequency_hz), 3),
            "gain_db": round(gain_db, 3),
            "q_factor": round(float(adjustment.q_factor), 3),
            "role": adjustment.role,
            "band": adjustment.band,
            "masking_pair": {
                "masked_channel_id": adjustment.masked_channel_id,
                "masker_channel_id": adjustment.masker_channel_id,
            },
            "old_value": {
                "source": READ_REQUIRED_SOURCE,
                "required_before_apply": True,
            },
            "proposed_value": {
                "frequency_hz": round(float(adjustment.frequency_hz), 3),
                "gain_db": round(gain_db, 3),
                "q_factor": round(float(adjustment.q_factor), 3),
            },
            "rollback_value": {
                "source": READ_REQUIRED_SOURCE,
                "required_before_apply": True,
            },
            "safety_evidence": {
                "masking_pair": True,
                "linked_cut_boost_pair": adjustment.paired_channel_id is not None,
                "frequency_match": True,
                "bounded_gain_delta": bounded,
                "current_masking_context_cache_freshness": freshness_status,
                "source_identity_downstream_trust": source_identity_trust,
                "sum_bus_check": "required_before_apply",
                "eq_band_readback": "required_before_apply",
            },
            "required_approval": "operator_supervised",
            "readback_required": True,
            "rollback_required": True,
            "live_write_allowed": False,
            "can_import_to_queue": False,
        }
