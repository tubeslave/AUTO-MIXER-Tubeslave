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
import math
from typing import Any, Dict, List, Mapping, Optional, Tuple

try:
    from source_identity_downstream_gate import evaluate_source_identity_downstream_trust
except ImportError:  # pragma: no cover - package import compatibility
    from backend.source_identity_downstream_gate import evaluate_source_identity_downstream_trust


METHOD_ID = "cross_adaptive_eq"
READ_REQUIRED_SOURCE = "cached_or_physical_eq_band_read_required_before_promotion"
DEFAULT_CHANNEL_PRIORITY = 3
MIN_ARRANGEMENT_ROLE_CONFIDENCE = 0.55
ARRANGEMENT_ROLE_PRIORITY_HINTS = {
    "lead": 1,
    "featured": 1,
    "low_anchor": 1,
    "rhythmic_anchor": 2,
    "harmonic_support": 3,
    "backing": 4,
    "texture": 4,
    "effect": 5,
}
SOURCE_ROLE_PRIORITY_HINTS = {
    "lead_vocal": 1,
    "bass": 1,
    "kick": 2,
    "snare": 2,
    "drums": 2,
    "vocal": 2,
    "guitar": 3,
    "keys": 3,
    "synth": 4,
    "backing_vocal": 4,
    "fx_return": 5,
}
NON_PROCESSING_ARRANGEMENT_ROLES = {"reference", "utility", "talkback"}


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
        return {"status": "not_provided", "provided": False}
    if not isinstance(payload, Mapping):
        return {"status": "unknown", "raw_type": type(payload).__name__}
    normalized = dict(payload)
    normalized["status"] = _freshness_status(payload)
    return normalized


def _freshness_is_no_authority(payload: Mapping[str, Any]) -> bool:
    return (
        payload.get("read_only") is True
        and payload.get("live_write") is False
        and payload.get("queries_console") is False
    )


def _normalize_token(value: Any) -> str:
    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _safe_int(value: Any, default: int = DEFAULT_CHANNEL_PRIORITY) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    return numeric if math.isfinite(numeric) else default


def _clip01(value: Any) -> float:
    return max(0.0, min(1.0, _safe_float(value)))


def _first_present(payload: Mapping[str, Any], keys: Tuple[str, ...], default: Any = "") -> Any:
    for key in keys:
        value = payload.get(key)
        if value is not None and value != "":
            return value
    return default


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


def _arrangement_payload_for_channel(
    channel_id: int,
    channel_arrangement_context: Optional[Mapping[Any, Mapping[str, Any]]],
) -> Mapping[str, Any]:
    if not isinstance(channel_arrangement_context, Mapping):
        return {}
    return (
        channel_arrangement_context.get(channel_id)
        or channel_arrangement_context.get(str(channel_id))
        or {}
    )


def _priority_hint_for_context(arrangement_role: str, source_role: str) -> Optional[int]:
    role = _normalize_token(arrangement_role)
    source = _normalize_token(source_role)
    if role in ARRANGEMENT_ROLE_PRIORITY_HINTS:
        return ARRANGEMENT_ROLE_PRIORITY_HINTS[role]
    if source in SOURCE_ROLE_PRIORITY_HINTS:
        return SOURCE_ROLE_PRIORITY_HINTS[source]
    return None


def _arrangement_context_by_channel(
    channel_ids: Tuple[int, ...],
    channel_arrangement_context: Optional[Mapping[Any, Mapping[str, Any]]],
) -> Tuple[Dict[int, Dict[str, Any]], Tuple[Dict[str, Any], ...], Tuple[str, ...]]:
    contexts: Dict[int, Dict[str, Any]] = {}
    blocked = []
    blockers = set()
    for channel_id in channel_ids:
        if not isinstance(channel_arrangement_context, Mapping):
            context = {
                "channel_id": channel_id,
                "status": "missing",
                "usable_for_unmasking_priority": False,
                "blockers": ("cross_adaptive_eq_arrangement_context_missing_for_channel",),
            }
            contexts[channel_id] = context
            blocked.append(context)
            blockers.add("cross_adaptive_eq_arrangement_context_missing_for_channel")
            continue

        payload = _arrangement_payload_for_channel(channel_id, channel_arrangement_context)
        if not isinstance(payload, Mapping) or not payload:
            context = {
                "channel_id": channel_id,
                "status": "missing",
                "usable_for_unmasking_priority": False,
                "blockers": ("cross_adaptive_eq_arrangement_context_missing_for_channel",),
            }
            contexts[channel_id] = context
            blocked.append(context)
            blockers.add("cross_adaptive_eq_arrangement_context_missing_for_channel")
            continue

        arrangement_role = _normalize_token(
            _first_present(
                payload,
                ("arrangement_role", "proposed_arrangement_role", "role"),
                "unknown",
            )
        )
        source_role = _normalize_token(
            _first_present(payload, ("source_role", "source_type", "instrument_role"), "unknown")
        )
        confidence_default = 0.75 if arrangement_role and arrangement_role != "unknown" else 0.0
        role_confidence = _clip01(
            _first_present(
                payload,
                ("arrangement_role_confidence", "role_confidence", "confidence"),
                confidence_default,
            )
        )
        priority_hint = _priority_hint_for_context(arrangement_role, source_role)
        channel_blockers = set()
        if arrangement_role in NON_PROCESSING_ARRANGEMENT_ROLES:
            channel_blockers.add("cross_adaptive_eq_arrangement_role_not_processable")
        if not arrangement_role or arrangement_role == "unknown":
            channel_blockers.add("cross_adaptive_eq_arrangement_role_unknown")
        if priority_hint is None:
            channel_blockers.add("cross_adaptive_eq_arrangement_priority_unknown")
        if role_confidence < MIN_ARRANGEMENT_ROLE_CONFIDENCE:
            channel_blockers.add("cross_adaptive_eq_arrangement_role_confidence_low")

        context = {
            "channel_id": channel_id,
            "status": "trusted" if not channel_blockers else "blocked",
            "arrangement_role": arrangement_role,
            "source_role": source_role,
            "role_confidence": round(role_confidence, 3),
            "priority_hint": priority_hint,
            "usable_for_unmasking_priority": not channel_blockers,
            "blockers": tuple(sorted(channel_blockers)),
        }
        contexts[channel_id] = context
        if channel_blockers:
            blocked.append(context)
            blockers.update(channel_blockers)

    return contexts, tuple(blocked), tuple(sorted(blockers))


def _effective_priorities(
    channel_ids: Tuple[int, ...],
    channel_priorities: Mapping[int, int],
    arrangement_context_by_channel: Mapping[int, Dict[str, Any]],
) -> Tuple[Dict[int, int], Dict[int, str], Dict[int, Dict[str, Any]]]:
    explicit_priorities = {
        _safe_int(channel_id): _safe_int(priority)
        for channel_id, priority in (channel_priorities or {}).items()
    }
    priorities: Dict[int, int] = {}
    priority_sources: Dict[int, str] = {}
    enriched_context: Dict[int, Dict[str, Any]] = {
        channel_id: dict(context)
        for channel_id, context in arrangement_context_by_channel.items()
    }

    for channel_id in channel_ids:
        context = enriched_context.get(channel_id, {})
        if channel_id in explicit_priorities:
            priorities[channel_id] = explicit_priorities[channel_id]
            priority_sources[channel_id] = "explicit_channel_priority"
        elif context.get("usable_for_unmasking_priority"):
            priorities[channel_id] = _safe_int(context.get("priority_hint"))
            priority_sources[channel_id] = "arrangement_role_hint"
        else:
            priorities[channel_id] = DEFAULT_CHANNEL_PRIORITY
            priority_sources[channel_id] = "default_priority"

        if context:
            context["effective_priority"] = priorities[channel_id]
            context["priority_source"] = priority_sources[channel_id]

    return priorities, priority_sources, enriched_context


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
        channel_arrangement_context: Optional[Mapping[Any, Mapping[str, Any]]] = None,
        arrangement_context_cache_freshness: Optional[Mapping[str, Any]] = None,
    ) -> CrossAdaptiveEQProposalResult:
        """Build proposal-only cross-channel unmasking actions.

        `calculate_corrections` remains the compatibility API for older offline
        and live code paths. This wrapper adds the proposal-only contract fields
        needed before any future operator review or supervised promotion.
        """

        current_state_freshness = _freshness_payload(current_state_cache_freshness)
        freshness_status = str(current_state_freshness.get("status") or "unknown")
        current_state_no_authority = _freshness_is_no_authority(current_state_freshness)
        arrangement_freshness = _freshness_payload(arrangement_context_cache_freshness)
        arrangement_freshness_status = str(arrangement_freshness.get("status") or "unknown")
        arrangement_no_authority = _freshness_is_no_authority(arrangement_freshness)
        metadata: Dict[str, Any] = {
            "method_id": METHOD_ID,
            "proposal_only": True,
            "live_write_allowed": False,
            "can_import_to_queue": False,
            "current_state_cache_freshness_status": freshness_status,
            "current_state_cache_freshness": current_state_freshness,
            "current_masking_context_cache_no_live_authority": current_state_no_authority,
            "arrangement_context_cache_freshness_status": arrangement_freshness_status,
            "arrangement_context_cache_freshness": arrangement_freshness,
            "current_arrangement_context_cache_no_live_authority": arrangement_no_authority,
            "old_value_source": READ_REQUIRED_SOURCE,
            "readback_required": True,
            "rollback_required": True,
            "blockers": (),
        }
        channel_ids = tuple(sorted(int(channel_id) for channel_id in channel_band_energy))
        (
            arrangement_context,
            blocked_arrangement_context,
            arrangement_context_blockers,
        ) = _arrangement_context_by_channel(channel_ids, channel_arrangement_context)
        metadata["arrangement_priority_context"] = list(arrangement_context.values())
        metadata["arrangement_context_blocked_channels"] = list(blocked_arrangement_context)
        metadata["arrangement_context_blocked_channel_count"] = len(
            blocked_arrangement_context
        )
        source_identity_by_channel, blocked_source_identity, source_identity_blockers = (
            _source_identity_gate_by_channel(channel_ids, channel_source_identity)
        )
        metadata["source_identity_downstream_trust"] = [
            {"channel_id": channel_id, **decision.to_dict()}
            for channel_id, decision in source_identity_by_channel.items()
        ]
        metadata["source_identity_blocked_channels"] = list(blocked_source_identity)
        metadata["source_identity_blocked_channel_count"] = len(blocked_source_identity)

        if freshness_status != "fresh":
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
        if not current_state_no_authority:
            metadata["adjustment_count"] = 0
            metadata["action_count"] = 0
            metadata["blockers"] = ("cross_adaptive_eq_context_cache_live_authority_present",)
            return CrossAdaptiveEQProposalResult(
                method_id=METHOD_ID,
                adjustments=(),
                suggested_actions=(),
                warnings=(
                    "cross-channel masking context cache has live authority; collect "
                    "read-only masking evidence before linked EQ proposals",
                ),
                metadata=metadata,
            )
        if arrangement_freshness_status != "fresh":
            metadata["adjustment_count"] = 0
            metadata["action_count"] = 0
            metadata["blockers"] = ("cross_adaptive_eq_arrangement_context_cache_not_fresh",)
            return CrossAdaptiveEQProposalResult(
                method_id=METHOD_ID,
                adjustments=(),
                suggested_actions=(),
                warnings=(
                    "cross-channel arrangement context cache is not fresh; "
                    "collect fresh arrangement-role evidence before linked EQ proposals",
                ),
                metadata=metadata,
            )
        if not arrangement_no_authority:
            metadata["adjustment_count"] = 0
            metadata["action_count"] = 0
            metadata["blockers"] = (
                "cross_adaptive_eq_arrangement_context_cache_live_authority_present",
            )
            return CrossAdaptiveEQProposalResult(
                method_id=METHOD_ID,
                adjustments=(),
                suggested_actions=(),
                warnings=(
                    "cross-channel arrangement context cache has live authority; collect "
                    "read-only arrangement-role evidence before linked EQ proposals",
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
        if arrangement_context_blockers:
            metadata["adjustment_count"] = 0
            metadata["action_count"] = 0
            metadata["blockers"] = arrangement_context_blockers
            return CrossAdaptiveEQProposalResult(
                method_id=METHOD_ID,
                adjustments=(),
                suggested_actions=(),
                warnings=(
                    "arrangement-role evidence blocks cross-channel unmasking; "
                    "review channel roles before linked EQ proposals",
                ),
                metadata=metadata,
            )

        effective_priorities, priority_sources, arrangement_context = _effective_priorities(
            channel_ids,
            channel_priorities,
            arrangement_context,
        )
        metadata["effective_channel_priorities"] = {
            str(channel_id): priority
            for channel_id, priority in effective_priorities.items()
        }
        metadata["channel_priority_sources"] = {
            str(channel_id): source
            for channel_id, source in priority_sources.items()
        }
        metadata["arrangement_priority_context"] = list(arrangement_context.values())

        adjustments = tuple(
            self.calculate_corrections(
                channel_band_energy=channel_band_energy,
                channel_priorities=effective_priorities,
            )
        )
        actions = tuple(
            self._proposal_action(
                item,
                freshness_status,
                current_state_no_authority,
                source_identity_by_channel,
                arrangement_freshness_status,
                arrangement_no_authority,
                arrangement_context,
            )
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
        current_state_no_authority: bool,
        source_identity_by_channel: Mapping[int, Any],
        arrangement_freshness_status: str,
        arrangement_no_authority: bool,
        arrangement_context_by_channel: Mapping[int, Mapping[str, Any]],
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
        masked_arrangement = arrangement_context_by_channel.get(adjustment.masked_channel_id)
        masker_arrangement = arrangement_context_by_channel.get(adjustment.masker_channel_id)
        arrangement_priority_context = {
            "masked_channel": dict(masked_arrangement) if masked_arrangement else None,
            "masker_channel": dict(masker_arrangement) if masker_arrangement else None,
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
                "current_masking_context_cache_no_live_authority": current_state_no_authority,
                "current_arrangement_context_cache_freshness": (
                    arrangement_freshness_status
                ),
                "current_arrangement_context_cache_no_live_authority": (
                    arrangement_no_authority
                ),
                "source_identity_downstream_trust": source_identity_trust,
                "arrangement_priority_context": arrangement_priority_context,
                "sum_bus_check": "required_before_apply",
                "eq_band_readback": "required_before_apply",
            },
            "required_approval": "operator_supervised",
            "readback_required": True,
            "rollback_required": True,
            "live_write_allowed": False,
            "can_import_to_queue": False,
        }
