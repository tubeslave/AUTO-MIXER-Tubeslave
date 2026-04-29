"""
Incremental AutoFOH musical detectors and action recommenders.

These analyzers build on the stem-aware feature extraction and the existing
typed safety layer. They stay intentionally lightweight so the current engine
can adopt them without a large architecture rewrite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from autofoh_analysis import EPS, StemContributionMatrix, calculate_mix_indexes
from autofoh_models import (
    AnalysisFeatures,
    ConfidenceRisk,
    DetectedProblem,
    NAMED_FREQUENCY_BANDS,
    RuntimeState,
    TargetCorridor,
)
from autofoh_safety import ChannelEQMove, ChannelFaderMove, TypedCorrectionAction


BAND_INDEX_LOOKUP = {
    "SUB": 1,
    "BASS": 1,
    "BODY": 1,
    "MUD": 2,
    "LOW_MID": 2,
    "PRESENCE": 3,
    "HARSHNESS": 3,
    "SIBILANCE": 4,
    "AIR": 4,
}

BAND_TARGET_LOOKUP = {
    "SUB": (50.0, 1.0),
    "BASS": (90.0, 1.0),
    "BODY": (180.0, 1.1),
    "MUD": (350.0, 1.3),
    "LOW_MID": (700.0, 1.2),
    "PRESENCE": (2800.0, 1.5),
    "HARSHNESS": (4200.0, 1.8),
    "SIBILANCE": (7500.0, 2.0),
    "AIR": (12000.0, 1.5),
}

INDEX_ATTR_LOOKUP = {
    "SUB": "sub_index",
    "BASS": "bass_index",
    "BODY": "body_index",
    "MUD": "mud_index",
    "PRESENCE": "presence_index",
    "HARSHNESS": "harshness_index",
    "SIBILANCE": "sibilance_index",
    "AIR": "air_index",
}


def _db_to_linear(db_value: float) -> float:
    return 10.0 ** (float(db_value) / 10.0)


def _linear_to_db(power_value: float) -> float:
    return 10.0 * __import__("math").log10(max(power_value, EPS))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _band_names(features: Iterable[AnalysisFeatures], attribute: str) -> List[str]:
    keys = set()
    for feature in features:
        keys.update(getattr(feature, attribute).keys())
    return sorted(keys)


def aggregate_analysis_features(
    features: Iterable[AnalysisFeatures],
    target_corridor: Optional[TargetCorridor] = None,
) -> AnalysisFeatures:
    feature_list = [feature for feature in features if feature is not None]
    if not feature_list:
        return AnalysisFeatures(confidence=0.0)

    named_levels: Dict[str, float] = {}
    for band in NAMED_FREQUENCY_BANDS:
        total_power = sum(
            _db_to_linear(feature.named_band_levels_db.get(band.name, -100.0))
            for feature in feature_list
        )
        named_levels[band.name] = _linear_to_db(total_power)

    slope_levels: Dict[str, float] = {}
    for band in NAMED_FREQUENCY_BANDS:
        total_power = sum(
            _db_to_linear(feature.slope_compensated_band_levels_db.get(band.name, -100.0))
            for feature in feature_list
        )
        slope_levels[band.name] = _linear_to_db(total_power)

    octave_levels: Dict[str, float] = {}
    for band_name in _band_names(feature_list, "octave_band_levels_db"):
        total_power = sum(
            _db_to_linear(feature.octave_band_levels_db.get(band_name, -100.0))
            for feature in feature_list
        )
        octave_levels[band_name] = _linear_to_db(total_power)

    rms_power = sum(_db_to_linear(feature.rms_db) for feature in feature_list)
    peak_db = max(feature.peak_db for feature in feature_list)
    rms_db = _linear_to_db(rms_power)

    return AnalysisFeatures(
        rms_db=rms_db,
        peak_db=peak_db,
        crest_factor_db=peak_db - rms_db,
        named_band_levels_db=named_levels,
        octave_band_levels_db=octave_levels,
        slope_compensated_band_levels_db=slope_levels,
        mix_indexes=calculate_mix_indexes(
            slope_levels,
            target_corridor=target_corridor,
        ),
        confidence=sum(feature.confidence for feature in feature_list) / len(feature_list),
    )


def aggregate_stem_features(
    channel_features: Mapping[int, AnalysisFeatures],
    channel_stems: Mapping[int, Sequence[str]],
    target_corridor: Optional[TargetCorridor] = None,
) -> Dict[str, AnalysisFeatures]:
    grouped: Dict[str, List[AnalysisFeatures]] = {}
    for channel_id, feature in channel_features.items():
        stems = channel_stems.get(channel_id) or ["UNKNOWN"]
        for stem in stems:
            grouped.setdefault(str(stem), []).append(feature)

    stem_features = {
        stem_name: aggregate_analysis_features(
            features,
            target_corridor=target_corridor,
        )
        for stem_name, features in grouped.items()
    }
    if channel_features:
        stem_features["MASTER"] = aggregate_analysis_features(
            channel_features.values(),
            target_corridor=target_corridor,
        )
    return stem_features


def dominant_channel_for_stem_band(
    channel_features: Mapping[int, AnalysisFeatures],
    channel_stems: Mapping[int, Sequence[str]],
    stem_name: str,
    band_name: str,
) -> Optional[int]:
    best_channel = None
    best_level = float("-inf")
    for channel_id, feature in channel_features.items():
        if stem_name not in (channel_stems.get(channel_id) or []):
            continue
        level = float(feature.named_band_levels_db.get(band_name, -100.0))
        if level > best_level:
            best_level = level
            best_channel = channel_id
    return best_channel


@dataclass
class DetectorRecommendation:
    problem: Optional[DetectedProblem] = None
    candidate_actions: List[TypedCorrectionAction] = field(default_factory=list)


@dataclass
class LeadMaskingResult:
    lead_channel_id: Optional[int] = None
    lead_intelligibility_score: float = 0.0
    lead_body_score: float = 0.0
    lead_sibilance_score: float = 0.0
    masking_score: float = 0.0
    culprit_stems_by_band: Dict[str, str] = field(default_factory=dict)
    candidate_actions: List[TypedCorrectionAction] = field(default_factory=list)
    problem: Optional[DetectedProblem] = None


@dataclass
class LowEndBalanceResult:
    dominant_issue: Optional[str] = None
    sub_index: float = 0.0
    bass_index: float = 0.0
    body_index: float = 0.0
    culprit_stem: Optional[str] = None
    culprit_channel_id: Optional[int] = None
    candidate_actions: List[TypedCorrectionAction] = field(default_factory=list)
    problem: Optional[DetectedProblem] = None


@dataclass
class _PersistenceState:
    count: int = 0
    active: bool = False


class LeadMaskingAnalyzer:
    def __init__(
        self,
        masking_threshold_db: float = 3.0,
        culprit_share_threshold: float = 0.35,
        persistence_required_cycles: int = 3,
        hysteresis_db: float = 1.0,
        lead_boost_db: float = 0.5,
    ):
        self.masking_threshold_db = masking_threshold_db
        self.culprit_share_threshold = culprit_share_threshold
        self.persistence_required_cycles = max(1, persistence_required_cycles)
        self.hysteresis_db = hysteresis_db
        self.lead_boost_db = max(0.1, lead_boost_db)
        self._states: Dict[int, _PersistenceState] = {}

    def analyze(
        self,
        channel_features: Mapping[int, AnalysisFeatures],
        channel_stems: Mapping[int, Sequence[str]],
        stem_features: Mapping[str, AnalysisFeatures],
        contribution_matrix: StemContributionMatrix,
        lead_channel_ids: Sequence[int],
        current_faders_db: Optional[Mapping[int, float]] = None,
        lead_priorities: Optional[Mapping[int, float]] = None,
        runtime_state: RuntimeState = RuntimeState.PRE_SHOW_CHECK,
    ) -> LeadMaskingResult:
        del runtime_state
        if not lead_channel_ids:
            return LeadMaskingResult()

        lead_priorities = lead_priorities or {}
        current_faders_db = current_faders_db or {}
        lead_channel_id = max(
            lead_channel_ids,
            key=lambda channel_id: (
                float(lead_priorities.get(channel_id, 0.0)),
                float(channel_features.get(channel_id, AnalysisFeatures()).rms_db),
            ),
        )
        lead_features = channel_features.get(lead_channel_id)
        if lead_features is None:
            return LeadMaskingResult()

        accompaniment_presence_power = 0.0
        for stem_name, feature in stem_features.items():
            if stem_name in {"MASTER", "LEAD"}:
                continue
            accompaniment_presence_power += _db_to_linear(
                feature.named_band_levels_db.get("PRESENCE", -100.0)
            )
        accompaniment_presence_db = _linear_to_db(accompaniment_presence_power)
        lead_presence_db = float(lead_features.named_band_levels_db.get("PRESENCE", -100.0))
        masking_score = max(0.0, accompaniment_presence_db - lead_presence_db)

        lead_intelligibility = _clamp01(0.5 + (lead_presence_db - accompaniment_presence_db) / 12.0)
        lead_body = _clamp01(1.0 - abs(lead_features.mix_indexes.body_index) / 6.0)
        lead_sibilance = _clamp01(1.0 - max(0.0, lead_features.mix_indexes.sibilance_index) / 6.0)

        culprit_stem, culprit_share = self._dominant_non_lead_stem(
            contribution_matrix,
            band_name="PRESENCE",
        )
        result = LeadMaskingResult(
            lead_channel_id=lead_channel_id,
            lead_intelligibility_score=lead_intelligibility,
            lead_body_score=lead_body,
            lead_sibilance_score=lead_sibilance,
            masking_score=masking_score,
            culprit_stems_by_band={"PRESENCE": culprit_stem} if culprit_stem else {},
        )

        state = self._states.setdefault(lead_channel_id, _PersistenceState())
        if masking_score >= self.masking_threshold_db:
            state.count += 1
        elif masking_score <= max(0.0, self.masking_threshold_db - self.hysteresis_db):
            state.count = 0
            state.active = False
            return result
        else:
            return result

        if state.count < self.persistence_required_cycles or state.active:
            return result

        state.active = True
        culprit_channel = None
        actions: List[TypedCorrectionAction] = []
        if culprit_stem:
            culprit_channel = dominant_channel_for_stem_band(
                channel_features,
                channel_stems,
                culprit_stem,
                "PRESENCE",
            )
        if culprit_channel is not None and culprit_share >= self.culprit_share_threshold:
            actions.append(
                ChannelEQMove(
                    channel_id=culprit_channel,
                    band=BAND_INDEX_LOOKUP["PRESENCE"],
                    freq_hz=BAND_TARGET_LOOKUP["PRESENCE"][0],
                    gain_db=-1.0,
                    q=BAND_TARGET_LOOKUP["PRESENCE"][1],
                    reason=f"Reduce {culprit_stem} masking under lead",
                )
            )
        else:
            current_fader = float(current_faders_db.get(lead_channel_id, -6.0))
            actions.append(
                ChannelFaderMove(
                    channel_id=lead_channel_id,
                    target_db=min(0.0, current_fader + self.lead_boost_db),
                    delta_db=self.lead_boost_db,
                    is_lead=True,
                    reason="Small lead trim for intelligibility",
                )
            )

        confidence = _clamp01(0.45 + (masking_score - self.masking_threshold_db) / 8.0)
        culprit_confidence = _clamp01(culprit_share if culprit_stem else 0.0)
        result.problem = DetectedProblem(
            problem_type="lead_masking",
            description="Lead presence is being masked by accompaniment",
            channel_id=lead_channel_id,
            stem="LEAD",
            band_name="PRESENCE",
            persistence_sec=float(state.count),
            features=lead_features,
            confidence_risk=ConfidenceRisk(
                problem_confidence=confidence,
                culprit_confidence=culprit_confidence,
                action_confidence=max(confidence, culprit_confidence),
                risk_score=0.25,
            ),
            expected_effect="Improve lead intelligibility without over-boosting lead",
        )
        result.candidate_actions = actions
        return result

    @staticmethod
    def _dominant_non_lead_stem(
        contribution_matrix: StemContributionMatrix,
        band_name: str,
    ) -> Tuple[Optional[str], float]:
        row = contribution_matrix.band_contributions.get(band_name, {})
        filtered = {
            stem_name: share
            for stem_name, share in row.items()
            if stem_name not in {"MASTER", "LEAD", "UNKNOWN"}
        }
        if not filtered:
            return None, 0.0
        stem_name, share = max(filtered.items(), key=lambda item: item[1])
        return stem_name, float(share)


class PersistentBandDetector:
    def __init__(
        self,
        *,
        problem_type: str,
        description: str,
        band_name: str,
        threshold_db: float,
        persistence_required_cycles: int = 3,
        hysteresis_db: float = 0.75,
        culprit_share_threshold: float = 0.35,
        action_gain_db: float = -1.0,
    ):
        self.problem_type = problem_type
        self.description = description
        self.band_name = band_name
        self.threshold_db = threshold_db
        self.persistence_required_cycles = max(1, persistence_required_cycles)
        self.hysteresis_db = hysteresis_db
        self.culprit_share_threshold = culprit_share_threshold
        self.action_gain_db = action_gain_db
        self._states: Dict[str, _PersistenceState] = {}

    def observe(
        self,
        *,
        master_features: AnalysisFeatures,
        contribution_matrix: StemContributionMatrix,
        channel_features: Mapping[int, AnalysisFeatures],
        channel_stems: Mapping[int, Sequence[str]],
        key: str = "MASTER",
    ) -> DetectorRecommendation:
        state = self._states.setdefault(key, _PersistenceState())
        excess_db = self._band_excess(master_features)

        if excess_db >= self.threshold_db:
            state.count += 1
        elif excess_db <= max(0.0, self.threshold_db - self.hysteresis_db):
            state.count = 0
            state.active = False
            return DetectorRecommendation()
        else:
            return DetectorRecommendation()

        if state.count < self.persistence_required_cycles or state.active:
            return DetectorRecommendation()

        state.active = True
        culprit_stem = contribution_matrix.dominant_stem(self.band_name)
        culprit_share = contribution_matrix.contribution(self.band_name, culprit_stem or "")
        culprit_channel = None
        actions: List[TypedCorrectionAction] = []
        if culprit_stem:
            culprit_channel = dominant_channel_for_stem_band(
                channel_features,
                channel_stems,
                culprit_stem,
                self.band_name,
            )
        if culprit_channel is not None and culprit_share >= self.culprit_share_threshold:
            target_freq_hz, target_q = BAND_TARGET_LOOKUP[self.band_name]
            actions.append(
                ChannelEQMove(
                    channel_id=culprit_channel,
                    band=BAND_INDEX_LOOKUP[self.band_name],
                    freq_hz=target_freq_hz,
                    gain_db=self.action_gain_db,
                    q=target_q,
                    reason=f"{self.problem_type} cleanup on {culprit_stem}",
                )
            )

        confidence = _clamp01(0.45 + (excess_db - self.threshold_db) / 8.0)
        recommendation = DetectorRecommendation(
            problem=DetectedProblem(
                problem_type=self.problem_type,
                description=self.description,
                stem=culprit_stem,
                band_name=self.band_name,
                persistence_sec=float(state.count),
                features=master_features,
                confidence_risk=ConfidenceRisk(
                    problem_confidence=confidence,
                    culprit_confidence=_clamp01(culprit_share),
                    action_confidence=max(confidence * 0.9, _clamp01(culprit_share)),
                    risk_score=0.35,
                ),
                expected_effect=f"Reduce sustained {self.band_name.lower()} excess from the main culprit stem",
            ),
            candidate_actions=actions,
        )
        if culprit_channel is not None:
            recommendation.problem.channel_id = culprit_channel
        return recommendation

    def _band_excess(self, features: AnalysisFeatures) -> float:
        attr_name = INDEX_ATTR_LOOKUP[self.band_name]
        return max(0.0, float(getattr(features.mix_indexes, attr_name)))


class MudExcessDetector(PersistentBandDetector):
    def __init__(self, threshold_db: float = 2.5, persistence_required_cycles: int = 3, hysteresis_db: float = 0.75):
        super().__init__(
            problem_type="mud_excess",
            description="Sustained 250-500 Hz excess is clouding the mix",
            band_name="MUD",
            threshold_db=threshold_db,
            persistence_required_cycles=persistence_required_cycles,
            hysteresis_db=hysteresis_db,
        )


class HarshnessExcessDetector(PersistentBandDetector):
    def __init__(self, threshold_db: float = 2.5, persistence_required_cycles: int = 3, hysteresis_db: float = 0.75):
        super().__init__(
            problem_type="harshness_excess",
            description="Sustained 3-6 kHz excess is making the mix harsh",
            band_name="HARSHNESS",
            threshold_db=threshold_db,
            persistence_required_cycles=persistence_required_cycles,
            hysteresis_db=hysteresis_db,
        )


class SibilanceExcessDetector(PersistentBandDetector):
    def __init__(self, threshold_db: float = 2.5, persistence_required_cycles: int = 3, hysteresis_db: float = 0.75):
        super().__init__(
            problem_type="sibilance_excess",
            description="Sustained 6-10 kHz excess is causing sibilance or cymbal bite",
            band_name="SIBILANCE",
            threshold_db=threshold_db,
            persistence_required_cycles=persistence_required_cycles,
            hysteresis_db=hysteresis_db,
        )


class LowEndAnalyzer:
    def __init__(
        self,
        sub_threshold_db: float = 4.0,
        bass_threshold_db: float = 3.0,
        body_threshold_db: float = 2.5,
        culprit_share_threshold: float = 0.35,
        persistence_required_cycles: int = 3,
        hysteresis_db: float = 0.75,
    ):
        self.sub_threshold_db = sub_threshold_db
        self.bass_threshold_db = bass_threshold_db
        self.body_threshold_db = body_threshold_db
        self.culprit_share_threshold = culprit_share_threshold
        self.persistence_required_cycles = max(1, persistence_required_cycles)
        self.hysteresis_db = hysteresis_db
        self._active_issue: Optional[str] = None
        self._state = _PersistenceState()

    def analyze(
        self,
        *,
        master_features: AnalysisFeatures,
        contribution_matrix: StemContributionMatrix,
        channel_features: Mapping[int, AnalysisFeatures],
        channel_stems: Mapping[int, Sequence[str]],
    ) -> LowEndBalanceResult:
        sub_index = float(master_features.mix_indexes.sub_index)
        bass_index = float(master_features.mix_indexes.bass_index)
        body_index = float(master_features.mix_indexes.body_index)

        candidates = []
        if sub_index >= self.sub_threshold_db:
            candidates.append(("sub_excess", "SUB", sub_index - self.sub_threshold_db))
        if bass_index >= self.bass_threshold_db:
            candidates.append(("bass_excess", "BASS", bass_index - self.bass_threshold_db))
        if body_index >= self.body_threshold_db:
            candidates.append(("body_excess", "BODY", body_index - self.body_threshold_db))

        if candidates:
            issue_name, band_name, _ = max(candidates, key=lambda item: item[2])
        else:
            issue_name = None
            band_name = None

        result = LowEndBalanceResult(
            dominant_issue=issue_name,
            sub_index=sub_index,
            bass_index=bass_index,
            body_index=body_index,
        )

        if issue_name is None:
            self._active_issue = None
            self._state = _PersistenceState()
            return result

        threshold = {
            "sub_excess": self.sub_threshold_db,
            "bass_excess": self.bass_threshold_db,
            "body_excess": self.body_threshold_db,
        }[issue_name]
        issue_value = {
            "sub_excess": sub_index,
            "bass_excess": bass_index,
            "body_excess": body_index,
        }[issue_name]

        if self._active_issue != issue_name:
            self._active_issue = issue_name
            self._state = _PersistenceState()

        if issue_value >= threshold:
            self._state.count += 1
        elif issue_value <= max(0.0, threshold - self.hysteresis_db):
            self._active_issue = None
            self._state = _PersistenceState()
            return result
        else:
            return result

        if self._state.count < self.persistence_required_cycles or self._state.active:
            return result

        self._state.active = True
        culprit_stem = contribution_matrix.dominant_stem(band_name)
        culprit_share = contribution_matrix.contribution(band_name, culprit_stem or "")
        culprit_channel = None
        actions: List[TypedCorrectionAction] = []
        if culprit_stem:
            culprit_channel = dominant_channel_for_stem_band(
                channel_features,
                channel_stems,
                culprit_stem,
                band_name,
            )
        if culprit_channel is not None and culprit_share >= self.culprit_share_threshold:
            target_freq_hz, target_q = BAND_TARGET_LOOKUP[band_name]
            actions.append(
                ChannelEQMove(
                    channel_id=culprit_channel,
                    band=BAND_INDEX_LOOKUP[band_name],
                    freq_hz=target_freq_hz,
                    gain_db=-1.0,
                    q=target_q,
                    reason=f"Low-end control for {issue_name} on {culprit_stem}",
                )
            )

        result.culprit_stem = culprit_stem
        result.culprit_channel_id = culprit_channel
        result.candidate_actions = actions
        result.problem = DetectedProblem(
            problem_type=issue_name,
            description=f"Low-end imbalance detected: {issue_name}",
            channel_id=culprit_channel,
            stem=culprit_stem,
            band_name=band_name,
            persistence_sec=float(self._state.count),
            features=master_features,
            confidence_risk=ConfidenceRisk(
                problem_confidence=_clamp01(0.45 + (issue_value - threshold) / 8.0),
                culprit_confidence=_clamp01(culprit_share),
                action_confidence=max(0.5, _clamp01(culprit_share)),
                risk_score=0.3,
            ),
            expected_effect="Reduce excessive low-end build-up without chasing the master bus",
        )
        return result
