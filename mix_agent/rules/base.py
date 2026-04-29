"""Explainable, conservative rule engine for mixing recommendations."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from mix_agent.config import load_genre_profile, load_metric_thresholds
from mix_agent.models import MixAction, MixAnalysis, QualityDashboard, RuleIssue


SEVERITY_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _score_from_penalty(penalty: float) -> float:
    return round(_clamp01(1.0 - penalty), 3)


def _profile_range(profile: Dict[str, Any], key: str, default: tuple[float, float]) -> tuple[float, float]:
    values = profile.get(key, default)
    if not isinstance(values, (list, tuple)) or len(values) != 2:
        return default
    return float(values[0]), float(values[1])


class MixRuleEngine:
    """Evaluate analysis snapshots against engineering rules.

    The engine does not optimize one metric.  It emits ranked issues with
    evidence, risk and conservative actions that can be reviewed or passed to
    the backend safety bridge.
    """

    def __init__(
        self,
        genre_profile: Optional[Dict[str, Any]] = None,
        thresholds: Optional[Dict[str, Any]] = None,
    ):
        self.profile = genre_profile or load_genre_profile("neutral")
        self.thresholds = thresholds or load_metric_thresholds()

    @classmethod
    def for_genre(cls, genre: str | None) -> "MixRuleEngine":
        return cls(genre_profile=load_genre_profile(genre))

    def evaluate(self, analysis: MixAnalysis) -> List[RuleIssue]:
        issues: List[RuleIssue] = []
        issues.extend(self._technical_rules(analysis))
        issues.extend(self._gain_staging_rules(analysis))
        issues.extend(self._balance_rules(analysis))
        issues.extend(self._tonal_rules(analysis))
        issues.extend(self._masking_rules(analysis))
        issues.extend(self._dynamics_rules(analysis))
        issues.extend(self._stereo_rules(analysis))
        issues.extend(self._reference_rules(analysis))
        issues.sort(key=lambda item: (SEVERITY_RANK.get(item.severity, 9), -item.confidence))
        return issues

    def _issue(
        self,
        *,
        issue_id: str,
        group: str,
        name: str,
        severity: str,
        explanation: str,
        evidence: Iterable[str],
        suggested_action: str,
        expected_improvement: str,
        affected_tracks: Iterable[str],
        confidence: float,
        risk: str,
        actions: Optional[List[MixAction]] = None,
        constraints: Optional[List[str]] = None,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> RuleIssue:
        return RuleIssue(
            id=issue_id,
            group=group,
            name=name,
            severity=severity,
            explanation=explanation,
            evidence=list(evidence),
            suggested_action=suggested_action,
            expected_improvement=expected_improvement,
            affected_tracks=list(affected_tracks),
            metrics_before_after=metrics or {},
            confidence=round(_clamp01(confidence), 3),
            do_not_apply_constraints=constraints
            or [
                "Do not apply automatically if musical intent or subjective clarity worsens.",
                "Loudness-match before judging improvement.",
            ],
            risk=risk,
            actions=actions or [],
        )

    def _action(
        self,
        action_id: str,
        action_type: str,
        target: str,
        parameters: Dict[str, Any],
        reason: str,
        expected: str,
        risk: str,
        confidence: float,
        mode: str = "recommend",
    ) -> MixAction:
        safe_ranges = self.thresholds.get("actions", {})
        return MixAction(
            id=action_id,
            action_type=action_type,
            target=target,
            parameters=parameters,
            safe_range=safe_ranges,
            mode=mode,
            reason=reason,
            expected_improvement=expected,
            risk=risk,
            confidence=confidence,
            reversible=True,
        )

    def _technical_rules(self, analysis: MixAnalysis) -> List[RuleIssue]:
        level = analysis.mix.level
        tech = self.thresholds.get("technical", {})
        issues: List[RuleIssue] = []
        true_peak = float(level.get("true_peak_dbtp", -100.0))
        clip_count = int(level.get("clip_count", 0))
        if true_peak > float(tech.get("master_true_peak_warning_dbtp", -1.0)) or clip_count:
            reduction = min(-0.5, float(tech.get("master_true_peak_warning_dbtp", -1.0)) - true_peak)
            action = self._action(
                "technical.reduce_mix_gain",
                "gain_adjustment",
                "mix",
                {"gain_db": max(-3.0, reduction)},
                "Prevent clipping/true-peak risk before any aesthetic decision.",
                "Restores digital headroom and lowers inter-sample peak risk.",
                "Too much reduction can bias A/B comparisons unless loudness-matched.",
                0.95,
                mode="safe_apply",
            )
            issues.append(
                self._issue(
                    issue_id="technical.true_peak_or_clipping",
                    group="I. artifacts_and_technical_defects",
                    name="True peak or clipping risk",
                    severity="critical",
                    explanation="The mix violates technical headroom before musical decisions are safe.",
                    evidence=[
                        f"True peak is {true_peak:.2f} dBTP.",
                        f"Clip count estimate is {clip_count}.",
                    ],
                    suggested_action="Lower the mix or offending bus conservatively, then re-analyze.",
                    expected_improvement="More headroom and lower artifact risk.",
                    affected_tracks=["mix"],
                    confidence=0.95,
                    risk=action.risk,
                    actions=[action],
                    metrics={"before": {"true_peak_dbtp": true_peak, "clip_count": clip_count}},
                )
            )
        dc = abs(float(level.get("dc_offset", 0.0)))
        if dc > float(tech.get("dc_offset_warning", 0.01)):
            issues.append(
                self._issue(
                    issue_id="technical.dc_offset",
                    group="I. artifacts_and_technical_defects",
                    name="DC offset warning",
                    severity="medium",
                    explanation="A measurable DC offset can reduce headroom and bias dynamics processors.",
                    evidence=[f"DC offset is {dc:.4f}."],
                    suggested_action="Apply a DC blocking high-pass filter before compression or limiting.",
                    expected_improvement="Restores symmetric headroom.",
                    affected_tracks=["mix"],
                    confidence=0.8,
                    risk="A filter set too high can thin the low end.",
                    actions=[
                        self._action(
                            "technical.dc_block_hpf",
                            "high_pass_filter",
                            "mix",
                            {"frequency_hz": 20.0, "slope_db_per_octave": 12},
                            "Remove DC and subsonic bias.",
                            "Improves technical headroom.",
                            "Avoid higher cutoff unless source requires rumble cleanup.",
                            0.75,
                        )
                    ],
                )
            )
        return issues

    def _gain_staging_rules(self, analysis: MixAnalysis) -> List[RuleIssue]:
        level = analysis.mix.level
        tech = self.thresholds.get("technical", {})
        headroom = float(level.get("headroom_db", 0.0))
        target = float(tech.get("mix_headroom_warning_db", 3.0))
        if headroom < target:
            return [
                self._issue(
                    issue_id="gain_staging.low_headroom",
                    group="A. gain_staging",
                    name="Low pre-master headroom",
                    severity="high",
                    explanation="The mix does not leave enough conservative headroom for downstream processing.",
                    evidence=[
                        f"Headroom is {headroom:.2f} dB; recommended minimum is {target:.2f} dB.",
                        f"PLR is {float(level.get('plr_db', 0.0)):.2f} dB.",
                    ],
                    suggested_action="Reduce mix/bus gain 0.5-2 dB; do not raise loudness to judge improvement.",
                    expected_improvement="Cleaner gain staging and safer A/B evaluation.",
                    affected_tracks=["mix"],
                    confidence=0.86,
                    risk="Lower level can sound worse unless A/B is loudness-matched.",
                    actions=[
                        self._action(
                            "gain_stage.lower_mix",
                            "gain_adjustment",
                            "mix",
                            {"gain_db": -1.0},
                            "Create conservative headroom before further processing.",
                            "Improves true-peak headroom.",
                            "Requires loudness-matched monitoring.",
                            0.86,
                            mode="safe_apply",
                        )
                    ],
                )
            ]
        return []

    def _balance_rules(self, analysis: MixAnalysis) -> List[RuleIssue]:
        rel = analysis.stem_relationships.get("relative_vocal_to_accompaniment_db")
        if rel is None:
            return []
        low, high = _profile_range(self.profile, "vocal_prominence_db", (1.0, 5.0))
        if rel < low:
            action_target = "accompaniment"
            action = self._action(
                "balance.vocal_masking_accompaniment_down",
                "gain_adjustment",
                action_target,
                {"gain_db": -0.75},
                "Improve vocal/lead readability without automatically boosting the lead.",
                "Raises relative vocal clarity while preserving headroom.",
                "Can reduce track energy if overused.",
                0.72,
            )
            return [
                self._issue(
                    issue_id="balance.vocal_low",
                    group="B. source_balance",
                    name="Lead/vocal may be under the accompaniment",
                    severity="medium",
                    explanation="The vocal or lead appears low relative to the accompaniment for this profile.",
                    evidence=[f"Vocal-to-accompaniment ratio is {rel:.2f} dB; profile expects {low:.1f}-{high:.1f} dB."],
                    suggested_action="First check masking/pan/envelope; then lower competing accompaniment 0.5-1 dB or ride the lead manually.",
                    expected_improvement="Better lyric/lead intelligibility without sacrificing headroom.",
                    affected_tracks=["lead_vocal", "accompaniment"],
                    confidence=0.72,
                    risk=action.risk,
                    actions=[action],
                )
            ]
        if rel > high:
            return [
                self._issue(
                    issue_id="balance.vocal_too_forward",
                    group="B. source_balance",
                    name="Lead/vocal may be too forward",
                    severity="low",
                    explanation="The lead/vocal sits above the profile tolerance and may detach from the track.",
                    evidence=[f"Vocal-to-accompaniment ratio is {rel:.2f} dB; profile expects {low:.1f}-{high:.1f} dB."],
                    suggested_action="Consider lowering vocal/lead 0.5 dB or adding context with short ambience, after listening.",
                    expected_improvement="More cohesive balance.",
                    affected_tracks=["lead_vocal"],
                    confidence=0.55,
                    risk="Reducing lead can hurt intelligibility.",
                )
            ]
        return []

    def _tonal_rules(self, analysis: MixAnalysis) -> List[RuleIssue]:
        spectral = analysis.mix.spectral
        ratios = spectral.get("band_energy_ratios", {})
        tonal = self.thresholds.get("tonal", {})
        issues: List[RuleIssue] = []
        if float(spectral.get("muddiness_proxy", 0.0)) > float(tonal.get("mud_ratio_warning", 0.22)):
            issues.append(
                self._issue(
                    issue_id="tonal.low_mid_buildup",
                    group="C. tonal_balance",
                    name="Low-mid buildup",
                    severity="medium",
                    explanation="The mix has excess low-mid density, a common cause of reduced clarity.",
                    evidence=[
                        f"Low-mid ratio is {ratios.get('low_mid', 0.0):.3f}.",
                        "Low-mid cleanup should be checked in full mix, not solo.",
                    ],
                    suggested_action="Try a broad 180-350 Hz cut of 1-2 dB on the likely bus, or dynamic EQ if buildup is section-dependent.",
                    expected_improvement="Less muddiness and better separation.",
                    affected_tracks=["mix"],
                    confidence=0.68,
                    risk="Too much reduction can make guitars/keys/vocals thin.",
                    actions=[
                        self._action(
                            "tonal.low_mid_cut",
                            "parametric_eq",
                            "mix",
                            {"frequency_hz": 260.0, "gain_db": -1.5, "q": 0.9},
                            "Reduce measured low-mid buildup.",
                            "Improves clarity while using a small reversible move.",
                            "Can remove warmth if over-applied.",
                            0.68,
                        )
                    ],
                )
            )
        if float(spectral.get("harshness_proxy", 0.0)) > float(tonal.get("harshness_ratio_warning", 0.18)):
            issues.append(
                self._issue(
                    issue_id="tonal.harsh_presence",
                    group="C. tonal_balance",
                    name="Harshness/presence risk",
                    severity="medium",
                    explanation="The upper-mid/presence region is elevated and may cause fatigue.",
                    evidence=[f"Harshness proxy is {float(spectral.get('harshness_proxy', 0.0)):.3f}."],
                    suggested_action="Use narrow EQ or dynamic EQ around 2.5-6 kHz on the offending stems; avoid global dulling.",
                    expected_improvement="Less fatigue while preserving intelligibility.",
                    affected_tracks=["mix"],
                    confidence=0.64,
                    risk="Over-cutting presence reduces vocal/guitar articulation.",
                    actions=[
                        self._action(
                            "tonal.harsh_dynamic_eq",
                            "dynamic_eq_placeholder",
                            "offending_stems",
                            {"frequency_hz": 3500.0, "gain_db": -1.0, "q": 2.0},
                            "Control harshness only where it occurs.",
                            "Keeps clarity while reducing fatigue.",
                            "Placeholder only unless a real dynamic EQ is available.",
                            0.64,
                        )
                    ],
                )
            )
        return issues

    def _masking_rules(self, analysis: MixAnalysis) -> List[RuleIssue]:
        masking = self.thresholds.get("masking", {})
        threshold = float(masking.get("spectral_overlap_warning", 0.82))
        conflicts = analysis.stem_relationships.get("stem_spectral_overlap_top", [])
        issues = []
        for conflict in conflicts[:3]:
            overlap = float(conflict.get("overlap", 0.0))
            if overlap < threshold:
                continue
            a = str(conflict.get("a", ""))
            b = str(conflict.get("b", ""))
            issues.append(
                self._issue(
                    issue_id=f"masking.{a}.{b}",
                    group="D. masking",
                    name="High stem spectral overlap",
                    severity="medium",
                    explanation=f"{a} and {b} occupy similar broad spectral space.",
                    evidence=[f"Spectral overlap proxy is {overlap:.2f}."],
                    suggested_action="Check balance and panorama first; then use small EQ or dynamic EQ only where conflict is audible.",
                    expected_improvement="Better separation without unnecessary solo-mode EQ.",
                    affected_tracks=[a, b],
                    confidence=min(0.8, 0.45 + overlap / 2.0),
                    risk="Blind EQ can weaken either source if the overlap is musically intentional.",
                    actions=[
                        self._action(
                            f"masking.dynamic_eq.{a}.{b}",
                            "sidechain_suggestion",
                            b,
                            {"trigger": a, "frequency_hz": 3000.0, "gain_db": -1.0, "q": 1.4},
                            "Reduce masking only when the priority source is active.",
                            "Improves source readability.",
                            "Requires listening check and a real sidechain/dynamic EQ path.",
                            0.58,
                        )
                    ],
                )
            )
        return issues

    def _dynamics_rules(self, analysis: MixAnalysis) -> List[RuleIssue]:
        level = analysis.mix.level
        dynamics = self.thresholds.get("dynamics", {})
        crest = float(level.get("crest_factor_db", 0.0))
        lra = float(level.get("loudness_range_lu", 0.0))
        issues = []
        if crest < float(dynamics.get("crest_factor_low_warning_db", 6.0)):
            issues.append(
                self._issue(
                    issue_id="dynamics.low_crest_factor",
                    group="E. dynamics",
                    name="Low crest factor",
                    severity="medium",
                    explanation="The mix may be over-compressed or transient-poor.",
                    evidence=[f"Crest factor is {crest:.2f} dB."],
                    suggested_action="Do not add loudness. Review bus compression/limiting and preserve drum transients where style needs punch.",
                    expected_improvement="Better punch and lower pumping risk.",
                    affected_tracks=["mix"],
                    confidence=0.62,
                    risk="Some dense genres intentionally use lower crest factor.",
                )
            )
        if lra > float(dynamics.get("lra_high_warning_lu", 18.0)):
            issues.append(
                self._issue(
                    issue_id="dynamics.high_lra",
                    group="E. dynamics",
                    name="Large macro-dynamic range",
                    severity="low",
                    explanation="Sections may translate unevenly across playback levels.",
                    evidence=[f"LRA is {lra:.2f} LU."],
                    suggested_action="Consider manual rides before compression; preserve section contrast.",
                    expected_improvement="More stable translation without flattening arrangement.",
                    affected_tracks=["mix"],
                    confidence=0.5,
                    risk="Over-smoothing can damage emotional arrangement contrast.",
                )
            )
        return issues

    def _stereo_rules(self, analysis: MixAnalysis) -> List[RuleIssue]:
        stereo = analysis.mix.stereo
        thresholds = self.thresholds.get("stereo", {})
        issues = []
        if bool(stereo.get("phase_cancellation_risk", False)):
            issues.append(
                self._issue(
                    issue_id="stereo.phase_or_mono_loss",
                    group="F. stereo_pan_mono",
                    name="Mono compatibility risk",
                    severity="high",
                    explanation="Stereo information may collapse poorly in mono.",
                    evidence=[
                        f"Correlation: {float(stereo.get('inter_channel_correlation', 1.0)):.2f}.",
                        f"Mono fold-down loss: {float(stereo.get('mono_fold_down_loss_db', 0.0)):.2f} dB.",
                    ],
                    suggested_action="Reduce risky width, check polarity/time effects, and loudness-match mono A/B.",
                    expected_improvement="Better translation to mono and small systems.",
                    affected_tracks=["mix"],
                    confidence=0.82,
                    risk="Narrowing everything removes contrast; fix the conflicting layer if possible.",
                    actions=[
                        self._action(
                            "stereo.reduce_width",
                            "stereo_width_adjustment",
                            "mix",
                            {"width_delta": -0.1},
                            "Reduce phase-cancelling side energy conservatively.",
                            "Improves mono compatibility.",
                            "May make the mix feel smaller.",
                            0.72,
                        )
                    ],
                )
            )
        low_width = float(stereo.get("low_frequency_stereo_width", 0.0))
        if low_width > float(thresholds.get("low_frequency_width_warning", 0.25)):
            issues.append(
                self._issue(
                    issue_id="stereo.low_end_width",
                    group="F. stereo_pan_mono",
                    name="Low-frequency stereo width warning",
                    severity="medium",
                    explanation="Sub/bass side energy can reduce club and mono translation.",
                    evidence=[f"Low-frequency width proxy is {low_width:.2f}."],
                    suggested_action="Keep sub/bass near mono or narrow only the low band.",
                    expected_improvement="More stable low-end translation.",
                    affected_tracks=["mix", "bass", "kick"],
                    confidence=0.7,
                    risk="Some synth or cinematic material intentionally uses wide low effects.",
                    actions=[
                        self._action(
                            "stereo.low_band_mono",
                            "mid_side_adjustment",
                            "mix",
                            {"band": "sub_bass", "side_gain_db": -1.5},
                            "Reduce low-frequency side energy.",
                            "Improves mono/large-system low-end stability.",
                            "Do not apply if wide low-end is intentional.",
                            0.7,
                        )
                    ],
                )
            )
        return issues

    def _reference_rules(self, analysis: MixAnalysis) -> List[RuleIssue]:
        if not analysis.reference.enabled:
            return []
        tolerance = self.profile.get("reference_tolerance", {})
        distance = float(analysis.reference.spectral_distance)
        max_distance = float(tolerance.get("spectral_distance", 0.35))
        if distance <= max_distance:
            return []
        largest_band = max(
            analysis.reference.band_differences_db.items(),
            key=lambda item: abs(item[1]),
        )
        return [
            self._issue(
                issue_id="reference.spectral_distance",
                group="reference_matching",
                name="Reference tonal distance",
                severity="low",
                explanation="After loudness-normalized comparison, tonal balance differs from the reference more than the genre prior expects.",
                evidence=[
                    f"Spectral distance is {distance:.2f}; profile tolerance is {max_distance:.2f}.",
                    f"Largest band delta: {largest_band[0]} {largest_band[1]:+.2f} dB.",
                ],
                suggested_action="Use reference as a tolerance guide only; adjust the largest audible band with small EQ moves after loudness-matched listening.",
                expected_improvement="Closer genre/reference translation without copying a protected recording.",
                affected_tracks=["mix"],
                confidence=0.56,
                risk="Reference matching can erase the song's intent if followed mechanically.",
            )
        ]


def build_quality_dashboard(analysis: MixAnalysis, issues: List[RuleIssue]) -> QualityDashboard:
    """Build independent quality scores from metrics and issue severity."""
    severity_penalty = {
        "critical": 0.28,
        "high": 0.18,
        "medium": 0.10,
        "low": 0.04,
        "info": 0.01,
    }

    def penalty_for(group_prefix: str) -> float:
        return sum(
            severity_penalty.get(issue.severity, 0.0)
            for issue in issues
            if issue.group.startswith(group_prefix)
        )

    level = analysis.mix.level
    stereo = analysis.mix.stereo
    technical_penalty = penalty_for("I.") + max(0.0, float(level.get("clip_count", 0)) / 100.0)
    gain_penalty = penalty_for("A.")
    masking_penalty = penalty_for("D.")
    tonal_penalty = penalty_for("C.")
    dynamics_penalty = penalty_for("E.")
    stereo_penalty = penalty_for("F.")
    reference_penalty = 0.0
    if analysis.reference.enabled:
        reference_penalty = min(0.5, float(analysis.reference.spectral_distance))
    artifact_risk = 1.0 - _score_from_penalty(technical_penalty)
    notes = []
    if not analysis.stems:
        notes.append("Balance and masking scores are limited because stems were not supplied.")
    if not analysis.reference.enabled:
        notes.append("Reference match score is neutral because no reference was supplied.")
    components = [
        _score_from_penalty(technical_penalty),
        _score_from_penalty(gain_penalty),
        _score_from_penalty(masking_penalty),
        _score_from_penalty(tonal_penalty),
        _score_from_penalty(dynamics_penalty),
        _score_from_penalty(stereo_penalty),
    ]
    return QualityDashboard(
        technical_health_score=components[0],
        gain_staging_score=components[1],
        balance_score=_score_from_penalty(penalty_for("B.")),
        tonal_balance_score=components[3],
        masking_score=components[2],
        dynamics_score=components[4],
        stereo_mono_score=components[5],
        space_clarity_score=0.5,
        reference_match_score=_score_from_penalty(reference_penalty) if analysis.reference.enabled else 0.5,
        translation_score=_score_from_penalty(
            stereo_penalty
            + (0.15 if bool(stereo.get("low_frequency_stereo_width_warning", False)) else 0.0)
        ),
        artifact_risk_score=round(_clamp01(artifact_risk), 3),
        overall_recommendation_confidence=round(float(sum(components) / len(components)), 3),
        notes=notes,
    )
