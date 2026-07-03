"""Candidate action generation for offline AI mixing tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ai_mixing_pipeline.models import CandidateAction, MixCandidate

PLANNER_SOURCE = "automix_toolkit_fxnorm_diffmst_deepafx_action_planner"
PLANNER_TECHNOLOGIES = [
    "automix-toolkit",
    "FxNorm-Automix",
    "Diff-MST",
    "DeepAFx-style differentiable effects",
]


class OfflineActionPlanner:
    """Generate bounded candidate actions without applying them."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = dict(config or {})

    def generate_candidates(
        self,
        *,
        stems: dict[str, Any],
        stem_roles: dict[str, str],
        initial_metrics: dict[str, Any],
        reference_profile: dict[str, Any] | None = None,
    ) -> list[MixCandidate]:
        """Create the standard offline candidate set."""

        candidates: list[MixCandidate] = [
            MixCandidate(
                candidate_id="000_initial_mix",
                label="no_change",
                render_filename="000_initial_mix.wav",
                explanation="No-change baseline candidate.",
            )
        ]
        candidates.append(self._gain_balance_candidate(stems, stem_roles, initial_metrics))
        candidates.append(self._eq_cleanup_candidate(stems, stem_roles, initial_metrics))
        candidates.append(self._compression_candidate(stems, stem_roles))
        candidates.append(self._fx_candidate(stems, stem_roles))
        candidates.append(self._vocal_clarity_candidate(stems, stem_roles))
        candidates.append(self._low_mid_candidate(stems, stem_roles))
        candidates.append(self._bass_kick_candidate(stems, stem_roles))
        for candidate in candidates:
            candidate.metadata["planner_role"] = "candidate_action_generation"
            candidate.metadata["planner_technologies"] = list(PLANNER_TECHNOLOGIES)
            for action in candidate.actions:
                action.source = PLANNER_SOURCE
        return candidates

    def participation_status(self) -> dict[str, Any]:
        """Report how the research action-planner layer participated."""

        repo_root = Path(__file__).resolve().parents[2]
        references = {
            "automix-toolkit": repo_root / "external" / "automix-toolkit",
            "FxNorm-Automix": repo_root / "external" / "FxNorm-automix",
            "Diff-MST": repo_root / "external" / "Diff-MST",
            "DeepAFx-style differentiable effects": repo_root / "external" / "dasp-pytorch",
        }
        detected = {
            name: path.exists()
            for name, path in references.items()
        }
        return {
            "name": "automix_toolkit_fxnorm_diffmst_deepafx",
            "role": "candidate_action_planner_research_base",
            "available": True,
            "technologies": list(PLANNER_TECHNOLOGIES),
            "detected_reference_code": detected,
            "fallback": "local bounded candidate generator",
            "warnings": [
                "Research planners participate through bounded candidate-action templates; no external planner dependency is required.",
                "Action planner proposes actions only; Sandbox Renderer and Safety Governor decide whether any candidate is accepted.",
            ],
        }

    def _gain_balance_candidate(
        self,
        stems: dict[str, Any],
        stem_roles: dict[str, str],
        initial_metrics: dict[str, Any],
    ) -> MixCandidate:
        actions: list[CandidateAction] = []
        for name in stems:
            role = stem_roles.get(name, "unknown")
            if "vocal" in role:
                gain = 0.6
            elif role in {"guitars", "keys", "playback", "drums"}:
                gain = -0.6
            elif role in {"kick", "bass", "snare"}:
                gain = -0.2
            else:
                gain = -0.3
            actions.append(
                CandidateAction(
                    action_type="gain_change",
                    target=name,
                    parameters={"gain_db": gain},
                    reason=f"Conservative gain-balance probe for {role}.",
                    safe_range={"gain_db": [-1.0, 1.0]},
                )
            )
        return MixCandidate(
            candidate_id="001_candidate_gain_balance",
            label="gain_balance",
            render_filename="001_candidate_gain_balance.wav",
            actions=actions,
            explanation="Small stem-level gain moves inspired by automix-toolkit/FxNorm balance ideas.",
        )

    def _eq_cleanup_candidate(
        self,
        stems: dict[str, Any],
        stem_roles: dict[str, str],
        initial_metrics: dict[str, Any],
    ) -> MixCandidate:
        actions: list[CandidateAction] = []
        for name in stems:
            role = stem_roles.get(name, "unknown")
            if role not in {"kick", "bass"}:
                actions.append(
                    CandidateAction(
                        action_type="high_pass_filter",
                        target=name,
                        parameters={"frequency_hz": 70.0 if "vocal" not in role else 95.0},
                        reason="Clean unnecessary low-frequency energy before mix-bus decisions.",
                        safe_range={"frequency_hz": [20.0, 140.0]},
                    )
                )
            if role in {"guitars", "keys", "drums", "vocal", "lead_vocal", "backing_vocal"}:
                actions.append(
                    CandidateAction(
                        action_type="eq_correction",
                        target=name,
                        parameters={"frequency_hz": 320.0, "gain_db": -0.9, "q": 0.9},
                        reason="Low-mid cleanup candidate on likely masking sources.",
                        safe_range={"gain_db": [-1.5, 0.0], "q": [0.5, 2.0]},
                    )
                )
        return MixCandidate(
            candidate_id="002_candidate_eq_cleanup",
            label="eq_cleanup",
            render_filename="002_candidate_eq_cleanup.wav",
            actions=actions,
            explanation="EQ cleanup candidate with small cuts and HPF moves only.",
        )

    def _compression_candidate(self, stems: dict[str, Any], stem_roles: dict[str, str]) -> MixCandidate:
        actions = []
        for name in stems:
            role = stem_roles.get(name, "unknown")
            if role in {"lead_vocal", "vocal", "bass", "kick", "snare", "drums"}:
                actions.append(
                    CandidateAction(
                        action_type="compression_correction",
                        target=name,
                        parameters={
                            "threshold_db": -18.0,
                            "ratio": 1.5 if "vocal" in role else 1.35,
                            "attack_ms": 15.0,
                            "release_ms": 140.0,
                            "makeup_db": 0.0,
                        },
                        reason="Gentle dynamics-control probe; no makeup gain boost.",
                        safe_range={"ratio": [1.0, 2.0], "makeup_db": [0.0, 0.0]},
                    )
                )
        return MixCandidate(
            candidate_id="003_candidate_compression",
            label="compression",
            render_filename="003_candidate_compression.wav",
            actions=actions,
            explanation="Gentle compression candidate; action planner proposes, renderer tests offline.",
        )

    def _fx_candidate(self, stems: dict[str, Any], stem_roles: dict[str, str]) -> MixCandidate:
        actions = []
        side = -0.25
        for name in stems:
            role = stem_roles.get(name, "unknown")
            if role in {"guitars", "keys", "backing_vocal"}:
                actions.append(
                    CandidateAction(
                        action_type="pan_change",
                        target=name,
                        parameters={"pan": side},
                        reason="Subtle space candidate without adding reverb/delay.",
                        safe_range={"pan": [-0.35, 0.35]},
                    )
                )
                side *= -1.0
        actions.append(
            CandidateAction(
                action_type="fx_send_change",
                target="all",
                parameters={"send_db": -1.0, "effect": "reverb_delay"},
                reason="Report-only FX wash reduction candidate.",
                safe_range={"send_db": [-3.0, 0.0]},
            )
        )
        return MixCandidate(
            candidate_id="004_candidate_fx",
            label="fx_space",
            render_filename="004_candidate_fx.wav",
            actions=actions,
            explanation="Space/FX candidate; send moves are logged but not sent to a console.",
        )

    def _vocal_clarity_candidate(self, stems: dict[str, Any], stem_roles: dict[str, str]) -> MixCandidate:
        actions: list[CandidateAction] = []
        for name in stems:
            role = stem_roles.get(name, "unknown")
            if "vocal" in role:
                actions.append(
                    CandidateAction(
                        action_type="gain_change",
                        target=name,
                        parameters={"gain_db": 0.5},
                        reason="Small vocal clarity lift candidate.",
                        safe_range={"gain_db": [0.0, 0.8]},
                    )
                )
            elif role in {"guitars", "keys", "playback"}:
                actions.append(
                    CandidateAction(
                        action_type="eq_correction",
                        target=name,
                        parameters={"frequency_hz": 2500.0, "gain_db": -0.7, "q": 1.1},
                        reason="Make vocal presence space by cutting competing stems.",
                        safe_range={"gain_db": [-1.0, 0.0]},
                    )
                )
        return MixCandidate(
            candidate_id="005_candidate_vocal_clarity",
            label="vocal_clarity",
            render_filename="005_candidate_vocal_clarity.wav",
            actions=actions,
            explanation="Vocal clarity candidate using source/stem moves, not master EQ.",
        )

    def _low_mid_candidate(self, stems: dict[str, Any], stem_roles: dict[str, str]) -> MixCandidate:
        actions = [
            CandidateAction(
                action_type="eq_correction",
                target=name,
                parameters={"frequency_hz": 280.0, "gain_db": -1.1, "q": 0.8},
                reason="Dedicated low-mid mud reduction candidate.",
                safe_range={"gain_db": [-1.5, 0.0]},
            )
            for name, role in stem_roles.items()
            if role not in {"kick", "bass"}
        ]
        return MixCandidate(
            candidate_id="006_candidate_low_mid_mud_reduction",
            label="low_mid_mud_reduction",
            render_filename="006_candidate_low_mid_mud_reduction.wav",
            actions=actions,
            explanation="Low-mid mud reduction candidate.",
        )

    def _bass_kick_candidate(self, stems: dict[str, Any], stem_roles: dict[str, str]) -> MixCandidate:
        actions: list[CandidateAction] = []
        for name, role in stem_roles.items():
            if role == "kick":
                actions.append(
                    CandidateAction(
                        action_type="gain_change",
                        target=name,
                        parameters={"gain_db": 0.4},
                        reason="Kick/bass balance probe: small kick anchor lift.",
                        safe_range={"gain_db": [0.0, 0.6]},
                    )
                )
            if role == "bass":
                actions.append(
                    CandidateAction(
                        action_type="eq_correction",
                        target=name,
                        parameters={"frequency_hz": 90.0, "gain_db": -0.5, "q": 1.0},
                        reason="Bass/kick balance probe: leave kick pocket.",
                        safe_range={"gain_db": [-0.8, 0.0]},
                    )
                )
        return MixCandidate(
            candidate_id="007_candidate_bass_kick_balance",
            label="bass_kick_balance",
            render_filename="007_candidate_bass_kick_balance.wav",
            actions=actions,
            explanation="Bass/kick balance candidate.",
        )
