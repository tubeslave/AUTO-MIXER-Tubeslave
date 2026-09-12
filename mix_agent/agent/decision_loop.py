"""Offline decision loop and conservative apply workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Optional

from backend.heuristics.spectral_ceiling_eq import (
    SpectralCeilingEQAnalyzer,
    SpectralCeilingEQConfig,
    SpectralCeilingEQProposal,
    normalize_role,
)
from mix_agent.actions import render_conservative_mix
from mix_agent.analysis import analyze_loaded_context, load_audio_context
from mix_agent.models import MixAction, SuggestionPlan
from mix_agent.reporting import write_json_report, write_markdown_report

from .evaluator import compare_before_after
from .planner import build_suggestion_plan


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_spectral_ceiling_config() -> SpectralCeilingEQConfig:
    config_path = REPO_ROOT / "config" / "automixer.yaml"
    if not config_path.exists():
        return SpectralCeilingEQConfig()
    try:
        import yaml

        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return SpectralCeilingEQConfig()
    section = payload.get("spectral_ceiling_eq", {}) if isinstance(payload, dict) else {}
    return SpectralCeilingEQConfig.from_mapping(section)


def _offline_eq_band_for_freq(freq_hz: float) -> int:
    if freq_hz < 180.0:
        return 1
    if freq_hz < 1200.0:
        return 2
    if freq_hz < 5500.0:
        return 3
    return 4


def _spectral_actions_for_proposal(
    proposal: SpectralCeilingEQProposal,
    target: str,
) -> List[MixAction]:
    mode = "safe_apply" if proposal.should_apply else "dry_run"
    status = "proposed" if proposal.should_apply else "logged_only"
    actions: List[MixAction] = []
    if proposal.low_cut_hz and proposal.role != "mix_bus":
        actions.append(
            MixAction(
                id=f"spectral_ceiling.{target}.hpf",
                action_type="high_pass_filter",
                target=target,
                parameters={"frequency_hz": float(proposal.low_cut_hz)},
                safe_range={"frequency_hz": [20.0, 400.0]},
                mode=mode,
                reason=f"Spectral ceiling roll-off for {proposal.role}",
                expected_improvement="Remove rumble before broad tonal EQ decisions.",
                risk="May thin source if role detection is wrong.",
                confidence=proposal.confidence,
                reversible=True,
                backend_mapping={"source": "spectral_ceiling_eq"},
                status=status,
            )
        )
    for idx, band in enumerate(proposal.bands, start=1):
        if band.action not in {"attenuate", "lift"}:
            continue
        actions.append(
            MixAction(
                id=f"spectral_ceiling.{target}.{idx}.{band.name}",
                action_type="parametric_eq",
                target=target,
                parameters={
                    "band": _offline_eq_band_for_freq(band.freq_hz),
                    "frequency_hz": float(band.freq_hz),
                    "gain_db": float(band.gain_db),
                    "q": float(band.q),
                },
                safe_range={
                    "gain_db": [
                        float(proposal.safety.get("max_cut_db", -3.0)),
                        float(proposal.safety.get("max_boost_db", 1.0)),
                    ],
                    "q": [0.5, float(proposal.safety.get("max_q", 2.0))],
                },
                mode=mode,
                reason=f"Spectral ceiling {band.action}: {band.reason}",
                expected_improvement=(
                    "Move broad smoothed spectrum toward the role target without match EQ."
                ),
                risk="Broad EQ can still hurt tone; verify in context.",
                confidence=proposal.confidence,
                reversible=True,
                backend_mapping={
                    "source": "spectral_ceiling_eq",
                    "rule": band.rule,
                    "target_profile": proposal.target_profile,
                },
                status=status,
            )
        )
    return actions


def _append_spectral_ceiling_eq(
    plan: SuggestionPlan,
    loaded,
) -> None:
    config = _load_spectral_ceiling_config()
    analyzer = SpectralCeilingEQAnalyzer(config)
    proposals = []
    lead_present = any(
        normalize_role(info.role) == "lead_vocal"
        for info in loaded.stem_info.values()
    )
    for name, audio in loaded.stems.items():
        info = loaded.stem_info[name]
        proposal = analyzer.analyze(
            audio,
            instrument_role=info.role,
            sample_rate=loaded.mix_sample_rate,
            track_id=name,
            role_confidence=1.0,
            lead_vocal_active=lead_present and normalize_role(info.role) != "lead_vocal",
            lead_vocal_confidence=1.0 if lead_present else 0.0,
        )
        proposals.append(proposal.to_dict())
        plan.actions.extend(_spectral_actions_for_proposal(proposal, name))

    if loaded.mix_audio is not None:
        mix_proposal = analyzer.analyze(
            loaded.mix_audio,
            instrument_role="mix_bus",
            sample_rate=loaded.mix_sample_rate,
            track_id="mix_bus",
            role_confidence=1.0,
            lead_vocal_active=lead_present,
            lead_vocal_confidence=1.0 if lead_present else 0.0,
        )
        proposals.append(mix_proposal.to_dict())

    plan.audit_trail.append({"spectral_ceiling_eq": proposals})


def analyze(
    *,
    stems: str | None = None,
    mix: str | None = None,
    reference: str | None = None,
    genre: str = "neutral",
    target_platform: str = "streaming",
) -> SuggestionPlan:
    """Load audio, analyze it and build a suggestion plan."""
    loaded = load_audio_context(
        stems_path=stems,
        mix_path=mix,
        reference_path=reference,
        genre=genre,
        target_platform=target_platform,
    )
    analysis = analyze_loaded_context(loaded)
    plan = build_suggestion_plan(analysis)
    _append_spectral_ceiling_eq(plan, loaded)
    return plan


def suggest(
    *,
    stems: str | None = None,
    mix: str | None = None,
    reference: str | None = None,
    genre: str = "neutral",
    target_platform: str = "streaming",
    out: str | None = None,
) -> SuggestionPlan:
    """Analyze and optionally write a JSON/Markdown report."""
    plan = analyze(
        stems=stems,
        mix=mix,
        reference=reference,
        genre=genre,
        target_platform=target_platform,
    )
    if out:
        if str(out).lower().endswith(".md"):
            write_markdown_report(plan, out)
        else:
            write_json_report(plan, out)
    return plan


def _load_actions_from_suggestions(path: str | Path) -> List[MixAction]:
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    action_payloads = payload.get("actions", payload if isinstance(payload, list) else [])
    return [
        MixAction(
            id=str(item.get("id", f"action_{idx}")),
            action_type=str(item.get("action_type", "")),
            target=str(item.get("target", "")),
            parameters=dict(item.get("parameters", {})),
            safe_range=dict(item.get("safe_range", {})),
            mode=str(item.get("mode", "recommend")),
            reason=str(item.get("reason", "")),
            expected_improvement=str(item.get("expected_improvement", "")),
            risk=str(item.get("risk", "")),
            confidence=float(item.get("confidence", 0.5)),
            reversible=bool(item.get("reversible", True)),
            backend_mapping=dict(item.get("backend_mapping", {})),
            status=str(item.get("status", "proposed")),
        )
        for idx, item in enumerate(action_payloads)
        if isinstance(item, dict)
    ]


def apply_conservative(
    *,
    stems: str,
    out: str,
    suggestions: str | None = None,
    reference: str | None = None,
    genre: str = "neutral",
    target_platform: str = "streaming",
    actions: Optional[Iterable[MixAction]] = None,
    report: str | None = None,
) -> SuggestionPlan:
    """Apply only conservative reversible offline operations and re-analyze."""
    loaded = load_audio_context(
        stems_path=stems,
        mix_path=None,
        reference_path=reference,
        genre=genre,
        target_platform=target_platform,
    )
    before_audio = loaded.mix_audio.copy()
    plan = build_suggestion_plan(analyze_loaded_context(loaded))
    _append_spectral_ceiling_eq(plan, loaded)
    selected_actions = list(actions or [])
    if suggestions:
        selected_actions = _load_actions_from_suggestions(suggestions)
    if not selected_actions:
        selected_actions = [
            action
            for action in plan.actions
            if action.action_type in {"gain_adjustment", "high_pass_filter", "parametric_eq", "pan_adjustment"}
        ]

    roles = {name: info.role for name, info in loaded.stem_info.items()}
    render_report = render_conservative_mix(
        loaded.stems,
        roles,
        loaded.mix_sample_rate,
        selected_actions,
        out,
    )
    after_loaded = load_audio_context(
        stems_path=None,
        mix_path=out,
        reference_path=reference,
        genre=genre,
        target_platform=target_platform,
    )
    after_audio = after_loaded.mix_audio
    plan.applied_actions = selected_actions
    plan.audit_trail.append({"render": render_report})
    plan.audit_trail.append(
        {
            "before_after": compare_before_after(
                before_audio,
                after_audio,
                loaded.mix_sample_rate,
            )
        }
    )
    if report:
        if str(report).lower().endswith(".md"):
            write_markdown_report(plan, report)
        else:
            write_json_report(plan, report)
    return plan
