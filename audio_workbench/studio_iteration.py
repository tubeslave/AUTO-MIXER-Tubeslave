"""Auditable STUDIO-only bridge from Perceptual Critic to offline mastering.

This module deliberately does not grant machine authority over subjective audio.
A candidate may be rendered for listening, but the baseline cannot be promoted here.
Rejected candidates are rolled back to the immutable baseline before mastering.
Post-hoc human calibration may be attached as annotation-only evidence; it cannot
rewrite the historical machine transition or baseline state.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from .autonomous_loop import resolve_candidate_iteration
from .mastering.offline import audio_digest, deliver_master
from .mixing.perceptual_calibration import classify_human_machine_disagreement


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_stereo(path: Path) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    if audio.ndim != 2 or audio.shape[1] != 2:
        raise ValueError(f"studio iteration requires stereo audio: {path}")
    if len(audio) == 0 or not np.isfinite(audio).all():
        raise ValueError(f"studio iteration requires finite non-empty audio: {path}")
    return audio, int(sr)


def _validate_plan(plan: dict[str, Any], critic_result: dict[str, Any]) -> str:
    target = str(critic_result.get("target", "")).strip()
    if not target:
        raise ValueError("critic_result.target is required")
    if str(plan.get("target", "")).strip() != target:
        raise ValueError("causal plan target differs from Perceptual Critic target")
    candidates = list(plan.get("candidates") or [])
    if not any(c.get("type") == "bypass" for c in candidates if isinstance(c, dict)):
        raise ValueError("causal plan must retain a bypass/no-change counterfactual")
    return target


def _decision_snapshot(report: dict[str, Any]) -> dict[str, Any]:
    """Copy fields that human calibration is forbidden to modify."""
    transition = dict(report.get("transition") or {})
    delivery = dict(report.get("delivery") or {})
    return {
        "status": report.get("status"),
        "baseline_promoted": report.get("baseline_promoted"),
        "baseline_after": report.get("baseline_after"),
        "delivery_role": delivery.get("role"),
        "delivery_source_id": delivery.get("source_id"),
        "rolled_back_to_baseline": delivery.get("rolled_back_to_baseline"),
        "transition_status": transition.get("status"),
        "transition_next_action": transition.get("next_action"),
        "transition_baseline_before": transition.get("baseline_before"),
        "transition_baseline_after": transition.get("baseline_after"),
        "transition_promote_baseline": transition.get("promote_baseline"),
        "transition_rollback_candidate": transition.get("rollback_candidate"),
        "transition_protected_regressions": deepcopy(transition.get("protected_regressions")),
    }


def add_human_calibration_annotation(
    report: dict[str, Any],
    human_review: dict[str, Any] | None,
) -> dict[str, Any]:
    """Attach post-listening calibration evidence without changing a decision.

    ``None`` is an explicit no-op so ordinary autonomous reports remain valid before
    a listener responds.  A supplied review is classified by the existing
    Perceptual Human Calibration layer and written under ``human_calibration`` only.
    Historical ``status``, transition, delivery routing and baseline fields remain
    byte-for-byte equivalent as Python values.  Invalid evidence fails closed.
    """
    if not isinstance(report, dict):
        raise TypeError("report must be a dictionary")
    if report.get("schema") != "studio-autonomous-iteration-v2":
        raise ValueError("unsupported studio iteration report schema")

    enriched = deepcopy(report)
    if human_review is None:
        return enriched
    if not isinstance(human_review, dict):
        raise TypeError("human_review must be a dictionary or None")
    if "human_calibration" in report:
        raise ValueError("iteration report already contains human calibration evidence")

    critic = report.get("critic")
    if not isinstance(critic, dict):
        raise ValueError("iteration report critic evidence is missing")
    before = _decision_snapshot(report)
    annotation = classify_human_machine_disagreement(critic, human_review)
    annotation = {
        **annotation,
        "role": "annotation_only",
        "source_report_schema": str(report["schema"]),
        "historical_machine_transition_preserved": True,
        "historical_delivery_routing_preserved": True,
        "historical_baseline_state_preserved": True,
        "may_change_iteration_status": False,
        "may_change_transition": False,
        "may_change_delivery": False,
    }
    enriched["human_calibration"] = annotation
    if _decision_snapshot(enriched) != before:
        raise RuntimeError("human calibration attempted to mutate autonomous decision state")
    return enriched


def persist_human_calibration_annotation(
    report_path: str | Path,
    human_review: dict[str, Any] | None,
) -> dict[str, Any]:
    """Persist annotation-only listening evidence into an existing iteration report."""
    path = Path(report_path)
    if not path.is_file():
        raise FileNotFoundError(f"iteration report does not exist: {path}")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid iteration report JSON: {path}") from exc
    enriched = add_human_calibration_annotation(report, human_review)
    if enriched == report:
        return enriched
    encoded = json.dumps(enriched, indent=2, allow_nan=False)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(encoded, encoding="utf-8")
    tmp.replace(path)
    return enriched


def run_studio_iteration(
    baseline_path: str | Path,
    candidate_path: str | Path,
    output_dir: str | Path,
    *,
    baseline_id: str,
    candidate_id: str,
    plan: dict[str, Any],
    critic_result: dict[str, Any],
    evaluation_confidence: float,
    target_lufs: float = -14.5,
    ceiling_dbtp: float = -1.2,
    master_name: str = "Iteration_Master",
) -> dict[str, Any]:
    """Persist one bounded perceptual iteration and route it through real mastering.

    Machine-safe or uncertain audible candidates are mastered only as audition
    candidates. A rejected candidate is never mastered; the unchanged baseline is
    routed through the same mastering delivery path instead. Baseline promotion is
    intentionally impossible in this autonomous wrapper because explicit human
    listening acceptance is not an input.
    """
    baseline = Path(baseline_path).resolve()
    candidate = Path(candidate_path).resolve()
    out = Path(output_dir)
    if out.exists():
        raise FileExistsError(f"iteration output directory already exists: {out}")
    if not str(baseline_id).strip() or not str(candidate_id).strip():
        raise ValueError("baseline_id and candidate_id are required")
    if baseline == candidate:
        raise ValueError("candidate path must be distinct from baseline path")
    if not 0 <= float(evaluation_confidence) <= 1:
        raise ValueError("evaluation_confidence must be in [0, 1]")

    target = _validate_plan(plan, critic_result)
    baseline_file_sha = _file_digest(baseline)
    candidate_file_sha = _file_digest(candidate)
    baseline_audio, baseline_sr = _load_stereo(baseline)
    candidate_audio, candidate_sr = _load_stereo(candidate)
    if baseline_sr != candidate_sr:
        raise ValueError("baseline and candidate sample rates differ")
    if baseline_audio.shape != candidate_audio.shape:
        raise ValueError("baseline and candidate shapes differ")

    baseline_audio_sha = audio_digest(baseline_audio, baseline_sr)
    candidate_audio_sha = audio_digest(candidate_audio, candidate_sr)
    if baseline_audio_sha == candidate_audio_sha:
        raise ValueError("candidate audio is identical to baseline")

    transition = resolve_candidate_iteration(
        plan,
        {"id": str(candidate_id)},
        critic_result,
        str(baseline_id),
        float(evaluation_confidence),
    )
    if transition.get("promote_baseline") or transition.get("baseline_after") != str(baseline_id):
        raise RuntimeError("autonomous STUDIO iteration cannot promote a subjective baseline")

    rollback = bool(transition.get("rollback_candidate")) or transition.get("status") == "rejected"
    mastering_source = baseline if rollback else candidate
    delivery_role = "baseline_rollback" if rollback else "candidate_audition"

    out.mkdir(parents=True, exist_ok=False)
    mastering = deliver_master(
        mastering_source,
        out / "master",
        name=master_name,
        target_lufs=target_lufs,
        ceiling_dbtp=ceiling_dbtp,
    )

    if _file_digest(baseline) != baseline_file_sha:
        raise RuntimeError("baseline file changed during autonomous iteration")
    if _file_digest(candidate) != candidate_file_sha:
        raise RuntimeError("candidate file changed during autonomous iteration")

    final_status = str(transition["status"])
    if str(mastering.get("status", "")).startswith("rejected"):
        final_status = "rejected_mastering"

    report = {
        "schema": "studio-autonomous-iteration-v2",
        "status": final_status,
        "target": target,
        "baseline": {
            "id": str(baseline_id),
            "file_sha256": baseline_file_sha,
            "audio_sha256": baseline_audio_sha,
        },
        "candidate": {
            "id": str(candidate_id),
            "file_sha256": candidate_file_sha,
            "audio_sha256": candidate_audio_sha,
        },
        "sample_rate": baseline_sr,
        "frames": int(len(baseline_audio)),
        "channels": 2,
        "plan": plan,
        "critic": critic_result,
        "transition": transition,
        "delivery": {
            "role": delivery_role,
            "source_id": str(baseline_id if rollback else candidate_id),
            "source_file_sha256": baseline_file_sha if rollback else candidate_file_sha,
            "rolled_back_to_baseline": rollback,
            "candidate_audition_exported": not rollback and not str(mastering.get("status", "")).startswith("rejected"),
            "mastering": mastering,
        },
        "baseline_promoted": False,
        "baseline_after": str(baseline_id),
        "requires_human_listening": True,
        "human_review": "pending",
        "input_files_unchanged": True,
        "live_control_used": False,
    }
    (out / "iteration_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    return report
