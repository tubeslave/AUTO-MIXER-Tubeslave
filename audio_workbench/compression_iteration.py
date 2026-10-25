"""STUDIO-only bridge for Compression Director v1.1 -> Perceptual Critic -> Autonomous Iteration.

The bridge creates a bounded family of three compressor auditions plus the mandatory
no-change counterfactual. Engineering gates may reject unsafe/invalid renders, but
no compression candidate can become a musical baseline without human listening.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
from typing import Any

import numpy as np

from . import causal
from .autonomous_loop import resolve_candidate_iteration
from .mixing.compression import as_audio
from .mixing.compression_director import audition_pair_plan
from .mixing.compression_director_v11 import (
    propose_calibrated_candidates,
    render_calibrated_core_candidate,
)
from .mixing.perceptual_critic import PerceptualSnapshot, accept_candidate
from .mastering.analyzer import true_peak_dbtp


VARIANT_IDS = ("preserve_transient", "balanced", "control")
PERCEPTUAL_TARGETS = {"vocal_intelligibility", "harshness", "punch", "climax"}


def _audio_sha256(x: np.ndarray, sr: int) -> str:
    """Stable digest of validated float32 PCM, sample rate, shape and channel layout."""
    audio = as_audio(x)
    h = hashlib.sha256()
    h.update(b"audio-workbench-pcm-f32-v1\0")
    h.update(np.asarray([int(sr)], dtype="<i8").tobytes())
    h.update(np.asarray(audio.shape, dtype="<i8").tobytes())
    h.update(np.ascontiguousarray(audio, dtype="<f4").tobytes())
    return h.hexdigest()


def _candidate_plan_entry(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "compressor",
        "id": str(candidate["id"]),
        "params": dict(candidate["compressor"]),
        "target_active_p95_gr_db": float(candidate["target_active_p95_gr_db"]),
        "requires_human_review": True,
        "requires_human_listening": True,
        "baseline_eligible": False,
    }


def prepare_compression_iteration(
    source: np.ndarray,
    sr: int,
    role: str,
    target: str,
    *,
    calibration_tolerance_db: float = .08,
    audition_ceiling_dbtp: float = -3.0,
    cause_confidence: float = .85,
    intervention_confidence: float = .85,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Render one immutable source into exactly three bounded v1.1 candidates.

    Returns JSON-safe evidence plus an in-memory render map. The causal plan contains
    a fourth entry, ``bypass/no_change``, but that counterfactual is not counted as a
    compression variant. No winner is ranked or selected here.
    """
    x = as_audio(source)
    if target not in PERCEPTUAL_TARGETS:
        raise ValueError(f"unsupported perceptual target: {target}")
    if not len(x):
        raise ValueError("compression iteration requires non-empty source audio")
    if not np.isfinite(calibration_tolerance_db) or calibration_tolerance_db <= 0:
        raise ValueError("calibration_tolerance_db must be finite and positive")
    if not np.isfinite(audition_ceiling_dbtp):
        raise ValueError("audition_ceiling_dbtp must be finite")
    for name, value in (("cause_confidence", cause_confidence),
                        ("intervention_confidence", intervention_confidence)):
        if not 0 <= float(value) <= 1:
            raise ValueError(f"{name} must be in [0, 1]")

    source_sha = _audio_sha256(x, sr)
    proposal = propose_calibrated_candidates(
        x, sr, role, tolerance_db=float(calibration_tolerance_db)
    )
    candidates = list(proposal.get("candidates") or [])
    ids = tuple(str(c.get("id")) for c in candidates)
    if ids != VARIANT_IDS:
        raise RuntimeError(f"unexpected compression candidate family: {ids!r}")

    plan = causal.make_plan(
        observation=f"Compression Director v1.1 evidence for {role}",
        hypothesis=(
            "one bounded compression timing/density variant may improve the declared "
            "perceptual target without a protected regression"
        ),
        target=target,
        interventions=[_candidate_plan_entry(c) for c in candidates],
        expected_effect=f"improve {target} while preserving protected mix metrics",
        protected_metrics=[
            "density", "width_db", "foreground_db", "harshness",
            "vocal_intelligibility", "punch_db", "climax_lift_db",
        ],
        confidence={"cause": float(cause_confidence),
                    "intervention": float(intervention_confidence)},
    )
    plan["requires_human_review"] = True
    plan["requires_human_listening"] = True
    plan["baseline_eligible"] = False
    plan["family"] = "compression-director-v1.1"

    renders: dict[str, np.ndarray] = {}
    evidence: list[dict[str, Any]] = []
    timing = dict(proposal.get("timing_censoring") or {})
    for candidate in candidates:
        cid = str(candidate["id"])
        y, render = render_calibrated_core_candidate(x, sr, candidate)
        render = dict(render)
        objective_failures: list[str] = []
        if y.shape != x.shape or not np.isfinite(y).all():
            objective_failures.append("invalid_render_shape_or_pcm")
        candidate_sha = _audio_sha256(y, sr)
        if candidate_sha == source_sha:
            objective_failures.append("candidate_identical_to_source")
        calibration = dict(candidate.get("calibration") or {})
        cal_error = calibration.get("error_db")
        if cal_error is None or not np.isfinite(float(cal_error)):
            objective_failures.append("calibration_unmeasured")
        elif abs(float(cal_error)) > float(calibration_tolerance_db) + 1e-9:
            objective_failures.append("active_p95_gr_target_missed")
        max_allowed = float(candidate["compressor"]["max_gr_db"])
        if float(render["max_gr_db"]) > max_allowed + .01:
            objective_failures.append("max_gr_budget_exceeded")

        pair = audition_pair_plan(x, y, sr, ceiling_dbtp=float(audition_ceiling_dbtp))
        if pair.get("status") != "measured" or pair.get("match_passed") is not True:
            objective_failures.append("audition_level_match_unavailable")
        audition_candidate_tp = pair.get("candidate_true_peak_after_dbtp")
        audition_reference_tp = pair.get("reference_true_peak_after_dbtp")
        if audition_candidate_tp is None or audition_reference_tp is None:
            objective_failures.append("audition_true_peak_unmeasured")
        elif (float(audition_candidate_tp) > audition_ceiling_dbtp + .01 or
              float(audition_reference_tp) > audition_ceiling_dbtp + .01):
            objective_failures.append("audition_true_peak_ceiling_exceeded")

        evidence.append({
            "id": cid,
            "source_audio_sha256": source_sha,
            "candidate_audio_sha256": candidate_sha,
            "sample_rate": int(sr),
            "frames": int(len(x)),
            "channels": int(x.shape[1]) if x.ndim == 2 else 1,
            "compressor": dict(candidate["compressor"]),
            "calibration": calibration,
            "render": render,
            "timing_censoring": timing,
            "candidate_true_peak_dbtp": float(true_peak_dbtp(y)),
            "audition_pair": pair,
            "objective_failures": objective_failures,
            "objective_gate_passed": not objective_failures,
            "requires_human_review": True,
            "requires_human_listening": True,
            "baseline_eligible": False,
        })
        renders[cid] = y

    if _audio_sha256(x, sr) != source_sha:
        raise RuntimeError("source PCM changed while preparing compression candidates")
    bypass = [c for c in plan["candidates"] if c.get("type") == "bypass"]
    if len(bypass) != 1 or bypass[0].get("label") != "no_change":
        raise RuntimeError("mandatory no-change counterfactual missing from causal plan")

    report = {
        "schema": "studio-compression-iteration-bridge-v1",
        "status": "candidates_prepared",
        "source": {
            "audio_sha256": source_sha,
            "sample_rate": int(sr),
            "frames": int(len(x)),
            "channels": int(x.shape[1]) if x.ndim == 2 else 1,
        },
        "role": role,
        "target": target,
        "director_schema": proposal.get("schema"),
        "director_analysis": proposal.get("analysis"),
        "timing_censoring": timing,
        "plan": plan,
        "candidate_order": list(VARIANT_IDS),
        "candidates": evidence,
        "selection_policy": (
            "objective engineering gates may reject; machine-safe candidates remain "
            "pending human listening; this bridge does not rank or promote a winner"
        ),
        "requires_human_review": True,
        "requires_human_listening": True,
        "baseline_eligible": False,
        "source_unchanged": True,
    }
    return report, renders


def evaluate_compression_candidate(
    prepared: dict[str, Any],
    candidate_id: str,
    before: PerceptualSnapshot,
    after: PerceptualSnapshot,
    *,
    evaluation_confidence: float,
    baseline_id: str = "compression-baseline",
) -> dict[str, Any]:
    """Run one prepared candidate through the real Perceptual Critic and autonomous gate.

    Objective rendering/calibration failure is a veto and forces rollback even if
    the perceptual proxy happens to improve. Passing machine evidence is still only
    an audition candidate because no human-review acceptance is accepted here.
    """
    if prepared.get("schema") != "studio-compression-iteration-bridge-v1":
        raise ValueError("unsupported prepared compression iteration schema")
    if not 0 <= float(evaluation_confidence) <= 1:
        raise ValueError("evaluation_confidence must be in [0, 1]")
    entries = [c for c in prepared.get("candidates", []) if c.get("id") == candidate_id]
    if len(entries) != 1:
        raise ValueError(f"unknown or duplicate compression candidate: {candidate_id}")
    evidence = entries[0]
    critic = dict(accept_candidate(before, after, str(prepared["target"])))
    objective_failures = list(evidence.get("objective_failures") or [])
    if objective_failures:
        critic["machine_decision"] = "rejected"
        critic["accept"] = False
        critic["failures"] = list(critic.get("failures") or []) + [
            f"compression:{failure}" for failure in objective_failures
        ]
        critic["protected_regressions"] = list(critic.get("protected_regressions") or []) + [
            f"compression:{failure}" for failure in objective_failures
        ]
        critic["evidence"] = list(critic.get("evidence") or []) + [{
            "metric": "compression_objective_gate",
            "role": "protected",
            "before": 1.0,
            "after": 0.0,
            "delta": -1.0,
            "passed": False,
            "failures": objective_failures,
        }]
    candidate = {
        "id": str(candidate_id),
        "requires_human_review": True,
        "requires_human_listening": True,
        "baseline_eligible": False,
        "compression_evidence": {
            "source_audio_sha256": evidence["source_audio_sha256"],
            "candidate_audio_sha256": evidence["candidate_audio_sha256"],
            "objective_gate_passed": bool(evidence["objective_gate_passed"]),
        },
    }
    transition = resolve_candidate_iteration(
        dict(prepared["plan"]), candidate, critic, str(baseline_id),
        float(evaluation_confidence),
    )
    if transition.get("promote_baseline") or transition.get("baseline_after") != str(baseline_id):
        raise RuntimeError("compression bridge cannot promote a baseline without human listening")
    return {
        "schema": "studio-compression-iteration-evaluation-v1",
        "candidate_id": str(candidate_id),
        "objective_gate_passed": bool(evidence["objective_gate_passed"]),
        "objective_failures": objective_failures,
        "critic": critic,
        "transition": transition,
        "baseline_promoted": False,
        "baseline_after": str(baseline_id),
        "requires_human_listening": True,
        "human_review": "pending",
    }
