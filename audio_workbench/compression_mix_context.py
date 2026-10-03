"""Full-mix context adapter for STUDIO compression candidates.

A source-level compression candidate is not treated as a mix improvement. This
adapter replaces exactly one already-routed stereo source contribution inside an
immutable offline mix, recomputes Perceptual Critic evidence on the whole mix and
only exports a level-matched audition pair when the existing autonomous gate says
human listening is the next action.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
from typing import Any

import numpy as np

from .compression_iteration import evaluate_compression_candidate
from .mixing.compression import as_audio
from .mixing.compression_director import audition_pair_plan, level_match_evidence
from .mixing.perceptual_critic import snapshot


def _stereo(x: np.ndarray, name: str) -> np.ndarray:
    audio = as_audio(x)
    if audio.ndim != 2 or audio.shape[1] != 2:
        raise ValueError(f"{name} must be samples x 2 stereo PCM")
    if not len(audio):
        raise ValueError(f"{name} must be non-empty")
    return audio


def _sha256(x: np.ndarray, sr: int) -> str:
    h = hashlib.sha256()
    h.update(b"audio-workbench-stereo-context-f32-v1\0")
    h.update(np.asarray([int(sr)], dtype="<i8").tobytes())
    h.update(np.asarray(x.shape, dtype="<i8").tobytes())
    h.update(np.ascontiguousarray(x, dtype="<f4").tobytes())
    return h.hexdigest()


def _gain_db(x: np.ndarray, gain_db: float) -> np.ndarray:
    return (x * np.float32(10 ** (float(gain_db) / 20))).astype(np.float32)


def _section_rms_db(x: np.ndarray, sr: int, window_s: float) -> list[float]:
    if not np.isfinite(window_s) or window_s <= 0:
        raise ValueError("section_window_s must be finite and positive")
    hop = max(1, int(round(sr * window_s)))
    out: list[float] = []
    for start in range(0, len(x), hop):
        q = x[start:start + hop].astype(np.float64)
        if not len(q):
            continue
        rms = float(np.sqrt(np.mean(q * q) + 1e-30))
        out.append(float(20 * np.log10(max(rms, 1e-15))))
    return out


def _same_shape(x: np.ndarray | None, reference: np.ndarray, name: str) -> np.ndarray | None:
    if x is None:
        return None
    audio = _stereo(x, name)
    if audio.shape != reference.shape:
        raise ValueError(f"{name} must match the full-mix shape")
    return audio


def _candidate_entry(prepared: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    entries = [c for c in prepared.get("candidates", []) if c.get("id") == candidate_id]
    if len(entries) != 1:
        raise ValueError(f"unknown or duplicate compression candidate: {candidate_id}")
    return entries[0]


def _prepared_with_context_failures(
    prepared: dict[str, Any], candidate_id: str, failures: list[str]
) -> dict[str, Any]:
    patched = deepcopy(prepared)
    entry = _candidate_entry(patched, candidate_id)
    combined = list(entry.get("objective_failures") or [])
    combined.extend(f"mix_context:{failure}" for failure in failures if f"mix_context:{failure}" not in combined)
    entry["objective_failures"] = combined
    entry["objective_gate_passed"] = not combined
    return patched


def evaluate_full_mix_compression_context(
    prepared: dict[str, Any],
    candidate_id: str,
    baseline_mix: np.ndarray,
    original_contribution: np.ndarray,
    candidate_contribution: np.ndarray,
    sr: int,
    *,
    source_group: str = "other",
    vocal_bus: np.ndarray | None = None,
    drums_bus: np.ndarray | None = None,
    early_room_before: np.ndarray | None = None,
    early_room_after: np.ndarray | None = None,
    evaluation_confidence: float = .9,
    baseline_id: str = "compression-baseline",
    audition_ceiling_dbtp: float = -3.0,
    source_match_tolerance_db: float = .05,
    section_window_s: float = 2.0,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Evaluate one routed compression candidate in the complete mix context.

    ``original_contribution`` and ``candidate_contribution`` must represent the same
    post-routing source contribution (same downstream EQ/pan/send convention). The
    candidate contribution is active-RMS matched to the original contribution before
    insertion, so the Perceptual Critic is not rewarded for a simple level change.

    ``source_group`` may be ``vocal``, ``drums`` or ``other``. When the replaced
    source belongs to a protected vocal/drum anchor, the corresponding bus is updated
    by the exact same replacement before snapshots are computed.
    """
    if prepared.get("schema") != "studio-compression-iteration-bridge-v1":
        raise ValueError("unsupported prepared compression iteration schema")
    _candidate_entry(prepared, candidate_id)
    if isinstance(sr, bool) or not isinstance(sr, (int, np.integer)) or sr < 1000:
        raise ValueError("sample rate must be an integer >= 1000 Hz")
    if source_group not in {"vocal", "drums", "other"}:
        raise ValueError("source_group must be vocal, drums or other")
    if not 0 <= float(evaluation_confidence) <= 1:
        raise ValueError("evaluation_confidence must be in [0, 1]")
    if not np.isfinite(audition_ceiling_dbtp):
        raise ValueError("audition_ceiling_dbtp must be finite")
    if not np.isfinite(source_match_tolerance_db) or source_match_tolerance_db < 0:
        raise ValueError("source_match_tolerance_db must be finite and non-negative")

    mix = _stereo(baseline_mix, "baseline_mix")
    original = _stereo(original_contribution, "original_contribution")
    candidate = _stereo(candidate_contribution, "candidate_contribution")
    if original.shape != mix.shape or candidate.shape != mix.shape:
        raise ValueError("source contributions must match the full-mix shape")
    prepared_source = dict(prepared.get("source") or {})
    if prepared_source.get("sample_rate") not in {None, int(sr)}:
        raise ValueError("prepared candidate sample rate differs from mix context")
    if prepared_source.get("frames") not in {None, int(len(mix))}:
        raise ValueError("prepared candidate frame count differs from mix context")

    vocal = _same_shape(vocal_bus, mix, "vocal_bus")
    drums = _same_shape(drums_bus, mix, "drums_bus")
    room_before = _same_shape(early_room_before, mix, "early_room_before")
    room_after = _same_shape(early_room_after, mix, "early_room_after")
    if room_after is None:
        room_after = room_before
    if source_group == "vocal" and vocal is None:
        raise ValueError("vocal source replacement requires vocal_bus")
    if source_group == "drums" and drums is None:
        raise ValueError("drum source replacement requires drums_bus")

    source_match = level_match_evidence(
        original, candidate, sr, ceiling_dbtp=None,
        tolerance_db=float(source_match_tolerance_db),
    )
    context_failures: list[str] = []
    if source_match.get("status") != "measured" or source_match.get("match_passed") is not True:
        context_failures.append("source_level_match_unavailable")
        matched_candidate = candidate.copy()
    else:
        matched_candidate = _gain_db(candidate, float(source_match["required_gain_db"]))
        measured = level_match_evidence(
            original, matched_candidate, sr, ceiling_dbtp=None,
            tolerance_db=float(source_match_tolerance_db),
        )
        source_match = {**source_match, "post_match": measured}
        if measured.get("match_passed") is not True:
            context_failures.append("source_level_match_failed_after_gain")

    # Float64 subtraction/summation keeps the replacement algebra auditable while
    # the stored context remains float32 audio like the rest of Workbench.
    rest64 = mix.astype(np.float64) - original.astype(np.float64)
    candidate_mix64 = rest64 + matched_candidate.astype(np.float64)
    rest = rest64.astype(np.float32)
    candidate_mix = candidate_mix64.astype(np.float32)
    baseline_reconstruction = rest64 + original.astype(np.float64)
    candidate_rest = candidate_mix64 - matched_candidate.astype(np.float64)
    baseline_error = float(np.max(np.abs(baseline_reconstruction - mix.astype(np.float64))))
    rest_error = float(np.max(np.abs(candidate_rest - rest64)))
    if baseline_error > 1e-7 or rest_error > 1e-7:
        context_failures.append("source_replacement_algebra_failed")
    max_mix_change = float(np.max(np.abs(candidate_mix.astype(np.float64) - mix.astype(np.float64))))
    if _sha256(candidate_mix, sr) == _sha256(mix, sr) or max_mix_change <= 1e-7:
        context_failures.append("candidate_mix_identical_to_baseline")

    vocal_after = vocal
    drums_after = drums
    if source_group == "vocal":
        vocal_after = (vocal.astype(np.float64) - original.astype(np.float64)
                       + matched_candidate.astype(np.float64)).astype(np.float32)
    elif source_group == "drums":
        drums_after = (drums.astype(np.float64) - original.astype(np.float64)
                       + matched_candidate.astype(np.float64)).astype(np.float32)

    before_snapshot = snapshot(
        mix, sr, vocal=vocal, drums=drums, early_room=room_before,
        section_rms_db=_section_rms_db(mix, sr, section_window_s),
    )
    after_snapshot = snapshot(
        candidate_mix, sr, vocal=vocal_after, drums=drums_after, early_room=room_after,
        section_rms_db=_section_rms_db(candidate_mix, sr, section_window_s),
    )

    mix_pair = audition_pair_plan(
        mix, candidate_mix, sr, ceiling_dbtp=float(audition_ceiling_dbtp)
    )
    if mix_pair.get("status") != "measured" or mix_pair.get("match_passed") is not True:
        context_failures.append("full_mix_audition_level_match_unavailable")
    ref_tp = mix_pair.get("reference_true_peak_after_dbtp")
    cand_tp = mix_pair.get("candidate_true_peak_after_dbtp")
    if ref_tp is None or cand_tp is None:
        context_failures.append("full_mix_audition_true_peak_unmeasured")
    elif (float(ref_tp) > audition_ceiling_dbtp + .01 or
          float(cand_tp) > audition_ceiling_dbtp + .01):
        context_failures.append("full_mix_audition_true_peak_ceiling_exceeded")

    patched = _prepared_with_context_failures(prepared, candidate_id, context_failures)
    evaluation = evaluate_compression_candidate(
        patched, candidate_id, before_snapshot, after_snapshot,
        evaluation_confidence=float(evaluation_confidence), baseline_id=str(baseline_id),
    )
    transition = dict(evaluation["transition"])
    if transition.get("promote_baseline") or transition.get("baseline_after") != str(baseline_id):
        raise RuntimeError("full-mix context adapter cannot promote a baseline without listening")

    audition_allowed = transition.get("next_action") == "human_listening"
    audition: dict[str, np.ndarray] = {}
    if audition_allowed:
        audition = {
            "reference": _gain_db(mix, float(mix_pair["reference_gain_db"])),
            "candidate": _gain_db(candidate_mix, float(mix_pair["candidate_total_gain_db"])),
        }

    report = {
        "schema": "studio-full-mix-compression-context-v1",
        "status": "pending_human_review" if audition_allowed else transition.get("status"),
        "candidate_id": str(candidate_id),
        "source_group": source_group,
        "source_match": source_match,
        "replacement": {
            "baseline_mix_sha256": _sha256(mix, sr),
            "rest_mix_sha256": _sha256(rest, sr),
            "original_contribution_sha256": _sha256(original, sr),
            "matched_candidate_contribution_sha256": _sha256(matched_candidate, sr),
            "candidate_mix_sha256": _sha256(candidate_mix, sr),
            "baseline_reconstruction_max_error": baseline_error,
            "candidate_rest_max_error": rest_error,
            "max_mix_change": max_mix_change,
            "other_mix_contributions_reused": True,
        },
        "before_snapshot": asdict(before_snapshot),
        "after_snapshot": asdict(after_snapshot),
        "context_failures": context_failures,
        "context_objective_gate_passed": not context_failures,
        "evaluation": evaluation,
        "audition_pair": mix_pair,
        "audition_export_allowed": bool(audition_allowed),
        "no_change_counterfactual_sha256": _sha256(mix, sr),
        "baseline_promoted": False,
        "baseline_after": str(baseline_id),
        "requires_human_review": True,
        "requires_human_listening": True,
        "human_review": "pending" if audition_allowed else "not_reached",
        "scope": (
            "whole-mix PerceptualSnapshot after one exact routed-source replacement; "
            "no mastering, no winner ranking, no autonomous subjective acceptance"
        ),
    }
    return report, audition
