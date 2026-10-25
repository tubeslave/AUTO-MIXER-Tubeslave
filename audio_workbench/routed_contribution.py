"""Authoritative full-session rerender boundary for STUDIO compression hypotheses.

Unlike ``compression_mix_context``, this module never estimates a routed source by
subtracting a dry contribution from a premaster. A caller supplies the real offline
session renderer. Baseline, no-change, candidate and source-muted counterfactuals
are rendered through that same graph, so sends/returns, source-dependent automation
and shared nonlinear buses are included in the evaluated mix.

The source-muted delta is diagnostic only. With nonlinear shared processing it is a
context-dependent marginal, not an additive stem and must never be summed as one.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
from typing import Any, Callable, Mapping

import numpy as np

from .compression_iteration import evaluate_compression_candidate
from .mixing.compression import as_audio
from .mixing.compression_director import audition_pair_plan, level_match_evidence
from .mixing.perceptual_critic import snapshot


@dataclass
class SessionRender:
    """One complete deterministic offline render of the routed session graph."""

    mix: np.ndarray
    vocal_bus: np.ndarray | None = None
    drums_bus: np.ndarray | None = None
    early_room: np.ndarray | None = None
    metadata: dict[str, Any] | None = None


SessionRenderer = Callable[[Mapping[str, np.ndarray]], SessionRender]


def _sha256(x: np.ndarray, sr: int, namespace: bytes = b"routed-session-f32-v1\0") -> str:
    h = hashlib.sha256()
    h.update(namespace)
    h.update(np.asarray([int(sr)], dtype="<i8").tobytes())
    h.update(np.asarray(x.shape, dtype="<i8").tobytes())
    h.update(np.ascontiguousarray(x, dtype="<f4").tobytes())
    return h.hexdigest()


def _stereo(x: np.ndarray, name: str) -> np.ndarray:
    audio = as_audio(x)
    if audio.ndim != 2 or audio.shape[1] != 2 or not len(audio):
        raise ValueError(f"{name} must be non-empty samples x 2 stereo PCM")
    return audio


def _same_shape_optional(x: np.ndarray | None, mix: np.ndarray, name: str) -> np.ndarray | None:
    if x is None:
        return None
    audio = _stereo(x, name)
    if audio.shape != mix.shape:
        raise ValueError(f"{name} must match rendered mix shape")
    return audio


def _checked_render(renderer: SessionRenderer, overrides: Mapping[str, np.ndarray], name: str) -> SessionRender:
    rendered = renderer(overrides)
    if not isinstance(rendered, SessionRender):
        raise TypeError("session renderer must return SessionRender")
    mix = _stereo(rendered.mix, f"{name}.mix")
    vocal = _same_shape_optional(rendered.vocal_bus, mix, f"{name}.vocal_bus")
    drums = _same_shape_optional(rendered.drums_bus, mix, f"{name}.drums_bus")
    room = _same_shape_optional(rendered.early_room, mix, f"{name}.early_room")
    return SessionRender(mix, vocal, drums, room, dict(rendered.metadata or {}))


def _gain_db(x: np.ndarray, gain_db: float) -> np.ndarray:
    gain = np.float32(10 ** (float(gain_db) / 20))
    return (x * gain).astype(np.float32, copy=False)


def _section_rms_db(x: np.ndarray, sr: int, window_s: float) -> list[float]:
    if not np.isfinite(window_s) or window_s <= 0:
        raise ValueError("section_window_s must be finite and positive")
    hop = max(1, int(round(sr * window_s)))
    result: list[float] = []
    for start in range(0, len(x), hop):
        q = x[start:start + hop].astype(np.float64)
        rms = float(np.sqrt(np.mean(q * q) + 1e-30))
        result.append(float(20 * np.log10(max(rms, 1e-15))))
    return result


def _candidate_entry(prepared: dict[str, Any], candidate_id: str) -> dict[str, Any]:
    entries = [c for c in prepared.get("candidates", []) if c.get("id") == candidate_id]
    if len(entries) != 1:
        raise ValueError(f"unknown or duplicate compression candidate: {candidate_id}")
    return entries[0]


def _prepared_with_failures(prepared: dict[str, Any], candidate_id: str,
                            failures: list[str]) -> dict[str, Any]:
    patched = deepcopy(prepared)
    entry = _candidate_entry(patched, candidate_id)
    combined = list(entry.get("objective_failures") or [])
    for failure in failures:
        label = f"routed_context:{failure}"
        if label not in combined:
            combined.append(label)
    entry["objective_failures"] = combined
    entry["objective_gate_passed"] = not combined
    return patched


def _max_error(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return float("inf")
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


def evaluate_routed_compression_context(
    prepared: dict[str, Any],
    candidate_id: str,
    source_id: str,
    original_source: np.ndarray,
    candidate_source: np.ndarray,
    sr: int,
    render_session: SessionRenderer,
    *,
    source_group: str = "other",
    evaluation_confidence: float = .9,
    baseline_id: str = "compression-baseline",
    audition_ceiling_dbtp: float = -3.0,
    source_match_tolerance_db: float = .05,
    rerender_tolerance: float = 1e-7,
    section_window_s: float = 2.0,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Evaluate one compression candidate by authoritative session rerender.

    ``render_session`` receives a mapping of source overrides and must execute the
    actual deterministic offline recipe. The function is called four times: exact
    baseline, exact no-change repetition, candidate, and source-muted counterfactual.
    No fixed routed contribution is inferred by subtraction for candidate creation.
    """
    if prepared.get("schema") != "studio-compression-iteration-bridge-v1":
        raise ValueError("unsupported prepared compression iteration schema")
    _candidate_entry(prepared, candidate_id)
    if not isinstance(source_id, str) or not source_id.strip():
        raise ValueError("source_id must be a non-empty string")
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
    if not np.isfinite(rerender_tolerance) or rerender_tolerance < 0:
        raise ValueError("rerender_tolerance must be finite and non-negative")

    original = as_audio(original_source)
    candidate = as_audio(candidate_source)
    if not len(original) or candidate.shape != original.shape:
        raise ValueError("candidate_source must match non-empty original_source shape")
    source_before = _sha256(original, sr, b"routed-source-f32-v1\0")
    candidate_before = _sha256(candidate, sr, b"routed-source-f32-v1\0")
    prepared_source = dict(prepared.get("source") or {})
    expected_channels = int(original.shape[1]) if original.ndim == 2 else 1
    if prepared_source.get("sample_rate") not in {None, int(sr)}:
        raise ValueError("prepared candidate sample rate differs from routed session")
    if prepared_source.get("frames") not in {None, int(len(original))}:
        raise ValueError("prepared candidate frame count differs from source")
    if prepared_source.get("channels") not in {None, expected_channels}:
        raise ValueError("prepared candidate channel count differs from source")

    source_match = level_match_evidence(
        original, candidate, sr, ceiling_dbtp=None,
        tolerance_db=float(source_match_tolerance_db),
    )
    failures: list[str] = []
    if source_match.get("status") != "measured" or source_match.get("match_passed") is not True:
        failures.append("source_level_match_unavailable")
        matched_candidate = candidate.copy()
    else:
        matched_candidate = _gain_db(candidate, float(source_match["required_gain_db"]))
        post_match = level_match_evidence(
            original, matched_candidate, sr, ceiling_dbtp=None,
            tolerance_db=float(source_match_tolerance_db),
        )
        source_match = {**source_match, "post_match": post_match}
        if post_match.get("match_passed") is not True:
            failures.append("source_level_match_failed_after_gain")

    baseline = _checked_render(render_session, {source_id: original.copy()}, "baseline")
    no_change = _checked_render(render_session, {source_id: original.copy()}, "no_change")
    candidate_render = _checked_render(
        render_session, {source_id: matched_candidate.copy()}, "candidate"
    )
    muted = _checked_render(
        render_session, {source_id: np.zeros_like(original)}, "source_muted"
    )
    if baseline.mix.shape != no_change.mix.shape or baseline.mix.shape != candidate_render.mix.shape:
        raise ValueError("all full-session renders must have the same mix shape")
    if baseline.mix.shape != muted.mix.shape:
        raise ValueError("source-muted render must match full-session mix shape")

    no_change_error = _max_error(baseline.mix, no_change.mix)
    no_change_hash_equal = _sha256(baseline.mix, sr) == _sha256(no_change.mix, sr)
    if not no_change_hash_equal or no_change_error > float(rerender_tolerance):
        failures.append("session_rerender_not_deterministic")
    mix_change = _max_error(baseline.mix, candidate_render.mix)
    if _sha256(baseline.mix, sr) == _sha256(candidate_render.mix, sr) or mix_change <= float(rerender_tolerance):
        failures.append("candidate_mix_identical_to_baseline")

    if source_group == "vocal" and (baseline.vocal_bus is None or candidate_render.vocal_bus is None):
        failures.append("vocal_anchor_bus_unavailable")
    if source_group == "drums" and (baseline.drums_bus is None or candidate_render.drums_bus is None):
        failures.append("drums_anchor_bus_unavailable")
    if (baseline.early_room is None) != (candidate_render.early_room is None):
        failures.append("early_room_presence_changed")

    before = snapshot(
        baseline.mix, sr,
        vocal=baseline.vocal_bus, drums=baseline.drums_bus,
        early_room=baseline.early_room,
        section_rms_db=_section_rms_db(baseline.mix, sr, section_window_s),
    )
    after = snapshot(
        candidate_render.mix, sr,
        vocal=candidate_render.vocal_bus, drums=candidate_render.drums_bus,
        early_room=candidate_render.early_room,
        section_rms_db=_section_rms_db(candidate_render.mix, sr, section_window_s),
    )
    audition_plan = audition_pair_plan(
        baseline.mix, candidate_render.mix, sr,
        ceiling_dbtp=float(audition_ceiling_dbtp),
    )
    if audition_plan.get("status") != "measured" or audition_plan.get("match_passed") is not True:
        failures.append("full_mix_audition_level_match_unavailable")
    for key in ("reference_true_peak_after_dbtp", "candidate_true_peak_after_dbtp"):
        value = audition_plan.get(key)
        if value is None:
            failures.append("full_mix_audition_true_peak_unmeasured")
            break
        if float(value) > float(audition_ceiling_dbtp) + .01:
            failures.append("full_mix_audition_true_peak_ceiling_exceeded")
            break

    patched = _prepared_with_failures(prepared, candidate_id, failures)
    evaluation = evaluate_compression_candidate(
        patched, candidate_id, before, after,
        evaluation_confidence=float(evaluation_confidence), baseline_id=str(baseline_id),
    )
    transition = dict(evaluation["transition"])
    if transition.get("promote_baseline") or transition.get("baseline_after") != str(baseline_id):
        raise RuntimeError("routed context renderer cannot promote a baseline without listening")

    audition_allowed = transition.get("next_action") == "human_listening" and not failures
    audition: dict[str, np.ndarray] = {}
    if audition_allowed:
        audition = {
            "reference": _gain_db(baseline.mix, float(audition_plan["reference_gain_db"])),
            "candidate": _gain_db(candidate_render.mix, float(audition_plan["candidate_total_gain_db"])),
        }

    if _sha256(original, sr, b"routed-source-f32-v1\0") != source_before:
        raise RuntimeError("original source changed during routed evaluation")
    if _sha256(candidate, sr, b"routed-source-f32-v1\0") != candidate_before:
        raise RuntimeError("candidate source changed during routed evaluation")

    baseline_marginal = baseline.mix.astype(np.float64) - muted.mix.astype(np.float64)
    candidate_marginal = candidate_render.mix.astype(np.float64) - muted.mix.astype(np.float64)
    report = {
        "schema": "studio-routed-compression-context-v1",
        "status": "pending_human_review" if audition_allowed else transition.get("status"),
        "candidate_id": str(candidate_id),
        "source_id": source_id,
        "source_group": source_group,
        "routing_authority": "full_session_rerender",
        "direct_source_subtraction_used_for_candidate_mix": False,
        "renderer_invocations": 4,
        "source_match": source_match,
        "rerender_identity": {
            "baseline_sha256": _sha256(baseline.mix, sr),
            "no_change_sha256": _sha256(no_change.mix, sr),
            "hash_equal": bool(no_change_hash_equal),
            "max_abs_error": no_change_error,
            "tolerance": float(rerender_tolerance),
        },
        "renders": {
            "candidate_mix_sha256": _sha256(candidate_render.mix, sr),
            "source_muted_mix_sha256": _sha256(muted.mix, sr),
            "max_mix_change": mix_change,
            "baseline_metadata": baseline.metadata,
            "candidate_metadata": candidate_render.metadata,
        },
        "counterfactual_marginal": {
            "baseline_minus_source_muted_max_abs": float(np.max(np.abs(baseline_marginal))),
            "candidate_minus_source_muted_max_abs": float(np.max(np.abs(candidate_marginal))),
            "meaning": (
                "context-dependent render-minus-muted diagnostic; not an additive stem "
                "when shared nonlinear buses or source-dependent routing are present"
            ),
        },
        "before_snapshot": asdict(before),
        "after_snapshot": asdict(after),
        "context_failures": failures,
        "context_objective_gate_passed": not failures,
        "evaluation": evaluation,
        "audition_pair": audition_plan,
        "audition_export_allowed": bool(audition_allowed),
        "baseline_promoted": False,
        "baseline_after": str(baseline_id),
        "requires_human_review": True,
        "requires_human_listening": True,
        "human_review": "pending" if audition_allowed else "not_reached",
        "scope": (
            "full offline session graph rerender with one source override; exact relative "
            "to supplied deterministic renderer; no mastering and no autonomous winner"
        ),
    }
    return report, audition
