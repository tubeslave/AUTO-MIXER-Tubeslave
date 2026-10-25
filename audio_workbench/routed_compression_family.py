"""STUDIO-only coordinator for a complete routed Compression Director family.

The coordinator does not rank or choose a musical winner. It evaluates the exact
three Compression Director v1.1 candidates through the authoritative session
renderer, records objective rejection/survival, and exposes level-matched audition
pairs only for candidates that reach the existing human-listening gate.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .compression_iteration import VARIANT_IDS
from .mixing.compression import as_audio
from .routed_contribution import SessionRenderer, evaluate_routed_compression_context


def evaluate_routed_compression_family(
    prepared: dict[str, Any],
    renders: Mapping[str, np.ndarray],
    source_id: str,
    original_source: np.ndarray,
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
) -> tuple[dict[str, Any], dict[str, dict[str, np.ndarray]]]:
    """Evaluate all three bounded candidates; never auto-select a winner.

    ``prepared`` and ``renders`` must come from ``prepare_compression_iteration``
    for the same source. Every candidate is passed through
    ``evaluate_routed_compression_context`` and therefore receives the same full
    rerender, objective-veto, Perceptual Critic and Autonomous Iteration gates.

    Returned auditions are keyed by candidate id and contain only candidates whose
    transition reached the human-listening gate. Their presence is not acceptance.
    """
    if prepared.get("schema") != "studio-compression-iteration-bridge-v1":
        raise ValueError("unsupported prepared compression iteration schema")
    order = tuple(str(x) for x in prepared.get("candidate_order") or ())
    if order != VARIANT_IDS:
        raise ValueError(f"unexpected compression candidate family: {order!r}")
    render_ids = tuple(str(x) for x in renders.keys())
    if set(render_ids) != set(VARIANT_IDS) or len(render_ids) != len(VARIANT_IDS):
        raise ValueError("renders must contain exactly the three Compression Director variants")
    original = as_audio(original_source)
    if not len(original):
        raise ValueError("original_source must be non-empty")

    reports: dict[str, dict[str, Any]] = {}
    auditions: dict[str, dict[str, np.ndarray]] = {}
    survivors: list[str] = []
    rejected: list[str] = []
    for candidate_id in VARIANT_IDS:
        candidate = as_audio(renders[candidate_id])
        if candidate.shape != original.shape:
            raise ValueError(f"candidate {candidate_id} shape differs from original source")
        report, pair = evaluate_routed_compression_context(
            prepared, candidate_id, source_id, original, candidate, sr, render_session,
            source_group=source_group,
            evaluation_confidence=float(evaluation_confidence),
            baseline_id=str(baseline_id),
            audition_ceiling_dbtp=float(audition_ceiling_dbtp),
            source_match_tolerance_db=float(source_match_tolerance_db),
            rerender_tolerance=float(rerender_tolerance),
            section_window_s=float(section_window_s),
        )
        reports[candidate_id] = report
        if report.get("baseline_promoted") is not False:
            raise RuntimeError("routed family evaluation cannot promote a baseline")
        transition = dict(report.get("evaluation", {}).get("transition") or {})
        if report.get("audition_export_allowed") is True:
            if transition.get("next_action") != "human_listening" or set(pair) != {"reference", "candidate"}:
                raise RuntimeError("audition export is inconsistent with human-listening gate")
            survivors.append(candidate_id)
            auditions[candidate_id] = pair
        else:
            rejected.append(candidate_id)

    if set(survivors).intersection(rejected) or set(survivors + rejected) != set(VARIANT_IDS):
        raise RuntimeError("candidate accounting is incomplete")
    summary = {
        "schema": "studio-routed-compression-family-v1",
        "status": "pending_human_review" if survivors else "all_candidates_rejected",
        "source_id": str(source_id),
        "source_group": str(source_group),
        "candidate_order": list(VARIANT_IDS),
        "surviving_auditions": survivors,
        "machine_rejected": rejected,
        "reports": reports,
        "winner": None,
        "ranking": None,
        "selection_policy": (
            "machine gates may veto candidates; surviving candidates are level-matched auditions only; "
            "no winner or baseline promotion is allowed without human listening"
        ),
        "baseline_promoted": False,
        "baseline_after": str(baseline_id),
        "requires_human_review": True,
        "requires_human_listening": True,
        "human_review": "pending" if survivors else "not_reached",
    }
    return summary, auditions
