"""Snare top/bottom coherence guard for the Ayaic offline pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .models import ensure_stereo


SNARE_PAIR_COHERENCE_STAGE = "snare_pair_coherence"
SNARE_GROUP = "snare"


def apply_snare_pair_coherence(stems: list[Any], *, report_only: bool = False) -> dict[str, Any]:
    """Keep snare top and bottom as one centered, polarity-correct close-mic pair."""

    pair = _find_snare_pair(stems)
    if pair is None:
        return {
            "enabled": False,
            "applied": False,
            "stage": SNARE_PAIR_COHERENCE_STAGE,
            "reason": "snare_top_bottom_pair_not_found",
            "pairs": [],
        }

    top, bottom = pair
    top_before = _state(top)
    bottom_before = _state(bottom)
    corr_before = _corr(_mono(top.audio), _mono(bottom.audio))
    actions: list[str] = []

    if not report_only:
        if top.group != SNARE_GROUP:
            top.group = SNARE_GROUP
            actions.append("top_group_set")
        if bottom.group != SNARE_GROUP:
            bottom.group = SNARE_GROUP
            actions.append("bottom_group_set")
        if abs(float(getattr(top, "pan", 0.0))) > 1e-6:
            top.pan = 0.0
            actions.append("top_pan_centered")
        if abs(float(getattr(bottom, "pan", 0.0))) > 1e-6:
            bottom.pan = 0.0
            actions.append("bottom_pan_centered")
        if bool(top.phase_invert) == bool(bottom.phase_invert):
            bottom.audio = -ensure_stereo(bottom.audio)
            bottom.phase_invert = not bool(bottom.phase_invert)
            actions.append("bottom_phase_inverted_relative_to_top")
    else:
        if top.group != SNARE_GROUP:
            actions.append("top_group_set")
        if bottom.group != SNARE_GROUP:
            actions.append("bottom_group_set")
        if abs(float(getattr(top, "pan", 0.0))) > 1e-6:
            actions.append("top_pan_centered")
        if abs(float(getattr(bottom, "pan", 0.0))) > 1e-6:
            actions.append("bottom_pan_centered")
        if bool(top.phase_invert) == bool(bottom.phase_invert):
            actions.append("bottom_phase_inverted_relative_to_top")

    top_after = _state(top)
    bottom_after = _state(bottom)
    corr_after = _corr(_mono(top.audio), _mono(bottom.audio))
    polarity_ok = bool(top_after["phase_invert"]) != bool(bottom_after["phase_invert"])
    centered_ok = abs(float(top_after["pan"])) <= 1e-6 and abs(float(bottom_after["pan"])) <= 1e-6
    grouped_ok = top_after["group"] == SNARE_GROUP and bottom_after["group"] == SNARE_GROUP
    applied = bool(actions and not report_only)

    pair_report = {
        "stage": SNARE_PAIR_COHERENCE_STAGE,
        "top_channel": int(top.channel_id),
        "bottom_channel": int(bottom.channel_id),
        "top_file": Path(top.path).name,
        "bottom_file": Path(bottom.path).name,
        "top_before": top_before,
        "bottom_before": bottom_before,
        "top_after": top_after,
        "bottom_after": bottom_after,
        "correlation_before": round(corr_before, 4),
        "correlation_after": round(corr_after, 4),
        "polarity_relation": "opposite_phase_flags" if polarity_ok else "same_phase_flags",
        "group": SNARE_GROUP if grouped_ok else "invalid",
        "actions": actions,
        "applied": applied,
        "report_only": bool(report_only),
        "reason": _reason(actions, polarity_ok, centered_ok, grouped_ok),
    }
    _append_note(top, bottom, pair_report)
    return {
        "enabled": True,
        "applied": applied,
        "stage": SNARE_PAIR_COHERENCE_STAGE,
        "method": "snare_top_bottom_center_group_polarity_guard",
        "pairs": [pair_report],
        "critic": {
            "decision": "ok" if polarity_ok and centered_ok and grouped_ok else "review",
            "polarity_ok": polarity_ok,
            "centered_ok": centered_ok,
            "grouped_ok": grouped_ok,
        },
    }


def _find_snare_pair(stems: list[Any]) -> tuple[Any, Any] | None:
    top = None
    bottom = None
    for stem in stems:
        role = _snare_role(stem.name)
        if role == "top" and top is None:
            top = stem
        elif role == "bottom" and bottom is None:
            bottom = stem
    if top is None or bottom is None:
        return None
    return top, bottom


def _snare_role(name: str) -> str | None:
    key = " ".join(Path(str(name)).stem.lower().replace("_", " ").replace("-", " ").split())
    if "snare" not in key and not key.startswith("sn "):
        return None
    if any(token in key for token in ("bottom", "bot", "btm", " b", "sn b")):
        return "bottom"
    if any(token in key for token in ("top", " t", "sn t")):
        return "top"
    return None


def _state(stem: Any) -> dict[str, Any]:
    return {
        "group": str(getattr(stem, "group", "")),
        "pan": round(float(getattr(stem, "pan", 0.0)), 3),
        "phase_invert": bool(getattr(stem, "phase_invert", False)),
        "delay_ms": round(float(getattr(stem, "delay_ms", 0.0)), 4),
    }


def _mono(audio: np.ndarray) -> np.ndarray:
    arr = ensure_stereo(audio)
    return np.mean(arr, axis=1).astype(np.float64)


def _corr(left: np.ndarray, right: np.ndarray) -> float:
    n = min(left.size, right.size)
    if n == 0:
        return 0.0
    a = left[:n] - float(np.mean(left[:n]))
    b = right[:n] - float(np.mean(right[:n]))
    den = float(np.sqrt(np.mean(a * a)) * np.sqrt(np.mean(b * b)) + 1e-12)
    return float(np.mean(a * b) / den)


def _reason(actions: list[str], polarity_ok: bool, centered_ok: bool, grouped_ok: bool) -> str:
    if polarity_ok and centered_ok and grouped_ok and not actions:
        return "snare_pair_already_centered_grouped_and_polarity_opposed"
    return (
        "snare_top_bottom pair forced to one centered snare group; "
        f"polarity_ok={polarity_ok}, centered_ok={centered_ok}, grouped_ok={grouped_ok}"
    )


def _append_note(top: Any, bottom: Any, report: dict[str, Any]) -> None:
    top_note = {
        "type": SNARE_PAIR_COHERENCE_STAGE,
        "pair_role": "snare_top",
        "bottom_file": report["bottom_file"],
        "group": report["group"],
        "polarity_relation": report["polarity_relation"],
        "reason": report["reason"],
    }
    bottom_note = {
        "type": SNARE_PAIR_COHERENCE_STAGE,
        "pair_role": "snare_bottom",
        "top_file": report["top_file"],
        "group": report["group"],
        "polarity_relation": report["polarity_relation"],
        "reason": report["reason"],
    }
    if hasattr(top, "notes"):
        top.notes.append(top_note)
    if hasattr(bottom, "notes"):
        bottom.notes.append(bottom_note)
