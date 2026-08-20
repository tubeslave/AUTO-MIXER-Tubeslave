"""Read-only instrument-space projection for the Product Layer.

The projection is deliberately descriptive.  It never imports a mixer
transport and it never turns a visual position into a console operation.
Unknown measurements remain ``None`` instead of being replaced with a
role-based or cosmetic guess.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import threading
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

try:
    from instrument_space_decision_support import build_instrument_space_decision_support
except ImportError:  # pragma: no cover - package import compatibility
    from backend.instrument_space_decision_support import build_instrument_space_decision_support


SCHEMA_VERSION = "automixer.instrument_space.v1"
MIN_LEVEL_DB = -60.0
MAX_LEVEL_DB = 0.0
MIN_CENTROID_HZ = 80.0
MAX_CENTROID_HZ = 12_000.0
DEFAULT_HISTORY_SNAPSHOTS = 12


HISTORY_AUTHORITY_DENIAL = {
    "read_only": True,
    "live_write": False,
    "queue_mutation": False,
    "can_import_to_queue": False,
    "queue_import_allowed": False,
    "can_apply_to_console": False,
    "live_apply_allowed": False,
    "live_authority_granted": False,
    "readiness_closure_allowed": False,
    "mutates_mixer": False,
}


def _finite_number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _pan_percent(value: Any) -> Optional[float]:
    """Match the existing Product Layer pan readback convention.

    Cached values in the normalized 0..1 interval are converted to the
    legacy dashboard percentage convention; native -100..100 values are
    preserved.  The original readback is also retained in the evidence row.
    """

    number = _finite_number(value)
    if number is None:
        return None
    if 0.0 <= number <= 1.0:
        return round((number - 0.5) * 100.0, 1)
    if -100.0 <= number <= 100.0:
        return round(number, 1)
    return None


def _centroid_from_channel(channel: Mapping[str, Any]) -> Optional[float]:
    direct = _finite_number(channel.get("spectral_centroid_hz"))
    if direct is not None and direct > 0.0:
        return direct
    spectral = channel.get("spectral")
    if isinstance(spectral, Mapping):
        nested = _finite_number(spectral.get("centroid_hz"))
        if nested is not None and nested > 0.0:
            return nested
    return None


def _frequency_position(centroid_hz: Optional[float]) -> Optional[float]:
    if centroid_hz is None:
        return None
    bounded = _clamp(centroid_hz, MIN_CENTROID_HZ, MAX_CENTROID_HZ)
    span = math.log10(MAX_CENTROID_HZ) - math.log10(MIN_CENTROID_HZ)
    return round((math.log10(bounded) - math.log10(MIN_CENTROID_HZ)) / span, 4)


def _depth_position(level_db: Optional[float]) -> Optional[float]:
    if level_db is None:
        return None
    bounded = _clamp(level_db, MIN_LEVEL_DB, MAX_LEVEL_DB)
    return round((bounded - MIN_LEVEL_DB) / (MAX_LEVEL_DB - MIN_LEVEL_DB), 4)


def _scene_object(channel: Mapping[str, Any]) -> Dict[str, Any]:
    channel_id = int(_finite_number(channel.get("id") or channel.get("number")) or 0)
    control = channel.get("control_state")
    control = control if isinstance(control, Mapping) else {}
    recognition = channel.get("ai_recognition")
    recognition = recognition if isinstance(recognition, Mapping) else {}
    read_coverage = channel.get("read_coverage")
    read_coverage = read_coverage if isinstance(read_coverage, Mapping) else {}

    pan_raw = _finite_number(control.get("pan"))
    pan_percent = _pan_percent(pan_raw)
    level_db = _finite_number(channel.get("level_db"))
    centroid_hz = _centroid_from_channel(channel)
    spectral = channel.get("spectral")
    spectral = spectral if isinstance(spectral, Mapping) else {}
    confidence = _finite_number(recognition.get("confidence"))

    coordinates = {
        "x_pan": round(pan_percent / 100.0, 4) if pan_percent is not None else None,
        "y_frequency": _frequency_position(centroid_hz),
        "z_level": _depth_position(level_db),
    }
    available_axes = [name for name, value in coordinates.items() if value is not None]

    return {
        "id": f"channel:{channel_id}",
        "channel_id": channel_id,
        "label": str(channel.get("name") or f"Ch {channel_id}"),
        "role": str(channel.get("role") or "Unknown"),
        "group": str(channel.get("group") or "CHANNELS"),
        "status": str(channel.get("status") or "unknown"),
        "coordinates": coordinates,
        "available_axes": available_axes,
        "evidence": {
            "pan": {
                "available": pan_percent is not None,
                "value": pan_percent,
                "unit": "percent",
                "raw_value": pan_raw,
                "source": "mixer_state_cache" if pan_percent is not None else "unavailable",
            },
            "frequency": {
                "available": centroid_hz is not None,
                "value": round(centroid_hz, 1) if centroid_hz is not None else None,
                "unit": "Hz",
                "source": spectral.get("source") or (
                    "channel_spectral_analysis" if centroid_hz is not None else "unavailable"
                ),
                "age_sec": spectral.get("age_sec"),
                "freshness_status": spectral.get("freshness_status") or "unknown",
                "reason": spectral.get("reason"),
            },
            "level": {
                "available": level_db is not None,
                "value": round(level_db, 1) if level_db is not None else None,
                "unit": "dBFS",
                "source": "audio_capture_meter" if level_db is not None else "unavailable",
            },
            "recognition": {
                "available": confidence is not None,
                "label": recognition.get("label"),
                "confidence_percent": round(_clamp(confidence, 0.0, 100.0), 1) if confidence is not None else None,
                "source": recognition.get("source") or "unavailable",
            },
            "read_coverage_status": str(read_coverage.get("status") or "unknown"),
        },
    }


class InstrumentSpaceHistory:
    """Bounded, in-memory history for read-only Product Layer snapshots.

    History never persists across a server restart and is reset whenever the
    mixer/audio source identity changes.  Stored scenes omit decision support
    and history fields so a snapshot cannot recursively acquire authority or
    grow without bound.
    """

    def __init__(self, max_snapshots: int = DEFAULT_HISTORY_SNAPSHOTS):
        self.max_snapshots = max(2, min(int(max_snapshots), 60))
        self._source_identity: Optional[str] = None
        self._snapshots: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    @staticmethod
    def _compact_scene(scene: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            key: copy.deepcopy(value)
            for key, value in scene.items()
            if key not in {"decision_support", "history"}
        }

    @staticmethod
    def _content_fingerprint(scene: Mapping[str, Any]) -> str:
        content = {
            "schema_version": scene.get("schema_version"),
            "source_context": scene.get("source_context"),
            "objects": scene.get("objects"),
        }
        encoded = json.dumps(
            content,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            default=lambda value: f"<non-json:{type(value).__name__}>",
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def observe(
        self,
        scene: Mapping[str, Any],
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        source_context = scene.get("source_context")
        source_context = source_context if isinstance(source_context, Mapping) else {}
        source_identity = str(source_context.get("identity") or "unbound")
        compact = self._compact_scene(scene)
        content_fingerprint = self._content_fingerprint(compact)
        generated_at = str(scene.get("generated_at") or "")
        snapshot_id = hashlib.sha256(
            f"{source_identity}|{generated_at}|{content_fingerprint}".encode("utf-8")
        ).hexdigest()

        with self._lock:
            source_changed = self._source_identity not in {None, source_identity}
            if source_changed:
                self._snapshots.clear()
            self._source_identity = source_identity
            previous_scene = (
                copy.deepcopy(self._snapshots[-1]["scene"])
                if self._snapshots
                else None
            )
            previous_snapshot_id = (
                self._snapshots[-1]["snapshot_id"] if self._snapshots else None
            )
            if not self._snapshots or self._snapshots[-1]["snapshot_id"] != snapshot_id:
                self._snapshots.append(
                    {
                        "snapshot_id": snapshot_id,
                        "generated_at": generated_at,
                        "content_fingerprint": content_fingerprint,
                        "scene": compact,
                    }
                )
                self._snapshots = self._snapshots[-self.max_snapshots :]

            summaries = [
                {
                    "snapshot_id": row["snapshot_id"],
                    "generated_at": row["generated_at"],
                    "content_fingerprint": row["content_fingerprint"],
                    "object_count": len(row["scene"].get("objects") or []),
                }
                for row in self._snapshots
            ]

        return previous_scene, {
            "type": "instrument_space_history",
            "schema_version": "automixer.instrument_space_history.v1",
            "status": "baseline_required" if previous_scene is None else "tracking",
            "reason": (
                "source_identity_changed_history_reset"
                if source_changed
                else (
                    "first_safe_snapshot_recorded"
                    if previous_scene is None
                    else None
                )
            ),
            "source_identity": source_identity,
            "source_changed": source_changed,
            "snapshot_count": len(summaries),
            "max_snapshots": self.max_snapshots,
            "current_snapshot_id": summaries[-1]["snapshot_id"],
            "baseline_snapshot_id": previous_snapshot_id,
            "snapshots": summaries,
            **HISTORY_AUTHORITY_DENIAL,
        }


def build_instrument_space(
    channels: Iterable[Mapping[str, Any]],
    *,
    generated_at: Optional[str] = None,
    source_context: Optional[Mapping[str, Any]] = None,
    history: Optional[InstrumentSpaceHistory] = None,
) -> Dict[str, Any]:
    """Build a deterministic, transport-free scene projection."""

    objects = [_scene_object(channel) for channel in channels]
    complete = sum(len(obj["available_axes"]) == 3 for obj in objects)
    plottable = sum(bool(obj["available_axes"]) for obj in objects)

    scene = {
        "type": "instrument_space",
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "source_context": copy.deepcopy(dict(source_context or {})),
        "mode": "read_only",
        "physical_localization": False,
        **HISTORY_AUTHORITY_DENIAL,
        "mapping": {
            "x_pan": "left/right pan readback",
            "y_frequency": "log-scaled spectral centroid (visual brightness, not elevation)",
            "z_level": "measured channel level (louder appears closer)",
            "unknown_policy": "unknown measurements stay null and render outside the semantic axis",
        },
        "summary": {
            "object_count": len(objects),
            "plottable_object_count": plottable,
            "complete_object_count": complete,
            "missing_pan_count": sum(obj["coordinates"]["x_pan"] is None for obj in objects),
            "missing_frequency_count": sum(obj["coordinates"]["y_frequency"] is None for obj in objects),
            "missing_level_count": sum(obj["coordinates"]["z_level"] is None for obj in objects),
        },
        "objects": objects,
    }
    previous_scene = None
    if isinstance(history, InstrumentSpaceHistory):
        previous_scene, scene["history"] = history.observe(scene)
    scene["decision_support"] = build_instrument_space_decision_support(
        scene,
        previous_scene=previous_scene,
    )
    return scene
