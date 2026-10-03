from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from itertools import product
from typing import Any

import numpy as np

from .pipeline import MasteringConfig, MasteringDirector


@dataclass(frozen=True)
class MasteringTargetSearchConfig:
    """Bounded search space for offline loudness targeting.

    The controller only varies linear pregain and maximizer drive. Tonal,
    transient and spatial decisions remain the responsibility of their own
    mastering modules/directors.
    """

    target_lufs: float = -14.0
    tolerance_lu: float = 0.5
    pregain_grid_db: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    maximizer_drive_grid_db: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0)
    max_candidates: int = 64
    shortlist_size: int = 3


@dataclass(frozen=True)
class MasteringTargetCandidate:
    pregain_db: float
    maximizer_drive_db: float


def _candidate_cost(row: dict[str, Any], target_lufs: float) -> float:
    """Prefer target accuracy with low limiter work and conservative drive."""
    evidence = row["safety"]["evidence"]
    measured = float(evidence["integrated_lufs"])
    target_error = abs(measured - float(target_lufs))
    final_gr = float(evidence.get("final_limiter_gr_db") or 0.0)
    band_gr = float(evidence.get("worst_band_limiter_gr_db") or 0.0)
    candidate = row["candidate"]
    return float(
        4.0 * target_error
        + 1.5 * final_gr
        + 0.75 * band_gr
        + 0.10 * abs(float(candidate["pregain_db"]))
        + 0.35 * abs(float(candidate["maximizer_drive_db"]))
    )


class MasteringTargetController:
    """Search a small, auditable mastering gain/drive space and fail closed.

    A candidate is eligible for the shortlist only when the existing mastering
    safety gate says it is machine-safe. Even the chosen machine-safe candidate
    remains pending human listening because mastering is an audible transform.
    If no candidate is safe, the controller returns the untouched source audio.
    """

    def __init__(
        self,
        base_config: MasteringConfig | None = None,
        search_config: MasteringTargetSearchConfig | None = None,
    ):
        self.base_config = base_config or MasteringConfig()
        self.search_config = search_config or MasteringTargetSearchConfig()
        if not self.base_config.maximizer:
            raise ValueError("MasteringTargetController requires maximizer evidence")
        count = len(self.search_config.pregain_grid_db) * len(self.search_config.maximizer_drive_grid_db)
        if count <= 0:
            raise ValueError("mastering target search grids must not be empty")
        if count > int(self.search_config.max_candidates):
            raise ValueError(f"mastering target search exceeds max_candidates: {count}")
        if self.search_config.shortlist_size < 1:
            raise ValueError("shortlist_size must be positive")

    def search(self, x: np.ndarray, sr: int) -> tuple[np.ndarray, dict[str, Any]]:
        source = np.asarray(x, dtype=np.float32)
        rows: list[dict[str, Any]] = []
        feasible_audio: list[tuple[float, np.ndarray, dict[str, Any]]] = []

        for pregain_db, drive_db in product(
            self.search_config.pregain_grid_db,
            self.search_config.maximizer_drive_grid_db,
        ):
            candidate = MasteringTargetCandidate(float(pregain_db), float(drive_db))
            config = replace(
                self.base_config,
                pregain_db=candidate.pregain_db,
                maximizer_drive_db=candidate.maximizer_drive_db,
                target_lufs=float(self.search_config.target_lufs),
                loudness_tolerance_lu=float(self.search_config.tolerance_lu),
            )
            audio, report = MasteringDirector(config).render(source, sr)
            safety = report["safety"]
            row = {
                "candidate": asdict(candidate),
                "machine_safe": bool(safety["machine_safe"]),
                "verdict": str(safety["verdict"]),
                "protected_regressions": list(safety["protected_regressions"]),
                "safety": safety,
                "after": report["after"],
                "budget": report["budget"],
            }
            row["cost"] = _candidate_cost(row, self.search_config.target_lufs) if row["machine_safe"] else None
            rows.append(row)
            if row["machine_safe"]:
                feasible_audio.append((float(row["cost"]), audio, row))

        feasible_audio.sort(key=lambda item: item[0])
        shortlist = [item[2] for item in feasible_audio[: self.search_config.shortlist_size]]

        if not feasible_audio:
            return source.copy(), {
                "status": "rejected",
                "chosen": None,
                "shortlist": [],
                "evaluated": len(rows),
                "rejected": len(rows),
                "rolled_back_to_source": True,
                "requires_human_listening": False,
                "baseline_eligible": False,
                "search_config": asdict(self.search_config),
                "rows": rows,
                "rule": "No machine-safe loudness candidate: preserve the source and revise the bounded search or mastering hypothesis.",
            }

        _, chosen_audio, chosen = feasible_audio[0]
        return chosen_audio, {
            "status": "pending_human_review",
            "chosen": chosen,
            "shortlist": shortlist,
            "evaluated": len(rows),
            "rejected": len(rows) - len(feasible_audio),
            "rolled_back_to_source": False,
            "requires_human_listening": True,
            "baseline_eligible": False,
            "search_config": asdict(self.search_config),
            "rows": rows,
            "rule": "Machine safety selects a low-cost candidate; level-matched human listening is still required before baseline promotion.",
        }
