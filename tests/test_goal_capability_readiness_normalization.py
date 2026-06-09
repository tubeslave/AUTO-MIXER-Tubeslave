"""Tests for shared readiness payload normalization helpers."""

from backend.goal_capability_readiness_normalization import (
    normalize_goal_capability_readiness_hotspots,
    normalize_goal_capability_readiness_rows,
)


def test_normalize_goal_capability_readiness_rows_converts_types_and_filters():
    normalized = normalize_goal_capability_readiness_rows(
        {
            "eq": {
                "goal_area": "eq",
                "required_row_count": "4",
                "ready_row_count": 2,
                "gap_row_count": 1.0,
                "readiness_score": 87.9,
                "blocker_row_count": True,
                "target_live_writable_row_count": 5,
                "target_proof_gap_count": "3",
                "status_counts": {"blocked": "2", "ready": 3.2},
                "ready_row_ids": [1, 2, "3"],
                "gap_row_ids": ["4", 5],
                "blocker_rows": ["cg_eq_1", 7],
                "blocker_row_ids": [9],
                "full_autonomous_status": "candidate",
                "full_autonomous_status_label": "ready",
                "drop_me": {"nested": "value"},
                "empty_goal_area": "",
            },
            "bad_area": "not-a-row",
        }
    )

    assert normalized == {
        "eq": {
            "goal_area": "eq",
            "required_row_count": 4,
            "ready_row_count": 2,
            "gap_row_count": 1,
            "readiness_score": 87,
            "blocker_row_count": 1,
            "target_live_writable_row_count": 5,
            "target_proof_gap_count": 3,
            "status_counts": {"blocked": 2, "ready": 3},
            "ready_row_ids": ["1", "2", "3"],
            "gap_row_ids": ["4", "5"],
            "blocker_rows": ["cg_eq_1", "7"],
            "blocker_row_ids": ["9"],
            "full_autonomous_status": "candidate",
            "full_autonomous_status_label": "ready",
        },
    }

    assert "" not in normalized


def test_normalize_goal_capability_readiness_rows_with_invalid_row_types():
    assert normalize_goal_capability_readiness_rows("bad") == {}
    assert normalize_goal_capability_readiness_rows({"eq": None}) == {}


def test_normalize_goal_capability_readiness_hotspots_converts_and_skips_empty():
    normalized = normalize_goal_capability_readiness_hotspots(
        [
            {"goal_area": "eq", "gap_row_count": "6", "blocker_row_ids": [1, "cg_eq_2"]},
            {"goal_area": "input_gain", "readiness_score": 44.9, "blocker_row_count": True},
            {"goal_area": "", "gap_row_count": 1},
            "bad-entry",
        ]
    )

    assert normalized == [
        {"goal_area": "eq", "gap_row_count": 6, "blocker_row_ids": ["1", "cg_eq_2"]},
        {"goal_area": "input_gain", "readiness_score": 44, "blocker_row_count": 1},
    ]


def test_normalize_goal_capability_readiness_hotspots_invalid_input():
    assert normalize_goal_capability_readiness_hotspots("bad") == []
